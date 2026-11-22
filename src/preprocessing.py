"""
Module tiền xử lý văn bản tiếng Việt & kiểm toán chất lượng dữ liệu (Zero Data Leakage).

Các chức năng chính:
1. Làm sạch văn bản tiếng Việt chuẩn mực (Unicode NFC, xóa URL/HTML/khoảng trắng thừa,
   giữ nguyên dấu câu và con số có ý nghĩa trong văn bản tài chính/ESG).
2. Tách từ tiếng Việt bằng `pyvi.ViTokenizer` CÓ ĐIỀU KIỆN theo từng kiến trúc mô hình
   (đảm bảo đồng nhất 100% giữa Train, Val, Test, Pseudo-Labeling và Demo).
3. Kiểm toán & khử trùng lặp (Deduplication + Leakage Audit):
   - Loại bỏ câu trùng lặp nội bộ (kể cả câu bị mâu thuẫn nhãn) trong tập train.
   - Loại bỏ hoàn toàn mọi câu trong tập train và tập unlabeled bị trùng khớp với tập test
     nhằm đảm bảo đánh giá khách quan, không rò rỉ dữ liệu.
4. Chia tập Train Pool / Validation cố định (Stratified) và tạo 7 tập con (subsets) cho Run 1.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import html
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABEL2ID, ID2LABEL


# Biểu thức chính quy biên dịch sẵn giúp xử lý nhanh trên tập dữ liệu lớn (>80k dòng)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_CONTROL_CHARS_PATTERN = re.compile(r"[\r\n\t\u200b\u200c\u200d\ufeff]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
# Dùng để tạo khóa so sánh trùng lặp (chỉ giữ chữ cái tiếng Việt và số, bỏ dấu câu & khoảng trắng thừa)
_NORM_KEY_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_vietnamese_text(text: object) -> str:
    """
    Làm sạch cơ bản một câu tiếng Việt nhưng vẫn bảo toàn dấu câu và số liệu ESG.

    Các bước xử lý:
    - Chuyển giá trị NaN/None thành chuỗi rỗng.
    - Giải mã thực thể HTML (&amp;, &quot;...) và chuẩn hóa Unicode về dạng dựng sẵn (NFC).
    - Xóa URL, thẻ HTML, ký tự điều khiển ẩn và gom khoảng trắng thừa về 1 dấu cách.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)
    s = _URL_PATTERN.sub(" ", s)
    s = _HTML_TAG_PATTERN.sub(" ", s)
    s = _CONTROL_CHARS_PATTERN.sub(" ", s)
    s = _MULTI_SPACE_PATTERN.sub(" ", s).strip()
    return s


def make_dedup_key(text: str) -> str:
    """
    Tạo khóa chuẩn hóa để phát hiện trùng lặp chính xác và trùng lặp chỉ khác chữ hoa/dấu câu.
    """
    s = clean_vietnamese_text(text).lower()
    s = _NORM_KEY_PATTERN.sub(" ", s)
    s = _MULTI_SPACE_PATTERN.sub(" ", s).strip()
    return s


class VietnameseESGPreprocessor:
    """
    Bộ tiền xử lý thống nhất cho từng mô hình Transformer.

    - Nếu `use_word_segmenter=True` (ví dụ PhoBERT): làm sạch -> tách từ bằng `pyvi.ViTokenizer`.
    - Nếu `use_word_segmenter=False` (Vi-ELECTRA, ViSoBERT, ViDeBERTa, mBERT, DistilBERT, RoBERTa):
      chỉ áp dụng làm sạch chuẩn hóa Unicode NFC, giữ nguyên văn bản tự nhiên cho subword tokenizer.
    """

    def __init__(self, use_word_segmenter: bool = False):
        self.use_word_segmenter = use_word_segmenter
        self._vi_tokenizer = None
        if self.use_word_segmenter:
            try:
                from pyvi import ViTokenizer
                self._vi_tokenizer = ViTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Chưa cài thư viện 'pyvi' để tách từ cho PhoBERT. "
                    "Hãy chạy: pip install pyvi"
                ) from exc

    def clean_text(self, text: object) -> str:
        """Làm sạch chuẩn hóa 1 câu tiếng Việt."""
        return clean_vietnamese_text(text)

    def segment_text(self, text: str) -> str:
        """Tách từ tiếng Việt bằng pyvi nếu cờ `use_word_segmenter=True`."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        if self.use_word_segmenter and self._vi_tokenizer is not None:
            return self._vi_tokenizer.tokenize(cleaned)
        return cleaned

    def transform_series(self, texts: pd.Series) -> pd.Series:
        """Áp dụng tiền xử lý nhất quán cho toàn bộ một cột Series văn bản."""
        return texts.apply(self.segment_text)

    def transform_list(self, texts: List[str]) -> List[str]:
        """Áp dụng tiền xử lý nhất quán cho một danh sách câu (dùng khi inference/demo)."""
        return [self.segment_text(t) for t in texts]


def audit_and_deduplicate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    text_col: str = "vi_text",
    label_col: str = "labels",
    min_words: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Kiểm toán chất lượng dữ liệu, làm sạch và chống rò rỉ dữ liệu (Zero Data Leakage).

    Quy tắc xử lý:
    1. Tập `test_df` là mốc đánh giá chuẩn: chỉ làm sạch text (KHÔNG xóa dòng để giữ nguyên 3.000 mẫu).
    2. Tập `train_df`:
       - Xóa giá trị thiếu, câu rỗng và câu ngắn hơn `min_words` từ.
       - Phát hiện và xóa các câu trùng lặp nội bộ (nếu 1 câu có nhiều nhãn mâu thuẫn -> xóa toàn bộ nhóm mâu thuẫn;
         nếu trùng nhãn -> giữ lại 1 bản duy nhất).
       - Xóa sạch mọi câu trùng khớp với `test_df` (loại bỏ rò rỉ train <-> test).
    3. Tập `unlabeled_df` (`overall_data.csv`):
       - Xóa giá trị thiếu, câu rỗng, câu ngắn hơn `min_words` từ.
       - Khử trùng lặp nội bộ.
       - Xóa sạch mọi câu trùng với `train_clean` hoặc `test_clean` (chống rò rỉ vào pseudo-labeling).
       - Gắn mã định danh duy nhất `text_id` để theo dõi xuyên suốt 2 vòng Pseudo-Labeling.

    Returns:
        (train_clean, test_clean, unlabeled_clean, audit_report)
    """
    report: Dict = {
        "train_raw_rows": int(len(train_df)),
        "test_raw_rows": int(len(test_df)),
        "unlabeled_raw_rows": int(len(unlabeled_df)),
    }

    # --- Bước 1: Chuẩn hóa tập Test (giữ nguyên toàn bộ mẫu benchmark) ---
    test_clean = test_df[[text_col, label_col]].copy()
    test_clean[text_col] = test_clean[text_col].apply(clean_vietnamese_text)
    test_clean[label_col] = test_clean[label_col].astype(int)
    test_clean["_dedup_key"] = test_clean[text_col].apply(make_dedup_key)
    test_keys = set(k for k in test_clean["_dedup_key"] if k)
    report["test_unique_keys"] = int(len(test_keys))
    report["test_internal_duplicates"] = int(len(test_clean) - len(test_keys))

    # --- Bước 2: Làm sạch & kiểm toán tập Train ---
    train_work = train_df[[text_col, label_col]].copy()
    train_work = train_work.dropna(subset=[text_col, label_col]).copy()
    train_work[text_col] = train_work[text_col].apply(clean_vietnamese_text)
    train_work[label_col] = train_work[label_col].astype(int)
    train_work = train_work[train_work[label_col].isin(ID2LABEL.keys())].copy()

    # Đếm & lọc câu rỗng hoặc quá ngắn
    word_counts_train = train_work[text_col].apply(lambda s: len(s.split()))
    too_short_train = int((word_counts_train < min_words).sum())
    train_work = train_work[word_counts_train >= min_words].copy()
    report["train_removed_short_or_empty"] = int(
        (report["train_raw_rows"] - len(train_work))
    )
    report["train_short_under_min_words"] = too_short_train

    # Tạo khóa khử trùng lặp
    train_work["_dedup_key"] = train_work[text_col].apply(make_dedup_key)

    # Phát hiện câu trùng lặp nhưng bị gán nhãn mâu thuẫn (label conflict)
    label_nunique = train_work.groupby("_dedup_key")[label_col].nunique()
    conflicting_keys = set(label_nunique[label_nunique > 1].index)
    rows_conflict = int(train_work["_dedup_key"].isin(conflicting_keys).sum())
    train_work = train_work[~train_work["_dedup_key"].isin(conflicting_keys)].copy()
    report["train_removed_label_conflicts"] = rows_conflict

    # Khử trùng lặp nội bộ cùng nhãn (giữ bản đầu tiên)
    before_dedup = len(train_work)
    train_work = train_work.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    report["train_removed_internal_duplicates"] = int(before_dedup - len(train_work))

    # Loại bỏ rò rỉ trùng lặp với tập Test
    leak_mask_train = train_work["_dedup_key"].isin(test_keys)
    report["train_removed_test_leakage"] = int(leak_mask_train.sum())
    train_work = train_work[~leak_mask_train].copy()

    train_keys = set(train_work["_dedup_key"])
    train_clean = train_work[[text_col, label_col]].reset_index(drop=True)
    report["train_clean_rows"] = int(len(train_clean))

    # --- Bước 3: Làm sạch & kiểm toán tập Unlabeled (overall_data.csv) ---
    unl_work = unlabeled_df[[text_col]].copy()
    unl_work = unl_work.dropna(subset=[text_col]).copy()
    unl_work[text_col] = unl_work[text_col].apply(clean_vietnamese_text)

    word_counts_unl = unl_work[text_col].apply(lambda s: len(s.split()))
    unl_work = unl_work[word_counts_unl >= min_words].copy()
    report["unlabeled_removed_short_or_empty"] = int(
        report["unlabeled_raw_rows"] - len(unl_work)
    )

    unl_work["_dedup_key"] = unl_work[text_col].apply(make_dedup_key)

    # Khử trùng lặp nội bộ trong tập Unlabeled
    before_unl_dedup = len(unl_work)
    unl_work = unl_work.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    report["unlabeled_removed_internal_duplicates"] = int(
        before_unl_dedup - len(unl_work)
    )

    # Loại bỏ câu trùng với tập Test hoặc tập Train
    leak_unl_test = unl_work["_dedup_key"].isin(test_keys)
    leak_unl_train = unl_work["_dedup_key"].isin(train_keys)
    report["unlabeled_removed_test_leakage"] = int(leak_unl_test.sum())
    report["unlabeled_removed_train_overlap"] = int((leak_unl_train & ~leak_unl_test).sum())

    unl_work = unl_work[~(leak_unl_test | leak_unl_train)].copy()
    unlabeled_clean = unl_work[[text_col]].reset_index(drop=True)
    # Gắn mã text_id cố định để theo dõi mẫu xuyên suốt Vòng 1 và Vòng 2
    unlabeled_clean.insert(0, "text_id", [f"unl_{i:06d}" for i in range(len(unlabeled_clean))])
    report["unlabeled_clean_rows"] = int(len(unlabeled_clean))

    # Dọn cột khóa phụ trên tập Test trước khi trả về
    test_clean = test_clean[[text_col, label_col]].reset_index(drop=True)
    report["test_clean_rows"] = int(len(test_clean))

    return train_clean, test_clean, unlabeled_clean, report


def split_train_val_and_subsets(
    train_clean_df: pd.DataFrame,
    val_ratio: float = 0.10,
    n_subsets: int = 7,
    subset_size: int = 10_000,
    seed: int = 42,
    label_col: str = "labels",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    """
    Tách tập train sạch thành:
    - `train_pool_df` (90%): Kho dữ liệu huấn luyện gốc.
    - `val_df` (10%): Tập Validation cố định (Stratified) dùng chung cho tất cả các vòng.
    - `subsets`: 7 tập con (mỗi tập tối đa `subset_size` mẫu) lấy ngẫu nhiên từ `train_pool_df`
      với các seed khác nhau (`seed + i`) để tạo sự đa dạng giữa 7 mô hình ở Run 1.
    """
    train_pool_df, val_df = train_test_split(
        train_clean_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_clean_df[label_col],
    )
    train_pool_df = train_pool_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    actual_subset_size = min(subset_size, len(train_pool_df))
    subsets: List[pd.DataFrame] = []
    for i in range(n_subsets):
        sub_df = train_pool_df.sample(
            n=actual_subset_size,
            random_state=seed + i,
            replace=False,
        ).reset_index(drop=True)
        subsets.append(sub_df)

    return train_pool_df, val_df, subsets
