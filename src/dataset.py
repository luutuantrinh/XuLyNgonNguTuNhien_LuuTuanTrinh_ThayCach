"""
PyTorch Dataset cho bài toán phân loại 4 lớp ESG tối ưu bộ nhớ & tốc độ trên Windows JupyterLab.

Thiết kế:
- Hỗ trợ cả chế độ có nhãn (Train / Val / Test) và không nhãn (Inference trên tập Unlabeled).
- Tiền tokenize theo lô (batch tokenization) một lần trong `__init__` với `truncation=True`
  và KHÔNG đệm cố định lên `max_length`, kết hợp `DataCollatorWithPadding` (dynamic padding)
  giúp giảm mạnh VRAM và tăng tốc ~2x khi `dataloader_num_workers=0` trên Windows.
"""

from typing import Dict, List, Optional, Sequence
import pandas as pd
from torch.utils.data import Dataset


class ESGTextDataset(Dataset):
    """Dataset PyTorch cho văn bản ESG hỗ trợ dynamic padding."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        labels: Optional[Sequence[int]] = None,
        max_length: int = 256,
    ):
        """
        Khởi tạo và tiền tokenize toàn bộ văn bản (chưa pad để dùng dynamic padding).

        Args:
            texts: Chuỗi/Series/List các văn bản đã qua `VietnameseESGPreprocessor`.
            tokenizer: Tokenizer của mô hình Transformer tương ứng.
            labels: Nhãn số nguyên 0..3 (None nếu là tập không nhãn).
            max_length: Độ dài token tối đa (mặc định 256).
        """
        text_list: List[str] = [str(t) if pd.notna(t) else "" for t in texts]
        self.encodings = tokenizer(
            text_list,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=True,
        )
        self.labels: Optional[List[int]] = (
            [int(y) for y in labels] if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
