"""
Lớp bao (Wrapper) thống nhất cho 7 kiến trúc Transformer phân loại 4 lớp ESG.

Các cải tiến chính so với mã nguồn gốc:
- Dùng `AutoModelForSequenceClassification` và `AutoTokenizer` chuẩn hóa, tương thích
  `transformers 5.24.0` cho cả 7 họ kiến trúc (PhoBERT, DistilBERT, mBERT, RoBERTa,
  Vi-ELECTRA, ViSoBERT, ViDeBERTa).
- Suy luận theo lô (`predict_proba`) tối ưu cho RTX 5070:
  + Tự động sắp xếp tạm câu theo độ dài (length-sorted dynamic padding) rồi khôi phục
    đúng thứ tự gốc -> tăng tốc 2-3x trên 54.487 câu không nhãn.
  + Dùng `torch.inference_mode()` + `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
    để tiết kiệm VRAM và tránh tràn số NaN ở DeBERTa-v2.
- Hỗ trợ suy luận 1 câu (`predict_one`) chuyển đúng tensor lên GPU/CPU (khắc phục lỗi
  device mismatch của mã nguồn cũ).
- Dọn sạch VRAM triệt để qua `release_memory()`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import gc
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

# Tắt monitor thread nền của tqdm (tránh lỗi ZMQ 'Too many open files' trên Windows JupyterLab)
tqdm.monitor_interval = 0
# Chỉ hiển thị lỗi thực sự từ transformers (ẩn bảng UNEXPECTED/MISSING dài khi gắn classifier head mới)
hf_logging.set_verbosity_error()

from src.config import ID2LABEL, LABEL2ID, NUM_LABELS


def get_device() -> torch.device:
    """Trả về thiết bị tính toán ưu tiên GPU CUDA nếu khả dụng."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESGModelWrapper:
    """Wrapper chuẩn hóa tải, lưu và suy luận cho các mô hình Transformer ESG."""

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        num_labels: int = NUM_LABELS,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.metadata: Dict = {}
        self.device = get_device()

    def load(self) -> "ESGModelWrapper":
        """Tải trọng số mô hình và tokenizer từ HuggingFace Hub hoặc checkpoint cục bộ."""
        print(f"[ESGModelWrapper] Đang tải mô hình & tokenizer từ: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dtype=torch.float32,  # Luôn nạp trọng số gốc ở FP32 (tránh Transformers v5 nạp ViDeBERTa ở FP16 gây đóng băng gradient)
            ignore_mismatched_sizes=True,
        )

        # Nạp thêm metadata nếu đang load từ thư mục checkpoint đã lưu
        meta_path = Path(self.model_name_or_path) / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        return self

    def save(self, save_dir: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Lưu trọng số mô hình, tokenizer và tệp thông tin `model_metadata.json`."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Mô hình chưa được khởi tạo hoặc tải trước khi lưu.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        meta_payload = {
            "source_model": self.model_name_or_path,
            "num_labels": self.num_labels,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            **(metadata or {}),
        }
        with open(save_path / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        print(f"[ESGModelWrapper] Đã lưu mô hình & metadata tại: {save_path}")

    def predict_proba(
        self,
        texts: List[str],
        batch_size: int = 64,
        max_length: int = 256,
        use_bf16: bool = True,
        show_progress: bool = True,
        desc: str = "Suy luận",
    ) -> np.ndarray:
        """
        Dự đoán ma trận xác suất 4 lớp shape (N, 4) cho danh sách văn bản đã tiền xử lý.

        Tối ưu tốc độ: sắp xếp tạm các câu theo độ dài ký tự trước khi chia batch để
        giảm lượng padding thừa, sau đó khôi phục đúng thứ tự chỉ số ban đầu.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Mô hình chưa được tải. Hãy gọi `.load()` trước.")

        n_samples = len(texts)
        if n_samples == 0:
            return np.zeros((0, self.num_labels), dtype=np.float32)

        self.model.eval()
        self.model.to(self.device)

        # Sắp xếp chỉ số theo độ dài câu để dynamic padding đạt hiệu suất cao nhất
        clean_texts = [str(t) if t is not None else "" for t in texts]
        order = np.argsort([len(t) for t in clean_texts])
        sorted_texts = [clean_texts[i] for i in order]

        use_amp_bf16 = (
            use_bf16
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        use_amp_fp16 = (
            (not use_amp_bf16)
            and self.device.type == "cuda"
            and use_bf16
        )

        sorted_probs: List[np.ndarray] = []
        batch_starts = range(0, n_samples, batch_size)
        iterator = (
            tqdm(batch_starts, desc=desc, unit="batch")
            if show_progress
            else batch_starts
        )

        with torch.inference_mode():
            for start in iterator:
                batch_texts = sorted_texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                if use_amp_bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(**encoded).logits
                elif use_amp_fp16:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = self.model(**encoded).logits
                else:
                    logits = self.model(**encoded).logits

                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                sorted_probs.append(probs)

        all_sorted_probs = np.vstack(sorted_probs).astype(np.float32)

        # Khôi phục đúng thứ tự gốc ban đầu
        restored_probs = np.empty_like(all_sorted_probs)
        restored_probs[order] = all_sorted_probs
        return restored_probs

    def predict_one(self, text: str, max_length: int = 256) -> Dict:
        """Dự đoán cho một câu đơn lẻ (tự đồng bộ thiết bị GPU/CPU)."""
        probs = self.predict_proba(
            [text],
            batch_size=1,
            max_length=max_length,
            use_bf16=True,
            show_progress=False,
        )[0]
        class_id = int(np.argmax(probs))
        return {
            "class_id": class_id,
            "class_name": ID2LABEL[class_id],
            "confidence": float(probs[class_id]),
            "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(self.num_labels)},
        }

    def count_parameters(self) -> Dict[str, int]:
        """Đếm tổng số tham số và số tham số huấn luyện được của mô hình."""
        if self.model is None:
            return {"total": 0, "trainable": 0}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    def release_memory(self) -> None:
        """Giải phóng mô hình khỏi RAM và VRAM GPU để tránh tích tụ phân mảnh bộ nhớ."""
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
