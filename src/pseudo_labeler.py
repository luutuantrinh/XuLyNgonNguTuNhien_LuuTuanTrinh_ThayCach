"""
Module True Unsupervised Pseudo-Labeling trên dữ liệu không nhãn (`unlabeled_clean.csv`).

Khác với mã nguồn cũ (vốn đòi cột `labels` có sẵn để đối chiếu `pred == original_label`),
module này thực hiện gán nhãn giả không giám sát thực thụ:
- Đọc toàn bộ văn bản không nhãn kèm định danh `text_id`.
- Áp dụng ĐÚNG bộ tiền xử lý (`use_word_segmenter`) trùng khớp 100% với lúc huấn luyện của từng mô hình.
- Suy luận theo lô có `BF16` + length-sorted dynamic padding (rất nhanh & mát trên RTX 5070).
- Lưu trọn vẹn 4 xác suất `prob_0..prob_3`, nhãn dự đoán `pred_label`, độ tin cậy `confidence`,
  biên độ chênh lệch `margin = p(1) - p(2)` và độ bất định `entropy` ra tệp `.parquet` gọn nhẹ.
"""

from pathlib import Path
from typing import Dict
import json
import time

import numpy as np
import pandas as pd

from src.config import ID2LABEL, ModelSpec, PipelineConfig
from src.model_wrapper import ESGModelWrapper
from src.preprocessing import VietnameseESGPreprocessor


class UnsupervisedPseudoLabeler:
    """Sinh phân phối xác suất 4 lớp cho tập không nhãn từ một checkpoint đã huấn luyện."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def predict_unlabeled(
        self,
        model_spec: ModelSpec,
        checkpoint_dir: Path,
        unlabeled_df: pd.DataFrame,
        output_path: Path,
        round_tag: str = "r1",
    ) -> pd.DataFrame:
        """
        Dự đoán xác suất đầy đủ 4 lớp trên toàn bộ `unlabeled_df` và lưu `.parquet`.

        Args:
            model_spec: Thông số mô hình từ `MODEL_REGISTRY`.
            checkpoint_dir: Thư mục checkpoint đã train (vd `checkpoints/run1/phobert-base`).
            unlabeled_df: DataFrame chứa cột `text_id` và `vi_text`.
            output_path: Đường dẫn file `.parquet` đầu ra.
            round_tag: Mã vòng (`'r1'` hoặc `'r2'`).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = output_path.with_suffix(".json")

        # Công tắc bỏ qua nếu file dự đoán của mô hình này đã tồn tại
        if self.config.skip_if_done and output_path.exists() and meta_path.exists():
            print(f"[Bỏ qua Pseudo] [{round_tag}] {model_spec.display_name} đã có tệp {output_path.name} -> nạp lại.")
            return pd.read_parquet(output_path)

        print(
            f"\n[Pseudo-Labeling {round_tag}] Mô hình: {model_spec.display_name} | "
            f"Số câu: {len(unlabeled_df):,} | pyvi={model_spec.use_word_segmenter}"
        )

        # Bước 1: Tiền xử lý văn bản đồng nhất với lúc huấn luyện
        preprocessor = VietnameseESGPreprocessor(use_word_segmenter=model_spec.use_word_segmenter)
        processed_texts = preprocessor.transform_list(unlabeled_df[self.config.text_column].tolist())

        # Bước 2: Tải checkpoint & chạy suy luận theo lô tối ưu BF16
        hw = self.config.get_hardware_settings()
        wrapper = ESGModelWrapper(checkpoint_dir).load()
        t0 = time.perf_counter()
        probs = wrapper.predict_proba(
            processed_texts,
            batch_size=hw["inference_batch_size"],
            max_length=self.config.max_length,
            use_bf16=hw["bf16"] and (not model_spec.force_fp32),
            show_progress=True,
            desc=f"Pseudo [{round_tag}] {model_spec.display_name}",
        )
        elapsed = time.perf_counter() - t0
        wrapper.release_memory()

        # Bước 3: Tính các đại lượng độ tin cậy (confidence, margin, entropy)
        sorted_probs = np.sort(probs, axis=1)
        pred_labels = np.argmax(probs, axis=1).astype(int)
        confidence = sorted_probs[:, -1].astype(np.float32)
        margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)
        entropy = (-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)).astype(np.float32)

        result_df = pd.DataFrame({
            "text_id": unlabeled_df["text_id"].values,
            "pred_label": pred_labels,
            "pred_label_name": [ID2LABEL[int(c)] for c in pred_labels],
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "prob_2": probs[:, 2],
            "prob_3": probs[:, 3],
        })
        result_df.to_parquet(output_path, index=False)

        # Lưu thống kê metadata cho từng mô hình
        threshold_ref = (
            self.config.round1_confidence_threshold
            if round_tag == "r1"
            else self.config.round2_confidence_threshold
        )
        high_conf_mask = result_df["confidence"] >= threshold_ref
        meta: Dict = {
            "round_tag": round_tag,
            "model_short_name": model_spec.short_name,
            "model_display_name": model_spec.display_name,
            "total_unlabeled": int(len(result_df)),
            "inference_seconds": round(float(elapsed), 2),
            "samples_per_second": round(float(len(result_df) / max(elapsed, 1e-6)), 1),
            "mean_confidence": round(float(result_df["confidence"].mean()), 4),
            "median_confidence": round(float(result_df["confidence"].median()), 4),
            "mean_entropy": round(float(result_df["entropy"].mean()), 4),
            "threshold_ref": threshold_ref,
            "count_above_threshold": int(high_conf_mask.sum()),
            "pct_above_threshold": round(float(high_conf_mask.mean() * 100.0), 2),
            "pred_distribution_all": result_df["pred_label_name"].value_counts().to_dict(),
            "pred_distribution_above_threshold": result_df.loc[high_conf_mask, "pred_label_name"].value_counts().to_dict(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(
            f"  => Xong {model_spec.display_name}: {meta['count_above_threshold']:,}/{len(result_df):,} "
            f"mẫu đạt conf >= {threshold_ref} ({meta['pct_above_threshold']}%) | "
            f"Tốc độ={meta['samples_per_second']} mẫu/s ({elapsed:.1f}s)"
        )

        # Nghỉ ngắn làm mát GPU sau bước suy luận nặng
        cooldown = min(int(hw.get("cooldown_seconds", 0)), 8)
        if cooldown > 0:
            time.sleep(cooldown)

        return result_df
