"""
Module Đánh giá mô hình toàn diện, Tính ECE (Calibration), Benchmark tốc độ & Trích xuất lỗi.

Các chức năng chính:
1. `make_hf_compute_metrics()`: Callback tính Accuracy, Macro-F1, Weighted-F1, Precision,
   Recall cho HuggingFace `Trainer` (ưu tiên `f1_macro` do tập train mất cân bằng IR=2.29).
2. `ESGEvaluator`:
   - Tính toàn bộ chỉ số tổng thể, chỉ số từng lớp (4 lớp ESG), Confusion Matrix và
     Expected Calibration Error (ECE — 10 bins).
   - Chấm điểm độc lập một checkpoint đã train trên Hold-out Test (`test_clean.csv`),
     áp dụng ĐÚNG bộ tiền xử lý của mô hình đó và lưu cả JSON kết quả lẫn xác suất dự đoán
     (`.parquet`) để phục vụ Final Ensemble & Error Analysis ở B7–B8.
   - Trích xuất Top-K mẫu dự đoán sai có độ tin cậy cao nhất và benchmark độ trễ suy luận.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ID2LABEL, LABEL_NAMES, METRICS_DIR, ModelSpec, PipelineConfig
from src.model_wrapper import ESGModelWrapper
from src.preprocessing import VietnameseESGPreprocessor


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính bộ chỉ số phân loại đa lớp chuẩn (có `zero_division=0` tránh cảnh báo)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_hf_compute_metrics():
    """Trả về hàm callback `compute_metrics(eval_pred)` tương thích HuggingFace Trainer."""
    def _compute(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=1)
        return compute_classification_metrics(labels, preds)
    return _compute


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Tính Expected Calibration Error (ECE) và dữ liệu từng bin để vẽ Reliability Diagram.

    ECE càng gần 0 chứng tỏ xác suất tự tin mô hình báo ra càng phản ánh đúng tỉ lệ đoán trúng thực tế.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info: List[Dict] = []
    n_total = len(y_true)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count > 0:
            bin_acc = float(accuracies[mask].mean())
            bin_conf = float(confidences[mask].mean())
            ece += (count / n_total) * abs(bin_acc - bin_conf)
            bins_info.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "accuracy": bin_acc,
                "avg_confidence": bin_conf,
            })
    return float(ece), bins_info


class ESGEvaluator:
    """Bộ đánh giá chuẩn khoa học cho từng mô hình đơn lẻ và hệ thống Ensemble."""

    def __init__(self, label_names: List[str] = None):
        self.label_names = list(label_names) if label_names is not None else list(LABEL_NAMES)

    def compute_all_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> Dict:
        """Tính toàn bộ chỉ số tổng thể, từng lớp, Confusion Matrix và ECE."""
        y_true = np.asarray(y_true, dtype=int)
        y_probs = np.asarray(y_probs, dtype=np.float32)
        y_pred = np.argmax(y_probs, axis=1)

        overall = compute_classification_metrics(y_true, y_pred)
        ece, calibration_bins = compute_ece(y_true, y_probs, n_bins=10)
        overall["ece"] = ece

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.label_names))),
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.label_names))))

        return {
            "overall_metrics": overall,
            "per_class_metrics": {name: report[name] for name in self.label_names},
            "confusion_matrix": cm.tolist(),
            "calibration_bins": calibration_bins,
            "num_samples": int(len(y_true)),
        }

    def evaluate_checkpoint(
        self,
        model_spec: ModelSpec,
        checkpoint_dir: Path,
        test_df: pd.DataFrame,
        config: PipelineConfig,
        run_tag: str,
        metrics_dir: Path = METRICS_DIR,
    ) -> Dict:
        """
        Đánh giá một checkpoint đã lưu trên Hold-out Test (`test_clean.csv`).

        - Nếu đã có kết quả JSON + Parquet và `config.skip_if_done=True` -> nạp lại ngay.
        - Áp dụng đúng `use_word_segmenter` của `model_spec`.
        - Lưu `metrics/{run_tag}_{short_name}_test_eval.json` và `..._test_probs.parquet`.
        """
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        eval_json_path = metrics_dir / f"{run_tag}_{model_spec.short_name}_test_eval.json"
        probs_parquet_path = metrics_dir / f"{run_tag}_{model_spec.short_name}_test_probs.parquet"

        if config.skip_if_done and eval_json_path.exists() and probs_parquet_path.exists():
            print(f"[Bỏ qua Eval] Đã có kết quả test cho {run_tag} / {model_spec.display_name} -> nạp lại từ đĩa.")
            with open(eval_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Tiền xử lý test set đồng nhất với lúc train của từng mô hình
        preprocessor = VietnameseESGPreprocessor(use_word_segmenter=model_spec.use_word_segmenter)
        processed_texts = preprocessor.transform_list(test_df[config.text_column].tolist())
        y_true = test_df[config.label_column].astype(int).to_numpy()

        hw = config.get_hardware_settings()
        wrapper = ESGModelWrapper(checkpoint_dir).load()
        t0 = time.perf_counter()
        probs = wrapper.predict_proba(
            processed_texts,
            batch_size=hw["eval_batch_size"],
            max_length=config.max_length,
            use_bf16=hw["bf16"] and (not model_spec.force_fp32),
            show_progress=True,
            desc=f"Đánh giá Test [{run_tag}] {model_spec.display_name}",
        )
        elapsed = time.perf_counter() - t0
        params_info = wrapper.count_parameters()
        wrapper.release_memory()

        result = self.compute_all_metrics(y_true, probs)
        result.update({
            "run_tag": run_tag,
            "model_short_name": model_spec.short_name,
            "model_display_name": model_spec.display_name,
            "hf_name": model_spec.hf_name,
            "use_word_segmenter": model_spec.use_word_segmenter,
            "checkpoint_dir": str(checkpoint_dir),
            "eval_seconds": round(float(elapsed), 3),
            "ms_per_sample": round(float(elapsed / max(len(y_true), 1) * 1000.0), 3),
            "samples_per_second": round(float(len(y_true) / max(elapsed, 1e-6)), 2),
            "num_parameters": params_info["total"],
        })

        # Lưu xác suất 4 lớp trên tập test để dùng cho Final Ensemble & Error Analysis
        probs_df = pd.DataFrame({
            "true_label": y_true,
            "pred_label": np.argmax(probs, axis=1),
            "confidence": np.max(probs, axis=1),
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "prob_2": probs[:, 2],
            "prob_3": probs[:, 3],
        })
        probs_df.to_parquet(probs_parquet_path, index=False)

        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        ov = result["overall_metrics"]
        print(
            f"  => [{run_tag}] {model_spec.display_name:<15} | "
            f"Acc={ov['accuracy']*100:.2f}% | Macro-F1={ov['f1_macro']*100:.2f}% | "
            f"Weighted-F1={ov['f1_weighted']*100:.2f}% | ECE={ov['ece']:.4f} | "
            f"Tốc độ={result['samples_per_second']:.1f} mẫu/s"
        )
        return result

    def extract_top_errors(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        text_col: str = "vi_text",
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Trích xuất Top-K mẫu dự đoán sai có độ tin cậy cao nhất để phân tích lỗi (Error Analysis)."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.argmax(y_probs, axis=1)
        conf = np.max(y_probs, axis=1)

        wrong_mask = y_pred != y_true
        err_df = pd.DataFrame({
            "row_index": np.where(wrong_mask)[0],
            "vi_text": df[text_col].iloc[wrong_mask].values,
            "true_id": y_true[wrong_mask],
            "true_label": [ID2LABEL[int(i)] for i in y_true[wrong_mask]],
            "pred_id": y_pred[wrong_mask],
            "pred_label": [ID2LABEL[int(i)] for i in y_pred[wrong_mask]],
            "pred_confidence": conf[wrong_mask],
            "true_class_prob": y_probs[wrong_mask, y_true[wrong_mask]],
        })
        return err_df.sort_values("pred_confidence", ascending=False).head(top_k).reset_index(drop=True)

    def benchmark_inference(
        self,
        model_wrapper: ESGModelWrapper,
        sample_texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        use_bf16: bool = True,
    ) -> Dict:
        """Đo thời gian suy luận trung bình (warm-up trước 1 batch để đo chính xác trên GPU)."""
        # Warm-up GPU
        _ = model_wrapper.predict_proba(
            sample_texts[: min(batch_size, len(sample_texts))],
            batch_size=batch_size,
            max_length=max_length,
            use_bf16=use_bf16,
            show_progress=False,
        )
        t0 = time.perf_counter()
        _ = model_wrapper.predict_proba(
            sample_texts,
            batch_size=batch_size,
            max_length=max_length,
            use_bf16=use_bf16,
            show_progress=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "num_samples": len(sample_texts),
            "total_seconds": round(float(elapsed), 4),
            "ms_per_sample": round(float(elapsed / max(len(sample_texts), 1) * 1000.0), 3),
            "samples_per_second": round(float(len(sample_texts) / max(elapsed, 1e-6)), 2),
        }


def mcnemar_significance_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    """
    Kiểm định thống kê McNemar (có hiệu chỉnh liên tục Edwards) so sánh hai bộ phân loại A và B
    trên cùng tập kiểm thử Hold-out Test Set (3.000 mẫu).

    Returns:
        Dict gồm: n_a_only_correct (n10), n_b_only_correct (n01), chi2_stat, p_value, significant_005.
    """
    from scipy.stats import chi2

    y_true = np.asarray(y_true, dtype=int)
    a_ok = np.asarray(pred_a, dtype=int) == y_true
    b_ok = np.asarray(pred_b, dtype=int) == y_true

    n10 = int(np.sum(a_ok & (~b_ok)))  # A đúng, B sai
    n01 = int(np.sum((~a_ok) & b_ok))  # A sai, B đúng
    denom = n10 + n01
    if denom == 0:
        stat, p_val = 0.0, 1.0
    else:
        stat = float(((abs(n10 - n01) - 1.0) ** 2) / denom) if abs(n10 - n01) >= 1 else 0.0
        p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "a_only_correct": n10,
        "b_only_correct": n01,
        "discordant_total": denom,
        "chi2_statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant_at_005": bool(p_val < 0.05),
    }
