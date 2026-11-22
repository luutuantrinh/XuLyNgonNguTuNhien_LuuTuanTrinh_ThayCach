"""
Module Ensemble Consensus Voting (Hard Voting, Equal Soft Voting & Weighted Soft Voting).

Phục vụ 2 tác vụ cốt lõi:
1. **Bỏ phiếu đồng thuận lọc nhãn giả (`vote_pseudo_labels` — Bước B4 & B6):**
   - Kết hợp dự đoán của các mô hình đang bật (3, 4 hoặc 7 mô hình) trên từng `text_id`.
   - Áp dụng bộ lọc kép nghiêm ngặt chống nhiễu cho True Unsupervised Pseudo-Labeling:
     (a) Số mô hình đồng thuận cứng (`vote_count >= ceil(N_models * min_vote_ratio)`).
     (b) Nhãn bỏ phiếu mềm có trọng số (`weighted_soft_label`) TRÙNG KHỚP với nhãn bỏ phiếu cứng.
     (c) Độ tin cậy Ensemble `confidence >= confidence_threshold` (0.94 ở R1, 0.91 ở R2).
     (d) Biên độ chênh lệch xác suất `margin = p(1) - p(2) >= min_margin` (chặn câu vùng biên).
   - Tự động tính ma trận đồng thuận cặp (Pairwise Agreement % và **Cohen's Kappa**) giữa
     các mô hình và xuất đồng thời `ensemble_pseudo_{tag}.csv`, `remain_data_{tag}.csv`,
     `ensemble_report_{tag}.json`.
2. **Suy luận Final Ensemble trên tập Test (`predict_ensemble_test` — Bước B7 & B8).**
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import ID2LABEL, LABEL_NAMES, NUM_LABELS


class EnsembleVoter:
    """Bộ bỏ phiếu đồng thuận đa mô hình hỗ trợ trọng số theo hiệu năng Validation."""

    def __init__(
        self,
        confidence_threshold: float = 0.94,
        min_vote_ratio: float = 0.60,
        min_margin: float = 0.50,
        model_weights: Optional[Dict[str, float]] = None,
        use_weighted_voting: bool = True,
        min_model_val_f1: float = 0.50,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_vote_ratio = min_vote_ratio
        self.min_margin = min_margin
        self.model_weights = model_weights or {}
        self.use_weighted_voting = use_weighted_voting
        # Cổng kiểm định chất lượng (Quality Gate): tự động loại mô hình bị collapse khỏi Ensemble
        self.min_model_val_f1 = min_model_val_f1

    def _get_normalized_weights(self, model_names: List[str]) -> np.ndarray:
        """Chuẩn hóa vector trọng số bỏ phiếu sao cho tổng bằng 1.0."""
        if self.use_weighted_voting and all(m in self.model_weights for m in model_names):
            raw = np.array([max(float(self.model_weights[m]), 1e-4) for m in model_names], dtype=np.float64)
        else:
            raw = np.ones(len(model_names), dtype=np.float64)
        return raw / raw.sum()

    def compute_pairwise_agreement(
        self,
        preds_matrix: np.ndarray,
        model_names: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tính 2 ma trận (M x M) đo mức độ tương đồng giữa các cặp mô hình:
        - `agreement_df`: Tỉ lệ phần trăm dự đoán giống hệt nhau (0..100%).
        - `kappa_df`: Hệ số đồng thuận Cohen's Kappa (hiệu chỉnh yếu tố trùng ngẫu nhiên).
        """
        m = len(model_names)
        agree = np.zeros((m, m), dtype=np.float64)
        kappa = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                if i == j:
                    agree[i, j] = 100.0
                    kappa[i, j] = 1.0
                elif j > i:
                    ag = float((preds_matrix[i] == preds_matrix[j]).mean() * 100.0)
                    kp = float(cohen_kappa_score(preds_matrix[i], preds_matrix[j]))
                    agree[i, j] = agree[j, i] = round(ag, 2)
                    kappa[i, j] = kappa[j, i] = round(kp, 4)

        return (
            pd.DataFrame(agree, index=model_names, columns=model_names),
            pd.DataFrame(kappa, index=model_names, columns=model_names),
        )

    def vote_pseudo_labels(
        self,
        per_model_files: Dict[str, Path],
        unlabeled_df: pd.DataFrame,
        output_dir: Path,
        round_tag: str = "r1",
        text_col: str = "vi_text",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Tổng hợp kết quả của tất cả mô hình trên tập không nhãn và lọc nhãn giả đạt chuẩn.

        Returns:
            (ensemble_pseudo_df, remain_df, report_dict)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cổng kiểm định chất lượng: loại bỏ tự động mô hình bị collapse (Val Macro-F1 < min_model_val_f1)
        excluded_models = []
        valid_model_names = []
        for m in per_model_files.keys():
            val_score = self.model_weights.get(m, None)
            if val_score is not None and float(val_score) < self.min_model_val_f1:
                excluded_models.append((m, float(val_score)))
                print(
                    f"[Cảnh báo Quality Gate] Loại {m} khỏi Ensemble Voting do Val Macro-F1 = "
                    f"{val_score*100:.2f}% < ngưỡng {self.min_model_val_f1*100:.0f}% (dấu hiệu collapse)."
                )
            else:
                valid_model_names.append(m)

        model_names = valid_model_names
        n_models = len(model_names)
        if n_models < 2:
            raise ValueError(f"Cần ít nhất 2 mô hình đạt chuẩn để thực hiện Ensemble Voting (hiện có {n_models}).")

        min_votes = int(math.ceil(n_models * self.min_vote_ratio))
        weights = self._get_normalized_weights(model_names)

        # Căn chỉnh toàn bộ dự đoán theo đúng thứ tự `text_id` của `unlabeled_df`
        base_ids = unlabeled_df["text_id"].values
        n_samples = len(base_ids)

        probs_stack = np.zeros((n_models, n_samples, NUM_LABELS), dtype=np.float32)
        preds_stack = np.zeros((n_models, n_samples), dtype=np.int32)
        prob_cols = [f"prob_{c}" for c in range(NUM_LABELS)]

        for idx, name in enumerate(model_names):
            df_m = pd.read_parquet(per_model_files[name]).set_index("text_id").loc[base_ids]
            probs_stack[idx] = df_m[prob_cols].to_numpy(dtype=np.float32)
            preds_stack[idx] = df_m["pred_label"].to_numpy(dtype=np.int32)

        # 1. Tính Ma trận đồng thuận cặp & Cohen's Kappa giữa các mô hình
        agree_df, kappa_df = self.compute_pairwise_agreement(preds_stack, model_names)

        # 2. Hard Majority Voting (đếm số phiếu bầu cho từng lớp 0..3)
        vote_counts_per_class = np.zeros((n_samples, NUM_LABELS), dtype=np.int32)
        for c in range(NUM_LABELS):
            vote_counts_per_class[:, c] = (preds_stack == c).sum(axis=0)
        hard_label = np.argmax(vote_counts_per_class, axis=1)
        max_votes = np.max(vote_counts_per_class, axis=1)

        # 3. Weighted Soft Voting: trung bình xác suất có trọng số theo Val Macro-F1
        weighted_probs = np.tensordot(weights, probs_stack, axes=([0], [0])).astype(np.float32)
        soft_label = np.argmax(weighted_probs, axis=1)
        sorted_soft = np.sort(weighted_probs, axis=1)
        ensemble_conf = sorted_soft[:, -1]
        ensemble_margin = sorted_soft[:, -1] - sorted_soft[:, -2]
        ensemble_entropy = -np.sum(
            weighted_probs * np.log(np.clip(weighted_probs, 1e-12, 1.0)), axis=1
        )

        # Độ tin cậy trung bình chỉ tính trên các mô hình đồng ý với nhãn hard_label
        agree_mask = preds_stack == hard_label[None, :]
        # Lấy xác suất của nhãn hard_label ở từng mô hình
        row_idx = np.arange(n_samples)
        model_probs_for_hard = np.array(
            [probs_stack[m, row_idx, hard_label] for m in range(n_models)]
        )
        agree_conf_sum = (model_probs_for_hard * agree_mask).sum(axis=0)
        agree_conf_mean = agree_conf_sum / np.maximum(max_votes, 1)

        # 4. Áp dụng 4 điều kiện lọc đồng thuận nghiêm ngặt
        keep_mask = (
            (max_votes >= min_votes)
            & (hard_label == soft_label)
            & (ensemble_conf >= self.confidence_threshold)
            & (ensemble_margin >= self.min_margin)
        )

        # Thống kê phân bố số phiếu bầu trên TOÀN BỘ tập không nhãn (để vẽ biểu đồ B4_03)
        vote_dist_all = {
            int(v): int((max_votes == v).sum()) for v in range(1, n_models + 1)
        }
        vote_dist_kept = {
            int(v): int(((max_votes == v) & keep_mask).sum()) for v in range(1, n_models + 1)
        }

        # Xây dựng DataFrame kết quả nhãn giả đạt chuẩn (sắp xếp giảm dần theo độ tin cậy)
        ensemble_df = pd.DataFrame({
            "text_id": base_ids[keep_mask],
            text_col: unlabeled_df[text_col].values[keep_mask],
            "labels": hard_label[keep_mask].astype(int),
            "label_name": [ID2LABEL[int(c)] for c in hard_label[keep_mask]],
            "confidence": ensemble_conf[keep_mask].astype(np.float32),
            "agree_avg_confidence": agree_conf_mean[keep_mask].astype(np.float32),
            "margin": ensemble_margin[keep_mask].astype(np.float32),
            "entropy": ensemble_entropy[keep_mask].astype(np.float32),
            "vote_count": max_votes[keep_mask].astype(int),
            "total_models": n_models,
            "prob_0": weighted_probs[keep_mask, 0],
            "prob_1": weighted_probs[keep_mask, 1],
            "prob_2": weighted_probs[keep_mask, 2],
            "prob_3": weighted_probs[keep_mask, 3],
            "pseudo_labeled": True,
            "source": f"ensemble_{round_tag}",
        }).sort_values(
            by=["confidence", "vote_count", "margin"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        # Phần dữ liệu chưa đạt chuẩn giữ lại cho vòng kế tiếp
        remain_df = unlabeled_df.loc[~keep_mask, ["text_id", text_col]].reset_index(drop=True)

        # Lưu file đầu ra
        pseudo_csv = output_dir / f"ensemble_pseudo_{round_tag}.csv"
        remain_csv = output_dir / f"remain_data_{round_tag}.csv"
        report_json = output_dir / f"ensemble_report_{round_tag}.json"

        ensemble_df.to_csv(pseudo_csv, index=False, encoding="utf-8-sig")
        remain_df.to_csv(remain_csv, index=False, encoding="utf-8-sig")

        report: Dict = {
            "round_tag": round_tag,
            "models": model_names,
            "model_weights": {m: round(float(w), 4) for m, w in zip(model_names, weights)},
            "min_votes_required": min_votes,
            "confidence_threshold": self.confidence_threshold,
            "min_margin": self.min_margin,
            "input_unlabeled_count": int(n_samples),
            "selected_pseudo_count": int(len(ensemble_df)),
            "selected_ratio_pct": round(float(len(ensemble_df) / max(n_samples, 1) * 100.0), 2),
            "remain_count": int(len(remain_df)),
            "avg_confidence": round(float(ensemble_df["confidence"].mean()), 4) if len(ensemble_df) else 0.0,
            "avg_margin": round(float(ensemble_df["margin"].mean()), 4) if len(ensemble_df) else 0.0,
            "avg_entropy": round(float(ensemble_df["entropy"].mean()), 4) if len(ensemble_df) else 0.0,
            "label_distribution": {
                str(c): int((ensemble_df["labels"] == c).sum()) for c in range(NUM_LABELS)
            },
            "label_name_distribution": {
                ID2LABEL[c]: int((ensemble_df["labels"] == c).sum()) for c in range(NUM_LABELS)
            },
            "vote_distribution_all": {str(k): v for k, v in vote_dist_all.items()},
            "vote_distribution_kept": {str(k): v for k, v in vote_dist_kept.items()},
            "pairwise_agreement_pct": agree_df.to_dict(),
            "pairwise_cohen_kappa": kappa_df.to_dict(),
            "mean_pairwise_kappa": round(
                float(kappa_df.values[np.triu_indices(n_models, k=1)].mean()), 4
            ),
        }

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 76)
        print(f"[ENSEMBLE VOTING {round_tag.upper()}] Hoàn tất bỏ phiếu {n_models} mô hình!")
        print(
            f"  - Điều kiện lọc : vote >= {min_votes}/{n_models} | conf >= {self.confidence_threshold} "
            f"| margin >= {self.min_margin} | Hard == Soft"
        )
        print(
            f"  - Kết quả       : Giữ {len(ensemble_df):,}/{n_samples:,} mẫu "
            f"({report['selected_ratio_pct']}%) | Còn dư {len(remain_df):,} mẫu"
        )
        print(
            f"  - Chất lượng    : Avg Conf={report['avg_confidence']:.4f} | "
            f"Avg Margin={report['avg_margin']:.4f} | Mean Kappa={report['mean_pairwise_kappa']:.4f}"
        )
        print(f"  - Phân bố nhãn  : {report['label_name_distribution']}")
        print("=" * 76)

        return ensemble_df, remain_df, report

    def predict_ensemble_test(
        self,
        per_model_probs: Dict[str, np.ndarray],
        strategy: str = "weighted_soft",
    ) -> np.ndarray:
        """
        Kết hợp ma trận xác suất (N, 4) của nhiều mô hình trên tập Test cho Bước B7 & Demo.

        Hỗ trợ 3 chiến lược:
        - `'weighted_soft'`: Trung bình xác suất có trọng số theo `Val Macro-F1`.
        - `'equal_soft'`: Trung bình xác suất đồng đều giữa các mô hình.
        - `'hard_voting'`: Bỏ phiếu đa số (phá vỡ hòa bằng weighted soft probabilities).
        """
        model_names = list(per_model_probs.keys())
        probs_stack = np.stack([per_model_probs[m] for m in model_names], axis=0)

        if strategy == "equal_soft":
            return probs_stack.mean(axis=0).astype(np.float32)

        weights = self._get_normalized_weights(model_names)
        weighted_probs = np.tensordot(weights, probs_stack, axes=([0], [0])).astype(np.float32)

        if strategy == "weighted_soft":
            return weighted_probs

        if strategy == "hard_voting":
            preds = np.argmax(probs_stack, axis=2)  # (M, N)
            n_samples = preds.shape[1]
            vote_mat = np.zeros((n_samples, NUM_LABELS), dtype=np.float32)
            for c in range(NUM_LABELS):
                vote_mat[:, c] = (preds == c).sum(axis=0)
            # Cộng thêm trọng số nhỏ từ weighted_probs để phá vỡ hòa mượt mà và giữ ý nghĩa xác suất
            combined = vote_mat / len(model_names) + 1e-4 * weighted_probs
            return combined / combined.sum(axis=1, keepdims=True)

        raise ValueError(f"Chiến lược Ensemble không hợp lệ: {strategy!r}")
