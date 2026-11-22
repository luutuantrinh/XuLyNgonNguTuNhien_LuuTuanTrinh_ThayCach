"""
Module Gộp dữ liệu có chọn lọc & cân bằng lớp (Selective Class-Balanced Combiner).

Khắc phục triệt để các lỗi của mã nguồn cũ:
- Luôn chọn đúng `head(count)` mẫu có độ tin cậy cao nhất của từng lớp (từ file `ensemble_pseudo`
  đã sắp xếp giảm dần theo `confidence`).
- Tính thống kê chính xác số lượng thực thêm (`len(selected_data)`, sửa lỗi cộng nhầm toàn bộ lớp).
- Hỗ trợ gợi ý tự động số lượng cần thêm để kéo 4 lớp về mục tiêu cân bằng (`target_per_class`),
  đồng thời cho phép người dùng ghi đè thủ công (`add_counts`) ngay trong Notebook.
- Xáo trộn ngẫu nhiên với `seed` cố định để các batch huấn luyện phân bố đều cả 4 lớp.
"""

from pathlib import Path
from typing import Dict, Tuple
import json

import pandas as pd

from src.config import ID2LABEL, NUM_LABELS


class SelectiveDataCombiner:
    """Gộp tập huấn luyện hiện có với các mẫu nhãn giả sạch nhất theo từng lớp ESG."""

    def __init__(self, text_col: str = "vi_text", label_col: str = "labels", seed: int = 42):
        self.text_col = text_col
        self.label_col = label_col
        self.seed = seed

    def suggest_balanced_counts(
        self,
        original_df: pd.DataFrame,
        pseudo_df: pd.DataFrame,
        target_per_class: int,
    ) -> Dict[int, int]:
        """
        Tự động đề xuất số mẫu pseudo cần thêm cho từng lớp để tiến gần mốc `target_per_class`.

        Quy tắc:
        - Nếu lớp gốc đã >= `target_per_class`: chỉ thêm một lượng nhỏ hoặc 0 (không làm tăng thêm mất cân bằng).
        - Nếu lớp gốc < `target_per_class`: lấy tối đa `min(còn thiếu, số lượng khả dụng trong pseudo_df)`.
        """
        orig_counts = original_df[self.label_col].value_counts().to_dict()
        pseudo_counts = pseudo_df[self.label_col].value_counts().to_dict()

        suggested: Dict[int, int] = {}
        print("=== GỢI Ý SỐ LƯỢNG GỘP CÂN BẰNG LỚP ===")
        print(f"{'Lớp':<18} | {'Hiện có':>8} | {'Pseudo có sẵn':>13} | {'Đề xuất thêm':>12} | {'Tổng sau gộp':>12}")
        print("-" * 74)
        for c in range(NUM_LABELS):
            have = int(orig_counts.get(c, 0))
            avail = int(pseudo_counts.get(c, 0))
            needed = max(0, target_per_class - have)
            add_n = min(needed, avail)
            suggested[c] = add_n
            print(
                f"{c} - {ID2LABEL[c]:<14} | {have:>8,} | {avail:>13,} | {add_n:>12,} | {have + add_n:>12,}"
            )
        return suggested

    def combine(
        self,
        original_df: pd.DataFrame,
        pseudo_df: pd.DataFrame,
        add_counts: Dict[int, int],
        output_dir: Path,
        round_name: str = "round1",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Gộp `original_df` với top-`add_counts[c]` mẫu có độ tin cậy cao nhất của từng lớp trong `pseudo_df`.

        Returns:
            (combined_df, remaining_pseudo_df, distribution_info)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        orig = original_df.copy()
        pseudo = pseudo_df.copy()

        # Đảm bảo pseudo đã được sắp xếp giảm dần theo confidence để .head(k) lấy mẫu sạch nhất
        if "confidence" in pseudo.columns:
            pseudo = pseudo.sort_values("confidence", ascending=False).reset_index(drop=True)

        # Gắn metadata chuẩn hóa cho phần dữ liệu gốc nếu chưa có
        if "source" not in orig.columns:
            orig["source"] = "original"
        if "pseudo_labeled" not in orig.columns:
            orig["pseudo_labeled"] = False
        if "confidence" not in orig.columns:
            orig["confidence"] = 1.0

        # Kiểm tra hợp lệ số lượng yêu cầu
        pseudo_counts = pseudo[self.label_col].value_counts().to_dict()
        for c, req in add_counts.items():
            avail = int(pseudo_counts.get(int(c), 0))
            if int(req) < 0 or int(req) > avail:
                raise ValueError(
                    f"Lớp {c} ({ID2LABEL[int(c)]}): yêu cầu thêm {req:,} mẫu nhưng chỉ có sẵn {avail:,} mẫu!"
                )

        # Chọn top-k mẫu cho từng lớp
        selected_parts = []
        used_indices = set()
        actual_added: Dict[int, int] = {}

        for c in range(NUM_LABELS):
            k = int(add_counts.get(c, 0))
            if k > 0:
                class_slice = pseudo[pseudo[self.label_col] == c].head(k).copy()
                class_slice["source"] = f"pseudo_{round_name}"
                class_slice["pseudo_labeled"] = True
                selected_parts.append(class_slice)
                used_indices.update(class_slice.index.tolist())
                actual_added[c] = int(len(class_slice))
            else:
                actual_added[c] = 0

        selected_pseudo_df = (
            pd.concat(selected_parts, ignore_index=True)
            if selected_parts
            else pd.DataFrame(columns=[self.text_col, self.label_col, "confidence", "pseudo_labeled", "source"])
        )
        remaining_pseudo_df = pseudo.drop(index=list(used_indices)).reset_index(drop=True)

        # Chỉ giữ các cột chuẩn thống nhất và gộp lại
        keep_cols = [self.text_col, self.label_col, "confidence", "pseudo_labeled", "source"]
        combined_df = pd.concat(
            [orig[keep_cols], selected_pseudo_df[keep_cols]],
            ignore_index=True,
        )
        # Xáo trộn ngẫu nhiên với seed cố định
        combined_df = combined_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        combined_df[self.label_col] = combined_df[self.label_col].astype(int)

        # Xây dựng báo cáo phân bố chi tiết (chính xác 100%)
        orig_counts = orig[self.label_col].value_counts().to_dict()
        final_counts = combined_df[self.label_col].value_counts().to_dict()
        remain_counts = remaining_pseudo_df[self.label_col].value_counts().to_dict()

        distribution_info: Dict = {
            "round_name": round_name,
            "original_total": int(len(orig)),
            "added_pseudo_total": int(len(selected_pseudo_df)),
            "combined_total": int(len(combined_df)),
            "remaining_pseudo_total": int(len(remaining_pseudo_df)),
            "per_class": {},
            "source_breakdown": combined_df["source"].value_counts().to_dict(),
        }

        print(f"\n=== KẾT QUẢ GỘP DỮ LIỆU ({round_name.upper()}) ===")
        for c in range(NUM_LABELS):
            o_n = int(orig_counts.get(c, 0))
            a_n = int(actual_added.get(c, 0))
            f_n = int(final_counts.get(c, 0))
            r_n = int(remain_counts.get(c, 0))
            distribution_info["per_class"][str(c)] = {
                "label_name": ID2LABEL[c],
                "before": o_n,
                "added_pseudo": a_n,
                "after": f_n,
                "remaining_pseudo": r_n,
            }
            print(
                f"  Lớp {c} ({ID2LABEL[c]:<11}): {o_n:>6,} (trước) + {a_n:>5,} (pseudo) "
                f"= {f_n:>6,} (sau) | Dư pseudo: {r_n:>5,}"
            )

        # Tính tỉ số mất cân bằng (Imbalance Ratio) trước & sau khi gộp
        ir_before = max(orig_counts.values()) / max(min(orig_counts.values()), 1)
        ir_after = max(final_counts.values()) / max(min(final_counts.values()), 1)
        distribution_info["imbalance_ratio_before"] = round(float(ir_before), 3)
        distribution_info["imbalance_ratio_after"] = round(float(ir_after), 3)
        print(
            f"  => Tổng mẫu: {len(orig):,} -> {len(combined_df):,} (+{len(selected_pseudo_df):,} pseudo) | "
            f"Imbalance Ratio (IR): {ir_before:.2f} -> {ir_after:.2f}"
        )

        # Lưu ra đĩa
        combined_csv = output_dir / f"combined_{round_name}.csv"
        remain_pseudo_csv = output_dir / f"remaining_pseudo_{round_name}.csv"
        dist_json = output_dir / f"distribution_{round_name}.json"

        combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        remaining_pseudo_df.to_csv(remain_pseudo_csv, index=False, encoding="utf-8-sig")
        with open(dist_json, "w", encoding="utf-8") as f:
            json.dump(distribution_info, f, ensure_ascii=False, indent=2)

        print(f"  [Đã lưu] {combined_csv.name}, {remain_pseudo_csv.name}, {dist_json.name}")
        return combined_df, remaining_pseudo_df, distribution_info
