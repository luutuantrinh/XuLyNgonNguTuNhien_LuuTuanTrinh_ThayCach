"""
Bộ công cụ trực quan hóa dữ liệu & kết quả chuẩn báo cáo đồ án cuối khóa (300 DPI PNG + PDF).

Mọi biểu đồ đều:
- Sử dụng bảng màu ESG nhất quán (Irrelevant, Environment, Social, Governance).
- Hỗ trợ hiển thị tiếng Việt có dấu sắc nét trên Windows.
- Tự động lưu đồng thời 2 định dạng `.png` (300 DPI cho Word/Slide) và `.pdf` (vector cho LaTeX)
  vào đúng thư mục con trong `workspace_outputs/figures/B2_eda/` ... `B8_final_report/`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ID2LABEL, LABEL_NAMES


# Bảng màu đồng nhất cho 4 lớp ESG xuyên suốt toàn bộ đồ án
ESG_PALETTE: Dict[str, str] = {
    "Irrelevant": "#6C757D",   # Xám trung tính
    "Environment": "#2E8B57",  # Xanh lá môi trường
    "Social": "#1F77B4",       # Xanh dương xã hội
    "Governance": "#D95F02",   # Cam đất quản trị
}

# Danh sách từ dừng (stopword) tiếng Việt cơ bản để lọc nhiễu khi vẽ Top N-gram & WordCloud
VIETNAMESE_STOPWORDS = {
    "và", "của", "các", "có", "là", "trong", "cho", "được", "với", "những", "này",
    "một", "đã", "tại", "từ", "để", "về", "như", "trên", "khi", "theo", "không",
    "cũng", "đến", "còn", "nhiều", "người", "năm", "sẽ", "đang", "hay", "hoặc",
    "nhưng", "vì", "nên", "nếu", "thì", "bị", "do", "tại", "bằng", "qua", "ra",
    "vào", "lại", "sau", "trước", "giữa", "cùng", "đó", "đây", "tôi", "chúng",
    "chúng tôi", "công", "ty", "công ty", "việc", "mình", "họ", "các bạn", "rằng",
    "nhằm", "gồm", "bao", "bao gồm", "tất", "cả", "tất cả", "mọi", "mỗi", "hơn",
    "nhất", "rất", "quá", "chỉ", "đều", "được", "phải", "cần", "thể", "có thể",
    "nào", "nơi", "đâu", "làm", "tạo", "đem", "mang", "sự", "tính", "mức", "số",
    "theo đó", "ngoài", "ngoài ra", "tuy", "nhiên", "tuy nhiên", "trong đó", "đồng",
    "thời", "đồng thời", "nhưng", "vẫn", "chưa", "hiện", "nay", "hiện nay", "trở",
    "lên", "xuống", "đi", "tới", "bởi", "vậy", "thế", "nên", "mà", "thực", "hiện",
    "hoạt", "động", "các hoạt", "bên", "dưới", "trên", "nêu", "nói", "biết", "thấy",
}


def setup_plot_style() -> None:
    """
    Thiết lập phong cách biểu đồ chuẩn báo cáo khoa học, hỗ trợ đầy đủ tiếng Việt trên Windows.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        # Ưu tiên các font phổ biến trên Windows có hỗ trợ đầy đủ dấu tiếng Việt
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Tahoma"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def save_figure(
    fig: plt.Figure,
    save_dir: Path,
    filename_stem: str,
    dpi: int = 300,
    save_pdf: bool = True,
) -> List[Path]:
    """
    Lưu biểu đồ ra file PNG (300 DPI) và tùy chọn thêm file PDF (vector) phục vụ báo cáo.

    Returns:
        Danh sách đường dẫn các file hình ảnh đã lưu.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    png_path = save_dir / f"{filename_stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    saved_paths.append(png_path)

    if save_pdf:
        pdf_path = save_dir / f"{filename_stem}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_paths.append(pdf_path)

    print(f"[Đã lưu biểu đồ] {', '.join(str(p.name) for p in saved_paths)} -> {save_dir}")
    return saved_paths


# ============================================================================
# NHÓM BIỂU ĐỒ ĐỢT 1 — EDA & KIỂM TOÁN DỮ LIỆU (NOTEBOOK B2)
# ============================================================================
def plot_class_distribution_comparison(
    datasets: Dict[str, pd.DataFrame],
    label_col: str = "labels",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B2_01_phan_bo_lop_esg",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B2.1: So sánh phân bố 4 lớp ESG trên các tập dữ liệu (Train Pool, Validation, Test)
    kèm cả số lượng mẫu tuyệt đối và tỉ lệ phần trăm trên đỉnh mỗi cột.
    """
    setup_plot_style()
    n_sets = len(datasets)
    fig, axes = plt.subplots(1, n_sets, figsize=(5.5 * n_sets, 5.2), sharey=False)
    if n_sets == 1:
        axes = [axes]

    colors = [ESG_PALETTE[name] for name in LABEL_NAMES]

    for ax, (title, df) in zip(axes, datasets.items()):
        counts = df[label_col].value_counts().reindex(range(len(LABEL_NAMES)), fill_value=0)
        total = int(counts.sum())
        bars = ax.bar(LABEL_NAMES, counts.values, color=colors, edgecolor="#333333", linewidth=0.7)

        ax.set_title(f"{title}\n(Tổng: {total:,} mẫu)", pad=12)
        ax.set_xlabel("Nhãn ESG")
        ax.set_ylabel("Số lượng mẫu")
        ax.set_ylim(0, max(counts.values) * 1.22 if total > 0 else 1)
        ax.tick_params(axis="x", rotation=12)

        for bar, val in zip(bars, counts.values):
            pct = (val / total * 100.0) if total > 0 else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(counts.values) * 0.025,
                f"{int(val):,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle("Phân bố 4 lớp ESG trên các tập dữ liệu sau khi làm sạch & chia chuẩn", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_text_length_distribution(
    train_df: pd.DataFrame,
    unlabeled_df: Optional[pd.DataFrame] = None,
    text_col: str = "vi_text",
    label_col: str = "labels",
    max_length_ref: int = 256,
    save_dir: Optional[Path] = None,
    filename_stem: str = "B2_02_phan_bo_do_dai_van_ban",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B2.2: Phân bố độ dài câu (theo số từ) của tập Train theo từng lớp ESG và tập Unlabeled,
    đánh dấu phân vị 95%, 99% và ngưỡng tham chiếu `max_length=256`.
    """
    setup_plot_style()
    work = train_df[[text_col, label_col]].copy()
    work["word_count"] = work[text_col].astype(str).apply(lambda s: len(s.split()))
    work["label_name"] = work[label_col].map(ID2LABEL)

    p95 = float(np.percentile(work["word_count"], 95))
    p99 = float(np.percentile(work["word_count"], 99))
    pct_under_256 = float((work["word_count"] <= max_length_ref).mean() * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # Trái: Histogram + KDE tổng thể (so sánh Train và Unlabeled nếu có)
    ax0 = axes[0]
    clip_upper = int(max(p99 * 1.25, 120))
    sns.histplot(
        work["word_count"].clip(upper=clip_upper),
        bins=45,
        kde=True,
        stat="density",
        color="#1F77B4",
        alpha=0.45,
        label=f"Tập Train sạch (n={len(work):,})",
        ax=ax0,
    )
    if unlabeled_df is not None and len(unlabeled_df) > 0:
        unl_wc = unlabeled_df[text_col].astype(str).apply(lambda s: len(s.split()))
        sns.kdeplot(
            unl_wc.clip(upper=clip_upper),
            color="#D95F02",
            linewidth=2.2,
            label=f"Tập Unlabeled sạch (n={len(unl_wc):,})",
            ax=ax0,
        )

    ax0.axvline(p95, color="#2E8B57", linestyle="--", linewidth=1.6, label=f"Phân vị 95% = {p95:.0f} từ")
    ax0.axvline(p99, color="#8E44AD", linestyle=":", linewidth=1.8, label=f"Phân vị 99% = {p99:.0f} từ")
    if max_length_ref <= clip_upper:
        ax0.axvline(max_length_ref, color="#C0392B", linestyle="-.", linewidth=1.8, label=f"Ngưỡng tham chiếu = {max_length_ref}")
    ax0.set_title(f"Phân bố độ dài câu theo số từ\n({pct_under_256:.2f}% câu Train ≤ {max_length_ref} từ)", pad=10)
    ax0.set_xlabel(f"Số từ trong câu (cắt hiển thị tại {clip_upper} từ)")
    ax0.set_ylabel("Mật độ phân bố (Density)")
    ax0.legend(loc="upper right", frameon=True)

    # Phải: Boxplot độ dài theo từng lớp ESG
    ax1 = axes[1]
    sns.boxplot(
        data=work,
        x="label_name",
        y="word_count",
        order=LABEL_NAMES,
        hue="label_name",
        palette=ESG_PALETTE,
        legend=False,
        showfliers=False,
        ax=ax1,
    )
    ax1.set_title("So sánh độ dài câu theo từng lớp ESG (ẩn ngoại lai)", pad=10)
    ax1.set_xlabel("Nhãn ESG")
    ax1.set_ylabel("Số từ trong câu")

    # Ghi giá trị trung vị (median) lên từng hộp
    medians = work.groupby("label_name")["word_count"].median()
    for idx, name in enumerate(LABEL_NAMES):
        med = medians.get(name, 0)
        ax1.text(idx, med + 1.2, f"Med={med:.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.suptitle("Phân tích độ dài văn bản — Cơ sở chọn max_length = 256 token", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_top_keywords_by_class(
    df: pd.DataFrame,
    text_col: str = "vi_text",
    label_col: str = "labels",
    top_n: int = 12,
    save_dir: Optional[Path] = None,
    filename_stem: str = "B2_03_tu_khoa_dac_trung_4_lop",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B2.3: Top-N từ/cụm từ (1-2 gram) có điểm TF-IDF trung bình cao nhất của từng lớp ESG.
    Giúp minh họa trực quan đặc trưng từ vựng phân biệt 4 lớp trong báo cáo.
    """
    setup_plot_style()

    # Tách từ tiếng Việt đơn giản, loại bỏ chữ số thuần và stopword
    def _clean_for_tfidf(s: str) -> str:
        s = re.sub(r"\d+", " ", str(s).lower())
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        tokens = [w for w in s.split() if len(w) > 1 and w not in VIETNAMESE_STOPWORDS]
        return " ".join(tokens)

    corpus = df[text_col].apply(_clean_for_tfidf)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=15_000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes_flat = axes.flatten()

    for idx, label_name in enumerate(LABEL_NAMES):
        ax = axes_flat[idx]
        mask = (df[label_col].values == idx)
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        # Điểm đặc trưng = trung bình TF-IDF lớp đó trừ đi 0.35 * trung bình các lớp còn lại
        mean_in = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        mean_out = np.asarray(tfidf_matrix[~mask].mean(axis=0)).ravel()
        distinctive_score = mean_in - 0.35 * mean_out

        top_indices = np.argsort(distinctive_score)[-top_n:]
        top_terms = feature_names[top_indices]
        top_scores = distinctive_score[top_indices]

        ax.barh(top_terms, top_scores, color=ESG_PALETTE[label_name], edgecolor="#333333", linewidth=0.6)
        ax.set_title(f"Lớp {idx}: {label_name} (n={int(mask.sum()):,})", pad=8)
        ax.set_xlabel("Điểm đặc trưng TF-IDF")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(f"Top-{top_n} từ/cụm từ đặc trưng nhất của từng lớp ESG (TF-IDF 1–2 gram)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_wordclouds_by_class(
    df: pd.DataFrame,
    text_col: str = "vi_text",
    label_col: str = "labels",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B2_04_wordcloud_4_lop_esg",
    dpi: int = 300,
    save_pdf: bool = True,
) -> Optional[plt.Figure]:
    """
    Hình B2.4: Vẽ 4 WordCloud tiếng Việt cho 4 lớp ESG (tự động bỏ qua an toàn nếu chưa cài `wordcloud`).
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[Cảnh báo] Chưa cài gói 'wordcloud'. Bỏ qua hình WordCloud (cài bằng: pip install wordcloud).")
        return None

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes_flat = axes.flatten()

    # Tìm font hỗ trợ tiếng Việt trên Windows cho WordCloud
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)

    colormaps = ["Greys", "Greens", "Blues", "Oranges"]
    for idx, label_name in enumerate(LABEL_NAMES):
        ax = axes_flat[idx]
        texts = df.loc[df[label_col] == idx, text_col].astype(str)
        combined_text = " ".join(texts.tolist()).lower()
        combined_text = re.sub(r"\d+|[^\w\s]", " ", combined_text, flags=re.UNICODE)

        wc = WordCloud(
            width=800,
            height=420,
            background_color="white",
            stopwords=VIETNAMESE_STOPWORDS,
            colormap=colormaps[idx],
            max_words=80,
            font_path=font_path,
            random_state=42,
        ).generate(combined_text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"WordCloud — Lớp {idx}: {label_name}", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle("Mây từ vựng (WordCloud) theo 4 nhóm chủ đề ESG", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_data_cleaning_audit(
    audit_report: Dict,
    save_dir: Optional[Path] = None,
    filename_stem: str = "B2_05_kiem_toan_lam_sach_du_lieu",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B2.5: Biểu đồ kiểm toán dữ liệu trước và sau làm sạch + khử trùng lặp/chống rò rỉ
    cho tập Train (`train.csv`) và tập Unlabeled (`overall_data.csv`).
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # Trái: Tập Train
    train_stages = [
        "Gốc\n(train.csv)",
        "Bỏ rỗng/\ncâu ngắn",
        "Bỏ mâu thuẫn\nnhãn",
        "Bỏ trùng\nnội bộ",
        "Bỏ rò rỉ\nvới Test",
        "Sạch\n(train_clean)",
    ]
    train_vals = [
        audit_report["train_raw_rows"],
        audit_report["train_removed_short_or_empty"],
        audit_report["train_removed_label_conflicts"],
        audit_report["train_removed_internal_duplicates"],
        audit_report["train_removed_test_leakage"],
        audit_report["train_clean_rows"],
    ]
    train_colors = ["#1F77B4", "#E67E22", "#E74C3C", "#F39C12", "#C0392B", "#2E8B57"]
    bars0 = axes[0].bar(train_stages, train_vals, color=train_colors, edgecolor="#333333", linewidth=0.7)
    axes[0].set_title("Kiểm toán chất lượng tập Train (Zero Data Leakage)", pad=10)
    axes[0].set_ylabel("Số lượng mẫu")
    axes[0].set_ylim(0, max(train_vals) * 1.18)
    for bar, v in zip(bars0, train_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + max(train_vals) * 0.02, f"{v:,}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    # Phải: Tập Unlabeled
    unl_stages = [
        "Gốc\n(overall_data)",
        "Bỏ rỗng/\ncâu ngắn",
        "Bỏ trùng\nnội bộ",
        "Bỏ trùng\nTest",
        "Bỏ trùng\nTrain",
        "Sạch\n(unlabeled)",
    ]
    unl_vals = [
        audit_report["unlabeled_raw_rows"],
        audit_report["unlabeled_removed_short_or_empty"],
        audit_report["unlabeled_removed_internal_duplicates"],
        audit_report["unlabeled_removed_test_leakage"],
        audit_report["unlabeled_removed_train_overlap"],
        audit_report["unlabeled_clean_rows"],
    ]
    unl_colors = ["#1F77B4", "#E67E22", "#F39C12", "#C0392B", "#8E44AD", "#2E8B57"]
    bars1 = axes[1].bar(unl_stages, unl_vals, color=unl_colors, edgecolor="#333333", linewidth=0.7)
    axes[1].set_title("Kiểm toán chất lượng tập Unlabeled (overall_data.csv)", pad=10)
    axes[1].set_ylabel("Số lượng mẫu")
    axes[1].set_ylim(0, max(unl_vals) * 1.18)
    for bar, v in zip(bars1, unl_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + max(unl_vals) * 0.02, f"{v:,}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    fig.suptitle("Báo cáo Kiểm toán Làm sạch & Chống Rò rỉ Dữ liệu", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


# ============================================================================
# NHÓM BIỂU ĐỒ ĐỢT 2 — LEARNING CURVES & HIỆU NĂNG BASELINE / RUN 1 (NOTEBOOK B3)
# ============================================================================
def plot_learning_curves_grid(
    histories: Dict[str, pd.DataFrame],
    title: str = "Đường cong huấn luyện (Learning Curves) — Train/Val Loss & Val Macro-F1",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B3_01_learning_curves_run1",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B3.1: Vẽ lưới Learning Curves cho tất cả mô hình trong một vòng huấn luyện.
    Trục trái hiển thị Train Loss & Val Loss; trục phải hiển thị Val Macro-F1 kèm chấm sao tại Best Epoch.
    """
    setup_plot_style()
    model_names = [m for m, df in histories.items() if df is not None and not df.empty]
    n_models = len(model_names)
    if n_models == 0:
        raise ValueError("Không có dữ liệu lịch sử huấn luyện nào để vẽ Learning Curves.")

    n_cols = min(3, n_models) if n_models <= 6 else 4
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, name in enumerate(model_names):
        ax = axes_flat[idx]
        df = histories[name].sort_values("epoch")
        epochs = df["epoch"].values

        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"], marker="o", linewidth=2.0, color="#1F77B4", label="Train Loss")
        if "eval_loss" in df.columns:
            ax.plot(epochs, df["eval_loss"], marker="s", linestyle="--", linewidth=2.0, color="#D95F02", label="Val Loss")

        ax.set_title(name, fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="#333333")
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax.twinx()
        if "eval_f1_macro" in df.columns:
            f1_pct = df["eval_f1_macro"].values * 100.0
            ax2.plot(epochs, f1_pct, marker="^", linewidth=2.2, color="#2E8B57", label="Val Macro-F1 (%)")
            best_i = int(np.argmax(f1_pct))
            ax2.scatter([epochs[best_i]], [f1_pct[best_i]], color="#C0392B", s=110, marker="*", zorder=6, label=f"Best ({f1_pct[best_i]:.2f}%)")
            ax2.set_ylabel("Val Macro-F1 (%)", color="#2E8B57")
            ax2.set_ylim(max(0, min(f1_pct) - 6), min(100, max(f1_pct) + 4))
            ax2.grid(False)

        # Gom chú giải từ cả 2 trục
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.2, frameon=True, framealpha=0.88)

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_run_benchmark_comparison(
    summary_df: pd.DataFrame,
    title: str = "So sánh Hiệu năng trên Hold-out Test Set (Ablation Baseline vs Run 1)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B3_02_so_sanh_hieu_nang_run1",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B3.2: Biểu đồ cột nhóm so sánh Accuracy (%) và Macro-F1 (%) trên `test_clean.csv`
    giữa các mô hình Run 1 (10k subset) và Ablation Baseline (24.6k train_pool).
    """
    setup_plot_style()
    df = summary_df.copy()
    df["label"] = df["run_tag"].str.upper() + " — " + df["model_display_name"]
    df = df.sort_values("f1_macro_pct", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.55 * len(df) + 1.8)))
    y = np.arange(len(df))
    h = 0.36

    bars_f1 = ax.barh(y + h / 2, df["f1_macro_pct"], height=h, color="#2E8B57", edgecolor="#333333", linewidth=0.6, label="Macro-F1 (%)")
    bars_acc = ax.barh(y - h / 2, df["accuracy_pct"], height=h, color="#1F77B4", edgecolor="#333333", linewidth=0.6, label="Accuracy (%)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=10)
    ax.set_xlabel("Điểm số trên Hold-out Test (%)")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)

    min_x = max(0, float(min(df["f1_macro_pct"].min(), df["accuracy_pct"].min())) - 8)
    ax.set_xlim(min_x, 102)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bars in (bars_f1, bars_acc):
        for bar in bars:
            val = bar.get_width()
            ax.text(val + 0.35, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


# ============================================================================
# NHÓM BIỂU ĐỒ ĐỢT 3 — CHẤT LƯỢNG PSEUDO-LABELING & ENSEMBLE (NOTEBOOK B4)
# ============================================================================
def plot_pseudo_confidence_distribution(
    per_model_confs: Dict[str, np.ndarray],
    threshold: float = 0.94,
    title: str = "Phân bố Độ tin cậy (Confidence) của các Mô hình trên Tập Không nhãn",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B4_01_phan_bo_confidence_vong1",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B4.1 / B6.3: Vẽ đường cong phân phối tích lũy ngược (1 - ECDF: Tỉ lệ % mẫu đạt Confidence >= x)
    kết hợp biểu đồ cột ngang tỉ lệ vượt ngưỡng lọc τ.

    Lý do dùng (1 - ECDF) thay cho KDE thuần: mô hình Transformer sau fine-tune có phân bố độ tin cậy
    lệch trái rất mạnh (70–94% mẫu dồn sát 0.99–1.00) khiến KDE dựng đỉnh cực cao tại 1.0 và ép phẳng
    toàn bộ phần còn lại. Đường cong (1 - ECDF) hiển thị sắc nét hành vi của cả 7 mô hình trên toàn miền.
    """
    setup_plot_style()
    names = list(per_model_confs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))

    # Trái: Đường cong tích lũy ngược P(Confidence >= x) (%) trên miền [0.40, 1.00]
    ax0 = axes[0]
    palette = sns.color_palette("tab10", len(names))
    x_grid = np.linspace(0.40, 1.00, 300)

    for color, name in zip(palette, names):
        confs = np.asarray(per_model_confs[name], dtype=np.float32)
        if len(confs) == 0:
            continue
        # Tính tỉ lệ % mẫu có confidence >= x tại từng điểm lưới
        survival_pct = (confs[:, None] >= x_grid[None, :]).mean(axis=0) * 100.0
        pct_at_tau = float((confs >= threshold).mean() * 100.0)
        ax0.plot(
            x_grid,
            survival_pct,
            linewidth=2.3,
            color=color,
            label=f"{name} ({pct_at_tau:.1f}% ≥ {threshold:.2f})",
        )
        # Chấm tròn đánh dấu đúng giao điểm tại ngưỡng lọc τ
        ax0.scatter([threshold], [pct_at_tau], color=color, s=42, zorder=5, edgecolor="white", linewidth=0.8)

    ax0.axvline(threshold, color="#C0392B", linestyle="--", linewidth=2.0, label=f"Ngưỡng lọc τ = {threshold:.2f}")
    ax0.set_title("Đường cong Tích lũy Ngược — Tỉ lệ mẫu đạt Độ tin cậy ≥ x", pad=10)
    ax0.set_xlabel("Ngưỡng độ tin cậy x (Confidence)")
    ax0.set_ylabel("Tỉ lệ mẫu đạt Confidence ≥ x (%)")
    ax0.set_xlim(0.40, 1.005)
    ax0.set_ylim(-2, 105)
    ax0.grid(True, linestyle="--", alpha=0.45)
    ax0.legend(loc="lower left", fontsize=8.4, frameon=True, framealpha=0.92)

    # Phải: Tỉ lệ % mẫu vượt ngưỡng confidence theo từng mô hình
    ax1 = axes[1]
    pct_above = [float((np.asarray(per_model_confs[m]) >= threshold).mean() * 100.0) for m in names]
    bars = ax1.barh(names, pct_above, color=palette, edgecolor="#333333", linewidth=0.6)
    ax1.set_title(f"Tỉ lệ mẫu đạt ngưỡng tin cậy ≥ {threshold:.2f} (%)", pad=10)
    ax1.set_xlabel("Tỉ lệ mẫu vượt ngưỡng (%)")
    ax1.set_xlim(0, min(108, max(pct_above) * 1.18 + 6))
    for bar, v in zip(bars, pct_above):
        ax1.text(v + 0.9, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontweight="bold", fontsize=9.5)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_pairwise_agreement_heatmap(
    agreement_df: pd.DataFrame,
    kappa_df: pd.DataFrame,
    title: str = "Ma trận Đồng thuận Cặp giữa các Mô hình (Tỉ lệ % & Cohen's Kappa)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B4_02_ma_tran_dong_thuan_kappa",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B4.2: Heatmap kép so sánh Tỉ lệ đồng thuận (%) và Hệ số Cohen's Kappa giữa các mô hình.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    sns.heatmap(
        agreement_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=50,
        vmax=100,
        square=True,
        cbar_kws={"label": "Đồng thuận (%)"},
        ax=axes[0],
    )
    axes[0].set_title("Tỉ lệ dự đoán trùng khớp (%)", pad=10)
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(
        kappa_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.4,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Cohen's Kappa"},
        ax=axes[1],
    )
    axes[1].set_title("Hệ số đồng thuận Cohen's Kappa", pad=10)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_vote_count_distribution(
    vote_dist_all: Dict[str, int],
    vote_dist_kept: Dict[str, int],
    min_votes_required: int,
    total_models: int,
    save_dir: Optional[Path] = None,
    filename_stem: str = "B4_03_phan_bo_so_phieu_bau",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B4.3: Phân bố số lượng câu theo số phiếu đồng thuận (1/M -> M/M) — so sánh tổng số câu
    và số câu vượt qua bộ lọc Ensemble (conf + margin + Hard==Soft).
    """
    setup_plot_style()
    votes = list(range(1, total_models + 1))
    all_vals = [int(vote_dist_all.get(str(v), vote_dist_all.get(v, 0))) for v in votes]
    kept_vals = [int(vote_dist_kept.get(str(v), vote_dist_kept.get(v, 0))) for v in votes]
    labels = [f"{v}/{total_models}" for v in votes]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(votes))
    w = 0.38

    b1 = ax.bar(x - w / 2, all_vals, width=w, color="#95A5A6", edgecolor="#333333", linewidth=0.6, label="Tổng số câu không nhãn")
    b2 = ax.bar(x + w / 2, kept_vals, width=w, color="#2E8B57", edgecolor="#333333", linewidth=0.6, label="Đạt chuẩn giữ lại (Conf + Margin + Hard=Soft)")

    ax.axvline(min_votes_required - 1.5, color="#C0392B", linestyle="--", linewidth=1.8, label=f"Ngưỡng đồng thuận tối thiểu ({min_votes_required}/{total_models})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Mức đồng thuận giữa các mô hình (Số phiếu / Tổng số mô hình)")
    ax.set_ylabel("Số lượng câu")
    ax.set_title("Phân bố mức độ đồng thuận (Ensemble Consensus Voting)", fontsize=13.5, fontweight="bold", pad=12)
    ax.set_ylim(0, max(all_vals) * 1.18 if max(all_vals) > 0 else 10)
    ax.legend(loc="upper left", frameon=True)

    for bars in (b1, b2):
        for bar in bars:
            v = int(bar.get_height())
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + max(all_vals) * 0.015, f"{v:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_before_after_combine_distribution(
    distribution_info: Dict,
    title: str = "Hiệu chỉnh Mất cân bằng Lớp & Prior Shift sau Combine Vòng 1",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B4_04_can_bang_lop_sau_vong1",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B4.4: Biểu đồ cột chồng so sánh phân bố 4 lớp Trước (Original) và Sau khi bổ sung Pseudo Vòng 1.
    """
    setup_plot_style()
    per_class = distribution_info["per_class"]
    before_vals = [int(per_class[str(c)]["before"]) for c in range(len(LABEL_NAMES))]
    added_vals = [int(per_class[str(c)]["added_pseudo"]) for c in range(len(LABEL_NAMES))]
    after_vals = [int(per_class[str(c)]["after"]) for c in range(len(LABEL_NAMES))]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(LABEL_NAMES))

    ax.bar(x, before_vals, width=0.55, color="#1F77B4", edgecolor="#333333", linewidth=0.7, label="Dữ liệu gốc (Train Pool)")
    ax.bar(x, added_vals, bottom=before_vals, width=0.55, color="#F39C12", edgecolor="#333333", linewidth=0.7, label="Nhãn giả bổ sung (Pseudo Vòng 1)")

    max_h = max(after_vals) if after_vals else 1
    ax.set_ylim(0, max_h * 1.26)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu")
    ir_b = distribution_info.get("imbalance_ratio_before", 0)
    ir_a = distribution_info.get("imbalance_ratio_after", 0)
    ax.set_title(
        f"{title}\n(Tổng: {distribution_info['original_total']:,} → {distribution_info['combined_total']:,} mẫu | "
        f"Imbalance Ratio: {ir_b:.2f} → {ir_a:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=28,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True)

    for i in range(len(LABEL_NAMES)):
        ax.text(x[i], before_vals[i] / 2, f"{before_vals[i]:,}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        if added_vals[i] > max_h * 0.04:
            ax.text(x[i], before_vals[i] + added_vals[i] / 2, f"+{added_vals[i]:,}", ha="center", va="center", color="#222222", fontweight="bold", fontsize=9.5)
        ax.text(x[i], after_vals[i] + max_h * 0.02, f"Tổng: {after_vals[i]:,}", ha="center", va="bottom", color="#111111", fontweight="bold", fontsize=10)

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


# ============================================================================
# NHÓM BIỂU ĐỒ ĐỢT 3 — RUN 2 & PSEUDO-LABELING VÒNG 2 (NOTEBOOK B5 & B6)
# ============================================================================
def plot_run_delta_comparison(
    run1_df: pd.DataFrame,
    run2_df: pd.DataFrame,
    metric_col: str = "f1_macro_pct",
    metric_label: str = "Macro-F1 (%)",
    title: str = "Bước nhảy Hiệu năng trên Hold-out Test Set: Run 1 → Run 2",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B5_02_buoc_nhay_run1_sang_run2",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B5.2: Biểu đồ Dumbbell so sánh mức tăng trưởng (Delta Macro-F1 & Delta Accuracy)
    của từng mô hình từ Run 1 (10k subset) sang Run 2 (Combined Vòng 1).
    """
    setup_plot_style()
    m1 = run1_df[["model_display_name", "accuracy_pct", "f1_macro_pct"]].rename(
        columns={"accuracy_pct": "acc_r1", "f1_macro_pct": "f1_r1"}
    )
    m2 = run2_df[["model_display_name", "accuracy_pct", "f1_macro_pct"]].rename(
        columns={"accuracy_pct": "acc_r2", "f1_macro_pct": "f1_r2"}
    )
    merged = pd.merge(m1, m2, on="model_display_name", how="inner")
    merged["delta_f1"] = merged["f1_r2"] - merged["f1_r1"]
    merged["delta_acc"] = merged["acc_r2"] - merged["acc_r1"]
    merged = merged.sort_values("f1_r2", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5.2, 0.72 * len(merged) + 1.8)), sharey=True)
    y = np.arange(len(merged))

    for ax, c1, c2, dcol, subtitle in [
        (axes[0], "f1_r1", "f1_r2", "delta_f1", "Macro-F1 trên Hold-out Test (%)"),
        (axes[1], "acc_r1", "acc_r2", "delta_acc", "Accuracy trên Hold-out Test (%)"),
    ]:
        for i, row in merged.iterrows():
            v1, v2, dv = float(row[c1]), float(row[c2]), float(row[dcol])
            line_col = "#2E8B57" if dv >= 0 else "#C0392B"
            ax.plot([v1, v2], [i, i], color=line_col, linewidth=2.8, alpha=0.85, zorder=2)
            ax.scatter([v1], [i], color="#7F8C8D", s=85, zorder=3, label="Run 1" if i == 0 else "")
            ax.scatter([v2], [i], color="#1F77B4", s=110, zorder=4, label="Run 2" if i == 0 else "")
            sign = "+" if dv >= 0 else ""
            ax.text(
                max(v1, v2) + 0.8,
                i,
                f"{v2:.2f}% ({sign}{dv:.2f}%)",
                va="center",
                ha="left",
                fontsize=9.2,
                fontweight="bold",
                color=line_col,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(merged["model_display_name"], fontsize=10.5, fontweight="bold")
        ax.set_title(subtitle, pad=24)
        ax.set_xlabel("Điểm số (%)")
        min_v = float(min(merged[c1].min(), merged[c2].min()))
        ax.set_xlim(max(0, min_v - 5), 116)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_unlabeled_data_funnel(
    total_unlabeled: int,
    r1_selected: int,
    r2_selected: int,
    r2_remain: int,
    title: str = "Biểu đồ Phễu Khai thác Dữ liệu Không nhãn (Data Funnel qua 2 Vòng)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B6_01_bieu_do_pheu_unlabeled",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B6.1: Theo dõi dòng chảy 54.487 câu không nhãn qua Vòng 1 và Vòng 2 Pseudo-Labeling.
    """
    setup_plot_style()
    total_used = r1_selected + r2_selected
    stages = [
        "1. Tập Không nhãn sạch\n(unlabeled_clean.csv)",
        "2. Đạt chuẩn Ensemble Vòng 1\n(Conf ≥ 0.94, Margin ≥ 0.50)",
        "3. Còn lại sau Vòng 1\n(remain_data_r1.csv)",
        "4. Đạt chuẩn Ensemble Vòng 2\n(Conf ≥ 0.91, Margin ≥ 0.45)",
        "5. Tổng nhãn giả sạch\n(Vòng 1 + Vòng 2)",
        "6. Loại bỏ (Không đủ tin cậy)\n(remain_data_r2.csv)",
    ]
    values = [total_unlabeled, r1_selected, total_unlabeled - r1_selected, r2_selected, total_used, r2_remain]
    colors = ["#34495E", "#2E8B57", "#7F8C8D", "#1F77B4", "#8E44AD", "#E67E22"]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    y = np.arange(len(stages))[::-1]
    bars = ax.barh(y, values, color=colors, edgecolor="#333333", linewidth=0.7, height=0.62)

    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=10, fontweight="bold")
    ax.set_xlabel("Số lượng câu văn bản")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)
    ax.set_xlim(0, total_unlabeled * 1.22)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, v in zip(bars, values):
        pct = (v / max(total_unlabeled, 1)) * 100.0
        ax.text(
            v + total_unlabeled * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9.8,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_three_tier_class_distribution(
    dist_r1: Dict,
    dist_r2: Dict,
    title: str = "Phân bố Tập Huấn luyện Cuối cùng (Original + Pseudo Vòng 1 + Pseudo Vòng 2)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B6_02_phan_bo_du_lieu_3_tang",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B6.2: Biểu đồ cột chồng 3 tầng cho thấy quá trình mở rộng & cân bằng đều 4 lớp ESG
    từ Dữ liệu gốc -> +Pseudo Vòng 1 -> +Pseudo Vòng 2.
    """
    setup_plot_style()
    orig_vals = [int(dist_r1["per_class"][str(c)]["before"]) for c in range(len(LABEL_NAMES))]
    p1_vals = [int(dist_r1["per_class"][str(c)]["added_pseudo"]) for c in range(len(LABEL_NAMES))]
    p2_vals = [int(dist_r2["per_class"][str(c)]["added_pseudo"]) for c in range(len(LABEL_NAMES))]
    bottom_p2 = [orig_vals[i] + p1_vals[i] for i in range(len(LABEL_NAMES))]
    totals = [bottom_p2[i] + p2_vals[i] for i in range(len(LABEL_NAMES))]

    fig, ax = plt.subplots(figsize=(11, 6.0))
    x = np.arange(len(LABEL_NAMES))

    ax.bar(x, orig_vals, width=0.56, color="#1F77B4", edgecolor="#333333", linewidth=0.7, label="Dữ liệu gốc (Train Pool)")
    ax.bar(x, p1_vals, bottom=orig_vals, width=0.56, color="#F39C12", edgecolor="#333333", linewidth=0.7, label="Pseudo Vòng 1 (τ=0.94)")
    ax.bar(x, p2_vals, bottom=bottom_p2, width=0.56, color="#8E44AD", edgecolor="#333333", linewidth=0.7, label="Pseudo Vòng 2 (τ=0.91)")

    max_h = max(totals) if totals else 1
    ax.set_ylim(0, max_h * 1.26)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu")
    ir_orig = dist_r1.get("imbalance_ratio_before", 0)
    ir_final = dist_r2.get("imbalance_ratio_after", 0)
    ax.set_title(
        f"{title}\n(Tổng: {sum(orig_vals):,} → {sum(totals):,} mẫu | Imbalance Ratio: {ir_orig:.2f} → {ir_final:.2f})",
        fontsize=13,
        fontweight="bold",
        pad=28,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=True)

    for i in range(len(LABEL_NAMES)):
        ax.text(x[i], orig_vals[i] / 2, f"{orig_vals[i]:,}", ha="center", va="center", color="white", fontweight="bold", fontsize=9.8)
        if p1_vals[i] > max_h * 0.045:
            ax.text(x[i], orig_vals[i] + p1_vals[i] / 2, f"+{p1_vals[i]:,}", ha="center", va="center", color="#111111", fontweight="bold", fontsize=9.2)
        if p2_vals[i] > max_h * 0.045:
            ax.text(x[i], bottom_p2[i] + p2_vals[i] / 2, f"+{p2_vals[i]:,}", ha="center", va="center", color="white", fontweight="bold", fontsize=9.2)
        ax.text(x[i], totals[i] + max_h * 0.02, f"Tổng: {totals[i]:,}", ha="center", va="bottom", color="#111111", fontweight="bold", fontsize=10)

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


# ============================================================================
# NHÓM BIỂU ĐỒ ĐỢT 4 — RUN 3, FINAL ENSEMBLE & TỔNG KẾT BÁO CÁO (B7 & B8)
# ============================================================================
def plot_run3_and_ensemble_comparison(
    run3_and_ens_df: pd.DataFrame,
    title: str = "So sánh 7 Mô hình Đơn lẻ (Run 3) vs Các chiến lược Final Ensemble trên Test Set",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B7_01_so_sanh_don_le_va_ensemble",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B7.1: Biểu đồ cột ngang so sánh Macro-F1 (%) và Accuracy (%) giữa 7 mô hình Run 3
    và các chiến lược Final Ensemble (tô màu nổi bật cho nhóm Ensemble).
    """
    setup_plot_style()
    df = run3_and_ens_df.sort_values("f1_macro_pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.4, max(6.4, 0.58 * len(df) + 2.2)))
    y = np.arange(len(df))
    h = 0.36

    is_ens = df["run_tag"].astype(str).str.contains("ensemble", case=False)
    f1_colors = ["#8E44AD" if e else "#2E8B57" for e in is_ens]
    acc_colors = ["#D95F02" if e else "#1F77B4" for e in is_ens]

    bars_f1 = ax.barh(y + h / 2, df["f1_macro_pct"], height=h, color=f1_colors, edgecolor="#333333", linewidth=0.6)
    bars_acc = ax.barh(y - h / 2, df["accuracy_pct"], height=h, color=acc_colors, edgecolor="#333333", linewidth=0.6)

    # Tạo chú giải gọn theo màu sắc
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2E8B57", edgecolor="#333333", label="Macro-F1 — Mô hình đơn lẻ (Run 3)"),
        Patch(facecolor="#1F77B4", edgecolor="#333333", label="Accuracy — Mô hình đơn lẻ (Run 3)"),
        Patch(facecolor="#8E44AD", edgecolor="#333333", label="Macro-F1 — Final Ensemble"),
        Patch(facecolor="#D95F02", edgecolor="#333333", label="Accuracy — Final Ensemble"),
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(df["model_display_name"], fontsize=10.2, fontweight="bold")
    ax.set_xlabel("Điểm số trên Hold-out Test Set (%)")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=54)
    min_x = max(0, float(min(df["f1_macro_pct"].min(), df["accuracy_pct"].min())) - 6)
    ax.set_xlim(min_x, 102)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=9.2)

    for bars in (bars_f1, bars_acc):
        for bar in bars:
            v = bar.get_width()
            ax.text(v + 0.25, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", ha="left", fontsize=8.8, fontweight="bold")

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_inference_benchmark(
    bench_df: pd.DataFrame,
    title: str = "Benchmark Tốc độ Suy luận (Throughput) & Kích thước Tham số trên RTX 5070",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B7_02_benchmark_toc_do_suy_luan",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B7.2: Biểu đồ kép so sánh tốc độ suy luận (mẫu/giây) và quy mô tham số (Triệu params).
    """
    setup_plot_style()
    df = bench_df.sort_values("samples_per_second", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)

    b0 = axes[0].barh(df["model_display_name"], df["samples_per_second"], color="#1F77B4", edgecolor="#333333", linewidth=0.6)
    axes[0].set_title("Thông lượng suy luận (mẫu / giây — càng cao càng tốt)", pad=10)
    axes[0].set_xlabel("Số mẫu / giây")
    axes[0].set_xlim(0, df["samples_per_second"].max() * 1.22)
    for bar, v in zip(b0, df["samples_per_second"]):
        axes[0].text(v + df["samples_per_second"].max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{v:,.0f} mẫu/s", va="center", ha="left", fontsize=9.2, fontweight="bold")

    params_m = df["num_parameters"] / 1e6
    b1 = axes[1].barh(df["model_display_name"], params_m, color="#E67E22", edgecolor="#333333", linewidth=0.6)
    axes[1].set_title("Quy mô kiến trúc (Triệu tham số)", pad=10)
    axes[1].set_xlabel("Triệu tham số (M)")
    axes[1].set_xlim(0, params_m.max() * 1.22)
    for bar, v in zip(b1, params_m):
        axes[1].text(v + params_m.max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.1f}M", va="center", ha="left", fontsize=9.2, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_ablation_and_3rounds_progress(
    all_runs_df: pd.DataFrame,
    title: str = "Tiến trình Hiệu năng qua 3 Vòng Pseudo-Labeling so với Đối chứng Baseline & Final Ensemble",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_01_tien_trinh_3_vong_va_ablation",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.1: Biểu đồ chủ lực của đồ án — theo dõi Macro-F1 & Accuracy qua Run 1 -> Run 2 -> Run 3,
    đặt cạnh đường ngang mốc Supervised Baseline tốt nhất và Final Weighted Ensemble!
    """
    setup_plot_style()
    rounds = ["run1", "run2", "run3"]
    round_labels = ["Run 1\n(10k Subset)", "Run 2\n(34.4k + Pseudo R1)", "Run 3\n(39.2k + Pseudo R2)"]

    single_df = all_runs_df[all_runs_df["run_tag"].isin(rounds)].copy()
    base_df = all_runs_df[all_runs_df["run_tag"] == "baseline"]
    ens_df = all_runs_df[all_runs_df["run_tag"] == "final_ensemble"]

    models = list(single_df["model_display_name"].unique())
    palette = sns.color_palette("tab10", len(models))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    x = np.arange(len(rounds))

    for ax, metric_col, y_label in [
        (axes[0], "f1_macro_pct", "Macro-F1 trên Hold-out Test (%)"),
        (axes[1], "accuracy_pct", "Accuracy trên Hold-out Test (%)"),
    ]:
        for color, m_name in zip(palette, models):
            sub = single_df[single_df["model_display_name"] == m_name].set_index("run_tag")
            vals = [float(sub.loc[r, metric_col]) if r in sub.index else np.nan for r in rounds]
            # Ghi chú riêng nếu Run 1 bị collapse (ViDeBERTa-base) -> nối bằng nét chấm mờ để không che vùng 78-92%
            valid_vals = [v for v in vals if not np.isnan(v) and v > 50]
            if len(valid_vals) < len(vals) and vals[0] < 50:
                ax.plot(x[1:], vals[1:], marker="o", linewidth=2.3, markersize=7, color=color, label=f"{m_name} (R2→R3)")
            else:
                ax.plot(x, vals, marker="o", linewidth=2.3, markersize=7, color=color, label=m_name)

        # Vạch mốc đối chứng Ablation Baseline cao nhất
        if not base_df.empty:
            best_base_row = base_df.loc[base_df[metric_col].idxmax()]
            bv = float(best_base_row[metric_col])
            ax.axhline(
                bv,
                color="#7F8C8D",
                linestyle="--",
                linewidth=2.0,
                label=f"Best Baseline ({best_base_row['model_display_name']}: {bv:.2f}%)",
            )

        # Vạch đỉnh Final Ensemble cao nhất
        if not ens_df.empty:
            best_ens_row = ens_df.loc[ens_df[metric_col].idxmax()]
            ev = float(best_ens_row[metric_col])
            ax.axhline(
                ev,
                color="#8E44AD",
                linestyle="-.",
                linewidth=2.3,
                label=f"Best Ensemble ({best_ens_row['model_display_name']}: {ev:.2f}%)",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(round_labels, fontsize=10.2, fontweight="bold")
        ax.set_ylabel(y_label)
        ax.set_title(y_label, pad=10)
        # Zoom vào dải hoạt động đẹp [77%, max + 2%] để nhìn rõ từng bước tiến
        high_vals = single_df[single_df[metric_col] > 50][metric_col]
        y_min = max(65, float(high_vals.min()) - 2.5) if not high_vals.empty else 70
        y_max = min(100, float(all_runs_df[metric_col].max()) + 2.2)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="lower right", fontsize=8.3, frameon=True, framealpha=0.92)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_radar_charts_grid(
    all_runs_df: pd.DataFrame,
    title: str = "Biểu đồ Radar 5 trục — Tiến bộ qua 3 Vòng của 7 Mô hình & Final Ensemble",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_02_radar_charts_7_models",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.2: Lưới Radar 2x4 (7 mô hình đơn lẻ qua 3 vòng + ô thứ 8 so sánh Best Baseline vs Final Ensemble).
    """
    setup_plot_style()
    metric_cols = [
        "accuracy_pct",
        "f1_irrelevant_pct",
        "f1_environment_pct",
        "f1_social_pct",
        "f1_governance_pct",
    ]
    axis_labels = ["Accuracy", "F1\nIrrelevant", "F1\nEnvironment", "F1\nSocial", "F1\nGovernance"]
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    single_models = list(all_runs_df[all_runs_df["run_tag"] == "run3"]["model_display_name"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), subplot_kw=dict(polar=True))
    axes_flat = axes.flatten()

    run_styles = [
        ("run1", "Run 1", "#95A5A6", "--"),
        ("run2", "Run 2", "#1F77B4", "-."),
        ("run3", "Run 3", "#2E8B57", "-"),
    ]

    for idx, m_name in enumerate(single_models[:7]):
        ax = axes_flat[idx]
        for rtag, rlabel, col, ls in run_styles:
            sub = all_runs_df[(all_runs_df["run_tag"] == rtag) & (all_runs_df["model_display_name"] == m_name)]
            if sub.empty:
                continue
            vals = sub.iloc[0][metric_cols].astype(float).tolist()
            if max(vals) < 50:
                continue  # Không vẽ vòng bị collapse để tránh kéo tâm radar về 0
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2.2, linestyle=ls, color=col, label=rlabel)
            if rtag == "run3":
                ax.fill(angles, vals, color=col, alpha=0.14)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axis_labels, fontsize=9.2, fontweight="bold")
        ax.set_ylim(55, 100)
        ax.set_yticks([65, 75, 85, 95])
        ax.set_yticklabels(["65", "75", "85", "95"], fontsize=8, color="#555555")
        ax.set_title(m_name, fontsize=12.5, fontweight="bold", pad=16)
        ax.tick_params(axis="x", pad=8)
        if idx == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), fontsize=9, frameon=True)

    # Ô thứ 8: So sánh Best Baseline vs Best Single Run 3 vs Best Final Ensemble
    ax_last = axes_flat[7]
    for rtag, rlabel, col, ls in [
        ("baseline", "Best Baseline", "#7F8C8D", "--"),
        ("run3", "Best Single R3", "#1F77B4", "-."),
        ("final_ensemble", "Best Ensemble", "#8E44AD", "-"),
    ]:
        sub = all_runs_df[all_runs_df["run_tag"] == rtag]
        if sub.empty:
            continue
        best = sub.loc[sub["f1_macro_pct"].idxmax()]
        vals = best[metric_cols].astype(float).tolist()
        vals += vals[:1]
        ax_last.plot(angles, vals, linewidth=2.4, linestyle=ls, color=col, label=rlabel)
        if rtag == "final_ensemble":
            ax_last.fill(angles, vals, color=col, alpha=0.16)

    ax_last.set_xticks(angles[:-1])
    ax_last.set_xticklabels(axis_labels, fontsize=9.2, fontweight="bold")
    ax_last.set_ylim(65, 100)
    ax_last.set_yticks([70, 80, 90, 100])
    ax_last.set_yticklabels(["70", "80", "90", "100"], fontsize=8, color="#555555")
    ax_last.set_title("Tổng kết: Baseline vs R3 vs Ensemble", fontsize=12, fontweight="bold", pad=16, color="#8E44AD")
    ax_last.tick_params(axis="x", pad=8)
    ax_last.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.8, frameon=True)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_per_class_f1_heatmap(
    run3_and_ens_df: pd.DataFrame,
    title: str = "Heatmap F1-Score từng lớp ESG (%) — 7 Mô hình Run 3 & Final Ensemble",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_03_heatmap_f1_theo_lop",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.3: Heatmap so sánh chi tiết F1-Score của 4 lớp ESG + Macro-F1 + Accuracy.
    """
    setup_plot_style()
    df = run3_and_ens_df.sort_values("f1_macro_pct", ascending=False).copy()
    cols = [
        "f1_irrelevant_pct",
        "f1_environment_pct",
        "f1_social_pct",
        "f1_governance_pct",
        "f1_macro_pct",
        "accuracy_pct",
    ]
    col_labels = ["Irrelevant\n(F1 %)", "Environment\n(F1 %)", "Social\n(F1 %)", "Governance\n(F1 %)", "Macro-F1\n(%)", "Accuracy\n(%)"]
    heat_df = df.set_index("model_display_name")[cols]
    heat_df.columns = col_labels

    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.55 * len(heat_df) + 1.6)))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=max(65, float(heat_df.values.min()) - 2),
        vmax=100,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Điểm số (%)"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=12)
    ax.set_ylabel("Mô hình / Chiến lược Ensemble")
    ax.set_xlabel("Chỉ số đánh giá trên Hold-out Test Set (3.000 mẫu)")
    plt.tight_layout()

    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_dual_confusion_matrix(
    cm_best_single: np.ndarray,
    single_name: str,
    cm_ensemble: np.ndarray,
    ensemble_name: str,
    title: str = "Ma trận Nhầm lẫn (Confusion Matrix) trên Hold-out Test Set (3.000 mẫu)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_04_confusion_matrix_best_va_ensemble",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.4: So sánh song song Confusion Matrix (hiển thị cả số lượng thực tế & % theo hàng Recall)
    giữa Best Single Model (Run 3) và Best Final Ensemble.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    for ax, cm_raw, sub_title in [
        (axes[0], np.asarray(cm_best_single), f"Best Single Model: {single_name}"),
        (axes[1], np.asarray(cm_ensemble), f"Best Final Ensemble: {ensemble_name}"),
    ]:
        row_sums = np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
        cm_pct = cm_raw / row_sums * 100.0
        annot = np.empty_like(cm_raw, dtype=object)
        for i in range(cm_raw.shape[0]):
            for j in range(cm_raw.shape[1]):
                annot[i, j] = f"{int(cm_raw[i, j]):,}\n({cm_pct[i, j]:.1f}%)"

        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            square=True,
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            cbar_kws={"label": "Tỉ lệ theo nhãn thực (%)"},
            ax=ax,
        )
        ax.set_title(sub_title, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("Nhãn dự đoán (Predicted)")
        ax.set_ylabel("Nhãn thực tế (Ground Truth)")

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_accuracy_latency_tradeoff(
    tradeoff_df: pd.DataFrame,
    title: str = "Đánh đổi Hiệu năng (Macro-F1) vs Độ trễ Suy luận (Latency) & Đường biên Pareto",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_05_tradeoff_hieu_nang_va_toc_do",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.5: Biểu đồ Scatter Trade-off giữa ms/mẫu (trục X) và Macro-F1 % (trục Y),
    kích thước bong bóng theo số tham số (M), kèm đường biên tối ưu Pareto cho khuyến nghị triển khai.
    """
    setup_plot_style()
    df = tradeoff_df.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))

    is_ens = df["run_tag"].astype(str).str.contains("ensemble", case=False)
    colors = ["#8E44AD" if e else "#1F77B4" for e in is_ens]
    sizes = np.clip((df["num_parameters"] / 1e6) * 2.0, 120, 620)

    ax.scatter(
        df["ms_per_sample"],
        df["f1_macro_pct"],
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolors="#222222",
        linewidth=1.1,
        zorder=4,
    )

    # Tìm và vẽ đường biên tối ưu Pareto (độ trễ thấp nhất cho mỗi mức Macro-F1 cao nhất)
    sorted_p = df.sort_values("ms_per_sample")
    pareto_pts = []
    best_f1 = -1.0
    for _, r in sorted_p.iterrows():
        if float(r["f1_macro_pct"]) > best_f1:
            best_f1 = float(r["f1_macro_pct"])
            pareto_pts.append((float(r["ms_per_sample"]), float(r["f1_macro_pct"])))
    if len(pareto_pts) >= 2:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, linestyle="--", color="#2E8B57", linewidth=2.0, label="Đường biên tối ưu Pareto", zorder=3)

    # Sắp đặt vị trí nhãn thông minh kèm đường dẫn mảnh (leader line) tránh chồng chéo tuyệt đối
    custom_offsets = {
        "DistilBERT-m": (14, 28, "left"),
        "mBERT-cased": (44, 14, "left"),
        "PhoBERT-base": (62, -4, "left"),
        "ViSoBERT": (58, 2, "left"),
        "Vi-ELECTRA": (-12, -42, "left"),
        "RoBERTa-base": (26, -12, "left"),
        "ViDeBERTa-base": (26, -4, "left"),
        "Ensemble — Top-3 Weighted (PhoBERT+Vi+mBERT)": (16, 26, "left"),
        "Ensemble — Equal Soft (7M)": (28, 36, "left"),
        "Ensemble — Weighted Soft (7M)": (28, -2, "left"),
        "Ensemble — Hard Voting (7M)": (28, -40, "left"),
    }
    for idx, r in df.iterrows():
        name = str(r["model_display_name"])
        dx, dy, ha = custom_offsets.get(name, (12, 14 if idx % 2 == 0 else -24, "left"))
        short_label = name.replace("Ensemble — Top-3 Weighted (PhoBERT+Vi+mBERT)", "Ensemble — Top-3 Weighted\n(PhoBERT+Vi-ELECTRA+mBERT)")
        ax.annotate(
            f"{short_label}\n({r['f1_macro_pct']:.2f}%, {r['ms_per_sample']:.2f}ms)",
            (r["ms_per_sample"], r["f1_macro_pct"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#7F8C8D", lw=0.9, shrinkA=0, shrinkB=5, alpha=0.85),
            zorder=5,
        )

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1F77B4", edgecolor="#222222", label="Mô hình đơn lẻ (Run 3)"),
        Patch(facecolor="#8E44AD", edgecolor="#222222", label="Hệ thống Ensemble (Run 3)"),
    ]
    if len(pareto_pts) >= 2:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#2E8B57", linestyle="--", linewidth=2.0, label="Đường biên Pareto"))

    ax.set_xlabel("Độ trễ suy luận trung bình (ms / câu — càng nhỏ càng nhanh)")
    ax.set_ylabel("Macro-F1 trên Hold-out Test Set (%)")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=14)
    ax.set_xlim(0.0, df["ms_per_sample"].max() * 1.44)
    ax.set_ylim(df["f1_macro_pct"].min() - 1.0, df["f1_macro_pct"].max() + 1.55)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig


def plot_calibration_reliability_diagram(
    models_calibration: Dict[str, Dict],
    title: str = "Biểu đồ Hiệu chuẩn Xác suất (Reliability Diagram & Chỉ số ECE)",
    save_dir: Optional[Path] = None,
    filename_stem: str = "B8_06_calibration_reliability",
    dpi: int = 300,
    save_pdf: bool = True,
) -> plt.Figure:
    """
    Hình B8.6: Vẽ Reliability Diagram so sánh độ hiệu chuẩn xác suất của Best Single Model vs Final Ensemble.
    Đường càng sát đường chéo y = x (ECE càng nhỏ) thì xác suất tự tin mô hình báo ra càng đáng tin cậy.
    """
    setup_plot_style()
    names = list(models_calibration.keys())
    n_cols = len(names)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.2 * n_cols, 5.4), sharey=True)
    if n_cols == 1:
        axes = [axes]

    colors = ["#1F77B4", "#8E44AD", "#2E8B57", "#D95F02"]
    for idx, (ax, name) in enumerate(zip(axes, names)):
        info = models_calibration[name]
        bins = info["calibration_bins"]
        ece = float(info["ece"])

        confs = [b["avg_confidence"] for b in bins]
        accs = [b["accuracy"] for b in bins]
        widths = [b["bin_high"] - b["bin_low"] for b in bins]
        centers = [(b["bin_high"] + b["bin_low"]) / 2.0 for b in bins]

        ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.6, label="Hiệu chuẩn lý tưởng (y = x)")
        ax.bar(
            centers,
            accs,
            width=[w * 0.82 for w in widths],
            color=colors[idx % len(colors)],
            alpha=0.72,
            edgecolor="#222222",
            linewidth=0.7,
            label="Độ chính xác thực tế theo bin",
        )
        # Vẽ khoảng cách sai lệch hiệu chuẩn (Calibration Gap)
        ax.plot(confs, accs, marker="o", color="#C0392B", linewidth=1.8, markersize=5.5, label=f"Đường hiệu chuẩn (ECE = {ece:.4f})")

        ax.set_title(f"{name}\n(ECE = {ece:.4f} — càng gần 0 càng tốt)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Độ tin cậy dự đoán trung bình (Confidence)")
        if idx == 0:
            ax.set_ylabel("Độ chính xác thực tế (Empirical Accuracy)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper left", fontsize=8.8, frameon=True)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    if save_dir is not None:
        save_figure(fig, save_dir, filename_stem, dpi=dpi, save_pdf=save_pdf)
    return fig
