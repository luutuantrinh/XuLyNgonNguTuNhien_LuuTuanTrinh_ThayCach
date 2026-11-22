"""
Cấu hình trung tâm, công tắc linh hoạt và tối ưu phần cứng cho dự án Vi-ESG-Bert.

Phần cứng máy hiện tại:
- GPU: NVIDIA GeForce RTX 5070 (12 GB VRAM, kiến trúc Blackwell -> hỗ trợ native BF16)
- CPU: 20 luồng logic (Windows -> đặt dataloader_num_workers=0 để ổn định trên JupyterLab)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
import os
import random
import numpy as np

# Chống rò rỉ socket ZMQ ('Too many open files') trên Windows JupyterLab & tắt cảnh báo thừa của HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN & NƠI LƯU TRỮ BIỂU ĐỒ BÁO CÁO
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu gốc (chỉ đọc, tuyệt đối không ghi đè)
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_PATH = DATA_DIR / "train.csv"
RAW_TEST_PATH = DATA_DIR / "test.csv"
RAW_UNLABELED_PATH = DATA_DIR / "data_pseudo" / "overall_data.csv"

# Thư mục lưu toàn bộ đầu ra trong quá trình chạy
OUTPUT_DIR = PROJECT_ROOT / "workspace_outputs"
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
SUBSETS_DIR = PROCESSED_DATA_DIR / "subsets"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PSEUDO_ROUND1_DIR = OUTPUT_DIR / "pseudo_round1"
PSEUDO_ROUND2_DIR = OUTPUT_DIR / "pseudo_round2"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Phân chia thư mục lưu biểu đồ trực quan hóa theo từng bước Notebook (phục vụ báo cáo)
FIGURE_SUBDIRS: Dict[str, Path] = {
    "B2": FIGURES_DIR / "B2_eda",
    "B3": FIGURES_DIR / "B3_baseline_run1",
    "B4": FIGURES_DIR / "B4_pseudo_round1",
    "B5": FIGURES_DIR / "B5_run2",
    "B6": FIGURES_DIR / "B6_pseudo_round2",
    "B7": FIGURES_DIR / "B7_run3_ensemble",
    "B8": FIGURES_DIR / "B8_final_report",
}


def get_figure_dir(step_key: str) -> Path:
    """Trả về và tự tạo thư mục lưu biểu đồ ứng với mã bước ('B2' .. 'B8')."""
    target = FIGURE_SUBDIRS.get(step_key, FIGURES_DIR)
    target.mkdir(parents=True, exist_ok=True)
    return target


# ============================================================================
# 2. NHÃN ESG CHUẨN HÓA (ĐÃ SỬA LỖI CHÍNH TẢ 'Environment')
# ============================================================================
LABEL2ID: Dict[str, int] = {
    "Irrelevant": 0,
    "Environment": 1,
    "Social": 2,
    "Governance": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES: List[str] = [ID2LABEL[i] for i in range(len(ID2LABEL))]
LABEL_NAMES_VI: Dict[int, str] = {
    0: "Không liên quan (Irrelevant)",
    1: "Môi trường (Environment)",
    2: "Xã hội (Social)",
    3: "Quản trị (Governance)",
}
NUM_LABELS: int = len(LABEL_NAMES)


# ============================================================================
# 3. CÔNG TẮC PHẦN CỨNG (CHỐNG QUÁ TẢI GPU / RAM TRÊN RTX 5070 12GB)
# ============================================================================
HardwareMode = Literal["eco", "balanced", "performance", "cpu"]

HARDWARE_PROFILES: Dict[str, Dict] = {
    # Chế độ tiết kiệm/mát nhất: dùng ~3.5-4.5 GB VRAM, phù hợp vừa train vừa làm việc khác
    "eco": {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 4,   # Batch hiệu dụng = 32
        "eval_batch_size": 16,
        "inference_batch_size": 32,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,        # Tránh treo/ngốn RAM đa tiến trình trên Windows JupyterLab
        "cooldown_seconds": 20,             # Nghỉ tản nhiệt giữa các mô hình
    },
    # MẶC ĐỊNH: Tối ưu cân bằng cho RTX 5070 12GB (~5-7 GB VRAM, mát máy & tốc độ cao)
    "balanced": {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 2,   # Batch hiệu dụng = 32
        "eval_batch_size": 32,
        "inference_batch_size": 64,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "cooldown_seconds": 15,
    },
    # Chế độ hiệu năng cao: dùng ~8-10 GB VRAM (vẫn nằm trong ngưỡng an toàn 12 GB)
    "performance": {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,   # Batch hiệu dụng = 32
        "eval_batch_size": 64,
        "inference_batch_size": 128,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "cooldown_seconds": 5,
    },
    # Chế độ dự phòng khi không có GPU CUDA
    "cpu": {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "eval_batch_size": 16,
        "inference_batch_size": 16,
        "bf16": False,
        "fp16": False,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "cooldown_seconds": 0,
    },
}


# ============================================================================
# 4. DANH MỤC 7 MÔ HÌNH TRANSFORMER & CÔNG TẮC BẬT/TẮT LINH HOẠT
# ============================================================================
@dataclass(frozen=True)
class ModelSpec:
    """Thông số chuẩn hóa cho từng mô hình trong hệ thống Ensemble."""
    short_name: str
    hf_name: str
    family: str
    use_word_segmenter: bool
    subset_index: int
    display_name: str
    enabled: bool = True
    force_fp32: bool = False                      # Ép FP32 nếu kiến trúc không ổn định dưới AMP
    learning_rate_override: Optional[float] = None # Ghi đè learning rate riêng nếu kiến trúc cần LR nhỏ hơn
    batch_size_override: Optional[int] = None      # Giảm micro-batch riêng cho mô hình tốn VRAM (tự tăng grad_accum)
    force_activation_checkpointing: bool = False   # Bật riêng activation checkpointing cho kiến trúc nặng (DeBERTa-v2)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "phobert-base": ModelSpec(
        short_name="phobert-base",
        hf_name="vinai/phobert-base",
        family="phobert",
        use_word_segmenter=True,   # Chỉ bật tách từ pyvi cho PhoBERT (pretrain ở mức từ)
        subset_index=1,
        display_name="PhoBERT-base",
    ),
    "distilbert-base-multilingual-cased": ModelSpec(
        short_name="distilbert-base-multilingual-cased",
        hf_name="distilbert/distilbert-base-multilingual-cased",
        family="distilbert",
        use_word_segmenter=False,
        subset_index=2,
        display_name="DistilBERT-m",
    ),
    "bert-base-multilingual-cased": ModelSpec(
        short_name="bert-base-multilingual-cased",
        hf_name="google-bert/bert-base-multilingual-cased",
        family="bert",
        use_word_segmenter=False,
        subset_index=3,
        display_name="mBERT-cased",
    ),
    "roberta-base": ModelSpec(
        short_name="roberta-base",
        hf_name="FacebookAI/roberta-base",
        family="roberta",
        use_word_segmenter=False,
        subset_index=4,
        display_name="RoBERTa-base",
    ),
    "electra-base-vn": ModelSpec(
        short_name="electra-base-vn",
        hf_name="NlpHUST/electra-base-vn",
        family="electra",
        use_word_segmenter=False,
        subset_index=5,
        display_name="Vi-ELECTRA",
    ),
    "visobert": ModelSpec(
        short_name="visobert",
        hf_name="uitnlp/visobert",
        family="xlm-roberta",
        use_word_segmenter=False,
        subset_index=6,
        display_name="ViSoBERT",
    ),
    "videberta-base": ModelSpec(
        short_name="videberta-base",
        hf_name="Fsoft-AIC/videberta-base",
        family="deberta-v2",
        use_word_segmenter=False,
        subset_index=7,
        display_name="ViDeBERTa-base",
        force_fp32=False,                    # Đã có FP32 master weights ở model_wrapper -> dùng AMP BF16 để giảm 50% VRAM
        learning_rate_override=1.5e-5,       # LR ổn định cho Disentangled Relative Attention
        batch_size_override=16,              # Run 2 dùng bs=8 chỉ tốn 3.17 GB VRAM -> nâng lên 16 (giữ act_ckpt=True) để nhanh gấp ~2x
        force_activation_checkpointing=True, # Giữ VRAM activation DeBERTa-v2 ở mức an toàn ~5-6 GB / 12 GB
    ),
}

# Các gói chọn nhanh mô hình (đổi 1 tham số model_preset trong notebook)
MODEL_PRESETS: Dict[str, List[str]] = {
    "fast_3": ["electra-base-vn", "phobert-base", "visobert"],
    "vietnamese_4": ["electra-base-vn", "phobert-base", "visobert", "videberta-base"],
    "full_7": list(MODEL_REGISTRY.keys()),
}


# ============================================================================
# 5. CẤU HÌNH THỰC NGHIỆM (PIPELINE SWITCHES)
# ============================================================================
@dataclass
class PipelineConfig:
    """Cấu hình toàn cục cho toàn bộ quy trình B1 -> B8."""

    # --- Công tắc vận hành chính ---
    hardware_mode: HardwareMode = "balanced"
    model_preset: str = "full_7"          # "fast_3" | "vietnamese_4" | "full_7"
    skip_if_done: bool = True             # Tự bỏ qua model/bước đã có kết quả
    run_ablation_baseline: bool = True    # Train đối chứng trên 100% train_pool (không pseudo-label)
    ablation_preset: str = "fast_3"       # Chạy Baseline cho nhóm đại diện ("fast_3"|"vietnamese_4"|"full_7") để giảm tải GPU
    seed: int = 42

    # --- Thiết lập dữ liệu ---
    text_column: str = "vi_text"
    label_column: str = "labels"
    val_ratio: float = 0.10               # 10% tập train sạch làm tập Validation cố định
    subset_size: int = 10_000             # Số mẫu mỗi tập con cho Run 1
    min_word_count: int = 5               # Ngưỡng tối thiểu số từ để lọc câu nhiễu/rác

    # --- Siêu tham số huấn luyện (Tương thích chuẩn Transformers v5.24+) ---
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: float = 0.10            # Transformers v5 bỏ warmup_ratio, truyền tỉ lệ float vào warmup_steps
    num_epochs_baseline: int = 8
    num_epochs_run1: int = 8
    num_epochs_run2: int = 8
    num_epochs_run3: int = 10
    early_stopping_patience: int = 2
    metric_for_best_model: str = "f1_macro"

    # --- Ngưỡng Pseudo-Labeling, Ensemble & Hiệu chỉnh Prior Shift (từ kết quả EDA B2) ---
    round1_confidence_threshold: float = 0.94
    round2_confidence_threshold: float = 0.91
    round1_min_margin: float = 0.50       # Biên độ p(1) - p(2) >= 0.50 để loại câu nhập nhằng vùng Irrelevant/Social-Governance
    round2_min_margin: float = 0.45
    round1_min_vote_ratio: float = 0.60   # Tối thiểu >= 5/7 model (hoặc >= 2/3 ở fast_3) đồng thuận
    round2_min_vote_ratio: float = 0.60
    round1_target_per_class: int = 9_500  # Kéo 4 lớp về cân bằng (~38k mẫu) nhằm bù Prior Shift lớp Environment (+11.31%)
    round2_target_per_class: int = 10_500 # Mục tiêu vòng cuối cân bằng đều (~42k mẫu)
    use_weighted_voting: bool = True      # Trọng số bỏ phiếu theo Val Macro-F1

    # --- Xuất biểu đồ báo cáo ---
    figure_dpi: int = 300
    save_pdf_figures: bool = True

    def get_hardware_settings(self) -> Dict:
        """Lấy bảng cấu hình phần cứng tương ứng với hardware_mode đang chọn."""
        return HARDWARE_PROFILES[self.hardware_mode].copy()

    def get_active_models(self) -> List[ModelSpec]:
        """Lấy danh sách các mô hình đang được bật theo model_preset."""
        keys = MODEL_PRESETS[self.model_preset]
        return [MODEL_REGISTRY[k] for k in keys if MODEL_REGISTRY[k].enabled]

    def get_ablation_models(self) -> List[ModelSpec]:
        """Lấy danh sách các mô hình chạy đối chứng Ablation Baseline ở Bước B3."""
        keys = MODEL_PRESETS[self.ablation_preset]
        return [MODEL_REGISTRY[k] for k in keys if MODEL_REGISTRY[k].enabled]

    def summary(self) -> str:
        """In tóm tắt gọn gàng cấu hình hiện hành để kiểm tra trước khi chạy."""
        hw = self.get_hardware_settings()
        models = [m.short_name for m in self.get_active_models()]
        abl_models = [m.short_name for m in self.get_ablation_models()]
        return (
            "=== CẤU HÌNH HỆ THỐNG VI-ESG-BERT ===\n"
            f"- Chế độ phần cứng : {self.hardware_mode} "
            f"(train_bs={hw['train_batch_size']} x accum={hw['gradient_accumulation_steps']}, "
            f"bf16={hw['bf16']}, grad_ckpt={hw['gradient_checkpointing']}, nghỉ={hw['cooldown_seconds']}s)\n"
            f"- Gói mô hình      : {self.model_preset} ({len(models)} mô hình: {', '.join(models)})\n"
            f"- Ablation Baseline: bật={self.run_ablation_baseline} | gói={self.ablation_preset} ({', '.join(abl_models)})\n"
            f"- Công tắc an toàn : skip_if_done={self.skip_if_done} | seed={self.seed}\n"
            f"- Siêu tham số     : max_length={self.max_length} | lr={self.learning_rate} | warmup={self.warmup_steps} | metric={self.metric_for_best_model}\n"
            f"- Pseudo & Cân bằng: R1(conf>={self.round1_confidence_threshold}, margin>={self.round1_min_margin}, target={self.round1_target_per_class:,}/lớp) | "
            f"R2(conf>={self.round2_confidence_threshold}, target={self.round2_target_per_class:,}/lớp)\n"
            f"- Thư mục biểu đồ  : {FIGURES_DIR} (DPI={self.figure_dpi}, PDF={self.save_pdf_figures})"
        )


def set_global_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch để đảm bảo tái lập kết quả."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_directories() -> None:
    """Khởi tạo đầy đủ toàn bộ cây thư mục đầu ra và thư mục lưu biểu đồ báo cáo."""
    dirs = [
        PROCESSED_DATA_DIR,
        SUBSETS_DIR,
        CHECKPOINTS_DIR / "baseline",
        CHECKPOINTS_DIR / "run1",
        CHECKPOINTS_DIR / "run2",
        CHECKPOINTS_DIR / "run3",
        PSEUDO_ROUND1_DIR / "per_model",
        PSEUDO_ROUND2_DIR / "per_model",
        METRICS_DIR,
        FIGURES_DIR,
        *FIGURE_SUBDIRS.values(),
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
