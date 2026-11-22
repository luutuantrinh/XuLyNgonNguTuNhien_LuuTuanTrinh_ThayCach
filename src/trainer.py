"""
Bộ máy huấn luyện chuẩn khoa học (ESGTrainer) — Tương thích Transformers v5.24.0 & RTX 5070.

Các điểm chuẩn hóa quan trọng:
1. Tương thích 100% API `transformers 5.24.0`:
   - Dùng `eval_strategy="epoch"`, `save_strategy="epoch"`, `logging_strategy="epoch"`.
   - Dùng `warmup_steps=config.warmup_steps` (tỉ lệ float 0.10) thay vì `warmup_ratio` đã bị xóa ở v5.
   - Dùng `processing_class=tokenizer` thay vì `tokenizer=...` đã bị xóa ở v5.
2. Đảm bảo Zero Data Leakage:
   - Chỉ dùng `val.csv` để đánh giá theo epoch, kích hoạt `EarlyStoppingCallback` và chọn
     checkpoint tốt nhất theo `eval_f1_macro`. Tuyệt đối không chạm `test_clean.csv` trong khi train.
3. Chống quá tải phần cứng & dọn dẹp gọn gàng:
   - Dùng `bf16=True` (native trên RTX 5070), dynamic padding (`DataCollatorWithPadding`),
     `dataloader_num_workers=0` (an toàn cho Windows JupyterLab).
   - Xóa các thư mục trung gian `checkpoint-*` sau khi đã lưu best model để tiết kiệm ổ đĩa.
   - Dọn VRAM GPU + nghỉ tản nhiệt `cooldown_seconds` giây sau mỗi mô hình.
   - Hỗ trợ `skip_if_done=True` để chạy tiếp an toàn nếu bị ngắt phiên JupyterLab.
"""

from pathlib import Path
from typing import Dict, List
import gc
import json
import shutil
import time

import pandas as pd
import torch
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


class VietnameseEpochLoggerCallback(TrainerCallback):
    """
    Callback in tiến độ gọn gàng theo từng 25% epoch & tổng kết cuối epoch ngay trên luồng chính.
    Thay thế NotebookProgressCallback mặc định để chống rò rỉ ZMQ socket trên Windows JupyterLab.
    """

    def __init__(self, model_name: str, total_epochs: int):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self._last_train_loss = None
        self._epoch_start_time = None
        self._steps_per_epoch = None
        self._last_milestone = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps and self.total_epochs > 0:
            self._steps_per_epoch = max(1, int(round(state.max_steps / self.total_epochs)))
        print(
            f"  [{self.model_name}] Đã tải & tokenize xong — bắt đầu huấn luyện "
            f"(~{self._steps_per_epoch or '?'} steps/epoch)...",
            flush=True,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.perf_counter()
        self._last_milestone = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not self._steps_per_epoch:
            return
        step_in_epoch = state.global_step % self._steps_per_epoch
        if step_in_epoch == 0 and state.global_step > 0:
            return
        pct = int((step_in_epoch / self._steps_per_epoch) * 100)
        for m in (25, 50, 75):
            if pct >= m and self._last_milestone < m:
                self._last_milestone = m
                cur_ep = int(state.epoch) + 1 if state.epoch is not None else 1
                elapsed = (time.perf_counter() - self._epoch_start_time) if self._epoch_start_time else 0.0
                print(
                    f"    [{self.model_name}] Epoch {cur_ep}/{self.total_epochs} — tiến độ {m}% "
                    f"(step {step_in_epoch}/{self._steps_per_epoch}, đã chạy {elapsed:.0f}s)",
                    flush=True,
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self._last_train_loss = float(logs["loss"])
        if "eval_loss" in logs:
            ep = int(round(float(logs.get("epoch", state.epoch or 0))))
            tr_loss = f"{self._last_train_loss:.4f}" if self._last_train_loss is not None else "N/A"
            val_loss = float(logs["eval_loss"])
            val_f1 = float(logs.get("eval_f1_macro", 0.0)) * 100.0
            val_acc = float(logs.get("eval_accuracy", 0.0)) * 100.0
            ep_sec = (time.perf_counter() - self._epoch_start_time) if self._epoch_start_time else 0.0
            print(
                f"  [{self.model_name}] Epoch {ep}/{self.total_epochs} | "
                f"Train Loss={tr_loss} | Val Loss={val_loss:.4f} | "
                f"Val Macro-F1={val_f1:.2f}% | Val Acc={val_acc:.2f}% ({ep_sec/60:.1f} phút)",
                flush=True,
            )

from src.config import (
    METRICS_DIR,
    ModelSpec,
    PipelineConfig,
    set_global_seed,
)
from src.dataset import ESGTextDataset
from src.evaluator import make_hf_compute_metrics
from src.model_wrapper import ESGModelWrapper
from src.preprocessing import VietnameseESGPreprocessor


def _extract_epoch_history(log_history: List[Dict]) -> pd.DataFrame:
    """
    Gom nhật ký `trainer.state.log_history` theo từng epoch thành bảng gọn để vẽ Learning Curves.
    """
    by_epoch: Dict[int, Dict] = {}
    for entry in log_history:
        if "epoch" not in entry:
            continue
        ep = int(round(float(entry["epoch"])))
        if ep not in by_epoch:
            by_epoch[ep] = {"epoch": ep}
        # Ghi nhận train loss ở cuối epoch
        if "loss" in entry:
            by_epoch[ep]["train_loss"] = float(entry["loss"])
        # Ghi nhận các chỉ số validation
        if "eval_loss" in entry:
            for k in [
                "eval_loss",
                "eval_accuracy",
                "eval_f1_macro",
                "eval_f1_weighted",
                "eval_precision_weighted",
                "eval_recall_weighted",
            ]:
                if k in entry:
                    by_epoch[ep][k] = float(entry[k])

    if not by_epoch:
        return pd.DataFrame()
    return pd.DataFrame([by_epoch[k] for k in sorted(by_epoch.keys())])


def _cleanup_intermediate_checkpoints(checkpoint_dir: Path) -> None:
    """Xóa các thư mục tạm `checkpoint-*` của HF Trainer sau khi đã lưu best model chính thức."""
    for sub in checkpoint_dir.glob("checkpoint-*"):
        if sub.is_dir():
            shutil.rmtree(sub, ignore_errors=True)


class ESGTrainer:
    """Điều phối huấn luyện mô hình ESG an toàn cho RTX 5070 trên JupyterLab."""

    def __init__(self, config: PipelineConfig, metrics_dir: Path = METRICS_DIR):
        self.config = config
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def train_single_model(
        self,
        model_spec: ModelSpec,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        run_tag: str,
        num_epochs: int,
        checkpoint_dir: Path,
        data_source_label: str = "",
    ) -> Dict:
        """
        Huấn luyện một mô hình từ checkpoint pretrained gốc (tránh confirmation bias giữa các vòng).

        Args:
            model_spec: Cấu hình mô hình từ `MODEL_REGISTRY`.
            train_df: Tập huấn luyện (subset hoặc train_pool hoặc combined).
            val_df: Tập Validation cố định (`val.csv`).
            run_tag: Nhãn vòng chạy (vd `'baseline'`, `'run1'`, `'run2'`, `'run3'`).
            num_epochs: Số epoch tối đa (có EarlyStopping chặn dừng sớm).
            checkpoint_dir: Thư mục lưu best model của riêng mô hình trong vòng này.
            data_source_label: Ghi chú tên tập train lưu vào metadata.
        """
        checkpoint_dir = Path(checkpoint_dir)
        history_csv_path = self.metrics_dir / f"{run_tag}_{model_spec.short_name}_history.csv"
        train_meta_path = self.metrics_dir / f"{run_tag}_{model_spec.short_name}_train_summary.json"
        saved_meta_path = checkpoint_dir / "model_metadata.json"

        # Công tắc bỏ qua nếu mô hình ở vòng này đã hoàn tất trước đó
        if (
            self.config.skip_if_done
            and saved_meta_path.exists()
            and history_csv_path.exists()
            and train_meta_path.exists()
        ):
            print(
                f"[Bỏ qua Train] [{run_tag}] {model_spec.display_name} đã hoàn tất trước đó "
                f"-> nạp lại tóm tắt từ {train_meta_path.name}"
            )
            with open(train_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print("\n" + "=" * 78)
        print(
            f"[BẮT ĐẦU TRAIN] Vòng: {run_tag} | Mô hình: {model_spec.display_name} ({model_spec.hf_name})\n"
            f"  - Train: {len(train_df):,} mẫu ({data_source_label}) | Val cố định: {len(val_df):,} mẫu\n"
            f"  - Tiền xử lý tách từ pyvi: {model_spec.use_word_segmenter} | Epochs tối đa: {num_epochs}"
        )
        print("=" * 78)

        set_global_seed(self.config.seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        hw = self.config.get_hardware_settings()

        # Xác định chế độ mixed precision, batch size & bảo vệ VRAM theo phần cứng + kiến trúc mô hình
        has_cuda = torch.cuda.is_available() and self.config.hardware_mode != "cpu"
        if has_cuda:
            # Khóa trần VRAM tối đa 82% (~9.8 GB / 12 GB) để ngăn tuyệt đối Windows đẩy sang Shared RAM gây đứng lag máy
            try:
                torch.cuda.set_per_process_memory_fraction(0.82, device=0)
            except Exception:
                pass

        if model_spec.force_fp32:
            use_bf16 = False
            use_fp16 = False
            print(f"  [Ổn định số học] Bật chế độ FP32 thuần cho {model_spec.display_name}.")
        else:
            use_bf16 = bool(hw["bf16"] and has_cuda and torch.cuda.is_bf16_supported())
            use_fp16 = bool((not use_bf16) and hw["fp16"] and has_cuda)

        effective_lr = (
            model_spec.learning_rate_override
            if model_spec.learning_rate_override is not None
            else self.config.learning_rate
        )
        base_effective_bs = hw["train_batch_size"] * hw["gradient_accumulation_steps"]
        effective_train_bs = (
            min(hw["train_batch_size"], model_spec.batch_size_override)
            if model_spec.batch_size_override is not None
            else hw["train_batch_size"]
        )
        effective_grad_accum = max(1, base_effective_bs // effective_train_bs)
        effective_eval_bs = (
            min(hw["eval_batch_size"], model_spec.batch_size_override * 2)
            if model_spec.batch_size_override is not None
            else hw["eval_batch_size"]
        )
        use_act_ckpt = bool(hw["gradient_checkpointing"] or model_spec.force_activation_checkpointing)
        print(
            f"  [Bảo vệ GPU] bs={effective_train_bs} x grad_accum={effective_grad_accum} "
            f"(Effective BS={effective_train_bs * effective_grad_accum}) | "
            f"bf16={use_bf16} | act_ckpt={use_act_ckpt} | lr={effective_lr} | trần VRAM=82%"
        )

        if has_cuda:
            torch.cuda.reset_peak_memory_stats()

        # Bước 1: Tiền xử lý văn bản nhất quán cho cả Train và Validation
        preprocessor = VietnameseESGPreprocessor(use_word_segmenter=model_spec.use_word_segmenter)
        train_texts = preprocessor.transform_series(train_df[self.config.text_column])
        val_texts = preprocessor.transform_series(val_df[self.config.text_column])

        # Bước 2: Khởi tạo mô hình từ pretrained gốc & tạo PyTorch Datasets
        wrapper = None
        trainer = None
        train_dataset = None
        val_dataset = None

        try:
            wrapper = ESGModelWrapper(model_spec.hf_name).load()
            train_dataset = ESGTextDataset(
                texts=train_texts,
                tokenizer=wrapper.tokenizer,
                labels=train_df[self.config.label_column].astype(int),
                max_length=self.config.max_length,
            )
            val_dataset = ESGTextDataset(
                texts=val_texts,
                tokenizer=wrapper.tokenizer,
                labels=val_df[self.config.label_column].astype(int),
                max_length=self.config.max_length,
            )
            data_collator = DataCollatorWithPadding(
                tokenizer=wrapper.tokenizer,
                pad_to_multiple_of=8 if has_cuda else None,
            )

            # Bước 3: Thiết lập TrainingArguments chuẩn Transformers v5.24.0 + bảo vệ VRAM
            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=effective_train_bs,
                per_device_eval_batch_size=effective_eval_bs,
                gradient_accumulation_steps=effective_grad_accum,
                learning_rate=effective_lr,
                max_grad_norm=1.0,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                lr_scheduler_type="linear",
                bf16=use_bf16,
                fp16=use_fp16,
                activation_checkpointing=use_act_ckpt,
                dataloader_num_workers=hw["dataloader_num_workers"],
                dataloader_pin_memory=has_cuda,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=True,
                save_total_limit=1,
                report_to="none",
                seed=self.config.seed,
                data_seed=self.config.seed,
                disable_tqdm=True,  # Tắt NotebookProgressCallback tránh rò rỉ socket ZMQ trên Windows JupyterLab
            )

            trainer = Trainer(
                model=wrapper.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=wrapper.tokenizer,
                data_collator=data_collator,
                compute_metrics=make_hf_compute_metrics(),
                callbacks=[
                    VietnameseEpochLoggerCallback(
                        model_name=model_spec.display_name,
                        total_epochs=num_epochs,
                    ),
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience
                    ),
                ],
            )

            # Bước 4: Huấn luyện và đánh giá trên Validation bằng best checkpoint
            t0 = time.perf_counter()
            train_output = trainer.train()
            val_metrics = trainer.evaluate()
            train_seconds = time.perf_counter() - t0
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3) if has_cuda else 0.0
            )

            # Bước 5: Trích xuất & lưu lịch sử từng epoch để vẽ Learning Curves
            history_df = _extract_epoch_history(trainer.state.log_history)
            history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")

            best_epoch = (
                int(history_df.loc[history_df[f"eval_{self.config.metric_for_best_model}"].idxmax(), "epoch"])
                if (not history_df.empty and f"eval_{self.config.metric_for_best_model}" in history_df.columns)
                else int(round(train_output.metrics.get("epoch", num_epochs)))
            )

            # Bước 6: Lưu best model sạch & xóa thư mục tạm checkpoint-* để tiết kiệm dung lượng ổ đĩa
            train_summary = {
                "run_tag": run_tag,
                "model_short_name": model_spec.short_name,
                "model_display_name": model_spec.display_name,
                "hf_name": model_spec.hf_name,
                "use_word_segmenter": model_spec.use_word_segmenter,
                "data_source": data_source_label,
                "train_size": int(len(train_df)),
                "val_size": int(len(val_df)),
                "hardware_mode": self.config.hardware_mode,
                "bf16": use_bf16,
                "fp16": use_fp16,
                "activation_checkpointing": use_act_ckpt,
                "effective_train_bs": effective_train_bs,
                "effective_grad_accum": effective_grad_accum,
                "num_epochs_configured": num_epochs,
                "best_epoch": best_epoch,
                "train_seconds": round(float(train_seconds), 2),
                "peak_vram_gb": round(float(peak_vram_gb), 2),
                "train_metrics": {k: float(v) for k, v in train_output.metrics.items() if isinstance(v, (int, float))},
                "best_val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
                "checkpoint_dir": str(checkpoint_dir),
                "history_csv": str(history_csv_path),
            }

            wrapper.model = trainer.model
            wrapper.save(checkpoint_dir, metadata=train_summary)
            _cleanup_intermediate_checkpoints(checkpoint_dir)

            with open(train_meta_path, "w", encoding="utf-8") as f:
                json.dump(train_summary, f, ensure_ascii=False, indent=2)

            print(
                f"[HOÀN TẤT TRAIN] [{run_tag}] {model_spec.display_name} | "
                f"Best Epoch={best_epoch} | Val Macro-F1={val_metrics.get('eval_f1_macro', 0)*100:.2f}% | "
                f"Val Acc={val_metrics.get('eval_accuracy', 0)*100:.2f}% | "
                f"VRAM đỉnh={peak_vram_gb:.2f} GB | Thời gian={train_seconds/60:.1f} phút"
            )
        finally:
            # Bước 7: LUÔN giải phóng sạch VRAM dù train xong, lỗi hay bấm Interrupt giữa chừng
            if trainer is not None:
                del trainer
            if train_dataset is not None:
                del train_dataset
            if val_dataset is not None:
                del val_dataset
            if wrapper is not None:
                wrapper.release_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cooldown = int(hw.get("cooldown_seconds", 0))
        if has_cuda and cooldown > 0:
            print(f"[Bảo vệ phần cứng] Đã dọn VRAM — nghỉ tản nhiệt {cooldown} giây trước bước tiếp theo...")
            time.sleep(cooldown)

        return train_summary
