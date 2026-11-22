# Vi-ESG-Bert — Đồ án Cuối khóa (Phiên bản Chuẩn hóa cho JupyterLab)

Dự án phân loại văn bản tiếng Việt theo **4 lớp ESG** (*Irrelevant, Environment, Social, Governance*) bằng **7 mô hình Transformer** kết hợp kỹ thuật **Iterative Ensemble Pseudo-Labeling (3 vòng)**, đảm bảo **Zero Data Leakage**, có **Ablation Study đối chứng**, tối ưu phần cứng **RTX 5070 (12 GB VRAM)** và xuất trọn bộ **biểu đồ trực quan hóa 300 DPI** phục vụ báo cáo cuối khóa.

---

## 1. Hướng dẫn cài đặt môi trường `.venv` (JupyterLab + RTX 5070)

Mở **PowerShell** tại thư mục dự án `D:\Document 2025 Backup\XuLyNgonNguTuNhien\Vi-ESG-Bert-LuuTuanTrinh`:

```powershell
# 1. Cho phép chạy script PowerShell (chỉ cần chạy 1 lần)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Tạo và kích hoạt môi trường ảo .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Cài PyTorch hỗ trợ CUDA 12.8 (bắt buộc cho RTX 5070 kiến trúc Blackwell)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Cài toàn bộ thư viện của dự án + JupyterLab
pip install -r requirements.txt

# 5. Khởi chạy JupyterLab
jupyter lab
```

> Sau khi mở JupyterLab, chạy ngay notebook **`notebooks/B1_Kiem_tra_Moi_truong_va_GPU.ipynb`** để xác nhận GPU RTX 5070, `bf16`, các thư viện và dữ liệu gốc đều sẵn sàng 100%.

---

## 2. Cấu hình phần cứng tối ưu & Công tắc linh hoạt (`src/config.py`)

Máy hiện tại có **NVIDIA GeForce RTX 5070 (12 GB VRAM)** + **20 luồng CPU (Windows)**. Toàn bộ công tắc điều chỉnh nằm tập trung tại `src/config.py` (và có thể ghi đè trực tiếp ngay cell đầu mỗi notebook):

- **Công tắc chế độ phần cứng (`hardware_mode`) — tránh quá tải GPU/RAM:**
  - `"eco"`: Nhẹ & mát nhất (`bs=8 × grad_accum=4`, `gradient_checkpointing=True`, nghỉ tản nhiệt `20s` giữa mỗi model).
  - `"balanced"` *(mặc định khuyên dùng)*: `bs=16 × grad_accum=2` (Effective BS = 32), `bf16=True`, `dataloader_num_workers=0` (an toàn cho Windows JupyterLab), chỉ dùng ~5–7 GB / 12 GB VRAM, nghỉ `15s` + dọn VRAM sau mỗi model.
  - `"performance"`: Nhanh tối đa (`bs=32`, nghỉ `5s`).
  - `"cpu"`: Dự phòng khi chạy trên máy không có GPU.
- **Công tắc chọn nhóm mô hình (`model_preset`):**
  - `"fast_3"`: 3 mô hình tiếng Việt chủ lực (`electra-base-vn`, `phobert-base`, `visobert`).
  - `"vietnamese_4"`: 4 mô hình tiếng Việt (+ `videberta-base`).
  - `"full_7"` *(mặc định)*: Đủ cả 7 mô hình Transformer (4 tiếng Việt + 3 đa ngôn ngữ/tiếng Anh).
- **Công tắc vận hành an toàn:**
  - `skip_if_done = True`: Tự động bỏ qua bước/model đã chạy xong nếu phiên JupyterLab bị ngắt giữa chừng.
  - `run_ablation_baseline = True`: Bật/tắt vòng đối chứng train trên 100% `train_pool` gốc (không dùng pseudo-label).

---

## 3. Nơi lưu trữ Biểu đồ Trực quan hóa phục vụ Báo cáo

Tất cả biểu đồ sinh ra từ `src/viz_utils.py` đều được **tự động lưu đồng thời 2 định dạng**:
- **`.png` (300 DPI)**: Sắc nét, chèn thẳng vào **Word / PowerPoint / Canva**.
- **`.pdf` (Vector)**: Phóng to không vỡ nét, chèn thẳng vào **LaTeX / báo cáo khoa học**.

Cây thư mục lưu trữ biểu đồ được phân tách theo từng bước notebook tại `workspace_outputs/figures/`:

```text
workspace_outputs/figures/
├── B2_eda/              # 4 biểu đồ EDA: phân bố lớp, độ dài câu, WordCloud/Top N-gram, kiểm toán làm sạch
├── B3_baseline_run1/    # Learning Curves (Baseline & Run 1) + bảng xếp hạng 7 model Run 1
├── B4_pseudo_round1/    # Confidence KDE, Ma trận Cohen's Kappa, Vote Count, Stacked Bar trước/sau Combine Vòng 1
├── B5_run2/             # Learning Curves Run 2 + biểu đồ bước nhảy hiệu năng Run 1 -> Run 2
├── B6_pseudo_round2/    # Biểu đồ phễu dữ liệu (Data Funnel) + Stacked Bar 3 tầng (Original + Pseudo 1 + Pseudo 2)
├── B7_run3_ensemble/    # So sánh 7 model đơn lẻ vs Final Ensemble + Benchmark tốc độ suy luận
└── B8_final_report/     # 6 biểu đồ tổng kết báo cáo: Tiến trình 3 vòng + Ablation, Radar Charts, Heatmap F1, Confusion Matrix, Latency vs F1, Calibration
```

---

## 4. Lộ trình 8 Notebook chạy theo đúng thứ tự (`B1` → `B8`)

Toàn bộ notebook nằm trong thư mục **`notebooks/`**, đánh số `B1_` → `B8_` để chạy tuần tự từ trên xuống dưới:

| Bước | File Notebook (`notebooks/`) | Nhiệm vụ chính | Trạng thái triển khai (Cuốn chiếu) |
| :---: | :--- | :--- | :---: |
| **B1** | `B1_Kiem_tra_Moi_truong_va_GPU.ipynb` | Kiểm tra `.venv`, PyTorch CUDA/RTX 5070/BF16, thư viện & dữ liệu gốc | **Đợt 1 — Sẵn sàng chạy** |
| **B2** | `B2_EDA_Lam_sach_va_Chia_Du_lieu.ipynb` | Kiểm toán trùng lặp/rò rỉ, làm sạch tiếng Việt, chia `train_pool`/`val`/`7 subsets`, xuất **4 biểu đồ EDA** | **Đợt 1 — Sẵn sàng chạy** |
| **B3** | `B3_Train_Baseline_va_Run1.ipynb` | Train Ablation Baseline (100% `train_pool`) + Run 1 (7 subsets), vẽ Learning Curves & kết quả Run 1 | **Đợt 2 — Sẵn sàng chạy** |
| **B4** | `B4_PseudoLabel_va_Ensemble_Vong1.ipynb` | Dự đoán tập không nhãn, Ensemble Voting Vòng 1 (`0.94`), gộp cân bằng lớp + 4 biểu đồ chất lượng | **Đợt 2/3 — Sẵn sàng chạy** |
| **B5** | `B5_Train_Run2.ipynb` | Train lại 7 mô hình từ pretrained gốc (`FP32` master weights) trên `combined_round1.csv` + so sánh Run 1 → Run 2 | **Đợt 3 — Sẵn sàng chạy** |
| **B6** | `B6_PseudoLabel_va_Ensemble_Vong2.ipynb` | Pseudo-Labeling & Ensemble Vòng 2 (`0.91`) trên `remain_data_r1`, gộp cân bằng lớp + Data Funnel & Stacked 3 tầng | **Đợt 3 — Sẵn sàng chạy** |
| **B7** | `B7_Train_Run3_va_Danh_gia_Final_Ensemble.ipynb` | Train Vòng 3 cuối cùng, đánh giá Final Ensemble (4 chiến lược + McNemar test) trên `test_clean.csv` + Benchmark Latency/VRAM | **Đợt 4 — Sẵn sàng chạy** |
| **B8** | `B8_Truc_quan_hoa_Tong_hop_Phan_tich_Loi_va_Demo.ipynb` | Xuất trọn bộ 6 biểu đồ tổng kết báo cáo + Phân tích lỗi Top-15 + Demo nhập văn bản ESG tương tác | **Đợt 4 — Sẵn sàng chạy** |
| **Báo cáo** | `BAO_CAO_TONG_HOP_KHOA_HOC.ipynb` | **Notebook Báo cáo Khoa học Tổng hợp (chuẩn Thạc sĩ)** — tổng hợp trọn vẹn phân tích định lượng, kiểm định McNemar, phân tích lỗi và nhúng trực tiếp toàn bộ biểu đồ từ `B1` → `B8` | **✅ Hoàn tất 100% B1 → B8 (27 hình)** |

---

## 5. Quy ước mã nguồn trong dự án
- Toàn bộ comment, docstring, chú thích markdown và thông báo log đều viết bằng **tiếng Việt** để dễ đọc, dễ giải thích khi bảo vệ đồ án.
- Tên thư mục dữ liệu gốc (`data/`) luôn giữ nguyên trạng; toàn bộ kết quả chạy đều ghi vào `workspace_outputs/`.
- Tập `data/test.csv` (3.000 mẫu) được **khóa kín** làm Hold-out Test Set, tuyệt đối không dùng để chọn checkpoint hay tinh chỉnh siêu tham số.
