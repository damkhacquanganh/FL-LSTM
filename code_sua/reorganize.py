"""
reorganize.py — Tổ chức lại thư mục code_sua/ theo cấu trúc chuyên nghiệp.
Chạy: python reorganize.py
"""
import os
import shutil
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. TẠO CẤU TRÚC THƯ MỤC MỚI
# ============================================================
dirs = [
    "src",
    "experiments",
    "visualization",
    "demo",
    os.path.join("results", "figures"),
    os.path.join("results", "metrics"),
    os.path.join("results", "models"),
    "legacy",
]
for d in dirs:
    path = os.path.join(ROOT, d)
    os.makedirs(path, exist_ok=True)
    print(f"  [DIR] Tạo {d}/")

# ============================================================
# 2. ĐỊNH NGHĨA BẢNG MAPPING: (file_gốc → thư_mục_mới/tên_mới)
# ============================================================

# --- src/ : Module cốt lõi (PyTorch) ---
src_files = {
    "model_pytorch.py":                "src/model.py",
    "preprocess_custom.py":            "src/preprocessing.py",
    "training_pytorch.py":             "src/training.py",
    "evaluation_pytorch.py":           "src/evaluation.py",
    "federated_learning_pytorch.py":   "src/federated.py",
}

# --- experiments/ : Các script chạy thí nghiệm ---
exp_files = {
    "run_adaptive_fedprox.py":     "experiments/run_adaptive_fedprox.py",
    "run_100_rounds_alpha01.py":   "experiments/run_100_rounds_alpha01.py",
    "run_ablation_studies.py":     "experiments/run_ablation_studies.py",
    "run_baselines.py":            "experiments/run_baselines.py",
    "run_tuning_mu.py":            "experiments/run_tuning_mu.py",
}

# --- visualization/ : Vẽ biểu đồ ---
viz_files = {
    "plot_convergence.py":     "visualization/plot_convergence.py",
    "plot_convergence_all.py": "visualization/plot_convergence_all.py",
    "plot_cm_fedprox.py":      "visualization/plot_cm_fedprox.py",
    "plot_roc_curve.py":       "visualization/plot_roc_curve.py",
    "plot_comparison.py":      "visualization/plot_comparison.py",
    "plot_non_iid.py":         "visualization/plot_non_iid.py",
    "plot_results.py":         "visualization/plot_results.py",
}

# --- demo/ ---
demo_files = {
    "demo_prototype_ids.py":   "demo/demo_prototype_ids.py",
}

# --- legacy/ : Code Keras/TF cũ ---
legacy_files = {
    "model.py":                     "legacy/model_keras.py",
    "main.py":                      "legacy/main_keras.py",
    "main_pytorch.py":              "legacy/main_pytorch_v1.py",
    "main_centralized_pytorch.py":  "legacy/main_centralized_pytorch.py",
    "evaluation.py":                "legacy/evaluation_keras.py",
    "training.py":                  "legacy/training_keras.py",
    "federated_learning.py":        "legacy/federated_learning_keras.py",
    "run_cent_fedavg_only.py":      "legacy/run_cent_fedavg_only.py",
    "run_fedprox_only.py":          "legacy/run_fedprox_only.py",
    "run_experiments_auto.py":      "legacy/run_experiments_auto.py",
    "run_non_iid.py":               "legacy/run_non_iid.py",
    "fl_lstm_wsn_K3_T3.h5":        "legacy/fl_lstm_wsn_K3_T3.h5",
}

# --- results/figures/ : Ảnh biểu đồ ---
figure_patterns = ["*.png"]

# --- results/metrics/ : Kết quả số liệu ---
metric_files = {
    "adaptive_fedprox_results.json": "results/metrics/adaptive_fedprox_results.json",
    "ablation_results.json":         "results/metrics/ablation_results.json",
    "results_100_rounds.json":       "results/metrics/results_100_rounds.json",
    "fedprox_results.txt":           "results/metrics/fedprox_results.txt",
    "non_iid_results.txt":           "results/metrics/non_iid_results.txt",
    "data_distribution_log.txt":     "results/metrics/data_distribution_log.txt",
    "cent_fedavg_convergence.npz":   "results/metrics/cent_fedavg_convergence.npz",
    "fedprox_convergence.npz":       "results/metrics/fedprox_convergence.npz",
    "non_iid_convergence.npz":       "results/metrics/non_iid_convergence.npz",
}

# --- results/models/ : Model weights ---
model_files = {
    "centralized_wsn.pt":               "results/models/centralized_wsn.pt",
    "fedavg_wsn.pt":                     "results/models/fedavg_wsn.pt",
    "fedprox_wsn.pt":                    "results/models/fedprox_wsn.pt",
    "fl_lstm_pytorch_wsn_K10_T100.pt":   "results/models/fl_lstm_pytorch_wsn_K10_T100.pt",
}

# Files to delete
delete_files = ["tuning_mu_bad.npz"]

# ============================================================
# 3. THỰC HIỆN COPY
# ============================================================
def safe_copy(src_name, dst_rel):
    src_path = os.path.join(ROOT, src_name)
    dst_path = os.path.join(ROOT, dst_rel)
    if not os.path.exists(src_path):
        print(f"  [SKIP] {src_name} (không tồn tại)")
        return False
    if os.path.exists(dst_path):
        print(f"  [SKIP] {dst_rel} (đã tồn tại)")
        return False
    shutil.copy2(src_path, dst_path)
    print(f"  [COPY] {src_name} → {dst_rel}")
    return True

print("\n" + "="*60)
print("BƯỚC 1: Copy các module cốt lõi → src/")
print("="*60)
for old, new in src_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 2: Copy các thí nghiệm → experiments/")
print("="*60)
for old, new in exp_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 3: Copy các visualization → visualization/")
print("="*60)
for old, new in viz_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 4: Copy demo → demo/")
print("="*60)
for old, new in demo_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 5: Copy ảnh → results/figures/")
print("="*60)
for pattern in figure_patterns:
    for f in glob.glob(os.path.join(ROOT, pattern)):
        fname = os.path.basename(f)
        dst_rel = os.path.join("results", "figures", fname)
        safe_copy(fname, dst_rel)

print("\n" + "="*60)
print("BƯỚC 6: Copy kết quả số liệu → results/metrics/")
print("="*60)
for old, new in metric_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 7: Copy model weights → results/models/")
print("="*60)
for old, new in model_files.items():
    safe_copy(old, new)

print("\n" + "="*60)
print("BƯỚC 8: Copy code cũ → legacy/")
print("="*60)
for old, new in legacy_files.items():
    safe_copy(old, new)

# ============================================================
# 4. TẠO src/__init__.py
# ============================================================
print("\n" + "="*60)
print("BƯỚC 9: Tạo src/__init__.py")
print("="*60)
init_content = '''"""
FL-LSTM IDS — Federated Learning + LSTM cho Phát hiện Xâm nhập IoT
===================================================================
Modules:
    model          — Kiến trúc LSTM_IDS 3 tầng (PyTorch)
    preprocessing  — Pipeline tiền xử lý 7 bước (Zero Data Leakage)
    training       — Huấn luyện cục bộ (Local Training) với FedProx Proximal Term
    evaluation     — Đánh giá mô hình (Accuracy, F1, FPR, RMSE, Confusion Matrix)
    federated      — Vòng lặp Federated Learning (Weighted FedAvg)
"""
from .model import LSTM_IDS, create_lstm_model_pytorch, clone_model_with_weights_pytorch
from .model import get_model_weights_pytorch, set_model_weights_pytorch
from .preprocessing import preprocess_data
from .training import train_client_pytorch
from .evaluation import evaluate_model_pytorch
from .federated import weighted_fedavg_pytorch, run_federated_training_pytorch
'''
init_path = os.path.join(ROOT, "src", "__init__.py")
with open(init_path, "w", encoding="utf-8") as f:
    f.write(init_content)
print(f"  [CREATE] src/__init__.py")

# ============================================================
# 5. CẬP NHẬT IMPORT TRONG CÁC FILE ĐÃ COPY
# ============================================================
print("\n" + "="*60)
print("BƯỚC 10: Cập nhật import statements")
print("="*60)

# Bảng thay thế import (old → new)
import_replacements = {
    # Model imports
    "from model_pytorch import":          "from src.model import",
    "from model_pytorch import LSTM_IDS": "from src.model import LSTM_IDS",
    "import model_pytorch":               "import src.model as model_pytorch",
    
    # Preprocessing imports  
    "from preprocess_custom import":      "from src.preprocessing import",
    "import preprocess_custom":           "import src.preprocessing as preprocess_custom",
    
    # Training imports
    "from training_pytorch import":       "from src.training import",
    "import training_pytorch":            "import src.training as training_pytorch",
    
    # Evaluation imports
    "from evaluation_pytorch import":     "from src.evaluation import",
    "import evaluation_pytorch":          "import src.evaluation as evaluation_pytorch",
    
    # Federated imports
    "from federated_learning_pytorch import": "from src.federated import",
    "import federated_learning_pytorch":      "import src.federated as federated_learning_pytorch",
}

# Thêm sys.path vào đầu file nếu cần
sys_path_header = '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
'''

# Các thư mục cần cập nhật import
update_dirs = ["experiments", "visualization", "demo"]

for subdir in update_dirs:
    dir_path = os.path.join(ROOT, subdir)
    if not os.path.isdir(dir_path):
        continue
    for fname in os.listdir(dir_path):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(dir_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        original = content
        
        # Thực hiện thay thế import
        for old_imp, new_imp in import_replacements.items():
            content = content.replace(old_imp, new_imp)
        
        # Thêm sys.path nếu file có import từ src và chưa có sys.path
        if "from src." in content and "sys.path.insert" not in content:
            # Tìm vị trí sau docstring hoặc đầu file
            if content.startswith('"""'):
                end_doc = content.find('"""', 3)
                if end_doc != -1:
                    insert_pos = end_doc + 3
                    content = content[:insert_pos] + "\n" + sys_path_header + content[insert_pos:]
                else:
                    content = sys_path_header + content
            elif content.startswith("'''"):
                end_doc = content.find("'''", 3)
                if end_doc != -1:
                    insert_pos = end_doc + 3
                    content = content[:insert_pos] + "\n" + sys_path_header + content[insert_pos:]
                else:
                    content = sys_path_header + content
            else:
                content = sys_path_header + content
        
        if content != original:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [UPDATE] {subdir}/{fname} — imports cập nhật")
        else:
            print(f"  [OK]     {subdir}/{fname} — không cần thay đổi")

# ============================================================
# 5b. CẬP NHẬT IMPORT TRONG src/ (internal references)
# ============================================================
# src/federated.py imports from model_pytorch and training_pytorch
src_internal_replacements = {
    "from model_pytorch import": "from .model import",
    "from training_pytorch import": "from .training import",
    "from evaluation_pytorch import": "from .evaluation import",
    "from preprocess_custom import": "from .preprocessing import",
}

for fname in os.listdir(os.path.join(ROOT, "src")):
    if not fname.endswith(".py") or fname == "__init__.py":
        continue
    fpath = os.path.join(ROOT, "src", fname)
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    original = content
    for old_imp, new_imp in src_internal_replacements.items():
        content = content.replace(old_imp, new_imp)
    if content != original:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [UPDATE] src/{fname} — internal imports cập nhật")

# ============================================================
# 6. CẬP NHẬT requirements.txt
# ============================================================
print("\n" + "="*60)
print("BƯỚC 11: Cập nhật requirements.txt")
print("="*60)
req_content = """# Dependencies cho FL-LSTM IDS
# Framework: PyTorch (thay thế TensorFlow)
torch>=2.0
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
imbalanced-learn>=0.10
matplotlib>=3.5
seaborn>=0.12
"""
req_path = os.path.join(ROOT, "requirements.txt")
with open(req_path, "w", encoding="utf-8") as f:
    f.write(req_content)
print(f"  [UPDATE] requirements.txt (PyTorch thay TensorFlow)")

# ============================================================
# 7. TẠO README.md
# ============================================================
print("\n" + "="*60)
print("BƯỚC 12: Tạo README.md")
print("="*60)
readme_content = """# FL-LSTM IDS — Federated Learning + LSTM cho Phát hiện Xâm nhập Mạng IoT

## Mô tả
Hệ thống Phát hiện Xâm nhập (IDS) phi tập trung cho mạng Wireless Sensor Network (WSN),
sử dụng Federated Learning kết hợp LSTM và thuật toán FedProx Adaptive-μ.

## Cấu trúc thư mục

```
code_sua/
├── src/                  # Module cốt lõi (Model, Training, Evaluation, FL)
├── experiments/          # Các script chạy thí nghiệm
├── visualization/        # Script vẽ biểu đồ
├── demo/                 # Demo bảo vệ luận văn
├── results/              # Kết quả (ảnh, số liệu, model weights)
│   ├── figures/
│   ├── metrics/
│   └── models/
├── legacy/               # Code Keras/TF cũ (tham khảo)
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy thí nghiệm chính (Adaptive-μ FedProx)

```bash
cd code_sua
python -m experiments.run_adaptive_fedprox
```

## Demo bảo vệ luận văn

```bash
cd code_sua
python -m demo.demo_prototype_ids
```

## Tác giả
- Học viên: [Tên sinh viên]
- GVHD: PGS.TS. Nguyễn Thị Mỹ Bình
- Đơn vị: Trường Đại học Công nghiệp Hà Nội — Khoa CNTT — 2025

## Tài liệu tham khảo chính
- Anwar, S. et al. (2025). Federated Learning-based Intrusion Detection in IoT-WSN. PeerJ CS 11:e2751.
"""
readme_path = os.path.join(ROOT, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"  [CREATE] README.md")

# ============================================================
# 8. XÓA FILE RÁC
# ============================================================
print("\n" + "="*60)
print("BƯỚC 13: Xóa file rác")
print("="*60)
for fname in delete_files:
    fpath = os.path.join(ROOT, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f"  [DELETE] {fname}")

# Xóa __pycache__
pycache = os.path.join(ROOT, "__pycache__")
if os.path.isdir(pycache):
    shutil.rmtree(pycache)
    print(f"  [DELETE] __pycache__/")

# ============================================================
# HOÀN TẤT
# ============================================================
print("\n" + "="*60)
print("✅ TỔ CHỨC LẠI THƯ MỤC HOÀN TẤT!")
print("="*60)
print("""
Lưu ý:
  - Các file GỐC vẫn được giữ nguyên (chưa xóa) để bạn kiểm tra.
  - Sau khi xác nhận mọi thứ hoạt động tốt, bạn có thể xóa các file gốc.
  - Chạy thí nghiệm: python -m experiments.run_adaptive_fedprox
  - Chạy demo:       python -m demo.demo_prototype_ids
""")
