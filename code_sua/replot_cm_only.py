"""
replot_cm_only.py — Vẽ lại Confusion Matrix từ số liệu đã có (KHÔNG cần train lại)
===================================================================================
Bug: CLASS_NAMES cũ ['Normal','Blackhole','Flooding','Grayhole','Scheduling']
     không khớp với thứ tự LabelEncoder alphabetical.

Thứ tự ĐÚNG theo LabelEncoder: ['Blackhole','Flooding','Grayhole','Normal','Scheduling']
  → Index 0 = Blackhole, 1 = Flooding, 2 = Grayhole, 3 = Normal, 4 = TDMA/Scheduling

Số liệu trong ma trận là ĐÚNG, chỉ cần đổi nhãn trục là xong.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Thứ tự ĐÚNG từ LabelEncoder alphabetical
CLASS_NAMES = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'Scheduling']

# ============================================================
# Số liệu từ ảnh cũ (đọc từ cm_adaptive_iid.png)
# Hàng = True label (theo thứ tự đúng: BH, FL, GH, NM, SC)
# Cột = Predicted label (theo thứ tự đúng: BH, FL, GH, NM, SC)
# ============================================================
cm_iid = np.array([
    [435,     0,   0,     0,   0],   # True=Blackhole
    [  0,   137,   0,     0,   0],   # True=Flooding
    [250,     1, 379,     0,   0],   # True=Grayhole
    [  1,    43, 353, 14024,  16],   # True=Normal (chiếm đa số)
    [  1,     0,   1,    24, 255],   # True=Scheduling
])

# Số liệu từ cm_adaptive_noniid.png
cm_noniid = np.array([
    [200,     0, 235,     0,   0],   # True=Blackhole
    [  0,    92,  43,     0,   2],   # True=Flooding
    [ 24,    13, 593,     0,   0],   # True=Grayhole
    [  0,    74, 347, 14005,  11],   # True=Normal
    [  2,     4,   1,    24, 250],   # True=Scheduling
])

def plot_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Tính recall (% theo hàng)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm.astype(float) / row_sums * 100

    # Vẽ heatmap với số đếm tuyệt đối
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, linecolor='gray')

    # Viết số vào từng ô: count + %
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j + 0.5, i + 0.38, f'{count}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
            ax.text(j + 0.5, i + 0.65, f'({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
    ax.set_ylabel('True Label', fontsize=11, labelpad=10)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Thống kê nhanh
print("=== KIỂM TRA PHÂN PHỐI TEST SET ===")
for i, name in enumerate(CLASS_NAMES):
    total_iid = cm_iid[i].sum()
    correct_iid = cm_iid[i, i]
    total_noniid = cm_noniid[i].sum()
    correct_noniid = cm_noniid[i, i]
    print(f"  {name:12s}: IID {correct_iid}/{total_iid} ({correct_iid/total_iid*100:.1f}%)"
          f"  |  Non-IID {correct_noniid}/{total_noniid} ({correct_noniid/total_noniid*100:.1f}%)")

print(f"\nTổng test samples: {cm_iid.sum()}")
print(f"Accuracy IID:     {np.diag(cm_iid).sum()/cm_iid.sum()*100:.2f}%")
print(f"Accuracy Non-IID: {np.diag(cm_noniid).sum()/cm_noniid.sum()*100:.2f}%")

# Vẽ lại
plot_cm(cm_iid,
        'Confusion Matrix — FedProx Adaptive-μ (IID)',
        'cm_adaptive_iid.png')

plot_cm(cm_noniid,
        'Confusion Matrix — FedProx Adaptive-μ (Non-IID  α=0.1)',
        'cm_adaptive_noniid.png')

print("\nDone! Không cần train lại.")
