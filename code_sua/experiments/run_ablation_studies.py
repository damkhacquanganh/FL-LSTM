"""
run_ablation_studies.py — Ablation Study (Optimized GPU + Professional Charts)
====================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import copy, warnings, json, os
warnings.filterwarnings('ignore')

# Import từ project
from src.preprocessing import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from src.model import LSTM_IDS, clone_model_with_weights_pytorch, set_model_weights_pytorch
from src.federated import weighted_fedavg_pytorch
from src.training import train_client_pytorch
from src.evaluation import evaluate_model_pytorch
from run_non_iid import create_non_iid_dirichlet

# ============================================================
# CẤU HÌNH
# ============================================================
MAX_SAMPLES  = 80000
BATCH_SIZE   = 2048
ROUNDS       = 30     # Dựa trên convergence_comparison_all.png: hội tụ tại ~round 10
LOCAL_EPOCHS = 1
K_CLIENTS    = 10
CENT_EPOCHS  = 15
SEED         = 42
RESULTS_FILE = 'ablation_results.json'

# ============================================================
# STYLE BIỂU ĐỒ CHUYÊN NGHIỆP
# ============================================================
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

COLORS = {
    'centralized': '#1565C0',  # Deep Blue
    'fedavg':      '#D32F2F',  # Deep Red
    'fedprox':     '#2E7D32',  # Deep Green
    'highlight':   '#FF6F00',  # Amber
    'gray':        '#78909C',  # Blue Gray
}

# ============================================================
# HÀM HUẤN LUYỆN
# ============================================================
def train_centralized(X_train, y_train, X_test, y_test, device, epochs=CENT_EPOCHS):
    print(f"\n--- Training CENTRALIZED LSTM ({epochs} epochs) ---")
    input_size = X_train.shape[2]
    num_classes = y_train.shape[1]
    model = LSTM_IDS(input_size, num_classes).to(device)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

    dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    acc_history = []
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        metrics = evaluate_model_pytorch(model, X_test, y_test, device)
        acc_history.append(metrics['accuracy'])
        print(f"  Epoch {epoch+1}/{epochs} - Acc: {metrics['accuracy']*100:.2f}%")

    final_metrics = evaluate_model_pytorch(model, X_test, y_test, device)
    return acc_history, final_metrics


def train_federated(X_train, y_train, X_test, y_test, alpha, mu, algo, device, rounds=ROUNDS):
    print(f"\n--- {algo.upper()} | alpha={alpha} | mu={mu} | {rounds} rounds ---")
    client_data = create_non_iid_dirichlet(X_train, y_train, K=K_CLIENTS, alpha=alpha)

    input_size = X_train.shape[2]
    num_classes = y_train.shape[1]
    global_model = LSTM_IDS(input_size, num_classes).to(device)
    acc_history = []

    for t in range(rounds):
        client_weights, client_sizes = [], []
        for k in range(K_CLIENTS):
            X_k, y_k = client_data[k]
            if len(X_k) < 10:
                continue
            local_model = clone_model_with_weights_pytorch(global_model)
            gw = [p.data.clone() for p in global_model.parameters()] if algo == 'fedprox' else None

            w_k, n_k = train_client_pytorch(
                local_model, X_k, y_k,
                epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE,
                mu=mu, global_weights=gw, device=device
            )
            client_weights.append(w_k)
            client_sizes.append(n_k)

        set_model_weights_pytorch(global_model, weighted_fedavg_pytorch(client_weights, client_sizes))
        metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
        acc_history.append(metrics['accuracy'])
        print(f"  Round {t+1}/{rounds} - Acc: {metrics['accuracy']*100:.2f}%")

    final_metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
    return acc_history, final_metrics


# ============================================================
# THỰC NGHIỆM 1: SO SÁNH 3 PHƯƠNG PHÁP
# ============================================================
def exp1_three_way_comparison(X_train, y_train, X_test, y_test, device):
    print("\n" + "="*60)
    print("  THỰC NGHIỆM 1: CENTRALIZED vs FEDAVG vs FEDPROX")
    print("="*60)

    cent_h, cent_m  = train_centralized(X_train, y_train, X_test, y_test, device)
    avg_h, avg_m    = train_federated(X_train, y_train, X_test, y_test, alpha=0.5, mu=0.0, algo='fedavg', device=device)
    prox_h, prox_m  = train_federated(X_train, y_train, X_test, y_test, alpha=0.5, mu=0.0001, algo='fedprox', device=device)

    # In bảng tổng kết
    print("\n" + "="*65)
    print(f"{'Metric':<18} {'Centralized':>12} {'FedAvg':>12} {'FedProx':>12}")
    print("="*65)
    for key, label in [('accuracy','Accuracy'), ('precision','Precision'), ('recall','Recall'), ('f1_score','F1-Score'), ('FPR','FPR'), ('RMSE','RMSE')]:
        scale = 100 if key in ['accuracy','precision','recall','f1_score','FPR'] else 1
        unit  = '%' if key in ['accuracy','precision','recall','f1_score','FPR'] else ''
        print(f"{label:<18} {cent_m[key]*scale:>11.4f}{unit} {avg_m[key]*scale:>10.4f}{unit} {prox_m[key]*scale:>10.4f}{unit}")
    print("="*65)

    # ---- VẼ BIỂU ĐỒ 1: CONVERGENCE (2 subplots) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Thực nghiệm 1: So sánh Centralized vs FedAvg vs FedProx\n(Non-IID Dirichlet α=0.5)',
                 fontsize=14, fontweight='bold', y=1.02)

    fed_x  = list(range(1, ROUNDS + 1))      # 30 rounds cho FL
    cent_x = list(range(1, CENT_EPOCHS + 1)) # 15 epochs cho Centralized

    # --- Tìm điểm hội tụ (khi delta acc < 0.5%) ---
    def find_convergence(history, threshold=0.005):
        for i in range(5, len(history)):
            recent = history[max(0,i-4):i+1]
            if max(recent) - min(recent) < threshold:
                return i + 1
        return len(history)

    avg_conv  = find_convergence(avg_h)
    prox_conv = find_convergence(prox_h)

    # --- 1A: Đường hội tụ ---
    ax1.plot(cent_x, cent_h, 's--', label='Centralized (15 Epochs)', color=COLORS['centralized'],
             linewidth=2.5, markersize=5, zorder=3)
    ax1.plot(fed_x, avg_h,  'o-',  label='FedAvg (α=0.5)', color=COLORS['fedavg'],
             linewidth=2.5, markersize=5, zorder=3)
    ax1.plot(fed_x, prox_h, '^-',  label='FedProx (α=0.5, μ=0.0001)', color=COLORS['fedprox'],
             linewidth=2.5, markersize=5, zorder=3)

    # Vùng tô hội tụ (sau round 10 từ ảnh 100-round reference)
    ax1.axvspan(10, ROUNDS, alpha=0.06, color='green', label='Convergence zone (from 100-round ref.)')
    ax1.axvline(x=10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(10.3, 0.08, 'Convergence\nzone (≥Round 10)', fontsize=8.5, color='gray', style='italic')

    # Annotation điểm cuối
    ax1.annotate(f'{prox_h[-1]*100:.1f}%',
                 xy=(ROUNDS, prox_h[-1]), xytext=(-25, 6), textcoords='offset points',
                 fontsize=9, color=COLORS['fedprox'], fontweight='bold')
    ax1.annotate(f'{avg_h[-1]*100:.1f}%',
                 xy=(ROUNDS, avg_h[-1]), xytext=(-25, -14), textcoords='offset points',
                 fontsize=9, color=COLORS['fedavg'], fontweight='bold')
    ax1.annotate(f'{cent_h[-1]*100:.1f}%',
                 xy=(CENT_EPOCHS, cent_h[-1]), xytext=(-25, 6), textcoords='offset points',
                 fontsize=9, color=COLORS['centralized'], fontweight='bold')

    ax1.set_xlabel('Epoch / Communication Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Convergence Curves (30 Rounds)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.set_ylim([0.0, 1.08])
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_xticks(range(0, ROUNDS+1, 5))
    ax1.set_xlim([0, ROUNDS + 1])

    # --- 1B: Bar chart các metrics ---
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cent_vals = [cent_m['accuracy'], cent_m['precision'], cent_m['recall'], cent_m['f1_score']]
    avg_vals  = [avg_m['accuracy'],  avg_m['precision'],  avg_m['recall'],  avg_m['f1_score']]
    prox_vals = [prox_m['accuracy'], prox_m['precision'], prox_m['recall'], prox_m['f1_score']]

    x = np.arange(len(metric_names))
    width = 0.25
    bars1 = ax2.bar(x - width, [v*100 for v in cent_vals], width, label='Centralized',
                    color=COLORS['centralized'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x,         [v*100 for v in avg_vals],  width, label='FedAvg',
                    color=COLORS['fedavg'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars3 = ax2.bar(x + width, [v*100 for v in prox_vals], width, label='FedProx',
                    color=COLORS['fedprox'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Thêm giá trị số trên đầu mỗi cột
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, fontsize=11)
    ax2.set_ylabel('Score (%)', fontsize=12)
    ax2.set_title('(b) Performance Metrics', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim([70, 105])

    plt.tight_layout()
    plt.savefig('exp1_three_way_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n=> Saved: exp1_three_way_comparison.png")

    return {'cent_h': cent_h, 'avg_h': avg_h, 'prox_h': prox_h,
            'cent_m': cent_m, 'avg_m': avg_m, 'prox_m': prox_m}


# ============================================================
# THỰC NGHIỆM 2: GRID SEARCH MU
# ============================================================
def exp2_mu_grid_search(X_train, y_train, X_test, y_test, device):
    print("\n" + "="*60)
    print("  THỰC NGHIỆM 2: PHÂN TÍCH THAM SỐ μ (FedProx)")
    print("="*60)

    mu_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0]
    mu_labels = ['μ=0.1', 'μ=0.01', 'μ=0.001', 'μ=0.0001', 'μ=0.00001', 'μ=0\n(FedAvg)']
    results = {}
    histories = {}

    for mu in mu_values:
        algo = 'fedprox' if mu > 0 else 'fedavg'
        h, m = train_federated(X_train, y_train, X_test, y_test, alpha=0.5, mu=mu, algo=algo, device=device)
        results[str(mu)] = m
        histories[str(mu)] = h

    # In bảng tổng kết
    print("\n" + "="*80)
    print(f"{'μ Value':<12}", end="")
    for mu in mu_values:
        print(f" {'μ='+str(mu):>12}", end="")
    print("\n" + "="*80)
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        label = key.replace('_',' ').title()
        print(f"{label:<12}", end="")
        for mu in mu_values:
            print(f" {results[str(mu)][key]*100:>11.2f}%", end="")
        print()
    print("="*80)

    # ---- VẼ BIỂU ĐỒ 2: MU ANALYSIS (2 subplots) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Thực nghiệm 2: Ảnh hưởng của tham số μ đến hiệu năng FedProx\n(Non-IID Dirichlet α=0.5)',
                 fontsize=14, fontweight='bold', y=1.02)

    # --- 2A: Bar chart Accuracy theo mu ---
    accuracies = [results[str(mu)]['accuracy'] * 100 for mu in mu_values]
    best_idx = np.argmax(accuracies)
    bar_colors = [COLORS['gray']] * len(mu_values)
    bar_colors[best_idx] = COLORS['fedprox']       # Tô xanh cho giá trị tốt nhất
    bar_colors[-1] = COLORS['fedavg']               # Tô đỏ cho FedAvg (mu=0)

    bars = ax1.bar(mu_labels, accuracies, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, width=0.6)

    # Thêm giá trị số trên đầu cột
    for i, bar in enumerate(bars):
        height = bar.get_height()
        fontw = 'bold' if i == best_idx else 'normal'
        ax1.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight=fontw)
        if i == best_idx:
            ax1.annotate('★ BEST',
                         xy=(bar.get_x() + bar.get_width() / 2, height + 0.8),
                         ha='center', fontsize=9, color=COLORS['fedprox'], fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy theo giá trị μ', fontsize=13, fontweight='bold')

    # Tính toán ylim thông minh
    min_acc = min(accuracies)
    ax1.set_ylim([max(0, min_acc - 3), 100])

    # --- 2B: Convergence curves cho tất cả mu ---
    rounds_x = range(1, ROUNDS + 1)
    line_styles = ['-', '--', '-.', '-', ':', '-']
    markers = ['v', 'D', 'p', '^', 'x', 'o']
    color_list = ['#E91E63', '#FF9800', '#9C27B0', COLORS['fedprox'], '#00BCD4', COLORS['fedavg']]

    for i, mu in enumerate(mu_values):
        label = f'μ={mu}' if mu > 0 else 'μ=0 (FedAvg)'
        lw = 3 if i == best_idx else 1.5
        alpha_val = 1.0 if i == best_idx else 0.6
        ax2.plot(rounds_x, histories[str(mu)], linestyle=line_styles[i], marker=markers[i],
                 label=label, color=color_list[i], linewidth=lw, markersize=5, alpha=alpha_val)

    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('(b) Convergence Curves theo μ', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_xticks(range(1, ROUNDS+1))

    plt.tight_layout()
    plt.savefig('exp2_mu_grid_search.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n=> Saved: exp2_mu_grid_search.png")

    return results, histories


# ============================================================
# THỰC NGHIỆM 3: GRID SEARCH ALPHA
# ============================================================
def exp3_alpha_grid_search(X_train, y_train, X_test, y_test, device):
    print("\n" + "="*60)
    print("  THỰC NGHIỆM 3: ẢNH HƯỞNG ĐỘ PHÂN TÁN DỮ LIỆU (α)")
    print("="*60)

    alpha_values = [0.1, 0.3, 0.5, 1.0, 5.0]
    avg_results = {}
    prox_results = {}

    for a in alpha_values:
        _, m_avg  = train_federated(X_train, y_train, X_test, y_test, alpha=a, mu=0.0, algo='fedavg', device=device)
        _, m_prox = train_federated(X_train, y_train, X_test, y_test, alpha=a, mu=0.0001, algo='fedprox', device=device)
        avg_results[str(a)]  = m_avg
        prox_results[str(a)] = m_prox

    # In bảng tổng kết
    print("\n" + "="*70)
    print(f"{'Alpha':<10} {'FedAvg Acc':>12} {'FedProx Acc':>12} {'Improvement':>14}")
    print("="*70)
    for a in alpha_values:
        avg_acc = avg_results[str(a)]['accuracy'] * 100
        prox_acc = prox_results[str(a)]['accuracy'] * 100
        diff = prox_acc - avg_acc
        sign = "+" if diff >= 0 else ""
        print(f"α = {a:<6} {avg_acc:>11.2f}% {prox_acc:>11.2f}% {sign}{diff:>12.2f}%")
    print("="*70)

    # ---- VẼ BIỂU ĐỒ 3: ALPHA ANALYSIS (2 subplots) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Thực nghiệm 3: Ảnh hưởng của độ phân tán dữ liệu (Dirichlet α)\nFedAvg vs FedProx',
                 fontsize=14, fontweight='bold', y=1.02)

    alpha_labels = [f'α={a}' for a in alpha_values]
    avg_accs  = [avg_results[str(a)]['accuracy'] * 100 for a in alpha_values]
    prox_accs = [prox_results[str(a)]['accuracy'] * 100 for a in alpha_values]

    # --- 3A: Grouped Bar Chart ---
    x = np.arange(len(alpha_values))
    width = 0.35
    bars1 = ax1.bar(x - width/2, avg_accs,  width, label='FedAvg', color=COLORS['fedavg'],
                    alpha=0.85, edgecolor='white', linewidth=1)
    bars2 = ax1.bar(x + width/2, prox_accs, width, label='FedProx (μ=0.0001)', color=COLORS['fedprox'],
                    alpha=0.85, edgecolor='white', linewidth=1)

    # Thêm giá trị số trên đầu cột
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_labels, fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy: FedAvg vs FedProx', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')

    # Zoom trục Y thông minh
    all_vals = avg_accs + prox_accs
    min_val = min(all_vals)
    ax1.set_ylim([max(0, min_val - 5), 100])

    # Thêm annotation chú thích
    ax1.annotate('← Dữ liệu lệch nhiều    Dữ liệu đồng đều →',
                 xy=(0.5, 0.02), xycoords='axes fraction',
                 ha='center', fontsize=9, style='italic', color='gray')

    # --- 3B: Line chart + Improvement ---
    ax2.plot(alpha_labels, avg_accs,  'o-', label='FedAvg', color=COLORS['fedavg'],
             linewidth=2.5, markersize=8)
    ax2.plot(alpha_labels, prox_accs, '^-', label='FedProx', color=COLORS['fedprox'],
             linewidth=2.5, markersize=8)

    # Tô vùng chênh lệch
    ax2.fill_between(range(len(alpha_values)), avg_accs, prox_accs,
                     alpha=0.15, color=COLORS['fedprox'], label='FedProx improvement')

    ax2.set_xlabel('Dirichlet α', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Xu hướng theo mức độ Non-IID', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim([max(0, min_val - 5), 100])

    plt.tight_layout()
    plt.savefig('exp3_alpha_grid_search.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n=> Saved: exp3_alpha_grid_search.png")

    return avg_results, prox_results


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading WSN-DS dataset...")

    df = load_dataset("wsn")
    target = auto_detect_target(df)
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, target_column=target)

    # Chạy 3 thực nghiệm
    r1 = exp1_three_way_comparison(X_train, y_train, X_test, y_test, device)
    r2_results, r2_histories = exp2_mu_grid_search(X_train, y_train, X_test, y_test, device)
    r3_avg, r3_prox = exp3_alpha_grid_search(X_train, y_train, X_test, y_test, device)

    # Lưu kết quả ra file JSON
    def serialize_metrics(m):
        """Serialize metrics dict, bỏ qua confusion_matrix vì không JSON được."""
        return {k: float(v) for k, v in m.items() if k != 'confusion_matrix'}

    all_results = {
        'exp1': {
            'centralized': serialize_metrics(r1['cent_m']),
            'fedavg':      serialize_metrics(r1['avg_m']),
            'fedprox':     serialize_metrics(r1['prox_m']),
            'cent_history': [float(x) for x in r1['cent_h']],
            'avg_history':  [float(x) for x in r1['avg_h']],
            'prox_history': [float(x) for x in r1['prox_h']],
        },
        'exp2': {str(k): serialize_metrics(v) for k, v in r2_results.items()},
        'exp3': {
            'fedavg':  {str(k): serialize_metrics(v) for k, v in r3_avg.items()},
            'fedprox': {str(k): serialize_metrics(v) for k, v in r3_prox.items()},
        }
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=> Saved results to: {RESULTS_FILE}")

    print("\n" + "="*60)
    print("  ✅ TẤT CẢ THỰC NGHIỆM ĐÃ HOÀN THÀNH!")
    print("  📊 Files đã tạo:")
    print("     1. exp1_three_way_comparison.png")
    print("     2. exp2_mu_grid_search.png")
    print("     3. exp3_alpha_grid_search.png")
    print("     4. ablation_results.json (kết quả số)")
    print("="*60)


if __name__ == '__main__':
    main()
