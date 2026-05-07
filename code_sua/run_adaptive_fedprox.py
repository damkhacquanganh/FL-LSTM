"""
run_adaptive_fedprox.py — Thực nghiệm Adaptive-μ FedProx
==========================================================
Ý tưởng: μ tự động điều chỉnh theo gradient divergence mỗi round
  - Divergence cao (data lệch nhiều) → μ tăng → kìm client drift mạnh hơn
  - Divergence thấp (data đồng nhất) → μ giảm → gần FedAvg

So sánh:
  1. FedAvg          (baseline - Anwar et al. 2025)
  2. FedProx (cố định μ=0.001)
  3. FedProx-Adaptive (μ tự điều chỉnh) ← NOVELTY
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import copy, warnings, json
warnings.filterwarnings('ignore')

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import LSTM_IDS, clone_model_with_weights_pytorch, set_model_weights_pytorch
from federated_learning_pytorch import weighted_fedavg_pytorch
from evaluation_pytorch import evaluate_model_pytorch
from run_non_iid import create_non_iid_dirichlet

# ============================================================
# CẤU HÌNH
# ============================================================
MAX_SAMPLES  = 80000
BATCH_SIZE   = 2048
ROUNDS       = 30
LOCAL_EPOCHS = 1
K_CLIENTS    = 10
SEED         = 42

# Adaptive-μ hyperparameters
MU_MIN   = 0.00001   # μ nhỏ nhất (gần FedAvg khi IID)
MU_MAX   = 0.05      # μ lớn nhất (kìm mạnh khi Non-IID nặng)
MU_FIXED = 0.001     # μ cố định cho FedProx baseline

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
    'centralized':'#5E35B1',  # Tím (Purple) cho baseline cao nhất
    'fedavg':    '#D32F2F',   # Đỏ
    'fedprox':   '#1565C0',   # Xanh dương
    'adaptive':  '#2E7D32',   # Xanh lá — proposed method
    'highlight': '#FF6F00',
}

# ============================================================
# HÀM TÍNH GRADIENT DIVERGENCE
# ============================================================
def compute_divergence(local_weights_list, global_weights):
    """
    Đo gradient divergence: trung bình khoảng cách L2
    giữa local model và global model.
    Server đã có toàn bộ thông tin này → không vi phạm privacy.

    divergence = mean_k( ||w_k - w_global||^2 )
    """
    divergences = []
    for w_k in local_weights_list:
        diff = 0.0
        for p_k, p_g in zip(w_k, global_weights):
            diff += torch.norm(p_k.float() - p_g.float()).item() ** 2
        divergences.append(diff)
    return float(np.mean(divergences))


def adaptive_mu(divergence, mu_min=MU_MIN, mu_max=MU_MAX, scale=0.1):
    """
    Tính μ thích nghi từ divergence score.
    Dùng sigmoid để map divergence → [mu_min, mu_max].

    Khi divergence cao  → μ gần mu_max (kìm mạnh)
    Khi divergence thấp → μ gần mu_min (như FedAvg)
    """
    sig = 1.0 / (1.0 + np.exp(-scale * (divergence - 50)))
    return mu_min + (mu_max - mu_min) * sig


# ============================================================
# HÀM TRAIN CENTRALIZED (Baseline Upper Bound)
# ============================================================
def train_centralized(X_train, y_train, X_test, y_test, device, epochs=15):
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


# ============================================================
# HÀM TRAIN CLIENT (tự viết để kiểm soát μ)
# ============================================================
def train_client_local(model, X_k, y_k, mu, global_weights, device):
    """Train 1 client với FedProx loss."""
    X_t = torch.tensor(X_k, dtype=torch.float32)
    y_t = torch.tensor(np.argmax(y_k, axis=1), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)

            # Proximal term: (μ/2) * ||w - w_global||²
            if mu > 0 and global_weights is not None:
                prox = sum(
                    torch.norm(p - g.to(device)) ** 2
                    for p, g in zip(model.parameters(), global_weights)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    weights = [p.data.clone().cpu() for p in model.parameters()]
    return weights, len(X_k)


# ============================================================
# HÀM TRAIN FEDERATED (3 chế độ)
# ============================================================
def train_federated_mode(client_data, X_train, y_train, X_test, y_test,
                         alpha, mode, device):
    """
    mode: 'fedavg' | 'fedprox_fixed' | 'fedprox_adaptive'
    """
    label_map = {
        'fedavg':           f'FedAvg (α={alpha})',
        'fedprox_fixed':    f'FedProx fixed μ={MU_FIXED} (α={alpha})',
        'fedprox_adaptive': f'FedProx Adaptive-μ (α={alpha})',
    }
    print(f"\n--- {label_map[mode]} | {ROUNDS} rounds ---")

    input_size  = X_train.shape[2]
    num_classes = y_train.shape[1]
    global_model = LSTM_IDS(input_size, num_classes).to(device)

    acc_history = []
    mu_history  = []   # Theo dõi μ thay đổi như thế nào (chỉ cho adaptive)

    for t in range(ROUNDS):
        client_weights_list = []
        client_sizes        = []
        local_weight_tensors = []  # Dùng để tính divergence

        gw = [p.data.clone().cpu() for p in global_model.parameters()]

        for k in range(K_CLIENTS):
            X_k, y_k = client_data[k]
            if len(X_k) < 10:
                continue

            local_model = clone_model_with_weights_pytorch(global_model)

            if mode == 'fedavg':
                mu_t = 0.0
            elif mode == 'fedprox_fixed':
                mu_t = MU_FIXED
            else:  # adaptive — mu sẽ được tính sau khi có divergence
                mu_t = MU_FIXED  # placeholder, sẽ update bên dưới

            w_k, n_k = train_client_local(local_model, X_k, y_k,
                                          mu=mu_t,
                                          global_weights=gw if mu_t > 0 else None,
                                          device=device)
            client_weights_list.append(w_k)
            client_sizes.append(n_k)
            local_weight_tensors.append([torch.tensor(w) for w in w_k])

        # --- Tính divergence và cập nhật μ cho adaptive mode ---
        if mode == 'fedprox_adaptive' and len(local_weight_tensors) > 0:
            gw_t = [torch.tensor(w) for w in gw]
            div  = compute_divergence(local_weight_tensors, gw_t)
            mu_t = adaptive_mu(div)
            mu_history.append(mu_t)

            # Re-train với μ đúng (round 1 dùng mu_fixed, từ round 2 dùng adaptive)
            if t > 0:
                client_weights_list = []
                client_sizes = []
                for k in range(K_CLIENTS):
                    X_k, y_k = client_data[k]
                    if len(X_k) < 10:
                        continue
                    local_model = clone_model_with_weights_pytorch(global_model)
                    w_k, n_k = train_client_local(local_model, X_k, y_k,
                                                  mu=mu_t, global_weights=gw,
                                                  device=device)
                    client_weights_list.append(w_k)
                    client_sizes.append(n_k)
            if (t+1) % 5 == 0:
                print(f"    [Round {t+1}] divergence={div:.2f} → μ_adaptive={mu_t:.6f}")
        else:
            mu_history.append(MU_FIXED if mode == 'fedprox_fixed' else 0.0)

        # Aggregate
        set_model_weights_pytorch(global_model,
            weighted_fedavg_pytorch(client_weights_list, client_sizes))

        metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
        acc_history.append(metrics['accuracy'])
        if (t+1) % 5 == 0 or t == 0:
            print(f"  Round {t+1}/{ROUNDS} - Acc: {metrics['accuracy']*100:.2f}%")

    final_metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
    return acc_history, mu_history, final_metrics


# ============================================================
# THỬ NGHIỆM CHÍNH: So sánh 3 phương pháp × 5 alpha
# ============================================================
def run_adaptive_experiment(X_train, y_train, X_test, y_test, device):
    alpha_values = [0.1, 0.3, 0.5, 1.0, 5.0]

    print("\n" + "="*60)
    print("  RUNNING CENTRALIZED BASELINE")
    print("="*60)
    cent_h, cent_m = train_centralized(X_train, y_train, X_test, y_test, device, epochs=15)

    results = {
        'centralized':      {'all': cent_m},
        'fedavg':           {},
        'fedprox_fixed':    {},
        'fedprox_adaptive': {},
    }
    histories = {
        'fedavg':           {},
        'fedprox_fixed':    {},
        'fedprox_adaptive': {},
    }
    mu_histories = {}

    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"  α = {alpha}")
        print(f"{'='*60}")

        print(f"[!] Tạo phân chia dữ liệu Non-IID CHUNG cho α={alpha}...")
        client_data = create_non_iid_dirichlet(X_train, y_train, K=K_CLIENTS, alpha=alpha)

        for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
            h, mu_h, m = train_federated_mode(
                client_data, X_train, y_train, X_test, y_test, alpha, mode, device)
            results[mode][str(alpha)]   = m
            histories[mode][str(alpha)] = h
            mu_histories[f'{mode}_{alpha}'] = mu_h

    # ---- In bảng tổng kết ----
    print("\n" + "="*90)
    print(f"CENTRALIZED ACCURACY: {cent_m['accuracy']*100:.2f}% (Upper Bound)")
    print("="*90)
    print(f"{'Alpha':<8} {'FedAvg':>12} {'FedProx-Fixed':>15} {'FedProx-Adaptive':>18} {'Improvement':>12}")
    print("="*90)
    for a in alpha_values:
        avg_acc   = results['fedavg'][str(a)]['accuracy'] * 100
        fixed_acc = results['fedprox_fixed'][str(a)]['accuracy'] * 100
        adapt_acc = results['fedprox_adaptive'][str(a)]['accuracy'] * 100
        gain      = adapt_acc - avg_acc
        sign      = "+" if gain >= 0 else ""
        print(f"α={a:<6} {avg_acc:>11.2f}% {fixed_acc:>14.2f}% {adapt_acc:>17.2f}% {sign}{gain:>10.2f}%")
    print("="*90)

    return results, histories, mu_histories, alpha_values, cent_h, cent_m


# ============================================================
# VẼ BIỂU ĐỒ
# ============================================================
def plot_results(results, histories, mu_histories, alpha_values, cent_h, cent_m):
    alpha_labels = [f'α={a}' for a in alpha_values]

    cent_acc    = cent_m['accuracy'] * 100
    avg_accs    = [results['fedavg'][str(a)]['accuracy'] * 100 for a in alpha_values]
    fixed_accs  = [results['fedprox_fixed'][str(a)]['accuracy'] * 100 for a in alpha_values]
    adapt_accs  = [results['fedprox_adaptive'][str(a)]['accuracy'] * 100 for a in alpha_values]

    # ---- Figure 1: Accuracy comparison across alpha ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Adaptive-μ FedProx: So sánh hiệu năng theo độ phân tán dữ liệu\n'
                 'FedAvg vs FedProx(fixed μ) vs FedProx(Adaptive-μ)',
                 fontsize=14, fontweight='bold', y=1.02)

    # --- Bar chart ---
    x = np.arange(len(alpha_values))
    width = 0.25
    b1 = ax1.bar(x - width, avg_accs,   width, label='FedAvg (baseline)',
                 color=COLORS['fedavg'], alpha=0.85, edgecolor='white')
    b2 = ax1.bar(x,         fixed_accs,  width, label=f'FedProx (μ={MU_FIXED} fixed)',
                 color=COLORS['fedprox'], alpha=0.85, edgecolor='white')
    b3 = ax1.bar(x + width, adapt_accs, width, label='FedProx-Adaptive (proposed)',
                 color=COLORS['adaptive'], alpha=0.85, edgecolor='white')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_labels, fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Accuracy theo mức độ Non-IID', fontsize=13, fontweight='bold')
    
    # Vẽ thêm đường line Centralized
    ax1.axhline(y=cent_acc, color=COLORS['centralized'], linewidth=2, linestyle='--', label=f'Centralized ({cent_acc:.1f}%)', alpha=0.8)
    
    ax1.legend(fontsize=9, loc='lower right')
    min_v = min(avg_accs + fixed_accs + adapt_accs + [cent_acc])
    ax1.set_ylim([max(0, min_v - 3), 105])
    ax1.annotate('← Dữ liệu lệch nhiều         Dữ liệu đồng đều →',
                 xy=(0.5, 0.02), xycoords='axes fraction',
                 ha='center', fontsize=9, style='italic', color='gray')

    # --- Line chart: improvement over FedAvg ---
    gain_fixed  = [f - a for f, a in zip(fixed_accs,  avg_accs)]
    gain_adapt  = [f - a for f, a in zip(adapt_accs,  avg_accs)]

    ax2.axhline(y=0, color=COLORS['fedavg'], linewidth=2, linestyle='--',
                label='FedAvg (baseline = 0)', alpha=0.7)
    ax2.plot(alpha_labels, gain_fixed, 'D--', color=COLORS['fedprox'],
             linewidth=2.5, markersize=8, label=f'FedProx fixed μ={MU_FIXED}')
    ax2.plot(alpha_labels, gain_adapt, '^-',  color=COLORS['adaptive'],
             linewidth=2.5, markersize=8, label='FedProx-Adaptive (proposed)')
    ax2.fill_between(range(len(alpha_values)), gain_fixed, gain_adapt,
                     alpha=0.15, color=COLORS['adaptive'])

    for i, (gf, ga) in enumerate(zip(gain_fixed, gain_adapt)):
        ax2.annotate(f'{ga:+.2f}%', xy=(i, ga), xytext=(0, 8),
                     textcoords='offset points', ha='center',
                     fontsize=9, color=COLORS['adaptive'], fontweight='bold')

    ax2.set_ylabel('Accuracy cải thiện so với FedAvg (%)', fontsize=12)
    ax2.set_title('(b) Mức cải thiện của FedProx-Adaptive\nso với FedAvg', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.set_xticks(range(len(alpha_values)))
    ax2.set_xticklabels(alpha_labels)

    plt.tight_layout()
    plt.savefig('exp4_adaptive_fedprox_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Saved: exp4_adaptive_fedprox_comparison.png")

    # ---- Figure 2: Convergence curves tại alpha=0.1 (worst case) ----
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Convergence Curves: FedAvg vs FedProx(fixed) vs FedProx(Adaptive)\n'
                  'Tại các mức độ Non-IID khác nhau',
                  fontsize=13, fontweight='bold', y=1.02)

    selected_alphas = [0.1, 0.5, 5.0]
    for ax, a in zip(axes, selected_alphas):
        rounds_x = list(range(1, ROUNDS + 1))
        
        # Centralized line
        ax.axhline(y=cent_m['accuracy']*100, color=COLORS['centralized'], linewidth=2.5, linestyle=':', 
                   label=f'Centralized ({cent_acc:.1f}%)', alpha=0.8)
                   
        ax.plot(rounds_x, histories['fedavg'][str(a)], 'o-',
                color=COLORS['fedavg'], linewidth=2, markersize=4,
                label='FedAvg', alpha=0.9)
        ax.plot(rounds_x, histories['fedprox_fixed'][str(a)], 'D--',
                color=COLORS['fedprox'], linewidth=2, markersize=4,
                label=f'FedProx (μ={MU_FIXED})', alpha=0.9)
        ax.plot(rounds_x, histories['fedprox_adaptive'][str(a)], '^-',
                color=COLORS['adaptive'], linewidth=2.5, markersize=4,
                label='FedProx-Adaptive', alpha=1.0)

        # Annotation giá trị cuối
        for mode, color, offset in [
            ('fedavg', COLORS['fedavg'], -14),
            ('fedprox_fixed', COLORS['fedprox'], -28),
            ('fedprox_adaptive', COLORS['adaptive'], 6),
        ]:
            val = histories[mode][str(a)][-1]
            ax.annotate(f'{val*100:.1f}%',
                        xy=(ROUNDS, val),
                        xytext=(-20, offset), textcoords='offset points',
                        fontsize=8.5, color=color, fontweight='bold')

        ax.set_title(f'α = {a}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Communication Round', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8.5, loc='lower right')
        ax.set_xticks(range(0, ROUNDS+1, 5))
        ax.set_xlim([0, ROUNDS + 1])

        # Highlight convergence zone
        ax.axvspan(10, ROUNDS, alpha=0.04, color='green')
        ax.axvline(x=10, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    plt.tight_layout()
    plt.savefig('exp4_adaptive_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Saved: exp4_adaptive_convergence_curves.png")

    # ---- Figure 3: μ thay đổi theo round (tại alpha=0.1) ----
    fig3, ax = plt.subplots(figsize=(10, 5))
    fig3.suptitle('Adaptive-μ: Giá trị μ tự động điều chỉnh theo divergence\n'
                  'Tại α=0.1 (Non-IID nặng nhất)',
                  fontsize=13, fontweight='bold')

    mu_adapt_01 = mu_histories.get('fedprox_adaptive_0.1', [])
    if mu_adapt_01:
        rounds_x = list(range(1, len(mu_adapt_01) + 1))
        ax.plot(rounds_x, mu_adapt_01, 's-', color=COLORS['adaptive'],
                linewidth=2.5, markersize=6, label='Adaptive μ (α=0.1)')
        ax.axhline(y=MU_FIXED, color=COLORS['fedprox'], linewidth=2,
                   linestyle='--', label=f'Fixed μ = {MU_FIXED}', alpha=0.8)
        ax.axhline(y=MU_MIN, color='gray', linewidth=1.5,
                   linestyle=':', label=f'μ_min = {MU_MIN}', alpha=0.6)
        ax.axhline(y=MU_MAX, color='orange', linewidth=1.5,
                   linestyle=':', label=f'μ_max = {MU_MAX}', alpha=0.6)

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('μ value', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, ROUNDS+1, 5))

    plt.tight_layout()
    plt.savefig('exp4_adaptive_mu_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Saved: exp4_adaptive_mu_evolution.png")


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading WSN-DS dataset...")

    df     = load_dataset("wsn")
    target = auto_detect_target(df)
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, target_column=target)

    print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Cấu hình: ROUNDS={ROUNDS}, LOCAL_EPOCHS={LOCAL_EPOCHS}, "
          f"K_CLIENTS={K_CLIENTS}, MU_FIXED={MU_FIXED}")
    print(f"Adaptive-μ: MU_MIN={MU_MIN}, MU_MAX={MU_MAX}")

    results, histories, mu_histories, alpha_values, cent_h, cent_m = run_adaptive_experiment(
        X_train, y_train, X_test, y_test, device)

    plot_results(results, histories, mu_histories, alpha_values, cent_h, cent_m)

    # Lưu kết quả JSON
    def ser(m):
        return {k: float(v) for k, v in m.items() if k != 'confusion_matrix'}

    all_results = {
        'centralized': ser(cent_m)
    }
    for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
        all_results[mode] = {str(a): ser(results[mode][str(a)]) for a in alpha_values}

    with open('adaptive_fedprox_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("  ✅ THỰC NGHIỆM ADAPTIVE-μ HOÀN THÀNH!")
    print("  📊 Files đã tạo:")
    print("     1. exp4_adaptive_fedprox_comparison.png")
    print("     2. exp4_adaptive_convergence_curves.png")
    print("     3. exp4_adaptive_mu_evolution.png")
    print("     4. adaptive_fedprox_results.json")
    print("="*60)


if __name__ == '__main__':
    main()
