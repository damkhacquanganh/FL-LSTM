import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, json
warnings.filterwarnings('ignore')

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import LSTM_IDS, clone_model_with_weights_pytorch, set_model_weights_pytorch
from federated_learning_pytorch import weighted_fedavg_pytorch
from evaluation_pytorch import evaluate_model_pytorch
from run_non_iid import create_non_iid_dirichlet

# ============================================================
# CẤU HÌNH CHO 100 ROUNDS - ALPHA 0.1 - FULL DATA
# ============================================================
MAX_SAMPLES  = 80000      # Giới hạn 80k mẫu để mô phỏng môi trường IoT thực tế và tăng tốc độ
BATCH_SIZE   = 2048
ROUNDS       = 100       # 100 rounds cho FL
CENT_EPOCHS  = 50        # 50 epochs cho Centralized
LOCAL_EPOCHS = 1
K_CLIENTS    = 10
SEED         = 42
ALPHA        = 0.1       # Fix alpha = 0.1

# Adaptive-μ hyperparameters
MU_MIN   = 0.00001
MU_MAX   = 0.05
MU_FIXED = 0.001

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
    'centralized':'#5E35B1',  # Tím
    'fedavg':    '#D32F2F',   # Đỏ
    'fedprox':   '#1565C0',   # Xanh dương
    'adaptive':  '#2E7D32',   # Xanh lá
}

def compute_divergence(local_weights_list, global_weights):
    divergences = []
    for w_k in local_weights_list:
        diff = 0.0
        for p_k, p_g in zip(w_k, global_weights):
            diff += torch.norm(p_k.float() - p_g.float()).item() ** 2
        divergences.append(diff)
    return float(np.mean(divergences))

def adaptive_mu(divergence, mu_min=MU_MIN, mu_max=MU_MAX, scale=0.1):
    sig = 1.0 / (1.0 + np.exp(-scale * (divergence - 50)))
    return mu_min + (mu_max - mu_min) * sig

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

def train_client_local(model, X_k, y_k, mu, global_weights, device):
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

def train_federated_mode(client_data, X_train, y_train, X_test, y_test, alpha, mode, device):
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
    mu_history  = []

    for t in range(ROUNDS):
        client_weights_list = []
        client_sizes        = []
        local_weight_tensors = []

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
            else:
                mu_t = MU_FIXED

            w_k, n_k = train_client_local(local_model, X_k, y_k, mu=mu_t, global_weights=gw if mu_t > 0 else None, device=device)
            client_weights_list.append(w_k)
            client_sizes.append(n_k)
            local_weight_tensors.append([torch.tensor(w) for w in w_k])

        if mode == 'fedprox_adaptive' and len(local_weight_tensors) > 0:
            gw_t = [torch.tensor(w) for w in gw]
            div  = compute_divergence(local_weight_tensors, gw_t)
            mu_t = adaptive_mu(div)
            mu_history.append(mu_t)

            if t > 0:
                client_weights_list = []
                client_sizes = []
                for k in range(K_CLIENTS):
                    X_k, y_k = client_data[k]
                    if len(X_k) < 10:
                        continue
                    local_model = clone_model_with_weights_pytorch(global_model)
                    w_k, n_k = train_client_local(local_model, X_k, y_k, mu=mu_t, global_weights=gw, device=device)
                    client_weights_list.append(w_k)
                    client_sizes.append(n_k)
            if (t+1) % 5 == 0:
                print(f"    [Round {t+1}] divergence={div:.2f} → μ_adaptive={mu_t:.6f}")
        else:
            mu_history.append(MU_FIXED if mode == 'fedprox_fixed' else 0.0)

        set_model_weights_pytorch(global_model, weighted_fedavg_pytorch(client_weights_list, client_sizes))

        metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
        acc_history.append(metrics['accuracy'])
        if (t+1) % 5 == 0 or t == 0:
            print(f"  Round {t+1}/{ROUNDS} - Acc: {metrics['accuracy']*100:.2f}%")

    final_metrics = evaluate_model_pytorch(global_model, X_test, y_test, device)
    return acc_history, mu_history, final_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading WSN-DS FULL dataset...")
    df = load_dataset("wsn")
    target = auto_detect_target(df)
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)
    else:
        print(f"Using full dataset: {len(df)} samples")

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, target_column=target)
    print(f"\nX_train: {X_train.shape}, X_test: {X_test.shape}")

    # 1. Chạy Centralized
    cent_h, cent_m = train_centralized(X_train, y_train, X_test, y_test, device, epochs=CENT_EPOCHS)
    
    # 2. Tạo dữ liệu Non-IID CHUNG 1 LẦN cho cả 3 phương pháp
    print(f"\n[!] TẠO DỮ LIỆU NON-IID CHUNG CHO CẢ 3 PHƯƠNG PHÁP (alpha={ALPHA})...")
    
    # Ép cứng SEED để luôn ra cùng một kịch bản chia cắt (loại bỏ yếu tố may rủi)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    client_data = create_non_iid_dirichlet(X_train, y_train, K=K_CLIENTS, alpha=ALPHA)

    # LƯU VẾT PHÂN PHỐI DỮ LIỆU ĐỂ LÀM BẰNG CHỨNG
    with open("data_distribution_log.txt", "w", encoding="utf-8") as f:
        f.write(f"--- BẰNG CHỨNG PHÂN PHỐI DỮ LIỆU NON-IID (alpha={ALPHA}, Seed={SEED}) ---\n")
        f.write("Tập dữ liệu này được DÙNG CHUNG CỐ ĐỊNH cho cả FedAvg, FedProx và Adaptive.\n\n")
        for k in range(K_CLIENTS):
            X_k, y_k = client_data[k]
            if len(X_k) == 0:
                continue
            y_k_labels = np.argmax(y_k, axis=1)
            unique, counts = np.unique(y_k_labels, return_counts=True)
            dist = dict(zip(unique, counts))
            dist_str = ", ".join([f"Nhãn {lbl}: {cnt}" for lbl, cnt in dist.items()])
            log_line = f"Client {k+1:2d} | Tổng: {len(X_k):7d} mẫu | Phân phối: {{ {dist_str} }}"
            print(log_line)
            f.write(log_line + "\n")
            
    print("=> Đã lưu vết phân phối vào file 'data_distribution_log.txt'")

    # 3. Chạy Federated methods (FedAvg, FedProx fixed, FedProx adaptive)
    histories = {}
    mu_histories = {}
    metrics = {'centralized': cent_m}
    
    for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
        h, mu_h, m = train_federated_mode(client_data, X_train, y_train, X_test, y_test, ALPHA, mode, device)
        histories[mode] = h
        mu_histories[mode] = mu_h
        metrics[mode] = m

    # VẼ BIỂU ĐỒ HỘI TỤ (Convergence Curve 100 Rounds)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Convergence Comparison (100 Rounds) - Full Data\n'
                 f'Extreme Non-IID (α={ALPHA})', fontsize=14, fontweight='bold')

    # Centralized
    cent_acc = cent_m['accuracy'] * 100
    ax.axhline(y=cent_acc, color=COLORS['centralized'], linewidth=2.5, linestyle='--', 
               label=f'Centralized ({cent_acc:.1f}%)', alpha=0.8)
    
    # FL lines
    rounds_x = list(range(1, ROUNDS + 1))
    ax.plot(rounds_x, [v*100 for v in histories['fedavg']], '-',
            color=COLORS['fedavg'], linewidth=2, label=f"FedAvg ({metrics['fedavg']['accuracy']*100:.1f}%)", alpha=0.9)
    ax.plot(rounds_x, [v*100 for v in histories['fedprox_fixed']], '--',
            color=COLORS['fedprox'], linewidth=2.5, label=f"FedProx Fixed ({metrics['fedprox_fixed']['accuracy']*100:.1f}%)", alpha=0.9)
    ax.plot(rounds_x, [v*100 for v in histories['fedprox_adaptive']], '-',
            color=COLORS['adaptive'], linewidth=3, label=f"FedProx Adaptive ({metrics['fedprox_adaptive']['accuracy']*100:.1f}%)", alpha=1.0)

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([0, ROUNDS])
    
    # Highlight final 20 rounds
    ax.axvspan(ROUNDS-20, ROUNDS, alpha=0.05, color='green')

    plt.tight_layout()
    plt.savefig('exp5_100_rounds_alpha01.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("=> Saved: exp5_100_rounds_alpha01.png")

    # Lưu kết quả
    def ser(m):
        return {k: float(v) for k, v in m.items() if k != 'confusion_matrix'}
    
    out_json = {
        'metadata': {'alpha': ALPHA, 'rounds': ROUNDS, 'samples': len(df)},
        'centralized': ser(metrics['centralized']),
        'fedavg': ser(metrics['fedavg']),
        'fedprox_fixed': ser(metrics['fedprox_fixed']),
        'fedprox_adaptive': ser(metrics['fedprox_adaptive'])
    }
    with open('results_100_rounds.json', 'w') as f:
        json.dump(out_json, f, indent=2)

    print("\n✅ HOÀN THÀNH!")

if __name__ == '__main__':
    main()
