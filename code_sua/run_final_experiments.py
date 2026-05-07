"""
run_final_experiments.py — Thực nghiệm hoàn chỉnh cho báo cáo
================================================================
Chạy tất cả thực nghiệm cần thiết:
  1. IID: Centralized + FedAvg + FedProx + Adaptive-μ (+ Confusion Matrix)
  2. Non-IID α=0.1: Centralized + FedAvg + FedProx + Adaptive-μ (+ Confusion Matrix)
  3. Non-IID 3 runs (mean ± std) cho bảng thống kê

Output:
  - results_final_iid.json
  - results_final_noniid.json
  - results_stability_3runs.json
  - cm_adaptive_iid.png        (Confusion Matrix IID)
  - cm_adaptive_noniid.png     (Confusion Matrix Non-IID)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy, warnings, json, time
warnings.filterwarnings('ignore')

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import LSTM_IDS, clone_model_with_weights_pytorch, set_model_weights_pytorch
from federated_learning_pytorch import weighted_fedavg_pytorch
from evaluation_pytorch import evaluate_model_pytorch
from run_non_iid import create_non_iid_dirichlet

# ============================================================
MAX_SAMPLES  = 80000
BATCH_SIZE   = 2048
ROUNDS       = 100
CENT_EPOCHS  = 50
LOCAL_EPOCHS = 1
K_CLIENTS    = 10
SEED         = 42
ALPHA        = 0.1

MU_MIN   = 0.00001
MU_MAX   = 0.05
MU_FIXED = 0.001

CLASS_NAMES = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'Scheduling']  # Thứ tự alphabetical từ LabelEncoder

# ============================================================
# HELPER FUNCTIONS
# ============================================================

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
    print(f"\n--- Training CENTRALIZED ({epochs} epochs) ---")
    model = LSTM_IDS(X_train.shape[2], y_train.shape[1]).to(device)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            m = evaluate_model_pytorch(model, X_test, y_test, device)
            print(f"  Epoch {epoch+1}/{epochs} - Acc: {m['accuracy']*100:.2f}%")

    return evaluate_model_pytorch(model, X_test, y_test, device)

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
                prox = sum(torch.norm(p - g.to(device))**2 for p, g in zip(model.parameters(), global_weights))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()

    return [p.data.clone().cpu() for p in model.parameters()], len(X_k)

def run_fl(client_data, X_test, y_test, mode, device, rounds=ROUNDS, verbose=True):
    """Chay 1 FL experiment, tra ve final metrics (bao gom confusion_matrix)."""
    input_size = X_test.shape[2]
    num_classes = y_test.shape[1]
    global_model = LSTM_IDS(input_size, num_classes).to(device)

    for t in range(rounds):
        gw = [p.data.clone().cpu() for p in global_model.parameters()]
        client_weights_list = []
        client_sizes = []
        local_weight_tensors = []

        for k in range(len(client_data)):
            X_k, y_k = client_data[k]
            if len(X_k) < 10:
                continue
            local_model = clone_model_with_weights_pytorch(global_model)
            if mode == 'fedavg':
                mu_t = 0.0
            elif mode == 'fedprox_fixed':
                mu_t = MU_FIXED
            else:
                mu_t = MU_FIXED  # will be overwritten if adaptive

            w_k, n_k = train_client_local(local_model, X_k, y_k, mu=mu_t,
                                          global_weights=gw if mu_t > 0 else None, device=device)
            client_weights_list.append(w_k)
            client_sizes.append(n_k)
            local_weight_tensors.append([torch.tensor(w) for w in w_k])

        # Adaptive second pass
        if mode == 'fedprox_adaptive' and len(local_weight_tensors) > 0:
            gw_t = [torch.tensor(w) for w in gw]
            div = compute_divergence(local_weight_tensors, gw_t)
            mu_t = adaptive_mu(div)
            if t > 0:
                client_weights_list = []
                client_sizes = []
                for k in range(len(client_data)):
                    X_k, y_k = client_data[k]
                    if len(X_k) < 10:
                        continue
                    local_model = clone_model_with_weights_pytorch(global_model)
                    w_k, n_k = train_client_local(local_model, X_k, y_k, mu=mu_t, global_weights=gw, device=device)
                    client_weights_list.append(w_k)
                    client_sizes.append(n_k)

        set_model_weights_pytorch(global_model, weighted_fedavg_pytorch(client_weights_list, client_sizes))

        if verbose and ((t+1) % 20 == 0 or t == 0):
            m = evaluate_model_pytorch(global_model, X_test, y_test, device)
            print(f"  [{mode}] Round {t+1}/{rounds} - Acc: {m['accuracy']*100:.2f}%")

    return evaluate_model_pytorch(global_model, X_test, y_test, device)

def plot_confusion_matrix(cm, title, filename):
    """Ve va luu Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {filename}")

def ser(m):
    return {k: float(v) for k, v in m.items() if k != 'confusion_matrix'}

# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading WSN-DS...")
    df = load_dataset("wsn")
    target = auto_detect_target(df)
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, target_column=target)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ============================
    # PHAN 1: IID (du lieu chia deu)
    # ============================
    print("\n" + "="*60)
    print("PHAN 1: KICH BAN IID (du lieu dong nhat)")
    print("="*60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Chia IID: shuffle + split deu
    n = len(X_train)
    indices = np.random.permutation(n)
    client_data_iid = []
    chunk = n // K_CLIENTS
    for k in range(K_CLIENTS):
        idx = indices[k*chunk : (k+1)*chunk]
        client_data_iid.append((X_train[idx], y_train[idx]))

    # Centralized
    cent_m = train_centralized(X_train, y_train, X_test, y_test, device)

    # FedAvg, FedProx, Adaptive
    iid_results = {'centralized': ser(cent_m)}
    for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
        np.random.seed(SEED); torch.manual_seed(SEED)
        m = run_fl(client_data_iid, X_test, y_test, mode, device)
        iid_results[mode] = ser(m)
        print(f"  {mode}: Acc={m['accuracy']*100:.2f}%, F1={m['f1_score']*100:.2f}%, FPR={m['FPR']*100:.2f}%")

        # Confusion Matrix cho Adaptive IID
        if mode == 'fedprox_adaptive':
            plot_confusion_matrix(m['confusion_matrix'],
                                  'Confusion Matrix - FedProx Adaptive-mu (IID)',
                                  'cm_adaptive_iid.png')

    with open('results_final_iid.json', 'w') as f:
        json.dump(iid_results, f, indent=2)
    print("=> Saved: results_final_iid.json")

    # ============================
    # PHAN 2: NON-IID (alpha=0.1)
    # ============================
    print("\n" + "="*60)
    print(f"PHAN 2: KICH BAN NON-IID (alpha={ALPHA})")
    print("="*60)

    np.random.seed(SEED); torch.manual_seed(SEED)
    client_data_noniid = create_non_iid_dirichlet(X_train, y_train, K=K_CLIENTS, alpha=ALPHA)

    noniid_results = {'centralized': ser(cent_m)}
    for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
        np.random.seed(SEED); torch.manual_seed(SEED)
        m = run_fl(client_data_noniid, X_test, y_test, mode, device)
        noniid_results[mode] = ser(m)
        print(f"  {mode}: Acc={m['accuracy']*100:.2f}%, F1={m['f1_score']*100:.2f}%, FPR={m['FPR']*100:.2f}%")

        # Confusion Matrix cho Adaptive Non-IID
        if mode == 'fedprox_adaptive':
            plot_confusion_matrix(m['confusion_matrix'],
                                  f'Confusion Matrix - FedProx Adaptive-mu (Non-IID alpha={ALPHA})',
                                  'cm_adaptive_noniid.png')

    with open('results_final_noniid.json', 'w') as f:
        json.dump(noniid_results, f, indent=2)
    print("=> Saved: results_final_noniid.json")

    # ============================
    # PHAN 3: STABILITY TEST (3 runs Non-IID, seeds khac nhau)
    # ============================
    print("\n" + "="*60)
    print("PHAN 3: STABILITY TEST (3 runs, seeds khac nhau)")
    print("="*60)

    seeds = [42, 123, 2025]
    stability = {mode: {'accuracy': [], 'f1_score': []} for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']}

    for run_idx, seed in enumerate(seeds):
        print(f"\n--- Run {run_idx+1}/3 (seed={seed}) ---")
        np.random.seed(seed); torch.manual_seed(seed)
        df_run = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=seed)
        X_tr, X_te, y_tr, y_te, _, _ = preprocess_data(df_run, target_column=target)
        np.random.seed(seed); torch.manual_seed(seed)
        cd = create_non_iid_dirichlet(X_tr, y_tr, K=K_CLIENTS, alpha=ALPHA)

        for mode in ['fedavg', 'fedprox_fixed', 'fedprox_adaptive']:
            np.random.seed(seed); torch.manual_seed(seed)
            m = run_fl(cd, X_te, y_te, mode, device, verbose=False)
            stability[mode]['accuracy'].append(m['accuracy'])
            stability[mode]['f1_score'].append(m['f1_score'])
            print(f"  {mode}: Acc={m['accuracy']*100:.2f}%, F1={m['f1_score']*100:.2f}%")

    # Tinh mean +- std
    stability_summary = {}
    for mode in stability:
        accs = np.array(stability[mode]['accuracy']) * 100
        f1s  = np.array(stability[mode]['f1_score']) * 100
        stability_summary[mode] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std':  float(np.std(accs)),
            'f1_mean':       float(np.mean(f1s)),
            'f1_std':        float(np.std(f1s)),
            'raw_accuracy':  [float(x) for x in accs],
            'raw_f1':        [float(x) for x in f1s],
        }
        print(f"\n  {mode}: Acc = {np.mean(accs):.2f} +/- {np.std(accs):.2f}%, "
              f"F1 = {np.mean(f1s):.2f} +/- {np.std(f1s):.2f}%")

    with open('results_stability_3runs.json', 'w') as f:
        json.dump(stability_summary, f, indent=2)
    print("=> Saved: results_stability_3runs.json")

    print("\n" + "="*60)
    print("HOAN THANH TAT CA THUC NGHIEM!")
    print("="*60)

if __name__ == '__main__':
    main()
