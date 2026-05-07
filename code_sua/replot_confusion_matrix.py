"""
replot_confusion_matrix.py — Vẽ lại Confusion Matrix với CLASS_NAMES đúng
(Không cần train lại, chỉ load model và chạy evaluate)
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import LSTM_IDS
from evaluation_pytorch import evaluate_model_pytorch
from run_non_iid import create_non_iid_dirichlet

# Thứ tự ĐÚNG theo LabelEncoder alphabetical
# Từ log: ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA'] → [0, 1, 2, 3, 4]
CLASS_NAMES = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'Scheduling']

MAX_SAMPLES = 80000
K_CLIENTS = 10
SEED = 42
ALPHA = 0.1
MU_MIN = 0.00001
MU_MAX = 0.05
MU_FIXED = 0.001
ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 2048

# Verify từ LabelEncoder
print("Kiểm tra thứ tự nhãn từ LabelEncoder:")
from sklearn.preprocessing import LabelEncoder
import pandas as pd
le = LabelEncoder()
# WSN-DS labels
wsn_labels = ['Normal', 'Blackhole', 'Flooding', 'Grayhole', 'TDMA']
le.fit(wsn_labels)
print(f"  Thứ tự LabelEncoder: {list(le.classes_)}")
print(f"  → 0={le.classes_[0]}, 1={le.classes_[1]}, 2={le.classes_[2]}, 3={le.classes_[3]}, 4={le.classes_[4]}")
print(f"  CLASS_NAMES dùng: {CLASS_NAMES}")

def plot_confusion_matrix(cm, title, filename):
    """Ve va luu Confusion Matrix voi nhan dung."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Tính tỷ lệ % theo từng hàng (True class)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Vẽ heatmap với số đếm thực tế
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Count'})

    # Thêm % vào annotation
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j + 0.5, i + 0.75,
                    f'({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center',
                    fontsize=7, color='gray')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
    ax.set_ylabel('True Label', fontsize=11, labelpad=10)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  => Saved: {filename}")

# ============================================================
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from model_pytorch import clone_model_with_weights_pytorch, set_model_weights_pytorch
from federated_learning_pytorch import weighted_fedavg_pytorch

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

def compute_divergence(local_weights_list, global_weights):
    divergences = []
    for w_k in local_weights_list:
        diff = sum(torch.norm(p_k.float() - p_g.float()).item()**2 for p_k, p_g in zip(w_k, global_weights))
        divergences.append(diff)
    return float(np.mean(divergences))

def adaptive_mu(divergence):
    sig = 1.0 / (1.0 + np.exp(-0.1 * (divergence - 50)))
    return MU_MIN + (MU_MAX - MU_MIN) * sig

def run_fl_adaptive(client_data, X_test, y_test, device):
    """Chạy FedProx Adaptive và trả về metrics + confusion matrix."""
    input_size = X_test.shape[2]
    num_classes = y_test.shape[1]
    global_model = LSTM_IDS(input_size, num_classes).to(device)

    for t in range(ROUNDS):
        gw = [p.data.clone().cpu() for p in global_model.parameters()]
        client_weights_list, client_sizes, local_weight_tensors = [], [], []

        for k in range(len(client_data)):
            X_k, y_k = client_data[k]
            if len(X_k) < 10:
                continue
            local_model = clone_model_with_weights_pytorch(global_model)
            w_k, n_k = train_client_local(local_model, X_k, y_k, mu=MU_FIXED, global_weights=gw, device=device)
            client_weights_list.append(w_k)
            client_sizes.append(n_k)
            local_weight_tensors.append([torch.tensor(w) for w in w_k])

        # Adaptive second pass
        if len(local_weight_tensors) > 0:
            gw_t = [torch.tensor(w) for w in gw]
            div = compute_divergence(local_weight_tensors, gw_t)
            mu_t = adaptive_mu(div)
            if t > 0:
                client_weights_list, client_sizes = [], []
                for k in range(len(client_data)):
                    X_k, y_k = client_data[k]
                    if len(X_k) < 10:
                        continue
                    local_model = clone_model_with_weights_pytorch(global_model)
                    w_k, n_k = train_client_local(local_model, X_k, y_k, mu=mu_t, global_weights=gw, device=device)
                    client_weights_list.append(w_k)
                    client_sizes.append(n_k)

        set_model_weights_pytorch(global_model, weighted_fedavg_pytorch(client_weights_list, client_sizes))

        if (t+1) % 25 == 0 or t == 0:
            m = evaluate_model_pytorch(global_model, X_test, y_test, device)
            print(f"  Round {t+1}/{ROUNDS} - Acc: {m['accuracy']*100:.2f}%")

    return evaluate_model_pytorch(global_model, X_test, y_test, device)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = load_dataset("wsn")
    target = auto_detect_target(df)
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=SEED)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, target_column=target)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Kiểm tra phân phối test set
    y_test_labels = np.argmax(y_test, axis=1)
    unique, counts = np.unique(y_test_labels, return_counts=True)
    print("\nPhân phối TEST SET (sau SMOTE không áp dụng):")
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({CLASS_NAMES[u]}): {c} mẫu ({c/len(y_test_labels)*100:.1f}%)")

    # IID
    print("\n=== IID ===")
    n = len(X_train)
    np.random.seed(SEED); torch.manual_seed(SEED)
    indices = np.random.permutation(n)
    chunk = n // K_CLIENTS
    client_data_iid = [(X_train[indices[k*chunk:(k+1)*chunk]], y_train[indices[k*chunk:(k+1)*chunk]]) for k in range(K_CLIENTS)]

    np.random.seed(SEED); torch.manual_seed(SEED)
    m_iid = run_fl_adaptive(client_data_iid, X_test, y_test, device)
    print(f"IID: Acc={m_iid['accuracy']*100:.2f}%, F1={m_iid['f1_score']*100:.2f}%")
    plot_confusion_matrix(m_iid['confusion_matrix'],
                          'Confusion Matrix — FedProx Adaptive-μ (IID)',
                          'cm_adaptive_iid.png')

    # Non-IID
    print("\n=== NON-IID ===")
    np.random.seed(SEED); torch.manual_seed(SEED)
    client_data_noniid = create_non_iid_dirichlet(X_train, y_train, K=K_CLIENTS, alpha=ALPHA)

    np.random.seed(SEED); torch.manual_seed(SEED)
    m_noniid = run_fl_adaptive(client_data_noniid, X_test, y_test, device)
    print(f"Non-IID: Acc={m_noniid['accuracy']*100:.2f}%, F1={m_noniid['f1_score']*100:.2f}%")
    plot_confusion_matrix(m_noniid['confusion_matrix'],
                          f'Confusion Matrix — FedProx Adaptive-μ (Non-IID α=0.1)',
                          'cm_adaptive_noniid.png')

    print("\nDONE!")

if __name__ == '__main__':
    main()
