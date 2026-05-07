import torch
import torch.nn as nn
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import create_lstm_model_pytorch
from federated_learning_pytorch import run_federated_training_pytorch

def create_eval_fn(X_test, y_test, num_classes, device):
    def eval_fn(model):
        was_training = model.training
        model.eval()
        model = model.to(device)
        
        if num_classes > 2:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
            
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
        
        y_preds = []
        with torch.no_grad():
            for batch_X in dataloader:
                outputs = model(batch_X[0].to(device))
                if num_classes > 2:
                    _, batch_pred = torch.max(outputs, 1)
                else:
                    batch_pred = (torch.sigmoid(outputs) > 0.5).float()
                y_preds.append(batch_pred.cpu().numpy())
                
        y_pred_all = np.concatenate(y_preds)
        if was_training:
            model.train()
        return accuracy_score(y_true, y_pred_all)
    return eval_fn

# ============================================================
# PHAN PHOI DIRICHLET - CHUAN QUOC TE (NeurIPS, ICML, MLSys)
# ============================================================
def create_non_iid_dirichlet(X, y, K=10, alpha=0.5):
    """
    Chia du lieu Non-IID bang phan phoi Dirichlet.
    alpha nho => Non-IID manh (0.1 = cuc doan, 0.5 = vua phai, 1.0 = nhe)
    alpha lon  => Gan IID
    """
    print(f"\n[!] DANG TAO DU LIEU NON-IID BANG DIRICHLET (alpha={alpha})...")
    
    y_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y
    num_classes = len(np.unique(y_labels))
    
    # Khoi tao danh sach index cho moi client
    client_indices = [[] for _ in range(K)]
    
    for c in range(num_classes):
        # Lay tat ca index cua class c
        idx_c = np.where(y_labels == c)[0]
        np.random.shuffle(idx_c)
        
        # Sinh ty le phan chia theo Dirichlet
        proportions = np.random.dirichlet([alpha] * K)
        
        # Chuyen ty le thanh so luong thuc te
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        
        for k in range(K):
            client_indices[k].extend(splits[k].tolist())
    
    # Tao client_data
    client_data = []
    for k in range(K):
        idx = np.array(client_indices[k])
        np.random.shuffle(idx)
        client_data.append((X[idx], y[idx]))
        
        # Thong ke de in ra man hinh
        y_k = np.argmax(y[idx], axis=1) if len(y.shape) > 1 else y[idx]
        unique, counts = np.unique(y_k, return_counts=True)
        dist = dict(zip(unique.astype(int), counts))
        total = len(idx)
        print(f"  -> Client {k+1}: {total} mau | Phan phoi: {dist}")
        
    return client_data

def evaluate_final(model, X_test, y_test, num_classes, device, model_name):
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    y_preds = []
    with torch.no_grad():
        for batch_X in dataloader:
            outputs = model(batch_X[0].to(device))
            if num_classes > 2:
                _, batch_pred = torch.max(outputs, 1)
            else:
                batch_pred = (torch.sigmoid(outputs) > 0.5).float()
            y_preds.append(batch_pred.cpu().numpy())
    
    y_pred_cpu = np.concatenate(y_preds)
    
    if num_classes > 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    acc = accuracy_score(y_true, y_pred_cpu)
    f1 = f1_score(y_true, y_pred_cpu, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred_cpu)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm)) - np.diag(cm) + np.diag(cm)
    # Tinh FPR dung
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr_per_class = fp / (fp + tn + 1e-10)
    mean_fpr = np.nanmean(fpr_per_class)
    
    print(f"\n{'=' * 50}")
    print(f" KET QUA {model_name} (NON-IID)")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  Mean FPR:  {mean_fpr * 100:.2f}%")
    print(f"{'=' * 50}")
    
    return acc, f1, mean_fpr

def main():
    print("=" * 60)
    print(" THI NGHIEM NON-IID (CHUAN QUOC TE - DIRICHLET)")
    print(" Phuong phap: Dirichlet alpha=0.5 | KHONG dung SMOTE")
    print("=" * 60)
    
    # Co dinh random seed de ket qua co the tai tao (reproducible)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # KHONG dung SMOTE - giu phan phoi tu nhien (dung chuan hoc thuat)
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=False
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Chia du lieu Non-IID bang Dirichlet (alpha=0.1 = CUC DOAN / STRESS TEST)
    client_data = create_non_iid_dirichlet(X_train, y_train, K=10, alpha=0.1)
    eval_fn = create_eval_fn(X_test, y_test, num_classes, device)
    
    # =============================================
    # 1. CHAY FEDAVG NON-IID (100 rounds)
    # =============================================
    print("\n" + "="*50)
    print(" CHAY FEDAVG (STRESS TEST - alpha=0.1)")
    print("="*50)
    t1 = time.time()
    fedavg_model = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    fedavg_model, hist_fedavg = run_federated_training_pytorch(
        fedavg_model, client_data, num_rounds=100, local_epochs=5, 
        device=device, algo='fedavg', eval_fn=eval_fn
    )
    t_fedavg = time.time() - t1
    
    acc_avg, f1_avg, fpr_avg = evaluate_final(
        fedavg_model, X_test, y_test, num_classes, device, "FEDAVG"
    )
    
    # =============================================
    # 2. CHAY FEDPROX NON-IID (mu=0.01, 100 rounds)
    # =============================================
    print("\n" + "="*50)
    print(" CHAY FEDPROX (STRESS TEST - alpha=0.1) - mu=0.01")
    print("="*50)
    t2 = time.time()
    fedprox_model = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    fedprox_model, hist_fedprox = run_federated_training_pytorch(
        fedprox_model, client_data, num_rounds=100, local_epochs=5, 
        device=device, algo='fedprox', mu=0.01, eval_fn=eval_fn
    )
    t_fedprox = time.time() - t2
    
    acc_prox, f1_prox, fpr_prox = evaluate_final(
        fedprox_model, X_test, y_test, num_classes, device, "FEDPROX"
    )
    
    # =============================================
    # 3. LUU KET QUA
    # =============================================
    np.savez('non_iid_convergence.npz', 
             fedavg_acc=hist_fedavg['acc'], 
             fedprox_acc=hist_fedprox['acc'])
    
    # Luu ket qua chi tiet ra file txt
    with open('non_iid_results.txt', 'w', encoding='utf-8') as f:
        f.write("Non-IID Results (Stress Test alpha=0.1)\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nFedAvg:\n")
        f.write(f"  Accuracy:  {acc_avg * 100:.2f}%\n")
        f.write(f"  F1-Score:  {f1_avg * 100:.2f}%\n")
        f.write(f"  Mean FPR:  {fpr_avg * 100:.2f}%\n")
        f.write(f"  Time:      {t_fedavg:.1f}s ({t_fedavg/60:.1f} min)\n")
        f.write(f"\nFedProx (mu=0.01):\n")
        f.write(f"  Accuracy:  {acc_prox * 100:.2f}%\n")
        f.write(f"  F1-Score:  {f1_prox * 100:.2f}%\n")
        f.write(f"  Mean FPR:  {fpr_prox * 100:.2f}%\n")
        f.write(f"  Time:      {t_fedprox:.1f}s ({t_fedprox/60:.1f} min)\n")
        
        f.write(f"\nConvergence FedAvg:\n")
        for i, a in enumerate(hist_fedavg['acc']):
            f.write(f"  Round {i+1}: {a*100:.2f}%\n")
        f.write(f"\nConvergence FedProx:\n")
        for i, a in enumerate(hist_fedprox['acc']):
            f.write(f"  Round {i+1}: {a*100:.2f}%\n")
    
    print(f"\n{'=' * 60}")
    print(f" TONG KET NON-IID (Dirichlet alpha=0.5)")
    print(f"{'=' * 60}")
    print(f"  FedAvg  -> Acc: {acc_avg*100:.2f}% | F1: {f1_avg*100:.2f}%")
    print(f"  FedProx -> Acc: {acc_prox*100:.2f}% | F1: {f1_prox*100:.2f}%")
    print(f"{'=' * 60}")
    print("Da luu: non_iid_convergence.npz + non_iid_results.txt")
    print("HOAN TAT!")

if __name__ == "__main__":
    main()
