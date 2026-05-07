"""
run_fedprox_only.py — Chỉ chạy FedProx (Centralized & FedAvg đã xong)
Sử dụng mu=0.0001 (giảm 100 lần so với lần trước)
"""
import torch
import torch.nn as nn
import numpy as np
import time
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

def evaluate_final(model, X_test, y_test, num_classes, device):
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
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr_per_class = fp / (fp + tn)
    mean_fpr = np.nanmean(fpr_per_class)
    
    return acc, f1, mean_fpr

def main():
    print("=" * 60)
    print(" CHỈ CHẠY FEDPROX (mu=0.0001)")
    print(" Centralized & FedAvg đã hoàn thành trước đó.")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. TIỀN XỬ LÝ DỮ LIỆU (giống hệt lần trước)
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Chia dữ liệu cho 10 clients
    K = 10
    client_data = []
    n_per_client = len(X_train) // K
    for i in range(K):
        start = i * n_per_client
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
    
    # Tạo hàm đánh giá hội tụ
    eval_fn = create_eval_fn(X_test, y_test, num_classes, device)
    
    # 2. CHẠY FEDPROX
    print("\n" + "=" * 40)
    print(" FEDPROX (mu=0.0001) — 100 Rounds")
    print("=" * 40)
    
    fedprox_model = create_lstm_model_pytorch(input_shape, num_classes)
    fedprox_model = fedprox_model.to(device)
    
    start_time = time.time()
    fedprox_model, hist_fedprox = run_federated_training_pytorch(
        fedprox_model, client_data,
        num_rounds=100, local_epochs=5, device=device,
        algo='fedprox', mu=0.0001,   # <<< ĐÃ GIẢM TỪ 0.01 XUỐNG 0.0001
        eval_fn=eval_fn
    )
    train_time = time.time() - start_time
    
    # 3. ĐÁNH GIÁ CUỐI CÙNG
    acc, f1, mean_fpr = evaluate_final(fedprox_model, X_test, y_test, num_classes, device)
    
    print(f"\n{'=' * 50}")
    print(f" KẾT QUẢ FEDPROX (mu=0.0001) TRÊN WSN-DS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  Mean FPR:  {mean_fpr * 100:.2f}%")
    print(f"  Time:      {train_time:.2f} seconds ({train_time/60:.1f} minutes)")
    print(f"{'=' * 50}")
    
    # 4. LƯU MÔ HÌNH VÀ LỊCH SỬ HỘI TỤ
    torch.save(fedprox_model.state_dict(), 'fedprox_wsn.pt')
    print("✅ Đã lưu mô hình: fedprox_wsn.pt")
    
    np.savez('fedprox_convergence.npz', 
             fedprox_acc=hist_fedprox['acc'],
             mu=0.0001)
    print("✅ Đã lưu lịch sử hội tụ: fedprox_convergence.npz")
    
    # 5. LƯU KẾT QUẢ RA FILE TXT (PHÒNG MẤT TERMINAL)
    with open('fedprox_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"FedProx Results (mu=0.0001) on WSN-DS\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Accuracy:  {acc * 100:.2f}%\n")
        f.write(f"F1-Score:  {f1 * 100:.2f}%\n")
        f.write(f"Mean FPR:  {mean_fpr * 100:.2f}%\n")
        f.write(f"Time:      {train_time:.2f}s ({train_time/60:.1f} min)\n")
        f.write(f"\nConvergence (Acc per round):\n")
        for i, a in enumerate(hist_fedprox['acc']):
            f.write(f"  Round {i+1}: {a*100:.2f}%\n")
    print("✅ Đã lưu kết quả ra file: fedprox_results.txt")
    
    print("\n🎉 HOÀN TẤT! Bạn có thể tắt terminal và đi ngủ yên tâm.")

if __name__ == "__main__":
    main()
