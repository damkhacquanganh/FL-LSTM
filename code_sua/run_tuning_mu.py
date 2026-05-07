import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

def plot_hyperparameter_tuning(acc_mu_bad, acc_mu_good):
    print("\nĐang vẽ biểu đồ Hyperparameter Tuning...")
    # Lấy 25 vòng đầu tiên
    rounds = range(1, 26)
    acc_bad_pct = [a * 100 for a in acc_mu_bad[:25]]
    acc_good_pct = [a * 100 for a in acc_mu_good[:25]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, acc_good_pct, 'b-o', label='Tối ưu: FedProx (μ = 0.0001)', linewidth=2, markersize=5)
    plt.plot(rounds, acc_bad_pct, 'r-s', label='Lỗi: FedProx (μ = 0.01)', linewidth=2, markersize=5)
    
    plt.title('Đánh giá ảnh hưởng của tham số Proximal (μ) trong 25 vòng đầu', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='center right', fontsize=11)
    
    output_filename = 'hyperparameter_tuning_mu.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Đã vẽ xong biểu đồ: {output_filename}")

def main():
    print("=" * 60)
    print(" BƯỚC 2: CHẠY THỬ NGHIỆM LỖI (mu=0.01) TRONG 25 VÒNG")
    print(" Mục đích: Lấy dữ liệu để chứng minh quá trình Fine-tuning")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(df, target_column, test_size=0.2, use_smote=True)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    K = 10
    client_data = []
    n_per_client = len(X_train) // K
    for i in range(K):
        start = i * n_per_client
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
    
    eval_fn = create_eval_fn(X_test, y_test, num_classes, device)
    
    # CHẠY FEDPROX VỚI MU LỖI (0.01) TRONG 25 VÒNG
    fedprox_model_bad = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    _, hist_bad = run_federated_training_pytorch(
        fedprox_model_bad, client_data, num_rounds=25, local_epochs=5, 
        device=device, algo='fedprox', mu=0.01, eval_fn=eval_fn
    )
    
    np.savez('tuning_mu_bad.npz', acc=hist_bad['acc'])
    print("✅ Đã lưu kết quả mu=0.01")
    
    # VẼ BIỂU ĐỒ SO SÁNH LUÔN VỚI FILE ĐÊM QUA
    if os.path.exists('fedprox_convergence.npz'):
        data_good = np.load('fedprox_convergence.npz')
        plot_hyperparameter_tuning(hist_bad['acc'], data_good['fedprox_acc'])
    else:
        print("Không tìm thấy file fedprox_convergence.npz của đêm qua để vẽ so sánh.")

if __name__ == "__main__":
    main()
