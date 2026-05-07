import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import time
import numpy as np

# Import từ các file hiện có
from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import create_lstm_model_pytorch
from federated_learning_pytorch import run_federated_training_pytorch

# Import logic cho Centralized
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def train_centralized(model, X_train, y_train, epochs=20, batch_size=512, device='cpu', eval_fn=None):
    model = model.to(device)
    model.train()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_indices = np.argmax(y_train, axis=1)
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
    else:
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'epoch': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train() # Đảm bảo mô hình luôn ở chế độ train khi bắt đầu epoch mới
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataloader.dataset)
        
        if eval_fn is not None:
            acc = eval_fn(model)
            history['epoch'].append(epoch + 1)
            history['acc'].append(acc)
            print(f"  [Centralized] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {acc*100:.2f}%")
        else:
            print(f"  [Centralized] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    return model, history

def create_eval_fn(X_test, y_test, num_classes, device):
    def eval_fn(model):
        was_training = model.training # Lưu lại trạng thái cũ
        model.eval()
        model = model.to(device) # Đảm bảo mô hình đang ở trên GPU
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        if num_classes > 2:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
            
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
        
        # Trả mô hình về trạng thái train nếu trước đó nó đang train
        if was_training:
            model.train()
            
        return accuracy_score(y_true, y_pred_all)
    return eval_fn

def evaluate_model(model, X_test, y_test, num_classes, device, model_name):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        if num_classes > 2:
            _, y_pred = torch.max(outputs, 1)
            y_true = np.argmax(y_test, axis=1)
        else:
            y_pred = (torch.sigmoid(outputs) > 0.5).float()
            y_true = y_test
            
    y_pred_cpu = y_pred.cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred_cpu)
    f1 = f1_score(y_true, y_pred_cpu, average='weighted')
    
    # Tính FPR
    cm = confusion_matrix(y_true, y_pred_cpu)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    fpr_per_class = fp / (fp + tn)
    mean_fpr = np.nanmean(fpr_per_class)
    
    print(f"\n[{model_name}] KẾT QUẢ ĐÁNH GIÁ:")
    print(f"  - Accuracy: {acc * 100:.2f}%")
    print(f"  - F1-Score: {f1 * 100:.2f}%")
    print(f"  - Mean FPR: {mean_fpr * 100:.2f}%\n")
    return acc, f1, mean_fpr

def main():
    print("="*60)
    print(" BỘ TỰ ĐỘNG CHẠY THỰC NGHIỆM: CENTRALIZED vs FEDAVG vs FEDPROX")
    print("="*60)
    
    dataset_name = 'wsn'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. TIỀN XỬ LÝ DỮ LIỆU
    df = load_dataset(dataset_name)
    target_column = auto_detect_target(df)
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Chuẩn bị dữ liệu cho FL (10 clients)
    K = 10
    client_data = []
    n_per_client = len(X_train) // K
    for i in range(K):
        start = i * n_per_client
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
        
    # Tạo hàm đánh giá hội tụ
    eval_fn = create_eval_fn(X_test, y_test, num_classes, device)
    
    results = {}
    
    # ==========================================
    # EXPERIMENT 1: CENTRALIZED LEARNING
    # ==========================================
    print("\n" + "="*40)
    print(" EXPERIMENT 1: CENTRALIZED LEARNING")
    print("="*40)
    centralized_model = create_lstm_model_pytorch(input_shape, num_classes)
    start_time = time.time()
    centralized_model, hist_cent = train_centralized(
        centralized_model, X_train, y_train, epochs=20, batch_size=512, device=device, eval_fn=eval_fn
    )
    train_time = time.time() - start_time
    print(f"Time taken: {train_time:.2f} seconds")
    
    acc, f1, fpr = evaluate_model(centralized_model, X_test, y_test, num_classes, device, "CENTRALIZED")
    results['Centralized'] = {'Acc': acc, 'F1': f1, 'FPR': fpr, 'Time': train_time}
    torch.save(centralized_model.state_dict(), f'centralized_wsn.pt')

    # ==========================================
    # EXPERIMENT 2: FEDERATED LEARNING (FEDAVG)
    # ==========================================
    print("\n" + "="*40)
    print(" EXPERIMENT 2: FEDERATED LEARNING (FEDAVG)")
    print("="*40)
    fedavg_model = create_lstm_model_pytorch(input_shape, num_classes)
    fedavg_model = fedavg_model.to(device)
    start_time = time.time()
    fedavg_model, hist_fedavg = run_federated_training_pytorch(
        fedavg_model, client_data, num_rounds=100, local_epochs=5, device=device, algo='fedavg', eval_fn=eval_fn
    )
    train_time = time.time() - start_time
    print(f"Time taken: {train_time:.2f} seconds")
    
    acc, f1, fpr = evaluate_model(fedavg_model, X_test, y_test, num_classes, device, "FEDAVG")
    results['FedAvg'] = {'Acc': acc, 'F1': f1, 'FPR': fpr, 'Time': train_time}
    torch.save(fedavg_model.state_dict(), f'fedavg_wsn.pt')

    # ==========================================
    # EXPERIMENT 3: FEDERATED LEARNING (FEDPROX)
    # ==========================================
    print("\n" + "="*40)
    print(" EXPERIMENT 3: FEDERATED LEARNING (FEDPROX)")
    print("="*40)
    fedprox_model = create_lstm_model_pytorch(input_shape, num_classes)
    fedprox_model = fedprox_model.to(device)
    start_time = time.time()
    fedprox_model, hist_fedprox = run_federated_training_pytorch(
        fedprox_model, client_data, num_rounds=100, local_epochs=5, device=device, algo='fedprox', mu=0.01, eval_fn=eval_fn
    )
    train_time = time.time() - start_time
    print(f"Time taken: {train_time:.2f} seconds")
    
    acc, f1, fpr = evaluate_model(fedprox_model, X_test, y_test, num_classes, device, "FEDPROX")
    results['FedProx'] = {'Acc': acc, 'F1': f1, 'FPR': fpr, 'Time': train_time}
    torch.save(fedprox_model.state_dict(), f'fedprox_wsn.pt')

    # Lưu lịch sử hội tụ để vẽ biểu đồ
    np.savez('convergence_history.npz', 
             cent_acc=hist_cent['acc'],
             fedavg_acc=hist_fedavg['acc'],
             fedprox_acc=hist_fedprox['acc'])
    print("\n  ✅ Đã lưu lịch sử hội tụ vào file convergence_history.npz")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*50)
    print(" BẢNG TỔNG KẾT KẾT QUẢ THỰC NGHIỆM TRÊN WSN-DS")
    print("="*50)
    print(f"{'Thuật toán':<15} | {'Accuracy (%)':<15} | {'F1-Score (%)':<15} | {'Mean FPR (%)':<15}")
    print("-" * 65)
    for algo, res in results.items():
        print(f"{algo:<15} | {res['Acc']*100:<15.2f} | {res['F1']*100:<15.2f} | {res['FPR']*100:<15.2f}")
    print("="*50)
    print("Hoàn tất! Các file trọng số .pt đã được lưu.")

if __name__ == "__main__":
    main()
