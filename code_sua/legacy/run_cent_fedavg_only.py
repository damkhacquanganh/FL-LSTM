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
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr_per_class = fp / (fp + tn)
    mean_fpr = np.nanmean(fpr_per_class)
    
    print(f"\n{'=' * 50}")
    print(f" KẾT QUẢ {model_name} TRÊN WSN-DS")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  Mean FPR:  {mean_fpr * 100:.2f}%")
    print(f"{'=' * 50}")
    
    return acc, f1, mean_fpr

def main():
    print("=" * 60)
    print(" CHẠY BÙ CENTRALIZED & FEDAVG")
    print(" Mục đích: Lấy lịch sử hội tụ để vẽ biểu đồ so sánh")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. TIỀN XỬ LÝ DỮ LIỆU
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    eval_fn = create_eval_fn(X_test, y_test, num_classes, device)
    
    # 2. CHẠY CENTRALIZED (15 Epochs)
    print("\n" + "="*40)
    print(" EXPERIMENT 1: CENTRALIZED LEARNING")
    print("="*40)
    centralized_model = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    criterion = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(centralized_model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    if num_classes > 2:
        y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    hist_cent = {'acc': [], 'loss': []}
    for epoch in range(15):
        centralized_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = centralized_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        acc = eval_fn(centralized_model)
        hist_cent['acc'].append(acc)
        hist_cent['loss'].append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/15 - Loss: {epoch_loss/len(train_loader):.4f} - Test Acc: {acc*100:.2f}%")
        
    evaluate_final(centralized_model, X_test, y_test, num_classes, device, "CENTRALIZED")
    torch.save(centralized_model.state_dict(), 'centralized_wsn.pt')

    # 3. CHẠY FEDAVG (100 Rounds)
    print("\n" + "="*40)
    print(" EXPERIMENT 2: FEDERATED LEARNING (FEDAVG)")
    print("="*40)
    K = 10
    client_data = []
    n_per_client = len(X_train) // K
    for i in range(K):
        start = i * n_per_client
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
        
    fedavg_model = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    fedavg_model, hist_fedavg = run_federated_training_pytorch(
        fedavg_model, client_data, num_rounds=100, local_epochs=5, device=device, algo='fedavg', eval_fn=eval_fn
    )
    
    evaluate_final(fedavg_model, X_test, y_test, num_classes, device, "FEDAVG")
    torch.save(fedavg_model.state_dict(), 'fedavg_wsn.pt')

    # 4. LƯU LỊCH SỬ
    np.savez('cent_fedavg_convergence.npz', 
             cent_acc=hist_cent['acc'],
             fedavg_acc=hist_fedavg['acc'])
    print("\n✅ Đã lưu lịch sử hội tụ vào file: cent_fedavg_convergence.npz")
    print("🎉 HOÀN TẤT BƯỚC CHẠY BÙ! Giờ đã đủ dữ liệu để vẽ biểu đồ 3 đường.")

if __name__ == "__main__":
    main()
