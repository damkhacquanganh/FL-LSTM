import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target, DATASET_CONFIG
from model_pytorch import create_lstm_model_pytorch
from evaluation_pytorch import evaluate_model_pytorch

def main():
    dataset_key = 'wsn'
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 65)
    print("  CENTRALIZED LSTM (Baseline Comparison)")
    print("=" * 65)
    
    # 1. Load Data
    df = load_dataset(dataset_key)
    target_column = auto_detect_target(df)
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True, random_state=seed
    )
    
    # 3. Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model_pytorch(input_shape, num_classes).to(device)
    
    # 4. Convert to PyTorch Dataloader
    print(f"\n🚀 Đang huấn luyện Centralized LSTM trên {device}...")
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_indices = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    y_tensor = torch.tensor(y_indices, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 # 20 epochs trên toàn bộ dữ liệu là đủ hội tụ cho Centralized
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        print(f"  Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss/len(dataset):.4f}")
        
    # 5. Evaluate
    print(f"\n{'─' * 65}\n📊 KẾT QUẢ CENTRALIZED\n{'─' * 65}")
    metrics = evaluate_model_pytorch(model, X_test, y_test, device=device)
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key:<15} {value:>10.5f}")

if __name__ == '__main__':
    main()
