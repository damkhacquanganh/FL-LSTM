import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_client_pytorch(client_model, X_train, y_train, epochs=5, batch_size=32, device='cpu', global_weights=None, mu=0.01):
    """
    Huấn luyện mô hình cho một client (Local Training) bằng PyTorch.
    
    Args:
        client_model (nn.Module): Mô hình của client.
        X_train (np.ndarray): Dữ liệu huấn luyện đầu vào.
        y_train (np.ndarray): Nhãn huấn luyện (đã one-hot encoded).
        epochs (int): Số local epochs (bài báo dùng E=5).
        batch_size (int): Kích thước batch.
        device (str): Thiết bị tính toán ('cuda' hoặc 'cpu').
        global_weights (list, optional): Trọng số của mô hình toàn cục (dành cho FedProx).
        mu (float, optional): Tham số Proximal (dành cho FedProx).
        
    Returns:
        list: Trọng số của mô hình sau khi huấn luyện.
        int: Số lượng mẫu dữ liệu (num_samples).
    """
    client_model = client_model.to(device)
    client_model.train()
    
    # Chuyển đổi dữ liệu sang PyTorch Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    # PyTorch CrossEntropyLoss yêu cầu nhãn là dạng số nguyên (class indices), không phải one-hot.
    # Vì y_train đang là one-hot (do preprocessing trước đó thiết kế cho Keras), ta chuyển đổi lại.
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_indices = np.argmax(y_train, axis=1)
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
    else:
        # Nếu binary thì dùng BCE (nhưng bài báo thường dùng CrossEntropy cho multiclass)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Hàm mất mát và bộ tối ưu hóa (Theo Table 3: Adam, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = client_model(batch_X)
            
            # Tính loss
            loss = criterion(outputs, batch_y)
            
            # Tính Proximal Term cho thuật toán FedProx
            if global_weights is not None and mu > 0:
                proximal_term = 0.0
                for local_param, global_param in zip(client_model.parameters(), global_weights):
                    proximal_term += ((local_param - global_param.to(device)) ** 2).sum()
                loss += (mu / 2.0) * proximal_term
                
            # Backward pass & update weights
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(dataloader.dataset)
        # Bỏ comment dòng dưới nếu muốn xem loss của từng epoch trong client
        # print(f"    Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Trả về trọng số mô hình trên CPU để tổng hợp (tránh tràn VRAM)
    client_model = client_model.to('cpu')
    weights = [param.data.clone() for param in client_model.parameters()]
    num_samples = len(X_train)
    
    return weights, num_samples
