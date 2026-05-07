import torch
import torch.nn as nn
import copy

class LSTM_IDS(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTM_IDS, self).__init__()
        
        # Lớp LSTM 1 (128 units)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Lớp LSTM 2 (64 units)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        # Lớp LSTM 3 (64 units)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Lớp Dense ẩn
        self.dense = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        # Lớp Output
        self.output = nn.Linear(64, num_classes)
        
        # Lưu ý: PyTorch dùng CrossEntropyLoss đã bao gồm Softmax,
        # nên ở output layer ta không cần hàm activation Softmax nữa.

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        
        # Ở lớp LSTM cuối, ta chỉ lấy đầu ra ở timestep cuối cùng
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Lấy output của timestep cuối cùng
        
        x = self.dropout2(x)
        
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        return x

def create_lstm_model_pytorch(input_shape, num_classes):
    """
    Khởi tạo mô hình LSTM 3 tầng theo cấu trúc bài báo Anwar et al. (2025)
    
    Args:
        input_shape (tuple): (timesteps, features)
        num_classes (int): Số lượng nhãn phân loại
        
    Returns:
        Mô hình PyTorch LSTM
    """
    # Trong PyTorch, nn.LSTM cần input_size (số lượng features)
    input_size = input_shape[1] 
    model = LSTM_IDS(input_size=input_size, num_classes=num_classes)
    return model

def clone_model_with_weights_pytorch(global_model):
    """Tạo bản sao của model với cùng trọng số (phục vụ FL)"""
    new_model = copy.deepcopy(global_model)
    return new_model

def get_model_weights_pytorch(model):
    """Lấy danh sách các tensor trọng số của mô hình"""
    return [param.data.clone() for param in model.parameters()]

def set_model_weights_pytorch(model, weights):
    """Cập nhật trọng số cho mô hình"""
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.data.copy_(weight)
