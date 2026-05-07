import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

def evaluate_model_pytorch(model, X_test, y_test, device='cpu'):
    """
    Đánh giá mô hình PyTorch, tính toán 6 metrics (bao gồm FPR, RMSE theo bài báo).
    
    Args:
        model (nn.Module): Mô hình đã huấn luyện.
        X_test (np.ndarray): Dữ liệu test.
        y_test (np.ndarray): Nhãn test (one-hot).
        device (str): Thiết bị chạy.
        
    Returns:
        dict: Chứa các metrics (accuracy, precision, recall, f1, FPR, RMSE, confusion_matrix).
    """
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Keras có .predict(), PyTorch phải tự tính
    with torch.no_grad():
        outputs = model(X_tensor)
        # Vì ta dùng CrossEntropyLoss, output chưa có softmax
        y_pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
    # Xử lý y_test thật từ one-hot -> index
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
        
    # Tính các metrics cơ bản (Macro average cho multiclass)
    acc = accuracy_score(y_true, y_pred_classes)
    prec = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
    
    # Tính Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Tính False Positive Rate (FPR)
    num_classes = cm.shape[0]
    fpr_list = []
    
    for i in range(num_classes):
        # FP: Tổng cột i trừ đi đường chéo
        FP = cm[:, i].sum() - cm[i, i]
        # TN: Tổng toàn bộ ma trận trừ đi hàng i và cột i
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fpr_list.append(fpr)
        
    mean_fpr = np.mean(fpr_list)
    
    # Tính RMSE (Dựa trên xác suất dự đoán và one-hot label gốc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_probs))
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'FPR': mean_fpr,
        'RMSE': rmse,
        'confusion_matrix': cm
    }
