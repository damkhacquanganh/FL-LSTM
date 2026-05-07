import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import time

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import create_lstm_model_pytorch

def main():
    print("=" * 65)
    print("  VẼ BIỂU ĐỒ CONFUSION MATRIX CHO FEDPROX")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Load Data & Preprocess
    print("Đang tải dữ liệu và tiền xử lý...")
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    _, X_test_3d, _, y_test, _, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True, random_state=42
    )

    # 2. Tạo model và nạp file trọng số (.pt)
    model_path = 'fedprox_wsn.pt'
    print(f"Đang nạp trọng số mô hình từ {model_path}...")
    input_shape = (X_test_3d.shape[1], X_test_3d.shape[2])
    model = create_lstm_model_pytorch(input_shape, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Lỗi: Không tìm thấy file {model_path} hoặc không nạp được. Chi tiết: {e}")
        return
        
    model.to(device)
    model.eval()

    # 3. Dự đoán
    print("Đang chạy dự đoán trên tập Test...")
    X_test_tensor = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred_cpu = y_pred.cpu().numpy()
        y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred_cpu)
    print(f"Độ chính xác (Accuracy) của mô hình: {acc*100:.4f}%")

    # 4. Vẽ Confusion Matrix
    print("Đang vẽ ma trận nhầm lẫn...")
    class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
    cm = confusion_matrix(y_true, y_pred_cpu)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 14})
    
    plt.title(f'Confusion Matrix - FedProx (Accuracy: {acc*100:.2f}%)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=14, fontweight='bold')
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    output_filename = 'confusion_matrix_fedprox.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Đã vẽ xong! Ảnh lưu tại: {output_filename}")

if __name__ == '__main__':
    main()
