import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

from preprocess_custom import preprocess_data
from main_pytorch import load_dataset, auto_detect_target
from model_pytorch import create_lstm_model_pytorch

def main():
    print("=" * 65)
    print("  VẼ BIỂU ĐỒ ROC-AUC CHO FL-LSTM (100 ROUNDS)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data & Preprocess
    print("Đang tải dữ liệu và tiền xử lý...")
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    _, X_test_3d, _, y_test, _, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True, random_state=42
    )

    # 2. Tạo model và nạp file trọng số (.pt)
    print("Đang nạp trọng số mô hình FL (fedprox_wsn.pt)...")
    input_shape = (X_test_3d.shape[1], X_test_3d.shape[2])
    model = create_lstm_model_pytorch(input_shape, num_classes)
    
    model_path = 'fedprox_wsn.pt'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Lỗi: Không tìm thấy file {model_path}. Vui lòng đảm bảo bạn đã chạy xong 100 rounds.")
        return
        
    model.to(device)
    model.eval()

    # 3. Dự đoán để lấy xác suất (Probabilities)
    print("Đang chạy dự đoán trên tập Test...")
    X_test_tensor = torch.tensor(X_test_3d, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        # Áp dụng softmax để lấy xác suất [0-1] cho mỗi lớp
        y_prob = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    # 4. Vẽ biểu đồ ROC
    print("Đang tính toán toán học và vẽ biểu đồ ROC...")
    class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
    
    # Tính ROC và AUC cho từng class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Cài đặt biểu đồ
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve - {class_names[i]} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Đường nét đứt xéo (Baseline = random guess)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tỉ lệ báo động nhầm)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Tỉ lệ bắt đúng tội phạm)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) - FL-LSTM (100 Rounds)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    output_filename = 'roc_curve_fedprox.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Đã vẽ xong! Ảnh lưu tại: {output_filename}")

if __name__ == '__main__':
    main()
