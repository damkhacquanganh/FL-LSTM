import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists('convergence_history.npz'):
        print("❌ Không tìm thấy file convergence_history.npz. Vui lòng chạy run_experiments_auto.py trước.")
        return

    # Load dữ liệu
    data = np.load('convergence_history.npz')
    cent_acc = data['cent_acc'] * 100      # Đổi sang %
    fedavg_acc = data['fedavg_acc'] * 100
    fedprox_acc = data['fedprox_acc'] * 100
    
    # Thiết lập x-axis (Epochs cho Centralized, Rounds cho Fed)
    # Vì Centralized chỉ có 20 epochs, Fed có 100 rounds, ta cần scale cho đẹp trên cùng biểu đồ.
    # Một cách thông dụng là vẽ theo % quá trình huấn luyện (từ 0 đến 100%)
    
    cent_x = np.linspace(0, 100, len(cent_acc))
    fed_x = np.linspace(0, 100, len(fedavg_acc))

    plt.figure(figsize=(10, 6))
    
    # Vẽ Centralized
    plt.plot(cent_x, cent_acc, label='Centralized Learning (20 Epochs)', 
             color='#2ca02c', linestyle='-', linewidth=2)
    
    # Vẽ FedAvg
    plt.plot(fed_x, fedavg_acc, label='FedAvg (100 Rounds)', 
             color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8)
             
    # Vẽ FedProx
    plt.plot(fed_x, fedprox_acc, label='FedProx (100 Rounds)', 
             color='#d62728', linestyle='-', linewidth=2.5)

    plt.title('Convergence Analysis: Accuracy vs Training Progress (WSN-DS)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Progress (%)', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Giới hạn trục Y để dễ nhìn độ chênh lệch ở đoạn hội tụ
    min_acc = min(np.min(fedavg_acc), np.min(fedprox_acc), np.min(cent_acc))
    plt.ylim([max(min_acc - 5, 0), 100.5])
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300)
    print("✅ Đã lưu biểu đồ hội tụ vào file convergence_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
