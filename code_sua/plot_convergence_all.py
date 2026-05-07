import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_convergence():
    # 1. Load dữ liệu Centralized & FedAvg
    if not os.path.exists('cent_fedavg_convergence.npz'):
        print("Lỗi: Không tìm thấy file 'cent_fedavg_convergence.npz'")
        return
    data_1 = np.load('cent_fedavg_convergence.npz')
    acc_cent = data_1['cent_acc']
    acc_fedavg = data_1['fedavg_acc']

    # 2. Load dữ liệu FedProx
    if not os.path.exists('fedprox_convergence.npz'):
        print("Lỗi: Không tìm thấy file 'fedprox_convergence.npz'")
        return
    data_2 = np.load('fedprox_convergence.npz')
    acc_fedprox = data_2['fedprox_acc']

    # 3. Chuyển đổi thành phần trăm
    acc_cent_pct = [a * 100 for a in acc_cent]
    acc_fedavg_pct = [a * 100 for a in acc_fedavg]
    acc_fedprox_pct = [a * 100 for a in acc_fedprox]

    # Vì Centralized chỉ chạy 15 epochs, ta mượn giá trị cuối cùng trải dài ra 100 rounds để dễ so sánh
    epochs_cent = range(1, len(acc_cent_pct) + 1)
    rounds_fl = range(1, 101)

    # 4. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))

    # Đường Centralized (chạy ngắn nhưng lấy làm Baseline)
    plt.plot(epochs_cent, acc_cent_pct, 'k--', label='Centralized (15 Epochs)', linewidth=2)
    # Đường kẻ ngang kéo dài Baseline
    plt.axhline(y=acc_cent_pct[-1], color='k', linestyle=':', alpha=0.5)

    # Đường FedAvg
    plt.plot(rounds_fl, acc_fedavg_pct, 'r-', label='FedAvg', linewidth=1.5, alpha=0.8)

    # Đường FedProx
    plt.plot(rounds_fl, acc_fedprox_pct, 'b-', label='FedProx (μ=0.0001)', linewidth=2)

    # 5. Trang trí biểu đồ
    plt.title('So sánh tốc độ hội tụ: Centralized vs FedAvg vs FedProx', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Rounds / Epochs', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    
    # Giới hạn trục Y để thấy rõ sự biến động ở phía trên
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)

    # 6. Lưu file
    output_filename = 'convergence_comparison_all.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Đã vẽ và lưu biểu đồ tại: {output_filename}")
    
    # Hiển thị trên màn hình nếu có giao diện
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_all_convergence()
