import numpy as np
import matplotlib.pyplot as plt
import os

def plot_non_iid_convergence():
    if not os.path.exists('non_iid_convergence.npz'):
        print("Lỗi: Không tìm thấy file 'non_iid_convergence.npz'")
        return
        
    data = np.load('non_iid_convergence.npz')
    acc_fedavg = data['fedavg_acc']
    acc_fedprox = data['fedprox_acc']

    acc_fedavg_pct = [a * 100 for a in acc_fedavg]
    acc_fedprox_pct = [a * 100 for a in acc_fedprox]

    rounds_fl = range(1, 101)

    plt.figure(figsize=(10, 6))

    plt.plot(rounds_fl, acc_fedavg_pct, 'r-', label='FedAvg (Non-IID)', linewidth=1.5, alpha=0.8)
    plt.plot(rounds_fl, acc_fedprox_pct, 'b-', label='FedProx (μ=0.0001, Non-IID)', linewidth=2)

    plt.title('Sự vượt trội của FedProx trong môi trường Non-IID (Dữ liệu chia lệch cực đoan)', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    
    # Giới hạn trục Y
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)

    output_filename = 'non_iid_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH NON-IID (Vòng 100)")
    print("="*50)
    print(f"  FedAvg Accuracy  : {acc_fedavg_pct[-1]:.2f}%")
    print(f"  FedProx Accuracy : {acc_fedprox_pct[-1]:.2f}%")
    print("="*50)
    print(f"✅ Đã lưu biểu đồ tại: {output_filename}")

if __name__ == "__main__":
    plot_non_iid_convergence()
