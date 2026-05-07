import matplotlib.pyplot as plt
import numpy as np

# Số liệu thu thập được từ các bước chạy
models = ['FL-LSTM\n(Đề tài của bạn)', 'Centralized\nLSTM', 'Decision\nTree', 'Random\nForest']
accuracy = [98.05, 98.98, 99.40, 99.62]
f1_score = [88.92, 93.76, 96.43, 97.62]

x = np.arange(len(models))  # Vị trí các nhãn trên trục x
width = 0.35  # Độ rộng của cột

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2ecc71')
rects2 = ax.bar(x + width/2, f1_score, width, label='F1-Score (%)', color='#3498db')

# Căn chỉnh và Thêm nhãn
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('So sánh Hiệu suất các Thuật toán trên bộ WSN-DS', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(80, 105) # Giới hạn trục y từ 80 đến 105 để thấy rõ sự khác biệt
ax.legend(loc='lower right', fontsize=12)

# Hàm gắn số liệu trực tiếp lên đỉnh cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Lưu thành ảnh
output_filename = 'model_comparison_bar_chart.png'
plt.savefig(output_filename, dpi=300)
print(f"✅ Đã vẽ và lưu biểu đồ cột so sánh vào: {output_filename}")
