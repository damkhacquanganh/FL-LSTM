import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ma trận nhầm lẫn từ kết quả chạy thực tế
cm = np.array([
    [ 1865,     0,   144,     0,     1],
    [    0,   630,     1,     0,     0],
    [  424,     0,  2353,     4,     1],
    [    2,   155,   234, 65651,   366],
    [    1,     0,     5,    87,  1234]
])

# Tên các nhãn (classes) dựa trên map: ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA'] -> [0, 1, 2, 3, 4]
class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)

plt.title('Confusion Matrix - WSN-DS (LSTM Federated Learning)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Lưu lại thành ảnh chất lượng cao
output_filename = 'confusion_matrix_wsn.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"✅ Đã vẽ và lưu biểu đồ vào file: {output_filename}")
