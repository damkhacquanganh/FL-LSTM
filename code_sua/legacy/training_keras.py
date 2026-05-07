"""
training.py — Huấn luyện cục bộ (Local Training) cho mỗi Client
=================================================================
Tái hiện đúng mô tả của Anwar et al. (2025):
  - Local epochs = 5 (Table 3)
  - Batch size = 32
  - Optimizer = Adam (lr=0.001)

Fix skeleton bugs:
  ✓ Epochs = 5 (skeleton dùng 10 — gây client drift)
  ✓ Trả về cả weights VÀ num_samples (cần cho Weighted FedAvg)
"""

from model import clone_model_with_weights


def train_client(global_model, X_local, y_local, local_epochs=5, batch_size=32):
    """
    Huấn luyện model cục bộ trên dữ liệu của 1 client.

    Flow theo FedAvg (McMahan 2017):
      1. Nhận global weights từ server
      2. Khởi tạo local model = bản sao global model
      3. Train local model trên dữ liệu riêng (E epochs)
      4. Trả về local weights + số lượng samples

    Parameters
    ----------
    global_model : tf.keras.Model
        Model toàn cục (đã compile).
    X_local : np.ndarray
        Dữ liệu features của client này, shape (n, 1, features).
    y_local : np.ndarray
        Nhãn tương ứng.
    local_epochs : int
        Số epochs train cục bộ. Default = 5 theo bài báo.
    batch_size : int
        Batch size. Default = 32 theo bài báo.

    Returns
    -------
    local_weights : list of np.ndarray
        Trọng số sau khi train xong.
    num_samples : int
        Số lượng mẫu huấn luyện tại client này.
        ⚠️ Skeleton gốc KHÔNG trả về giá trị này → không tính được Weighted FedAvg.
    """
    # Clone model để train riêng — không ảnh hưởng global model
    local_model = clone_model_with_weights(global_model)

    # Train cục bộ
    local_model.fit(
        X_local, y_local,
        epochs=local_epochs,
        batch_size=batch_size,
        verbose=0  # Tắt log từng client để output gọn
    )

    # Trả về weights + số lượng samples
    local_weights = local_model.get_weights()
    num_samples = len(X_local)

    return local_weights, num_samples
