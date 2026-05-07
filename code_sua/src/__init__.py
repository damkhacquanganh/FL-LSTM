"""
FL-LSTM IDS — Federated Learning + LSTM cho Phát hiện Xâm nhập IoT
===================================================================
Modules:
    model          — Kiến trúc LSTM_IDS 3 tầng (PyTorch)
    preprocessing  — Pipeline tiền xử lý 7 bước (Zero Data Leakage)
    training       — Huấn luyện cục bộ (Local Training) với FedProx Proximal Term
    evaluation     — Đánh giá mô hình (Accuracy, F1, FPR, RMSE, Confusion Matrix)
    federated      — Vòng lặp Federated Learning (Weighted FedAvg)
"""
from .model import LSTM_IDS, create_lstm_model_pytorch, clone_model_with_weights_pytorch
from .model import get_model_weights_pytorch, set_model_weights_pytorch
from .preprocessing import preprocess_data
from .training import train_client_pytorch
from .evaluation import evaluate_model_pytorch
from .federated import weighted_fedavg_pytorch, run_federated_training_pytorch
