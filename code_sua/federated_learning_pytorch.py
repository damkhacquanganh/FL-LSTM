import torch
from model_pytorch import clone_model_with_weights_pytorch, get_model_weights_pytorch, set_model_weights_pytorch
from training_pytorch import train_client_pytorch

def weighted_fedavg_pytorch(client_weights, client_sizes):
    """
    Thực hiện Weighted Federated Averaging (McMahan et al., 2017) cho PyTorch.
    Công thức: w_global = Σ (n_k / N) * w_k
    """
    total_samples = sum(client_sizes)
    
    # Khởi tạo danh sách trọng số mới bằng 0
    new_weights = [torch.zeros_like(w) for w in client_weights[0]]
    
    # Tính trung bình có trọng số cho từng client
    for weights, size in zip(client_weights, client_sizes):
        weight_factor = size / total_samples
        for i, param in enumerate(weights):
            new_weights[i] += param * weight_factor
            
    return new_weights

def run_federated_training_pytorch(global_model, client_data, num_rounds=100, local_epochs=5, device='cpu'):
    """
    Vòng lặp Federated Learning.
    
    Args:
        global_model (nn.Module): Mô hình toàn cục ban đầu.
        client_data (list): List các tuple (X_train, y_train) cho từng client.
        num_rounds (int): Số vòng giao tiếp FL.
        local_epochs (int): Số epoch huấn luyện tại mỗi client.
        device (str): Thiết bị chạy ('cuda' hoặc 'cpu').
        
    Returns:
        global_model: Mô hình toàn cục đã được huấn luyện.
        history: Lịch sử số liệu (hiện tại để trống, có thể mở rộng lưu validation).
    """
    num_clients = len(client_data)
    history = {'round': []}
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n[FL Round {round_num}/{num_rounds}] Bắt đầu huấn luyện trên {num_clients} clients...")
        
        client_weights = []
        client_sizes = []
        
        # 1. Gửi global model cho từng client và huấn luyện local
        for client_id, (X_train, y_train) in enumerate(client_data):
            print(f"  → Đang huấn luyện Client {client_id+1}/{num_clients} (samples: {len(X_train):,})...", end='', flush=True)
            
            # Tạo bản sao local model từ global model
            local_model = clone_model_with_weights_pytorch(global_model)
            
            # Huấn luyện
            weights, num_samples = train_client_pytorch(
                client_model=local_model,
                X_train=X_train,
                y_train=y_train,
                epochs=local_epochs,
                batch_size=512,
                device=device
            )
            
            client_weights.append(weights)
            client_sizes.append(num_samples)
            print(" Hoàn thành.")
            
        # 2. Tổng hợp trọng số tại Server bằng Weighted FedAvg
        print("  → Server đang tổng hợp trọng số (Weighted FedAvg)...", end='', flush=True)
        new_global_weights = weighted_fedavg_pytorch(client_weights, client_sizes)
        set_model_weights_pytorch(global_model, new_global_weights)
        print(" Hoàn thành.")
        
        history['round'].append(round_num)
        
    print("\n✅ Quá trình Federated Learning đã hoàn tất!")
    return global_model, history
