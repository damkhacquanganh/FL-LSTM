"""
federated_learning.py — Vòng lặp Federated Learning với Weighted FedAvg
=========================================================================
Tái hiện đúng thuật toán FedAvg của McMahan et al. (2017), AISTATS:

  W(t+1) = Σ(k=1..K) [nₖ/N × wₖ(t+1)]

  Trong đó:
    nₖ = số mẫu tại client k
    N  = tổng số mẫu tất cả clients
    wₖ = weights local của client k sau E epochs training

Fix skeleton bugs:
  ✓ Weighted average (nₖ/N) thay vì simple np.mean()
  ✓ Có vòng lặp FL rounds hoàn chỉnh (skeleton thiếu)
  ✓ Có logic chia data cho K clients (skeleton thiếu)
  ✓ Có progress log mỗi round
"""

import numpy as np
from training import train_client


def weighted_fedavg(client_results):
    """
    Tổng hợp trọng số theo Weighted FedAvg (McMahan 2017).

    Formula: W_global = Σ (nₖ/N) × wₖ

    Parameters
    ----------
    client_results : list of (weights, num_samples)
        Kết quả từ mỗi client: trọng số model + số lượng mẫu.

    Returns
    -------
    aggregated_weights : list of np.ndarray
        Trọng số tổng hợp có trọng số.
    """
    # Tính tổng N = tổng mẫu tất cả clients
    total_samples = sum(n for _, n in client_results)

    # Khởi tạo aggregated weights = zeros
    num_layers = len(client_results[0][0])
    aggregated_weights = [np.zeros_like(client_results[0][0][i])
                          for i in range(num_layers)]

    # Weighted sum: W = Σ (nₖ/N) × wₖ
    for local_weights, num_samples in client_results:
        weight_factor = num_samples / total_samples  # nₖ/N
        for i in range(num_layers):
            aggregated_weights[i] += weight_factor * local_weights[i]

    return aggregated_weights


def run_federated_training(global_model, client_data, num_rounds=100):
    """
    Chạy vòng lặp Federated Learning đầy đủ.

    Parameters
    ----------
    global_model : tf.keras.Model
        Model toàn cục ban đầu.
    client_data : list of (X_local, y_local)
        Dữ liệu của K clients. Mỗi phần tử = (X, y) của 1 client.
    num_rounds : int
        Số FL rounds. Default = 100 theo bài báo.

    Returns
    -------
    global_model : tf.keras.Model
        Model toàn cục sau T rounds training.
    history : dict
        Lịch sử training: round, num_samples per client.
    """
    K = len(client_data)
    history = {'rounds': [], 'total_samples': []}

    print(f"  FL Config: K={K} clients, T={num_rounds} rounds, "
          f"E=5 local epochs, Aggregation=Weighted FedAvg")
    print(f"  Samples per client: {[len(x) for x, y in client_data]}")
    print()

    for round_idx in range(1, num_rounds + 1):
        # =====================================================
        # Bước 1: Server broadcast global weights → K clients
        # Bước 2+3: Mỗi client train cục bộ → trả về weights
        # =====================================================
        client_results = []
        for k, (X_k, y_k) in enumerate(client_data):
            local_weights, num_samples = train_client(
                global_model, X_k, y_k,
                local_epochs=5,   # Theo bài báo Table 3
                batch_size=32     # Theo bài báo Table 3
            )
            client_results.append((local_weights, num_samples))

        # =====================================================
        # Bước 4: FedAvg Aggregation — Weighted Average
        #   W(t+1) = Σ (nₖ/N) × wₖ(t+1)
        # =====================================================
        aggregated_weights = weighted_fedavg(client_results)

        # =====================================================
        # Bước 5: Update global model
        # =====================================================
        global_model.set_weights(aggregated_weights)

        # Log progress
        total = sum(n for _, n in client_results)
        history['rounds'].append(round_idx)
        history['total_samples'].append(total)

        if round_idx % 10 == 0 or round_idx == 1:
            print(f"  Round {round_idx:3d}/{num_rounds} — "
                  f"Total samples: {total:,} — "
                  f"Weights aggregated (Weighted FedAvg) ✓")

    print(f"\n  ✅ Federated Learning hoàn tất: {num_rounds} rounds × {K} clients")
    return global_model, history
