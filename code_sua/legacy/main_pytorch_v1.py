"""
main_pytorch.py — Entry Point: FL-LSTM cho Phát hiện Xâm nhập IoT (PyTorch Version GPU)
============================================================
Tái hiện hoàn chỉnh bài báo Anwar et al. (2025) sử dụng PyTorch.
Hỗ trợ chạy native trên GPU (Windows/Linux) cực nhanh.

Cách chạy:
  .\venv_gpu\Scripts\python.exe main_pytorch.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from preprocess_custom import preprocess_data
from model_pytorch import create_lstm_model_pytorch
from federated_learning_pytorch import run_federated_training_pytorch
from evaluation_pytorch import evaluate_model_pytorch

def parse_args():
    parser = argparse.ArgumentParser(description="FL-LSTM PyTorch for IDS in IoT")
    parser.add_argument('--dataset', type=str, default='wsn', choices=['wsn', 'cic', 'unsw'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--clients', '-K', type=int, default=10)
    parser.add_argument('--rounds', '-T', type=int, default=100)
    parser.add_argument('--no-smote', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

DATASET_CONFIG = {
    'wsn': {'target_column': 'Class', 'description': 'WSN-DS (374,661 records, 23 features)'},
    'cic': {'target_column': ' Label', 'description': 'CIC-IDS-2017 (~2.8M records, 80 features)'},
    'unsw': {'target_column': 'label', 'description': 'UNSW-NB15 (~2.5M records, 49 features)'},
}

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_dataset(dataset_key, custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return pd.read_csv(custom_path)
    if dataset_key == 'wsn':
        path = os.path.join(DATA_ROOT, 'WSN-DS', 'WSN-DS.csv')
        if not os.path.exists(path): _dataset_not_found('WSN-DS', path)
        return pd.read_csv(path)
    elif dataset_key == 'cic':
        cic_dir = os.path.join(DATA_ROOT, 'CIC-IDS-2017')
        if not os.path.isdir(cic_dir): _dataset_not_found('CIC-IDS-2017', cic_dir)
        csv_files = [f for f in os.listdir(cic_dir) if f.endswith('.csv')]
        dfs = [pd.read_csv(os.path.join(cic_dir, f), low_memory=False) for f in sorted(csv_files)]
        return pd.concat(dfs, ignore_index=True)
    elif dataset_key == 'unsw':
        unsw_dir = os.path.join(DATA_ROOT, 'UNSW_NB15')
        train_path, test_path = os.path.join(unsw_dir, 'UNSW_NB15_training-set.csv'), os.path.join(unsw_dir, 'UNSW_NB15_testing-set.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            return pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
        else: _dataset_not_found('UNSW-NB15', unsw_dir)
    return None

def _dataset_not_found(name, path):
    print(f"\n❌ Không tìm thấy {name} tại: {path}")
    sys.exit(1)

def auto_detect_target(df):
    common_targets = ['Class', 'class', 'Label', 'label', ' Label', 'attack_type', 'Attack', 'attack', 'target', 'is_attack', 'intrusion', 'category']
    for col in common_targets:
        if col in df.columns: return col
    return df.columns[-1]

def main():
    args = parse_args()

    # Cài đặt thiết bị (GPU hoặc CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 65)
    print("  FL-LSTM for Intrusion Detection in IoT-based WSN")
    print("  Anwar et al. (2025) — PyTorch Version (GPU Accelerated)")
    print("=" * 65)
    print(f"🚀 DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. LOAD DATASET
    config = DATASET_CONFIG[args.dataset]
    print(f"\n📂 Dataset: {config['description']}")
    df = load_dataset(args.dataset, args.data_path)
    print(f"   Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")

    target_column = args.target or config['target_column']
    if target_column not in df.columns:
        target_column = auto_detect_target(df)
    print(f"   Target:  '{target_column}' — {df[target_column].nunique()} classes")

    # 2. PREPROCESSING
    print(f"\n{'─' * 65}\n📊 PREPROCESSING PIPELINE (tránh Data Leakage)\n{'─' * 65}")
    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=not args.no_smote, random_state=args.seed
    )

    # 3. GLOBAL MODEL
    print(f"\n{'─' * 65}\n🧠 MODEL ARCHITECTURE (Table 2 — bài báo)\n{'─' * 65}")
    input_shape = (X_train.shape[1], X_train.shape[2])
    global_model = create_lstm_model_pytorch(input_shape, num_classes)
    global_model = global_model.to(device)
    print(global_model)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 4. CHIA DỮ LIỆU
    K = args.clients
    print(f"\n{'─' * 65}\n🔗 FEDERATED SETUP — K={K} clients (IID split)\n{'─' * 65}")
    client_data = []
    n_per_client = len(X_train) // K
    for i in range(K):
        start = i * n_per_client
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
        print(f"  Client {i+1:2d}: samples={end - start:>7,} (index {start:,}–{end-1:,})")

    # 5. FEDERATED LEARNING
    print(f"\n{'─' * 65}\n🚀 FEDERATED LEARNING — {args.rounds} rounds\n{'─' * 65}")
    global_model, fl_history = run_federated_training_pytorch(
        global_model, client_data, num_rounds=args.rounds, local_epochs=5, device=device
    )

    # 6. ĐÁNH GIÁ
    print(f"\n{'─' * 65}\n📊 KẾT QUẢ ĐÁNH GIÁ (Table 4/5/6 — bài báo)\n{'─' * 65}")
    metrics = evaluate_model_pytorch(global_model, X_test, y_test, device=device)

    print(f"\n  {'Metric':<15} {'Value':>10}\n  {'─' * 28}")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key:<15} {value:>10.5f}")
    print(f"\n  Confusion Matrix:\n  {metrics['confusion_matrix']}")

    # 7. LƯU MODEL
    model_filename = f"fl_lstm_pytorch_{args.dataset}_K{K}_T{args.rounds}.pt"
    torch.save(global_model.state_dict(), model_filename)
    print(f"\n  ✅ Model đã lưu: {model_filename}")

    # 8. TÓM TẮT
    print(f"\n{'─' * 65}\n📋 TÓM TẮT\n{'─' * 65}")
    print(f"  Dataset:      {config['description']}")
    print(f"  FL Config:    K={K} clients, T={args.rounds} rounds, E=5 epochs")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  FPR:          {metrics['FPR']:.4f}")
    print(f"  RMSE:         {metrics['RMSE']:.4f}")
    print(f"{'─' * 65}")

if __name__ == '__main__':
    main()
