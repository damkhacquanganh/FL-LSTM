"""
main.py — Entry Point: FL-LSTM cho Phát hiện Xâm nhập IoT
============================================================
Tái hiện hoàn chỉnh bài báo Anwar et al. (2025)
  "Federated Learning with LSTM for Intrusion Detection in IoT-WSN"
  PeerJ Computer Science 11:e2751, DOI: 10.7717/peerj-cs.2751

Cách chạy:
  python main.py                    # Mặc định: WSN-DS, binary
  python main.py --dataset wsn      # WSN-DS dataset
  python main.py --dataset cic      # CIC-IDS-2017 dataset
  python main.py --dataset unsw     # UNSW-NB15 dataset
  python main.py --rounds 50        # Giảm số rounds để test nhanh
  python main.py --clients 5        # Giảm số clients

Hyperparameters theo bài báo (Table 3):
  K = 10 clients
  T = 100 FL rounds
  E = 5 local epochs
  Batch size = 32
  Learning rate = 0.001 (Adam)
  LSTM: 128 → 64 → 64
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess_custom import preprocess_data
from model import create_lstm_model
from federated_learning import run_federated_training
from evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="FL-LSTM for Intrusion Detection in IoT (Anwar et al. 2025)"
    )
    parser.add_argument(
        '--dataset', type=str, default='wsn',
        choices=['wsn', 'cic', 'unsw'],
        help='Dataset to use: wsn (WSN-DS), cic (CIC-IDS-2017), unsw (UNSW-NB15)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Custom path to CSV file (overrides --dataset)'
    )
    parser.add_argument(
        '--target', type=str, default=None,
        help='Target column name (auto-detected if not specified)'
    )
    parser.add_argument(
        '--clients', '-K', type=int, default=10,
        help='Number of FL clients (default: 10)'
    )
    parser.add_argument(
        '--rounds', '-T', type=int, default=100,
        help='Number of FL rounds (default: 100)'
    )
    parser.add_argument(
        '--no-smote', action='store_true',
        help='Disable SMOTE oversampling'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    return parser.parse_args()


# ================================================================
# Cấu hình dataset — tên file + tên cột label
# ================================================================
DATASET_CONFIG = {
    'wsn': {
        'target_column': 'Class',
        'description': 'WSN-DS (374,661 records, 23 features, 5 classes)'
    },
    'cic': {
        'target_column': ' Label',       # CIC-IDS-2017 có space trước Label
        'description': 'CIC-IDS-2017 (~2.8M records, 80 features)'
    },
    'unsw': {
        'target_column': 'label',        # UNSW-NB15: binary label
        'description': 'UNSW-NB15 (~2.5M records, 49 features)'
    },
}

# Đường dẫn data thực tế (tương đối từ thư mục code_sua/)
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_dataset(dataset_key, custom_path=None):
    """
    Load dataset theo cấu trúc thư mục thực tế.
    Hỗ trợ: WSN-DS (1 file), CIC-IDS-2017 (8 files), UNSW-NB15 (train+test).
    """
    if custom_path and os.path.exists(custom_path):
        return pd.read_csv(custom_path)

    if dataset_key == 'wsn':
        path = os.path.join(DATA_ROOT, 'WSN-DS', 'WSN-DS.csv')
        if not os.path.exists(path):
            _dataset_not_found('WSN-DS', path)
        return pd.read_csv(path)

    elif dataset_key == 'cic':
        cic_dir = os.path.join(DATA_ROOT, 'CIC-IDS-2017')
        if not os.path.isdir(cic_dir):
            _dataset_not_found('CIC-IDS-2017', cic_dir)
        csv_files = [f for f in os.listdir(cic_dir) if f.endswith('.csv')]
        if not csv_files:
            _dataset_not_found('CIC-IDS-2017', cic_dir)
        print(f"  CIC-IDS-2017: nối {len(csv_files)} file CSV...")
        dfs = []
        for f in sorted(csv_files):
            fp = os.path.join(cic_dir, f)
            dfs.append(pd.read_csv(fp, low_memory=False))
            print(f"    ✓ {f} ({len(dfs[-1]):,} rows)")
        return pd.concat(dfs, ignore_index=True)

    elif dataset_key == 'unsw':
        unsw_dir = os.path.join(DATA_ROOT, 'UNSW_NB15')
        train_path = os.path.join(unsw_dir, 'UNSW_NB15_training-set.csv')
        test_path = os.path.join(unsw_dir, 'UNSW_NB15_testing-set.csv')
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"  UNSW-NB15: nối training-set + testing-set...")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            print(f"    ✓ Training: {len(df_train):,} rows")
            print(f"    ✓ Testing:  {len(df_test):,} rows")
            return pd.concat([df_train, df_test], ignore_index=True)
        else:
            _dataset_not_found('UNSW-NB15', unsw_dir)

    return None


def _dataset_not_found(name, path):
    """In thông báo lỗi và thoát."""
    print(f"\n❌ Không tìm thấy {name} tại: {path}")
    print(f"   Download datasets:")
    print(f"   • WSN-DS:       https://www.kaggle.com/datasets/basimalhasan/wsn-ds")
    print(f"   • CIC-IDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html")
    print(f"   • UNSW-NB15:    https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    sys.exit(1)


def auto_detect_target(df):
    """Tự động phát hiện cột label từ tên cột phổ biến."""
    common_targets = [
        'Class', 'class', 'Label', 'label', ' Label',
        'attack_type', 'Attack', 'attack', 'target',
        'is_attack', 'intrusion', 'category'
    ]
    for col in common_targets:
        if col in df.columns:
            return col
    # Fallback: cột cuối cùng
    return df.columns[-1]


def main():
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 65)
    print("  FL-LSTM for Intrusion Detection in IoT-based WSN")
    print("  Anwar et al. (2025) — PeerJ Computer Science 11:e2751")
    print("=" * 65)

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    config = DATASET_CONFIG[args.dataset]

    print(f"\n📂 Dataset: {config['description']}")
    df = load_dataset(args.dataset, args.data_path)
    print(f"   Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns")


    # Xác định cột target
    target_column = args.target or config['target_column']
    if target_column not in df.columns:
        target_column = auto_detect_target(df)
    print(f"   Target:  '{target_column}' — {df[target_column].nunique()} classes")
    print(f"   Classes: {df[target_column].value_counts().to_dict()}")

    # ============================================================
    # 2. PREPROCESSING
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"📊 PREPROCESSING PIPELINE (tránh Data Leakage)")
    print(f"{'─' * 65}")

    X_train, X_test, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column,
        test_size=0.2,
        use_smote=not args.no_smote,
        random_state=args.seed
    )

    # ============================================================
    # 3. TẠO GLOBAL MODEL — LSTM 3 tầng (128 → 64 → 64)
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"🧠 MODEL ARCHITECTURE (Table 2 — bài báo)")
    print(f"{'─' * 65}")

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps=1, features)
    global_model = create_lstm_model(input_shape, num_classes)
    global_model.summary()

    # ============================================================
    # 4. CHIA DỮ LIỆU CHO K CLIENTS (IID split)
    # ============================================================
    K = args.clients
    print(f"\n{'─' * 65}")
    print(f"🔗 FEDERATED SETUP — K={K} clients (IID split)")
    print(f"{'─' * 65}")

    client_data = []
    n_per_client = len(X_train) // K

    for i in range(K):
        start = i * n_per_client
        # Client cuối nhận phần dư
        end = len(X_train) if i == K - 1 else start + n_per_client
        client_data.append((X_train[start:end], y_train[start:end]))
        print(f"  Client {i+1:2d}: samples={end - start:>7,}"
              f"  (index {start:,}–{end-1:,})")

    # ============================================================
    # 5. FEDERATED LEARNING — T rounds × K clients
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"🚀 FEDERATED LEARNING — {args.rounds} rounds")
    print(f"{'─' * 65}")

    global_model, fl_history = run_federated_training(
        global_model, client_data, num_rounds=args.rounds
    )

    # ============================================================
    # 6. ĐÁNH GIÁ — 6 metrics đầy đủ theo bài báo
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ (Table 4/5/6 — bài báo)")
    print(f"{'─' * 65}")

    metrics = evaluate_model(global_model, X_test, y_test)

    print(f"\n  {'Metric':<15} {'Value':>10}")
    print(f"  {'─' * 28}")
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            continue
        print(f"  {key:<15} {value:>10.5f}")

    print(f"\n  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")

    # ============================================================
    # 7. LƯU MODEL
    # ============================================================
    model_filename = f"fl_lstm_{args.dataset}_K{K}_T{args.rounds}.h5"
    global_model.save(model_filename)
    print(f"\n  ✅ Model đã lưu: {model_filename}")

    # ============================================================
    # 8. SO SÁNH VỚI CENTRALIZED (tùy chọn)
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"📋 TÓM TẮT")
    print(f"{'─' * 65}")
    print(f"  Dataset:      {config['description']}")
    print(f"  Architecture: LSTM 3 layers (128→64→64) + Dense(64)")
    print(f"  FL Config:    K={K} clients, T={args.rounds} rounds, E=5 epochs")
    print(f"  FedAvg:       Weighted (nₖ/N) — McMahan 2017")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  FPR:          {metrics['FPR']:.4f}")
    print(f"  RMSE:         {metrics['RMSE']:.4f}")
    print(f"{'─' * 65}")


if __name__ == '__main__':
    main()
