"""
preprocess_custom.py — Tiền xử lý dữ liệu cho FL-LSTM IDS
============================================================
Tái hiện đúng pipeline của Anwar et al. (2025), PeerJ CS 11:e2751

Pipeline 5 bước (thứ tự ĐÚNG để tránh Data Leakage):
  1. Data Cleaning (drop duplicates, handle Inf/NaN)
  2. Label Encoding (string → integer → one-hot nếu multiclass)
  3. Train/Test Split (80/20, stratified) ← TRƯỚC normalize
  4. MinMax Normalization — fit CHỈ trên Train, transform riêng Test
  5. SMOTE trên Train set (cân bằng lớp)
  6. Reshape 3D cho LSTM: (samples, timesteps=1, features)

Fix skeleton bugs:
  ✓ Data leakage: split TRƯỚC, normalize SAU
  ✓ Reshape 3D cho LSTM
  ✓ Return scaler để transform test set riêng
  ✓ Tự động detect tất cả cột số (không hardcode 3 cột)
  ✓ Xử lý multiclass + binary tự động
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# Không dùng tensorflow nữa
# from tensorflow.keras.utils import to_categorical

# SMOTE là optional — nếu không cài imbalanced-learn thì bỏ qua
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠️  imbalanced-learn chưa cài. SMOTE sẽ bị bỏ qua.")
    print("   Cài bằng: pip install imbalanced-learn")


def preprocess_data(df, target_column, test_size=0.2, use_smote=True, random_state=42):
    """
    Pipeline tiền xử lý hoàn chỉnh cho FL-LSTM IDS.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe đã load từ CSV.
    target_column : str
        Tên cột nhãn (label), ví dụ: 'class', 'attack_type', 'label'.
    test_size : float
        Tỷ lệ tập test (default 0.2 = 20%).
    use_smote : bool
        Có dùng SMOTE để cân bằng lớp không (default True).
    random_state : int
        Seed cho reproducibility.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, 1, n_features) — 3D cho LSTM
    X_test  : np.ndarray, shape (n_test, 1, n_features)
    y_train : np.ndarray — one-hot nếu multiclass, 1D nếu binary
    y_test  : np.ndarray
    scaler  : MinMaxScaler — đã fit trên train, dùng lại cho inference
    num_classes : int — số lớp phân loại
    """

    df = df.copy()

    # =========================================================
    # BƯỚC 1: Data Cleaning
    # =========================================================
    initial_shape = df.shape
    # Xóa duplicate rows
    df.drop_duplicates(inplace=True)

    # Thay Inf/-Inf → NaN, rồi fillna bằng median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Loại target_column khỏi numeric nếu nó là số
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Drop rows nếu label bị NaN
    df.dropna(subset=[target_column], inplace=True)

    print(f"  [Bước 1] Cleaning: {initial_shape} → {df.shape}"
          f" (xóa {initial_shape[0] - df.shape[0]} rows)")

    # =========================================================
    # BƯỚC 2: Label Encoding
    # =========================================================
    le = LabelEncoder()
    # Nếu label là string → encode thành số
    if df[target_column].dtype == object:
        df[target_column] = le.fit_transform(df[target_column])
        print(f"  [Bước 2] Label encoding: {list(le.classes_)}"
              f" → {list(range(len(le.classes_)))}")
    else:
        # Nếu đã là số, đảm bảo bắt đầu từ 0
        unique_labels = sorted(df[target_column].unique())
        if unique_labels[0] != 0:
            label_map = {old: new for new, old in enumerate(unique_labels)}
            df[target_column] = df[target_column].map(label_map)
            print(f"  [Bước 2] Re-map labels: {unique_labels}"
                  f" → {list(range(len(unique_labels)))}")
        else:
            print(f"  [Bước 2] Labels OK (đã là số): {unique_labels}")

    num_classes = df[target_column].nunique()
    print(f"           Số lớp (num_classes) = {num_classes}")

    # =========================================================
    # BƯỚC 3: Tách features (X) và label (y) → Train/Test Split
    #          TRƯỚC khi normalize → TRÁNH DATA LEAKAGE
    # =========================================================
    # Chỉ giữ các cột số làm features (loại bỏ cột string còn sót)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in feature_cols:
        feature_cols.remove(target_column)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_column].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  [Bước 3] Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}"
          f" (stratified, test_size={test_size})")

    # =========================================================
    # BƯỚC 4: MinMax Normalization [0, 1]
    #          fit() CHỈ trên Train → transform() riêng Test
    # =========================================================
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)    # fit + transform trên Train
    X_test = scaler.transform(X_test)          # CHỈ transform trên Test
    print(f"  [Bước 4] MinMax normalization: fit on Train, transform on Test ✓")

    # =========================================================
    # BƯỚC 5: SMOTE — cân bằng lớp (CHỈ trên Train set)
    # =========================================================
    if use_smote and HAS_SMOTE:
        # Kiểm tra xem có lớp nào quá ít mẫu không
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        if min_count < 6:
            # SMOTE cần ít nhất k_neighbors+1 = 6 mẫu
            print(f"  [Bước 5] SMOTE bỏ qua: lớp nhỏ nhất chỉ có {min_count} mẫu"
                  f" (cần ≥ 6)")
        else:
            smote = SMOTE(random_state=random_state)
            X_train_before = X_train.shape[0]
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  [Bước 5] SMOTE: {X_train_before} → {X_train.shape[0]} samples"
                  f" (balanced)")
    else:
        reason = "imbalanced-learn chưa cài" if not HAS_SMOTE else "user tắt"
        print(f"  [Bước 5] SMOTE bỏ qua ({reason})")

    # =========================================================
    # BƯỚC 6: One-hot encoding cho labels (nếu multiclass)
    # =========================================================
    if num_classes > 2:
        y_train = np.eye(num_classes)[y_train.astype(int)]
        y_test = np.eye(num_classes)[y_test.astype(int)]
        print(f"  [Bước 6] One-hot encoding: y shape → {y_train.shape[1]} classes")
    else:
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        print(f"  [Bước 6] Binary labels: giữ nguyên 0/1")

    # =========================================================
    # BƯỚC 7: Reshape 3D cho LSTM — (samples, timesteps=1, features)
    # =========================================================
    n_features = X_train.shape[1]
    X_train = X_train.reshape(-1, 1, n_features)
    X_test = X_test.reshape(-1, 1, n_features)
    print(f"  [Bước 7] Reshape 3D: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"           input_shape cho LSTM = (1, {n_features})")

    return X_train, X_test, y_train, y_test, scaler, num_classes
