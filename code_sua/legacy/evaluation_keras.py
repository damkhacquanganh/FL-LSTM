"""
evaluation.py — Đánh giá model với 6 metrics đầy đủ
=====================================================
Tái hiện đúng Table 4/5/6 của Anwar et al. (2025):
  1. Accuracy
  2. Precision (weighted)
  3. Recall (weighted)
  4. F1-Score (weighted)
  5. FPR (False Positive Rate) ← SKELETON THIẾU
  6. RMSE (Root Mean Squared Error) ← SKELETON THIẾU
  + Confusion Matrix

Fix skeleton bugs:
  ✓ Thêm FPR — tính từ Confusion Matrix: FP / (FP + TN)
  ✓ Thêm RMSE — tính từ predicted probabilities vs true labels
  ✓ Xử lý đúng multiclass (argmax one-hot) vs binary
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error
)


def evaluate_model(model, X_test, y_test):
    """
    Đánh giá model trên tập test với đầy đủ 6 metrics của bài báo.

    Parameters
    ----------
    model : tf.keras.Model
        Model đã train xong.
    X_test : np.ndarray
        Dữ liệu test, shape (n, 1, features).
    y_test : np.ndarray
        Nhãn test — one-hot nếu multiclass, 1D nếu binary.

    Returns
    -------
    metrics : dict
        Dictionary chứa tất cả metrics.
    """
    # Predict
    y_pred_prob = model.predict(X_test, verbose=0)

    # =========================================================
    # Chuyển đổi predictions → class labels
    # =========================================================
    is_multiclass = y_test.ndim > 1 and y_test.shape[1] > 1

    if is_multiclass:
        # Multiclass: argmax
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred_prob, axis=1)
        # Dùng y_test (one-hot) và y_pred_prob để tính RMSE
        y_true_for_rmse = y_test
        y_pred_for_rmse = y_pred_prob
    else:
        # Binary: threshold 0.5
        y_test_labels = y_test.astype(int).flatten()
        y_pred_labels = (y_pred_prob > 0.5).astype(int).flatten()
        # RMSE trên probability
        y_true_for_rmse = y_test.flatten()
        y_pred_for_rmse = y_pred_prob.flatten()

    # =========================================================
    # 4 Classification Metrics cơ bản
    # =========================================================
    acc = accuracy_score(y_test_labels, y_pred_labels)
    prec = precision_score(y_test_labels, y_pred_labels, average='weighted',
                           zero_division=0)
    rec = recall_score(y_test_labels, y_pred_labels, average='weighted',
                       zero_division=0)
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted',
                  zero_division=0)

    # =========================================================
    # FPR (False Positive Rate) — SKELETON THIẾU
    #   Binary:     FPR = FP / (FP + TN)
    #   Multiclass: FPR trung bình macro
    # =========================================================
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    fpr = _compute_fpr(cm)

    # =========================================================
    # RMSE (Root Mean Squared Error) — SKELETON THIẾU
    #   Tính trên predicted probabilities vs true labels
    # =========================================================
    rmse = np.sqrt(mean_squared_error(y_true_for_rmse, y_pred_for_rmse))

    metrics = {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1_score':  f1,
        'FPR':       fpr,
        'RMSE':      rmse,
        'confusion_matrix': cm
    }

    return metrics


def _compute_fpr(cm):
    """
    Tính False Positive Rate từ Confusion Matrix.

    Binary (2×2):
      FPR = FP / (FP + TN)

    Multiclass (n×n):
      FPR_k = Σ(j≠k) CM[j,k] / Σ(j≠k) Σ(all) CM[j,:]
      → Trung bình macro trên tất cả classes

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix, shape (n, n).

    Returns
    -------
    fpr : float
        False Positive Rate (trung bình nếu multiclass).
    """
    n_classes = cm.shape[0]

    if n_classes == 2:
        # Binary: [[TN, FP], [FN, TP]]
        TN, FP = cm[0, 0], cm[0, 1]
        if (FP + TN) == 0:
            return 0.0
        return FP / (FP + TN)
    else:
        # Multiclass: trung bình macro
        fpr_per_class = []
        for k in range(n_classes):
            # FP cho class k = tổng cột k, trừ đi TP (đường chéo)
            FP_k = cm[:, k].sum() - cm[k, k]
            # TN cho class k = tổng tất cả - (hàng k + cột k) + cm[k,k]
            TN_k = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
            if (FP_k + TN_k) > 0:
                fpr_per_class.append(FP_k / (FP_k + TN_k))
            else:
                fpr_per_class.append(0.0)
        return np.mean(fpr_per_class)
