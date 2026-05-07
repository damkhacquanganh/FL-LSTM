import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

from src.preprocessing import preprocess_data
from main_pytorch import load_dataset, auto_detect_target

def evaluate_sklearn_model(model, X_test, y_test, name="Model"):
    # sklearn cần y_test dạng 1D (chỉ số lớp)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    print(f"Đang dự đoán với {name}...")
    y_pred = model.predict(X_test)
    
    # Tính các chỉ số
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # Tính FPR trung bình macro
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr_array = fp / (fp + tn)
    fpr_macro = np.mean(fpr_array)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{'─' * 40}")
    print(f"KẾT QUẢ: {name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy:  {acc:.5f}")
    print(f"  Precision: {prec:.5f}")
    print(f"  Recall:    {rec:.5f}")
    print(f"  F1-Score:  {f1:.5f}")
    print(f"  FPR:       {fpr_macro:.5f}")
    print(f"  RMSE:      {rmse:.5f}")
    print(f"{'─' * 40}\n")

def main():
    print("=" * 65)
    print("  TRADITIONAL ML BASELINES (Random Forest & Decision Tree)")
    print("=" * 65)
    
    # 1. Load & Preprocess Data
    df = load_dataset('wsn')
    target_column = auto_detect_target(df)
    
    X_train_3d, X_test_3d, y_train, y_test, scaler, num_classes = preprocess_data(
        df, target_column, test_size=0.2, use_smote=True, random_state=42
    )
    
    # 2. Reshape từ 3D (của LSTM) về 2D cho scikit-learn
    X_train = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test = X_test_3d.reshape(X_test_3d.shape[0], -1)
    
    y_train_1d = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
    
    # 3. Decision Tree (Chạy cực nhanh)
    print("\n⏳ Đang huấn luyện Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train_1d)
    evaluate_sklearn_model(dt, X_test, y_test, "Decision Tree")
    
    # 4. Random Forest (Chạy hơi lâu tầm 3-5 phút do 1.3 triệu dòng)
    print("\n⏳ Đang huấn luyện Random Forest (n_estimators=50)...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # n_jobs=-1 để tận dụng full CPU
    rf.fit(X_train, y_train_1d)
    evaluate_sklearn_model(rf, X_test, y_test, "Random Forest")

if __name__ == '__main__':
    main()
