"""
model.py — Kiến trúc LSTM 3 tầng cho IDS
==========================================
Tái hiện đúng Table 2 của Anwar et al. (2025):
  Layer 1: LSTM(128, return_sequences=True)  + Dropout(0.2)
  Layer 2: LSTM(64,  return_sequences=True)  ← TẦNG NÀY SKELETON THIẾU
  Layer 3: LSTM(64,  return_sequences=False) + Dropout(0.2)
  Dense(64, relu) + Dropout(0.2)
  Output: Dense(1, sigmoid) hoặc Dense(num_classes, softmax)

Fix skeleton bugs:
  ✓ Thêm Layer 2 LSTM(64) bị thiếu → đủ 3 tầng
  ✓ Xử lý đúng binary vs multiclass output
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_lstm_model(input_shape, num_classes):
    """
    Tạo model LSTM 3 tầng theo kiến trúc bài báo.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features), ví dụ: (1, 23) cho WSN-DS.
    num_classes : int
        Số lớp phân loại. 2 = binary, >2 = multiclass.

    Returns
    -------
    model : tf.keras.Model
        Compiled LSTM model.
    """
    model = Sequential(name="FL_LSTM_IDS")

    # Layer 1: LSTM 128 units — học low-level temporal patterns
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True,
                   name="lstm_layer1"))
    model.add(Dropout(0.2, name="dropout_1"))

    # Layer 2: LSTM 64 units — học mid-level patterns
    # ⚠️ SKELETON GỐC THIẾU TẦNG NÀY
    model.add(LSTM(64, return_sequences=True, name="lstm_layer2"))

    # Layer 3: LSTM 64 units — tổng hợp toàn chuỗi → vector 64D
    model.add(LSTM(64, return_sequences=False, name="lstm_layer3"))
    model.add(Dropout(0.2, name="dropout_2"))

    # Dense Layer — phi tuyến hóa
    model.add(Dense(64, activation='relu', name="dense_hidden"))
    model.add(Dropout(0.2, name="dropout_3"))

    # Output Layer — binary hoặc multiclass
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid', name="output_binary"))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax', name="output_multiclass"))
        loss = 'categorical_crossentropy'

    model.compile(
        loss=loss,
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def clone_model_with_weights(model):
    """
    Clone model architecture + copy weights.
    Dùng trong FL để tạo local model cho mỗi client.
    """
    cloned = tf.keras.models.clone_model(model)
    cloned.set_weights(model.get_weights())
    # Compile lại vì clone_model không giữ compile config
    cloned.compile(
        loss=model.loss,
        optimizer='adam',
        metrics=['accuracy']
    )
    return cloned
