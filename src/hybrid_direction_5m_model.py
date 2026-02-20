import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    Activation, Dropout, Add,
    LSTM, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load 5-minute direction data
X_train = np.load("data/processed/X_train_5m_dir.npy")
X_test = np.load("data/processed/X_test_5m_dir.npy")
y_train = np.load("data/processed/y_train_5m_dir.npy")
y_test = np.load("data/processed/y_test_5m_dir.npy")

print("5-minute direction data loaded ✅")

# ==========================
# TCN BLOCK
# ==========================
def tcn_block(x, filters, kernel_size, dilation_rate):
    prev_x = x
    
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               padding="causal",
               dilation_rate=dilation_rate)(x)
    
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    
    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters, 1, padding="same")(prev_x)
    
    x = Add()([prev_x, x])
    
    return x

# ==========================
# MODEL
# ==========================
input_layer = Input(shape=(60, 6))

x = tcn_block(input_layer, 64, 3, 1)
x = tcn_block(x, 64, 3, 2)
x = tcn_block(x, 64, 3, 4)

x = LSTM(64)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("models", exist_ok=True)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("models/hybrid_direction_5m_model.h5", save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=256,
    callbacks=callbacks
)

print("5-minute Direction Model Training Completed ✅")
