import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Paths
INTERIM_PATH = "data/interim/btcusdt_with_indicators.csv"
PROCESSED_PATH = "data/processed/"
SCALER_PATH = "data/scaler/"

SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.8

# Create folders if not exist
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(SCALER_PATH, exist_ok=True)

# Load data
df = pd.read_csv(INTERIM_PATH)
df.dropna(inplace=True)

# Select features
features = ["close", "volume", "rsi", "macd", "ema", "volatility"]
data = df[features].values

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

joblib.dump(scaler, SCALER_PATH + "scaler.pkl")

# Sequence creation
X = []
y = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i][0])  # close price

X = np.array(X)
y = np.array(y)

print("Total sequences:", X.shape)

# Chronological split
train_size = int(len(X) * TRAIN_SPLIT)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Save
np.save(PROCESSED_PATH + "X_train.npy", X_train)
np.save(PROCESSED_PATH + "X_test.npy", X_test)
np.save(PROCESSED_PATH + "y_train.npy", y_train)
np.save(PROCESSED_PATH + "y_test.npy", y_test)

print("Preprocessing completed successfully ✅")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
