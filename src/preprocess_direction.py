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

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load
df = pd.read_csv(INTERIM_PATH)
df.dropna(inplace=True)

features = ["close", "volume", "rsi", "macd", "ema", "volatility"]
data = df[features].values

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

joblib.dump(scaler, SCALER_PATH + "scaler.pkl")

# Create direction labels
close_prices = df["close"].values
direction = np.where(np.diff(close_prices) > 0, 1, 0)

# Since diff reduces one row
scaled_data = scaled_data[:-1]

# Sequence creation
X = []
y = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(direction[i])

X = np.array(X)
y = np.array(y)

print("Total sequences:", X.shape)

# Chronological split
train_size = int(len(X) * TRAIN_SPLIT)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

np.save(PROCESSED_PATH + "X_train_dir.npy", X_train)
np.save(PROCESSED_PATH + "X_test_dir.npy", X_test)
np.save(PROCESSED_PATH + "y_train_dir.npy", y_train)
np.save(PROCESSED_PATH + "y_test_dir.npy", y_test)

print("Direction preprocessing completed ✅")
