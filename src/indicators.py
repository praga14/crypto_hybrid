import os
import pandas as pd
import numpy as np

# Paths
RAW_PATH = "data/raw/btcusdt_1m_raw.csv"
INTERIM_PATH = "data/interim/btcusdt_with_indicators.csv"

# Create folder if not exists
os.makedirs("data/interim", exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_PATH)

# ==============================
# EMA
# ==============================
df["ema"] = df["close"].ewm(span=14, adjust=False).mean()

# ==============================
# RSI
# ==============================
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

# ==============================
# MACD
# ==============================
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26

# ==============================
# Volatility (Rolling Std)
# ==============================
df["volatility"] = df["close"].rolling(window=14).std()

# Drop NaN
df.dropna(inplace=True)

# Save file
df.to_csv(INTERIM_PATH, index=False)

print("Indicators added successfully ✅")
print("File saved at:", INTERIM_PATH)
print(df.head())
