import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/btcusdt_5m_raw.csv"
INTERIM_PATH = "data/interim/btcusdt_5m_with_indicators.csv"

os.makedirs("data/interim", exist_ok=True)

df = pd.read_csv(RAW_PATH)

# EMA
df["ema"] = df["close"].ewm(span=14, adjust=False).mean()

# RSI
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26

# Volatility
df["volatility"] = df["close"].rolling(window=14).std()

df.dropna(inplace=True)

df.to_csv(INTERIM_PATH, index=False)

print("5-minute indicators added successfully ✅")
print(df.head())
