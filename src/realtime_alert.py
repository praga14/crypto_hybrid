import numpy as np
import pandas as pd
import joblib
import time
import os
import requests
from binance.client import Client
from tensorflow.keras.models import load_model
from datetime import datetime

# ======================
# CONFIG
# ======================
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE
sequence_length = 60
threshold = 0.003   # 0.3%
check_interval = 300  # 5 minutes (300 sec)

# Telegram config (fill later)
TELEGRAM_TOKEN = "8594607204:AAE0TMypceJQ-MkuL2jR2MbDAHVyZpesJeQ"
TELEGRAM_CHAT_ID = "1297702845"

# ======================
# Load model & scaler
# ======================
model = load_model("models/hybrid_model.h5")
scaler = joblib.load("data/scaler/scaler.pkl")

print("System Started ✅")

client = Client()

os.makedirs("logs", exist_ok=True)
log_file = "logs/trading_log.csv"

# Create log file if not exists
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        "time", "current_price", "predicted_price",
        "change_percent", "signal"
    ]).to_csv(log_file, index=False)

# ======================
# TELEGRAM FUNCTION
# ======================
def send_telegram(message):
    if TELEGRAM_TOKEN == "" or TELEGRAM_CHAT_ID == "":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

# ======================
# MAIN LOOP
# ======================
while True:
    try:
        klines = client.get_historical_klines(symbol, interval, "2 day ago UTC")

        df = pd.DataFrame(klines, columns=[
            "timestamp","open","high","low","close","volume",
            "close_time","qav","trades","tbbav","tbqav","ignore"
        ])

        df = df[["timestamp","open","high","low","close","volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)

        # Indicators
        df["ema"] = df["close"].ewm(span=14, adjust=False).mean()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26

        df["volatility"] = df["close"].rolling(window=14).std()
        df.dropna(inplace=True)

        features = ["close","volume","rsi","macd","ema","volatility"]
        data = df[features].values
        scaled = scaler.transform(data)

        last_sequence = scaled[-sequence_length:]
        X_input = np.expand_dims(last_sequence, axis=0)

        pred_scaled = model.predict(X_input)[0][0]

        close_min = scaler.data_min_[0]
        close_max = scaler.data_max_[0]

        pred_price = pred_scaled * (close_max - close_min) + close_min
        current_price = df["close"].iloc[-1]

        change = (pred_price - current_price) / current_price

        if change > threshold:
            signal = "BUY"
        elif change < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        now = datetime.now()

        print("\n===== ALERT =====")
        print("Time:", now)
        print("Current:", round(current_price,2))
        print("Predicted:", round(pred_price,2))
        print("Change %:", round(change*100,3))
        print("Signal:", signal)

        # Save log
        new_row = pd.DataFrame([{
            "time": now,
            "current_price": current_price,
            "predicted_price": pred_price,
            "change_percent": change*100,
            "signal": signal
        }])

        # Save log
        new_row.to_csv(log_file, mode="a", header=False, index=False)
         #if signal != "HOLD":
            #send_telegram(message)
        message = f"""
       
🚨 Crypto Alert 🚨
Time: {now}
Current: {current_price}
Predicted: {round(pred_price,2)}
Change: {round(change*100,3)}%
Signal: {signal}
"""

        # Send Telegram for ALL signals (for testing)
        send_telegram(message)

        time.sleep(check_interval)

    except Exception as e:
        print("Error:", e)
        time.sleep(check_interval)

