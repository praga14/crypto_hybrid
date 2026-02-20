import os
import pandas as pd
from binance.client import Client
from datetime import datetime

client = Client()

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE   # <-- 5 minute here
start_date = "1 Jan 2023"
end_date = datetime.now().strftime("%d %b %Y")

print("Downloading 5-minute data...")

klines = client.get_historical_klines(symbol, interval, start_date, end_date)

columns = [
    "timestamp","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]

df = pd.DataFrame(klines, columns=columns)
df = df[["timestamp","open","high","low","close","volume"]]

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

for col in ["open","high","low","close","volume"]:
    df[col] = df[col].astype(float)

os.makedirs("data/raw", exist_ok=True)

df.to_csv("data/raw/btcusdt_5m_raw.csv", index=False)

print("5-minute data saved successfully ✅")
print(df.head())
