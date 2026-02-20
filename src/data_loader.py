import os
import pandas as pd
from binance.client import Client
from datetime import datetime

# Create Binance client (No API key needed for public data)
client = Client()

# Parameters
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
start_date = "1 Jan 2023"   # Change if needed
end_date = datetime.now().strftime("%d %b %Y")

print("Downloading data from Binance...")

klines = client.get_historical_klines(symbol, interval, start_date, end_date)

print("Download completed!")

# Convert to DataFrame
columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore"
]

df = pd.DataFrame(klines, columns=columns)

# Keep only required columns
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Convert numeric columns
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

# Create directory if not exists
os.makedirs("data/raw", exist_ok=True)

# Save CSV
file_path = "data/raw/btcusdt_1m_raw.csv"
df.to_csv(file_path, index=False)

print(f"File saved at: {file_path}")
print(df.head())
