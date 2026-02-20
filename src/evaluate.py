import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# ==============================
# Load Data
# ==============================
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print("Test data loaded ✅")

# ==============================
# Load Model
# ==============================
model = load_model("models/hybrid_model.h5")
print("Model loaded ✅")

# ==============================
# Predict
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Load Scaler
# ==============================
scaler = joblib.load("data/scaler/scaler.pkl")

# We only need close price scaling range
close_min = scaler.data_min_[0]
close_max = scaler.data_max_[0]

# Inverse scaling
y_test_real = y_test * (close_max - close_min) + close_min
y_pred_real = y_pred.flatten() * (close_max - close_min) + close_min

# ==============================
# Metrics
# ==============================
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

print("\n===== FINAL REAL METRICS =====")
print(f"MAE  (Real Price Error): {mae:.2f}")
print(f"RMSE (Real Price Error): {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ==============================
# Directional Accuracy
# ==============================
direction_true = np.sign(np.diff(y_test_real))
direction_pred = np.sign(np.diff(y_pred_real))

direction_accuracy = np.mean(direction_true == direction_pred) * 100

print(f"Directional Accuracy: {direction_accuracy:.2f}%")

# ==============================
# Plot
# ==============================
plt.figure(figsize=(12,6))
plt.plot(y_test_real[:500], label="Actual")
plt.plot(y_pred_real[:500], label="Predicted")
plt.title("Actual vs Predicted Price (First 500 Samples)")
plt.legend()
plt.show()
