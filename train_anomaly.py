# train_anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

# Load historic simulated data
df = pd.read_csv("telemetry.csv", parse_dates=["timestamp"])

# Features for anomaly detection
X = df[["solar_power", "temperature", "battery_level"]].values

# Train model
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,     # 2% anomalies
    random_state=42
)
model.fit(X)

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/anomaly_iforest.joblib")

print("✔ Anomaly model saved at models/anomaly_iforest.joblib")
