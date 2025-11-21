# train_forecast.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

df = pd.read_csv("telemetry.csv", parse_dates=["timestamp"])

# Create lag features (use last 6 power values)
N_LAGS = 6
for lag in range(1, N_LAGS + 1):
    df[f"lag_{lag}"] = df["solar_power"].shift(lag)

df = df.dropna().reset_index(drop=True)

# Features = lag values + temperature + battery
features = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + ["temperature", "battery_level"]
X = df[features]
y = df["solar_power"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=120,
    random_state=42
)
model.fit(X_train, y_train)

print("Train score:", model.score(X_train, y_train))
print("Test score:", model.score(X_test, y_test))

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/forecast_rf.joblib")

print("✔ Forecast model saved at models/forecast_rf.joblib")
