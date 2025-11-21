# persist_telemetry.py
import csv
from pathlib import Path
from digital_twin import generate_twin_data
from datetime import datetime

OUT = Path("telemetry.csv")

# Create file with header if it does not exist
if not OUT.exists():
    with OUT.open("w", newline="") as f:
        csv.writer(f).writerow(["timestamp","solar_power","temperature","battery_level"])

def append_sample():
    d = generate_twin_data()
    row = [datetime.now().isoformat(), d["solar_power"], d["temperature"], d["battery_level"]]
    with OUT.open("a", newline="") as f:
        csv.writer(f).writerow(row)

# Generate dataset (500–2000 rows)
if __name__ == "__main__":
    for _ in range(1000):
        append_sample()

    print("telemetry.csv created with 1000 rows!")
