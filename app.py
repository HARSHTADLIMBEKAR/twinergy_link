"""
Twinergy - Industrial-Grade AI-Powered Predictive Maintenance Platform
Backend API Server with Digital Twin, ML Models, and Alert Engine
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time
from digital_twin import generate_twin_data, set_scenario, initialize_asset, get_asset_state, simulate_failure_progression
from ml_models import AdvancedAnomalyDetector, LSTMForecaster, RULPredictor, HealthScoreCalculator
from alert_engine import AlertEngine
from database import Database
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Initialize components
db = Database()
alert_engine = AlertEngine()
anomaly_detector = AdvancedAnomalyDetector()
forecaster = LSTMForecaster()
rul_predictor = RULPredictor()
health_calculator = HealthScoreCalculator()

# Load models if available
anomaly_detector.load()
forecaster.load()
rul_predictor.load()

# Asset management
ASSETS = ["ASSET-001", "ASSET-002", "ASSET-003"]
for asset_id in ASSETS:
    initialize_asset(asset_id, capacity_kw=500.0 if asset_id == "ASSET-001" else 750.0 if asset_id == "ASSET-002" else 1000.0)

# Background data collection thread
def background_data_collection():
    """Continuously collect and process telemetry data"""
    while True:
        try:
            for asset_id in ASSETS:
                data = generate_twin_data(asset_id)
                db.insert_telemetry(asset_id, data)
                
                # Run ML analysis
                historical = db.get_latest_telemetry(asset_id, limit=100)
                if len(historical) >= 10:
                    # Anomaly detection
                    features = np.array([[d.get('solar_power', 0), d.get('temperature', 0), d.get('battery_level', 0)] 
                                       for d in historical[-10:]])
                    if len(features) > 0:
                        predictions, scores = anomaly_detector.predict(features)
                        if len(scores) > 0:
                            anomaly_score = float(scores[-1])
                            is_anomaly = int(predictions[-1]) == -1
                            if is_anomaly or anomaly_score < 0.3:
                                db.insert_anomaly(asset_id, anomaly_score, "system_anomaly", 
                                                 {"solar_power": data.get('solar_power'), 
                                                  "temperature": data.get('temperature'),
                                                  "battery_level": data.get('battery_level')})
                    
                    # Health metrics
                    health = health_calculator.calculate_health(data, historical)
                    db.update_asset_health(asset_id, health['overall_health'])
                    
                    # Generate alerts
                    predictions_dict = {}
                    if len(historical) >= 24:
                        power_history = [h.get('solar_power', 0) for h in historical[-24:]]
                        pred_power, conf = forecaster.predict(power_history, 
                                                             {"temperature": data.get('temperature'),
                                                              "battery_level": data.get('battery_level')})
                        predictions_dict['predicted_power'] = pred_power
                        
                        rul_days, rul_conf = rul_predictor.predict(data)
                        predictions_dict['rul_days'] = rul_days
                        predictions_dict['confidence'] = rul_conf
                    
                    alerts = alert_engine.analyze_telemetry(data, historical, predictions_dict, 
                                                          {"anomaly_score": anomaly_score if 'anomaly_score' in locals() else 1.0})
                    for alert in alerts:
                        alert_id = db.insert_alert(alert['asset_id'], alert['alert_type'], alert['severity'], 
                                                  alert['message'], alert.get('metadata'))
                        # Store the database ID in the alert for reference
                        alert['db_id'] = alert_id
        except Exception as e:
            print(f"Background collection error: {e}")
        time.sleep(5)  # Collect every 5 seconds

# Start background thread
bg_thread = threading.Thread(target=background_data_collection, daemon=True)
bg_thread.start()

@app.route("/")
def home():
    return jsonify({
        "msg": "Twinergy Industrial-Grade Predictive Maintenance Platform",
        "version": "2.0.0",
        "status": "operational",
        "assets": len(ASSETS),
        "endpoints": ["/twin-data", "/assets", "/alerts", "/predictions", "/health", "/maintenance"]
    })

@app.route("/assets")
def get_assets():
    """Get all assets with their current status"""
    assets_data = []
    for asset_id in ASSETS:
        asset = db.get_asset(asset_id)
        if asset:
            latest_telemetry = db.get_latest_telemetry(asset_id, limit=1)
            health_metrics = db.get_health_metrics(asset_id)
            asset_state = get_asset_state(asset_id)
            
            assets_data.append({
                **asset,
                "current_telemetry": latest_telemetry[0] if latest_telemetry else None,
                "health_metrics": health_metrics,
                "asset_state": asset_state
            })
    return jsonify(assets_data)

@app.route("/twin-data")
def twin_data():
    """Get latest digital twin data for primary asset"""
    asset_id = request.args.get("asset_id", "ASSET-001")
    data = generate_twin_data(asset_id)
    
    # Get historical data for analysis
    historical = db.get_latest_telemetry(asset_id, limit=50)
    
    # Anomaly detection
    is_anom = False
    score = 1.0
    if len(historical) >= 10:
        features = np.array([[d.get('solar_power', 0), d.get('temperature', 0), d.get('battery_level', 0)] 
                           for d in historical[-10:]])
        if len(features) > 0:
            predictions, scores = anomaly_detector.predict(features)
            if len(scores) > 0:
                score = float(scores[-1])
                is_anom = int(predictions[-1]) == -1
    
    # Health score
    health = health_calculator.calculate_health(data, historical)
    
    # Predictions
    predictions_dict = {}
    if len(historical) >= 24:
        power_history = [h.get('solar_power', 0) for h in historical[-24:]]
        pred_power, conf = forecaster.predict(power_history, 
                                             {"temperature": data.get('temperature'),
                                              "battery_level": data.get('battery_level')})
        predictions_dict['predicted_power'] = pred_power
        predictions_dict['confidence'] = conf
        
        rul_days, rul_conf = rul_predictor.predict(data)
        predictions_dict['rul_days'] = rul_days
        predictions_dict['rul_confidence'] = rul_conf
    
    resp = data.copy()
    resp["is_anomaly"] = is_anom
    resp["anomaly_score"] = score
    resp["health"] = health
    resp["predictions"] = predictions_dict
    return jsonify(resp)

@app.route("/set-scenario", methods=["POST"])
def update_scenario():
    """Set failure scenario for testing"""
    body = request.json or {}
    s = body.get("scenario", "normal")
    asset_id = body.get("asset_id", "ASSET-001")
    set_scenario(s, asset_id)
    return jsonify({"msg": f"Scenario set to {s} for {asset_id}"})

@app.route("/anomaly")
def anomaly():
    """Legacy endpoint for backward compatibility"""
    asset_id = request.args.get("asset_id", "ASSET-001")
    data = generate_twin_data(asset_id)
    historical = db.get_latest_telemetry(asset_id, limit=10)
    
    is_anom = False
    score = 1.0
    if len(historical) >= 10:
        features = np.array([[d.get('solar_power', 0), d.get('temperature', 0), d.get('battery_level', 0)] 
                           for d in historical[-10:]])
        if len(features) > 0:
            predictions, scores = anomaly_detector.predict(features)
            if len(scores) > 0:
                score = float(scores[-1])
                is_anom = int(predictions[-1]) == -1
    
    alerts = []
    if is_anom or score < 0.3:
        alerts.append({
            "msg": "AI anomaly detected: abnormal system behaviour", 
            "severity": "HIGH"
        })
    
    return jsonify({
        "sample": data, 
        "is_anomaly": is_anom, 
        "anomaly_score": score,
        "alerts": alerts
    })

@app.route("/alerts")
def get_alerts():
    """Get active alerts for assets"""
    asset_id = request.args.get("asset_id")
    alerts = db.get_active_alerts(asset_id)
    prioritized = alert_engine.prioritize_alerts(alerts)
    return jsonify(prioritized)

@app.route("/alerts/<int:alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    """Mark alert as resolved"""
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE alerts SET resolved = 1, resolved_at = ? WHERE id = ?
    """, (datetime.now().isoformat(), alert_id))
    conn.commit()
    conn.close()
    return jsonify({"msg": "Alert resolved"})

@app.route("/predictions")
def get_predictions():
    """Get ML predictions for assets"""
    asset_id = request.args.get("asset_id", "ASSET-001")
    historical = db.get_latest_telemetry(asset_id, limit=100)
    
    if len(historical) < 10:
        return jsonify({"error": "Insufficient data"})
    
    latest = historical[0]
    
    # Power forecast
    power_history = [h.get('solar_power', 0) for h in historical[-24:]]
    pred_power, conf = forecaster.predict(power_history, 
                                         {"temperature": latest.get('temperature'),
                                          "battery_level": latest.get('battery_level')})
    
    # RUL prediction
    rul_days, rul_conf = rul_predictor.predict(latest)
    
    # Health score
    health = health_calculator.calculate_health(latest, historical)
    
    # Failure probability
    failure_prob = alert_engine.get_failure_probability(latest, {"rul_days": rul_days})
    
    return jsonify({
        "power_forecast": {
            "predicted_power": pred_power,
            "confidence": conf,
            "horizon_hours": 6
        },
        "rul": {
            "days_until_failure": rul_days,
            "confidence": rul_conf
        },
        "health": health,
        "failure_probability": failure_prob
    })

@app.route("/health/<asset_id>")
def get_health(asset_id):
    """Get comprehensive health metrics for asset"""
    historical = db.get_latest_telemetry(asset_id, limit=100)
    if not historical:
        return jsonify({"error": "No data available"})
    
    latest = historical[0]
    health = health_calculator.calculate_health(latest, historical)
    health_metrics = db.get_health_metrics(asset_id)
    
    return jsonify({
        "current_health": health,
        "historical_metrics": health_metrics,
        "asset_state": get_asset_state(asset_id)
    })

@app.route("/telemetry/<asset_id>")
def get_telemetry(asset_id):
    """Get telemetry history for asset"""
    limit = int(request.args.get("limit", 100))
    hours = request.args.get("hours")
    
    if hours:
        since = datetime.now() - timedelta(hours=int(hours))
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM telemetry 
            WHERE asset_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (asset_id, since.isoformat()))
        rows = cursor.fetchall()
        conn.close()
        return jsonify([dict(row) for row in rows])
    
    telemetry = db.get_latest_telemetry(asset_id, limit=limit)
    return jsonify(telemetry)

@app.route("/maintenance/schedule", methods=["POST"])
def schedule_maintenance():
    """Schedule maintenance for asset"""
    body = request.json or {}
    asset_id = body.get("asset_id")
    maintenance_type = body.get("type", "preventive")
    priority = body.get("priority", "medium")
    scheduled_date = body.get("scheduled_date")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO maintenance_schedule 
        (asset_id, scheduled_date, maintenance_type, priority, status)
        VALUES (?, ?, ?, ?, 'pending')
    """, (asset_id, scheduled_date, maintenance_type, priority))
    conn.commit()
    conn.close()
    return jsonify({"msg": "Maintenance scheduled"})

@app.route("/maintenance/schedule")
def get_maintenance_schedule():
    """Get maintenance schedule"""
    asset_id = request.args.get("asset_id")
    conn = db.get_connection()
    cursor = conn.cursor()
    if asset_id:
        cursor.execute("""
            SELECT * FROM maintenance_schedule 
            WHERE asset_id = ? AND status = 'pending'
            ORDER BY scheduled_date ASC
        """, (asset_id,))
    else:
        cursor.execute("""
            SELECT * FROM maintenance_schedule 
            WHERE status = 'pending'
            ORDER BY scheduled_date ASC
        """)
    rows = cursor.fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route("/failure-progression/<asset_id>")
def get_failure_progression(asset_id):
    """Simulate failure progression for predictive analysis"""
    days = int(request.args.get("days", 30))
    progression = simulate_failure_progression(asset_id, days_ahead=days)
    return jsonify(progression)

@app.route("/predictive-alert", methods=["POST"])
def predictive_alert():
    """Legacy endpoint for backward compatibility"""
    body = request.json or {}
    asset_id = body.get("asset_id", "ASSET-001")
    last6 = body.get("last6", [])
    temp = body.get("temperature", 0)
    battery = body.get("battery_level", 0)
    current_power = body.get("current_power", 0)
    
    if len(last6) >= 1:
        pred_power, conf = forecaster.predict(last6, {"temperature": temp, "battery_level": battery})
    else:
        pred_power = current_power * 0.98
        conf = 0.6
    
    # Generate alerts using alert engine
    telemetry = {"solar_power": current_power, "temperature": temp, "battery_level": battery, "asset_id": asset_id}
    predictions = {"predicted_power": pred_power}
    alerts = alert_engine.analyze_telemetry(telemetry, None, predictions, None)
    
    return jsonify({
        "predicted_power": pred_power,
        "confidence": conf,
        "alerts": [{"msg": a["message"], "severity": a["severity"]} for a in alerts]
    })

if __name__ == "__main__":
    print("=" * 60)
    print("Twinergy Industrial-Grade Predictive Maintenance Platform")
    print("Backend API Server Starting...")
    print(f"Monitoring {len(ASSETS)} assets")
    print("=" * 60)
    app.run(port=5000, debug=True, threaded=True)
