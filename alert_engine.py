"""
Intelligent Alert Engine for Predictive Maintenance
Analyzes telemetry, predictions, and anomalies to generate actionable alerts
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

class AlertEngine:
    """Advanced alert generation and management system"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict:
        """Initialize alert rules with thresholds and conditions"""
        return {
            "critical_temperature": {
                "threshold": 70,
                "warning": 60,
                "severity": "CRITICAL",
                "type": "temperature_overheat"
            },
            "low_battery": {
                "threshold": 15,
                "warning": 25,
                "severity": "HIGH",
                "type": "battery_critical"
            },
            "power_drop": {
                "threshold": 0.5,  # 50% drop
                "warning": 0.7,    # 30% drop
                "severity": "HIGH",
                "type": "power_anomaly"
            },
            "inverter_failure": {
                "threshold": 75,  # efficiency below 75%
                "warning": 85,
                "severity": "CRITICAL",
                "type": "inverter_degradation"
            },
            "panel_degradation": {
                "threshold": 60,  # health below 60%
                "warning": 75,
                "severity": "MEDIUM",
                "type": "panel_degradation"
            },
            "anomaly_detected": {
                "threshold": 0.3,  # anomaly score
                "warning": 0.5,
                "severity": "MEDIUM",
                "type": "anomaly"
            },
            "predicted_failure": {
                "threshold": 7,  # days until failure
                "warning": 14,
                "severity": "CRITICAL",
                "type": "failure_prediction"
            },
            "voltage_anomaly": {
                "threshold": 50,  # deviation from nominal
                "warning": 30,
                "severity": "HIGH",
                "type": "electrical_anomaly"
            },
        }
    
    def analyze_telemetry(self, telemetry: Dict, historical: List[Dict] = None, 
                         predictions: Dict = None, anomalies: Dict = None) -> List[Dict]:
        """
        Comprehensive analysis of telemetry data to generate alerts
        """
        alerts = []
        asset_id = telemetry.get("asset_id", "ASSET-001")
        timestamp = telemetry.get("timestamp", datetime.now().isoformat())
        
        # Temperature alerts
        temp = telemetry.get("temperature", 0)
        if temp >= self.alert_rules["critical_temperature"]["threshold"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "CRITICAL",
                "critical_temperature",
                f"Critical temperature detected: {temp}°C - Immediate cooling required",
                {"temperature": temp, "threshold": self.alert_rules["critical_temperature"]["threshold"]}
            ))
        elif temp >= self.alert_rules["critical_temperature"]["warning"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "HIGH",
                "temperature_warning",
                f"High temperature warning: {temp}°C - Monitor closely",
                {"temperature": temp}
            ))
        elif temp >= 45:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "LOW",
                "temperature_elevated",
                f"Temperature elevated: {temp}°C - Within normal range but monitor",
                {"temperature": temp}
            ))
        
        # Battery alerts
        battery = telemetry.get("battery_level", 100)
        if battery <= self.alert_rules["low_battery"]["threshold"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "CRITICAL",
                "battery_critical",
                f"Battery critically low: {battery}% - Risk of storage failure",
                {"battery_level": battery, "estimated_hours": battery * 0.5}
            ))
        elif battery <= self.alert_rules["low_battery"]["warning"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "MEDIUM",
                "battery_low",
                f"Battery level low: {battery}% - Schedule maintenance",
                {"battery_level": battery}
            ))
        elif battery <= 40:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "LOW",
                "battery_warning",
                f"Battery level below optimal: {battery}% - Monitor closely",
                {"battery_level": battery}
            ))
        
        # Power drop detection
        if historical and len(historical) >= 10:
            recent_avg = np.mean([h.get("solar_power", 0) for h in historical[-5:]])
            older_avg = np.mean([h.get("solar_power", 0) for h in historical[-10:-5]])
            if older_avg > 0:
                drop_ratio = recent_avg / older_avg
                if drop_ratio <= self.alert_rules["power_drop"]["threshold"]:
                    alerts.append(self._create_alert(
                        asset_id, timestamp,
                        "CRITICAL",
                        "power_drop",
                        f"Severe power drop detected: {((1-drop_ratio)*100):.1f}% reduction - Possible panel/inverter failure",
                        {"drop_ratio": drop_ratio, "recent_avg": recent_avg, "older_avg": older_avg}
                    ))
                elif drop_ratio <= self.alert_rules["power_drop"]["warning"]:
                    alerts.append(self._create_alert(
                        asset_id, timestamp,
                        "MEDIUM",
                        "power_degradation",
                        f"Power degradation detected: {((1-drop_ratio)*100):.1f}% reduction",
                        {"drop_ratio": drop_ratio}
                    ))
                elif drop_ratio <= 0.85:
                    alerts.append(self._create_alert(
                        asset_id, timestamp,
                        "LOW",
                        "power_trend",
                        f"Minor power trend detected: {((1-drop_ratio)*100):.1f}% reduction - Monitor",
                        {"drop_ratio": drop_ratio}
                    ))
        
        # Inverter efficiency
        inverter_eff = telemetry.get("inverter_efficiency", 100)
        if inverter_eff <= self.alert_rules["inverter_failure"]["threshold"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "CRITICAL",
                "inverter_failure",
                f"Inverter efficiency critically low: {inverter_eff}% - Inverter failure imminent",
                {"efficiency": inverter_eff, "estimated_failure_days": max(1, (inverter_eff - 70) * 2)}
            ))
        elif inverter_eff <= self.alert_rules["inverter_failure"]["warning"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "HIGH",
                "inverter_degradation",
                f"Inverter efficiency declining: {inverter_eff}% - Schedule inspection",
                {"efficiency": inverter_eff}
            ))
        elif inverter_eff <= 90:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "LOW",
                "inverter_maintenance",
                f"Inverter efficiency at {inverter_eff}% - Consider routine maintenance",
                {"efficiency": inverter_eff}
            ))
        
        # Panel health
        panel_health = telemetry.get("panel_health", 100)
        if panel_health <= self.alert_rules["panel_degradation"]["threshold"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "HIGH",
                "panel_degradation",
                f"Panel health critically low: {panel_health}% - Replacement recommended",
                {"panel_health": panel_health}
            ))
        elif panel_health <= self.alert_rules["panel_degradation"]["warning"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "MEDIUM",
                "panel_wear",
                f"Panel health declining: {panel_health}% - Monitor degradation",
                {"panel_health": panel_health}
            ))
        elif panel_health <= 85:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "LOW",
                "panel_maintenance",
                f"Panel health at {panel_health}% - Consider preventive maintenance",
                {"panel_health": panel_health}
            ))
        
        # Voltage anomalies
        voltage = telemetry.get("voltage", 400)
        voltage_deviation = abs(voltage - 400)
        if voltage_deviation >= self.alert_rules["voltage_anomaly"]["threshold"]:
            alerts.append(self._create_alert(
                asset_id, timestamp,
                "HIGH",
                "voltage_anomaly",
                f"Voltage anomaly detected: {voltage}V (deviation: {voltage_deviation}V) - Electrical fault possible",
                {"voltage": voltage, "deviation": voltage_deviation}
            ))
        
        # Anomaly detection alerts
        if anomalies:
            anomaly_score = anomalies.get("anomaly_score", 1.0)
            if anomaly_score <= self.alert_rules["anomaly_detected"]["threshold"]:
                alerts.append(self._create_alert(
                    asset_id, timestamp,
                    "HIGH",
                    "anomaly_detected",
                    f"AI-detected anomaly: Unusual system behavior pattern identified",
                    {"anomaly_score": anomaly_score, "anomaly_type": anomalies.get("anomaly_type", "unknown")}
                ))
        
        # Predictive failure alerts
        if predictions:
            rul_days = predictions.get("rul_days")
            if rul_days is not None and rul_days <= self.alert_rules["predicted_failure"]["threshold"]:
                alerts.append(self._create_alert(
                    asset_id, timestamp,
                    "CRITICAL",
                    "failure_prediction",
                    f"PREDICTIVE ALERT: System failure predicted within {rul_days:.1f} days - Immediate action required",
                    {"rul_days": rul_days, "confidence": predictions.get("confidence", 0.8), 
                     "failure_type": predictions.get("failure_type", "general")}
                ))
            elif rul_days is not None and rul_days <= self.alert_rules["predicted_failure"]["warning"]:
                alerts.append(self._create_alert(
                    asset_id, timestamp,
                    "HIGH",
                    "failure_warning",
                    f"Failure warning: System degradation indicates potential failure in {rul_days:.1f} days",
                    {"rul_days": rul_days, "confidence": predictions.get("confidence", 0.8)}
                ))
            
            # Power forecast alerts
            predicted_power = predictions.get("predicted_power")
            current_power = telemetry.get("solar_power", 0)
            if predicted_power and current_power > 0:
                forecast_drop = (current_power - predicted_power) / current_power
                if forecast_drop >= 0.4:  # 40% predicted drop
                    alerts.append(self._create_alert(
                        asset_id, timestamp,
                        "HIGH",
                        "power_forecast",
                        f"Power forecast indicates {forecast_drop*100:.1f}% drop in next 6 hours - Preventive action recommended",
                        {"current_power": current_power, "predicted_power": predicted_power, 
                         "forecast_drop": forecast_drop}
                    ))
        
        return alerts
    
    def _create_alert(self, asset_id: str, timestamp: str, severity: str, 
                     alert_type: str, message: str, metadata: Dict = None) -> Dict:
        """Create standardized alert object"""
        return {
            "asset_id": asset_id,
            "timestamp": timestamp,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {},
            "resolved": False,
            "id": f"{asset_id}-{alert_type}-{timestamp}"
        }
    
    def prioritize_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Sort alerts by priority (severity + recency)"""
        severity_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        def priority_score(alert):
            severity = alert.get("severity", "LOW")
            timestamp = alert.get("timestamp", "")
            try:
                time_delta = (datetime.now() - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds()
                recency_score = max(0, 1 - time_delta / 3600)  # Decay over 1 hour
            except:
                recency_score = 1.0
            return severity_weights.get(severity, 1) * 10 + recency_score
        
        return sorted(alerts, key=priority_score, reverse=True)
    
    def get_failure_probability(self, telemetry: Dict, predictions: Dict = None) -> float:
        """Calculate overall failure probability (0-1)"""
        risk_factors = []
        
        # Temperature risk
        temp = telemetry.get("temperature", 0)
        if temp > 70:
            risk_factors.append(0.4)
        elif temp > 60:
            risk_factors.append(0.2)
        
        # Battery risk
        battery = telemetry.get("battery_level", 100)
        if battery < 15:
            risk_factors.append(0.3)
        elif battery < 25:
            risk_factors.append(0.15)
        
        # Inverter risk
        inverter_eff = telemetry.get("inverter_efficiency", 100)
        if inverter_eff < 75:
            risk_factors.append(0.3)
        elif inverter_eff < 85:
            risk_factors.append(0.15)
        
        # Panel health risk
        panel_health = telemetry.get("panel_health", 100)
        if panel_health < 60:
            risk_factors.append(0.2)
        
        # Prediction-based risk
        if predictions:
            rul_days = predictions.get("rul_days")
            if rul_days and rul_days < 7:
                risk_factors.append(0.5)
            elif rul_days and rul_days < 14:
                risk_factors.append(0.3)
        
        # Combine risk factors (using complementary probability)
        if not risk_factors:
            return 0.05  # Base failure probability
        
        combined_risk = 1 - np.prod([1 - r for r in risk_factors])
        return min(0.95, max(0.05, combined_risk))

