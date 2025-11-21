"""
Advanced ML Models for Predictive Maintenance
- LSTM for time series forecasting
- Ensemble anomaly detection
- Remaining Useful Life (RUL) prediction
- Health score computation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using fallback models")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class AdvancedAnomalyDetector:
    """Ensemble anomaly detection using Isolation Forest and statistical methods"""
    
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray):
        """Train anomaly detection model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        joblib.dump(self.model, MODEL_DIR / "anomaly_ensemble.joblib")
        joblib.dump(self.scaler, MODEL_DIR / "anomaly_scaler.joblib")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        if not self.is_trained:
            return np.zeros(len(X)), np.ones(len(X)) * 0.5
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Normalize scores to 0-1 range (lower = more anomalous)
        scores_normalized = 1 / (1 + np.exp(-scores))
        
        return predictions, scores_normalized
    
    def load(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(MODEL_DIR / "anomaly_ensemble.joblib")
            self.scaler = joblib.load(MODEL_DIR / "anomaly_scaler.joblib")
            self.is_trained = True
        except:
            self.is_trained = False

class LSTMForecaster:
    """LSTM-based time series forecasting for power prediction"""
    
    def __init__(self, sequence_length: int = 24, forecast_horizon: int = 6):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
    
    def _create_sequences(self, data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, target_col: str = 'solar_power'):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available, using Random Forest fallback")
            return self._train_fallback(data, target_col)
        
        # Prepare data
        values = data[target_col].values.reshape(-1, 1)
        values_scaled = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = self._create_sequences(values_scaled, self.sequence_length)
        
        if len(X) < 10:
            return self._train_fallback(data, target_col)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        self.model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.is_trained = True
        self.model.save(str(MODEL_DIR / "lstm_forecaster.h5"))
        joblib.dump(self.scaler, MODEL_DIR / "lstm_scaler.joblib")
    
    def _train_fallback(self, data: pd.DataFrame, target_col: str):
        """Fallback to Random Forest if LSTM unavailable"""
        # Create lag features
        for lag in range(1, min(7, len(data))):
            data[f'lag_{lag}'] = data[target_col].shift(lag)
        
        data = data.dropna()
        if len(data) < 5:
            return
        
        features = [f'lag_{i}' for i in range(1, min(7, len(data.columns)))]
        if 'temperature' in data.columns:
            features.append('temperature')
        if 'battery_level' in data.columns:
            features.append('battery_level')
        
        X = data[features].values
        y = data[target_col].values
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        joblib.dump(self.model, MODEL_DIR / "forecast_rf.joblib")
    
    def predict(self, recent_data: List[float], additional_features: Dict = None) -> Tuple[float, float]:
        """Predict future value and confidence"""
        if not self.is_trained or len(recent_data) < self.sequence_length:
            # Simple fallback prediction
            if len(recent_data) > 0:
                trend = np.mean(recent_data[-3:]) if len(recent_data) >= 3 else recent_data[-1]
                return trend * 0.98, 0.6
            return 0.0, 0.0
        
        if isinstance(self.model, RandomForestRegressor):
            # RF fallback
            features = recent_data[-6:] if len(recent_data) >= 6 else recent_data + [recent_data[-1]] * (6 - len(recent_data))
            if additional_features:
                if 'temperature' in additional_features:
                    features.append(additional_features['temperature'])
                if 'battery_level' in additional_features:
                    features.append(additional_features['battery_level'])
            while len(features) < 8:
                features.append(features[-1])
            
            pred = self.model.predict([features[:8]])[0]
            return pred, 0.75
        
        # LSTM prediction
        if TENSORFLOW_AVAILABLE:
            seq = np.array(recent_data[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
            seq_scaled = self.scaler.transform(seq.reshape(-1, 1)).reshape(1, self.sequence_length, 1)
            pred_scaled = self.model.predict(seq_scaled, verbose=0)[0][0]
            pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            confidence = 0.85
            return max(0, pred), confidence
        
        return recent_data[-1] * 0.98, 0.6
    
    def load(self):
        """Load pre-trained model"""
        try:
            if TENSORFLOW_AVAILABLE:
                self.model = keras.models.load_model(MODEL_DIR / "lstm_forecaster.h5")
                self.scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
            else:
                self.model = joblib.load(MODEL_DIR / "forecast_rf.joblib")
            self.is_trained = True
        except:
            self.is_trained = False

class RULPredictor:
    """Remaining Useful Life (RUL) prediction using degradation modeling"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame):
        """Train RUL model based on degradation patterns"""
        # Create degradation features
        if len(historical_data) < 20:
            return
        
        # Calculate degradation indicators
        historical_data = historical_data.sort_values('timestamp')
        historical_data['power_degradation'] = historical_data['solar_power'].rolling(24).mean().pct_change()
        historical_data['temp_trend'] = historical_data['temperature'].rolling(24).mean().diff()
        historical_data['battery_degradation'] = historical_data['battery_level'].rolling(24).mean().diff()
        historical_data['efficiency_trend'] = historical_data.get('inverter_efficiency', 100).rolling(24).mean().diff()
        
        historical_data = historical_data.dropna()
        
        if len(historical_data) < 10:
            return
        
        # Features for RUL prediction
        features = ['solar_power', 'temperature', 'battery_level', 
                   'power_degradation', 'temp_trend', 'battery_degradation']
        if 'inverter_efficiency' in historical_data.columns:
            features.append('inverter_efficiency')
        
        X = historical_data[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Simulate RUL labels (days until failure threshold)
        # In production, this would come from actual failure data
        y = np.random.uniform(30, 365, len(historical_data))  # Placeholder
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        joblib.dump(self.model, MODEL_DIR / "rul_predictor.joblib")
        joblib.dump(self.scaler, MODEL_DIR / "rul_scaler.joblib")
    
    def predict(self, current_data: Dict) -> Tuple[float, float]:
        """Predict RUL in days and confidence"""
        if not self.is_trained:
            # Heuristic fallback
            health_score = self._estimate_health(current_data)
            rul = max(30, health_score * 3.65)  # Rough estimate
            return rul, 0.5
        
        features = np.array([[
            current_data.get('solar_power', 400),
            current_data.get('temperature', 35),
            current_data.get('battery_level', 70),
            current_data.get('power_degradation', 0),
            current_data.get('temp_trend', 0),
            current_data.get('battery_degradation', 0),
            current_data.get('inverter_efficiency', 95)
        ]])
        
        features_scaled = self.scaler.transform(features)
        rul = self.model.predict(features_scaled)[0]
        confidence = 0.8
        
        return max(0, rul), confidence
    
    def _estimate_health(self, data: Dict) -> float:
        """Estimate health score from current data"""
        power = data.get('solar_power', 400) / 500.0 * 100
        temp_health = max(0, 100 - (data.get('temperature', 35) - 30) * 2)
        battery_health = data.get('battery_level', 70)
        
        return (power * 0.4 + temp_health * 0.3 + battery_health * 0.3)
    
    def load(self):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(MODEL_DIR / "rul_predictor.joblib")
            self.scaler = joblib.load(MODEL_DIR / "rul_scaler.joblib")
            self.is_trained = True
        except:
            self.is_trained = False

class HealthScoreCalculator:
    """Calculate comprehensive health scores for assets"""
    
    @staticmethod
    def calculate_health(telemetry: Dict, historical_trend: List[Dict] = None) -> Dict:
        """Calculate overall health score and component health"""
        power = telemetry.get('solar_power', 400)
        temp = telemetry.get('temperature', 35)
        battery = telemetry.get('battery_level', 70)
        efficiency = telemetry.get('inverter_efficiency', 95)
        
        # Component health scores (0-100)
        power_health = min(100, (power / 500.0) * 100) if power > 0 else 0
        temp_health = max(0, 100 - (temp - 30) * 2) if temp > 30 else 100
        battery_health = battery
        efficiency_health = efficiency
        
        # Degradation analysis
        degradation_rate = 0.0
        if historical_trend and len(historical_trend) > 10:
            recent_power = [d.get('solar_power', 400) for d in historical_trend[-10:]]
            older_power = [d.get('solar_power', 400) for d in historical_trend[-20:-10]]
            if older_power:
                degradation_rate = ((np.mean(recent_power) - np.mean(older_power)) / np.mean(older_power)) * 100
        
        # Overall health (weighted average)
        overall_health = (
            power_health * 0.3 +
            temp_health * 0.25 +
            battery_health * 0.25 +
            efficiency_health * 0.2
        )
        
        # Adjust for degradation
        if degradation_rate < -5:  # Significant degradation
            overall_health *= 0.9
        
        overall_health = max(0, min(100, overall_health))
        
        return {
            'overall_health': round(overall_health, 2),
            'component_health': {
                'power': round(power_health, 2),
                'temperature': round(temp_health, 2),
                'battery': round(battery_health, 2),
                'efficiency': round(efficiency_health, 2)
            },
            'degradation_rate': round(degradation_rate, 2),
            'status': 'critical' if overall_health < 50 else 'warning' if overall_health < 70 else 'healthy'
        }

