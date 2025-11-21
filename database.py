"""
Industrial-grade database schema for Twinergy Digital Twin Platform
Supports multi-asset monitoring, telemetry, alerts, predictions, and maintenance scheduling
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json

DB_PATH = Path("twinergy.db")

class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database schema with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Assets table - represents solar installations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                location TEXT,
                capacity_kw REAL,
                installation_date TEXT,
                status TEXT DEFAULT 'active',
                health_score REAL DEFAULT 100.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Telemetry table - time-series sensor data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                solar_power REAL,
                temperature REAL,
                battery_level REAL,
                voltage REAL,
                current REAL,
                irradiance REAL,
                inverter_efficiency REAL,
                panel_health REAL,
                wind_speed REAL,
                humidity REAL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Alerts table - system alerts and warnings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                resolved_at TEXT,
                metadata TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Predictions table - ML model predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_value REAL,
                confidence REAL,
                horizon_hours INTEGER,
                metadata TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Anomalies table - detected anomalies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                anomaly_score REAL,
                anomaly_type TEXT,
                features TEXT,
                is_failure_prediction BOOLEAN DEFAULT 0,
                rul_days REAL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Health metrics table - computed health scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_health REAL,
                component_health TEXT,
                degradation_rate REAL,
                expected_failure_date TEXT,
                maintenance_priority TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Maintenance schedule table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                scheduled_date TEXT NOT NULL,
                maintenance_type TEXT NOT NULL,
                priority TEXT,
                estimated_duration_hours REAL,
                status TEXT DEFAULT 'pending',
                completed_at TEXT,
                notes TEXT,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                date TEXT NOT NULL,
                daily_energy_kwh REAL,
                efficiency_percent REAL,
                availability_percent REAL,
                peak_power_kw REAL,
                capacity_factor REAL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
            )
        """)
        
        # Create indexes for better query performance (after tables are created)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_asset_time 
            ON telemetry(asset_id, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_asset_resolved 
            ON alerts(asset_id, resolved, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_asset_time 
            ON predictions(asset_id, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_metrics_asset_time 
            ON health_metrics(asset_id, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
        # Initialize default asset if none exists
        self._init_default_asset()
    
    def _init_default_asset(self):
        """Initialize default solar assets"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM assets")
        if cursor.fetchone()[0] == 0:
            default_assets = [
                ("ASSET-001", "Solar Farm Alpha", "Location A", 500.0, "2023-01-15", "active", 100.0),
                ("ASSET-002", "Solar Farm Beta", "Location B", 750.0, "2023-03-20", "active", 100.0),
                ("ASSET-003", "Solar Farm Gamma", "Location C", 1000.0, "2023-06-10", "active", 100.0),
            ]
            cursor.executemany("""
                INSERT INTO assets (asset_id, name, location, capacity_kw, installation_date, status, health_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, default_assets)
        conn.commit()
        conn.close()
    
    def insert_telemetry(self, asset_id: str, data: Dict):
        """Insert telemetry data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry (
                asset_id, timestamp, solar_power, temperature, battery_level,
                voltage, current, irradiance, inverter_efficiency, panel_health,
                wind_speed, humidity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            asset_id,
            data.get("timestamp", datetime.now().isoformat()),
            data.get("solar_power", 0),
            data.get("temperature", 0),
            data.get("battery_level", 0),
            data.get("voltage", 0),
            data.get("current", 0),
            data.get("irradiance", 0),
            data.get("inverter_efficiency", 0),
            data.get("panel_health", 100),
            data.get("wind_speed", 0),
            data.get("humidity", 0)
        ))
        conn.commit()
        conn.close()
    
    def insert_alert(self, asset_id: str, alert_type: str, severity: str, message: str, metadata: Dict = None):
        """Insert alert and return the inserted alert with ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO alerts (asset_id, timestamp, alert_type, severity, message, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            asset_id,
            timestamp,
            alert_type,
            severity,
            message,
            json.dumps(metadata) if metadata else None
        ))
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alert_id
    
    def insert_prediction(self, asset_id: str, prediction_type: str, predicted_value: float, 
                         confidence: float, horizon_hours: int, metadata: Dict = None):
        """Insert prediction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (asset_id, timestamp, prediction_type, predicted_value, confidence, horizon_hours, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            asset_id,
            datetime.now().isoformat(),
            prediction_type,
            predicted_value,
            confidence,
            horizon_hours,
            json.dumps(metadata) if metadata else None
        ))
        conn.commit()
        conn.close()
    
    def insert_anomaly(self, asset_id: str, anomaly_score: float, anomaly_type: str, 
                      features: Dict, is_failure_prediction: bool = False, rul_days: float = None):
        """Insert anomaly detection result"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO anomalies (asset_id, timestamp, anomaly_score, anomaly_type, features, is_failure_prediction, rul_days)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            asset_id,
            datetime.now().isoformat(),
            anomaly_score,
            anomaly_type,
            json.dumps(features),
            is_failure_prediction,
            rul_days
        ))
        conn.commit()
        conn.close()
    
    def get_latest_telemetry(self, asset_id: str, limit: int = 100) -> List[Dict]:
        """Get latest telemetry data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM telemetry 
            WHERE asset_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (asset_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_active_alerts(self, asset_id: str = None) -> List[Dict]:
        """Get active (unresolved) alerts"""
        conn = self.get_connection()
        cursor = conn.cursor()
        if asset_id:
            cursor.execute("""
                SELECT id, asset_id, timestamp, alert_type, severity, message, resolved, metadata
                FROM alerts 
                WHERE asset_id = ? AND resolved = 0 
                ORDER BY timestamp DESC
            """, (asset_id,))
        else:
            cursor.execute("""
                SELECT id, asset_id, timestamp, alert_type, severity, message, resolved, metadata
                FROM alerts 
                WHERE resolved = 0 
                ORDER BY timestamp DESC
            """)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dict and parse metadata JSON
        alerts = []
        for row in rows:
            alert_dict = dict(row)
            # Parse metadata if it's a string
            if alert_dict.get('metadata') and isinstance(alert_dict['metadata'], str):
                try:
                    import json
                    alert_dict['metadata'] = json.loads(alert_dict['metadata'])
                except:
                    pass
            alerts.append(alert_dict)
        return alerts
    
    def get_health_metrics(self, asset_id: str) -> Optional[Dict]:
        """Get latest health metrics for asset"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM health_metrics 
            WHERE asset_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (asset_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_asset_health(self, asset_id: str, health_score: float):
        """Update asset health score"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE assets SET health_score = ? WHERE asset_id = ?
        """, (health_score, asset_id))
        conn.commit()
        conn.close()
    
    def get_asset(self, asset_id: str) -> Optional[Dict]:
        """Get asset information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def insert_health_metrics(self, asset_id: str, health_data: Dict):
        """Insert health metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO health_metrics 
            (asset_id, timestamp, overall_health, component_health, degradation_rate, expected_failure_date, maintenance_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            asset_id,
            datetime.now().isoformat(),
            health_data.get('overall_health', 100),
            json.dumps(health_data.get('component_health', {})),
            health_data.get('degradation_rate', 0),
            health_data.get('expected_failure_date'),
            health_data.get('status', 'healthy')
        ))
        conn.commit()
        conn.close()

