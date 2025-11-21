# Twinergy Architecture Documentation

## System Overview

Twinergy is built as a modern, scalable predictive maintenance platform with a clear separation between frontend and backend components.

## Technology Stack

### Backend
- **Framework**: Flask (Python)
- **Database**: SQLite (development), PostgreSQL (production-ready)
- **ML Libraries**: scikit-learn, TensorFlow/Keras, pandas, numpy
- **Data Processing**: pandas, numpy, scipy

### Frontend
- **Framework**: React 19.2
- **Visualization**: Recharts, Three.js (via @react-three/fiber)
- **HTTP Client**: Axios
- **Styling**: CSS3 with modern gradients and animations

## Component Architecture

### 1. Digital Twin Engine (`digital_twin.py`)

**Purpose**: Simulates realistic solar asset behavior with physics-based modeling

**Key Features**:
- Time-based solar irradiance modeling
- Multiple failure scenarios
- Progressive degradation simulation
- Multi-asset support

**Core Functions**:
- `generate_twin_data()`: Generate telemetry for an asset
- `set_scenario()`: Set failure scenario for testing
- `simulate_failure_progression()`: Forward-looking failure simulation

### 2. ML Models (`ml_models.py`)

#### AdvancedAnomalyDetector
- **Algorithm**: Isolation Forest (Ensemble)
- **Purpose**: Detect anomalous system behavior
- **Features**: StandardScaler normalization, ensemble approach

#### LSTMForecaster
- **Algorithm**: LSTM (with Random Forest fallback)
- **Purpose**: Predict future power output
- **Horizon**: 6 hours ahead
- **Fallback**: Random Forest if TensorFlow unavailable

#### RULPredictor
- **Algorithm**: Gradient Boosting Regressor
- **Purpose**: Predict Remaining Useful Life
- **Output**: Days until failure with confidence

#### HealthScoreCalculator
- **Method**: Weighted component health + degradation analysis
- **Components**: Power, temperature, battery, inverter efficiency
- **Output**: Overall health (0-100) + component breakdown

### 3. Alert Engine (`alert_engine.py`)

**Purpose**: Intelligent alert generation and management

**Alert Types**:
- Critical temperature
- Low battery
- Power drop
- Inverter failure
- Panel degradation
- Voltage anomalies
- AI-detected anomalies
- Predictive failure alerts

**Severity Levels**:
- CRITICAL: Immediate action required
- HIGH: Urgent attention needed
- MEDIUM: Monitor closely
- LOW: Informational

### 4. Database Layer (`database.py`)

**Schema**:
- `assets`: Asset metadata and status
- `telemetry`: Time-series sensor data
- `alerts`: System alerts and warnings
- `predictions`: ML model predictions
- `anomalies`: Detected anomalies
- `health_metrics`: Computed health scores
- `maintenance_schedule`: Maintenance planning
- `performance_metrics`: Daily performance stats

**Indexes**: Optimized for time-series queries

### 5. API Server (`app.py`)

**Endpoints**:
- `/`: System status
- `/assets`: Asset management
- `/twin-data`: Digital twin data
- `/alerts`: Alert management
- `/predictions`: ML predictions
- `/health/<asset_id>`: Health metrics
- `/telemetry/<asset_id>`: Historical data
- `/maintenance/schedule`: Maintenance planning
- `/failure-progression/<asset_id>`: Failure simulation

**Background Processing**:
- Continuous telemetry collection (5-second intervals)
- Real-time ML inference
- Automatic alert generation
- Health score updates

## Data Flow

```
1. Digital Twin generates telemetry
   ↓
2. Data stored in database
   ↓
3. ML models analyze data
   ↓
4. Alert engine evaluates conditions
   ↓
5. Alerts stored and sent to frontend
   ↓
6. Frontend displays real-time updates
```

## Frontend Architecture

### Component Structure

```
App.js (Main Container)
├── OverviewMini (Dashboard)
├── SolarTwin (3D Visualization)
├── LineGraph (Trends)
├── PredictiveTab (Analytics)
└── Alerts (Alert Management)
```

### State Management

- **Local State**: Component-level state with React hooks
- **API Calls**: Axios for HTTP requests
- **Real-time Updates**: 2-second polling interval

## Security Considerations

1. **API Security**: Add authentication middleware in production
2. **Database**: Use parameterized queries (already implemented)
3. **Input Validation**: Validate all API inputs
4. **CORS**: Configured for development, restrict in production
5. **Environment Variables**: Use .env for sensitive config

## Scalability

### Current Limitations
- SQLite database (single-writer)
- In-memory ML models
- Single-threaded Flask (development mode)

### Production Recommendations
1. **Database**: Migrate to PostgreSQL
2. **Caching**: Add Redis for frequently accessed data
3. **Message Queue**: Use RabbitMQ/Celery for background tasks
4. **Load Balancing**: Use nginx or similar
5. **Model Serving**: Consider TensorFlow Serving for ML models
6. **Monitoring**: Add Prometheus/Grafana
7. **Logging**: Structured logging with ELK stack

## Performance Optimizations

1. **Database Indexes**: Already implemented on key columns
2. **Query Optimization**: Limit results, use pagination
3. **Frontend**: React.memo for expensive components
4. **API Caching**: Cache predictions for short periods
5. **Background Processing**: Separate thread for data collection

## Deployment Architecture

### Development
```
Frontend (React Dev Server) → Backend (Flask Dev Server) → SQLite
```

### Production (Recommended)
```
Nginx → React Build
    ↓
Flask (Gunicorn) → PostgreSQL
    ↓
Redis (Cache)
    ↓
Celery Workers (Background Tasks)
```

## Monitoring & Observability

### Metrics to Track
- API response times
- ML inference latency
- Database query performance
- Alert generation rate
- System health scores
- Prediction accuracy

### Logging
- Application logs
- Error tracking
- Performance metrics
- User actions

## Future Enhancements

1. **Real-time WebSockets**: Replace polling with WebSocket connections
2. **Advanced ML**: Transformer models for time-series
3. **Multi-tenancy**: Support for multiple organizations
4. **Mobile App**: React Native mobile application
5. **Edge Computing**: Deploy models on edge devices
6. **Integration APIs**: Connect with SCADA systems
7. **Advanced Analytics**: Machine learning model explainability
8. **Automated Actions**: Trigger maintenance workflows

## Development Workflow

1. **Backend Development**
   - Write tests for new features
   - Update API documentation
   - Test with different scenarios

2. **Frontend Development**
   - Component-based development
   - Responsive design testing
   - Cross-browser compatibility

3. **Integration Testing**
   - Test API endpoints
   - Verify data flow
   - Test failure scenarios

4. **Deployment**
   - Build frontend
   - Run database migrations
   - Deploy backend
   - Configure reverse proxy

---

**Last Updated**: 2024
**Version**: 2.0.0


