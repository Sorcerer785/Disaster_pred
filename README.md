# AI Disaster Warning System

An intelligent system that predicts potential disasters based on comprehensive environmental data using advanced machine learning.

## Enhanced Features

- **Multi-dimensional Data Analysis**
  - Predictions based on rainfall, temperature, humidity, wind speed, soil moisture, and air pressure
  - Integration with OpenWeather API for real-time weather data
  - Potential for satellite imagery integration

- **Advanced Disaster Intelligence**
  - Multi-class classification predicting specific disaster types (Flood, Storm, Heatwave)
  - Severity assessment (Low, Medium, High) for better preparation
  - Foundation for time-series forecasting for future risk prediction

- **Interactive Map Dashboard**
  - Visualization of disaster risk areas with Leaflet.js
  - Color-coded markers based on disaster type and severity
  - Detailed information available via interactive popups

- **Real-time Monitoring**
  - User-friendly dashboards with detailed visualizations
  - Comprehensive data insights with Plotly graphs
  - Historical prediction tracking and analysis

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Initialize the database:
   ```
   python init_db.py
   ```
4. Train the prediction models:
   ```
   python train_model.py
   ```
5. Run the application:
   ```
   python app.py
   ```
6. Access the application at http://localhost:5000

## Usage

1. Register a new account or login
2. Enter city name to fetch real-time weather data or input custom parameters
3. View detailed disaster predictions including type and severity
4. Explore the map view for geographic visualization
5. Monitor historical predictions and trends

## API Integration

The system provides a RESTful API for integration with other systems:

```
POST /api/predict
```

Example request:
```json
{
  "rainfall": 250.0,
  "temperature": 30.0,
  "humidity": 85.0,
  "wind_speed": 15.0,
  "soil_moisture": 80.0,
  "air_pressure": 990.0
}
```

Example response:
```json
{
  "result": "Disaster Likely",
  "probability": [0.2, 0.8],
  "disaster_type": "Flood",
  "severity": "Medium",
  "features": {
    "rainfall": 250.0,
    "temperature": 30.0,
    "humidity": 85.0,
    "wind_speed": 15.0,
    "soil_moisture": 80.0,
    "air_pressure": 990.0
  }
}
```

## Model Information

The system uses three Random Forest Classifier models:

1. **Binary Disaster Prediction** - Predicts likelihood of any disaster
2. **Disaster Type Classification** - Identifies specific disaster types
3. **Severity Assessment** - Evaluates the potential impact level

## Future Development Roadmap

- Time-series forecasting using LSTM or Prophet for advance warnings
- Addition of satellite imagery analysis for enhanced accuracy
- SMS/Email alert notifications for high-risk scenarios
- Blockchain integration for transparent disaster relief fund management
- Community feedback loop for continuous model improvement

## Data Visualization

The dashboard provides visual insights into:
- Feature importance
- Prediction probability
- Historical prediction trends 