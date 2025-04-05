# AI Disaster Warning System

An intelligent system that predicts potential disasters based on weather conditions using machine learning.

## Features

- Predict disaster likelihood based on rainfall, temperature, humidity, and wind speed
- User authentication system for secure access
- Interactive data visualization dashboard
- Responsive and modern UI

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Access the application at http://localhost:5000

## Usage

1. Register a new account or login
2. Enter weather parameters in the prediction form
3. View disaster prediction results and visualizations
4. Explore historical predictions in your dashboard

## Model Information

The system uses a Random Forest Classifier trained on environmental data to predict disaster likelihood. The model considers:

- Rainfall (mm)
- Temperature (°C)
- Humidity (%)
- Wind Speed (m/s)

## Data Visualization

The dashboard provides visual insights into:
- Feature importance
- Prediction probability
- Historical prediction trends 