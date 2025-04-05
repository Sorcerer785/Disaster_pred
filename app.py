# app.py
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import json
import os
import requests
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, Prediction
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Load the trained models
with open("disaster_model.pkl", "rb") as f:
    disaster_model = pickle.load(f)

# Load disaster type model if exists
type_model = None
if os.path.exists("disaster_type_model.pkl"):
    with open("disaster_type_model.pkl", "rb") as f:
        type_model = pickle.load(f)

# Load severity model if exists
severity_model = None
if os.path.exists("severity_model.pkl"):
    with open("severity_model.pkl", "rb") as f:
        severity_model = pickle.load(f)

# Function to fetch weather data from OpenWeather API
def fetch_weather_data(city):
    # Using a sample API key - for a production app, use environment variables
    api_key = "4d8fb5b93d4af21d66a2948710284366"  # Sample OpenWeather API key
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # For Celsius
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            print(f"Weather data received: {data}")  # Debug info
            
            # Extract relevant weather data with better error handling
            rainfall = 0  # Default value
            
            # Handle rainfall data which might not be present
            if "rain" in data:
                rainfall = data["rain"].get("1h", 0) if isinstance(data["rain"], dict) else 0
            
            # Create weather data dictionary with safe fallbacks
            weather_data = {
                "temperature": data["main"].get("temp", 20),
                "humidity": data["main"].get("humidity", 50),
                "wind_speed": data["wind"].get("speed", 5),
                "air_pressure": data["main"].get("pressure", 1013),
                "soil_moisture": 50,  # Default value as this isn't provided by OpenWeather
                "rainfall": rainfall
            }
            
            return weather_data
        else:
            print(f"Error from OpenWeather API: {data}")  # Debug info
            flash(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('All fields are required!')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match!')
            return render_template('register.html')
            
        existing_user = User.get_by_username(username)
        if existing_user:
            flash('Username already exists!')
            return render_template('register.html')
            
        User.create(username, email, password)
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
        
    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.get_by_username(username)
        if user and user.verify_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password!')
            
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Dashboard page
@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.get_user_predictions(current_user.id)
    
    # Create visualization data
    if predictions:
        df = pd.DataFrame([
            {
                'Timestamp': p.timestamp,
                'Rainfall': p.rainfall,
                'Temperature': p.temperature,
                'Humidity': p.humidity,
                'Wind Speed': p.wind_speed,
                'Soil Moisture': p.soil_moisture if p.soil_moisture is not None else 0,
                'Air Pressure': p.air_pressure if p.air_pressure is not None else 0,
                'Result': p.result,
                'Disaster Type': p.disaster_type if p.disaster_type else 'None',
                'Severity': p.severity if p.severity else 'Low'
            } for p in predictions
        ])
        
        # Fix timestamps for better display
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        except:
            pass  # Keep original format if conversion fails
        
        # Get properly sized feature importance array
        model_features = disaster_model.n_features_in_
        feature_names = ['Rainfall', 'Temperature', 'Humidity', 'Wind Speed']
        if model_features > 4:
            feature_names.extend(['Soil Moisture', 'Air Pressure'])
        
        # Ensure we only use the actual number of features the model was trained with
        importances_values = disaster_model.feature_importances_
        if len(importances_values) < len(feature_names):
            feature_names = feature_names[:len(importances_values)]
        elif len(importances_values) > len(feature_names):
            importances_values = importances_values[:len(feature_names)]
        
        # Feature importance
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances_values
        }).sort_values('Importance', ascending=False)
        
        # Feature importance chart
        feature_fig = px.bar(
            importances, 
            x='Feature', 
            y='Importance',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        feature_fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            font=dict(size=14),
            height=450,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        feature_fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
        
        # Recent predictions
        recent_fig = px.scatter(
            df,
            x='Timestamp', 
            y='Rainfall',
            color='Result',
            size='Wind Speed',
            hover_data=['Temperature', 'Humidity', 'Soil Moisture', 'Air Pressure', 'Disaster Type', 'Severity'],
            title='Recent Predictions',
            color_discrete_map={"Disaster Likely": "#dc3545", "No Disaster Predicted": "#28a745"}
        )
        recent_fig.update_layout(
            xaxis_title="Time", 
            yaxis_title="Rainfall (mm)",
            font=dict(size=14),
            height=450,
            margin=dict(l=40, r=40, t=60, b=60),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        recent_fig.update_xaxes(tickangle=45)
        
        # Initialize graphs dictionary
        graphs = {
            'feature_importance': json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'recent_predictions': json.dumps(recent_fig, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Create chart of disaster types only if we have disaster predictions
        disaster_preds = df[df['Result'].str.contains('Likely')]
        if not disaster_preds.empty and 'Disaster Type' in df.columns:
            # Remove 'None' disaster type for better visualization
            disaster_preds = disaster_preds[disaster_preds['Disaster Type'] != 'None']
            
            if not disaster_preds.empty:
                type_counts = disaster_preds['Disaster Type'].value_counts().reset_index()
                type_counts.columns = ['Disaster Type', 'Count']
                
                # Only create chart if we have actual disaster types
                if not type_counts.empty:
                    type_fig = px.pie(
                        type_counts, 
                        values='Count', 
                        names='Disaster Type',
                        title='Disaster Type Distribution',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        hole=0.3
                    )
                    type_fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label', 
                        textfont_size=14,
                        insidetextorientation='radial'
                    )
                    type_fig.update_layout(
                        font=dict(size=14),
                        height=450,
                        margin=dict(l=40, r=40, t=60, b=40),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    graphs['disaster_types'] = json.dumps(type_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create severity distribution only if we have severity data
        if not disaster_preds.empty and 'Severity' in df.columns:
            severity_order = ['Low', 'Medium', 'High']
            severity_counts = disaster_preds['Severity'].value_counts().reset_index()
            severity_counts.columns = ['Severity', 'Count']
            
            # Only create chart if we have actual severity data
            if not severity_counts.empty:
                # Ensure proper order of severity levels
                severity_counts['Severity'] = pd.Categorical(
                    severity_counts['Severity'],
                    categories=severity_order,
                    ordered=True
                )
                severity_counts = severity_counts.sort_values('Severity')
                
                severity_fig = px.bar(
                    severity_counts,
                    x='Severity',
                    y='Count',
                    title='Severity Distribution',
                    color='Severity',
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                    text='Count'
                )
                severity_fig.update_layout(
                    xaxis_title="Severity Level", 
                    yaxis_title="Number of Predictions",
                    font=dict(size=14),
                    height=450,
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor='rgba(240, 240, 240, 0.5)'
                )
                severity_fig.update_traces(textposition='outside')
                graphs['severity'] = json.dumps(severity_fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graphs = None
    
    return render_template('dashboard.html', predictions=predictions, graphs=graphs)

# Weather data route to fetch from API
@app.route('/fetch-weather', methods=['POST'])
@login_required
def fetch_weather():
    city = request.form.get('city')
    if not city:
        flash('Please provide a city name')
        return redirect(url_for('dashboard'))
    
    weather_data = fetch_weather_data(city)
    if weather_data:
        # Get predictions to pass to the template
        predictions = Prediction.get_user_predictions(current_user.id)
        graphs = None
        
        if predictions:
            # Create visualizations (simplified reuse of dashboard code)
            df = pd.DataFrame([
                {
                    'Timestamp': p.timestamp,
                    'Rainfall': p.rainfall,
                    'Temperature': p.temperature,
                    'Humidity': p.humidity,
                    'Wind Speed': p.wind_speed,
                    'Soil Moisture': p.soil_moisture if p.soil_moisture is not None else 0,
                    'Air Pressure': p.air_pressure if p.air_pressure is not None else 0,
                    'Result': p.result,
                    'Disaster Type': p.disaster_type if p.disaster_type else 'None',
                    'Severity': p.severity if p.severity else 'Low'
                } for p in predictions
            ])
            
            # Simple charts for now
            feature_names = ['Rainfall', 'Temperature', 'Humidity', 'Wind Speed']
            importances_values = disaster_model.feature_importances_[:4]  # Just use first 4
            
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances_values
            }).sort_values('Importance', ascending=False)
            
            feature_fig = px.bar(importances, x='Feature', y='Importance', title='Feature Importance')
            recent_fig = px.scatter(df, x='Timestamp', y='Rainfall', color='Result', title='Recent Predictions')
            
            graphs = {
                'feature_importance': json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'recent_predictions': json.dumps(recent_fig, cls=plotly.utils.PlotlyJSONEncoder)
            }
            
        flash(f'Weather data fetched successfully for {city}')
        return render_template('dashboard.html', weather_data=weather_data, city=city, predictions=predictions, graphs=graphs)
    else:
        flash(f'Could not fetch weather data for {city}')
        return redirect(url_for('dashboard'))

# Prediction route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get form data and convert to floats
        rainfall = float(request.form.get('rainfall'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        soil_moisture = float(request.form.get('soil_moisture')) if request.form.get('soil_moisture') else None
        air_pressure = float(request.form.get('air_pressure')) if request.form.get('air_pressure') else None
        
        # Set default values for missing features
        if soil_moisture is None:
            soil_moisture = 50.0  # Default value
        if air_pressure is None:
            air_pressure = 1013.0  # Default value (average sea level pressure)
        
        # Create an input array for prediction
        features = np.array([[rainfall, temperature, humidity, wind_speed, soil_moisture, air_pressure]])
        
        # Check if model was trained with all features
        if disaster_model.n_features_in_ < 6:
            # Use only the original features
            features = features[:, :disaster_model.n_features_in_]
        
        # Predict using the loaded model
        prediction = disaster_model.predict(features)
        probabilities = disaster_model.predict_proba(features)[0]
        
        result = "Disaster Likely" if prediction[0] == 1 else "No Disaster Predicted"
        
        # Predict disaster type and severity if available and if disaster detected
        disaster_type = None
        severity = None
        
        if prediction[0] == 1:
            if type_model is not None:
                try:
                    disaster_type = type_model.predict(features)[0]
                except:
                    disaster_type = "Unknown"
                    
            if severity_model is not None:
                try:
                    severity = severity_model.predict(features)[0]
                except:
                    severity = "Low"
        
        # Save prediction to database
        pred = Prediction(
            user_id=current_user.id,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            soil_moisture=soil_moisture,
            air_pressure=air_pressure,
            result=result,
            probability=probabilities.tolist(),
            disaster_type=disaster_type,
            severity=severity
        )
        Prediction.save(pred)
        
        # For API requests
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify({
                'result': result,
                'probability': probabilities.tolist(),
                'disaster_type': disaster_type,
                'severity': severity,
                'features': {
                    'rainfall': rainfall,
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'soil_moisture': soil_moisture,
                    'air_pressure': air_pressure
                }
            })
        
        # Return prediction to template
        if disaster_type and severity:
            flash(f"Prediction: {result} | Type: {disaster_type} | Severity: {severity}")
        else:
            flash(f"Prediction: {result}")
        return redirect(url_for('dashboard'))
    except ValueError:
        flash("Error: Please enter valid numeric values.")
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for('dashboard'))

# API route for getting predictions
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        rainfall = float(data.get('rainfall'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        wind_speed = float(data.get('wind_speed'))
        soil_moisture = float(data.get('soil_moisture')) if data.get('soil_moisture') is not None else 50.0
        air_pressure = float(data.get('air_pressure')) if data.get('air_pressure') is not None else 1013.0
        
        features = np.array([[rainfall, temperature, humidity, wind_speed, soil_moisture, air_pressure]])
        
        # Check if model was trained with all features
        if disaster_model.n_features_in_ < 6:
            # Use only the original features
            features = features[:, :disaster_model.n_features_in_]
            
        prediction = disaster_model.predict(features)
        probabilities = disaster_model.predict_proba(features)[0]
        
        response = {
            'result': "Disaster Likely" if prediction[0] == 1 else "No Disaster Predicted",
            'probability': probabilities.tolist(),
            'features': {
                'rainfall': rainfall,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'soil_moisture': soil_moisture,
                'air_pressure': air_pressure
            }
        }
        
        # Add disaster type and severity if available
        if prediction[0] == 1:
            if type_model is not None:
                try:
                    response['disaster_type'] = type_model.predict(features)[0]
                except:
                    response['disaster_type'] = "Unknown"
                    
            if severity_model is not None:
                try:
                    response['severity'] = severity_model.predict(features)[0]
                except:
                    response['severity'] = "Low"
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Map-based dashboard route
@app.route('/map')
@login_required
def map_dashboard():
    predictions = Prediction.get_user_predictions(current_user.id, limit=50)  # Get more data for the map
    
    return render_template('map.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
