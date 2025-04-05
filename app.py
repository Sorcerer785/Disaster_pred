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
from datetime import datetime
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

# Load the trained model
with open("disaster_model.pkl", "rb") as f:
    model = pickle.load(f)

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
                'Result': p.result
            } for p in predictions
        ])
        
        # Feature importance
        importances = pd.DataFrame({
            'Feature': ['Rainfall', 'Temperature', 'Humidity', 'Wind Speed'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_fig = px.bar(
            importances, 
            x='Feature', 
            y='Importance',
            title='Feature Importance',
            color='Importance'
        )
        
        # Recent predictions
        recent_fig = px.scatter(
            df,
            x='Timestamp', 
            y='Rainfall',
            color='Result',
            size='Wind Speed',
            hover_data=['Temperature', 'Humidity'],
            title='Recent Predictions'
        )
        
        graphs = {
            'feature_importance': json.dumps(feature_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'recent_predictions': json.dumps(recent_fig, cls=plotly.utils.PlotlyJSONEncoder)
        }
    else:
        graphs = None
    
    return render_template('dashboard.html', predictions=predictions, graphs=graphs)

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
        
        # Create an input array for prediction
        features = np.array([[rainfall, temperature, humidity, wind_speed]])
        
        # Predict using the loaded model
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        
        result = "Disaster Likely" if prediction[0] == 1 else "No Disaster Predicted"
        
        # Save prediction to database
        pred = Prediction(
            user_id=current_user.id,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            result=result,
            probability=probabilities.tolist()
        )
        Prediction.save(pred)
        
        # For API requests
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify({
                'result': result,
                'probability': probabilities.tolist(),
                'features': {
                    'rainfall': rainfall,
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                }
            })
        
        # Return prediction to template
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
        
        features = np.array([[rainfall, temperature, humidity, wind_speed]])
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        
        return jsonify({
            'result': "Disaster Likely" if prediction[0] == 1 else "No Disaster Predicted",
            'probability': probabilities.tolist(),
            'features': {
                'rainfall': rainfall,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
