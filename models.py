import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import json

class User(UserMixin):
    def __init__(self, id, username, email, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                password_hash=user['password_hash']
            )
        return None

    @staticmethod
    def get_by_username(username):
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return User(
                id=user['id'],
                username=user['username'],
                email=user['email'],
                password_hash=user['password_hash']
            )
        return None

    @staticmethod
    def create(username, email, password):
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return User(user_id, username, email, password_hash)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction:
    def __init__(self, id=None, user_id=None, rainfall=None, temperature=None, 
                 humidity=None, wind_speed=None, soil_moisture=None, air_pressure=None,
                 result=None, timestamp=None, probability=None, disaster_type=None, severity=None):
        self.id = id
        self.user_id = user_id
        self.rainfall = rainfall
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.soil_moisture = soil_moisture
        self.air_pressure = air_pressure
        self.result = result
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.probability = probability
        self.disaster_type = disaster_type
        self.severity = severity

    @staticmethod
    def save(prediction):
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions 
               (user_id, rainfall, temperature, humidity, wind_speed, soil_moisture, air_pressure,
                result, timestamp, probability, disaster_type, severity) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction.user_id, prediction.rainfall, prediction.temperature, 
             prediction.humidity, prediction.wind_speed, prediction.soil_moisture, prediction.air_pressure,
             prediction.result, prediction.timestamp, json.dumps(prediction.probability) if prediction.probability else None,
             prediction.disaster_type, prediction.severity)
        )
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        prediction.id = prediction_id
        return prediction

    @staticmethod
    def get_user_predictions(user_id, limit=10):
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
            (user_id, limit)
        )
        predictions = []
        for row in cursor.fetchall():
            # Check if columns exist before accessing them
            soil_moisture = row['soil_moisture'] if 'soil_moisture' in row.keys() else None
            air_pressure = row['air_pressure'] if 'air_pressure' in row.keys() else None
            disaster_type = row['disaster_type'] if 'disaster_type' in row.keys() else None
            severity = row['severity'] if 'severity' in row.keys() else None
            
            pred = Prediction(
                id=row['id'],
                user_id=row['user_id'],
                rainfall=row['rainfall'],
                temperature=row['temperature'],
                humidity=row['humidity'],
                wind_speed=row['wind_speed'],
                soil_moisture=soil_moisture,
                air_pressure=air_pressure,
                result=row['result'],
                timestamp=row['timestamp'],
                probability=json.loads(row['probability']) if row['probability'] else None,
                disaster_type=disaster_type,
                severity=severity
            )
            predictions.append(pred)
        conn.close()
        return predictions 