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
                 humidity=None, wind_speed=None, result=None, timestamp=None, probability=None):
        self.id = id
        self.user_id = user_id
        self.rainfall = rainfall
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.result = result
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.probability = probability

    @staticmethod
    def save(prediction):
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions 
               (user_id, rainfall, temperature, humidity, wind_speed, result, timestamp, probability) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction.user_id, prediction.rainfall, prediction.temperature, 
             prediction.humidity, prediction.wind_speed, prediction.result, 
             prediction.timestamp, json.dumps(prediction.probability) if prediction.probability else None)
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
            pred = Prediction(
                id=row['id'],
                user_id=row['user_id'],
                rainfall=row['rainfall'],
                temperature=row['temperature'],
                humidity=row['humidity'],
                wind_speed=row['wind_speed'],
                result=row['result'],
                timestamp=row['timestamp'],
                probability=json.loads(row['probability']) if row['probability'] else None
            )
            predictions.append(pred)
        conn.close()
        return predictions 