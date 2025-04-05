import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    ''')
    
    # Drop and recreate predictions table with new columns
    cursor.execute('DROP TABLE IF EXISTS predictions')
    
    # Create predictions table with additional fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        rainfall REAL,
        temperature REAL,
        humidity REAL,
        wind_speed REAL,
        soil_moisture REAL,
        air_pressure REAL,
        result TEXT,
        timestamp TEXT,
        probability TEXT,
        disaster_type TEXT,
        severity TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create weather_data table for historical records
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS weather_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        location TEXT,
        rainfall REAL,
        temperature REAL,
        humidity REAL,
        wind_speed REAL,
        soil_moisture REAL,
        air_pressure REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db() 