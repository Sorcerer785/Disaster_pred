# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def generate_data(num_samples=1000):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic environmental features with new dimensions
    data = pd.DataFrame({
        'rainfall': np.random.uniform(0, 300, num_samples),       # in millimeters
        'temperature': np.random.uniform(10, 40, num_samples),    # in Celsius
        'humidity': np.random.uniform(20, 100, num_samples),      # in percentage
        'wind_speed': np.random.uniform(0, 20, num_samples),      # in m/s
        'soil_moisture': np.random.uniform(0, 100, num_samples),  # in percentage
        'air_pressure': np.random.uniform(980, 1050, num_samples) # in hPa
    })
    
    # Simulate disaster occurrence with more complex rules
    data['disaster'] = 0  # Default: No disaster
    
    # Flood conditions: high rainfall + high soil moisture
    flood_mask = (data['rainfall'] > 200) & (data['soil_moisture'] > 70)
    data.loc[flood_mask, 'disaster'] = 1
    data.loc[flood_mask, 'disaster_type'] = 'Flood'
    
    # Storm conditions: high wind speed + low air pressure
    storm_mask = (data['wind_speed'] > 15) & (data['air_pressure'] < 1000)
    data.loc[storm_mask, 'disaster'] = 1
    data.loc[storm_mask, 'disaster_type'] = 'Storm'
    
    # Heatwave conditions: high temperature + low humidity
    heatwave_mask = (data['temperature'] > 35) & (data['humidity'] < 30)
    data.loc[heatwave_mask, 'disaster'] = 1
    data.loc[heatwave_mask, 'disaster_type'] = 'Heatwave'
    
    # Add severity levels based on conditions
    data['severity'] = 'Low'  # Default severity
    
    # Define medium severity conditions
    medium_severity = (
        (flood_mask & (data['rainfall'] > 230)) |
        (storm_mask & (data['wind_speed'] > 17)) |
        (heatwave_mask & (data['temperature'] > 37))
    )
    data.loc[medium_severity, 'severity'] = 'Medium'
    
    # Define high severity conditions
    high_severity = (
        (flood_mask & (data['rainfall'] > 260)) |
        (storm_mask & (data['wind_speed'] > 19)) |
        (heatwave_mask & (data['temperature'] > 39))
    )
    data.loc[high_severity, 'severity'] = 'High'
    
    # Fill NaN values for non-disaster rows
    data['disaster_type'].fillna('None', inplace=True)
    
    return data

def train_model():
    # Generate synthetic dataset
    data = generate_data()
    
    # Train disaster prediction model (binary classification)
    # Select features and label
    X = data[['rainfall', 'temperature', 'humidity', 'wind_speed', 'soil_moisture', 'air_pressure']]
    y = data['disaster']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print("Disaster prediction accuracy: {:.2f}%".format(accuracy * 100))
    
    # Save the binary prediction model
    with open("disaster_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Disaster prediction model saved as disaster_model.pkl")
    
    # Train disaster type model (multi-class classification) - only on disaster=1 rows
    disaster_data = data[data['disaster'] == 1]
    X_type = disaster_data[['rainfall', 'temperature', 'humidity', 'wind_speed', 'soil_moisture', 'air_pressure']]
    y_type = disaster_data['disaster_type']
    
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type, y_type, test_size=0.2, random_state=42)
    
    type_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    type_clf.fit(X_type_train, y_type_train)
    
    type_accuracy = type_clf.score(X_type_test, y_type_test)
    print("Disaster type prediction accuracy: {:.2f}%".format(type_accuracy * 100))
    
    # Save the disaster type model
    with open("disaster_type_model.pkl", "wb") as f:
        pickle.dump(type_clf, f)
    print("Disaster type model saved as disaster_type_model.pkl")
    
    # Train severity model (multi-class classification) - only on disaster=1 rows
    X_severity = disaster_data[['rainfall', 'temperature', 'humidity', 'wind_speed', 'soil_moisture', 'air_pressure']]
    y_severity = disaster_data['severity']
    
    X_severity_train, X_severity_test, y_severity_train, y_severity_test = train_test_split(
        X_severity, y_severity, test_size=0.2, random_state=42
    )
    
    severity_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    severity_clf.fit(X_severity_train, y_severity_train)
    
    severity_accuracy = severity_clf.score(X_severity_test, y_severity_test)
    print("Severity prediction accuracy: {:.2f}%".format(severity_accuracy * 100))
    
    # Save the severity model
    with open("severity_model.pkl", "wb") as f:
        pickle.dump(severity_clf, f)
    print("Severity model saved as severity_model.pkl")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance.to_csv("feature_importance.csv", index=False)
    print("Feature importance saved as feature_importance.csv")

if __name__ == '__main__':
    train_model()
