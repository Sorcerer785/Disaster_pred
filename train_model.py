# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def generate_data(num_samples=1000):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic environmental features
    data = pd.DataFrame({
        'rainfall': np.random.uniform(0, 300, num_samples),       # in millimeters
        'temperature': np.random.uniform(10, 40, num_samples),      # in Celsius
        'humidity': np.random.uniform(20, 100, num_samples),        # in percentage
        'wind_speed': np.random.uniform(0, 20, num_samples)         # in m/s
    })
    
    # Simulate disaster occurrence:
    # Here, if rainfall > 200 mm and humidity > 80%, we mark it as a disaster.
    data['disaster'] = ((data['rainfall'] > 200) & (data['humidity'] > 80)).astype(int)
    return data

def train_model():
    # Generate synthetic dataset
    data = generate_data()
    
    # Select features and label
    X = data[['rainfall', 'temperature', 'humidity', 'wind_speed']]
    y = data['disaster']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    print("Test accuracy: {:.2f}%".format(accuracy * 100))
    
    # Save the model to disk
    with open("disaster_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model saved as disaster_model.pkl")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feature_importance.to_csv("feature_importance.csv", index=False)
    print("Feature importance saved as feature_importance.csv")

if __name__ == '__main__':
    train_model()
