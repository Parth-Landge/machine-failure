import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Import MLflow!
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri('sqlite:///mlflow.db')
# 1. Set up MLflow Experiment
# This creates a specific folder in your dashboard for this project
mlflow.set_experiment("Milling_Machine_Failure_Prediction")

def load_and_preprocess_data():
    """Load dataset and prepare features. Update the filename to match yours!"""
    # Replace with your actual dataset name, e.g., 'ai4i2020.csv'
    DATA_PATH = "predictive_maintenance.csv" 
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Successfully loaded {DATA_PATH}")
    except FileNotFoundError:
        print(f"⚠️ {DATA_PATH} not found. Using simulated data for testing...")
        # Fallback simulated data just in case the CSV is missing so the script doesn't crash
        df = pd.DataFrame({
            "Type": np.random.choice(['L', 'M', 'H'], 1000),
            "Air temperature [K]": np.random.uniform(295, 305, 1000),
            "Process temperature [K]": np.random.uniform(305, 315, 1000),
            "Rotational speed [rpm]": np.random.uniform(1200, 2800, 1000),
            "Torque [Nm]": np.random.uniform(20, 70, 1000),
            "Tool wear [min]": np.random.uniform(0, 250, 1000),
            "Machine failure": np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })

    # Encode Categorical Data ('L', 'M', 'H' -> 0, 1, 2)
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    
    # Select Features and Target
    X = df[['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    y = df['Machine failure']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("🚀 Starting Model Training with MLflow...")
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Define hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    # 2. Start the MLflow run (The Flight Recorder turns ON)
    with mlflow.start_run(run_name="RandomForest_Base_Run"):
        
        # 3. Log Parameters (Settings you chose)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        print("🧠 Training the Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # 4. Make Predictions & Calculate Metrics
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        
        # 5. Log Metrics (How well the model did)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        print(f"📊 Results: Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        # 6. Log the Model itself into MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Also save a local .pkl file just like you did before for your FastAPI
        joblib.dump(model, "model.pkl")
        print("💾 Model saved locally as model.pkl")
        
        print("✅ MLflow Run Complete! The flight recorder has saved the data.")