from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. Initialize the App
app = FastAPI(
    title="Predictive Maintenance Core API",
    description="Backend engine for processing grinding machine sensor data.",
    version="1.0"
)

# 2. Load the Model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Central AI Model Loaded Successfully.")
except Exception as e:
    model = None
    print(f"Critical Error: Could not load model. {e}")

# 3. Define the Incoming Data Schema
class SensorData(BaseModel):
    Machine_Type: str  
    Air_Temperature_K: float
    Process_Temperature_K: float
    Rotational_Speed_rpm: float
    Torque_Nm: float
    Tool_Wear_min: float

# 4. The Health Check Endpoint
@app.get("/")
def system_status():
    return {"status": "Online", "message": "Predictive Maintenance API is actively listening."}

# 5. The Prediction Endpoint
@app.post("/predict")
def predict_failure(data: SensorData):
    if not model:
        raise HTTPException(status_code=500, detail="Model is currently offline.")

    try:
        temp_diff = data.Process_Temperature_K - data.Air_Temperature_K
        power = data.Rotational_Speed_rpm * data.Torque_Nm * (2 * np.pi / 60)
        type_L = 1 if data.Machine_Type == "L" else 0
        type_M = 1 if data.Machine_Type == "M" else 0

        input_df = pd.DataFrame([{
            'Air temperature [K]': data.Air_Temperature_K,
            'Process temperature [K]': data.Process_Temperature_K,
            'Rotational speed [rpm]': data.Rotational_Speed_rpm,
            'Torque [Nm]': data.Torque_Nm,
            'Tool wear [min]': data.Tool_Wear_min,
            'Temperature_Difference': temp_diff,
            'Power [W]': power,
            'Type_L': type_L,
            'Type_M': type_M
        }])

        probability = float(model.predict_proba(input_df.values)[0][1])
        prediction = int(model.predict(input_df.values)[0])

        return {
            "critical_failure_imminent": bool(prediction),
            "failure_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")
