from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import os

from src.preprocess import extract_hrv_features
from src.inference import load_model, predict_stress

app = FastAPI(title="MindGuard Stress AI", description="Real-time PPG inference pipeline")

# Load model on startup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "stress_model.pkl")

# We will load the model globally so it doesn't reload on every request
try:
    stress_model = load_model(model_path)
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Warning: {e}")
    stress_model = None

class PPGDataPayload(BaseModel):
    ppg_signal: List[float]
    sampling_rate: int = 256

@app.post("/predict_stress")
async def predict_stress_endpoint(payload: PPGDataPayload):
    if stress_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
        
    try:
        import numpy as np
        ppg_array = np.array(payload.ppg_signal)
        
        # 1. Pipeline: Preprocess and extract features
        hrv_features = extract_hrv_features(ppg_array, sampling_rate=payload.sampling_rate)
        
        # 2. Pipeline: Inference
        stress_level = predict_stress(stress_model, hrv_features)
        
        # Mapping for readability
        stress_mapping = {0: "No Stress", 1: "Moderate Stress", 2: "Severe Stress"}
        
        return {
            "stress_level_int": stress_level,
            "stress_level_label": stress_mapping.get(stress_level, "Unknown"),
            "extracted_features": hrv_features
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
