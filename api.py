from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os
import uvicorn

app = FastAPI()

class Transaction(BaseModel):
    features: List[float]

# Try to load model, but don't crash if not found
model = None
scaler = None

model_path = 'fraud_model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print("⚠️ Model files not found - running in mock mode for CI/CD")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(t: Transaction):
    if model is None or scaler is None:
        # Mock prediction for CI/CD when model not available
        return {
            "is_fraudulent": False,
            "probability": 0.05,
            "risk": "LOW",
            "note": "Mock mode - model not loaded"
        }
    
    features = np.array(t.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = float(model.predict_proba(features_scaled)[0][1])
    pred = int(model.predict(features_scaled)[0])
    
    return {
        "is_fraudulent": bool(pred),
        "probability": prob,
        "risk": "HIGH" if prob > 0.5 else "LOW"
    }

@app.get("/metrics")
def metrics():
    return {
        "model_type": "XGBoost",
        "status": "loaded" if model is not None else "mock_mode",
        "performance": {
            "precision": 0.7810,
            "recall": 0.8367,
            "f1_score": 0.8079,
            "roc_auc": 0.9747
        } if model is not None else {}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
