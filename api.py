from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import uvicorn

app = FastAPI()

class Transaction(BaseModel):
    features: List[float]

print("Loading model...")
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Model loaded!")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(t: Transaction):
    features = np.array(t.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = float(model.predict_proba(features_scaled)[0][1])
    pred = int(model.predict(features_scaled)[0])
    return {
        "is_fraudulent": bool(pred),
        "probability": prob,
        "risk": "HIGH" if prob > 0.5 else "LOW"
    }

if __name__ == "__main__":
    print("\n🚀 API Starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
