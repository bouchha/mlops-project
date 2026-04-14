from fastapi import FastAPI
import joblib
from typing import List

app = FastAPI()

# Charger modèle
model = joblib.load("model.pkl")

# Endpoint santé
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint prédiction
@app.post("/predict")
def predict(data: List[float]):
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}