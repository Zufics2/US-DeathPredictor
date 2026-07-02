import json
import joblib
import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model import DeathPredictionModel

app = FastAPI(
    title='US Causes of Death Predictor',
    description='FastAPI + PyTorch Deep Learning Model',
    version='1.0.0'
)

app.mount("/static", StaticFiles(directory="static"), name="static")

x_scaler = joblib.load("artifacts/x_scaler.joblib")  
feature_columns = joblib.load("artifacts/feature_columns.pkl")

model = DeathPredictionModel(input_size=len(feature_columns))
state_dict = torch.load("artifacts/death_prediction_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

STATES = sorted([col.replace("State_", "") for col in feature_columns if col.startswith("State_")])
CAUSES = sorted([col.replace("Cause Name_", "") for col in feature_columns if col.startswith("Cause Name_")])

class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1990, le=2030, description="Год для прогноза")
    state: str = Field(..., description="Название штата")
    cause_name: str = Field(..., description="Причина смерти")

@app.get("/", response_class=HTMLResponse)
def frontend():
    with open("static/index.html", "r", encoding="utf-8") as file:
        return file.read()

@app.get("/meta")
def get_meta_data():
    """Возвращает списки доступных штатов и причин для выпадающих списков на фронте"""
    return {
        "states": STATES,
        "causes": CAUSES
    }

@app.get("/metrics")
def get_metrics():
    with open("artifacts/metrics.json", "r", encoding="utf-8") as file:
        return json.load(file)

@app.post("/predict")
def predict_death_rate(data: PredictionRequest):
    encoded_features = {col: 0.0 for col in feature_columns}
    
    if "Year" in encoded_features:
        encoded_features["Year"] = float(data.year)

    state_col = f"State_{data.state}"
    cause_col = f"Cause Name_{data.cause_name}"
    
    if state_col in encoded_features:
        encoded_features[state_col] = 1.0
    if cause_col in encoded_features:
        encoded_features[cause_col] = 1.0
        
    input_vector = [encoded_features[col] for col in feature_columns]
    
    scaled_year = x_scaler.transform([[data.year]])[0][0]
    input_vector[0] = scaled_year
    
    input_tensor = torch.tensor([input_vector], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor).numpy()
    
    rate = float(prediction[0][0])
    
    return {
        "predicted_age_adjusted_death_rate": round(max(0.0, rate), 2),
        "unit": "Deaths per 100,000 population"
    }