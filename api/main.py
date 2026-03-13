from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import PricePredictionRequest, PricePredictionResponse
from src.predict import load_model, predict_price
from pathlib import Path
import json

model_bundle = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_bundle = load_model()
    yield

app = FastAPI(
    title="Resilient Price Engine",
    description="Used car price prediction API with drift monitoring and automated retraining.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}

@app.get("/model-info")
def model_info() -> dict:
    metrics_path = Path("artifacts/metrics.json")

    if not metrics_path.exists():
        return {"status": "no model metadata found"}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "model": metrics.get("model"),
        "train_mae": metrics.get("train_mae"),
        "test_mae": metrics.get("test_mae"),
        "num_features": metrics.get("num_features"),
        "training_data_path": metrics.get("training_data_path"),
        "candidate": metrics.get("candidate"),
    }

@app.post("/predict", response_model=PricePredictionResponse)
def predict(request: PricePredictionRequest) -> PricePredictionResponse:
    try:
        raw_input = request.model_dump()
        predicted_price = predict_price(raw_input, model_bundle)
        return PricePredictionResponse(predicted_price=round(predicted_price, 2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))