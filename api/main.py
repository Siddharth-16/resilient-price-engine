from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.schemas import PricePredictionRequest, PricePredictionResponse
from src.predict import predict_price

app = FastAPI(title="Resilient Price Engine", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PricePredictionResponse)
def predict(request: PricePredictionRequest) -> PricePredictionResponse:
    try:
        raw_input = request.model_dump()
        predicted_price = predict_price(raw_input)
        return PricePredictionResponse(predicted_price=round(predicted_price, 2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))