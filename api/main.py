from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import PricePredictionRequest, PricePredictionResponse
from src.config import ARTIFACTS_DIR
from src.predict import MODEL_PATH, load_model, predict_price


class ModelStore:
    def __init__(self) -> None:
        self.model = None
        self.model_mtime_ns: int | None = None

    def reload(self) -> None:
        self.model = load_model()
        self.model_mtime_ns = MODEL_PATH.stat().st_mtime_ns

    def get(self):
        current_mtime = MODEL_PATH.stat().st_mtime_ns
        if self.model is None or self.model_mtime_ns != current_mtime:
            self.reload()
        return self.model


model_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store.reload()
    yield


app = FastAPI(
    title="Resilient Price Engine",
    description="Used car price prediction API with drift monitoring and safe model promotion.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="production model unavailable")
    return {"status": "ok", "model_ready": True}


@app.get("/model-info")
def model_info() -> dict:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {"status": "no model metadata found"}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "model": metrics.get("model"),
        "train_mae": metrics.get("train_mae"),
        "test_mae": metrics.get("test_mae"),
        "evaluation_mae": metrics.get("evaluation_mae"),
        "num_features": metrics.get("num_features"),
        "training_data_path": metrics.get("training_data_path"),
        "candidate": metrics.get("candidate"),
        "promoted_from_candidate": metrics.get("promoted_from_candidate", False),
    }


@app.post("/predict", response_model=PricePredictionResponse)
def predict(request: PricePredictionRequest) -> PricePredictionResponse:
    try:
        predicted_price = predict_price(request.model_dump(), model_store.get())
        return PricePredictionResponse(predicted_price=round(predicted_price, 2))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="prediction failed") from exc
