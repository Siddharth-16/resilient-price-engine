from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
)

from api.schemas import (
    PricePredictionRequest,
    PricePredictionResponse,
)

from src.config import (
    ARTIFACTS_DIR,
)

from src.predict import (
    MODEL_PATH,
    load_model,
    predict_price,
)

# Model store

class ModelStore:
    """
    Keep the production model in memory and automatically reload it
    when the model artifact changes.

    This allows drift-triggered model promotion to replace
    price_model.joblib without requiring an API restart.
    """

    def __init__(
        self,
    ) -> None:
        self.model = None

        self.model_mtime_ns: (
            int | None
        ) = None


    def reload(
        self,
    ) -> None:
        self.model = (
            load_model()
        )

        self.model_mtime_ns = (
            MODEL_PATH
            .stat()
            .st_mtime_ns
        )


    def get(
        self,
    ):
        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Production model not found "
                f"at {MODEL_PATH}. "
                "Run `python -m src.train` first."
            )

        current_mtime = (
            MODEL_PATH
            .stat()
            .st_mtime_ns
        )

        if (
            self.model is None
            or self.model_mtime_ns
            != current_mtime
        ):
            self.reload()

        return self.model


model_store = (
    ModelStore()
)

# Lifespan

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Load the production model on startup when available.

    The API can still start without a model; readiness endpoints and
    prediction requests will return 503 until training is completed.
    """

    if MODEL_PATH.exists():
        model_store.reload()

    yield

# FastAPI application

app = FastAPI(
    title=(
        "Resilient Price Engine"
    ),
    description=(
        "Used-car price prediction API "
        "with drift monitoring, "
        "candidate retraining, and "
        "safe model promotion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Health

@app.get(
    "/health"
)
def health_check() -> dict:
    """
    Lightweight readiness check.

    Returns 503 if the production model has not been trained yet.
    """

    if not MODEL_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "production model unavailable"
            ),
        )

    return {
        "status": "ok",
        "model_ready": True,
    }

# Model metadata

@app.get(
    "/model-info"
)
def model_info() -> dict:
    """
    Return metadata for the current production model.

    Supports both baseline training metadata and the latest
    drift-promotion information.
    """

    metrics_path = (
        ARTIFACTS_DIR
        / "metrics.json"
    )

    if not metrics_path.exists():

        return {
            "status": (
                "no model metadata found"
            )
        }

    try:
        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "unable to read "
                "model metadata"
            ),
        ) from exc


    train_metrics = (
        metrics.get(
            "train_metrics"
        )
        or {}
    )

    evaluation_metrics = (
        metrics.get(
            "evaluation_metrics"
        )
        or {}
    )

    latest_cohort_metrics = (
        metrics.get(
            "latest_cohort_metrics"
        )
        or {}
    )


    return {
        "model": (
            metrics.get(
                "model"
            )
        ),

        "hyperparameters": (
            metrics.get(
                "hyperparameters"
            )
        ),

        "num_features": (
            metrics.get(
                "num_features"
            )
        ),

        "train_rows": (
            metrics.get(
                "train_rows"
            )
        ),

        "evaluation_rows": (
            metrics.get(
                "evaluation_rows"
            )
        ),

        "split_strategy": (
            metrics.get(
                "split_strategy"
            )
        ),

        "feature_overlap_pct": (
            metrics.get(
                "feature_overlap_pct"
            )
        ),

        "training_metrics": {
            "mae": (
                train_metrics.get(
                    "mae"
                )
            ),
            "median_ae": (
                train_metrics.get(
                    "median_ae"
                )
            ),
            "rmse": (
                train_metrics.get(
                    "rmse"
                )
            ),
            "r2": (
                train_metrics.get(
                    "r2"
                )
            ),
        },

        "baseline_evaluation_metrics": {
            "mae": (
                evaluation_metrics.get(
                    "mae"
                )
            ),
            "median_ae": (
                evaluation_metrics.get(
                    "median_ae"
                )
            ),
            "rmse": (
                evaluation_metrics.get(
                    "rmse"
                )
            ),
            "r2": (
                evaluation_metrics.get(
                    "r2"
                )
            ),
        },

        "latest_cohort": (
            metrics.get(
                "latest_cohort"
            )
        ),

        "latest_cohort_metrics": {
            "mae": (
                latest_cohort_metrics.get(
                    "mae"
                )
            ),
            "median_ae": (
                latest_cohort_metrics.get(
                    "median_ae"
                )
            ),
            "rmse": (
                latest_cohort_metrics.get(
                    "rmse"
                )
            ),
            "r2": (
                latest_cohort_metrics.get(
                    "r2"
                )
            ),
        },

        "candidate": (
            metrics.get(
                "candidate",
                False,
            )
        ),
    }

# Prediction

@app.post(
    "/predict",
    response_model=(
        PricePredictionResponse
    ),
)
def predict(
    request: (
        PricePredictionRequest
    ),
) -> PricePredictionResponse:
    """
    Predict the price for one vehicle.

    ModelStore automatically reloads the production artifact if
    drift_detection.py has promoted a newer candidate model.
    """

    try:
        model = (
            model_store.get()
        )

        raw_input = (
            request.model_dump()
        )

        predicted_price = (
            predict_price(
                raw_input,
                model,
            )
        )

        return (
            PricePredictionResponse(
                predicted_price=round(
                    predicted_price,
                    2,
                )
            )
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(
                exc
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "prediction failed"
            ),
        ) from exc