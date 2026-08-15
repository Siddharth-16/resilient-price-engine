from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def prepare_input(raw_input: dict) -> pd.DataFrame:
    return pd.DataFrame([raw_input])


def predict_price(raw_input: dict, model) -> float:
    X = prepare_input(raw_input)
    prediction = model.predict(X)[0]
    return float(prediction)
