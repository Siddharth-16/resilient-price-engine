from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "model_features.joblib"

MODEL = None
FEATURE_COLUMNS = None


def load_artifacts() -> None:
    global MODEL, FEATURE_COLUMNS

    if MODEL is None:
        MODEL = joblib.load(MODEL_PATH)

    if FEATURE_COLUMNS is None:
        FEATURE_COLUMNS = joblib.load(FEATURES_PATH)


def prepare_input(raw_input: dict) -> pd.DataFrame:
    load_artifacts()

    df = pd.DataFrame([raw_input])

    if "year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = 2024 - df["year"]
        df = df.drop(columns=["year"])

    df = pd.get_dummies(df, drop_first=False)

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


def predict_price(raw_input: dict) -> float:
    load_artifacts()

    X = prepare_input(raw_input)
    prediction = MODEL.predict(X)[0]
    return float(prediction)