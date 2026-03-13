from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR

MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "model_features.joblib"

def load_model() -> tuple[object, list[str]]:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns

def prepare_input(raw_input: dict, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    if "year" in df.columns and "car_age" not in df.columns:
        df["car_age"] = 2024 - df["year"]
        df = df.drop(columns=["year"])

    df = pd.get_dummies(df, drop_first=False)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

def predict_price(raw_input: dict, model_bundle: tuple[object, list[str]]) -> float:
    model, feature_columns = model_bundle

    X = prepare_input(raw_input, feature_columns)
    prediction = model.predict(X)[0]

    return float(prediction)