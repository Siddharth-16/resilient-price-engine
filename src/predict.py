from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.preprocess import (
    AGE_REFERENCE_YEAR,
    normalize_categorical_columns,
)


MODEL_PATH = (
    ARTIFACTS_DIR
    / "price_model.joblib"
)


def load_model():
    """
    Load the current production sklearn Pipeline.

    The saved artifact already contains:
        preprocessing
        OneHotEncoder
        estimator

    Therefore no separate feature-columns artifact or manual
    one-hot encoding is required.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at "
            f"{MODEL_PATH}. "
            "Run `python -m src.train` first."
        )

    return joblib.load(
        MODEL_PATH
    )


def prepare_input(
    raw_input: dict,
) -> pd.DataFrame:
    """
    Convert an API request into the feature schema expected by the
    trained sklearn Pipeline.

    Public API:
        year

    Internal model representation:
        car_age = 2022 - year

    Categorical normalization is shared with training to prevent
    training-serving skew.
    """
    input_data = raw_input.copy()

    if "year" not in input_data:
        raise ValueError(
            "Prediction input must contain vehicle model year."
        )

    year = float(
        input_data.pop("year")
    )

    input_data["car_age"] = (
        AGE_REFERENCE_YEAR - year
    )

    prepared = pd.DataFrame(
        [input_data]
    )

    prepared = normalize_categorical_columns(
        prepared
    )

    return prepared


def predict_price(
    raw_input: dict,
    model,
) -> float:
    """
    Predict the price of one vehicle using the production Pipeline.
    """

    X = prepare_input(
        raw_input
    )

    prediction = (
        model.predict(
            X
        )[0]
    )

    return float(
        prediction
    )