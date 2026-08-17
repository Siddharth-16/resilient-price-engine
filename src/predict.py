from __future__ import annotations

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR


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
    Convert one raw API request into the DataFrame schema expected
    by the sklearn Pipeline.

    Encoding is handled inside the saved Pipeline.
    """

    return pd.DataFrame(
        [raw_input]
    )


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