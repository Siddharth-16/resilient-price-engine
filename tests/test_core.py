from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from api.schemas import PricePredictionRequest
from src.drift_detection import (
    build_simulated_drift_batch,
    detect_drift,
    should_promote,
)
from src.predict import predict_price, prepare_input
from src.train import build_pipeline, evaluate_model, load_data


def sample_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manufacturer": ["ford", "ford", "toyota", "toyota", "honda", "honda", "bmw", "bmw"],
            "model": ["f150", "focus", "camry", "corolla", "civic", "accord", "x3", "x5"],
            "fuel": ["gas"] * 8,
            "title_status": ["clean"] * 8,
            "transmission": ["automatic"] * 8,
            "drive": ["4wd", "fwd", "fwd", "fwd", "fwd", "fwd", "awd", "awd"],
            "type": ["truck", "sedan", "sedan", "sedan", "sedan", "sedan", "suv", "suv"],
            "paint_color": ["white", "blue", "black", "red", "white", "black", "blue", "black"],
            "state": ["ny", "ny", "ca", "ca", "tx", "tx", "ny", "nj"],
            "odometer": [90000, 70000, 60000, 50000, 40000, 55000, 30000, 25000],
            "car_age": [8, 7, 6, 5, 4, 5, 3, 2],
            "price": [22000, 14000, 19000, 18000, 20000, 17500, 33000, 41000],
        }
    )


def test_pipeline_predicts_with_unseen_category() -> None:
    df = sample_training_frame()
    X = df.drop(columns=["price"])
    model = build_pipeline(X)
    model.fit(X, df["price"])

    unseen = X.iloc[[0]].copy()
    unseen["manufacturer"] = "never-seen-brand"
    prediction = model.predict(unseen)[0]

    assert prediction > 0


def test_prediction_accepts_raw_features() -> None:
    df = sample_training_frame()
    X = df.drop(columns=["price"])
    model = build_pipeline(X)
    model.fit(X, df["price"])

    prediction = predict_price(X.iloc[0].to_dict(), model)
    assert isinstance(prediction, float)


def test_prepare_input_preserves_raw_schema() -> None:
    row = {"manufacturer": "ford", "odometer": 1000, "car_age": 2}
    prepared = prepare_input(row)
    assert prepared.columns.tolist() == ["manufacturer", "odometer", "car_age"]


def test_evaluate_model_returns_nonnegative_mae() -> None:
    df = sample_training_frame()
    X = df.drop(columns=["price"])
    model = build_pipeline(X)
    model.fit(X, df["price"])
    assert evaluate_model(model, df) >= 0


def test_load_data_requires_price_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"odometer": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="price"):
        load_data(path)


def test_detect_drift_flags_large_numeric_shift() -> None:
    reference = pd.DataFrame({"odometer": list(range(200)), "state": ["ny"] * 200})
    incoming = pd.DataFrame({"odometer": list(range(1000, 1200)), "state": ["ca"] * 200})
    assert "odometer" in detect_drift(reference, incoming)


def test_detect_drift_ignores_categorical_columns() -> None:
    reference = pd.DataFrame({"state": ["ny"] * 50})
    incoming = pd.DataFrame({"state": ["ca"] * 50})
    assert detect_drift(reference, incoming) == []


def test_simulated_drift_changes_monitored_features() -> None:
    incoming = pd.DataFrame({"odometer": [100.0], "car_age": [4.0], "price": [10000]})
    drifted = build_simulated_drift_batch(incoming)
    assert drifted.loc[0, "odometer"] == 150.0
    assert drifted.loc[0, "car_age"] == 9.0
    assert drifted.loc[0, "price"] == 10000


def test_promotion_requires_better_candidate() -> None:
    assert should_promote(1800.0, 1700.0)
    assert not should_promote(1800.0, 1900.0)


def test_request_rejects_negative_odometer() -> None:
    with pytest.raises(ValidationError):
        PricePredictionRequest(odometer=-1, car_age=4)


def test_request_rejects_empty_categorical_value() -> None:
    with pytest.raises(ValidationError):
        PricePredictionRequest(manufacturer="", odometer=1000, car_age=4)
