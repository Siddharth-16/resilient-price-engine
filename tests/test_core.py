from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from api.schemas import PricePredictionRequest

from src.drift_detection import (
    calculate_feature_overlap,
    detect_categorical_drift,
    detect_numeric_drift,
    should_promote,
)

from src.predict import (
    predict_price,
    prepare_input,
)

from src.train import (
    build_pipeline,
    calculate_regression_metrics,
    evaluate_model,
    group_train_test_split,
    load_data,
)

# Test data

def sample_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manufacturer": [
                "ford",
                "ford",
                "toyota",
                "toyota",
                "honda",
                "honda",
                "bmw",
                "bmw",
            ],
            "model": [
                "f150",
                "focus",
                "camry",
                "corolla",
                "civic",
                "accord",
                "x3",
                "x5",
            ],
            "fuel": [
                "gas",
            ] * 8,
            "title_status": [
                "clean",
            ] * 8,
            "transmission": [
                "automatic",
            ] * 8,
            "drive": [
                "4wd",
                "fwd",
                "fwd",
                "fwd",
                "fwd",
                "fwd",
                "awd",
                "awd",
            ],
            "type": [
                "truck",
                "sedan",
                "sedan",
                "sedan",
                "sedan",
                "sedan",
                "suv",
                "suv",
            ],
            "paint_color": [
                "white",
                "blue",
                "black",
                "red",
                "white",
                "black",
                "blue",
                "black",
            ],
            "state": [
                "ny",
                "ny",
                "ca",
                "ca",
                "tx",
                "tx",
                "ny",
                "nj",
            ],
            "odometer": [
                90000,
                70000,
                60000,
                50000,
                40000,
                55000,
                30000,
                25000,
            ],
            "car_age": [
                8,
                7,
                6,
                5,
                4,
                5,
                3,
                2,
            ],
            "price": [
                22000,
                14000,
                19000,
                18000,
                20000,
                17500,
                33000,
                41000,
            ],
        }
    )

# Pipeline

def test_pipeline_predicts_with_unseen_category() -> None:
    df = sample_training_frame()

    X = df.drop(
        columns=["price"]
    )

    model = build_pipeline(
        X
    )

    model.fit(
        X,
        df["price"],
    )

    unseen = (
        X.iloc[[0]]
        .copy()
    )

    unseen[
        "manufacturer"
    ] = "never-seen-brand"

    prediction = (
        model.predict(
            unseen
        )[0]
    )

    assert prediction > 0

# Prediction

def test_prediction_accepts_raw_features() -> None:
    df = sample_training_frame()

    X = df.drop(
        columns=["price"]
    )

    model = build_pipeline(
        X
    )

    model.fit(
        X,
        df["price"],
    )

    raw_input = (
        X.iloc[0]
        .to_dict()
    )

    car_age = raw_input.pop(
        "car_age"
    )

    raw_input["year"] = int(
        2022 - car_age
    )

    # Also verify inference normalization.
    raw_input[
        "manufacturer"
    ] = " Ford "

    prediction = predict_price(
        raw_input,
        model,
    )

    assert isinstance(
        prediction,
        float,
    )

    assert prediction > 0


def valid_prediction_request() -> dict:
    return {
        "manufacturer": "ford",
        "model": "f-150",
        "fuel": "gas",
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "4wd",
        "type": "truck",
        "paint_color": "white",
        "state": "ny",
        "odometer": 90000,
        "year": 2018,
    }


def test_prediction_request_rejects_post_2022_vehicle() -> None:
    payload = valid_prediction_request()

    payload["year"] = 2023

    with pytest.raises(
        ValidationError
    ):
        PricePredictionRequest(
            **payload
        )


def test_prediction_request_rejects_pre_2000_vehicle() -> None:
    payload = valid_prediction_request()

    payload["year"] = 1999

    with pytest.raises(
        ValidationError
    ):
        PricePredictionRequest(
            **payload
        )


def test_prediction_request_rejects_odometer_above_training_domain() -> None:
    payload = valid_prediction_request()

    payload[
        "odometer"
    ] = 300_001

    with pytest.raises(
        ValidationError
    ):
        PricePredictionRequest(
            **payload
        )


def test_inference_normalization_matches_training_format() -> None:
    raw = {
        "manufacturer": " FORD ",
        "model": " F-150 ",
        "fuel": " GAS ",
        "title_status": " CLEAN ",
        "transmission": " AUTOMATIC ",
        "drive": " 4WD ",
        "type": " TRUCK ",
        "paint_color": " WHITE ",
        "state": " NY ",
        "odometer": 90000,
        "year": 2014,
    }

    prepared = prepare_input(
        raw
    )

    assert (
        prepared.loc[
            0,
            "manufacturer",
        ]
        == "ford"
    )

    assert (
        prepared.loc[
            0,
            "state",
        ]
        == "ny"
    )

    assert (
        prepared.loc[
            0,
            "car_age",
        ]
        == 8
    )


def test_prepare_input_derives_age_and_normalizes_categories() -> None:
    row = {
        "manufacturer": " Ford ",
        "model": " F-150 ",
        "odometer": 1000,
        "year": 2020,
    }

    prepared = prepare_input(
        row
    )

    assert "year" not in prepared.columns

    assert prepared.loc[
        0,
        "car_age",
    ] == 2

    assert prepared.loc[
        0,
        "manufacturer",
    ] == "ford"

    assert prepared.loc[
        0,
        "model",
    ] == "f-150"

# Regression metrics

def test_regression_metrics_are_calculated_correctly() -> None:
    y_true = [
        100.0,
        200.0,
        300.0,
    ]

    predictions = [
        110.0,
        190.0,
        310.0,
    ]

    metrics = (
        calculate_regression_metrics(
            y_true,
            predictions,
        )
    )

    assert set(
        metrics.keys()
    ) == {
        "mae",
        "median_ae",
        "rmse",
        "r2",
    }

    assert metrics["mae"] == pytest.approx(
        10.0
    )

    assert metrics["median_ae"] == pytest.approx(
        10.0
    )

    assert metrics["rmse"] == pytest.approx(
        10.0
    )

    assert metrics["r2"] <= 1.0


def test_evaluate_model_returns_metrics_dict() -> None:
    df = sample_training_frame()

    X = df.drop(
        columns=["price"]
    )

    model = build_pipeline(
        X
    )

    model.fit(
        X,
        df["price"],
    )

    metrics = evaluate_model(
        model,
        df,
    )

    assert set(
        metrics.keys()
    ) == {
        "mae",
        "median_ae",
        "rmse",
        "r2",
    }

    assert metrics["mae"] >= 0

    assert metrics["median_ae"] >= 0

    assert metrics["rmse"] >= 0

# Loading

def test_load_data_requires_price_column(
    tmp_path: Path,
) -> None:

    path = (
        tmp_path
        / "bad.csv"
    )

    pd.DataFrame(
        {
            "odometer": [
                1,
                2,
            ]
        }
    ).to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="price",
    ):
        load_data(
            path
        )

# Numeric drift

def test_numeric_drift_flags_large_odometer_shift() -> None:
    reference = pd.DataFrame(
        {
            "odometer": list(
                range(
                    0,
                    200,
                )
            )
        }
    )

    incoming = pd.DataFrame(
        {
            "odometer": list(
                range(
                    1000,
                    1200,
                )
            )
        }
    )

    results = detect_numeric_drift(
        reference,
        incoming,
    )

    assert len(
        results
    ) == 1

    result = results[0]

    assert result[
        "feature"
    ] == "odometer"

    assert result[
        "drifted"
    ] is True

    assert result[
        "ks_statistic"
    ] >= 0.10


def test_numeric_drift_does_not_monitor_car_age() -> None:
    reference = pd.DataFrame(
        {
            "odometer": [
                10000,
                20000,
                30000,
                40000,
            ],
            "car_age": [
                1,
                1,
                1,
                1,
            ],
        }
    )

    incoming = pd.DataFrame(
        {
            "odometer": [
                10000,
                20000,
                30000,
                40000,
            ],
            "car_age": [
                20,
                20,
                20,
                20,
            ],
        }
    )

    results = detect_numeric_drift(
        reference,
        incoming,
    )

    monitored_features = [
        result["feature"]
        for result in results
    ]

    assert "car_age" not in monitored_features

    assert monitored_features == [
        "odometer"
    ]

# Categorical drift

def test_categorical_drift_flags_large_shift() -> None:
    reference = pd.DataFrame(
        {
            "manufacturer": (
                ["ford"] * 100
                + ["toyota"] * 100
            ),
            "fuel": [
                "gas"
            ] * 200,
            "transmission": [
                "automatic"
            ] * 200,
            "drive": [
                "fwd"
            ] * 200,
            "type": [
                "sedan"
            ] * 200,
        }
    )

    incoming = pd.DataFrame(
        {
            "manufacturer": [
                "bmw"
            ] * 200,
            "fuel": [
                "electric"
            ] * 200,
            "transmission": [
                "manual"
            ] * 200,
            "drive": [
                "awd"
            ] * 200,
            "type": [
                "suv"
            ] * 200,
        }
    )

    results = (
        detect_categorical_drift(
            reference,
            incoming,
        )
    )

    drifted_features = [
        result["feature"]
        for result in results
        if result["drifted"]
    ]

    assert "manufacturer" in drifted_features

    assert "fuel" in drifted_features

    assert "transmission" in drifted_features


def test_categorical_drift_stable_distribution() -> None:
    reference = pd.DataFrame(
        {
            "manufacturer": [
                "ford",
                "toyota",
            ] * 100,
            "fuel": [
                "gas",
            ] * 200,
            "transmission": [
                "automatic",
            ] * 200,
            "drive": [
                "fwd",
            ] * 200,
            "type": [
                "sedan",
            ] * 200,
        }
    )

    incoming = reference.copy()

    results = (
        detect_categorical_drift(
            reference,
            incoming,
        )
    )

    assert all(
        result["drifted"]
        is False
        for result in results
    )

# Group-aware splitting

def test_group_split_has_zero_feature_overlap() -> None:
    df = pd.DataFrame(
        {
            "manufacturer": [
                "ford",
                "ford",
                "toyota",
                "toyota",
                "honda",
                "honda",
                "bmw",
                "bmw",
            ],
            "model": [
                "f150",
                "f150",
                "camry",
                "camry",
                "civic",
                "civic",
                "x3",
                "x3",
            ],
            "fuel": [
                "gas",
            ] * 8,
            "title_status": [
                "clean",
            ] * 8,
            "transmission": [
                "automatic",
            ] * 8,
            "drive": [
                "fwd",
            ] * 8,
            "type": [
                "sedan",
            ] * 8,
            "paint_color": [
                "white",
            ] * 8,
            "state": [
                "ny",
            ] * 8,
            "odometer": [
                100000,
                100000,
                80000,
                80000,
                60000,
                60000,
                40000,
                40000,
            ],
            "car_age": [
                10,
                10,
                8,
                8,
                6,
                6,
                4,
                4,
            ],
            "price": [
                10000,
                10000,
                15000,
                15000,
                20000,
                20000,
                25000,
                25000,
            ],
        }
    )

    (
        train_df,
        eval_df,
    ) = group_train_test_split(
        df,
        train_size=0.70,
    )

    (
        overlap_rows,
        overlap_pct,
    ) = calculate_feature_overlap(
        train_df,
        eval_df,
    )

    assert overlap_rows == 0

    assert overlap_pct == 0.0

# Promotion gate

def test_promotion_accepts_meaningfully_better_candidate() -> None:
    production_metrics = {
        "mae": 1800.0,
        "median_ae": 1000.0,
        "rmse": 2500.0,
        "r2": 0.70,
    }

    candidate_metrics = {
        "mae": 1700.0,
        "median_ae": 900.0,
        "rmse": 2400.0,
        "r2": 0.72,
    }

    (
        promoted,
        improvement_pct,
    ) = should_promote(
        production_metrics,
        candidate_metrics,
    )

    assert promoted is True

    assert improvement_pct > 1.0


def test_promotion_rejects_worse_candidate() -> None:
    production_metrics = {
        "mae": 1800.0,
        "median_ae": 1000.0,
        "rmse": 2500.0,
        "r2": 0.70,
    }

    candidate_metrics = {
        "mae": 1900.0,
        "median_ae": 900.0,
        "rmse": 2400.0,
        "r2": 0.72,
    }

    (
        promoted,
        improvement_pct,
    ) = should_promote(
        production_metrics,
        candidate_metrics,
    )

    assert promoted is False

    assert improvement_pct < 0


def test_promotion_rejects_improvement_below_threshold() -> None:
    production_metrics = {
        "mae": 1800.0,
        "median_ae": 1000.0,
        "rmse": 2500.0,
        "r2": 0.70,
    }

    candidate_metrics = {
        # ~0.56% improvement — below 1% gate
        "mae": 1790.0,
        "median_ae": 900.0,
        "rmse": 2400.0,
        "r2": 0.72,
    }

    (
        promoted,
        improvement_pct,
    ) = should_promote(
        production_metrics,
        candidate_metrics,
    )

    assert improvement_pct < 1.0

    assert promoted is False

# API schema

def test_request_rejects_negative_odometer() -> None:
    with pytest.raises(
        ValidationError
    ):
        PricePredictionRequest(
            odometer=-1,
            car_age=4,
        )


def test_request_rejects_negative_car_age() -> None:
    with pytest.raises(
        ValidationError
    ):
        PricePredictionRequest(
            odometer=1000,
            car_age=-1,
        )