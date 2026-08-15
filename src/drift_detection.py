from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR

REFERENCE_DATA = ARTIFACTS_DIR / "reference_data.csv"
TRAINING_DATA = ARTIFACTS_DIR / "training_data.csv"
EVALUATION_DATA = ARTIFACTS_DIR / "evaluation_data.csv"
NEW_DATA = Path("data/new_data.csv")
RETRAIN_DATA = PROCESSED_DATA_DIR / "retrain_data.csv"

PROD_MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "candidate_price_model.joblib"
PROD_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CANDIDATE_METRICS_PATH = ARTIFACTS_DIR / "candidate_metrics.json"

TARGET = "price"
DRIFT_P_VALUE = 0.05
MIN_DRIFTED_COLUMNS = 2
MIN_MAE_IMPROVEMENT = 0.0


def detect_drift(reference: pd.DataFrame, new_data: pd.DataFrame) -> list[str]:
    drifted_columns: list[str] = []
    common_columns = [column for column in reference.columns if column in new_data.columns]

    numeric_columns = [
        column
        for column in common_columns
        if pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(new_data[column])
    ]

    for column in numeric_columns:
        reference_values = reference[column].dropna()
        new_values = new_data[column].dropna()
        if reference_values.empty or new_values.empty:
            continue
        _, p_value = ks_2samp(reference_values, new_values)
        if p_value < DRIFT_P_VALUE:
            drifted_columns.append(column)

    return drifted_columns


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_model(model, evaluation_data: pd.DataFrame) -> float:
    if TARGET not in evaluation_data.columns:
        raise ValueError(f"Evaluation data must contain '{TARGET}'.")
    X_eval = evaluation_data.drop(columns=[TARGET])
    y_eval = evaluation_data[TARGET]
    return float(mean_absolute_error(y_eval, model.predict(X_eval)))


def should_promote(prod_mae: float, candidate_mae: float, min_improvement: float = 0.0) -> bool:
    return candidate_mae <= prod_mae - min_improvement


def promote_candidate_if_better() -> bool:
    evaluation_data = pd.read_csv(EVALUATION_DATA)
    prod_model = joblib.load(PROD_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    prod_mae = evaluate_model(prod_model, evaluation_data)
    candidate_mae = evaluate_model(candidate_model, evaluation_data)

    candidate_metrics = load_metrics(CANDIDATE_METRICS_PATH)
    candidate_metrics["evaluation_mae"] = candidate_mae
    candidate_metrics["promotion_baseline_mae"] = prod_mae
    CANDIDATE_METRICS_PATH.write_text(
        json.dumps(candidate_metrics, indent=2), encoding="utf-8"
    )

    print(f"Production evaluation MAE: {prod_mae:.2f}")
    print(f"Candidate evaluation MAE: {candidate_mae:.2f}")

    if not should_promote(prod_mae, candidate_mae, MIN_MAE_IMPROVEMENT):
        print("Candidate model is not better. Keeping current production model.")
        return False

    print("Candidate model is better. Promoting candidate to production.")
    shutil.copy2(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)

    promoted_metrics = candidate_metrics | {
        "candidate": False,
        "test_mae": candidate_mae,
        "evaluation_mae": candidate_mae,
        "promoted_from_candidate": True,
    }
    PROD_METRICS_PATH.write_text(json.dumps(promoted_metrics, indent=2), encoding="utf-8")
    return True


def build_simulated_drift_batch(new_data: pd.DataFrame) -> pd.DataFrame:
    drifted = new_data.copy()
    if "odometer" in drifted.columns:
        drifted["odometer"] = drifted["odometer"] * 1.5
    if "car_age" in drifted.columns:
        drifted["car_age"] = drifted["car_age"] + 5
    return drifted


def run_monitoring_cycle() -> None:
    reference = pd.read_csv(REFERENCE_DATA)
    incoming = pd.read_csv(NEW_DATA)
    simulated_incoming = build_simulated_drift_batch(incoming)

    drift_features = simulated_incoming.drop(columns=[TARGET], errors="ignore")
    drifted_columns = detect_drift(reference, drift_features)
    print("Drifted columns:", drifted_columns)

    if len(drifted_columns) < MIN_DRIFTED_COLUMNS:
        print("No significant drift detected")
        return

    if TARGET not in simulated_incoming.columns:
        print("Drift detected, but labels are unavailable; skipping retraining safely.")
        return

    print("Drift detected — preparing labeled retraining dataset")
    training_data = pd.read_csv(TRAINING_DATA)
    retrain_data = pd.concat([training_data, simulated_incoming], ignore_index=True)
    RETRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    retrain_data.to_csv(RETRAIN_DATA, index=False)

    subprocess.run(
        [
            "python",
            "-m",
            "src.train",
            "--data-path",
            str(RETRAIN_DATA),
            "--candidate",
        ],
        check=True,
    )
    promote_candidate_if_better()


if __name__ == "__main__":
    run_monitoring_cycle()
