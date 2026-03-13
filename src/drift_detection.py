from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR

REFERENCE_DATA = ARTIFACTS_DIR / "reference_data.csv"
NEW_DATA = Path("data/new_data.csv")
ORIGINAL_DATA = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"
RETRAIN_DATA = PROCESSED_DATA_DIR / "retrain_data.csv"

PROD_MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "candidate_price_model.joblib"

PROD_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CANDIDATE_METRICS_PATH = ARTIFACTS_DIR / "candidate_metrics.json"


def detect_drift(reference: pd.DataFrame, new_data: pd.DataFrame) -> list[str]:
    drifted_columns = []

    common_columns = [col for col in reference.columns if col in new_data.columns]

    numeric_columns = [
        col for col in common_columns
        if pd.api.types.is_numeric_dtype(reference[col])
        and pd.api.types.is_numeric_dtype(new_data[col])
    ]

    for col in numeric_columns:
        _, p_value = ks_2samp(reference[col], new_data[col])
        if p_value < 0.05:
            drifted_columns.append(col)

    return drifted_columns


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def promote_candidate_if_better() -> None:
    prod_metrics = load_metrics(PROD_METRICS_PATH)
    candidate_metrics = load_metrics(CANDIDATE_METRICS_PATH)

    prod_mae = prod_metrics["test_mae"]
    candidate_mae = candidate_metrics["test_mae"]

    print(f"Production test MAE: {prod_mae:.2f}")
    print(f"Candidate test MAE: {candidate_mae:.2f}")

    if candidate_mae < prod_mae:
        print("Candidate model is better. Promoting candidate to production.")
        shutil.copy2(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)
        shutil.copy2(CANDIDATE_METRICS_PATH, PROD_METRICS_PATH)
    else:
        print("Candidate model is not better. Keeping current production model.")


if __name__ == "__main__":
    reference = pd.read_csv(REFERENCE_DATA)
    new_data = pd.read_csv(NEW_DATA)

    # simulate drift for demo
    new_data["odometer"] = new_data["odometer"] * 1.5
    new_data["car_age"] = new_data["car_age"] + 5

    drifted_columns = detect_drift(reference, new_data)

    print("Drifted columns:", drifted_columns)

    if len(drifted_columns) >= 2:
        print("Drift detected — preparing retraining dataset")

        original_data = pd.read_csv(ORIGINAL_DATA)

        # simple demo retraining dataset
        retrain_data = pd.concat(
            [original_data, original_data.sample(frac=0.2, random_state=42)],
            ignore_index=True,
        )
        retrain_data.to_csv(RETRAIN_DATA, index=False)

        print("Training candidate model...")
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
    else:
        print("No significant drift detected")