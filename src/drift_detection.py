from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR

REFERENCE_DATA = ARTIFACTS_DIR / "reference_data.csv"
NEW_DATA = Path("data/new_data.csv")
ORIGINAL_DATA = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"
RETRAIN_DATA = PROCESSED_DATA_DIR / "retrain_data.csv"


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


if __name__ == "__main__":
    reference = pd.read_csv(REFERENCE_DATA)
    new_data = pd.read_csv(NEW_DATA)

    new_data["odometer"] = new_data["odometer"] * 1.5
    new_data["car_age"] = new_data["car_age"] + 5

    drifted_columns = detect_drift(reference, new_data)

    print("Drifted columns:", drifted_columns)

    if len(drifted_columns) >= 2:
        print("Drift detected — preparing retraining dataset")

        original_data = pd.read_csv(ORIGINAL_DATA)

        retrain_data = pd.concat([original_data, original_data.sample(frac=0.1, random_state=42)], ignore_index=True)
        retrain_data.to_csv(RETRAIN_DATA, index=False)

        print(f"Retraining on: {RETRAIN_DATA}")
        subprocess.run(
            ["python", "-m", "src.train", "--data-path", str(RETRAIN_DATA)],
            check=True,
        )
    else:
        print("No significant drift detected")