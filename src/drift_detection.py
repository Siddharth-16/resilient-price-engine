from __future__ import annotations

import subprocess

import pandas as pd
from scipy.stats import ks_2samp

from src.config import REFERENCED_DATA, NEW_DATA


def detect_drift(reference: pd.DataFrame, new_data: pd.DataFrame) -> list[str]:
    drifted_columns = []

    common_columns = [col for col in reference.columns if col in new_data.columns]

    numeric_columns = [
        col for col in common_columns
        if pd.api.types.is_numeric_dtype(reference[col])
        and pd.api.types.is_numeric_dtype(new_data[col])
    ]

    for col in numeric_columns:
        stat, p_value = ks_2samp(reference[col], new_data[col])

        if p_value < 0.05:
            drifted_columns.append(col)

    return drifted_columns


if __name__ == "__main__":
    reference = pd.read_csv(REFERENCED_DATA)
    new_data = pd.read_csv(NEW_DATA)
    new_data["odometer"] = new_data["odometer"] * 1.5
    new_data["car_age"] = new_data["car_age"] + 5

    drifted_columns = detect_drift(reference, new_data)

    print("Drifted columns:", drifted_columns)

    if len(drifted_columns) >= 2:
        print("Drift detected — retraining model")
        subprocess.run(["python", "-m", "src.train"], check=True)
    else:
        print("No significant drift detected")