from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis_vehicle_data.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_predictions.csv"
)


def import_csv_files() -> None:
    """Import analysis and prediction datasets into MySQL."""

    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found in the project .env file."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {DATA_PATH}. "
            "Run `python -m src.preprocess` first."
        )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction dataset not found at {PREDICTION_PATH}. "
            "Run `python sql_analysis/export_predictions.py` first."
        )

    listings = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)

    if listings.empty:
        raise ValueError("The analysis dataset is empty.")

    if predictions.empty:
        raise ValueError("The prediction dataset is empty.")

    required_listing_columns = {
        "id",
        "year",
        "price",
        "odometer",
        "car_age",
    }

    missing_listing_columns = sorted(
        required_listing_columns.difference(listings.columns)
    )

    if missing_listing_columns:
        raise ValueError(
            "Analysis dataset is missing required columns: "
            f"{missing_listing_columns}"
        )

    required_prediction_columns = {
        "price",
        "predicted_price",
        "residual",
        "absolute_error",
    }

    missing_prediction_columns = sorted(
        required_prediction_columns.difference(
            predictions.columns
        )
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction dataset is missing required columns: "
            f"{missing_prediction_columns}"
        )

    engine = create_engine(database_url)

    with engine.begin() as connection:
        listings.to_sql(
            "listings",
            con=connection,
            if_exists="replace",
            index=False,
        )

        predictions.to_sql(
            "model_predictions",
            con=connection,
            if_exists="replace",
            index=False,
        )

    print(
        f"Imported {len(listings):,} rows into `listings`."
    )

    print(
        f"Imported {len(predictions):,} rows into "
        "`model_predictions`."
    )


if __name__ == "__main__":
    import_csv_files()