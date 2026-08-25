from pathlib import Path
import sys

import joblib
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import (
    EVALUATION_DATA_PATH,
    PROD_MODEL_PATH,
    TARGET,
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_predictions.csv"
)


# ------------------------------------------------------------
# Export model predictions for SQL analysis
# ------------------------------------------------------------

def export_predictions() -> pd.DataFrame:
    """
    Generate predictions on the held-out evaluation dataset and
    export row-level prediction errors for downstream SQL analysis.

    Residual definition:

        residual = actual_price - predicted_price

    Interpretation:

        residual > 0
            Model underpredicted the vehicle price.

        residual < 0
            Model overpredicted the vehicle price.

        residual = 0
            Prediction exactly matched the listing price.
    """

    # --------------------------------------------------------
    # Validate required artifacts
    # --------------------------------------------------------

    if not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at "
            f"{EVALUATION_DATA_PATH}.\n"
            "Run `python -m src.train` first."
        )

    if not PROD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found at "
            f"{PROD_MODEL_PATH}.\n"
            "Run `python -m src.train` first."
        )

    # --------------------------------------------------------
    # Load held-out evaluation data
    # --------------------------------------------------------

    df = pd.read_csv(EVALUATION_DATA_PATH)

    if df.empty:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found "
            "in evaluation dataset."
        )

    # --------------------------------------------------------
    # Load production model
    # --------------------------------------------------------

    model = joblib.load(PROD_MODEL_PATH)

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    X = df.drop(columns=[TARGET])

    predictions = model.predict(X)

    # --------------------------------------------------------
    # Build analysis dataset
    # --------------------------------------------------------

    result = df.copy()

    result["predicted_price"] = predictions

    # residual = actual - predicted
    result["residual"] = (
        result[TARGET]
        - result["predicted_price"]
    )

    # Prediction error regardless of direction
    result["absolute_error"] = (
        result["residual"].abs()
    )

    # Error relative to actual listing price
    result["absolute_percentage_error"] = (
        result["absolute_error"]
        / result[TARGET]
        * 100
    )

    # Positive value means listing price is below
    # the model's predicted price.
    result["listing_gap_pct"] = (
        (
            result["predicted_price"]
            - result[TARGET]
        )
        / result["predicted_price"]
        * 100
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("Prediction export complete")
    print("-" * 50)

    print(
        f"Evaluation rows: "
        f"{len(result):,}"
    )

    print(
        f"Mean absolute error: "
        f"${result['absolute_error'].mean():,.2f}"
    )

    print(
        f"Median absolute error: "
        f"${result['absolute_error'].median():,.2f}"
    )

    print(
        f"Mean residual: "
        f"${result['residual'].mean():,.2f}"
    )

    print(
        f"Saved predictions to: "
        f"{OUTPUT_PATH}"
    )

    return result


if __name__ == "__main__":
    export_predictions()