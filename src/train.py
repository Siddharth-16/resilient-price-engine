from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE
from src.utils import ensure_dir

PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "model_features.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_processed_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {file_path}\n"
            "Make sure you saved the cleaned dataset from your notebook first."
        )
    return pd.read_csv(file_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "price" not in df.columns:
        raise ValueError("Expected target column 'price' in processed dataset.")

    X = df.drop(columns=["price"])
    X = pd.get_dummies(X, drop_first=True)
    y = df["price"]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    return {
        "model": "RandomForestRegressor",
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
    }


def save_artifacts(
    model: RandomForestRegressor,
    feature_columns: list[str],
    metrics: dict,
) -> None:
    ensure_dir(ARTIFACTS_DIR)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    print(f"Loading processed dataset from: {PROCESSED_DATASET_PATH}")
    df = load_processed_data(PROCESSED_DATASET_PATH)

    print(f"Dataset shape: {df.shape}")

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Training RandomForestRegressor...")
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    save_artifacts(model, X.columns.tolist(), metrics)

    print("Training complete.")
    print(f"Train MAE: {metrics['train_mae']:.2f}")
    print(f"Test MAE:  {metrics['test_mae']:.2f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved features to: {FEATURES_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()