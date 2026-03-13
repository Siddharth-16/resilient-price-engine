from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, TEST_SIZE, RANDOM_STATE
from src.utils import ensure_dir

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("price_prediction")

DEFAULT_DATA_PATH = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"

PROD_MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "candidate_price_model.joblib"

PROD_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CANDIDATE_METRICS_PATH = ARTIFACTS_DIR / "candidate_metrics.json"

FEATURES_PATH = ARTIFACTS_DIR / "model_features.joblib"
REFERENCE_DATA_PATH = ARTIFACTS_DIR / "reference_data.csv"
NEW_DATA_PATH = Path("data/new_data.csv")


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in processed dataset.")

    y = df["price"]
    X = df.drop(columns=["price"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train(data_path: Path, candidate: bool = False) -> None:
    print(f"Loading processed dataset from: {data_path}")
    df = load_data(data_path)

    print("Preprocessing dataset...")
    X, y = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    ensure_dir(ARTIFACTS_DIR)

    if not candidate:
        X_train.to_csv(REFERENCE_DATA_PATH, index=False)
        ensure_dir(NEW_DATA_PATH.parent)
        X_test.to_csv(NEW_DATA_PATH, index=False)

    with mlflow.start_run():
        print("Training model...")
        model = RandomForestRegressor(
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )

        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("candidate", candidate)
        mlflow.log_param("training_data_path", str(data_path))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        model.fit(X_train, y_train)

        print("Evaluating model...")
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_metric("num_features", X.shape[1])

        if candidate:
            model_path = CANDIDATE_MODEL_PATH
            metrics_path = CANDIDATE_METRICS_PATH
        else:
            model_path = PROD_MODEL_PATH
            metrics_path = PROD_METRICS_PATH

        print("Saving artifacts...")
        joblib.dump(model, model_path)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)

        metrics = {
            "model": "RandomForestRegressor",
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "num_features": int(X.shape[1]),
            "training_data_path": str(data_path),
            "candidate": candidate,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(FEATURES_PATH))

        print("Training complete.")
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()

    train(Path(args.data_path), candidate=args.candidate)