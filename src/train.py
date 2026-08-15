from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from src.utils import ensure_dir

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("price_prediction")

DEFAULT_DATA_PATH = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"

PROD_MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "candidate_price_model.joblib"
PROD_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CANDIDATE_METRICS_PATH = ARTIFACTS_DIR / "candidate_metrics.json"
REFERENCE_DATA_PATH = ARTIFACTS_DIR / "reference_data.csv"
TRAINING_DATA_PATH = ARTIFACTS_DIR / "training_data.csv"
EVALUATION_DATA_PATH = ARTIFACTS_DIR / "evaluation_data.csv"
NEW_DATA_PATH = Path("data/new_data.csv")

TARGET = "price"
EVAL_SIZE = 0.15
DRIFT_SIZE = 0.15


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected '{TARGET}' column in processed dataset.")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=50,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_model(df: pd.DataFrame) -> Pipeline:
    X, y = split_features_target(df)
    model = build_pipeline(X)
    model.fit(X, y)
    return model


def evaluate_model(model: Pipeline, df: pd.DataFrame) -> float:
    X, y = split_features_target(df)
    return float(mean_absolute_error(y, model.predict(X)))


def _write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_production(df: pd.DataFrame, data_path: Path) -> None:
    # Reserve a stable promotion set and a separate simulated incoming batch.
    train_df, holdout_df = train_test_split(
        df,
        test_size=EVAL_SIZE + DRIFT_SIZE,
        random_state=RANDOM_STATE,
    )
    evaluation_df, drift_df = train_test_split(
        holdout_df,
        test_size=DRIFT_SIZE / (EVAL_SIZE + DRIFT_SIZE),
        random_state=RANDOM_STATE,
    )

    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(NEW_DATA_PATH.parent)

    train_df.to_csv(TRAINING_DATA_PATH, index=False)
    evaluation_df.to_csv(EVALUATION_DATA_PATH, index=False)
    drift_df.to_csv(NEW_DATA_PATH, index=False)
    train_df.drop(columns=[TARGET]).to_csv(REFERENCE_DATA_PATH, index=False)

    X_train, y_train = split_features_target(train_df)
    model = build_pipeline(X_train)

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("training_data_path", str(data_path))
        mlflow.log_param("random_state", RANDOM_STATE)

        model.fit(X_train, y_train)
        train_mae = float(mean_absolute_error(y_train, model.predict(X_train)))
        eval_mae = evaluate_model(model, evaluation_df)

        preprocessor = model.named_steps["preprocessor"]
        num_features = len(preprocessor.get_feature_names_out())

        metrics = {
            "model": "RandomForestRegressor",
            "train_mae": train_mae,
            "test_mae": eval_mae,
            "evaluation_mae": eval_mae,
            "train_rows": int(len(train_df)),
            "evaluation_rows": int(len(evaluation_df)),
            "drift_rows": int(len(drift_df)),
            "num_features": int(num_features),
            "training_data_path": str(data_path),
            "candidate": False,
        }

        joblib.dump(model, PROD_MODEL_PATH)
        _write_metrics(PROD_METRICS_PATH, metrics)

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("evaluation_mae", eval_mae)
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("evaluation_rows", len(evaluation_df))
        mlflow.log_metric("num_features", num_features)
        mlflow.log_artifact(str(PROD_MODEL_PATH))
        mlflow.log_artifact(str(PROD_METRICS_PATH))

    print(f"Train MAE: {train_mae:.2f}")
    print(f"Evaluation MAE: {eval_mae:.2f}")
    print(f"Saved production model to: {PROD_MODEL_PATH}")


def train_candidate(df: pd.DataFrame, data_path: Path) -> None:
    ensure_dir(ARTIFACTS_DIR)
    X, y = split_features_target(df)
    model = build_pipeline(X)

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("candidate", True)
        mlflow.log_param("training_data_path", str(data_path))

        model.fit(X, y)
        train_mae = float(mean_absolute_error(y, model.predict(X)))
        metrics = {
            "model": "RandomForestRegressor",
            "train_mae": train_mae,
            "train_rows": int(len(df)),
            "training_data_path": str(data_path),
            "candidate": True,
        }

        joblib.dump(model, CANDIDATE_MODEL_PATH)
        _write_metrics(CANDIDATE_METRICS_PATH, metrics)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_rows", len(df))
        mlflow.log_artifact(str(CANDIDATE_MODEL_PATH))
        mlflow.log_artifact(str(CANDIDATE_METRICS_PATH))

    print(f"Candidate train MAE: {train_mae:.2f}")
    print(f"Saved candidate model to: {CANDIDATE_MODEL_PATH}")


def train(data_path: Path, candidate: bool = False) -> None:
    print(f"Loading processed dataset from: {data_path}")
    df = load_data(data_path)
    if candidate:
        train_candidate(df, data_path)
    else:
        train_production(df, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    train(Path(args.data_path), candidate=args.candidate)
