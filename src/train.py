from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR, TEST_SIZE, RANDOM_STATE
from src.utils import ensure_dir

DATA_PATH = PROCESSED_DATA_DIR / "clean_vehicle_data.csv"
MODEL_PATH = ARTIFACTS_DIR / "price_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "model_features.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "price" not in df.columns:
        raise ValueError("Expected 'price' column in processed dataset.")

    y = df["price"]
    X = df.drop(columns=["price"])

    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train() -> None:
    print("Loading processed dataset...")
    df = load_data()

    print("Preprocessing dataset...")
    X, y = preprocess_data(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)

    ensure_dir(ARTIFACTS_DIR)

    print("Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)

    metrics = {
        "model": "RandomForestRegressor",
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "num_features": int(X.shape[1]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved features to: {FEATURES_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    train()