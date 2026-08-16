from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from src.utils import ensure_dir

# MLflow

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "price_prediction"
)

# Paths

DEFAULT_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "baseline_2000_2012.csv"
)

PROD_MODEL_PATH = (
    ARTIFACTS_DIR
    / "price_model.joblib"
)

CANDIDATE_MODEL_PATH = (
    ARTIFACTS_DIR
    / "candidate_price_model.joblib"
)

PROD_METRICS_PATH = (
    ARTIFACTS_DIR
    / "metrics.json"
)

CANDIDATE_METRICS_PATH = (
    ARTIFACTS_DIR
    / "candidate_metrics.json"
)

BENCHMARK_RESULTS_PATH = (
    ARTIFACTS_DIR
    / "model_benchmark.json"
)

# Historical tuning experiment.
TUNING_RESULTS_PATH = (
    ARTIFACTS_DIR
    / "extra_trees_tuning.json"
)

REFERENCE_DATA_PATH = (
    ARTIFACTS_DIR
    / "reference_data.csv"
)

TRAINING_DATA_PATH = (
    ARTIFACTS_DIR
    / "training_data.csv"
)

EVALUATION_DATA_PATH = (
    ARTIFACTS_DIR
    / "evaluation_data.csv"
)

# Training configuration

TARGET = "price"

EVALUATION_SIZE = 0.20

# Feature grouping

# These are exactly the features used by the model.
# Rows with identical values across all these columns are considered
# one group and must stay entirely on one side of the train/eval split.
FEATURE_GROUP_COLUMNS = [
    "manufacturer",
    "model",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
    "odometer",
    "car_age",
]

# Model names

RANDOM_FOREST = (
    "RandomForestRegressor"
)

EXTRA_TREES = (
    "ExtraTreesRegressor"
)

XGBOOST = (
    "XGBRegressor"
)

BENCHMARK_MODELS = [
    RANDOM_FOREST,
    EXTRA_TREES,
    XGBOOST,
]

# Model configurations

RANDOM_FOREST_PARAMS = {
    "n_estimators": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


EXTRA_TREES_PARAMS = {
    "n_estimators": 50,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Data helpers

def load_data(
    data_path: Path,
) -> pd.DataFrame:

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at "
            f"{data_path}. "
            "Run `python -m src.preprocess` first."
        )

    df = pd.read_csv(
        data_path
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column "
            f"'{TARGET}' in {data_path}"
        )

    if df.empty:
        raise ValueError(
            f"Dataset at {data_path} is empty."
        )

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y

# Group-aware splitting

def validate_group_columns(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in FEATURE_GROUP_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing grouping columns: "
            f"{missing_columns}"
        )


def create_feature_groups(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Create a deterministic group identifier for each unique
    model-feature combination.
    """

    validate_group_columns(
        df
    )

    return pd.util.hash_pandas_object(
        df[
            FEATURE_GROUP_COLUMNS
        ],
        index=False,
    )


def group_train_test_split(
    df: pd.DataFrame,
    train_size: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Group-aware train/evaluation split.

    Identical feature vectors are kept entirely on one side,
    preventing exact feature overlap between training and evaluation.
    """

    groups = (
        create_feature_groups(
            df
        )
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=RANDOM_STATE,
    )

    (
        train_indices,
        evaluation_indices,
    ) = next(
        splitter.split(
            df,
            groups=groups,
        )
    )

    train_df = (
        df.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    evaluation_df = (
        df.iloc[
            evaluation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        train_df,
        evaluation_df,
    )


def calculate_feature_overlap(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> tuple[int, float]:
    """
    Verify that no exact model-feature vector exists on both sides
    of the split.
    """

    train_unique = (
        train_df[
            FEATURE_GROUP_COLUMNS
        ]
        .drop_duplicates()
    )

    overlap = (
        evaluation_df.merge(
            train_unique,
            on=FEATURE_GROUP_COLUMNS,
            how="inner",
        )
    )

    overlap_rows = int(
        len(overlap)
    )

    overlap_pct = (
        overlap_rows
        / len(evaluation_df)
        * 100
        if len(evaluation_df)
        else 0.0
    )

    return (
        overlap_rows,
        float(overlap_pct),
    )

# Production configuration helpers

def get_production_model_name() -> str:

    if not PROD_METRICS_PATH.exists():
        return EXTRA_TREES

    try:
        metrics = json.loads(
            PROD_METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

        model_name = metrics.get(
            "model",
            EXTRA_TREES,
        )

        if model_name in BENCHMARK_MODELS:
            return model_name

    except (
        json.JSONDecodeError,
        OSError,
    ):
        pass

    return EXTRA_TREES


def get_production_hyperparameters() -> dict | None:

    if not PROD_METRICS_PATH.exists():
        return None

    try:
        metrics = json.loads(
            PROD_METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

        hyperparameters = (
            metrics.get(
                "hyperparameters"
            )
        )

        if isinstance(
            hyperparameters,
            dict,
        ):
            return hyperparameters

    except (
        json.JSONDecodeError,
        OSError,
    ):
        pass

    return None

# Model construction

def create_estimator(
    model_name: str,
    hyperparameters: dict | None = None,
):

    if model_name == RANDOM_FOREST:

        params = (
            RANDOM_FOREST_PARAMS.copy()
        )

        if hyperparameters:
            params.update(
                hyperparameters
            )

        params["random_state"] = (
            RANDOM_STATE
        )

        params["n_jobs"] = -1

        return RandomForestRegressor(
            **params
        )

    if model_name == EXTRA_TREES:

        params = (
            EXTRA_TREES_PARAMS.copy()
        )

        if hyperparameters:
            params.update(
                hyperparameters
            )

        params["random_state"] = (
            RANDOM_STATE
        )

        params["n_jobs"] = -1

        return ExtraTreesRegressor(
            **params
        )

    if model_name == XGBOOST:

        params = (
            XGBOOST_PARAMS.copy()
        )

        if hyperparameters:
            params.update(
                hyperparameters
            )

        params["random_state"] = (
            RANDOM_STATE
        )

        params["n_jobs"] = -1

        return XGBRegressor(
            **params
        )

    raise ValueError(
        f"Unsupported model: "
        f"{model_name}"
    )


def build_pipeline(
    X: pd.DataFrame,
    model_name: str | None = None,
    hyperparameters: dict | None = None,
) -> Pipeline:

    if model_name is None:

        model_name = (
            get_production_model_name()
        )

        if hyperparameters is None:
            hyperparameters = (
                get_production_hyperparameters()
            )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                    categorical_columns,
                ),
                (
                    "numeric",
                    "passthrough",
                    numeric_columns,
                ),
            ],
            remainder="drop",
        )
    )

    estimator = (
        create_estimator(
            model_name=model_name,
            hyperparameters=hyperparameters,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )

# Metrics

def calculate_regression_metrics(
    y_true,
    predictions,
) -> dict:

    mae = float(
        mean_absolute_error(
            y_true,
            predictions,
        )
    )

    median_ae = float(
        median_absolute_error(
            y_true,
            predictions,
        )
    )

    rmse = float(
        mean_squared_error(
            y_true,
            predictions,
        )
        ** 0.5
    )

    r2 = float(
        r2_score(
            y_true,
            predictions,
        )
    )

    return {
        "mae": mae,
        "median_ae": median_ae,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
) -> dict:

    X, y = (
        split_features_target(
            df
        )
    )

    predictions = (
        model.predict(
            X
        )
    )

    return (
        calculate_regression_metrics(
            y,
            predictions,
        )
    )

# File helper

def write_metrics(
    path: Path,
    metrics: dict,
) -> None:

    path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

# MLflow model logging

def log_mlflow_model(
    model: Pipeline,
    X_sample: pd.DataFrame,
) -> None:

    sample_input = (
        X_sample.head(5)
    )

    sample_predictions = (
        model.predict(
            sample_input
        )
    )

    signature = infer_signature(
        sample_input,
        sample_predictions,
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=sample_input,
    )

# Benchmark printing

def print_benchmark_results(
    benchmark_results: dict,
) -> None:

    print()
    print("=" * 90)
    print(
        "BASELINE MODEL BENCHMARK"
    )
    print("=" * 90)

    print(
        f"{'Model':<28}"
        f"{'MAE':>15}"
        f"{'Median AE':>15}"
        f"{'RMSE':>15}"
        f"{'R²':>12}"
    )

    print("-" * 90)

    for model_name in BENCHMARK_MODELS:

        metrics = (
            benchmark_results[
                model_name
            ]
        )

        print(
            f"{model_name:<28}"
            f"{metrics['mae']:>15,.2f}"
            f"{metrics['median_ae']:>15,.2f}"
            f"{metrics['rmse']:>15,.2f}"
            f"{metrics['r2']:>12.4f}"
        )

    print("=" * 90)

# Model benchmark

def benchmark_models(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> tuple[
    Pipeline,
    str,
    dict,
]:

    X_train, y_train = (
        split_features_target(
            train_df
        )
    )

    benchmark_results = {}
    fitted_models = {}

    for model_name in (
        BENCHMARK_MODELS
    ):

        print()
        print(
            f"Training "
            f"{model_name}..."
        )

        model = (
            build_pipeline(
                X_train,
                model_name=model_name,
                hyperparameters=None,
            )
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            f"Evaluating "
            f"{model_name}..."
        )

        evaluation_metrics = (
            evaluate_model(
                model,
                evaluation_df,
            )
        )

        benchmark_results[
            model_name
        ] = evaluation_metrics

        fitted_models[
            model_name
        ] = model

        print(
            f"  MAE: "
            f"${evaluation_metrics['mae']:,.2f}"
        )

        print(
            f"  Median AE: "
            f"${evaluation_metrics['median_ae']:,.2f}"
        )

        print(
            f"  RMSE: "
            f"${evaluation_metrics['rmse']:,.2f}"
        )

        print(
            f"  R²: "
            f"{evaluation_metrics['r2']:.4f}"
        )

    selected_model_name = min(
        benchmark_results,
        key=lambda name: (
            benchmark_results[
                name
            ]["mae"]
        ),
    )

    selected_model = (
        fitted_models[
            selected_model_name
        ]
    )

    return (
        selected_model,
        selected_model_name,
        benchmark_results,
    )


# Production training

def train_production(
    df: pd.DataFrame,
    data_path: Path,
) -> None:

    # Group-aware baseline split

    (
        train_df,
        evaluation_df,
    ) = group_train_test_split(
        df,
        train_size=(
            1.0
            - EVALUATION_SIZE
        ),
    )

    (
        overlap_rows,
        overlap_pct,
    ) = calculate_feature_overlap(
        train_df,
        evaluation_df,
    )

    print()
    print(
        "Baseline group-aware split:"
    )

    print(
        f"  Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"  Evaluation rows: "
        f"{len(evaluation_df):,}"
    )

    print(
        f"  Feature overlap rows: "
        f"{overlap_rows:,}"
    )

    print(
        f"  Feature overlap rate: "
        f"{overlap_pct:.2f}%"
    )

    if overlap_rows != 0:
        raise RuntimeError(
            "Group-aware split failed: "
            "training/evaluation feature "
            "overlap is not zero."
        )

    ensure_dir(
        ARTIFACTS_DIR
    )

    # Save baseline datasets

    train_df.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )

    evaluation_df.to_csv(
        EVALUATION_DATA_PATH,
        index=False,
    )

    train_df.drop(
        columns=[TARGET]
    ).to_csv(
        REFERENCE_DATA_PATH,
        index=False,
    )

    X_train, y_train = (
        split_features_target(
            train_df
        )
    )

    with mlflow.start_run(
        run_name=(
            "baseline_model_selection"
        )
    ):

        mlflow.log_param(
            "cohort",
            "2000-2012",
        )

        mlflow.log_param(
            "candidate",
            False,
        )

        mlflow.log_param(
            "split_strategy",
            "group_shuffle_split",
        )

        mlflow.log_param(
            "evaluation_size",
            EVALUATION_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "selection_metric",
            "mae",
        )

        mlflow.log_param(
            "benchmark_models",
            ",".join(
                BENCHMARK_MODELS
            ),
        )

        mlflow.log_metric(
            "feature_overlap_pct",
            overlap_pct,
        )

        # Benchmark

        (
            selected_model,
            selected_model_name,
            benchmark_results,
        ) = benchmark_models(
            train_df,
            evaluation_df,
        )

        print_benchmark_results(
            benchmark_results
        )

        benchmark_report = {
            "cohort": "2000-2012",

            "split_strategy": (
                "group_shuffle_split"
            ),

            "feature_group_columns": (
                FEATURE_GROUP_COLUMNS
            ),

            "training_rows": int(
                len(train_df)
            ),

            "evaluation_rows": int(
                len(evaluation_df)
            ),

            "feature_overlap_rows": (
                overlap_rows
            ),

            "feature_overlap_pct": (
                overlap_pct
            ),

            "selection_metric": (
                "mae"
            ),

            "results": (
                benchmark_results
            ),

            "selected_model": (
                selected_model_name
            ),
        }

        write_metrics(
            BENCHMARK_RESULTS_PATH,
            benchmark_report,
        )

        # Store actual selected hyperparameters

        if (
            selected_model_name
            == EXTRA_TREES
        ):

            selected_hyperparameters = {
                "n_estimators": 50,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": 1.0,
            }

        elif (
            selected_model_name
            == RANDOM_FOREST
        ):

            selected_hyperparameters = {
                "n_estimators": 50,
            }

        elif (
            selected_model_name
            == XGBOOST
        ):

            selected_hyperparameters = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": (
                    "reg:squarederror"
                ),
            }

        else:
            raise ValueError(
                "Unsupported selected model."
            )

        # Evaluation metrics already calculated during benchmark

        selected_evaluation_metrics = (
            benchmark_results[
                selected_model_name
            ]
        )

        print()
        print(
            "Calculating training metrics "
            "for selected production model..."
        )

        train_predictions = (
            selected_model.predict(
                X_train
            )
        )

        train_metrics = (
            calculate_regression_metrics(
                y_train,
                train_predictions,
            )
        )

        preprocessor = (
            selected_model.named_steps[
                "preprocessor"
            ]
        )

        num_features = len(
            preprocessor
            .get_feature_names_out()
        )

        # Production metadata

        production_metrics = {
            "model": (
                selected_model_name
            ),

            "cohort": (
                "2000-2012"
            ),

            "split_strategy": (
                "group_shuffle_split"
            ),

            "feature_group_columns": (
                FEATURE_GROUP_COLUMNS
            ),

            "feature_overlap_rows": (
                overlap_rows
            ),

            "feature_overlap_pct": (
                overlap_pct
            ),

            "train_rows": int(
                len(train_df)
            ),

            "evaluation_rows": int(
                len(evaluation_df)
            ),

            "num_features": int(
                num_features
            ),

            "training_data_path": (
                str(data_path)
            ),

            "candidate": False,

            "selection_metric": (
                "mae"
            ),

            "hyperparameters": (
                selected_hyperparameters
            ),

            "train_metrics": (
                train_metrics
            ),

            "evaluation_metrics": (
                selected_evaluation_metrics
            ),

            # Compatibility field.
            "evaluation_mae": (
                selected_evaluation_metrics[
                    "mae"
                ]
            ),

            "tuning_decision": {
                "selected_estimators": 50,
                "tested_estimators": 100,
                "previous_50_tree_mae": (
                    1455.86
                ),
                "previous_100_tree_mae": (
                    1451.69
                ),
                "previous_improvement_pct": (
                    0.29
                ),
                "reason": (
                    "100 trees improved MAE "
                    "by only about 0.29%; "
                    "50 trees retained for "
                    "cheaper retraining."
                ),
            },
        }

        # Save production model and metadata

        joblib.dump(
            selected_model,
            PROD_MODEL_PATH,
        )

        write_metrics(
            PROD_METRICS_PATH,
            production_metrics,
        )

        # MLflow

        log_mlflow_model(
            selected_model,
            X_train,
        )

        for (
            model_name,
            metrics,
        ) in benchmark_results.items():

            safe_name = (
                model_name
                .replace(
                    "Regressor",
                    "",
                )
                .lower()
            )

            mlflow.log_metric(
                f"{safe_name}_mae",
                metrics["mae"],
            )

            mlflow.log_metric(
                f"{safe_name}_median_ae",
                metrics["median_ae"],
            )

            mlflow.log_metric(
                f"{safe_name}_rmse",
                metrics["rmse"],
            )

            mlflow.log_metric(
                f"{safe_name}_r2",
                metrics["r2"],
            )

        mlflow.log_param(
            "selected_model",
            selected_model_name,
        )

        for (
            parameter,
            value,
        ) in (
            selected_hyperparameters.items()
        ):

            mlflow.log_param(
                f"production_{parameter}",
                value,
            )

        mlflow.log_metric(
            "evaluation_mae",
            selected_evaluation_metrics[
                "mae"
            ],
        )

        mlflow.log_metric(
            "evaluation_median_ae",
            selected_evaluation_metrics[
                "median_ae"
            ],
        )

        mlflow.log_metric(
            "evaluation_rmse",
            selected_evaluation_metrics[
                "rmse"
            ],
        )

        mlflow.log_metric(
            "evaluation_r2",
            selected_evaluation_metrics[
                "r2"
            ],
        )

        mlflow.log_metric(
            "train_mae",
            train_metrics[
                "mae"
            ],
        )

        mlflow.log_metric(
            "train_median_ae",
            train_metrics[
                "median_ae"
            ],
        )

        mlflow.log_metric(
            "train_rmse",
            train_metrics[
                "rmse"
            ],
        )

        mlflow.log_metric(
            "train_r2",
            train_metrics[
                "r2"
            ],
        )

        mlflow.log_artifact(
            str(
                PROD_METRICS_PATH
            )
        )

        mlflow.log_artifact(
            str(
                BENCHMARK_RESULTS_PATH
            )
        )

        # Preserve previous tuning report if present.
        if TUNING_RESULTS_PATH.exists():

            mlflow.log_artifact(
                str(
                    TUNING_RESULTS_PATH
                )
            )

    # Console summary

    print()
    print("=" * 90)
    print(
        "BASELINE TRAINING COMPLETE"
    )
    print("=" * 90)

    print(
        f"Total baseline rows: "
        f"{len(df):,}"
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Evaluation rows: "
        f"{len(evaluation_df):,}"
    )

    print(
        f"Feature overlap rate: "
        f"{overlap_pct:.2f}%"
    )

    print(
        f"Encoded features: "
        f"{num_features:,}"
    )

    print()
    print(
        f"Selected production model: "
        f"{selected_model_name}"
    )

    print()
    print(
        "Production hyperparameters:"
    )

    for (
        parameter,
        value,
    ) in (
        selected_hyperparameters.items()
    ):

        print(
            f"  {parameter}: "
            f"{value}"
        )

    print()
    print(
        "Training metrics:"
    )

    print(
        f"  MAE: "
        f"${train_metrics['mae']:,.2f}"
    )

    print(
        f"  Median AE: "
        f"${train_metrics['median_ae']:,.2f}"
    )

    print(
        f"  RMSE: "
        f"${train_metrics['rmse']:,.2f}"
    )

    print(
        f"  R²: "
        f"{train_metrics['r2']:.4f}"
    )

    print()
    print(
        "Evaluation metrics:"
    )

    print(
        f"  MAE: "
        f"${selected_evaluation_metrics['mae']:,.2f}"
    )

    print(
        f"  Median AE: "
        f"${selected_evaluation_metrics['median_ae']:,.2f}"
    )

    print(
        f"  RMSE: "
        f"${selected_evaluation_metrics['rmse']:,.2f}"
    )

    print(
        f"  R²: "
        f"{selected_evaluation_metrics['r2']:.4f}"
    )

    print()
    print(
        f"Saved production model to: "
        f"{PROD_MODEL_PATH}"
    )

    print(
        f"Saved benchmark report to: "
        f"{BENCHMARK_RESULTS_PATH}"
    )

# Standalone candidate training

def train_candidate(
    df: pd.DataFrame,
    data_path: Path,
) -> None:

    ensure_dir(
        ARTIFACTS_DIR
    )

    X_train, y_train = (
        split_features_target(
            df
        )
    )

    model_name = (
        get_production_model_name()
    )

    hyperparameters = (
        get_production_hyperparameters()
    )

    model = build_pipeline(
        X_train,
        model_name=model_name,
        hyperparameters=hyperparameters,
    )

    print(
        f"Training candidate "
        f"{model_name}..."
    )

    print(
        f"Using hyperparameters: "
        f"{hyperparameters}"
    )

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        model,
        CANDIDATE_MODEL_PATH,
    )

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    num_features = len(
        preprocessor
        .get_feature_names_out()
    )

    metrics = {
        "model": (
            model_name
        ),

        "hyperparameters": (
            hyperparameters
        ),

        "train_rows": int(
            len(df)
        ),

        "num_features": int(
            num_features
        ),

        "training_data_path": (
            str(data_path)
        ),

        "candidate": True,
    }

    write_metrics(
        CANDIDATE_METRICS_PATH,
        metrics,
    )

    print()
    print(
        "Candidate training complete."
    )

# Main

def train(
    data_path: Path,
    candidate: bool = False,
) -> None:

    print(
        f"Loading processed dataset from: "
        f"{data_path}"
    )

    df = load_data(
        data_path
    )

    if candidate:

        train_candidate(
            df,
            data_path,
        )

    else:

        train_production(
            df,
            data_path,
        )

# CLI

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and train "
            "used-car price models."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--candidate",
        action="store_true",
    )

    args = parser.parse_args()

    train(
        args.data_path,
        candidate=args.candidate,
    )