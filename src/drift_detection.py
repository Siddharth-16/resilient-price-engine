from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from scipy.spatial.distance import (
    jensenshannon,
)
from scipy.stats import (
    ks_2samp,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
)

from src.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

from src.train import (
    CANDIDATE_METRICS_PATH,
    CANDIDATE_MODEL_PATH,
    FEATURE_GROUP_COLUMNS,
    PROD_METRICS_PATH,
    PROD_MODEL_PATH,
    TARGET,
    build_pipeline,
    calculate_regression_metrics,
    get_production_hyperparameters,
    get_production_model_name,
    split_features_target,
)

from src.utils import ensure_dir

# Paths

REFERENCE_DATA_PATH = (
    ARTIFACTS_DIR
    / "reference_data.csv"
)

TRAINING_DATA_PATH = (
    ARTIFACTS_DIR
    / "training_data.csv"
)

DRIFT_REPORT_PATH = (
    ARTIFACTS_DIR
    / "drift_report.json"
)

# Simulated sequential cohorts

INCOMING_COHORTS = [
    (
        "2013-2015",
        PROCESSED_DATA_DIR
        / "incoming_2013_2015.csv",
    ),
    (
        "2016-2018",
        PROCESSED_DATA_DIR
        / "incoming_2016_2018.csv",
    ),
    (
        "2019-2022",
        PROCESSED_DATA_DIR
        / "incoming_2019_2022.csv",
    ),
]

# Incoming split

INCOMING_TRAIN_SIZE = 0.70

# Numeric drift

KS_STAT_THRESHOLD = 0.10

P_VALUE_THRESHOLD = 0.05


# car_age is deliberately excluded from drift triggering because
# cohorts are defined by vehicle year and car_age is derived from year.
NUMERIC_DRIFT_COLUMNS = [
    "odometer",
]

# Categorical drift

CATEGORICAL_DRIFT_COLUMNS = [
    "manufacturer",
    "fuel",
    "transmission",
    "drive",
    "type",
]

JS_DISTANCE_THRESHOLD = 0.10

# Retraining / promotion

MIN_DRIFTED_FEATURES = 2

MIN_IMPROVEMENT_PCT = 1.0

# Loading helpers

def load_dataframe(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at "
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    if df.empty:
        raise ValueError(
            f"Dataset at {path} "
            "is empty."
        )

    return df


def load_model(
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at "
            f"{path}. "
            "Run `python -m src.train` first."
        )

    return joblib.load(
        path
    )

# Evaluation

def evaluate_model(
    model,
    df: pd.DataFrame,
) -> dict:

    if TARGET not in df.columns:

        raise ValueError(
            f"Expected target column "
            f"'{TARGET}'."
        )

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

# Group-aware incoming cohort split

def create_feature_groups(
    df: pd.DataFrame,
) -> pd.Series:

    missing_columns = [
        column
        for column
        in FEATURE_GROUP_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing feature grouping "
            f"columns: {missing_columns}"
        )

    return pd.util.hash_pandas_object(
        df[
            FEATURE_GROUP_COLUMNS
        ],
        index=False,
    )


def split_incoming_cohort(
    incoming_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Group-aware 70/30 split.

    Identical ML feature vectors cannot occur in both the adaptation
    set and current holdout.
    """

    groups = (
        create_feature_groups(
            incoming_df
        )
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=INCOMING_TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )

    (
        train_indices,
        evaluation_indices,
    ) = next(
        splitter.split(
            incoming_df,
            groups=groups,
        )
    )

    adaptation_df = (
        incoming_df
        .iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    evaluation_df = (
        incoming_df
        .iloc[
            evaluation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        adaptation_df,
        evaluation_df,
    )


def calculate_feature_overlap(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> tuple[
    int,
    float,
]:

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

# Numeric drift

def detect_numeric_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
) -> list[dict]:

    results = []

    for column in (
        NUMERIC_DRIFT_COLUMNS
    ):

        if (
            column
            not in reference.columns
            or column
            not in incoming.columns
        ):
            continue

        reference_values = (
            reference[
                column
            ]
            .dropna()
            .to_numpy()
        )

        incoming_values = (
            incoming[
                column
            ]
            .dropna()
            .to_numpy()
        )

        if (
            len(reference_values) == 0
            or len(incoming_values) == 0
        ):
            continue

        (
            ks_statistic,
            p_value,
        ) = ks_2samp(
            reference_values,
            incoming_values,
        )

        drifted = (
            p_value
            < P_VALUE_THRESHOLD
            and ks_statistic
            >= KS_STAT_THRESHOLD
        )

        results.append(
            {
                "feature": column,
                "type": "numeric",

                "ks_statistic": float(
                    ks_statistic
                ),

                "p_value": float(
                    p_value
                ),

                "threshold": float(
                    KS_STAT_THRESHOLD
                ),

                "drifted": bool(
                    drifted
                ),
            }
        )

    return results


# Categorical drift

def categorical_distribution(
    series: pd.Series,
) -> pd.Series:

    normalized = (
        series
        .fillna(
            "unknown"
        )
        .astype(
            str
        )
        .str.strip()
        .str.lower()
        .replace(
            "",
            "unknown",
        )
    )

    return (
        normalized.value_counts(
            normalize=True
        )
    )


def detect_categorical_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
) -> list[dict]:

    results = []

    for column in (
        CATEGORICAL_DRIFT_COLUMNS
    ):

        if (
            column
            not in reference.columns
            or column
            not in incoming.columns
        ):
            continue

        reference_distribution = (
            categorical_distribution(
                reference[
                    column
                ]
            )
        )

        incoming_distribution = (
            categorical_distribution(
                incoming[
                    column
                ]
            )
        )

        categories = (
            reference_distribution
            .index
            .union(
                incoming_distribution
                .index
            )
        )

        reference_probabilities = (
            reference_distribution
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy()
        )

        incoming_probabilities = (
            incoming_distribution
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy()
        )

        js_distance = float(
            jensenshannon(
                reference_probabilities,
                incoming_probabilities,
                base=2,
            )
        )

        drifted = (
            js_distance
            >= JS_DISTANCE_THRESHOLD
        )

        results.append(
            {
                "feature": column,

                "type": (
                    "categorical"
                ),

                "js_distance": (
                    js_distance
                ),

                "threshold": float(
                    JS_DISTANCE_THRESHOLD
                ),

                "drifted": bool(
                    drifted
                ),
            }
        )

    return results


def get_drifted_features(
    results: list[dict],
) -> list[str]:

    return [
        result["feature"]
        for result in results
        if result["drifted"]
    ]

# Candidate training data

def build_candidate_training_data(
    historical_training_df: pd.DataFrame,
    adaptation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Candidate receives:

    - all historically available labeled data
    - 70% adaptation portion of current cohort

    Current 30% holdout is not included.
    """

    return pd.concat(
        [
            historical_training_df,
            adaptation_df,
        ],
        ignore_index=True,
    )

# Candidate training

def train_candidate(
    candidate_training_df: pd.DataFrame,
):

    X_train, y_train = (
        split_features_target(
            candidate_training_df
        )
    )

    model_name = (
        get_production_model_name()
    )

    hyperparameters = (
        get_production_hyperparameters()
    )

    candidate_model = (
        build_pipeline(
            X_train,
            model_name=model_name,
            hyperparameters=hyperparameters,
        )
    )

    print()
    print(
        f"Training candidate "
        f"{model_name} on "
        f"{len(candidate_training_df):,} "
        "rows..."
    )

    print(
        "Using production "
        f"hyperparameters: "
        f"{hyperparameters}"
    )

    candidate_model.fit(
        X_train,
        y_train,
    )

    print(
        "Candidate training finished."
    )

    joblib.dump(
        candidate_model,
        CANDIDATE_MODEL_PATH,
    )

    candidate_metadata = {
        "model": (
            model_name
        ),

        "hyperparameters": (
            hyperparameters
        ),

        "train_rows": int(
            len(
                candidate_training_df
            )
        ),

        "candidate": True,
    }

    CANDIDATE_METRICS_PATH.write_text(
        json.dumps(
            candidate_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return candidate_model

# Promotion

def calculate_mae_improvement_pct(
    production_metrics: dict,
    candidate_metrics: dict,
) -> float:

    production_mae = (
        production_metrics[
            "mae"
        ]
    )

    candidate_mae = (
        candidate_metrics[
            "mae"
        ]
    )

    if production_mae <= 0:
        return 0.0

    return float(
        (
            production_mae
            - candidate_mae
        )
        / production_mae
        * 100
    )


def should_promote(
    production_metrics: dict,
    candidate_metrics: dict,
) -> tuple[
    bool,
    float,
]:

    improvement_pct = (
        calculate_mae_improvement_pct(
            production_metrics,
            candidate_metrics,
        )
    )

    promoted = (
        candidate_metrics[
            "mae"
        ]
        < production_metrics[
            "mae"
        ]
        and improvement_pct
        >= MIN_IMPROVEMENT_PCT
    )

    return (
        promoted,
        improvement_pct,
    )


def promote_candidate(
    candidate_model,
) -> None:

    joblib.dump(
        candidate_model,
        PROD_MODEL_PATH,
    )


def update_production_metrics(
    cohort_name: str,
    evaluation_metrics: dict,
) -> None:

    metrics = {}

    if PROD_METRICS_PATH.exists():

        try:
            metrics = json.loads(
                PROD_METRICS_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except (
            json.JSONDecodeError,
            OSError,
        ):
            metrics = {}

    metrics[
        "latest_cohort"
    ] = cohort_name

    metrics[
        "latest_cohort_metrics"
    ] = evaluation_metrics

    metrics[
        "latest_cohort_mae"
    ] = evaluation_metrics[
        "mae"
    ]

    PROD_METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )


# Printing

def print_metrics(
    label: str,
    metrics: dict,
) -> None:

    print(
        f"{label}:"
    )

    print(
        f"  MAE: "
        f"${metrics['mae']:,.2f}"
    )

    print(
        f"  Median AE: "
        f"${metrics['median_ae']:,.2f}"
    )

    print(
        f"  RMSE: "
        f"${metrics['rmse']:,.2f}"
    )

    print(
        f"  R²: "
        f"{metrics['r2']:.4f}"
    )


# Process one cohort

def process_cohort(
    cohort_name: str,
    cohort_path: Path,
    production_model,
    cumulative_labeled_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> tuple[
    object,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    print()
    print("=" * 76)

    print(
        f"Processing incoming cohort: "
        f"{cohort_name}"
    )

    print("=" * 76)

    incoming_df = (
        load_dataframe(
            cohort_path
        )
    )

    print(
        f"Incoming rows: "
        f"{len(incoming_df):,}"
    )

    incoming_features = (
        incoming_df.drop(
            columns=[TARGET]
        )
    )

    # Drift detection

    numeric_results = (
        detect_numeric_drift(
            reference_df,
            incoming_features,
        )
    )

    categorical_results = (
        detect_categorical_drift(
            reference_df,
            incoming_features,
        )
    )

    drift_results = (
        numeric_results
        + categorical_results
    )

    drifted_features = (
        get_drifted_features(
            drift_results
        )
    )

    numeric_drifted = (
        get_drifted_features(
            numeric_results
        )
    )

    categorical_drifted = (
        get_drifted_features(
            categorical_results
        )
    )

    print()
    print(
        "Numeric drift results:"
    )

    for result in (
        numeric_results
    ):

        status = (
            "DRIFT"
            if result["drifted"]
            else "stable"
        )

        print(
            f"  {result['feature']}: "
            f"KS="
            f"{result['ks_statistic']:.4f}, "
            f"p="
            f"{result['p_value']:.3e} "
            f"[{status}]"
        )

    print()
    print(
        "Categorical drift results:"
    )

    for result in (
        categorical_results
    ):

        status = (
            "DRIFT"
            if result["drifted"]
            else "stable"
        )

        print(
            f"  {result['feature']}: "
            f"JS="
            f"{result['js_distance']:.4f} "
            f"[{status}]"
        )

    print()
    print(
        f"Total drifted features: "
        f"{len(drifted_features)}"
    )

    if drifted_features:

        print(
            "Drifted: "
            + ", ".join(
                drifted_features
            )
        )

    retraining_triggered = (
        len(drifted_features)
        >= MIN_DRIFTED_FEATURES
    )

    # GROUP-AWARE current cohort split
    (
        adaptation_df,
        evaluation_df,
    ) = split_incoming_cohort(
        incoming_df
    )

    (
        overlap_rows,
        overlap_pct,
    ) = calculate_feature_overlap(
        adaptation_df,
        evaluation_df,
    )

    print()
    print(
        f"Incoming adaptation rows: "
        f"{len(adaptation_df):,}"
    )

    print(
        f"Untouched evaluation rows: "
        f"{len(evaluation_df):,}"
    )

    print(
        f"Feature overlap rows: "
        f"{overlap_rows:,}"
    )

    print(
        f"Feature overlap rate: "
        f"{overlap_pct:.2f}%"
    )

    if overlap_rows != 0:

        raise RuntimeError(
            "Group-aware incoming split failed. "
            "Feature overlap is not zero."
        )

    # Evaluate incumbent

    print()
    print(
        "Evaluating current "
        "production model..."
    )

    production_metrics = (
        evaluate_model(
            production_model,
            evaluation_df,
        )
    )

    print()

    print_metrics(
        "Production holdout metrics",
        production_metrics,
    )

    candidate_metrics = None

    candidate_training_rows = None

    improvement_pct = 0.0

    promoted = False

    # Retraining

    if retraining_triggered:

        print()
        print(
            "Meaningful drift detected."
        )

        candidate_training_df = (
            build_candidate_training_data(
                cumulative_labeled_df,
                adaptation_df,
            )
        )

        candidate_training_rows = int(
            len(
                candidate_training_df
            )
        )

        candidate_model = (
            train_candidate(
                candidate_training_df
            )
        )

        print()
        print(
            "Evaluating candidate "
            "on untouched holdout..."
        )

        candidate_metrics = (
            evaluate_model(
                candidate_model,
                evaluation_df,
            )
        )

        print()

        print_metrics(
            "Candidate holdout metrics",
            candidate_metrics,
        )

        (
            promoted,
            improvement_pct,
        ) = should_promote(
            production_metrics,
            candidate_metrics,
        )

        print()

        print(
            f"MAE improvement: "
            f"{improvement_pct:.2f}%"
        )

        if promoted:

            print(
                "Candidate passed "
                "promotion gate."
            )

            promote_candidate(
                candidate_model
            )

            production_model = (
                candidate_model
            )

            update_production_metrics(
                cohort_name,
                candidate_metrics,
            )

        else:

            print(
                "Candidate did not pass "
                "promotion gate."
            )

    else:

        print()
        print(
            "No meaningful drift detected. "
            "Skipping retraining."
        )

    # Walk-forward history
    # After this cohort's evaluation has completed,
    # ALL rows become historical labeled data available to future
    # cohorts.

    cumulative_labeled_df = (
        pd.concat(
            [
                cumulative_labeled_df,
                incoming_df,
            ],
            ignore_index=True,
        )
    )

    # Reference distribution also advances with completed observations.
    reference_df = (
        cumulative_labeled_df
        .drop(
            columns=[TARGET]
        )
        .copy()
    )

    result = {
        "cohort": (
            cohort_name
        ),

        "rows": int(
            len(incoming_df)
        ),

        "adaptation_rows": int(
            len(adaptation_df)
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

        "numeric_drifted_features": (
            numeric_drifted
        ),

        "categorical_drifted_features": (
            categorical_drifted
        ),

        "drifted_features": (
            drifted_features
        ),

        "num_drifted_features": int(
            len(
                drifted_features
            )
        ),

        "numeric_drift_results": (
            numeric_results
        ),

        "categorical_drift_results": (
            categorical_results
        ),

        "retraining_triggered": (
            retraining_triggered
        ),

        "candidate_training_rows": (
            candidate_training_rows
        ),

        "production_metrics": (
            production_metrics
        ),

        "candidate_metrics": (
            candidate_metrics
        ),

        "mae_improvement_pct": (
            improvement_pct
        ),

        "promoted": (
            promoted
        ),
    }

    return (
        production_model,
        cumulative_labeled_df,
        reference_df,
        result,
    )


# Full sequential drift simulation

def run_drift_pipeline() -> None:

    ensure_dir(
        ARTIFACTS_DIR
    )

    print(
        "Loading current production model..."
    )

    production_model = (
        load_model(
            PROD_MODEL_PATH
        )
    )

    print(
        "Loading baseline labeled "
        "training data..."
    )

    cumulative_labeled_df = (
        load_dataframe(
            TRAINING_DATA_PATH
        )
    )

    print(
        "Loading production drift reference..."
    )

    reference_df = (
        load_dataframe(
            REFERENCE_DATA_PATH
        )
    )

    production_model_name = (
        get_production_model_name()
    )

    production_hyperparameters = (
        get_production_hyperparameters()
    )

    print()
    print(
        f"Production model: "
        f"{production_model_name}"
    )

    print(
        f"Production hyperparameters: "
        f"{production_hyperparameters}"
    )

    report = {
        "simulation": (
            "simulated distribution shift "
            "using progressively newer "
            "vehicle cohorts"
        ),

        "configuration": {
            "production_model": (
                production_model_name
            ),

            "production_hyperparameters": (
                production_hyperparameters
            ),

            "split_strategy": (
                "group_shuffle_split"
            ),

            "feature_group_columns": (
                FEATURE_GROUP_COLUMNS
            ),

            "incoming_train_size": (
                INCOMING_TRAIN_SIZE
            ),

            "numeric_drift_columns": (
                NUMERIC_DRIFT_COLUMNS
            ),

            "categorical_drift_columns": (
                CATEGORICAL_DRIFT_COLUMNS
            ),

            "ks_stat_threshold": (
                KS_STAT_THRESHOLD
            ),

            "p_value_threshold": (
                P_VALUE_THRESHOLD
            ),

            "js_distance_threshold": (
                JS_DISTANCE_THRESHOLD
            ),

            "min_drifted_features": (
                MIN_DRIFTED_FEATURES
            ),

            "promotion_metric": (
                "mae"
            ),

            "min_improvement_pct": (
                MIN_IMPROVEMENT_PCT
            ),
        },

        "cohorts": [],
    }

    with mlflow.start_run(
        run_name=(
            "sequential_drift_evaluation"
        )
    ):

        mlflow.log_param(
            "production_model",
            production_model_name,
        )

        mlflow.log_param(
            "split_strategy",
            "group_shuffle_split",
        )

        mlflow.log_param(
            "incoming_train_size",
            INCOMING_TRAIN_SIZE,
        )

        mlflow.log_param(
            "promotion_metric",
            "mae",
        )

        mlflow.log_param(
            "min_improvement_pct",
            MIN_IMPROVEMENT_PCT,
        )

        for (
            cohort_name,
            cohort_path,
        ) in INCOMING_COHORTS:

            (
                production_model,
                cumulative_labeled_df,
                reference_df,
                cohort_result,
            ) = process_cohort(
                cohort_name=(
                    cohort_name
                ),
                cohort_path=(
                    cohort_path
                ),
                production_model=(
                    production_model
                ),
                cumulative_labeled_df=(
                    cumulative_labeled_df
                ),
                reference_df=(
                    reference_df
                ),
            )

            report[
                "cohorts"
            ].append(
                cohort_result
            )

            safe_name = (
                cohort_name.replace(
                    "-",
                    "_",
                )
            )

            mlflow.log_metric(
                f"{safe_name}_feature_overlap_pct",
                cohort_result[
                    "feature_overlap_pct"
                ],
            )

            for (
                metric_name,
                metric_value,
            ) in (
                cohort_result[
                    "production_metrics"
                ].items()
            ):

                mlflow.log_metric(
                    (
                        f"{safe_name}_"
                        f"production_"
                        f"{metric_name}"
                    ),
                    metric_value,
                )

            candidate_metrics = (
                cohort_result[
                    "candidate_metrics"
                ]
            )

            if (
                candidate_metrics
                is not None
            ):

                for (
                    metric_name,
                    metric_value,
                ) in (
                    candidate_metrics.items()
                ):

                    mlflow.log_metric(
                        (
                            f"{safe_name}_"
                            f"candidate_"
                            f"{metric_name}"
                        ),
                        metric_value,
                    )

                mlflow.log_metric(
                    (
                        f"{safe_name}_"
                        "mae_improvement_pct"
                    ),
                    cohort_result[
                        "mae_improvement_pct"
                    ],
                )

    DRIFT_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Summary

    print()
    print("=" * 100)

    print(
        "SEQUENTIAL DRIFT "
        "EVALUATION COMPLETE"
    )

    print("=" * 100)

    for cohort in (
        report["cohorts"]
    ):

        print()
        print(
            cohort[
                "cohort"
            ]
        )

        print(
            f"  Feature overlap: "
            f"{cohort['feature_overlap_pct']:.2f}%"
        )

        print(
            f"  Drifted features: "
            f"{cohort['num_drifted_features']}"
        )

        print()

        print_metrics(
            "  Production",
            cohort[
                "production_metrics"
            ],
        )

        if (
            cohort[
                "candidate_metrics"
            ]
            is not None
        ):

            print()

            print_metrics(
                "  Candidate",
                cohort[
                    "candidate_metrics"
                ],
            )

            print(
                f"  MAE improvement: "
                f"{cohort['mae_improvement_pct']:.2f}%"
            )

            print(
                f"  Promoted: "
                f"{cohort['promoted']}"
            )

    print()
    print(
        f"Saved drift report to: "
        f"{DRIFT_REPORT_PATH}"
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":

    run_drift_pipeline()