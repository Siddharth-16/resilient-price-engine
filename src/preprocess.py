from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, PROCESSED_DATA_DIR
from src.utils import ensure_dir

MIN_YEAR = 2000
MAX_YEAR = 2022
MIN_PRICE = 500
MAX_PRICE = 100_000
MAX_ODOMETER = 300_000

CATEGORICAL_COLUMNS = [
    "manufacturer",
    "model",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
]

MODEL_COLUMNS = [
    *CATEGORICAL_COLUMNS,
    "odometer",
    "car_age",
    "price",
]

# Vehicle model year is used only to simulate progressively newer serving cohorts.
# It is not passed directly to the model; the model receives car_age instead.
COHORTS = {
    "baseline_2000_2012": (2000, 2012),
    "incoming_2013_2015": (2013, 2015),
    "incoming_2016_2018": (2016, 2018),
    "incoming_2019_2022": (2019, 2022),
}


def _record(stats: dict[str, int | float], label: str, df: pd.DataFrame) -> None:
    stats[label] = int(len(df))


def preprocess(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int | float]]:
    """Clean raw Craigslist vehicle listings and return model-ready rows.

    Vehicle model year is returned separately so the cleaned dataset can be
    partitioned into a historical baseline and progressively newer cohorts
    without passing `year` directly to the estimator.
    """
    required_source_columns = {"year", "price", "odometer", *CATEGORICAL_COLUMNS}
    missing = sorted(required_source_columns.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    stats: dict[str, int | float] = {}
    df = raw_df.copy()
    _record(stats, "raw_rows", df)

    # Craigslist `id` represents the listing itself. Deduplicating on model
    # features would incorrectly collapse distinct listings with identical
    # vehicle attributes and distort the feature distributions used for drift.
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
        stats["deduplication_key"] = "id"
    else:
        df = df.drop_duplicates(keep="first")
        stats["deduplication_key"] = "full_raw_row"
    _record(stats, "after_deduplication", df)

    # Coerce key numeric fields so malformed values become NaN and are removed.
    for column in ["year", "price", "odometer"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["year", "price", "odometer"])
    _record(stats, "after_required_numeric_fields", df)

    df = df[df["year"].between(MIN_YEAR, MAX_YEAR)]
    _record(stats, "after_year_filter", df)

    df = df[df["price"].between(MIN_PRICE, MAX_PRICE)]
    _record(stats, "after_price_filter", df)

    df = df[df["odometer"].between(0, MAX_ODOMETER)]
    _record(stats, "after_odometer_filter", df)

    # Normalize categoricals rather than dropping otherwise usable listings.
    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )

    df["year"] = df["year"].astype(int)
    df["price"] = df["price"].astype(float)
    df["odometer"] = df["odometer"].astype(float)
    df["car_age"] = MAX_YEAR - df["year"]

    years = df["year"].copy().reset_index(drop=True)
    ids = (
        df["id"].copy().reset_index(drop=True)
        if "id" in df.columns
        else pd.Series(range(len(df)), name="id")
    )
    clean_df = df[MODEL_COLUMNS].reset_index(drop=True)

    stats["final_rows"] = int(len(clean_df))
    stats["retention_rate_pct"] = (
        round(100 * len(clean_df) / int(stats["raw_rows"]), 2)
        if stats["raw_rows"]
        else 0.0
    )

    return clean_df, years, ids, stats


def save_outputs(
    clean_df: pd.DataFrame,
    years: pd.Series,
    ids: pd.Series,
    stats: dict[str, int | float],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    clean_path = output_dir / "clean_vehicle_data.csv"
    clean_df.to_csv(clean_path, index=False)

    analysis_df = clean_df.copy()

    analysis_df.insert(
        0,
        "year",
        years.to_numpy(),
    )

    analysis_df.insert(
        0,
        "id",
        ids.to_numpy(),
    )

    analysis_path = (
        output_dir
        / "analysis_vehicle_data.csv"
    )

    analysis_df.to_csv(
        analysis_path,
        index=False,
    )

    cohort_counts: dict[str, int] = {}
    for name, (start_year, end_year) in COHORTS.items():
        mask = years.between(start_year, end_year)
        cohort_df = clean_df.loc[mask].reset_index(drop=True)
        cohort_path = output_dir / f"{name}.csv"
        cohort_df.to_csv(cohort_path, index=False)
        cohort_counts[name] = int(len(cohort_df))

    report = {
        "filters": {
            "year": [MIN_YEAR, MAX_YEAR],
            "price": [MIN_PRICE, MAX_PRICE],
            "odometer": [0, MAX_ODOMETER],
        },
        "cleaning_stats": stats,
        "cohort_counts": cohort_counts,
    }
    report_path = output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Preprocessing complete")
    print(f"Raw rows: {int(stats['raw_rows']):,}")
    print(f"After deduplication: {int(stats['after_deduplication']):,}")
    print(f"Deduplication key: {stats['deduplication_key']}")
    print(f"After year filter: {int(stats['after_year_filter']):,}")
    print(f"After price filter: {int(stats['after_price_filter']):,}")
    print(f"After odometer filter: {int(stats['after_odometer_filter']):,}")
    print(f"Final usable rows: {int(stats['final_rows']):,}")
    print(f"Retention rate: {float(stats['retention_rate_pct']):.2f}%")
    print(f"Saved cleaned dataset to: {clean_path}")
    print("Cohorts:")
    for name, count in cohort_counts.items():
        print(f"  {name}: {count:,} rows")
    print(f"Saved preprocessing report to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Craigslist vehicle listings and create drift cohorts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_PATH,
        help=f"Raw vehicles CSV path (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Processed output directory (default: {PROCESSED_DATA_DIR})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {args.input}. Place vehicles.csv there first."
        )

    raw_df = pd.read_csv(args.input)
    clean_df, years, ids, stats = preprocess(raw_df)
    save_outputs(clean_df, years, ids, stats, args.output_dir)


if __name__ == "__main__":
    main()
