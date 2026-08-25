from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load variables from resilient-price-engine/.env
load_dotenv(PROJECT_ROOT / ".env")

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "analysis_vehicle_data.csv"
)
PREDICTION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_predictions.csv"
)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# Connect to MySQL
engine = create_engine(DATABASE_URL)

# Load CSV
df = pd.read_csv(DATA_PATH)
df_predictions = pd.read_csv(PREDICTION_PATH)

# Import to MySQL
df.to_sql("listings", con=engine, if_exists="replace", index=False)
df_predictions.to_sql( "model_predictions", con=engine, if_exists="replace", index=False, )

print(f"Imported {len(df_predictions):,} rows into listings")