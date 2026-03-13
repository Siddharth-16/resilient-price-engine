from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = RAW_DATA_DIR / "vehicles.csv"
REFERENCED_DATA = ARTIFACTS_DIR / "reference_data.csv"
NEW_DATA = DATA_DIR / "new_data.csv"

TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42