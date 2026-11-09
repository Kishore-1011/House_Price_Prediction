"""Paths and settings shared by every part of the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "housing.csv"
CLIENT_DATA = ROOT / "data" / "client" / "client_file.csv"
PREDICTIONS = ROOT / "data" / "predictions" / "output.csv"

MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

METRICS_PATH = ROOT / "reports" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

AREA = "area"
OTHER_NUMERIC = ["bedroom", "bathroom", "age"]
CATEGORICAL = ["locality"]
FEATURES = [AREA, *OTHER_NUMERIC, *CATEGORICAL]
TARGET = "price"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Invented neighbourhoods with invented prices per square foot.
LOCALITY_RATES = {
    "Lakeside": 6800,
    "Hillcrest": 6100,
    "Tech Park": 5600,
    "Riverview": 5000,
    "Old Town": 4300,
    "Greenfield": 3700,
}
