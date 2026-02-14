from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

DB_PATH = DATA_DIR / "dummy_classification.db"
TABLE_NAME = "classification_data"
TARGET_COLUMN = "target"

METRICS_CSV_PATH = ARTIFACTS_DIR / "model_comparison_metrics.csv"
METRICS_JSON_PATH = ARTIFACTS_DIR / "model_comparison_metrics.json"
SAMPLE_TEST_DATA_PATH = DATA_DIR / "sample_test_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_DISPLAY_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]
