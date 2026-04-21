import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for _d in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "movie-rating-prediction")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "ml-100k"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_TYPE = "svd"

MODEL_CONFIGS = {
    "svd": {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    "nmf": {"n_factors": 50, "n_epochs": 50},
    "knn": {"k": 40, "sim_options": {"name": "cosine", "user_based": True}},
}

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = [
    {"model_type": "svd", "n_factors": 50,  "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 100, "n_epochs": 50, "lr_all": 0.005, "reg_all": 0.02},
    {"model_type": "svd", "n_factors": 150, "n_epochs": 30, "lr_all": 0.01,  "reg_all": 0.02},
    {"model_type": "nmf", "n_factors": 50,  "n_epochs": 50},
    {"model_type": "nmf", "n_factors": 100, "n_epochs": 50},
    {"model_type": "knn", "k": 20, "sim_options": {"name": "cosine",   "user_based": True}},
    {"model_type": "knn", "k": 40, "sim_options": {"name": "cosine",   "user_based": True}},
    {"model_type": "knn", "k": 40, "sim_options": {"name": "pearson",  "user_based": True}},
]

# ---------------------------------------------------------------------------
# Airflow
# ---------------------------------------------------------------------------
AIRFLOW_DAG_ID = "movie_rating_training"
AIRFLOW_SCHEDULE = "@weekly"
