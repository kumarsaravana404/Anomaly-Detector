import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    try:
        directory.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print(f"Warning: Could not create directory {directory}: {e}")

# Data Generation Config
N_SAMPLES = 1000
ANOMALY_RATIO = 0.05
RANDOM_STATE = 42

# Model Config
CONTAMINATION = 0.05
N_ESTIMATORS = 100
MODEL_FILENAME = "isolation_forest_model.pkl"
SCALER_FILENAME = "scaler.pkl"

MODEL_PATH = MODELS_DIR / MODEL_FILENAME
SCALER_PATH = MODELS_DIR / SCALER_FILENAME

# Features
FEATURE_COLUMNS = [
    "login_hour",
    "login_attempts",
    "ip_frequency",
    "device_type",
    "login_success",
]

# API Config
API_HOST = "0.0.0.0"
API_PORT = 8000
