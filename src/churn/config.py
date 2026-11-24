"""Central configuration: paths, column names, and MLflow names."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ENGINEERED_FEATURES = ["avg_monthly_spend", "num_services", "is_long_contract"]

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-classifier")
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42
