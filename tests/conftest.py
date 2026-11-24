"""Shared fixtures. Tests use a small synthetic dataset so CI needs no network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CHOICES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({k: rng.choice(v, n) for k, v in CHOICES.items()})
    df["customerID"] = [f"C{i:04d}" for i in range(n)]
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    total[df["tenure"] == 0] = " "  # mimic the real dataset's blanks
    df["TotalCharges"] = total
    # Churn depends on contract and tenure, so a model can learn something.
    p = 0.15 + 0.4 * (df["Contract"] == "Month-to-month") - 0.004 * df["tenure"]
    df["Churn"] = np.where(rng.uniform(size=n) < p.clip(0.02, 0.95), "Yes", "No")
    return df
