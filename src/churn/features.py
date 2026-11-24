"""Feature engineering and model pipelines.

Everything lives inside one sklearn Pipeline, so the exact same
transformations run at training time and at serving time.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features: average spend, number of services, contract length."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
        X["avg_monthly_spend"] = X["TotalCharges"] / np.maximum(X["tenure"], 1)
        X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        X["is_long_contract"] = X["Contract"].isin(["One year", "Two year"]).astype(int)
        return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
    ])


def build_classifier(model_type: str, params: dict | None = None):
    params = dict(params or {})
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", **params)
    if model_type == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                      random_state=config.RANDOM_STATE, **params)
    if model_type == "hist_gb":
        return HistGradientBoostingClassifier(class_weight="balanced",
                                              random_state=config.RANDOM_STATE, **params)
    raise ValueError(f"Unknown model_type: {model_type}")


def build_pipeline(model_type: str = "hist_gb", params: dict | None = None) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", build_classifier(model_type, params)),
    ])


def split_xy(df: pd.DataFrame):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]
    return X, y
