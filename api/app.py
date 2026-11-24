"""FastAPI service that serves the ``champion`` model from the MLflow registry.

Run:  uvicorn api.app:app --port 8000
Docs: http://localhost:8000/docs
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from churn import config

MODEL_URI = os.getenv("MODEL_URI", f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
YesNo = Literal["Yes", "No"]
state: dict = {}


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.1,
    }}}


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]


def load_model():
    model = mlflow.sklearn.load_model(MODEL_URI)
    threshold = 0.5
    try:
        info = mlflow.models.get_model_info(MODEL_URI)
        cfg = mlflow.artifacts.load_dict(f"runs:/{info.run_id}/model_config.json")
        threshold = float(cfg["threshold"])
    except Exception as e:  # noqa: BLE001 - threshold file is optional
        print(f"No saved threshold, using 0.5 ({e})")
    info_out = {}
    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
        run = client.get_run(mv.run_id)
        info_out = {
            "name": config.MODEL_NAME, "version": mv.version,
            "model_type": run.data.tags.get("model_type"),
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()
                        if k.startswith(("test_", "cv_"))},
            "trained_at": run.info.start_time,
        }
    except Exception as e:  # noqa: BLE001 - e.g. MODEL_URI points at a run, not the registry
        print(f"No registry info for model ({e})")
    state.update(model=model, threshold=threshold, info=info_out)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load model from {MODEL_URI}: {e}")
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_methods=["*"], allow_headers=["*"],
)


def _predict(customers: list[Customer]) -> list[Prediction]:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Train and register a model first.")
    df = pd.DataFrame([c.model_dump() for c in customers])
    proba = state["model"].predict_proba(df)[:, 1]
    t = state["threshold"]
    return [Prediction(
        churn_probability=round(float(p), 4),
        will_churn=bool(p >= t),
        risk_level="high" if p >= max(t, 0.7) else "medium" if p >= t * 0.6 else "low",
    ) for p in proba]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state,
            "model_uri": MODEL_URI, "threshold": state.get("threshold")}


@app.get("/model-info")
def model_info():
    """Version, model type and metrics of the serving model (for dashboards)."""
    if "model" not in state:
        raise HTTPException(503, "Model not loaded.")
    return {**state.get("info", {}), "threshold": state["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    return _predict(customers)


@app.post("/reload")
def reload():
    """Pick up a newly promoted champion without restarting the service."""
    load_model()
    return health()
