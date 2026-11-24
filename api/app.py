"""ChurnGuard API and dashboard.

Run:        uvicorn api.app:app --port 8000
Dashboard:  http://localhost:8000
API docs:   http://localhost:8000/docs
"""
import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from api.schemas import BatchResult, Customer, MessageRequest, MessageResponse, Prediction
from api.service import ModelService
from churn.genai import retention_message

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"
MAX_CSV_ROWS = 5000

service = ModelService()


def load_model() -> None:
    service.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:  # noqa: BLE001 - start anyway; /health reports it
        print(f"WARNING: could not load model from {service.model_uri}: {e}")
    yield


app = FastAPI(
    title="ChurnGuard API",
    version="2.0.0",
    description="Predict customer churn, explain why, and recommend how to keep the customer.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def require_model() -> None:
    if not service.loaded:
        raise HTTPException(503, "Model not loaded. Train and register a model first (make pipeline).")


# ---------------- dashboard ----------------
@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "favicon.ico")


@app.get("/sample.csv", tags=["dashboard"], summary="Example CSV for batch scoring")
def sample_csv():
    return FileResponse(ROOT / "examples" / "customers_sample.csv", filename="customers_sample.csv")


# ---------------- status & model ----------------
@app.get("/health", tags=["status"])
def health():
    return {"status": "ok", "model_loaded": service.loaded, "model_uri": service.model_uri,
            "version": service.info.get("version"),
            "threshold": service.threshold if service.loaded else None}


@app.get("/model-info", tags=["model"])
def model_info():
    """Version, type, metrics, decision threshold and top churn drivers of the serving model."""
    require_model()
    return {**service.info, "threshold": service.threshold,
            "train_churn_rate": service.profile.get("churn_rate"), "drivers": service.drivers}


@app.get("/model/runs", tags=["model"])
def model_runs():
    """Recent training runs from MLflow, for comparing models."""
    return service.training_runs()


@app.get("/model/plots/{name}", tags=["model"])
def model_plot(name: str):
    """Diagnostic plots logged to MLflow with the champion model."""
    require_model()
    if service.plot_dir is None:
        raise HTTPException(404, "No plots logged for this model")
    path = (service.plot_dir / name).resolve()
    if path.parent != service.plot_dir.resolve() or not path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(path)


@app.post("/reload", tags=["model"])
def reload():
    """Pick up a newly promoted champion without restarting the service."""
    load_model()
    return health()


# ---------------- predictions ----------------
@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(customer: Customer):
    """Score one customer: probability, risk, priority, reasons and recommended actions."""
    require_model()
    return service.predict([customer.model_dump()], source="api")[0]


@app.post("/predict/batch", response_model=BatchResult, tags=["predict"])
def predict_batch(customers: list[Customer]):
    require_model()
    if len(customers) > MAX_CSV_ROWS:
        raise HTTPException(413, f"At most {MAX_CSV_ROWS} customers per request")
    results = service.predict([c.model_dump() for c in customers], source="batch")
    return {"summary": service.summarize(results), "results": results}


@app.post("/predict/csv", response_model=BatchResult, tags=["predict"])
async def predict_csv(file: UploadFile = File(...)):  # noqa: B008 - FastAPI dependency
    """Upload a CSV of customers; returns results sorted by revenue at risk."""
    require_model()
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read CSV: {e}") from e
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(413, f"At most {MAX_CSV_ROWS} rows per file")
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "customerID" in df:
        df["customerID"] = df["customerID"].astype(str)
    try:
        customers = [Customer(**r).model_dump() for r in df.to_dict("records")]
    except ValidationError as e:
        first = e.errors()[0]
        raise HTTPException(422, f"Invalid data in column '{first['loc'][0]}': {first['msg']}") from e
    results = sorted(service.predict(customers, source="csv"), key=lambda r: -r["revenue_at_risk"])
    return {"summary": service.summarize(results), "results": results}


@app.post("/retention-message", response_model=MessageResponse, tags=["retention"])
def write_retention_message(req: MessageRequest):
    """Draft a personalized retention message (Gemini if GEMINI_API_KEY is set, else a template)."""
    require_model()
    rec = req.customer.model_dump()
    pred = service.predict([rec], source="message")[0]
    return retention_message(rec, pred["reasons"], pred["actions"], req.tone)


# ---------------- monitoring ----------------
@app.get("/monitoring", tags=["monitoring"])
def monitoring():
    """Prediction volume, risk mix and data drift (PSI) of live traffic vs training data."""
    if service.store is None:
        return {"enabled": False}
    summary = service.store.summary(service.profile if service.loaded else None)
    recent = service.store.load(8)
    summary["recent"] = [
        {"ts": r.ts, "source": r.source, "probability": round(r.probability, 4), "risk_level": r.risk_level,
         **{k: v for k, v in json.loads(r.features).items()
            if k in ("tenure", "Contract", "MonthlyCharges")}}
        for r in recent.itertuples()
    ]
    return {"enabled": True, **summary}
