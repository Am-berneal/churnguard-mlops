"""Production monitoring: log every prediction and check for data drift.

Predictions go to a small SQLite database (no extra service needed). Drift is
measured with the Population Stability Index (PSI): how far the distribution
of live inputs has moved from the training data.

    PSI < 0.10   stable
    PSI < 0.25   moderate shift: keep an eye on it
    PSI >= 0.25  significant shift: retrain or investigate
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from churn import config

EPS = 1e-4
MIN_SAMPLES = 300  # below this, sampling noise alone pushes PSI past 0.1
DRIFT_FEATURES_NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
DRIFT_FEATURES_CAT = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                      "OnlineSecurity", "PaperlessBilling"]


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float | None) -> str:
    if value is None:
        return "insufficient data"
    return "stable" if value < 0.10 else "moderate" if value < 0.25 else "significant"


def numeric_drift(ref: dict, live: pd.Series) -> float:
    edges = np.array(ref["edges"], float)
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(pd.to_numeric(live, errors="coerce").fillna(0), bins=edges)
    return psi(ref["proportions"], counts / max(counts.sum(), 1))


def categorical_drift(ref: dict, live: pd.Series) -> float:
    live_p = live.astype(str).value_counts(normalize=True)
    cats = sorted(set(ref["proportions"]) | set(live_p.index))
    return psi([ref["proportions"].get(c, 0) for c in cats], [live_p.get(c, 0) for c in cats])


def drift_report(profile: dict, live: pd.DataFrame) -> list[dict]:
    rows = []
    enough = len(live) >= MIN_SAMPLES
    for col in DRIFT_FEATURES_NUM:
        v = numeric_drift(profile["numeric"][col], live[col]) if enough else None
        rows.append({"feature": col, "type": "numeric", "psi": None if v is None else round(v, 4),
                     "status": psi_status(v),
                     "train_mean": round(profile["numeric"][col]["mean"], 2),
                     "live_mean": round(float(pd.to_numeric(live[col]).mean()), 2) if len(live) else None})
    for col in DRIFT_FEATURES_CAT:
        v = categorical_drift(profile["categorical"][col], live[col]) if enough else None
        rows.append({"feature": col, "type": "categorical",
                     "psi": None if v is None else round(v, 4), "status": psi_status(v),
                     "train_top": profile["categorical"][col]["mode"],
                     "live_top": str(live[col].mode().iloc[0]) if len(live) else None})
    return rows


class PredictionStore:
    """Thread-safe append-only log of predictions in SQLite."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.DATA_DIR / "predictions.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT,
                model_version TEXT, probability REAL, risk_level TEXT, features TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def log(self, X: pd.DataFrame, proba, levels, source: str, model_version: str | None) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
        rows = [(ts, source, str(model_version), float(p), lvl,
                 json.dumps({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}))
                for p, lvl, r in zip(proba, levels, X[cols].to_dict("records"), strict=True)]
        with self._lock, self._conn() as c:
            c.executemany("INSERT INTO predictions (ts, source, model_version, probability, "
                          "risk_level, features) VALUES (?,?,?,?,?,?)", rows)

    def load(self, limit: int = 5000) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", c, params=(limit,))
        return df

    def summary(self, profile: dict | None, limit: int = 5000) -> dict:
        df = self.load(limit)
        if df.empty:
            return {"n_predictions": 0, "risk_mix": {}, "drift": [], "daily": []}
        feats = pd.DataFrame([json.loads(f) for f in df["features"]])
        df["day"] = df["ts"].str[:10]
        daily = (df.groupby("day").agg(n=("id", "size"), avg_probability=("probability", "mean"))
                 .reset_index().sort_values("day").tail(14))
        drift = drift_report(profile, feats) if profile else []
        worst = max((d["psi"] for d in drift if d["psi"] is not None), default=None)
        return {
            "n_predictions": int(len(df)),
            "avg_probability": round(float(df["probability"].mean()), 4),
            "risk_mix": {k: int(v) for k, v in df["risk_level"].value_counts().items()},
            "by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
            "daily": [{"day": r.day, "n": int(r.n), "avg_probability": round(r.avg_probability, 4)}
                      for r in daily.itertuples()],
            "drift": drift,
            "overall_status": psi_status(worst),
            "min_samples": MIN_SAMPLES,
            "last_prediction": df["ts"].iloc[0],
        }
