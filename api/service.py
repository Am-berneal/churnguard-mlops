"""Model service: loads the champion model and turns inputs into decisions.

Keeps all ML logic out of the HTTP layer, so ``app.py`` only does routing.
"""
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from churn import config
from churn.explain import explain, global_drivers
from churn.monitoring import PredictionStore
from churn.profile import build_profile, reference_customer
from churn.retention import decide

MODEL_URI = os.getenv("MODEL_URI", f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


class ModelService:
    def __init__(self, model_uri: str = MODEL_URI):
        self.model_uri = model_uri
        self.model = None
        self.threshold = 0.5
        self.profile: dict | None = None
        self.reference: dict | None = None
        self.info: dict = {}
        self.drivers: list[dict] = []
        self.plot_dir: Path | None = None
        log = os.getenv("LOG_PREDICTIONS", "1") != "0"
        self.store = PredictionStore(os.getenv("PREDICTIONS_DB")) if log else None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # ---------- loading ----------
    def load(self) -> None:
        self.model = mlflow.sklearn.load_model(self.model_uri)
        run_id = mlflow.models.get_model_info(self.model_uri).run_id
        self.threshold = self._artifact_dict(run_id, "model_config.json", {}).get("threshold", 0.5)
        self.profile = (self._artifact_dict(run_id, "reference_profile.json", None)
                        or self._profile_from_data())
        self.reference = reference_customer(self.profile)
        self.info = self._registry_info(run_id)
        self.plot_dir = self._download_plots(run_id)
        self.drivers = self._global_drivers()

    @staticmethod
    def _artifact_dict(run_id: str, path: str, default):
        try:
            return mlflow.artifacts.load_dict(f"runs:/{run_id}/{path}")
        except Exception:  # noqa: BLE001 - optional artifact
            return default

    @staticmethod
    def _profile_from_data() -> dict:
        train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
        return build_profile(train[FEATURES], train[config.TARGET])

    def _registry_info(self, run_id: str) -> dict:
        info = {"name": config.MODEL_NAME, "run_id": run_id}
        try:
            client = mlflow.MlflowClient()
            mv = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
            run = client.get_run(mv.run_id)
            info.update({
                "version": mv.version,
                "model_type": run.data.tags.get("model_type"),
                "params": {k: v for k, v in run.data.params.items() if k not in ("model_type",)},
                "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()
                            if k.startswith(("test_", "cv_"))},
                "trained_at": run.info.start_time,
            })
        except Exception as e:  # noqa: BLE001 - e.g. MODEL_URI points at a run
            print(f"No registry info ({e})")
        return info

    @staticmethod
    def _download_plots(run_id: str) -> Path | None:
        try:
            dst = Path(tempfile.mkdtemp(prefix="churnguard_plots_"))
            return Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="plots", dst_path=str(dst)))
        except Exception:  # noqa: BLE001
            return None

    def _global_drivers(self) -> list[dict]:
        path = config.PROCESSED_DIR / "test.parquet"
        if not path.exists():
            return []
        return global_drivers(self.model, pd.read_parquet(path)[FEATURES], self.reference)

    # ---------- predicting ----------
    def predict(self, records: list[dict], source: str = "api") -> list[dict]:
        df = pd.DataFrame(records)
        X = df[FEATURES].copy()
        proba = self.model.predict_proba(X)[:, 1]
        reasons = explain(self.model, X, self.reference)
        out = []
        for rec, p, r in zip(records, proba, reasons, strict=True):
            d = decide(float(p), float(rec["MonthlyCharges"]), r, self.threshold)
            out.append({
                "customerID": rec.get("customerID"),
                "churn_probability": round(float(p), 4),
                "will_churn": bool(p >= self.threshold),
                "risk_level": d.risk_level, "priority": d.priority,
                "revenue_at_risk": d.revenue_at_risk, "reasons": r, "actions": d.actions,
            })
        if self.store is not None:
            self.store.log(X, proba, [o["risk_level"] for o in out], source, self.info.get("version"))
        return out

    @staticmethod
    def summarize(results: list[dict]) -> dict:
        levels = [r["risk_level"] for r in results]
        n = max(len(results), 1)
        return {
            "n_customers": len(results),
            "n_high": levels.count("high"), "n_medium": levels.count("medium"),
            "n_low": levels.count("low"),
            "n_p1": sum(r["priority"] == "P1" for r in results),
            "revenue_at_risk_total": round(sum(r["revenue_at_risk"] for r in results), 2),
            "avg_churn_probability": round(sum(r["churn_probability"] for r in results) / n, 4),
        }

    def training_runs(self, limit: int = 20) -> list[dict]:
        try:
            runs = mlflow.search_runs(
                experiment_names=[config.EXPERIMENT_NAME],
                filter_string="tags.stage = 'training'",
                order_by=["attributes.start_time DESC"], max_results=limit)
        except Exception:  # noqa: BLE001
            return []
        champion = self.info.get("run_id")
        cols = {"metrics.cv_roc_auc_mean": "cv_roc_auc", "metrics.test_roc_auc": "test_roc_auc",
                "metrics.test_f1": "test_f1", "metrics.test_recall": "test_recall"}
        out = []
        for _, r in runs.iterrows():
            row = {"run_id": r.run_id, "run_name": r.get("tags.mlflow.runName"),
                   "model_type": r.get("tags.model_type"), "champion": r.run_id == champion}
            row.update({v: (round(float(r[k]), 4) if k in r and pd.notna(r[k]) else None)
                        for k, v in cols.items()})
            out.append(row)
        return out
