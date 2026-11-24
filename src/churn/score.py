"""Batch scoring: rank a whole customer list by who to save first.

    python -m churn.score examples/customers_sample.csv -o scored.csv

Each customer gets a churn probability, risk level, priority, revenue at risk,
top reasons and a recommended action. The output is sorted by revenue at risk,
so the retention team can start from the top. Each batch is logged as an MLflow run.
"""
import argparse
from pathlib import Path

import mlflow
import pandas as pd

from churn import config
from churn.explain import explain
from churn.profile import build_profile, reference_customer
from churn.retention import decide
from churn.tracking import init_experiment

MODEL_URI = f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}"


def load_assets(model_uri: str = MODEL_URI):
    """Model, decision threshold and reference profile for a registered model."""
    model = mlflow.sklearn.load_model(model_uri)
    info = mlflow.models.get_model_info(model_uri)
    try:
        threshold = float(mlflow.artifacts.load_dict(
            f"runs:/{info.run_id}/model_config.json")["threshold"])
    except Exception:  # noqa: BLE001
        threshold = 0.5
    try:
        profile = mlflow.artifacts.load_dict(f"runs:/{info.run_id}/reference_profile.json")
    except Exception:  # noqa: BLE001 - models trained before profiles existed
        train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
        profile = build_profile(train, train[config.TARGET])
    return model, threshold, profile, info.run_id


def score_frame(df: pd.DataFrame, model, threshold: float, profile: dict) -> pd.DataFrame:
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    proba = model.predict_proba(X)[:, 1]
    reasons = explain(model, X, reference_customer(profile))

    rows = []
    for i, (p, r) in enumerate(zip(proba, reasons, strict=True)):
        d = decide(float(p), float(X.iloc[i]["MonthlyCharges"]), r, threshold)
        rows.append({
            "churn_probability": round(float(p), 4),
            "risk_level": d.risk_level,
            "priority": d.priority,
            "revenue_at_risk": d.revenue_at_risk,
            "top_reasons": "; ".join(x["reason"] for x in r),
            "recommended_action": d.actions[0]["action"] if d.actions else "",
        })
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)


def main(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    init_experiment()
    df = pd.read_csv(input_path)
    model, threshold, profile, run_id = load_assets()
    scored = score_frame(df, model, threshold, profile)
    output_path = output_path or str(Path(input_path).with_name(Path(input_path).stem + "_scored.csv"))
    scored.to_csv(output_path, index=False)

    high = scored[scored["risk_level"] == "high"]
    with mlflow.start_run(run_name="batch_scoring", nested=True):
        mlflow.set_tags({"stage": "scoring", "model_run_id": run_id})
        mlflow.log_param("input", Path(input_path).name)
        mlflow.log_metrics({
            "n_customers": len(scored), "n_high_risk": len(high),
            "avg_churn_probability": float(scored["churn_probability"].mean()),
            "revenue_at_risk_total": float(scored["revenue_at_risk"].sum()),
        })
        mlflow.log_artifact(output_path, "scoring")

    print(f"Scored {len(scored)} customers -> {output_path}")
    print(f"High risk: {len(high)} | revenue at risk (12 mo): ${scored['revenue_at_risk'].sum():,.0f}")
    cols = ["customerID", "churn_probability", "priority", "revenue_at_risk", "recommended_action"]
    print(scored[[c for c in cols if c in scored]].head(5).to_string(index=False))
    return scored


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="CSV with the 19 customer columns (customerID optional)")
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()
    main(a.input, a.output)
