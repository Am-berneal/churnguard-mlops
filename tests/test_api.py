import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app as api_module
from api.schemas import EXAMPLE_CUSTOMER
from churn.data import clean
from churn.features import build_pipeline, split_xy
from churn.monitoring import PredictionStore
from churn.profile import build_profile, reference_customer

svc = api_module.service


@pytest.fixture
def client(raw_df, tmp_path, monkeypatch):
    X, y = split_xy(clean(raw_df))
    model = build_pipeline("logreg").fit(X, y)
    profile = build_profile(X, y)

    def fake_load():  # skip the MLflow registry: inject a freshly trained model
        svc.model, svc.threshold, svc.profile = model, 0.5, profile
        svc.reference = reference_customer(profile)
        svc.info = {"name": "test", "version": "1", "metrics": {"test_roc_auc": 0.8}}
        svc.drivers, svc.plot_dir = [], None

    monkeypatch.setattr(api_module, "load_model", fake_load)
    monkeypatch.setattr(svc, "store", PredictionStore(tmp_path / "pred.db"))
    monkeypatch.setattr(svc, "model", None)
    with TestClient(api_module.app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/model-info").json()
    assert info["threshold"] == 0.5 and info["version"] == "1"


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ChurnGuard" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_predict_returns_explanation_and_actions(client):
    body = client.post("/predict", json=EXAMPLE_CUSTOMER).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["customerID"] == "DEMO-0001"
    assert body["risk_level"] in {"low", "medium", "high"}
    assert isinstance(body["reasons"], list) and isinstance(body["actions"], list)
    assert body["revenue_at_risk"] >= 0


def test_predict_batch_has_summary(client):
    body = client.post("/predict/batch", json=[EXAMPLE_CUSTOMER] * 3).json()
    assert body["summary"]["n_customers"] == 3 and len(body["results"]) == 3


def test_predict_csv_sorted_by_revenue(client, raw_df):
    buf = io.StringIO()
    raw_df.drop(columns=["Churn"]).head(25).to_csv(buf, index=False)
    r = client.post("/predict/csv", files={"file": ("c.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200
    rev = [x["revenue_at_risk"] for x in r.json()["results"]]
    assert rev == sorted(rev, reverse=True) and len(rev) == 25


def test_predict_csv_rejects_bad_values(client, raw_df):
    df = raw_df.drop(columns=["Churn"]).head(3).copy()
    df.loc[0, "Contract"] = "Weekly"
    r = client.post("/predict/csv", files={"file": ("c.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 422 and "Contract" in r.json()["detail"]


@pytest.mark.parametrize("bad", [{"tenure": -1}, {"Contract": "Weekly"}, {"MonthlyCharges": "abc"}])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json={**EXAMPLE_CUSTOMER, **bad}).status_code == 422


def test_retention_message(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/retention-message", json={"customer": EXAMPLE_CUSTOMER, "tone": "concise"})
    assert r.status_code == 200 and r.json()["message"].startswith("Hi there")


def test_monitoring_counts_predictions(client):
    client.post("/predict/batch", json=[EXAMPLE_CUSTOMER] * 4)
    m = client.get("/monitoring").json()
    assert m["n_predictions"] == 4 and len(m["recent"]) == 4
    assert m["overall_status"] == "insufficient data"  # fewer than 30 samples


def test_plot_missing_returns_404(client):
    assert client.get("/model/plots/roc.png").status_code == 404


def test_endpoints_return_503_without_model(monkeypatch):
    monkeypatch.setattr(api_module, "load_model", lambda: None)
    monkeypatch.setattr(svc, "model", None)
    with TestClient(api_module.app) as c:
        assert c.post("/predict", json=EXAMPLE_CUSTOMER).status_code == 503
        assert c.get("/model-info").status_code == 503


def test_example_customer_matches_schema():
    cols = set(EXAMPLE_CUSTOMER) - {"customerID"}
    assert len(cols) == 19
    assert pd.DataFrame([EXAMPLE_CUSTOMER]).shape[1] == 20
