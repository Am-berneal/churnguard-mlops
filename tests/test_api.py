import pytest
from fastapi.testclient import TestClient

from api import app as api_module
from churn.data import clean
from churn.features import build_pipeline, split_xy

EXAMPLE = api_module.Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client(raw_df, monkeypatch):
    X, y = split_xy(clean(raw_df))
    model = build_pipeline("logreg").fit(X, y)
    # Skip the registry: inject a freshly trained model.
    monkeypatch.setattr(api_module, "load_model", lambda: api_module.state.update(
        model=model, threshold=0.5, info={"name": "test", "version": "1"}))
    api_module.state.clear()
    with TestClient(api_module.app) as c:
        yield c
    api_module.state.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    assert client.get("/model-info").json()["threshold"] == 0.5


def test_predict_single(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_batch(client):
    r = client.post("/predict/batch", json=[EXAMPLE, EXAMPLE])
    assert r.status_code == 200 and len(r.json()) == 2


@pytest.mark.parametrize("bad", [{"tenure": -1}, {"Contract": "Weekly"}, {"MonthlyCharges": "abc"}])
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json={**EXAMPLE, **bad}).status_code == 422


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(api_module, "load_model", lambda: None)
    api_module.state.clear()
    with TestClient(api_module.app) as c:
        assert c.post("/predict", json=EXAMPLE).status_code == 503
