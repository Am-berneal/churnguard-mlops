import numpy as np

from churn import genai
from churn.data import clean
from churn.features import split_xy
from churn.monitoring import PredictionStore, drift_report, psi, psi_status
from churn.profile import build_profile


def test_psi_zero_for_identical_and_grows_with_shift():
    e = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(e, e) == 0
    assert psi(e, [0.1, 0.2, 0.3, 0.4]) < psi(e, [0.01, 0.04, 0.15, 0.8])


def test_psi_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.2) == "moderate"
    assert psi_status(0.4) == "significant"
    assert psi_status(None) == "insufficient data"


def test_drift_detected_on_shifted_data(raw_df):
    X, y = split_xy(clean(raw_df))
    profile = build_profile(X, y)
    same = {d["feature"]: d["status"] for d in drift_report(profile, X)}
    assert same["tenure"] == "stable"

    shifted = X.copy()
    shifted["tenure"] = (shifted["tenure"] * 0.2).astype(int)
    shifted["Contract"] = "Month-to-month"
    drift = {d["feature"]: d["status"] for d in drift_report(profile, shifted)}
    assert drift["tenure"] == "significant"
    assert drift["Contract"] == "significant"


def test_drift_needs_minimum_samples(raw_df):
    X, y = split_xy(clean(raw_df))
    report = drift_report(build_profile(X, y), X.head(5))
    assert all(d["status"] == "insufficient data" for d in report)


def test_prediction_store_roundtrip(tmp_path, raw_df):
    X, y = split_xy(clean(raw_df))
    store = PredictionStore(tmp_path / "p.db")
    store.log(X.head(40), [0.9] * 10 + [0.1] * 30, ["high"] * 10 + ["low"] * 30, "test", "1")
    s = store.summary(build_profile(X, y))
    assert s["n_predictions"] == 40
    assert s["risk_mix"] == {"high": 10, "low": 30}
    assert s["by_source"] == {"test": 40}
    assert len(s["drift"]) > 0


ACTIONS = [{"action": "Offer a discounted 1-year contract", "offer": "a discounted 1-year plan",
            "channel": "Phone call", "cost": "medium", "because": "Month-to-month contract"}]
CUSTOMER = {"tenure": 2, "Contract": "Month-to-month", "InternetService": "Fiber optic",
            "MonthlyCharges": 95.5}


def test_template_message_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = genai.retention_message(CUSTOMER, [], ACTIONS)
    assert out["source"] == "template"
    assert "a discounted 1-year plan" in out["message"]


def test_gemini_used_when_key_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "_call_gemini", lambda prompt, key: "Hi there, from Gemini")
    out = genai.retention_message(CUSTOMER, [], ACTIONS)
    assert out["source"].startswith("gemini") and "Gemini" in out["message"]


def test_gemini_failure_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(prompt, key):
        raise TimeoutError

    monkeypatch.setattr(genai, "_call_gemini", boom)
    out = genai.retention_message(CUSTOMER, [], ACTIONS)
    assert out["source"] == "template" and "TimeoutError" in out["note"]


def test_prompt_excludes_identifiers():
    prompt = genai.build_prompt({**CUSTOMER, "customerID": "SECRET-123"}, [], ACTIONS, "friendly")
    assert "SECRET-123" not in prompt
