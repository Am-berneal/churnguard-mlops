import pandas as pd
import pytest

from churn.data import clean
from churn.explain import _describe, explain, global_drivers
from churn.features import build_pipeline, split_xy
from churn.profile import build_profile, reference_customer
from churn.retention import PLAYBOOK, decide, priority, recommend, revenue_at_risk, risk_level


@pytest.fixture
def fitted(raw_df):
    X, y = split_xy(clean(raw_df))
    model = build_pipeline("logreg").fit(X, y)
    return model, X, reference_customer(build_profile(X, y))


def test_profile_has_distributions(raw_df):
    X, y = split_xy(clean(raw_df))
    p = build_profile(X, y)
    assert set(p["numeric"]) >= {"tenure", "MonthlyCharges"}
    assert abs(sum(p["numeric"]["tenure"]["proportions"]) - 1) < 1e-9
    assert p["categorical"]["Contract"]["mode"] in {"Month-to-month", "One year", "Two year"}


def test_explain_returns_sorted_positive_reasons(fitted):
    model, X, ref = fitted
    reasons = explain(model, X.head(20), ref, top_k=3)
    assert len(reasons) == 20
    for r in reasons:
        assert len(r) <= 3
        impacts = [x["impact"] for x in r]
        assert impacts == sorted(impacts, reverse=True)
        assert all(i > 0 for i in impacts)


def test_month_to_month_is_a_reason_for_risky_customer(fitted):
    model, X, ref = fitted
    risky = X.head(1).copy()
    risky[["Contract", "tenure", "TotalCharges"]] = ["Month-to-month", 1, 70.0]
    assert "Contract" in [x["feature"] for x in explain(model, risky, ref)[0]]


def test_reason_text_is_readable():
    row = pd.Series({"tenure": 1, "TotalCharges": 70.0, "MonthlyCharges": 70.0, "Contract": "Month-to-month"})
    assert _describe("tenure", row) == "Short tenure (1 month)"
    assert _describe("Contract", row) == "Month-to-month contract"


def test_global_drivers_ranked(fitted):
    model, X, ref = fitted
    d = global_drivers(model, X, ref, sample=50)
    assert d and d[0]["mean_impact"] >= d[-1]["mean_impact"]


@pytest.mark.parametrize("p,level", [(0.9, "high"), (0.6, "medium"), (0.1, "low")])
def test_risk_levels(p, level):
    assert risk_level(p, threshold=0.58) == level


def test_p1_requires_high_risk_and_value():
    assert priority("high", 0.9, 1000, 0.58) == "P1"
    assert priority("high", 0.9, 100, 0.58) == "P2"
    assert priority("medium", 0.65, 1000, 0.58) == "P2"   # predicted churn, not P1
    assert priority("medium", 0.4, 1000, 0.58) == "P3"
    assert priority("low", 0.1, 1000, 0.58) == "—"


def test_revenue_at_risk():
    assert revenue_at_risk(0.5, 100) == 600.0


def test_recommend_uses_playbook_and_dedupes():
    reasons = [{"feature": "Contract", "reason": "Month-to-month contract", "impact": 0.3},
               {"feature": "TechSupport", "reason": "No tech support add-on", "impact": 0.1}]
    actions = recommend(reasons, "high")
    assert [a["action"] for a in actions] == [PLAYBOOK["Contract"]["action"],
                                              PLAYBOOK["TechSupport"]["action"]]
    assert all({"offer", "channel", "because"} <= set(a) for a in actions)


def test_low_risk_gets_no_actions():
    assert recommend([{"feature": "Contract", "reason": "x", "impact": 0.2}], "low") == []


def test_decide_bundles_everything():
    d = decide(0.85, 100.0, [{"feature": "Contract", "reason": "M2M", "impact": 0.3}], 0.58).to_dict()
    assert d["risk_level"] == "high" and d["priority"] == "P1" and d["actions"]
