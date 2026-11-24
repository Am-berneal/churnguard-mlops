"""Per-customer explanations: *why* is this customer at risk?

Model-agnostic counterfactual attribution. For each feature (or group of
linked features) we swap the customer's value for the value of a typical
low-risk customer and measure how much the churn probability drops.
A big drop means that feature is a major reason for the risk.

It only needs ``predict_proba``, so it works for every model type, and all
swaps for a batch are scored in a single vectorised call.
"""
import numpy as np
import pandas as pd

from churn import config

# Features that only make sense when changed together.
FEATURE_GROUPS: dict[str, list[str]] = {
    "tenure": ["tenure", "TotalCharges"],
    "MonthlyCharges": ["MonthlyCharges"],
    "SeniorCitizen": ["SeniorCitizen"],
    **{c: [c] for c in config.CATEGORICAL_FEATURES},
}

# Values that mean "this customer looks like a loyal one" for each group.
LOW_RISK_VALUES: dict[str, dict] = {
    "Contract": {"Contract": "Two year"},
    "PaymentMethod": {"PaymentMethod": "Credit card (automatic)"},
    "PaperlessBilling": {"PaperlessBilling": "No"},
    "OnlineSecurity": {"OnlineSecurity": "Yes"},
    "TechSupport": {"TechSupport": "Yes"},
    "OnlineBackup": {"OnlineBackup": "Yes"},
    "DeviceProtection": {"DeviceProtection": "Yes"},
    "InternetService": {"InternetService": "DSL"},
    "SeniorCitizen": {"SeniorCitizen": 0},
}


def _months(n) -> str:
    n = int(n)
    return f"{n} month" if n == 1 else f"{n} months"


def _describe(group: str, row: pd.Series) -> str:
    v = row.get(group)
    texts = {
        "tenure": f"Short tenure ({_months(row['tenure'])})" if row["tenure"] < 24
        else f"Tenure of {_months(row['tenure'])}",
        "Contract": f"{v} contract",
        "MonthlyCharges": f"High monthly charges (${row['MonthlyCharges']:.2f})",
        "PaymentMethod": f"Pays by {str(v).lower()}",
        "InternetService": f"{v} internet" if v != "No" else "No internet service",
        "OnlineSecurity": "No online security add-on",
        "TechSupport": "No tech support add-on",
        "OnlineBackup": "No online backup add-on",
        "DeviceProtection": "No device protection add-on",
        "PaperlessBilling": "Paperless billing",
        "SeniorCitizen": "Senior citizen",
        "Partner": "No partner on the account" if v == "No" else "Has a partner",
        "Dependents": "No dependents on the account" if v == "No" else "Has dependents",
        "StreamingTV": f"Streaming TV: {v}",
        "StreamingMovies": f"Streaming movies: {v}",
        "MultipleLines": f"Multiple lines: {v}",
        "PhoneService": f"Phone service: {v}",
        "gender": f"Gender: {v}",
    }
    return texts.get(group, f"{group}: {v}")


def _counterfactual(group: str, row: pd.Series, reference: dict) -> dict:
    if group in LOW_RISK_VALUES:
        return LOW_RISK_VALUES[group]
    if group == "tenure":
        tenure = max(float(reference["tenure"]), 36.0)
        return {"tenure": tenure, "TotalCharges": tenure * float(row["MonthlyCharges"])}
    return {c: reference[c] for c in FEATURE_GROUPS[group]}


def explain(model, X: pd.DataFrame, reference: dict, top_k: int = 3,
            min_impact: float = 0.01) -> list[list[dict]]:
    """Return, for every row of ``X``, the top reasons that push churn risk up.

    Each reason is ``{"feature", "reason", "impact"}`` where ``impact`` is the
    drop in churn probability (0–1) if that factor looked like a loyal customer's.
    """
    X = X[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES].reset_index(drop=True)
    base = model.predict_proba(X)[:, 1]

    variants, index = [], []
    for i, row in X.iterrows():
        for group in FEATURE_GROUPS:
            change = _counterfactual(group, row, reference)
            if all(str(row[k]) == str(v) for k, v in change.items()):
                continue  # already the low-risk value: nothing to explain
            new = row.copy()
            for k, v in change.items():
                new[k] = v
            variants.append(new)
            index.append((i, group))

    results: list[list[dict]] = [[] for _ in range(len(X))]
    if not variants:
        return results
    swapped = model.predict_proba(pd.DataFrame(variants).astype(X.dtypes.to_dict()))[:, 1]

    for (i, group), p in zip(index, swapped, strict=True):
        impact = float(base[i] - p)
        if impact >= min_impact:
            results[i].append({"feature": group, "reason": _describe(group, X.loc[i]),
                               "impact": round(impact, 4)})
    return [sorted(r, key=lambda d: -d["impact"])[:top_k] for r in results]


def global_drivers(model, X: pd.DataFrame, reference: dict, sample: int = 300) -> list[dict]:
    """Average impact of each factor across a sample, for dashboards."""
    Xs = X.sample(min(sample, len(X)), random_state=0)
    totals: dict[str, list[float]] = {}
    for reasons in explain(model, Xs, reference, top_k=len(FEATURE_GROUPS), min_impact=0.0):
        for r in reasons:
            totals.setdefault(r["feature"], []).append(r["impact"])
    out = [{"feature": f, "mean_impact": round(float(np.sum(v)) / len(Xs), 4)}
           for f, v in totals.items()]
    return sorted(out, key=lambda d: -d["mean_impact"])
