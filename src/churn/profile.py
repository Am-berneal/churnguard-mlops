"""Reference profile of the training data.

Saved next to every logged model as ``reference_profile.json``. It is used for:
- explanations: the "typical customer" each feature is compared against
- monitoring: the training distributions that live traffic is checked against
"""
import numpy as np
import pandas as pd

from churn import config

N_BINS = 10


def build_profile(X: pd.DataFrame, y: pd.Series | None = None) -> dict:
    profile = {"n_rows": len(X), "numeric": {}, "categorical": {}}
    if y is not None:
        profile["churn_rate"] = float(y.mean())

    for col in config.NUMERIC_FEATURES:
        s = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)))
        counts, _ = np.histogram(s, bins=edges)
        profile["numeric"][col] = {
            "median": float(s.median()),
            "mean": float(s.mean()),
            "edges": edges.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }

    for col in config.CATEGORICAL_FEATURES:
        freq = X[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = {
            "mode": str(freq.index[0]),
            "proportions": {str(k): float(v) for k, v in freq.items()},
        }
    return profile


def reference_customer(profile: dict) -> dict:
    """A 'typical' customer: median of each numeric feature, mode of each category."""
    ref = {c: v["median"] for c, v in profile["numeric"].items()}
    ref.update({c: v["mode"] for c, v in profile["categorical"].items()})
    return ref
