"""Send realistic traffic to a running API, to demo the monitoring page.

    python -m churn.simulate --n 300            # normal traffic (no drift)
    python -m churn.simulate --n 300 --drift    # shifted traffic (drift is detected)

``--drift`` simulates a market change: newer customers, higher prices and more
month-to-month contracts than the model saw during training.
"""
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

from churn import config


def make_traffic(n: int, drift: bool, seed: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(config.PROCESSED_DIR / "test.parquet")
    df = df.sample(n, replace=n > len(df), random_state=seed).reset_index(drop=True)
    if drift:
        rng = np.random.default_rng(seed)
        df["tenure"] = (df["tenure"] * rng.uniform(0.2, 0.6, n)).astype(int)
        df["MonthlyCharges"] = (df["MonthlyCharges"] * rng.uniform(1.15, 1.4, n)).round(2)
        df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
        flip = rng.uniform(size=n) < 0.5
        df.loc[flip, "Contract"] = "Month-to-month"
    return df.drop(columns=[config.TARGET])


def send(df: pd.DataFrame, url: str, batch: int = 100) -> None:
    for start in range(0, len(df), batch):
        body = json.dumps(df.iloc[start:start + batch].to_dict("records")).encode()
        req = urllib.request.Request(f"{url}/predict/batch", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:  # noqa: S310 - user-supplied local URL
            s = json.load(r)["summary"]
        print(f"sent {start + s['n_customers']}/{len(df)} · high risk in batch: {s['n_high']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--drift", action="store_true")
    p.add_argument("--url", default="http://localhost:8000")
    a = p.parse_args()
    send(make_traffic(a.n, a.drift), a.url)
