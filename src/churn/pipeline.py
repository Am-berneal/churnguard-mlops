"""Full pipeline: prepare data -> baselines -> tune -> register champion.

Entry point: ``python -m churn.pipeline --n-trials 30``
"""
import argparse

from churn import data, register, tune
from churn.tracking import project_run
from churn.train import train_and_log


def main(n_trials: int = 30, test_size: float = 0.2) -> None:
    data.main(test_size)

    print("\n== Baselines ==")
    for model_type in ["logreg", "random_forest", "hist_gb"]:
        train_and_log(model_type)

    print("\n== Tuning ==")
    tune.main("hist_gb", n_trials)

    print("\n== Registry ==")
    register.main()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    a = p.parse_args()
    with project_run():
        main(a.n_trials, a.test_size)
