"""Register the best model and promote it to the ``champion`` alias.

A new model only becomes champion if it beats the current champion on
the chosen metric by at least ``--min-improvement``. The default metric is
cross-validated ROC AUC, so the test set is never used to pick models.

Entry point: ``python -m churn.register --metric cv_roc_auc_mean``
"""
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn import config
from churn.tracking import init_experiment, project_run


def find_best_run(metric: str):
    runs = mlflow.search_runs(
        [init_experiment()],
        filter_string=f"metrics.{metric} > 0 and tags.stage = 'training'",
        order_by=[f"metrics.{metric} DESC"], max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No training runs with a logged model found.")
    return runs.iloc[0]


def current_champion(client: MlflowClient, metric: str):
    """Return (run_id, score) of the current champion, or (None, None)."""
    try:
        mv = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    except MlflowException:
        return None, None
    return mv.run_id, client.get_run(mv.run_id).data.metrics.get(metric)


def main(metric: str = "cv_roc_auc_mean", min_improvement: float = 0.0) -> None:
    client = MlflowClient()
    best = find_best_run(metric)
    score = best[f"metrics.{metric}"]
    champ_run, champ = current_champion(client, metric)
    champ_txt = f"{champ:.4f}" if champ is not None else "none"
    print(f"Best candidate run {best.run_id}: {metric}={score:.4f}; current champion: {champ_txt}")

    if champ_run == best.run_id:
        print("Best run is already the champion. Nothing to do.")
        return
    if champ is not None and score <= champ + min_improvement:
        print("Candidate does not beat the champion. Nothing promoted.")
        return

    mv = mlflow.register_model(f"runs:/{best.run_id}/model", config.MODEL_NAME)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, mv.version)
    client.set_model_version_tag(config.MODEL_NAME, mv.version, metric, f"{score:.4f}")
    client.update_model_version(
        config.MODEL_NAME, mv.version,
        description=f"{best['tags.model_type']} model, {metric}={score:.4f}",
    )
    print(f"Registered {config.MODEL_NAME} v{mv.version} as '{config.CHAMPION_ALIAS}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="cv_roc_auc_mean")
    p.add_argument("--min-improvement", type=float, default=0.0)
    a = p.parse_args()
    with project_run():
        main(a.metric, a.min_improvement)
