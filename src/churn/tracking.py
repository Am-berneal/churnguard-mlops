"""MLflow helpers that work both standalone and under ``mlflow run``.

``mlflow run`` creates a run and passes it to the script through the
``MLFLOW_RUN_ID`` and ``MLFLOW_EXPERIMENT_ID`` environment variables. We attach
to that run and open every pipeline step as a nested child run, so one
``mlflow run`` shows up in the UI as one tidy tree.
"""
import os
from contextlib import contextmanager

import mlflow

from churn import config


def init_experiment() -> str:
    """Use the experiment chosen by ``mlflow run`` if any, else our default."""
    exp_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    if exp_id:
        return exp_id
    return mlflow.set_experiment(config.EXPERIMENT_NAME).experiment_id


@contextmanager
def project_run():
    """Attach to the run created by ``mlflow run`` (no-op when run directly)."""
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    init_experiment()
    if run_id:
        with mlflow.start_run(run_id=run_id):
            yield
    else:
        yield
