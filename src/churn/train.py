"""Train one model, evaluate it and log everything to MLflow.

Entry point: ``python -m churn.train --model-type hist_gb``
"""
import argparse
import json
import warnings

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn import config
from churn.evaluate import (
    best_threshold,
    compute_metrics,
    log_feature_importance,
    log_plots,
)
from churn.features import build_pipeline, split_xy
from churn.profile import build_profile
from churn.tracking import init_experiment, project_run

# Types skops must be told to trust when saving. Safe here because these are
# models this pipeline trained itself, never files from an unknown source.
TRUSTED_TYPES = [
    "churn.features.FeatureEngineer",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def load_splits():
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(config.PROCESSED_DIR / "test.parquet")
    return split_xy(train), split_xy(test)


def train_and_log(model_type: str, params: dict | None = None, run_name: str | None = None,
                  log_model: bool = True) -> dict:
    (X_train, y_train), (X_test, y_test) = load_splits()
    params = params or {}

    with mlflow.start_run(run_name=run_name or f"train_{model_type}", nested=True) as run:
        mlflow.set_tags({"model_type": model_type, "stage": "training"})
        mlflow.log_params({"model_type": model_type, **params})

        pipe = build_pipeline(model_type, params)
        cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mlflow.log_metrics({"cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})

        pipe.fit(X_train, y_train)
        train_proba = pipe.predict_proba(X_train)[:, 1]
        threshold = best_threshold(y_train, train_proba)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, test_proba, threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_param("decision_threshold", round(threshold, 3))

        if log_model:
            log_plots(y_test, test_proba, threshold)
            log_feature_importance(pipe, X_test, y_test)
            # Inputs never contain missing integers (validated upstream), so MLflow's
            # "integer columns can't hold NaN" hint doesn't apply here.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Integer columns.*")
                signature = infer_signature(X_test, test_proba)
                mlflow.sklearn.log_model(
                    pipe, name="model", signature=signature,
                    input_example=X_test.head(3), pyfunc_predict_fn="predict_proba",
                    # Ship our package with the model so it loads anywhere.
                    code_paths=[str(config.ROOT / "src" / "churn")],
                    skops_trusted_types=TRUSTED_TYPES,
                )
            mlflow.log_dict({"threshold": threshold}, "model_config.json")
            # Training distributions: used for explanations and drift monitoring.
            mlflow.log_dict(build_profile(X_train, y_train), "reference_profile.json")

        print(f"[{model_type}] cv_auc={cv_auc.mean():.4f} test_auc={metrics['roc_auc']:.4f} "
              f"f1={metrics['f1']:.4f} run_id={run.info.run_id}")
        return {"run_id": run.info.run_id, "cv_roc_auc": cv_auc.mean(), **metrics}


def main(model_type: str, params: str) -> None:
    init_experiment()
    train_and_log(model_type, json.loads(params))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="hist_gb", choices=["logreg", "random_forest", "hist_gb"])
    p.add_argument("--params", default="{}", help='JSON, e.g. \'{"max_depth": 5}\'')
    a = p.parse_args()
    with project_run():
        main(a.model_type, a.params)
