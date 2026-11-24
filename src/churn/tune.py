"""Hyperparameter tuning with Optuna; every trial is a nested MLflow run.

Entry point: ``python -m churn.tune --model-type hist_gb --n-trials 30``
"""
import argparse

import mlflow
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn import config
from churn.features import build_pipeline
from churn.tracking import init_experiment, project_run
from churn.train import load_splits, train_and_log

SEARCH_SPACES = {
    "hist_gb": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 10),
        "max_iter": t.suggest_int("max_iter", 100, 600, step=50),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    },
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    },
    "logreg": lambda t: {
        "C": t.suggest_float("C", 1e-3, 100, log=True),
    },
}


def main(model_type: str = "hist_gb", n_trials: int = 30) -> str:
    init_experiment()
    (X_train, y_train), _ = load_splits()
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    with mlflow.start_run(run_name=f"tune_{model_type}", nested=True):
        mlflow.set_tags({"model_type": model_type, "stage": "tuning"})
        mlflow.log_params({"model_type": model_type, "n_trials": n_trials})

        def objective(trial):
            params = SEARCH_SPACES[model_type](trial)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                score = cross_val_score(build_pipeline(model_type, params), X_train, y_train,
                                        cv=cv, scoring="roc_auc", n_jobs=-1).mean()
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", score)
            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", study.best_value)
        print(f"Best CV ROC AUC {study.best_value:.4f} with {study.best_params}")

        # Refit the best configuration as a full, model-logging child run.
        result = train_and_log(model_type, study.best_params,
                               run_name=f"best_{model_type}")
        mlflow.set_tag("best_run_id", result["run_id"])
        return result["run_id"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="hist_gb", choices=list(SEARCH_SPACES))
    p.add_argument("--n-trials", type=int, default=30)
    a = p.parse_args()
    with project_run():
        main(a.model_type, a.n_trials)
