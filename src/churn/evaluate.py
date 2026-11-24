"""Metrics and diagnostic plots logged as MLflow artifacts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def best_threshold(y_true, y_proba) -> float:
    """Threshold that maximises F1 on the given data."""
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def log_plots(y_true, y_proba, threshold: float, prefix: str = "test") -> None:
    y_pred = (y_proba >= threshold).astype(int)
    for name, fn in {
        "roc_curve": lambda ax: RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax),
        "pr_curve": lambda ax: PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax),
        "confusion_matrix": lambda ax: ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, display_labels=["Stay", "Churn"], cmap="Blues"),
    }.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        fn(ax)
        ax.set_title(f"{name.replace('_', ' ').title()} ({prefix})")
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{prefix}_{name}.png")
        plt.close(fig)


def log_feature_importance(pipeline, X, y, n_repeats: int = 5) -> None:
    """Permutation importance on raw input columns (model-agnostic)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                    scoring="roc_auc", random_state=42, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(X.columns)[order][::-1], result.importances_mean[order][::-1])
    ax.set_xlabel("Mean drop in ROC AUC")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)
