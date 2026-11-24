"""Download, validate, clean and split the Telco churn dataset.

Entry point: ``python -m churn.data --test-size 0.2``
Outputs train/test parquet files and logs dataset metadata to MLflow.
"""
import argparse
import urllib.request

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config
from churn.tracking import init_experiment, project_run


def download(force: bool = False) -> pd.DataFrame:
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force or not config.RAW_PATH.exists():
        print(f"Downloading dataset from {config.DATA_URL}")
        urllib.request.urlretrieve(config.DATA_URL, config.RAW_PATH)
    return pd.read_csv(config.RAW_PATH)


def validate(df: pd.DataFrame) -> None:
    """Fail fast if the schema is not what the pipeline expects."""
    required = {config.ID_COL, config.TARGET, *config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df[config.ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found")
    bad = set(df[config.TARGET].unique()) - {"Yes", "No", 0, 1}
    if bad:
        raise ValueError(f"Unexpected target values: {bad}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is a string with blanks for brand-new customers (tenure 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
    return df


def split(df: pd.DataFrame, test_size: float):
    return train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=config.RANDOM_STATE
    )


def main(test_size: float = 0.2) -> None:
    init_experiment()
    with mlflow.start_run(run_name="prepare_data", nested=True):
        df = download()
        validate(df)
        df = clean(df)
        train, test = split(df, test_size)

        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_path = config.PROCESSED_DIR / "train.parquet"
        test_path = config.PROCESSED_DIR / "test.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        mlflow.log_params({"test_size": test_size, "n_rows": len(df), "n_cols": df.shape[1]})
        mlflow.log_metrics({
            "churn_rate": float(df[config.TARGET].mean()),
            "n_train": len(train),
            "n_test": len(test),
        })
        for frame, name, context in [(train, "train", "training"), (test, "test", "evaluation")]:
            ds = mlflow.data.from_pandas(frame, name=name, targets=config.TARGET)
            mlflow.log_input(ds, context=context)
        print(f"Saved {len(train)} train / {len(test)} test rows to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.2)
    with project_run():
        main(p.parse_args().test_size)
