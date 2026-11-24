import pytest

from churn.data import clean
from churn.evaluate import best_threshold, compute_metrics
from churn.features import FeatureEngineer, build_pipeline, split_xy


def test_feature_engineer_adds_columns(raw_df):
    X, _ = split_xy(clean(raw_df))
    out = FeatureEngineer().transform(X)
    for col in ["avg_monthly_spend", "num_services", "is_long_contract"]:
        assert col in out
    assert out["num_services"].between(0, 8).all()
    assert (out["avg_monthly_spend"] >= 0).all()


def test_feature_engineer_handles_zero_tenure(raw_df):
    X, _ = split_xy(clean(raw_df))
    X.loc[X.index[0], "tenure"] = 0
    out = FeatureEngineer().transform(X)
    assert out["avg_monthly_spend"].notna().all()


@pytest.mark.parametrize("model_type", ["logreg", "random_forest", "hist_gb"])
def test_pipeline_trains_and_predicts(raw_df, model_type):
    X, y = split_xy(clean(raw_df))
    pipe = build_pipeline(model_type).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert compute_metrics(y, proba)["roc_auc"] > 0.6  # learns the planted signal


def test_unknown_category_does_not_crash(raw_df):
    X, y = split_xy(clean(raw_df))
    pipe = build_pipeline("logreg").fit(X, y)
    X_new = X.head(1).copy()
    X_new["PaymentMethod"] = "Crypto"
    assert pipe.predict_proba(X_new).shape == (1, 2)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_pipeline("xgboost_v99")


def test_best_threshold_in_range(raw_df):
    X, y = split_xy(clean(raw_df))
    proba = build_pipeline("logreg").fit(X, y).predict_proba(X)[:, 1]
    assert 0.1 <= best_threshold(y, proba) <= 0.9
