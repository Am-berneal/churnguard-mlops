import pytest

from churn import config
from churn.data import clean, split, validate


def test_validate_accepts_good_data(raw_df):
    validate(raw_df)


def test_validate_rejects_missing_column(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        validate(raw_df.drop(columns=["tenure"]))


def test_validate_rejects_duplicate_ids(raw_df):
    raw_df.loc[1, "customerID"] = raw_df.loc[0, "customerID"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(raw_df)


def test_clean_fixes_total_charges_and_target(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df[config.TARGET].unique()) <= {0, 1}


def test_split_is_stratified(raw_df):
    df = clean(raw_df)
    train, test = split(df, 0.25)
    assert len(test) == pytest.approx(len(df) * 0.25, abs=1)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.05
