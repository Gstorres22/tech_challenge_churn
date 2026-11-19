from churn.config import FEATURES, TARGET
from churn.data import clean
from churn.features import build_preprocessor


def test_clean_converts_types_and_drops_id(sample_raw_df):
    df = clean(sample_raw_df)

    assert "customerID" not in df.columns
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 1  # " " vira NaN
    assert set(df[TARGET]) == {0, 1}
    assert set(df["SeniorCitizen"]) == {"Yes", "No"}


def test_preprocessor_outputs_no_missing_values(sample_raw_df):
    df = clean(sample_raw_df)
    X = build_preprocessor().fit_transform(df[FEATURES])

    assert X.shape[0] == len(df)
    assert not (X != X).any()  # sem NaN após imputação
