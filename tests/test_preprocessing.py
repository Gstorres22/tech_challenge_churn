import pytest

from churn.config import FEATURES, TARGET
from churn.data import clean, load_raw
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


def test_clean_works_without_target_column_for_inference(sample_raw_df):
    df_inference = sample_raw_df.drop(columns=[TARGET])
    cleaned = clean(df_inference)

    assert TARGET not in cleaned.columns
    assert "customerID" not in cleaned.columns
    assert set(cleaned["SeniorCitizen"]) == {"Yes", "No"}


def test_load_raw_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "missing.csv")


def test_load_raw_reads_csv_dataframe(tmp_path, sample_raw_df):
    csv_path = tmp_path / "telco.csv"
    sample_raw_df.to_csv(csv_path, index=False)

    loaded = load_raw(csv_path)

    assert len(loaded) == len(sample_raw_df)
    assert list(loaded.columns) == list(sample_raw_df.columns)
