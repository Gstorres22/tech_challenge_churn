from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from churn.config import FEATURES, SEED, TARGET
from churn.data import clean
from churn.train import build_pipeline, get_candidate_models

EXPECTED_METRICS_KEYS = {
    "champion_model",
    "threshold",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "net_retained_revenue",
    "roi",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "cv_train_roc_auc_mean",
    "overfitting_gap",
    "baseline_target_all_net_revenue",
    "gain_vs_target_all",
    "holdout_default_threshold",
    "holdout_optimal_threshold",
    "business_metrics",
    "dataset_info",
    "all_models",
}


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Dataset sintético realista com 100 clientes e duas classes balanceáveis em CV."""
    rng = np.random.default_rng(SEED)
    n_rows = 100
    n_churn = 36
    rows: list[dict[str, object]] = []

    for idx in range(n_rows):
        is_churn = idx < n_churn
        tenure = (
            0
            if idx in (0, n_churn)
            else int(rng.integers(1, 14) if is_churn else rng.integers(24, 73))
        )
        monthly = round(
            float(rng.uniform(75.0, 110.0) if is_churn else rng.uniform(20.0, 65.0)),
            2,
        )
        total_charges = " " if tenure == 0 else f"{tenure * monthly:.2f}"

        rows.append(
            {
                "customerID": f"{idx:04d}-SYNTH",
                "gender": str(rng.choice(["Female", "Male"])),
                "SeniorCitizen": int(rng.choice([0, 1], p=[0.6, 0.4] if is_churn else [0.9, 0.1])),
                "Partner": "No" if is_churn else "Yes",
                "Dependents": "No" if is_churn else str(rng.choice(["Yes", "No"])),
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": str(rng.choice(["Yes", "No"])),
                "InternetService": ("Fiber optic" if is_churn else str(rng.choice(["DSL", "No"]))),
                "OnlineSecurity": (
                    "No" if is_churn else str(rng.choice(["Yes", "No internet service"]))
                ),
                "OnlineBackup": "No" if is_churn else "Yes",
                "DeviceProtection": "No" if is_churn else "Yes",
                "TechSupport": (
                    "No" if is_churn else str(rng.choice(["Yes", "No internet service"]))
                ),
                "StreamingTV": "Yes" if is_churn else "No",
                "StreamingMovies": "Yes" if is_churn else "No",
                "Contract": (
                    "Month-to-month" if is_churn else str(rng.choice(["One year", "Two year"]))
                ),
                "PaperlessBilling": "Yes" if is_churn else "No",
                "PaymentMethod": (
                    "Electronic check"
                    if is_churn
                    else str(rng.choice(["Bank transfer (automatic)", "Credit card (automatic)"]))
                ),
                "MonthlyCharges": monthly,
                "TotalCharges": total_charges,
                "Churn": "Yes" if is_churn else "No",
            }
        )

    order = rng.permutation(n_rows)
    return pd.DataFrame([rows[i] for i in order])


def test_split_data_is_stratified_deterministic_and_disjoint(synthetic_raw_df):
    from churn.train import split_data

    df = clean(synthetic_raw_df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=SEED)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(df, test_size=0.2, random_state=SEED)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns
    assert set(X_train.columns) == set(FEATURES)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert X_train.equals(X_train_2)
    assert X_test.equals(X_test_2)
    assert y_train.equals(y_train_2)
    assert y_test.equals(y_test_2)


def test_no_data_leakage_from_test_set_extremes(tmp_path, synthetic_raw_df):
    from churn.train import split_data, train_and_evaluate

    cleaned = clean(synthetic_raw_df)
    X_train, X_test, _, _ = split_data(cleaned, test_size=0.2, random_state=SEED)
    test_indices = X_test.index

    clean_csv = tmp_path / "clean.csv"
    synthetic_raw_df.to_csv(clean_csv, index=False)

    poisoned_raw_df = synthetic_raw_df.copy()
    poisoned_raw_df.loc[test_indices, "MonthlyCharges"] = 1e9
    poisoned_raw_df.loc[test_indices, "TotalCharges"] = "1000000000.0"
    poisoned_csv = tmp_path / "poisoned.csv"
    poisoned_raw_df.to_csv(poisoned_csv, index=False)

    clean_res = train_and_evaluate(
        raw_data_path=clean_csv,
        model_output_path=tmp_path / "clean_model.joblib",
        metrics_output_path=tmp_path / "clean_metrics.json",
        comparison_output_path=tmp_path / "clean_comparison.csv",
        cv_folds=3,
    )
    poisoned_res = train_and_evaluate(
        raw_data_path=poisoned_csv,
        model_output_path=tmp_path / "poisoned_model.joblib",
        metrics_output_path=tmp_path / "poisoned_metrics.json",
        comparison_output_path=tmp_path / "poisoned_comparison.csv",
        cv_folds=3,
    )

    assert clean_res["champion_model"] == poisoned_res["champion_model"]

    prep_clean = joblib.load(tmp_path / "clean_model.joblib").named_steps["preprocessor"]
    prep_poisoned = joblib.load(tmp_path / "poisoned_model.joblib").named_steps["preprocessor"]

    num_clean = prep_clean.named_transformers_["num"]
    num_poisoned = prep_poisoned.named_transformers_["num"]

    assert np.allclose(
        num_clean.named_steps["imputer"].statistics_,
        num_poisoned.named_steps["imputer"].statistics_,
    )
    assert np.allclose(
        num_clean.named_steps["scaler"].mean_,
        num_poisoned.named_steps["scaler"].mean_,
    )
    assert np.allclose(
        num_clean.named_steps["scaler"].scale_,
        num_poisoned.named_steps["scaler"].scale_,
    )
    assert np.allclose(
        prep_clean.transform(X_train),
        prep_poisoned.transform(X_train),
    )


def test_pipeline_handles_unseen_categories_and_blank_total_charges(synthetic_raw_df):
    from churn.train import split_data

    df = clean(synthetic_raw_df)
    X_train, _, y_train, _ = split_data(df, test_size=0.2, random_state=SEED)

    pipeline = build_pipeline(get_candidate_models()["logistic_regression"])
    pipeline.fit(X_train, y_train)

    unseen_raw = synthetic_raw_df.iloc[:3].copy()
    unseen_raw["Contract"] = "Three year"
    unseen_raw["PaymentMethod"] = "Pix"
    unseen_raw["InternetService"] = "Satellite"
    unseen_raw["TotalCharges"] = " "
    X_unseen = clean(unseen_raw)[FEATURES]

    probs = pipeline.predict_proba(X_unseen)[:, 1]
    assert len(probs) == 3
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_evaluate_candidates_cv_evaluates_all_three_families(synthetic_raw_df):
    from churn.train import evaluate_candidates_cv, split_data

    df = clean(synthetic_raw_df)
    X_train, _, y_train, _ = split_data(df, test_size=0.2, random_state=SEED)

    df_summary, details = evaluate_candidates_cv(X_train, y_train, cv_folds=3, random_state=SEED)

    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 3
    assert isinstance(details, dict)
    assert set(details) == {"logistic_regression", "random_forest", "mlp"}
    numeric_cols = df_summary.select_dtypes(include="number")
    assert not numeric_cols.isna().any().any()


def test_train_and_evaluate_end_to_end_in_tmp_path(tmp_path, synthetic_raw_df):
    from churn.train import split_data, train_and_evaluate

    raw_csv = tmp_path / "telco_raw.csv"
    model_path = tmp_path / "models" / "churn_model.joblib"
    metrics_path = tmp_path / "models" / "metrics.json"
    comparison_path = tmp_path / "reports" / "model_comparison.csv"
    synthetic_raw_df.to_csv(raw_csv, index=False)

    result = train_and_evaluate(
        raw_data_path=raw_csv,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        comparison_output_path=comparison_path,
        cv_folds=3,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert comparison_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert EXPECTED_METRICS_KEYS.issubset(metrics.keys())
    assert EXPECTED_METRICS_KEYS.issubset(result.keys())
    assert metrics["champion_model"] in {"logistic_regression", "random_forest", "mlp"}
    assert 0.0 < float(metrics["threshold"]) < 1.0

    comparison_df = pd.read_csv(comparison_path)
    assert len(comparison_df) == 3

    _, X_test, _, _ = split_data(clean(synthetic_raw_df), test_size=0.2, random_state=SEED)
    loaded = joblib.load(model_path)
    probs = loaded.predict_proba(X_test)[:, 1]
    assert len(probs) == len(X_test)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
