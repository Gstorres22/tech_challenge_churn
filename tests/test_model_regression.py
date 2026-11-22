from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from churn.config import MODEL_PATH, RAW_DATA_FILE, SEED, TEST_SIZE
from churn.data import clean, load_raw
from churn.predict import predict_proba

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


def test_model_artifacts_exist_and_within_size_limit():
    from churn.config import METRICS_PATH

    assert MODEL_PATH.exists(), f"Modelo obrigatório ausente em {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Métricas obrigatórias ausentes em {METRICS_PATH}"
    assert MODEL_PATH.stat().st_size < 5 * 1024 * 1024


def test_metrics_json_meets_quality_and_business_floors():
    from churn.config import METRICS_PATH, MIN_F1, MIN_RECALL, MIN_ROC_AUC

    assert METRICS_PATH.exists(), f"Métricas obrigatórias ausentes em {METRICS_PATH}"
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert EXPECTED_METRICS_KEYS.issubset(metrics.keys())
    assert float(metrics["roc_auc"]) >= MIN_ROC_AUC
    assert float(metrics["f1"]) >= MIN_F1
    assert float(metrics["recall"]) >= MIN_RECALL
    assert float(metrics["net_retained_revenue"]) > 0.0
    assert float(metrics["gain_vs_target_all"]) > 0.0
    assert float(metrics["overfitting_gap"]) <= 0.12


def test_live_holdout_evaluation_matches_metrics_json():
    from churn.config import (
        BUSINESS_ACTION_COST,
        BUSINESS_LTV_MONTHS,
        BUSINESS_RETENTION_SUCCESS_RATE,
        METRICS_PATH,
        MIN_F1,
        MIN_RECALL,
        MIN_ROC_AUC,
    )
    from churn.evaluate import business_metrics, classification_metrics
    from churn.train import split_data

    assert RAW_DATA_FILE.exists(), f"Dataset bruto ausente em {RAW_DATA_FILE}"
    assert MODEL_PATH.exists(), f"Modelo obrigatório ausente em {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Métricas obrigatórias ausentes em {METRICS_PATH}"

    df = clean(load_raw(RAW_DATA_FILE))
    _, X_test, _, y_test = split_data(df, test_size=TEST_SIZE, random_state=SEED)

    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    threshold = float(metrics["threshold"])

    cls_live = classification_metrics(y_test, y_proba, threshold=threshold)
    biz_live = business_metrics(
        y_test,
        y_proba,
        X_test["MonthlyCharges"],
        threshold=threshold,
        ltv_months=BUSINESS_LTV_MONTHS,
        retention_success_rate=BUSINESS_RETENTION_SUCCESS_RATE,
        action_cost=BUSINESS_ACTION_COST,
    )

    assert cls_live["roc_auc"] >= MIN_ROC_AUC
    assert cls_live["f1"] >= MIN_F1
    assert cls_live["recall"] >= MIN_RECALL
    assert biz_live["net_retained_revenue"] > 0.0
    assert biz_live["gain_vs_target_all"] > 0.0

    for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"):
        assert cls_live[key] == pytest.approx(float(metrics[key]), abs=1e-4)

    for key in ("net_retained_revenue", "roi", "gain_vs_target_all"):
        assert float(biz_live[key]) == pytest.approx(float(metrics[key]), abs=1e-4)


def test_inference_is_deterministic_and_directionally_sane():
    from churn.train import split_data

    assert RAW_DATA_FILE.exists(), f"Dataset bruto ausente em {RAW_DATA_FILE}"
    assert MODEL_PATH.exists(), f"Modelo obrigatório ausente em {MODEL_PATH}"

    df = clean(load_raw(RAW_DATA_FILE))
    _, X_test, _, _ = split_data(df, test_size=TEST_SIZE, random_state=SEED)
    model = joblib.load(MODEL_PATH)

    probs_first = model.predict_proba(X_test)[:, 1]
    probs_second = model.predict_proba(X_test)[:, 1]
    assert np.allclose(probs_first, probs_second, atol=1e-12)

    high_risk_customer = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.5,
        "TotalCharges": 98.5,
    }
    low_risk_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3740.0,
    }

    prob_high, prob_low = predict_proba([high_risk_customer, low_risk_customer], model=model)
    assert 0.0 <= prob_low < prob_high <= 1.0
