from __future__ import annotations

import pytest

from churn.evaluate import classification_metrics


def test_business_config_constants():
    from churn.config import (
        BUSINESS_ACTION_COST,
        BUSINESS_LTV_MONTHS,
        BUSINESS_RETENTION_SUCCESS_RATE,
    )

    assert BUSINESS_LTV_MONTHS == 6
    assert float(BUSINESS_RETENTION_SUCCESS_RATE) == pytest.approx(0.30)
    assert float(BUSINESS_ACTION_COST) == pytest.approx(30.0)


def test_classification_metrics_includes_pr_auc():
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert set(metrics) == {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"}
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_business_metrics_exact_tp_fp_fn_tn_calculations():
    from churn.evaluate import business_metrics

    # C0: TP (100.0), C1: FP (80.0), C2: FN (20.0), C3: TN (40.0), C4: TN (60.0)
    y_true = [1, 0, 1, 0, 0]
    y_proba = [0.8, 0.7, 0.3, 0.2, 0.1]
    monthly_charges = [100.0, 80.0, 20.0, 40.0, 60.0]

    result = business_metrics(
        y_true,
        y_proba,
        monthly_charges,
        threshold=0.5,
        ltv_months=6,
        retention_success_rate=0.30,
        action_cost=30.0,
    )

    assert isinstance(result["targeted_customers"], int)
    assert isinstance(result["true_positives_caught"], int)
    assert isinstance(result["false_positives"], int)
    assert result["targeted_customers"] == 2
    assert result["true_positives_caught"] == 1
    assert result["false_positives"] == 1

    # gross = 100 * 6 * 0.30 = 180.0; cost = 2 * 30 = 60.0; net = 120.0; roi = 2.0
    assert result["gross_saved_revenue"] == pytest.approx(180.0)
    assert result["campaign_cost"] == pytest.approx(60.0)
    assert result["net_retained_revenue"] == pytest.approx(120.0)
    assert result["roi"] == pytest.approx(2.0)

    # baseline target-all: (100 + 20) * 6 * 0.30 - 5 * 30 = 216.0 - 150.0 = 66.0
    assert result["baseline_target_all_net_revenue"] == pytest.approx(66.0)
    assert result["gain_vs_target_all"] == pytest.approx(54.0)


def test_business_metrics_zero_campaign_cost_returns_zero_roi():
    from churn.evaluate import business_metrics

    y_true = [0, 1, 0]
    y_proba = [0.2, 0.4, 0.1]
    monthly_charges = [50.0, 70.0, 40.0]

    no_targets = business_metrics(y_true, y_proba, monthly_charges, threshold=0.9)
    assert no_targets["targeted_customers"] == 0
    assert no_targets["campaign_cost"] == pytest.approx(0.0)
    assert no_targets["net_retained_revenue"] == pytest.approx(0.0)
    assert no_targets["roi"] == pytest.approx(0.0)

    free_campaign = business_metrics(
        y_true,
        [0.9, 0.9, 0.1],
        monthly_charges,
        threshold=0.5,
        action_cost=0.0,
    )
    assert free_campaign["campaign_cost"] == pytest.approx(0.0)
    assert free_campaign["roi"] == pytest.approx(0.0)


def test_business_metrics_only_false_positives_produces_negative_net_revenue():
    from churn.evaluate import business_metrics

    result = business_metrics(
        y_true=[0, 0, 1],
        y_proba=[0.9, 0.8, 0.1],
        monthly_charges=[100.0, 100.0, 100.0],
        threshold=0.5,
        ltv_months=6,
        retention_success_rate=0.30,
        action_cost=30.0,
    )

    assert result["true_positives_caught"] == 0
    assert result["false_positives"] == 2
    assert result["gross_saved_revenue"] == pytest.approx(0.0)
    assert result["campaign_cost"] == pytest.approx(60.0)
    assert result["net_retained_revenue"] == pytest.approx(-60.0)
    assert result["roi"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("y_true", "y_proba", "charges", "kwargs"),
    [
        ([], [], [], {}),
        ([0, 1], [0.2], [50.0, 60.0], {}),
        ([0, 1], [0.2, 0.8], [50.0], {}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"threshold": -0.1}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"threshold": 1.1}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"ltv_months": 0}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"retention_success_rate": -0.1}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"retention_success_rate": 1.2}),
        ([0, 1], [0.2, 0.8], [50.0, 60.0], {"action_cost": -1.0}),
    ],
)
def test_business_metrics_validation_errors(y_true, y_proba, charges, kwargs):
    from churn.evaluate import business_metrics

    with pytest.raises(ValueError):
        business_metrics(y_true, y_proba, charges, **kwargs)


def test_find_optimal_threshold_beats_default_05_on_imbalanced_profitable_case():
    from churn.evaluate import business_metrics, find_optimal_threshold

    # 20% churn: limiar 0.50 captura apenas 1 TP (+150);
    # limiar entre 0.15 e 0.31 captura 2 TPs e 1 FP (+270).
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_proba = [0.65, 0.31, 0.35, 0.14, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01]
    monthly_charges = [100.0] * len(y_true)

    opt_threshold, opt_metrics = find_optimal_threshold(y_true, y_proba, monthly_charges)
    default_metrics = business_metrics(y_true, y_proba, monthly_charges, threshold=0.5)

    assert isinstance(opt_threshold, float)
    assert opt_threshold < 0.50
    assert default_metrics["net_retained_revenue"] == pytest.approx(150.0)
    assert opt_metrics["net_retained_revenue"] == pytest.approx(270.0)
    assert opt_metrics["net_retained_revenue"] > default_metrics["net_retained_revenue"]


def test_find_optimal_threshold_respects_min_recall_constraint():
    from churn.evaluate import find_optimal_threshold

    y_true = [1, 1, 0, 0]
    y_proba = [0.8, 0.4, 0.3, 0.1]
    monthly_charges = [100.0, 40.0, 100.0, 100.0]

    unconstrained_t, _ = find_optimal_threshold(
        y_true,
        y_proba,
        monthly_charges,
        thresholds=[0.35, 0.50],
        action_cost=90.0,
    )
    constrained_t, constrained_metrics = find_optimal_threshold(
        y_true,
        y_proba,
        monthly_charges,
        thresholds=[0.35, 0.50],
        action_cost=90.0,
        min_recall=0.90,
    )

    assert unconstrained_t == pytest.approx(0.50)
    assert constrained_t == pytest.approx(0.35)
    assert constrained_metrics["net_retained_revenue"] == pytest.approx(72.0)
