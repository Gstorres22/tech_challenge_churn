"""Métricas de avaliação (técnicas e de negócio)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import (
    BUSINESS_ACTION_COST,
    BUSINESS_LTV_MONTHS,
    BUSINESS_RETENTION_SUCCESS_RATE,
)


def _validate_business_inputs(
    y_true,
    y_proba,
    monthly_charges,
    *,
    threshold: float,
    ltv_months: int,
    retention_success_rate: float,
    action_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valida vetores de entrada e parâmetros financeiros da avaliação."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    charges_arr = np.asarray(monthly_charges, dtype=float)

    if y_true_arr.ndim != 1 or y_proba_arr.ndim != 1 or charges_arr.ndim != 1:
        raise ValueError("y_true, y_proba e monthly_charges devem ser vetores unidimensionais.")
    if y_true_arr.size == 0:
        raise ValueError("y_true, y_proba e monthly_charges não podem ser vazios.")
    if not (y_true_arr.shape[0] == y_proba_arr.shape[0] == charges_arr.shape[0]):
        raise ValueError("y_true, y_proba e monthly_charges devem ter o mesmo tamanho.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold deve estar em [0.0, 1.0]; recebido {threshold}.")
    if isinstance(ltv_months, bool) or ltv_months <= 0:
        raise ValueError(f"ltv_months deve ser maior que 0; recebido {ltv_months}.")
    if not (0.0 <= retention_success_rate <= 1.0):
        raise ValueError(
            f"retention_success_rate deve estar em [0.0, 1.0]; recebido {retention_success_rate}."
        )
    if action_cost < 0.0:
        raise ValueError(f"action_cost não pode ser negativo; recebido {action_cost}.")

    return y_true_arr, y_proba_arr, charges_arr


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Calcula as métricas técnicas a partir das probabilidades previstas."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold deve estar em [0.0, 1.0]; recebido {threshold}.")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_true_arr.size == 0 or y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true e y_proba devem ser não vazios e ter o mesmo tamanho.")

    y_pred = (y_proba_arr >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_proba_arr)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }


def business_metrics(
    y_true,
    y_proba,
    monthly_charges,
    *,
    threshold: float = 0.5,
    ltv_months: int = BUSINESS_LTV_MONTHS,
    retention_success_rate: float = BUSINESS_RETENTION_SUCCESS_RATE,
    action_cost: float = BUSINESS_ACTION_COST,
) -> dict[str, float | int]:
    """Calcula o impacto financeiro líquido de uma campanha de retenção orientada ao limiar."""
    y_true_arr, y_proba_arr, charges_arr = _validate_business_inputs(
        y_true,
        y_proba,
        monthly_charges,
        threshold=threshold,
        ltv_months=ltv_months,
        retention_success_rate=retention_success_rate,
        action_cost=action_cost,
    )

    y_pred = y_proba_arr >= threshold
    tp_mask = (y_true_arr == 1) & y_pred
    fp_mask = (y_true_arr == 0) & y_pred

    targeted_customers = int(np.sum(y_pred))
    true_positives_caught = int(np.sum(tp_mask))
    false_positives = int(np.sum(fp_mask))

    gross_saved_revenue = float(np.sum(charges_arr[tp_mask]) * ltv_months * retention_success_rate)
    campaign_cost = float(targeted_customers * action_cost)
    net_retained_revenue = float(gross_saved_revenue - campaign_cost)
    roi = float(net_retained_revenue / campaign_cost) if campaign_cost > 0.0 else 0.0

    all_churn_mask = y_true_arr == 1
    baseline_gross = float(
        np.sum(charges_arr[all_churn_mask]) * ltv_months * retention_success_rate
    )
    baseline_target_all_net_revenue = float(baseline_gross - (len(y_true_arr) * action_cost))
    gain_vs_target_all = float(net_retained_revenue - baseline_target_all_net_revenue)

    return {
        "gross_saved_revenue": gross_saved_revenue,
        "campaign_cost": campaign_cost,
        "net_retained_revenue": net_retained_revenue,
        "roi": roi,
        "targeted_customers": targeted_customers,
        "true_positives_caught": true_positives_caught,
        "false_positives": false_positives,
        "baseline_target_all_net_revenue": baseline_target_all_net_revenue,
        "gain_vs_target_all": gain_vs_target_all,
    }


def find_optimal_threshold(
    y_true,
    y_proba,
    monthly_charges,
    *,
    thresholds: Sequence[float] | np.ndarray | None = None,
    ltv_months: int = BUSINESS_LTV_MONTHS,
    retention_success_rate: float = BUSINESS_RETENTION_SUCCESS_RATE,
    action_cost: float = BUSINESS_ACTION_COST,
    min_f1: float = 0.0,
    min_recall: float = 0.0,
) -> tuple[float, dict[str, float | int]]:
    """Seleciona o limiar que maximiza net_retained_revenue sob pisos técnicos factíveis."""
    grid = (
        np.round(np.arange(0.15, 0.86, 0.01), 2)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    if grid.size == 0:
        raise ValueError("thresholds não pode ser vazio.")

    candidates: list[tuple[float, dict[str, float | int]]] = []
    for candidate in grid:
        threshold_val = float(candidate)
        cls = classification_metrics(y_true, y_proba, threshold=threshold_val)
        biz = business_metrics(
            y_true,
            y_proba,
            monthly_charges,
            threshold=threshold_val,
            ltv_months=ltv_months,
            retention_success_rate=retention_success_rate,
            action_cost=action_cost,
        )
        candidates.append((threshold_val, {**cls, **biz}))

    feasible = [
        (threshold_val, metrics)
        for threshold_val, metrics in candidates
        if float(metrics["f1"]) >= min_f1 and float(metrics["recall"]) >= min_recall
    ]
    pool = feasible if feasible else candidates

    best_threshold, best_metrics = max(
        pool,
        key=lambda item: (
            round(float(item[1]["net_retained_revenue"]), 9),
            round(float(item[1]["f1"]), 9),
        ),
    )
    return float(best_threshold), best_metrics
