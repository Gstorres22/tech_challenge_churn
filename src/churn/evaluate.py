"""Métricas de avaliação (técnicas e de negócio)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Calcula as métricas técnicas a partir das probabilidades previstas."""
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


# TODO (Etapa 1): definir a métrica de negócio (ex.: custo de churn evitado vs.
# custo de ação de retenção) e implementá-la aqui.
