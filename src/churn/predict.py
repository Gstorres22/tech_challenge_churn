"""Carga do modelo serializado e inferência."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from churn.config import MODEL_PATH
from churn.data import clean


@lru_cache(maxsize=1)
def load_model(path: Path | str = MODEL_PATH):
    """Carrega o pipeline treinado (pré-processamento + modelo) uma única vez."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode o treino primeiro.")
    return joblib.load(path)


def predict_proba(records: list[dict], model=None) -> list[float]:
    """Retorna a probabilidade de churn (classe 1) para cada cliente."""
    model = model or load_model()
    df = clean(pd.DataFrame(records))
    return model.predict_proba(df)[:, 1].tolist()
