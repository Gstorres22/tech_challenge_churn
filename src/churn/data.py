"""Carga e limpeza do dataset bruto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import ID_COLUMN, RAW_DATA_FILE, TARGET


def load_raw(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto do Telco Customer Churn."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Veja data/README.md para baixar."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza mínima e determinística, compartilhada entre treino e inferência.

    - ``TotalCharges`` vem como texto com espaços em branco para clientes novos:
      converte para numérico (vazios viram NaN e são tratados pelo imputer).
    - ``SeniorCitizen`` vem como 0/1: vira categoria "Yes"/"No" para padronizar.
    - ``Churn`` (se presente) vira 0/1.
    - Remove a coluna de ID, que não é feature.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].map({1: "Yes", 0: "No"}).fillna(df["SeniorCitizen"])
        )

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype("int64")

    return df.drop(columns=[ID_COLUMN], errors="ignore")
