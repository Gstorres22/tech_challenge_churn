"""Configurações centrais do projeto: caminhos, seed e definição das features."""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------ reprodutibilidade
SEED: int = 42

# ------------------------------------------------------------------ caminhos
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

RAW_DATA_FILE: Path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Pode ser sobrescrito por variável de ambiente (útil no container / AWS).
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", MODELS_DIR / "churn_model.joblib"))

# ------------------------------------------------------------------ dataset (Telco - IBM)
ID_COLUMN: str = "customerID"
TARGET: str = "Churn"

NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ------------------------------------------------------------------ treino
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
