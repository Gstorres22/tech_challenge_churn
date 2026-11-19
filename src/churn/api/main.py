"""API de inferência (FastAPI) com os endpoints /health e /predict.

Execução local:
    uvicorn churn.api.main:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from churn import __version__
from churn.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from churn.predict import load_model, predict_proba

THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

app = FastAPI(
    title="Churn Prediction API",
    description="Retorna a propensão de churn de clientes de telecom.",
    version=__version__,
)


def _model_available() -> bool:
    try:
        load_model()
    except FileNotFoundError:
        return False
    return True


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica se a API está no ar e se o modelo foi carregado."""
    return HealthResponse(status="ok", model_loaded=_model_available())


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Recebe os dados de um cliente e retorna a probabilidade de churn."""
    if not _model_available():
        raise HTTPException(status_code=503, detail="Modelo ainda não disponível.")

    proba = predict_proba([customer.model_dump()])[0]
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn=proba >= THRESHOLD,
        threshold=THRESHOLD,
    )
