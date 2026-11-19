"""Contratos de entrada e saída da API (Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """Dados de um cliente, no mesmo formato das colunas do dataset Telco."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 79.85,
                "TotalCharges": 958.2,
            }
        }
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
