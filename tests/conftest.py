import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Amostra mínima no formato do CSV bruto do Telco Customer Churn."""
    base = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No",
    }
    new_customer = {
        **base,
        "customerID": "0002-TEST",
        "tenure": 0,
        "TotalCharges": " ",
        "SeniorCitizen": 1,
        "Churn": "Yes",
    }
    return pd.DataFrame([base, new_customer])
