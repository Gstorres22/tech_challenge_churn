from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn.api.main import THRESHOLD, app
from churn.config import FEATURES, SEED, TARGET
from churn.data import clean
from churn.train import build_pipeline

client = TestClient(app)

VALID_PAYLOAD = {
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


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_payload():
    response = client.post("/predict", json={"tenure": -1})

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_missing(monkeypatch):
    def _raise_not_found():
        raise FileNotFoundError("modelo ausente")

    monkeypatch.setattr("churn.api.main.load_model", _raise_not_found)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503


def test_predict_returns_probability_and_flag_when_model_loaded(monkeypatch, sample_raw_df):
    df = clean(sample_raw_df)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipeline.fit(df[FEATURES], df[TARGET])

    monkeypatch.setattr("churn.api.main.load_model", lambda: pipeline)
    monkeypatch.setattr("churn.predict.load_model", lambda: pipeline)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold"] == THRESHOLD
    assert body["churn"] is (body["churn_probability"] >= body["threshold"])
