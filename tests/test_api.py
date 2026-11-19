from fastapi.testclient import TestClient

from churn.api.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_payload():
    response = client.post("/predict", json={"tenure": -1})

    assert response.status_code == 422
