
from fastapi.testclient import TestClient

from question_B.level1.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_input():
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["risk_probability"] <= 1
    assert "id" in result


def test_predict_invalid_input():
    payload = {
        "age": -5,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
