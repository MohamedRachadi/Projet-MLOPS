import requests

def test_mlflow_serving():
    response = requests.post(
        "http://localhost:5000/invocations",
        json={"inputs": [{"feature1": 1.0, "feature2": 2.0}]}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()