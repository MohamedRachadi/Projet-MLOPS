import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_invalid_data():
    response = client.post("/predict", json={
        "MedInc": "string",  # Mauvais type de donnée
        "HouseAge": 30, 
        "AveRooms": 6.5, 
        "AveBedrms": 2.0, 
        "Population": 1000, 
        "AveOccup": 3.0, 
        "Latitude": 34.5, 
        "Longitude": -118.5
    })
    assert response.status_code == 422  # Unprocessable Entity (erreur validation)


def test_predict():
    response = client.post("/predict", json={
        "MedInc": 3.2, 
        "HouseAge": 30, 
        "AveRooms": 6.5, 
        "AveBedrms": 2.0, 
        "Population": 1000, 
        "AveOccup": 3.0, 
        "Latitude": 34.5, 
        "Longitude": -118.5
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
