import requests

def test_predict():
    data = {
        "MedInc": 3.2,
        "HouseAge": 30,
        "AveRooms": 6.5,
        "AveBedrms": 2.0,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.5,
        "Longitude": -118.5
    }
    #response = requests.post("http://host.docker.internal:8000/predict", json=data)
    response = requests.post("http://localhost:8000/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
