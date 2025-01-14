import requests
import os

def test_model_serving():
    # L'URL de votre API MLflow servie
    #url = "http://localhost:5000/predict"  # Utilisation de /predict
    #url = "http://host.docker.internal:5000/predict"
    url = "http://localhost:5000/predict" if os.getenv("DOCKER", "false") == "false" else "http://host.docker.internal:5000/predict"

    # Données d'entrée pour tester le modèle (ici, un exemple basé sur le jeu de données California Housing)
    test_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984126984,
        "AveBedrms": 1.0,  # Assurez-vous d'inclure toutes les données nécessaires
        "Population": 322,
        "AveOccup": 0.972027972,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    # Requête POST à l'API
    response = requests.post(url, json=test_data, timeout=20)

    # Vérification de la réponse
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    print("Model served successfully! Response:", response.json())
