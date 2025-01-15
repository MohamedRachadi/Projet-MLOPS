import requests

def test_api():
    # Définir l'URL de l'API
    api_url = "http://127.0.0.1:8000/predict"

    # Données de test simples
    api_test_data = {
        "MedInc": 3.5,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.5,
        "Population": 500,
        "AveOccup": 2.5,
        "Latitude": 37.5,
        "Longitude": -120.0
    }

    # Envoyer une requête POST à l'API
    response = requests.post(api_url, json=api_test_data)

    # Vérifier que la réponse est un succès
    assert response.status_code == 200, "L'API n'a pas répondu correctement."

    # Afficher la réponse pour le débogage
    print("Réponse de l'API :", response.json())