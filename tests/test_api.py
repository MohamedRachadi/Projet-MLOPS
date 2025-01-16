import requests

def test_api():
    # Définir l'URL de l'API
    api_url = "http://127.0.0.1:8000/predict"

    # Liste de différents jeux de données de test
    test_cases = [
        {
            "MedInc": 3.5,
            "HouseAge": 20,
            "AveRooms": 6.0,
            "AveBedrms": 1.5,
            "Population": 500,
            "AveOccup": 2.5,
            "Latitude": 37.5,
            "Longitude": -120.0
        },
        {
            "MedInc": 8.2,
            "HouseAge": 35,
            "AveRooms": 7.5,
            "AveBedrms": 2.0,
            "Population": 1500,
            "AveOccup": 3.0,
            "Latitude": 36.8,
            "Longitude": -118.5
        },
        {
            "MedInc": 1.2,
            "HouseAge": 5,
            "AveRooms": 4.0,
            "AveBedrms": 1.2,
            "Population": 200,
            "AveOccup": 1.5,
            "Latitude": 38.2,
            "Longitude": -121.0
        },
        {
            "MedInc": 12.0,
            "HouseAge": 50,
            "AveRooms": 8.5,
            "AveBedrms": 2.5,
            "Population": 5000,
            "AveOccup": 4.0,
            "Latitude": 37.2,
            "Longitude": -119.0
        },
        {
            "MedInc": 5.0,
            "HouseAge": 25,
            "AveRooms": 5.0,
            "AveBedrms": 1.8,
            "Population": 800,
            "AveOccup": 2.8,
            "Latitude": 36.5,
            "Longitude": -122.0
        }
    ]

    for i, api_test_data in enumerate(test_cases, 1):
        print(f"\n--------------------- Test Case {i} ---------------------n")
        print(f"Input Data: {api_test_data}\n")

        # Envoyer une requête POST à l'API
        response = requests.post(api_url, json=api_test_data)

        # Vérifier que la réponse est un succès
        assert response.status_code == 200, f"L'API n'a pas répondu correctement pour le cas {i}."

        # Afficher la réponse pour le débogage
        response_json = response.json()
        print(f"\nRéponse de l'API pour le cas {i} :", response_json)

        # Vérifications supplémentaires sur la réponse
        assert "prediction" in response_json, f"La clé 'prediction' est absente dans la réponse pour le cas {i}."
        prediction = response_json["prediction"]
        print(f"Prédiction reçue : {prediction}")
        assert isinstance(prediction, str) and prediction.startswith("$"), \
            f"La prédiction n'est pas dans un format valide pour le cas {i}."