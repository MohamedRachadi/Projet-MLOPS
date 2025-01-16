import mlflow
import pandas as pd

def test_model_prediction():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Charger le modèle directement depuis MLflow
    model_name = "Best_Model_RandomForestRegressor"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

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

    for i, data in enumerate(test_cases, 1):
        print(f"\n--------------------- Test Case {i} ---------------------n")
        print(f"Input Data: {data}")

        # Convertir les données en DataFrame
        test_data = pd.DataFrame([data])

        # Faire une prédiction
        prediction = model.predict(test_data)

        # Vérifier que la prédiction est valide
        assert len(prediction) == 1, f"Le modèle n'a pas retourné une seule valeur pour le cas {i}."
        assert prediction[0] > 0, f"La prédiction du modèle doit être un nombre positif pour le cas {i}."

        # Afficher les résultats pour débogage
        print(f"Prédiction pour le cas {i} : {prediction[0]}")