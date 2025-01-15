import mlflow
import pandas as pd

def test_model_prediction():
    # Charger le modèle directement depuis MLflow
    model_name = "Best_Model_RandomForestRegressor"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

    # Données de test pour le modèle
    test_data = pd.DataFrame([{
        "MedInc": 3.5,
        "HouseAge": 20,
        "AveRooms": 6.0,
        "AveBedrms": 1.5,
        "Population": 500,
        "AveOccup": 2.5,
        "Latitude": 37.5,
        "Longitude": -120.0
    }])

    # Faire une prédiction
    prediction = model.predict(test_data)

    # Vérifier que la prédiction est valide
    assert len(prediction) == 1, "Le modèle n'a pas retourné une seule valeur."
    assert prediction[0] > 0, "La prédiction du modèle doit être un nombre positif."

    print("Prédiction du modèle :", prediction[0])