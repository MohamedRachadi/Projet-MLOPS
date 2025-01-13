from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd

import os

# Récupérer l'adresse de MLflow selon l'environnement
tracking_uri = "http://127.0.0.1:5000"  # En local
if os.getenv("DOCKER", "false") == "true":
    tracking_uri = "http://host.docker.internal:5000"  # Dans Docker

# Création de l'application FastAPI
app = FastAPI()

# Charger le modèle depuis MLflow
model = None
try:
    mlflow.set_tracking_uri(tracking_uri)
    model_name = "Best_Model_RandomForestRegressor"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
except Exception as e:
    print(f"Erreur de connexion ou de chargement du modèle : {str(e)}")

# Vérifier si le modèle a bien été chargé
if model is None:
    print("Le modèle n'a pas pu être chargé correctement.")
    raise HTTPException(status_code=500, detail="Le modèle MLflow n'a pas pu être chargé.")
    
# Définir un schéma Pydantic pour la validation des données entrantes
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Endpoint pour la prédiction
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible pour la prédiction.")
    
    # Convertir les données entrantes en dataframe
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])

    # Faire une prédiction
    prediction = model.predict(df)
    prediction_in_dollars_str = f"${prediction[0]*100000:,.2f}"  
    # Retourner la prédiction sous forme de réponse JSON
    return {"prediction": prediction_in_dollars_str}
