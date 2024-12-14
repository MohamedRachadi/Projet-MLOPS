from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# Création de l'application FastAPI
app = FastAPI()

# Charger le modèle depuis MLflow
model = mlflow.sklearn.load_model("models:/best_model/1")  # Charger le modèle avec la version appropriée

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
    """
    Endpoint qui prend des données sous forme JSON et retourne la prédiction du modèle.
    """
    # Convertir les données entrantes en dataframe
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])

    # Faire une prédiction
    prediction = model.predict(df)
    
    # Retourner la prédiction sous forme de réponse JSON
    return {"prediction": prediction.tolist()}

# Lancer l'API avec Uvicorn
# Tu dois exécuter dans ton terminal:
# uvicorn api:app --reload
