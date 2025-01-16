import streamlit as st
import requests

# URL de ton API FastAPI, assure-toi qu'elle soit en marche (par exemple, localhost si tu testes localement)
#API_URL = "http://127.0.0.1:8000/predict"
API_URL = "http://api-container:8000/predict"

# Fonction pour obtenir la prédiction via l'API
def get_prediction(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Erreur lors de la récupération de la prédiction"}

# Interface utilisateur Streamlit
st.title("Prédiction de prix immobilier")

# Saisie des paramètres par l'utilisateur
MedInc = st.number_input("Revenu médian", value=0.0)
HouseAge = st.number_input("Âge de la maison", value=0.0)
AveRooms = st.number_input("Nombre moyen de chambres", value=0.0)
AveBedrms = st.number_input("Nombre moyen de chambres à coucher", value=0.0)
Population = st.number_input("Population", value=0.0)
AveOccup = st.number_input("Occupants moyens par logement", value=0.0)
Latitude = st.number_input("Latitude", value=0.0)
Longitude = st.number_input("Longitude", value=-0.0)

# Créer un dictionnaire avec les données de l'utilisateur
data = {
    "MedInc": MedInc,
    "HouseAge": HouseAge,
    "AveRooms": AveRooms,
    "AveBedrms": AveBedrms,
    "Population": Population,
    "AveOccup": AveOccup,
    "Latitude": Latitude,
    "Longitude": Longitude
}

# Bouton pour faire la prédiction
if st.button("Faire la prédiction"):
    result = get_prediction(data)
    
    if "prediction" in result:
        st.write(f"Le prix de la maison estimé est de {result['prediction']}")
    else:
        st.write(f"Erreur: {result['error']}")
