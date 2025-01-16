import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DataDriftTable


# Générer des données simulées en production avec des variations plus réalistes
def generate_synthetic_data_varying(n_samples=500):
    np.random.seed(42)

    synthetic_data = pd.DataFrame({
        "MedInc": np.random.uniform(1, 15, n_samples) * np.random.uniform(0.9, 1.1),  # Décalage léger
        "HouseAge": np.random.randint(1, 60, n_samples) + np.random.choice([-5, 5, 0], n_samples, p=[0.1, 0.1, 0.8]),  # Décalage aléatoire
        "AveRooms": np.random.uniform(1, 10, n_samples) * np.random.uniform(0.8, 1.2),  # Légère augmentation ou réduction
        "AveBedrms": np.random.uniform(1, 5, n_samples) + np.random.normal(0, 0.5, n_samples),  # Ajout d'un bruit gaussien
        "Population": np.random.randint(100, 10000, n_samples) + np.random.randint(-500, 500, n_samples),  # Décalage
        "AveOccup": np.random.uniform(1, 10, n_samples) * np.random.uniform(0.95, 1.1),  # Variation aléatoire
        "Latitude": np.random.uniform(32, 42, n_samples) + np.random.normal(0, 0.1, n_samples),  # Bruit autour de la latitude
        "Longitude": np.random.uniform(-124, -114, n_samples) + np.random.normal(0, 0.1, n_samples),  # Bruit autour de la longitude
        "MedHouseVal": np.random.uniform(0.5, 5, n_samples) * np.random.uniform(0.9, 1.15),  # Déviation légère
    })

    return synthetic_data


# Générer les données de production
production_data = generate_synthetic_data_varying(n_samples=1000)

# Charger les données d'entraînement
train_data = pd.read_csv("../data/X_train.csv")

# Créer un rapport Evidently avec DataDriftTable
data_drift_report = Report(metrics=[DataDriftTable()])
data_drift_report.run(reference_data=train_data, current_data=production_data)

# Sauvegarder un rapport HTML
data_drift_report.save_html("data_drift_table_report.html")

print("Rapport de data drift généré : data_drift_table_report.html")