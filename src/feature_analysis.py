import shap
import mlflow
import mlflow.sklearn
import pandas as pd

# Charger le modèle depuis MLflow Model Registry
mlflow.set_tracking_uri("http://localhost:5000")
model_name = "Best_Model_RandomForestRegressor" 
model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

# Charger les données d'entraînement pour calculer les SHAP values
data_path = "../data/Std data"
X_train = pd.read_csv(f"{data_path}/X_train.csv")

# Initialiser le SHAP Explainer pour un modèle RandomForest
explainer = shap.TreeExplainer(model)

# Calculer les valeurs SHAP pour l'ensemble de données d'entraînement
shap_values = explainer.shap_values(X_train)

# Résumé des valeurs SHAP pour l'ensemble des features
shap.summary_plot(shap_values, X_train)

# Visualisation de l'impact des features pour la première observation de X_train
shap.initjs()  # Pour initialiser les visualisations interactives
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train.iloc[0])

# Visualisation de la dépendance de la feature 'MedInc' avec les valeurs SHAP
shap.dependence_plot("MedInc", shap_values[1], X_train)
