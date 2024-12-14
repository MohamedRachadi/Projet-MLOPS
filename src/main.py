import pandas as pd
from sklearn.model_selection import train_test_split
from model_training import compare_models
import mlflow
import mlflow.sklearn

# Charger les données
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')

# Comparer les modèles et obtenir le meilleur modèle
best_model = compare_models(X_train, y_train, X_test, y_test)

# Enregistrer le meilleur modèle dans le Model Registry
from model_training import register_best_model
register_best_model(best_model)

# Afficher que l'enregistrement est fait
print("Le meilleur modèle a été enregistré dans MLflow.")
