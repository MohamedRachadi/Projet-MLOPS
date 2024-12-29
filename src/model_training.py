import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Crée une fonction générique qui pourra entraîner plusieurs types de modèles
def train_model(model, X_train, y_train, X_test, y_test):
    """
    Fonction pour entraîner un modèle et calculer les métriques de performance.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, mae, r2


#Mettre en place MLflow pour suivre les expériences
def log_experiment(model_name, model, X_train, y_train, X_test, y_test, hyperparameters=None):
    mlflow.set_experiment("California Housing Project")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        if hyperparameters:
            mlflow.log_params(hyperparameters)

        trained_model, rmse, mae, r2 = train_model(model, X_train, y_train, X_test, y_test)

        # Log des métriques
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        input_example = pd.DataFrame({
            "MedInc": [1.0],
            "HouseAge": [15.0],
            "AveRooms": [6.0],
            "AveBedrms": [2.0],
            "Population": [300.0],
            "AveOccup": [4.0],
            "Latitude": [37.0],
            "Longitude": [-122.0]
        })

        # Log du modèle avec un exemple d'entrée
        mlflow.sklearn.log_model(trained_model, "model", input_example=input_example)
        
        return trained_model, rmse, mae, r2


# Optimisation bayésienne pour chaque modèle
def optimize_model(model, param_space, X_train, y_train):
    opt = BayesSearchCV(
        model, 
        param_space, 
        n_iter=20, 
        scoring='neg_mean_squared_error',  # Utilisation de MSE négatif pour maximiser le score
        cv=3,  # Validation croisée à 3 plis
        n_jobs=-1, 
        random_state=42
    )
    
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.best_params_, -opt.best_score_


#Tester différents modèles
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    # Définir les espaces d'hyperparamètres pour chaque modèle
    param_spaces = {
        "Linear Regression": {
            'fit_intercept': [True, False]
        },
        "Random Forest": {
            'n_estimators': Integer(50, 150),
            'max_depth': Integer(5, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5)
        },
        "Gradient Boosting": {
            'n_estimators': Integer(50, 150),
            'learning_rate': Real(0.001, 0.01, prior='uniform'),
            'max_depth': Integer(3, 8),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5)
        }
    }

    best_model = None
    best_rmse = float('inf')
    best_model_name = ""

    # Comparer les modèles avec optimisation bayésienne
    for model_name, model in models.items():
        print(f"\n--------------------------------------------- Optimizing {model_name}... ---------------------------------------------\n")
        best_model, best_params, best_score = optimize_model(model, param_spaces[model_name], X_train, y_train)

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best score (RMSE) for {model_name}: {best_score}")
        
        # Log des hyperparamètres et métriques
        log_experiment(model_name, best_model, X_train, y_train, X_test, y_test, hyperparameters=best_params)

        # Garder le meilleur modèle basé sur RMSE
        if best_score < best_rmse:
            best_rmse = best_score
            best_model_name = model_name
            final_best_model = best_model

    # Retourner le meilleur modèle
    print(f"Best model: {best_model_name} with RMSE: {best_rmse}")
    print("\n ----------------------------------------------------------------------------- \n")
    return final_best_model

def register_best_model(best_model):
    # Définir un nom spécifique pour le modèle dans le Model Registry
    model_name = f"California_Housing_Best_Model_{best_model.__class__.__name__}"  # Utilisation du nom de classe du modèle

    # Exemple d'entrée sous forme de DataFrame
    input_example = pd.DataFrame({
        "MedInc": [1.0],            # Exemple de valeur pour chaque caractéristique
        "HouseAge": [15.0],
        "AveRooms": [6.0],
        "AveBedrms": [2.0],
        "Population": [300.0],
        "AveOccup": [4.0],
        "Latitude": [37.0],
        "Longitude": [-122.0]
    })

    with mlflow.start_run(run_name=f"Best Model Registration: {model_name}"):
        # Enregistrer le modèle avec l'exemple d'entrée
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="best_model",  # L'endroit où le modèle sera stocké dans le run
            registered_model_name=model_name,  # Nom pour le Model Registry
            input_example=input_example  # L'exemple d'entrée pour déduire la signature du modèle
        )
        print(f"Le meilleur modèle a été enregistré dans le Model Registry avec le nom '{model_name}'.")