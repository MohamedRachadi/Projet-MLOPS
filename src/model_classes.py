import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from dataclasses import dataclass


# Cette classe gère l'entraînement des modèles et le calcul des métriques de performance.
@dataclass
class ModelTrainer:
    """
    Classe pour entraîner des modèles et calculer leurs métriques de performance.
    """
    @staticmethod
    def train_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, rmse, mae, r2


# Cette classe configure et gère le suivi des expériences dans MLflow.
@dataclass
class MLFlowLogger:
    """
    Classe pour gérer le suivi des expériences dans MLflow.
    """
    experiment_name: str

    def __post_init__(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.experiment_name)

    def log_experiment(self, model_name, model, X_train, y_train, X_test, y_test, hyperparameters=None):
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            if hyperparameters:
                mlflow.log_params(hyperparameters)

            trainer = ModelTrainer()
            trained_model, rmse, mae, r2 = trainer.train_model(model, X_train, y_train, X_test, y_test)

            # Log des métriques
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Exemple d'entrée
            input_example = pd.DataFrame({
                "MedInc": [1.0], "HouseAge": [15.0], "AveRooms": [6.0],
                "AveBedrms": [2.0], "Population": [300.0], "AveOccup": [4.0],
                "Latitude": [37.0], "Longitude": [-122.0]
            })

            # Log du modèle
            mlflow.sklearn.log_model(trained_model, "model", input_example=input_example)

            return trained_model, rmse, mae, r2


# Cette classe s'occupe d'optimiser les hyperparamètres des modèles en utilisant une recherche bayésienne.
@dataclass
class ModelOptimizer:
    """
    Classe pour optimiser les hyperparamètres des modèles avec recherche bayésienne.
    """
    @staticmethod
    def optimize_model(model, param_space, X_train, y_train):
        opt = BayesSearchCV(
            model,
            param_space,
            n_iter=20,
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        opt.fit(X_train, y_train)
        return opt.best_estimator_, opt.best_params_, -opt.best_score_


# Cette classe compare différents modèles, les optimise, et suit leurs résultats avec MLflow
@dataclass
class ModelComparator:
    """
    Classe pour comparer différents modèles, les optimiser, et suivre les résultats.
    """
    logger: MLFlowLogger

    def compare_models(self, X_train, y_train, X_test, y_test):
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }

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
        best_model_params = {}
        final_best_model = None  # Initialize it here

        for model_name, model in models.items():
            print(f"\nOptimizing {model_name}...")

            optimizer = ModelOptimizer()
            best_model, best_params, best_score = optimizer.optimize_model(model, param_spaces[model_name], X_train, y_train)

            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best score (RMSE) for {model_name}: {best_score}")

            self.logger.log_experiment(model_name, best_model, X_train, y_train, X_test, y_test, hyperparameters=best_params)

            if best_score < best_rmse:
                best_rmse = best_score
                best_model_name = model_name
                final_best_model = best_model  # Update the final_best_model
                best_model_params = best_params  # Update the best_model_params

        print(f"Best model: {best_model_name} with RMSE: {best_rmse} and params: {best_model_params}")
        return final_best_model, best_model_params


# Cette classe permet d'enregistrer le meilleur modèle dans le registre MLflow.
@dataclass
class BestModelRegistry:
    """
    Classe pour enregistrer le meilleur modèle dans le registre MLflow.
    """
    @staticmethod
    def register_best_model(best_model, best_params):
        model_name = f"Best_Model_{best_model.__class__.__name__}"
        input_example = pd.DataFrame({
            "MedInc": [1.0], "HouseAge": [15.0], "AveRooms": [6.0],
            "AveBedrms": [2.0], "Population": [300.0], "AveOccup": [4.0],
            "Latitude": [37.0], "Longitude": [-122.0]
        })

        with mlflow.start_run(run_name=f"Best Model Registration: {model_name}"):
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name=model_name,
                input_example=input_example
            )
            mlflow.log_params(best_params)
            print(f"Le meilleur modèle a été enregistré avec le nom '{model_name}' et ses hyperparamètres.")






