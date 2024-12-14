import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib



#Crée une fonction générique qui pourra entraîner plusieurs types de modèles
def train_model(model, X_train, y_train, X_test, y_test):
    """
    Fonction pour entraîner un modèle et calculer les métriques de performance.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, mae, r2


#Mettre en place MLflow pour suivre les expériences
def log_experiment(model_name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)

        trained_model, rmse, mae, r2 = train_model(model, X_train, y_train, X_test, y_test)

        # Log des métriques
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(trained_model, "model")
        
        return trained_model, rmse, mae, r2


#Tester différents modèles
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_rmse = float('inf')

    # Comparer les modèles
    for model_name, model in models.items():
        trained_model, rmse, mae, r2 = log_experiment(model_name, model, X_train, y_train, X_test, y_test)

        print(f"{model_name}: RMSE={rmse}, MAE={mae}, R²={r2}")

        # Garder le meilleur modèle basé sur RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = trained_model

    # Retourner le meilleur modèle
    return best_model


#Enregistrer le meilleur modèle dans le Model Registry
def register_best_model(best_model):
    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "best_model")
        print("Le meilleur modèle a été enregistré dans le Model Registry.")
