import mlflow
from model_classes import ModelTrainer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# -------------------------------------------------------------------------------------------- #

data_path = "../data/Train-test data"
X_train = pd.read_csv(f"{data_path}/X_train.csv")
X_test = pd.read_csv(f"{data_path}/X_test.csv")
y_train = pd.read_csv(f"{data_path}/y_train.csv").squeeze()  
y_test = pd.read_csv(f"{data_path}/y_test.csv").squeeze()
print("Les données sont chargées.")

# -------------------------------------------------------------------------------------------- #

# Initialiser le modèle RandomForest avec les meilleurs paramètres
best_rf_model = RandomForestRegressor(max_depth=10, min_samples_leaf=4, n_estimators=150, random_state=42)

# Entraîner le modèle
trainer = ModelTrainer()
trained_rf_model, rmse, mae, r2 = trainer.train_model(best_rf_model, X_train, y_train, X_test, y_test)

# Enregistrer le modèle dans MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name="California Housing Project")
with mlflow.start_run(run_name="RandomForest_Final_Model"):
    mlflow.sklearn.log_model(trained_rf_model, "random_forest_model")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("max_depth", best_rf_model.max_depth)
    mlflow.log_param("min_samples_leaf", best_rf_model.min_samples_leaf)
    mlflow.log_param("n_estimators", best_rf_model.n_estimators)
    mlflow.log_param("random_state", best_rf_model.random_state)

print(f"RandomForest model trained and logged with RMSE: {rmse}, MAE: {mae}, R2: {r2}")

input_example = pd.DataFrame({
    "MedInc": [1.0], "HouseAge": [15.0], "AveRooms": [6.0],
    "AveBedrms": [2.0], "Population": [300.0], "AveOccup": [4.0],
    "Latitude": [37.0], "Longitude": [-122.0]
})

input_prediction = trained_rf_model.predict(input_example)
print(f"Prediction for input_example: {input_prediction}")