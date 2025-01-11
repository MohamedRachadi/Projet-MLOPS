import pandas as pd
from model_classes import ModelComparator, MLFlowLogger, BestModelRegistry

# -------------------------------------------------------------------------------------------- #

data_path = "../data/Std data"
X_train = pd.read_csv(f"{data_path}/X_train.csv")
X_test = pd.read_csv(f"{data_path}/X_test.csv")
y_train = pd.read_csv(f"{data_path}/y_train.csv").squeeze()  
y_test = pd.read_csv(f"{data_path}/y_test.csv").squeeze()
print("Les données sont chargées.")

# -------------------------------------------------------------------------------------------- #

logger = MLFlowLogger(experiment_name="California Housing Project")
comparator = ModelComparator(logger=logger)
print("Les classes sont initialisées.")

# -------------------------------------------------------------------------------------------- #

best_model, best_params = comparator.compare_models(X_train, y_train, X_test, y_test)
print("Comparaison des modèles terminée.")

# -------------------------------------------------------------------------------------------- #

registry = BestModelRegistry()
registry.register_best_model(best_model, best_params)
print("Le meilleur modèle a été enregistré avec succès.")


