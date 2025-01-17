# Projet MLOPS - Prédiction de Prix Immobilier

## Introduction
 Ce projet avait pour objectif de construire une solution de bout en bout pour la prédiction des prix immobiliers en utilisant des techniques avancées de machine learning, ainsi que des outils de déploiement et de surveillance des modèles.
 Il utilise le dataset California Housing pour prédire les prix des maisons.
## Fichiers Inclus

### 1. **01_exploration_et_preparation.ipynb**
Ce notebook contient l'analyse exploratoire des données ainsi que leur préparation.
Les principales étapes effectuées :
- Nettoyage des données.
- Identification des variables importantes.
- Visualisations pour mieux comprendre les relations entre les variables.

### 2. **SHAP.ipynb**
L'objectif de ce notebook est d'expliquer les prédictions du modèle en utilisant SHAP qui permet de :
- Comprendre l'impact de chaque variable sur les prédictions.
- Renforcer la transparence du modèle choisi.

### 3. **api.py**
Ce fichier contient l'implémentation de l'API via FastAPI pour gérer les prédictions. Fonctionnalités clés :
- Endpoint `/predict` permettant d'envoyer des paramètres et de recevoir une estimation.
- Chargement du meilleur modèle directement depuis MLflow.

### 4. **detect_data_drift.py**
Un script essentiel pour surveiller les performances du modèle en production. Il :
- Compare les données de production avec celles d'entraînement.
- Détecte les "drifts" de données et génère un rapport HTML.

### 5. **model_classes.py**
Ce fichier est au cœur du pipeline. Il contient :
- Les classes pour entraîner et évaluer les modèles.
- L'optimisation des hyperparamètres via la recherche bayésienne.
- L'enregistrement des expériences dans MLflow pour le suivi.

### 6. **streamlit_app.py**
L'interface conviviale pour interagir avec le modèle. L'application permet :
- De saisir des données en entrée (revenu, âge des maisons, etc.).
- De recevoir les prédictions directement via l'API.

### 7. **Dockerfile**
Un fichier Docker pour faciliter le déploiement. Il contient les instructions pour créer une image exécutable de l'application.

### 8. **requirements.txt**
Liste des dépendances Python nécessaires, comme :
- `fastapi` et `uvicorn` pour l'API.
- `scikit-learn` et `mlflow` pour les modèles et le suivi.
- `evidently` pour l'analyse des drifts.

### 9. **data_drift_table_report.html**
Le rapport généré automatiquement par `evidently` pour analyser les variations dans les données de production. Ce fichier est essentiel pour comprendre les impacts sur les performances du modèle.

## Modèle Choisi : Random Forest
Après avoir comparé plusieurs modèles, notamment Random Forest, Gradient Boosting et la Régression linéaire, le choix s'est porté sur Random Forest pour les raisons suivantes :

### Performances des Modèles :
- **Random Forest :**
  - RMSE : 0.51
  - MAE : 0.33
  - R²: 0.80
- **Gradient Boosting :**
  - RMSE : 0.54
  - MAE : 0.37
  - R² : 0.78
- **Régression Linéaire :**
  - RMSE : 0.75
  - MAE : 0.53
  - R² : 0.58

### Raisons du Choix :
1. **Précision Supérieure :** Random Forest présente un RMSE et un MAE inférieurs, indiquant une meilleure précision dans les prédictions.
2. **Stabilité :** Random Forest est moins sensible aux outliers grâce à l'agrégation d'arbres de décision.
3. **Interprétabilité :** Bien qu'il soit complexe, Random Forest offre des outils pour analyser l'importance des variables.
4. **Robustesse :** Contrairement à Gradient Boosting, Random Forest est moins sujet au surapprentissage, surtout avec des données bruitées.

Le modèle Random Forest a été enregistré dans MLflow pour un suivi rigoureux et un déploiement facile.

## Rapport Evidently : data_drift_table_report.html
Le rapport de dérive des données révèle que 100% des colonnes présentent un drift entre les données d’entraînement et de production. Les colonnes "AveBedrms" et "Population" montrent des dérives significatives, indiquant des changements dans les caractéristiques des logements et des migrations démographiques. La "MedInc" a également subi une dérive modérée. Le modèle risque d'être impacté par ces dérives, surtout pour les variables importantes. Il est recommandé de surveiller les performances du modèle, d’envisager un réentraînement si nécessaire et d’implémenter un suivi continu pour anticiper les futurs problèmes.

### Des solutions de réentraînement en Cas de Drift

1. **Réentraînement complet avec données mixtes** :
   - Cette approche consiste à combiner 70 % des nouvelles données de production avec 30 % des anciennes données d’entraînement. Le modèle est ensuite entièrement réentraîné sur cet ensemble combiné. Cette méthode permet de capturer à la fois les tendances anciennes et nouvelles, assurant que le modèle reste pertinent face aux évolutions des données. Cependant, elle nécessite des ressources importantes en termes de temps et de calcul. Elle est recommandée lorsque les ressources sont disponibles et que le drift est significatif à travers l'ensemble des données.

2. **Fine-tuning du modèle actuel** :
   - Le fine-tuning consiste à ajuster uniquement les poids du modèle existant en utilisant les nouvelles données. Ce processus est plus rapide et moins coûteux que le réentraînement complet, car il préserve les connaissances acquises sur les données historiques tout en adaptant le modèle aux nouvelles tendances. Cette approche est idéale lorsque le modèle reste globalement performant mais montre des dégradations sur certaines parties des données.

3. **Détection et pondération des données représentatives** :
   - Cette méthode consiste à identifier les segments de données où le drift est le plus marqué, puis à appliquer un réentraînement en pondérant ces segments de manière prioritaire. Cela permet au modèle de se concentrer sur les zones problématiques tout en économisant des ressources, ce qui est particulièrement utile lorsque le drift est localisé, comme sur certaines colonnes (par exemple, "Population" ou "MedInc").

## Pipeline CI/CD

Le pipeline CI/CD assure la qualité et l’automatisation du déploiement avec deux jobs principaux, **test** et **docker-build-and-push**, chacun comprenant des étapes spécifiques :

### 1. Job : Test

- **Étape 1 :** Récupération du code source avec `actions/checkout`.
- **Étape 2 :** Configuration de Python (version 3.11) et installation des dépendances avec `requirements.txt`.
- **Étape 3 :** Démarrage du serveur MLflow localement via SQLite.
- **Étape 4 :** Exécution du script `main.py` pour entraîner et enregistrer le modèle.
- **Étape 5 :** Démarrage de l’API FastAPI sur le port 8000.
- **Étape 6 :** Exécution des tests unitaires pour :
  - Le modèle (`tests/test_model.py`).
  - L’API (`tests/test_api.py`).

### 2. Job : Docker Build and Push

- **Étape 1 :** Authentification sur DockerHub avec les secrets (`DOCKER_USERNAME` et `DOCKER_PASSWORD`).
- **Étape 2 :** Création de l’image Docker pour l’application.
- **Étape 3 :** Publication de l’image Docker sur DockerHub avec le tag `latest`.

Ce pipeline est déclenché automatiquement sur chaque `push` ou `pull request` vers la branche `main` et garantit une intégration fluide et un déploiement automatisé.

## Instructions d’Utilisation
### Prérequis
- Python 3.9+
- Installer les dépendances :
  ```bash
  pip install -r requirements.txt
  ```

### Lancer l’API
1. Démarrer l'interface de MLflow :
```bash
mflow ui
```
2. Lancer l'API :
```bash
uvicorn api:app --reload
```
Endpoint : `http://127.0.0.1:8000/predict`  
Documentation : Ouvrir `http://127.0.0.1:8000/docs`

### Lancer l’Application Streamlit
1. Démarrer l'interface de MLflow :
```bash
mlflow ui
```
2. Lancer l'API :
```bash
uvicorn api:app --reload
```
3. Démarrer l'application Streamlit :
```bash
streamlit run streamlit_app.py
``` 

### Utilisation de Docker
1. Lancer mlflow :
```bash
mlflow ui
```
2. Télécharger l'image Docker :
```bash
docker pull mohamedrachadi/ml-api:latest  
```
3. Lancer le conteneur Docker :
```bash
docker run -p 8000:8000 mohamedrachadi/ml-api:latest
```
4. Démarrer l'application Streamlit :
```bash
streamlit run streamlit_app.py
```


