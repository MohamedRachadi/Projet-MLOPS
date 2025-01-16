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
Ce fichier est au cœur du pipeline de machine learning. Il contient :
- Les classes pour entraîner et évaluer les modèles.
- L'optimisation des hyperparamètres via la recherche bayésienne.
- L'enregistrement des expériences dans MLflow pour le suivi.

### 6. **streamlit_app.py**
Une interface conviviale pour interagir avec le modèle. L'application permet :
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
Un rapport généré automatiquement par `evidently` pour analyser les variations dans les données de production. Ce fichier est essentiel pour comprendre les impacts sur les performances du modèle.

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
Ce rapport est un outil essentiel pour surveiller les changements dans les données de production. Il fournit :
- **Un score de drift global** pour évaluer la gravité des changements.
- **Des visualisations par variable** pour identifier celles qui ont le plus changé.
- **Des recommandations** pour savoir si un réentraînement est nécessaire.

## Des solutions de réentraînement en Cas de Drift

1. **Surveillance continue du drift** :
   - Intégrer une surveillance automatique des rapports HTML générés par `Evidently`.
   - Configurer des seuils d'alerte sur le score de drift global ou par variable (comme dans le rapport `data_drift_table_report.html`).

2. **Collecte régulière de données** :
   - Collecter et stocker de nouvelles données de production pour enrichir le jeu de données d’entraînement.
   - S'assurer que ces nouvelles données reflètent les changements dans la distribution des variables.

3. **Réentraîner le modèle** :
   - Réentraîner le modèle en utilisant les nouvelles données collectées.
   - Comparer les performances du modèle réentraîné avec le modèle actuel avant de le déployer.

4. **Enrichissement du modèle** :
   - Ajouter de nouvelles variables pertinentes si nécessaire, pour capturer les nouvelles tendances détectées dans les données.

5. **Test A/B** :
   - Mettre en place un test A/B pour valider les performances du nouveau modèle sur une partie de la production avant un déploiement complet.

6. **Fine-tuning du modèle existant** :
   - Si le modèle choisi est un algorithme comme Random Forest, effectuer un ajustement léger en réutilisant les arbres existants avec des données actualisées.
   - Cette méthode est plus rapide et moins coûteuse qu’un réentraînement complet.

7. **Validation croisée rigoureuse** :
   - Utiliser une validation croisée avec des métriques comme RMSE, MAE et R² pour évaluer les performances des modèles mis à jour.

8. **Automatisation avec des pipelines MLOps** :
   - Automatiser le processus de détection de drift, de collecte de données et de réentraînement via des outils comme MLflow ou Kubeflow.
   - Par exemple, en intégrant un workflow où le rapport Evidently déclenche automatiquement le réentraînement.

9. **Mise à jour itérative** :
   - Mettre en place des mises à jour régulières du modèle (par exemple, tous les mois ou trimestres) pour limiter l’accumulation de drift.

## Instructions d’Utilisation
### Prérequis
- Python 3.9+
- Installer les dépendances :
  ```bash
  pip install -r requirements.txt
  ```

### Lancer l’API
```bash
uvicorn api:app --reload
```
Endpoint : `http://127.0.0.1:8000/predict`

### Lancer l’Application Streamlit
```bash
streamlit run streamlit_app.py
```

### Utilisation de Docker
1. Construire l'image :
   ```bash
   docker build -t prix-immobilier-app .
   ```
2. Lancer le conteneur :
   ```bash
   docker run -p 8000:8000 prix-immobilier-app
   ```


