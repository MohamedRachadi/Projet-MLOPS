# Utiliser une image officielle de Python comme image de base
FROM python:3.11-slim

# Définir un répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /app/

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application dans le conteneur
COPY . /app/

# Exposer le port que l'API va utiliser
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]