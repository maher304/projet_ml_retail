# Projet ML Retail

## 📌 Description
Projet de Machine Learning appliqué au secteur retail.
Objectif : Prétraiter les données, entraîner un modèle de prédiction et le déployer via une application Flask.

---

## ⚙️ Installation



### 1. Créer l'environnement virtuel
python -m venv venv

### 2. Activer l’environnement
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 4. Installer les dépendances
pip install -r requirements.txt

---

## 📂 Structure du projet

- data/ : données brutes et traitées
- notebooks/ : exploration et prototypage
- src/ : scripts Python production
- models/ : modèles sauvegardés
- app/ : application Flask
- reports/ : visualisations et rapports

---

## ▶️ Utilisation

### Prétraitement
python src/preprocessing.py

### Entraînement du modèle
python src/train_model.py

### Prédiction
python src/predict.py

### Lancer l’application Flask
python app/app.py
