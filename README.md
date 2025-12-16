# 🎮 DeepLearning Playtime Prediction

Ce dépôt documente un projet de Deep Learning visant à analyser et prédire le temps de jeu des utilisateurs. Il retrace une démarche complète de Data Science, incluant la critique des données, la détection de biais, et la réorientation stratégique vers des sources fiables.

## 📝 Contexte et Objectifs

L'objectif initial était de construire un modèle prédictif capable d'estimer l'engagement d'un joueur (temps de jeu hebdomadaire) en fonction de son profil démographique (âge, localisation, type de jeu favori), en utilisant un réseau de neurones artificiels (MLP).

## 🔄 Pivot Stratégique et Historique

### 1. Première Analyse (Branche `old`)
Lors de la première phase du projet, nous avons entraîné un Perceptron Multicouche sur un premier jeu de données.
* **Constat :** L'Analyse Exploratoire des Données (EDA) et les résultats du modèle n'ont montré **aucune corrélation** significative. La distribution des données semblait aléatoire et incohérente.
* **Conclusion :** Nous avons déduit que ce premier dataset était constitué de **données artificielles (synthétiques)** mal générées, rendant toute prédiction impossible.

> 📂 **Accès à l'archive :**
> Par souci de transparence scientifique, cette première analyse a été conservée.
> Vous pouvez retrouver le code et les conclusions de cette étape dans la branche **`old`**.
>
> ```bash
> git checkout old
> ```

### 2. Nouvelle Orientation (Branche `main`)
Face à ce constat, nous avons décidé de basculer sur des **données réelles** pour garantir la pertinence de nos modèles de Deep Learning.

Nous utilisons désormais le dataset **"Gaming Profiles 2025"**, regroupant des données authentiques de plateformes majeures.

* **Source :** [Kaggle - Gaming Profiles 2025 (Steam, PlayStation, Xbox)](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox?resource=download&select=steam)
* **Objectif actuel :** Appliquer notre architecture MLP sur ces comportements réels pour extraire de vrais patterns d'engagement.

## 🧠 Architecture Technique

Le cœur du projet repose sur l'utilisation de réseaux de neurones profonds :
* **Modèle :** Perceptron Multicouche (MLP - Multi-Layer Perceptron).
* **Technique :** Couches denses (Dense Layers), fonction d'activation **ReLU** et rétropropagation du gradient.
* **Stack :** Python 🐍, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook.

## 🚀 Installation et Utilisation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/VOTRE-NOM-UTILISATEUR/DeepLearning-Playtime-Prediction.git](https://github.com/VOTRE-NOM-UTILISATEUR/DeepLearning-Playtime-Prediction.git)
    ```

2.  **Installer les dépendances :**
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install pandas numpy matplotlib seaborn jupyter
    ```

3.  **Lancer l'analyse :**
    ```bash
    jupyter notebook Final.ipynb
    ```

## 👤 Auteur

Projet réalisé dans le cadre du cours de Deep Learning en spécialité IA.
