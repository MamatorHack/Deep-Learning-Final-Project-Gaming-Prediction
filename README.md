# 🎮 DeepLearning Playtime Prediction

Ce dépôt contient le projet final de Deep Learning visant à prédire le temps de jeu des utilisateurs en fonction de leur profil démographique, en utilisant un réseau de neurones artificiels (MLP).

## 📝 Description du Projet

L'objectif de ce projet est d'analyser les comportements des joueurs et de tenter de prédire leur engagement (temps de jeu hebdomadaire) à partir de données statiques (âge, localisation, type de jeu favori, etc.).

Nous avons suivi une approche complète de Data Science :
1.  **Exploratory Data Analysis (EDA) :** Visualisation de la répartition des données.
2.  **Prétraitement :** Nettoyage et normalisation des données pour le réseau de neurones.
3.  **Modélisation Deep Learning :** Implémentation d'un Perceptron Multicouche (MLP).
4.  **Interprétation :** Analyse des résultats et recommandations stratégiques.

## 🧠 Architecture Technique

Le cœur du projet repose sur un algorithme de Deep Learning :
* **Modèle :** Perceptron Multicouche (MLP - Multi-Layer Perceptron).
* **Architecture :** Couches cachées avec fonctions d'activation **ReLU** pour capturer la non-linéarité.
* **Méthode :** Apprentissage par rétropropagation du gradient.

### Technologies utilisées
* **Langage :** Python 🐍
* **Analyse de données :** Pandas, NumPy
* **Visualisation :** Seaborn, Matplotlib
* **Environnement :** Jupyter Notebook

## 📊 Résultats Clés & Conclusion

L'analyse menée dans ce notebook a permis de mettre en évidence un point crucial concernant le comportement des joueurs :

> 🚫 **Constat :** Il n'existe pas de corrélation prédictive forte entre le profil démographique simple (âge, pays) et le temps de jeu. Un joueur de 25 ans peut jouer 1h comme 50h par semaine, quel que soit son pays.

**Recommandation Stratégique (Cas LethalCompany) :**
Comme démontré dans la conclusion du projet, pour améliorer la prédiction, il est nécessaire de collecter des **données comportementales historiques** (ex: temps de jeu de la semaine précédente) plutôt que de se baser uniquement sur des données statiques.

## 🚀 Comment utiliser ce projet

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/VOTRE-NOM-UTILISATEUR/DeepLearning-Playtime-Prediction.git](https://github.com/VOTRE-NOM-UTILISATEUR/DeepLearning-Playtime-Prediction.git)
    ```
2.  **Installer les dépendances :**
    Assurez-vous d'avoir Python installé, puis installez les librairies nécessaires :
    ```bash
    pip install pandas numpy matplotlib seaborn jupyter
    ```
3.  **Lancer le notebook :**
    ```bash
    jupyter notebook Final.ipynb
    ```

## 👤 Auteur

Projet réalisé par **[VOTRE NOM]** dans le cadre du cours de Deep Learning.
