# 🎮 Vapor Market Analyzer v2.0

Une solution d'Intelligence Artificielle hybride pour prédire le succès, le prix et la hype des jeux vidéo sur le marché "Vapor". Ce projet a été réalisé dans le cadre du module Deep Learning (ING4).

## 🚀 Concept
Vapor Oracle utilise une approche **multi-framework** pour analyser les tendances du marché Steam. Au lieu d'utiliser un modèle unique, nous avons benchmarké 9 modèles différents pour sélectionner les meilleurs champions pour chaque tâche spécifique.

## 📊 Performances du Benchmark
D'après nos derniers tests (Dataset : 450k lignes), voici les modèles sélectionnés pour l'Oracle final :

| Tâche | Framework Champion | Score (Métrique) |
| :--- | :--- | :--- |
| **Prédiction du Prix** | Sklearn (MLP) | **MAE: 5.19** |
| **Volume de Hype** | TensorFlow | **MAE: 0.39** |
| **Succès Local** | Sklearn | **Acc: 0.64** |

## 🛠️ Architecture Hybride
- **Data Management** : Nettoyage avancé, filtrage du bruit et Multi-Hot Encoding pour la gestion multi-genres.
- **Inférence** : Système d'aide à la décision (SAD) intégrant une couche de cohérence pour éviter les prédictions irréalistes (ex: prix déconnecté du genre).
- **Technologies** : Scikit-Learn, TensorFlow, PyTorch, Pandas.

## 📦 Installation & Utilisation

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt

2. **Entraîner les modèles (Benchmark)** :
    ```bash
    python main.py

3. **Lancer l'Oracle (Aide à la décision)** :
    ```bash
    python oracle.py

## 🧠 L'Équipe de Développement

- Mathis Marsault
- Calixte Fouqué
- Axel Bonneau