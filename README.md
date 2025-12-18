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

## 📁 Données (Dataset)

Le dossier `data/` n'est pas inclus dans ce dépôt car les fichiers sont trop volumineux. Pour faire fonctionner le projet, vous devez télécharger les données sources manuellement.

**Procédure :**
1. Téléchargez le dataset depuis Kaggle : **[(https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox/data)]**
2. Créez un dossier nommé `data` à la racine du projet.
3. Extrayez le contenu téléchargé (seulement les fichiers du dossier `steam` si présent) et placez-les dans `data/`.

**Fichiers requis dans `data/` :**
- `games.csv`
- `prices.csv`
- `players.csv`
- `reviews.csv`

> **Note :** Assurez-vous que les noms des fichiers correspondent exactement à la liste ci-dessus pour que le script `data_manager.py` les trouve.

## 🧠 L'Équipe de Développement

- Mathis Marsault
- Calixte Fouqué
- Axel Bonneau
