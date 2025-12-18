import os
import joblib
import numpy as np
import tensorflow as tf

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input(prompt, min_val=None, max_val=None, is_float=False):
    """Garantit une saisie utilisateur propre et sécurisée."""
    while True:
        try:
            val = float(input(prompt)) if is_float else int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"⚠️ Valeur trop basse (Min: {min_val})")
                continue
            if max_val is not None and val > max_val:
                print(f"⚠️ Valeur trop haute (Max: {max_val})")
                continue
            return val
        except ValueError:
            print("❌ Entrée invalide. Veuillez saisir un nombre.")

def run_oracle():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models_saved')

    # 1. Chargement des composants
    try:
        sc_reg = joblib.load(os.path.join(model_dir, 'scaler_reg.pkl'))
        sc_cls = joblib.load(os.path.join(model_dir, 'scaler_class.pkl'))
        le_pop = joblib.load(os.path.join(model_dir, 'le_pop.pkl'))
        le_country = joblib.load(os.path.join(model_dir, 'le_country.pkl'))
        
        # Chargement hybride (Sklearn pour prix/pop, TF pour Hype)
        m_prix = joblib.load(os.path.join(model_dir, 'best_prix.pkl'))
        m_hype = tf.keras.models.load_model(os.path.join(model_dir, 'best_hype.keras'), compile=False)
        m_pop = joblib.load(os.path.join(model_dir, 'best_pop.pkl'))
    except Exception as e:
        print(f"❌ Erreur de chargement des modèles : {e}")
        return

    clear_screen()
    print("="*60)
    print("          VAPOR ORACLE v2.0 : SYSTÈME D'AIDE À LA DÉCISION")
    print("="*60)

    # 2. Saisie utilisateur assistée
    print("\n[1] PARAMÈTRES DE LANCEMENT")
    mois = get_input(" > Mois de sortie (1-12) : ", 1, 12)
    annee = get_input(" > Année de sortie (ex: 2026) : ", 2024, 2030)
    langues = get_input(" > Nombre de langues : ", 1, 50)
    prix_vise = get_input(" > Prix visé ($) : ", 0, 999, is_float=True)

    print("\n[2] PROFIL DU JEU (GENRES)")
    genres_list = ['Action', 'Adventure', 'Indie', 'RPG', 'Strategy', 'Simulation', 'Casual']
    for i, g in enumerate(genres_list):
        print(f"  {i+1}. {g}")
    print("\nEntrez les numéros des genres séparés par des virgules (ex: 1,4) :")
    choix = input(" > Choix : ")
    genre_vec = [0] * len(genres_list)
    indices = [int(i.strip()) - 1 for i in choix.split(',') if i.strip().isdigit()]
    for idx in indices:
        if 0 <= idx < len(genres_list): genre_vec[idx] = 1

    print("\n[3] ANALYSE GÉOGRAPHIQUE")
    while True:
        pays = input(" > Pays cible (ex: Japan, France, United States) : ").title()
        try:
            c_idx = le_country.transform([pays])[0]
            break
        except:
            print(f"⚠️ Pays '{pays}' inconnu du dataset. Réessayez.")

    # 3. Prétraitement et Inférence
    msin, mcos = np.sin(2*np.pi*mois/12), np.cos(2*np.pi*mois/12)
    X_reg = sc_reg.transform([[msin, mcos, annee, langues] + genre_vec])
    X_cls = sc_cls.transform([[c_idx, msin, mcos, annee, prix_vise, langues] + genre_vec])

    # Prédictions brutes
    p_prix = max(0.99, m_prix.predict(X_reg)[0])
    p_hype = max(0, np.expm1(m_hype.predict(X_reg, verbose=0)[0][0]))
    p_pop_idx = m_pop.predict(X_cls)[0]
    p_pop_name = le_pop.inverse_transform([p_pop_idx])[0]

    # 4. COUCHE DE COHÉRENCE (Intelligence Métier)
    # On pénalise le succès si le prix utilisateur est > 50% du prix conseillé par l'IA
    degre_incoherence = 0
    if prix_vise > p_prix * 1.5:
        degre_incoherence = 1 # Warning modéré
        if p_pop_name == "High": p_pop_name = "Medium"
        elif p_pop_name == "Medium": p_pop_name = "Low"
        
        if prix_vise > p_prix * 2.5:
            degre_incoherence = 2 # Warning critique
            p_pop_name = "Low"

    # 5. Affichage final épuré
    clear_screen()
    print("="*60)
    print("                  RAPPORT D'EXPERTISE VAPOR")
    print("="*60)
    print(f" CONFIGURATION : {mois}/{annee} | {langues} Langues | Marché {pays}")
    print(f" GENRES        : {', '.join([genres_list[i] for i, v in enumerate(genre_vec) if v == 1])}")
    print("-" * 60)
    
    print(f"\n📈 ANALYSE DES MODÈLES :")
    print(f" > PRIX CONSEILLÉ  : {p_prix:.2f} $")
    print(f" > HYPE ESTIMÉE    : {p_hype:.0f} avis")
    
    color = "🟢" if p_pop_name == "High" else "🟡" if p_pop_name == "Medium" else "🔴"
    print(f" > POTENTIEL LOCAL : {color} {p_pop_name.upper()}")
    
    print("\n" + "-"*60)
    print("📌 CONSEIL STRATÉGIQUE :")
    if degre_incoherence == 2:
        print("❌ REJET : Prix totalement incohérent. Succès impossible à ce tarif.")
    elif degre_incoherence == 1:
        print("⚠️ RISQUE : Prix trop élevé. Potentiel de succès dégradé par l'IA.")
    elif p_pop_name == "High":
        print("🚀 OPTIMAL : Excellent positionnement. Feu vert pour le lancement.")
    else:
        print("ℹ️ NEUTRE : Configuration standard. Volume de vente modéré.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_oracle()