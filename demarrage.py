import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import deque
from sklearn.model_selection import GridSearchCV


################################################# --- 1. Préparation données ---

fichiers = ['atp_matches_2024.csv','2025.csv', '2026.csv']
liste_df = []
for f in fichiers:
    try:
        liste_df.append(pd.read_csv(f))
    except FileNotFoundError:
        print(f"Fichier {f} non trouvé, ignoré.")

df_global = pd.concat(liste_df, axis=0, ignore_index=True)
df_global['tourney_date'] = pd.to_datetime(df_global['tourney_date'], format='%Y%m%d') #Transformation du type en date
df_global = df_global.sort_values('tourney_date').reset_index(drop=True)  # Tri pour être sur

col_utiles = ['surface', 'tourney_date', 'winner_id', 'winner_rank', 
              'loser_id', 'loser_rank', 'round', 'tourney_level', 'winner_name', 'loser_name']
df_filtre = df_global[col_utiles].copy()  # On sélectionne que les colonnes utiles (les stats connues après le match ne doit pas être conservé)

rank_par_defaut = 2000
df_filtre['winner_rank'] = df_filtre['winner_rank'].fillna(rank_par_defaut)
df_filtre['loser_rank'] = df_filtre['loser_rank'].fillna(rank_par_defaut)  # Tous les joueurs n'ayant pas de rang sont mis au rang 2000 par défaut

################################################# --- 2. Création dictionnaire stats ---

# On utilise des dictionnaires pour suivre l'évolution des joueurs au fil des matchs
stats_joueurs = {} # Forme par surface et mental
h2h = {}           # Match-ups directs
forme_5 = {}       # Forme générale (5 derniers matchs)

data_expert = []

for index, row in df_filtre.iterrows():
    w_id, l_id = row['winner_id'], row['loser_id']
    surf = str(row['surface']) # On récupère gagnant perdant et surface
    
    # Initialisation si nouveau joueur
    for p_id in [w_id, l_id]:
        if p_id not in stats_joueurs:
            stats_joueurs[p_id] = {'w_Clay': 0, 'l_Clay': 0, 'w_Hard': 0, 'l_hard': 0, 'w_Grass': 0, 'l_Grass': 0, 'w_GS': 0, 'l_GS': 0, 'w_Final': 0, 'l_Final': 0}
            forme_5[p_id] = deque(maxlen=5)

    # --- CALCUL DES RATIOS AVANT LE MATCH ---
    def get_surf_ratio(pid, s):
        w = stats_joueurs[pid].get(f'w_{s}', 0)
        l = stats_joueurs[pid].get(f'l_{s}', 0)
        return w / (w + l) if (w + l) > 0 else 0.5 # On récupère les stats et on en fait un ratio

    def get_mental_ratio(pid):
        w = stats_joueurs[pid]['w_GS'] + stats_joueurs[pid]['w_Final']
        l = stats_joueurs[pid]['l_GS'] + stats_joueurs[pid]['l_Final']
        return w / (w + l) if (w + l) > 0 else 0.5 # On récupère les stats et on en fait un ratio

    # H2H (Match-up)
    h2h_key = tuple(sorted((w_id, l_id)))
    if h2h_key not in h2h: h2h[h2h_key] = {w_id: 0, l_id: 0} # Initialisation du matchup
    
    # Mélange l'ordre des données pour que l'IA ne se dise pas "le premier joueur gagne forcément"
    if np.random.random() > 0.5:
        id_a, id_b, target = w_id, l_id, 1
        rank_a, rank_b = row['winner_rank'], row['loser_rank']
    else:
        id_a, id_b, target = l_id, w_id, 0
        rank_a, rank_b = row['loser_rank'], row['winner_rank']

    # Récupération des stats accumulées AVANT ce match
    total_h2h = h2h[h2h_key][id_a] + h2h[h2h_key][id_b] # h2h (dico des matchup) à la clé h2h_key (correspond à joueur A contre B) à la clé [id_A] (nb victoires de A) respectivement B
    h2h_ratio = h2h[h2h_key][id_a] / total_h2h if total_h2h > 0 else 0.5
    
    f_a = sum(forme_5[id_a])/len(forme_5[id_a]) if forme_5[id_a] else 0.5
    f_b = sum(forme_5[id_b])/len(forme_5[id_b]) if forme_5[id_b] else 0.5

    data_expert.append({
        'id_A': id_a, 'id_B': id_b, 'target': target, 'date': row['tourney_date'], 'surface': surf,
        'rank_A': rank_a, 'rank_B': rank_b,
        'forme_A': f_a, 'forme_B': f_b,
        'surf_ratio_A': get_surf_ratio(id_a, surf), 'surf_ratio_B': get_surf_ratio(id_b, surf),
        'mental_A': get_mental_ratio(id_a), 'mental_B': get_mental_ratio(id_b),
        'h2h_A': h2h_ratio
    })

    # --- MISE À JOUR DES DICTIONNAIRES APRÈS LE MATCH ---
    # Surface
    stats_joueurs[w_id][f'w_{surf}'] = stats_joueurs[w_id].get(f'w_{surf}', 0) + 1
    stats_joueurs[l_id][f'l_{surf}'] = stats_joueurs[l_id].get(f'l_{surf}', 0) + 1
    # Mental
    if row['tourney_level'] == 'G':
        stats_joueurs[w_id]['w_GS'] += 1
        stats_joueurs[l_id]['l_GS'] += 1
    if row['round'] in ['F', 'SF']:
        stats_joueurs[w_id]['w_Final'] += 1
        stats_joueurs[l_id]['l_Final'] += 1
    # H2H
    h2h[h2h_key][w_id] += 1
    # Forme Générale
    forme_5[w_id].append(1)
    forme_5[l_id].append(0)

df_final = pd.DataFrame(data_expert)

# --- 4. CALCULS DES DIFFÉRENCES ET ENCODAGE ---
df_final['rank_diff'] = df_final['rank_A'] - df_final['rank_B']
df_final['forme_diff'] = df_final['forme_A'] - df_final['forme_B']
df_final['surf_diff'] = df_final['surf_ratio_A'] - df_final['surf_ratio_B']
df_final['mental_diff'] = df_final['mental_A'] - df_final['mental_B']

df_final = pd.get_dummies(df_final, columns=['surface'], prefix='is')
df_final = df_final.fillna(df_final.mean(numeric_only=True))

# --- 5. SPLIT ET ENTRAÎNEMENT ---
train_df = df_final[df_final['date'].dt.year < 2026].copy()
test_df = df_final[df_final['date'].dt.year >= 2026].copy()

cols_to_drop = ['target', 'id_A', 'id_B', 'date']
X_train = train_df.drop(cols_to_drop, axis=1)
y_train = train_df['target']
X_test = test_df.drop(cols_to_drop, axis=1)
y_test = test_df['target']

model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=12, 
    min_samples_leaf=10, 
    random_state=42, # Fixe le hasard pour avoir toujours le même résultat
    n_jobs=-1        # Utilise tous les cœurs de ton processeur pour aller vite
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nPrécision sur les matchs 2025/2026 : {acc * 100:.2f}%")

importances = pd.DataFrame({

    'Feature': X_train.columns,

    'Importance': model.feature_importances_

}).sort_values(by='Importance', ascending=False)



#print("\nImportance des variables :")

#print(importances)

# --- 6. PRÉDICTION MATCH DIRECT ---
def predire_match_expert(nom_a, nom_b, surface, model, columns, df_hist):
    def get_id(nom):
        res = df_hist[df_hist['winner_name'].str.contains(nom, case=False, na=False)]
        if res.empty: res = df_hist[df_hist['loser_name'].str.contains(nom, case=False, na=False)]
        return res.iloc[-1]['winner_id'] if not res.empty else None

    id_a, id_b = get_id(nom_a), get_id(nom_b)
    if not id_a or not id_b: return "Joueur non trouvé."

    # On récupère les dernières valeurs connues dans nos dictionnaires de fin de boucle
    f_a = sum(forme_5[id_a])/len(forme_5[id_a]) if forme_5[id_a] else 0.5
    f_b = sum(forme_5[id_b])/len(forme_5[id_b]) if forme_5[id_b] else 0.5
    
    # H2H
    h_key = tuple(sorted((id_a, id_b)))
    h_a_val = 0.5
    if h_key in h2h:
        total = h2h[h_key][id_a] + h2h[h_key][id_b]
        h_a_val = h2h[h_key][id_a] / total if total > 0 else 0.5

    # On récupère Rank récents
    m_a = df_hist[(df_hist['winner_id'] == id_a) | (df_hist['loser_id'] == id_a)].iloc[-1]
    m_b = df_hist[(df_hist['winner_id'] == id_b) | (df_hist['loser_id'] == id_b)].iloc[-1]
    r_a = m_a['winner_rank'] if m_a['winner_id'] == id_a else m_a['loser_rank']
    r_b = m_b['winner_rank'] if m_b['winner_id'] == id_b else m_b['loser_rank']

    input_data = {
        'rank_A': r_a, 'rank_B': r_b,
        'forme_A': f_a, 'forme_B': f_b, 'h2h_A': h_a_val,
        'surf_ratio_A': 0.6, 'surf_ratio_B': 0.5, # Moyennes simplifiées pour l'exemple
        'rank_diff': r_a - r_b, 'forme_diff': f_a - f_b,
        f'is_{surface}': 1
    }
    
    df_p = pd.DataFrame([input_data]).reindex(columns=columns, fill_value=0)
    prob = model.predict_proba(df_p)[0]
    print(f"\nPROBABILITÉS : {nom_a} {prob[1]*100:.1f}% | {nom_b} {prob[0]*100:.1f}%")

predire_match_expert("Etcheverry", "Bergs", "Hard", model, X_train.columns, df_global)
