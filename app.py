import numpy as np
import pandas as pd
import os
os.system('cls')
from itertools import combinations
from collections import Counter
import joblib
import warnings
import streamlit as st
import altair as alt
import ast

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =====================================================================
# ------------- PAGINA WEB PER LA SCELTA DELLA PREVISIONE -------------
# =====================================================================

host_nations = ['United States', 'Mexico', 'Canada']

partite_reali = {
    ('Mexico', 'South Africa'): (2, 0),
    ('South Korea', 'Czechia'): (2, 1),
    ('Canada', 'Bosnia and Herzegovina'): (1, 1),
    ('United States', 'Paraguay'): (4, 1),
    ('Qatar', 'Switzerland'): (1, 1),
    ('Brazil', 'Morocco'): (1, 1),
    ('Haiti', 'Scotland'): (0, 1),
    ('Australia', 'Turkey'): (2, 0),
    ('Germany', 'Curaçao'): (7, 1),
    ('Netherlands', 'Japan'): (2, 2),
    ('Ivory Coast', 'Ecuador'): (1, 0),
    ('Sweden', 'Tunisia'): (5, 1),
    ('Spain', 'Cape Verde'): (0, 0),
    ('Belgium', 'Egypt'): (1, 1),
    ('Saudi Arabia', 'Uruguay'): (1, 1),
    ('Iran', 'New Zealand'): (2, 2),
    ('France', 'Senegal'): (3, 1),
    ('Iraq', 'Norway'): (1, 4),
    ('Argentina', 'Algeria'): (3, 0),
    ('Austria', 'Jordan'): (3, 1),
    ('Portugal', 'DR Congo'): (1, 1),
    ('England', 'Croatia'): (4, 2),
    ('Ghana', 'Panama'): (1, 0),
    ('Uzbekistan', 'Colombia'): (1, 3),
    ('Czechia', 'South Africa'): (1, 1),
    ('Switzerland', 'Bosnia and Herzegovina'): (4, 1),
    ('Canada', 'Qatar'): (6, 0),
    ('Mexico', 'South Korea'): (1, 0),
    ('United States', 'Australia'): (2, 0),
    ('Scotland', 'Morocco'): (0, 1),
    ('Brazil', 'Haiti'): (3, 0),
    ('Turkey', 'Paraguay'): (0, 1),
    ('Netherlands', 'Sweden'): (5, 1),
    ('Germany', 'Ivory Coast'): (2, 1),
    ('Ecuador', 'Curaçao'): (0, 0),
    ('Tunisia', 'Japan'): (0, 4),
    ('Spain', 'Saudi Arabia'): (4, 0),
    ('Belgium', 'Iran'): (0, 0),
    ('Uruguay', 'Cape Verde'): (2, 2),
    ('New Zealand', 'Egypt'): (1, 3),
    ('Argentina', 'Austria'): (2, 0),
    ('France', 'Iraq'): (3, 0),
    ('Norway', 'Senegal'): (3, 2),
    ('Jordan', 'Algeria'): (1, 2),
    ('Portugal', 'Uzbekistan'): (5, 0),
    ('England', 'Ghana'): (0, 0),
    ('Panama', 'Croatia'): (0, 1),
    ('Colombia', 'DR Congo'): (1, 0),
    ('Bosnia and Herzegovina', 'Qatar'): (3, 1),
    ('Switzerland', 'Canada'): (2, 1),
    ('Morocco', 'Haiti'): (4, 2),
    ('Scotland', 'Brazil'): (0, 3),
    ('South Africa', 'South Korea'): (1, 0),
    ('Czechia', 'Mexico'): (0, 3),
    ('Curaçao', 'Ivory Coast'): (0, 2),
    ('Ecuador', 'Germany'): (2, 1),
    ('Tunisia', 'Netherlands'): (1, 3),
    ('Japan', 'Sweden'): (1, 1),
    ('Paraguay', 'Australia'): (0, 0),
    ('Turkey', 'United States'): (3, 2),
    ('Norway', 'France'): (1, 4),
    ('Senegal', 'Iraq'): (5, 0),
    ('Cape Verde', 'Saudi Arabia'): (0, 0),
    ('Uruguay', 'Spain'): (0, 1),
    ('New Zealand', 'Belgium'): (1, 5),
    ('Egypt', 'Iran'): (1, 1),
    ('Panama', 'England'): (0, 2),
    ('Croatia', 'Ghana'): (2, 1),
    ('Colombia', 'Portugal'): (0, 0),
    ('DR Congo', 'Uzbekistan'): (3, 1),
    ('Algeria', 'Austria'): (3, 3),
    ('Jordan', 'Argentina'): (1, 3),
    ('South Africa', 'Canada'): (0, 1),
}

latest_elo = {
    'Argentina': 2144, 'Spain': 2144, 'France': 2123, 'England': 2028, 
    'Brazil': 2009, 'Colombia': 2006, 'Portugal': 1988, 'Netherlands': 1980, 
    'Norway': 1918, 'Germany': 1916, 'Switzerland': 1914, 'Mexico': 1912, 
    'Japan': 1910, 'Ecuador': 1902, 'Croatia': 1896, 'Morocco': 1877, 
    'Denmark': 1869, 'Italy': 1869, 'Belgium': 1869, 'Turkey': 1852, 
    'Uruguay': 1841, 'Austria': 1841, 'Senegal': 1842, 'Paraguay': 1815, 
    'Australia': 1800, 'United States': 1781, 'Algeria': 1780, 'Ukraine': 1780, 
    'Russia': 1772, 'Nigeria': 1767, 'Iran': 1766, 'Canada': 1748, 
    'Scotland': 1745, 'Greece': 1744, 'Ivory Coast': 1743, 'Sweden': 1742, 
    'Egypt': 1740, 'Serbia': 1734, 'Venezuela': 1733, 'South Korea': 1723, 
    'Chile': 1717, 'Kosovo': 1714, 'Hungary': 1710, 'Poland': 1710, 
    'Peru': 1699, 'Ireland': 1699, 'Wales': 1682, 'Slovenia': 1682, 
    'Czechia': 1680, 'Uzbekistan': 1677, 'Panama': 1668, 'Slovakia': 1667, 
    'DR Congo': 1666, 'Georgia': 1654, 'Israel': 1647, 'Romania': 1639, 
    'Jordan': 1632, 'Cape Verde': 1622, 'Bosnia and Herzegovina': 1622, 
    'Bolivia': 1621, 'Albania': 1616, 'Cameroon': 1614, 'Costa Rica': 1608, 
    'Northern Ireland': 1605, 'Saudi Arabia': 1596, 'North Macedonia': 1589, 
    'Mali': 1588, 'Iraq': 1561, 'Ghana': 1584, 'South Africa': 1575, 
    'Honduras': 1570, 'Iceland': 1568, 'Tunisia': 1562, 'New Zealand': 1549, 
    'Angola': 1542, 'United Arab Emirates': 1540, 'Finland': 1536, 
    'Burkina Faso': 1529, 'Jamaica': 1527, 'Belarus': 1522, 'Haiti': 1517, 
    'Guatemala': 1505, 'Oman': 1480, 'Syria': 1479, 'Palestine': 1465, 
    'Guinea': 1463, 'Montenegro': 1461, 'Bulgaria': 1458, 'Luxembourg': 1450, 
    'Northern Cyprus': 1442, 'Curaçao': 1438, 'Suriname': 1431, 
    'Kazakhstan': 1428, 'China': 1424, 'Kurdistan': 1424, 'Libya': 1420, 
    'Gambia': 1419, 'Bahrain': 1414, 'Qatar': 1411
}

@st.cache_data
def prepara_dati_live(df_storico, partite_giocate, nazioni_ospitanti, update_elo = True):
    """
    Aggiorna lo storico gol delle ultime 3 partite per ogni squadra e l'Elo.
    Restituisce un dizionario con Elo media gol aggiornata.
    """    
    live_elo = latest_elo.copy()

    # Inizializzazione storico gaol 
    live_goals = {}
    for _, row in df_storico.iterrows():
        try:
            live_goals[row['team']] = ast.literal_eval(row['last_3_goals'])
        except (ValueError, SyntaxError):
            live_goals[row['team']] = [1, 1, 1]
    
    # Aggiornamento con partite giocate
    for (home_team, away_team), (home_score, away_score) in partite_giocate.items():
        
        # Aggiornamento storico gol
        if home_team in live_goals:
            live_goals[home_team].append(home_score)
            live_goals[home_team] = live_goals[home_team][-3:]
            
        if away_team in live_goals:
            live_goals[away_team].append(away_score)
            live_goals[away_team] = live_goals[away_team][-3:]
            
    # Calcolo media goal attuale per ogni squadra
    live_avg_goals = {team: np.mean(goals) for team, goals in live_goals.items()}

    # --- Aggiornamento ELO ---
    if update_elo:
        K = 40  
        # Usiamo live_elo (non current_elo!)
        elo_home = live_elo.get(home_team, 1500)
        elo_away = live_elo.get(away_team, 1500)
        
        # Calcolo della differenza Elo corretta per il fattore campo
        elo_diff_adjusted = elo_home - elo_away
        if home_team in nazioni_ospitanti:
            elo_diff_adjusted += 80
        elif away_team in nazioni_ospitanti:
            elo_diff_adjusted -= 80
        
        # Probabilità di vittoria attesa corretta
        expected_home = 1 / (1 + 10 ** (-elo_diff_adjusted / 400))
        expected_away = 1 - expected_home
        
        # Determinazione del risultato (1 Vittoria Casa, 0 Vittoria Trasferta, 0.5 Pareggio)
        if home_score > away_score:
            w_home, w_away = 1, 0      
        elif home_score < away_score:
            w_home, w_away = 0, 1      
        else:
            w_home, w_away = 0.5, 0.5  
            
        # Aggiorniamo il dizionario live_elo
        live_elo[home_team] = round(elo_home + K * (w_home - expected_home), 1)
        live_elo[away_team] = round(elo_away + K * (w_away - expected_away), 1)

    return live_elo, live_avg_goals

@st.cache_resource
def carica_risorse():
    """
    Carica modelli e dataset di base.
    """

    modelli = joblib.load('modelli_wc2026.pkl')
    df_squadre = pd.read_csv('dati_squadre_correnti.csv') 
    
    return modelli, df_squadre

# =====================================================================
# --------------------------- MONDIALE 2026 ---------------------------
# =====================================================================

gironi_ufficiali = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Switzerland', 'Canada', 'Bosnia and Herzegovina', 'Qatar'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Egypt', 'Iran', 'Belgium', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Cape Verde', 'Saudi Arabia'],
    'I': ['France', 'Norway', 'Senegal', 'Iraq'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Colombia', 'Portugal', 'DR Congo', 'Uzbekistan'],
    'L': ['England', 'Ghana', 'Croatia', 'Panama']
}

def calcola_nuovo_elo_sim(elo_home, elo_away, outocome, is_knockout = False):
    """
    Calcola il nuovo punteggio Elo basato sul risultato della partita simulata.
    Il K-Factor è aumentato per le partite a eliminazione diretta (peso maggiore).
    """
        
    K = 30 if is_knockout else 20

    expected_home = 1 / (1 + 10 ** (-(elo_home - elo_away) / 400))
    expected_away = 1 - expected_home

    if outocome == 1:
        w_home, w_away = 1, 0
    elif outocome == 2:
        w_home, w_away = 0, 1
    else:
        w_home, w_away = 0.5, 0.5

    elo_home_new = elo_home + K * (w_home - expected_home)
    elo_away_new = elo_away + K * (w_away - expected_away)

    return elo_home_new, elo_away_new

def simulate_match_ensemble(home_team, away_team, modelli_dict, current_elo_dict, live_avg_goals, is_knockout = False):
    
    # Gestione squadra in casa
    scambio_effettuato = False

    if away_team in host_nations and home_team not in host_nations:
        vero_home, vero_away = away_team, home_team
        scambio_effettuato = True
    else:
        vero_home, vero_away = home_team, away_team

    is_neutral = 0 if vero_home in host_nations or vero_away in host_nations else 1

    elo_h_base = current_elo_dict.get(vero_home, 1500)
    elo_a_base = current_elo_dict.get(vero_away, 1500)

    # Inserimento della varianza stocastica 
    elo_h_sim = np.random.normal(loc = elo_h_base, scale = 15)
    elo_a_sim = np.random.normal(loc = elo_a_base, scale = 15)

    elo_diff = elo_h_sim - elo_a_sim
    goals_diff = live_avg_goals.get(vero_home, 1.0) - live_avg_goals.get(vero_away, 1.0)

    X_match = pd.DataFrame([[elo_diff, goals_diff, is_neutral]], columns=['elo_diff', 'goals_diff_last_3', 'neutral'])

    # Soft Ensemble
    p_rf = modelli_dict['RandomForest'].predict_proba(X_match)[0]
    p_xgb = modelli_dict['XGBoost'].predict_proba(X_match)[0]
    p_lgb = modelli_dict['LightGBM'].predict_proba(X_match)[0]
    probs = (p_rf + p_xgb + p_lgb) / 3

    if is_knockout:
        p_A = probs[1]
        p_B = probs[2]
        probs = [0.0, p_A / (p_A + p_B), p_B / (p_A + p_B)]

    probs = np.array(probs)
    probs = probs / np.sum(probs)

    outcome = np.random.choice([0, 1, 2], p=probs)

    new_elo_h, new_elo_a = calcola_nuovo_elo_sim(elo_h_base, elo_a_base, outcome, is_knockout)

    if scambio_effettuato:
        current_elo_dict[away_team] = new_elo_h
        current_elo_dict[home_team] = new_elo_a
        final_outcome = 2 if outcome == 1 else 1 if outcome == 2 else 0
    else:
        current_elo_dict[home_team] = new_elo_h
        current_elo_dict[away_team] = new_elo_a
        final_outcome = outcome

    return final_outcome

def simulate_group(nome_girone, teams, modelli_dict, current_elo_dict, live_avg_goals):
    """
    Simula tutte le partite di un girone e restituisce la classifica ordinata.
    """
     
    # Inizializzazione punti a 0 per ogni squadra
    punti = {team: 0 for team in teams}

    # Generazione delle combinazioni possibili di incontri
    matches = list(combinations(teams, 2))
    
    # Simulazione delle partite
    for team1, team2 in matches:
        
        # CORREZIONE: Qui passiamo correttamente live_avg_goals come 5° parametro!
        esito = simulate_match_ensemble(team1, team2, modelli_dict, current_elo_dict, live_avg_goals)
        
        if esito == 1:
            punti[team1] += 3       
        elif esito == 2:
            punti[team2] += 3       
        else:
            punti[team1] += 1       
            punti[team2] += 1
            
    # Ordinamento della classifica
    classifica_ordinata = sorted(punti.items(), key = lambda x: (x[1], current_elo_dict.get(x[0], 0)), reverse=True)
    
    return classifica_ordinata

def simulate_tournament_phase(modelli_dict, base_elo, live_avg_goals):
    current_tournament_elo = base_elo.copy()
    
    # FASE 1: GIRONI
    prime_classificate = []
    seconde_classificate = []
    terze_classificate = []
    
    # Simulazione separata dei gironi
    for group_name, teams in gironi_ufficiali.items():
        
        classifica = simulate_group(group_name, teams, modelli_dict, current_tournament_elo, live_avg_goals)
        
        squadra_prima = classifica[0][0]
        squadra_seconda = classifica[1][0]
        squadra_terza = classifica[2][0]
        punti_terza = classifica[2][1]
        
        prime_classificate.append((squadra_prima, group_name))
        seconde_classificate.append((squadra_seconda, group_name))
        
        dati_terza = {
            'team': (squadra_terza, group_name),
            'punti': punti_terza,
            'elo': current_tournament_elo.get(squadra_terza, 0)
        }
        terze_classificate.append(dati_terza)

    # Ripescaggio delle terze per punti e poi per Elo
    terze_ordinate = sorted(terze_classificate, key=lambda x: (x['punti'], x['elo']), reverse=True)
    migliori_8_terze = []
    for item in terze_ordinate[:8]:
        migliori_8_terze.append(item['team'])
    
    # FASE 2: CREAZIONE TABELLONE (SEDICESIMI)
    accoppiamenti = []
    
    prime_disponibili = list(prime_classificate)
    seconde_disponibili = list(seconde_classificate)
    terze_disponibili = list(migliori_8_terze)
    
    # Associazione delle terze con le prime 8 prime 
    for terza in terze_disponibili:
        nome_t, girone_t = terza
        for i, prima in enumerate(prime_disponibili):
            nome_p, girone_p = prima
            if girone_p != girone_t:
                accoppiamenti.append(nome_p)
                accoppiamenti.append(nome_t)
                prime_disponibili.pop(i) 
                break 
                
    # Associazione delle restanti 4 prime con 4 seconde 
    for prima in prime_disponibili:
        nome_p, girone_p = prima
        for i, seconda in enumerate(seconde_disponibili):
            nome_s, girone_s = seconda
            if girone_p != girone_s:
                accoppiamenti.append(nome_p)
                accoppiamenti.append(nome_s)
                seconde_disponibili.pop(i)
                break
                
    # Associazione delle restanti 8 seconde tra di loro
    while len(seconde_disponibili) > 1:
        squadra1 = seconde_disponibili.pop(0) 
        accoppiamento_trovato = False
        
        for i, squadra2 in enumerate(seconde_disponibili):
            if squadra1[1] != squadra2[1]: 
                accoppiamenti.append(squadra1[0])
                accoppiamenti.append(squadra2[0])
                seconde_disponibili.pop(i)
                accoppiamento_trovato = True
                break
                
        # Se non c'è scelta, vengono accoppiate le prime due rimaste
        if not accoppiamento_trovato:
            squadra2 = seconde_disponibili.pop(0)
            accoppiamenti.append(squadra1[0])
            accoppiamenti.append(squadra2[0])
            
    # FASE 3: ELIMINAZIONE DIRETTA
    
    def gioca_turno_semplice(squadre_in_gara):
        '''
        Prende una lista di squadre, le fa scontrare a due a due e ritorna i vincitori
        '''
        
        squadre_vincenti = []

        for i in range(0, len(squadre_in_gara), 2):
            team_a = squadre_in_gara[i]
            team_b = squadre_in_gara[i+1]
            
            esito = simulate_match_ensemble(team_a, team_b, modelli_dict, current_tournament_elo, live_avg_goals, is_knockout=True)
            
            if esito == 1:
                squadre_vincenti.append(team_a)
            else:
                squadre_vincenti.append(team_b)
                
        return squadre_vincenti
    
    sedicesimi_vincenti = gioca_turno_semplice(accoppiamenti)

    ottavi_vincenti = gioca_turno_semplice(sedicesimi_vincenti)

    quarti_vincenti = gioca_turno_semplice(ottavi_vincenti)

    semifinali_vincenti = gioca_turno_semplice(quarti_vincenti)

    campione = gioca_turno_semplice(semifinali_vincenti)
    
    return campione[0]

# =====================================================================
# ----------------------- INTERFACCIA STREAMLIT -----------------------
# =====================================================================

st.set_page_config(
    page_title="Oracolo Mondiali 2026",
    page_icon="🏆",
    layout="wide"
)

st.markdown("""
<style>
h1 a, h2 a, h3 a, h4 a {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

try:
    modelli_dict, df_squadre_base = carica_risorse()
    live_elo, live_avg_goals = prepara_dati_live(
        df_storico = df_squadre_base, 
        partite_giocate = partite_reali, 
        nazioni_ospitanti = host_nations,
        update_elo = True 
    )
except Exception as e:
    st.error(f"⚠️ Errore nel caricamento dei file o nell'aggiornamento live: {e}")
    st.stop()

st.title("🏆 Simulatore Predittivo - FIFA World Cup 2026")
st.markdown("Motore Live aggiornato in tempo reale | Modello Ensemble (RF + XGB + LGBM)")

st.sidebar.header("Vuoi sbancare con le bet?🤑💸" \
"Ci pensa Angelo! 😜")
modalita = st.sidebar.radio(
    "Scegli la modalità:",
    ["⚽ Previsione Match Singolo", "🏆🥇 Simulazione Torneo"]
)

# =====================================================================
# MODALITÀ 1: SINGOLO MATCH
# =====================================================================
if modalita == "⚽ Previsione Singolo Match":
    st.header("Previsione Singolo Match")
    
    # Estrazione elenco ordinato delle squadre dai dati aggiornati live
    elenco_squadre = sorted(list(live_elo.keys()))
    
    col1, col2 = st.columns(2)

    with col1:
        squadra_A = st.selectbox("Squadra A: ", elenco_squadre, index = None, placeholder = "Scegli la prima squadra...")
    with col2:
        squadra_B = st.selectbox("Squadra B: ", elenco_squadre, index = None, placeholder = "Scegli la seconda squadra...")
        
    is_knockout = st.toggle("Partita a eliminazione diretta (Supplementari/Rigori inclusi - Esclude il pareggio)")

    if st.button("Calcola Probabilità", type="primary"):
        if not squadra_A or not squadra_B:
            st.warning("⚠️ Seleziona entrambe le squadre per procedere!")
        elif squadra_A == squadra_B:
            st.warning("⚠️ Seleziona due squadre diverse! Una squadra non può giocare contro se stessa.")
        else:
            elo_A = live_elo.get(squadra_A, 1500)
            elo_B = live_elo.get(squadra_B, 1500)
            
            goals_A = live_avg_goals.get(squadra_A, 1.0)
            goals_B = live_avg_goals.get(squadra_B, 1.0)
            
            elo_diff = elo_A - elo_B
            goals_diff = goals_A - goals_B
            
            # Gestione squadra di casa
            is_neutral = 1
            if squadra_A in host_nations or squadra_B in host_nations:
                is_neutral = 0
            
            # Costruzione del vettore di input con i nomi corretti delle colonne
            X_input = pd.DataFrame(
                [[elo_diff, goals_diff, is_neutral]], 
                columns=['elo_diff', 'goals_diff_last_3', 'neutral']
            )
            
            # Inferenza con Soft Ensemble (Media aritmetica delle probabilità dei 3 modelli)
            prob_rf = modelli_dict['RandomForest'].predict_proba(X_input)[0]
            prob_xgb = modelli_dict['XGBoost'].predict_proba(X_input)[0]
            prob_lgb = modelli_dict['LightGBM'].predict_proba(X_input)[0]
            
            prob_ensemble = (prob_rf + prob_xgb + prob_lgb) / 3
            
            p_X = prob_ensemble[0]  
            p_A = prob_ensemble[1]  
            p_B = prob_ensemble[2]  
            
            # Gestione eliminazione diretta
            if is_knockout:
                p_A_norm = p_A / (p_A + p_B)
                p_B_norm = p_B / (p_A + p_B)
                p_A, p_B = p_A_norm, p_B_norm
                p_X = 0.0

            # Visualizzazione risultati 
            neutral_text = "No (Paese ospitante in campo)" if is_neutral == 0 else "Sì"

            st.success(f"""
            📊 **Parametri calcolati dall'algoritmo:**
            - **Differenza Elo:** {elo_diff:+.1f}
            - **Differenza Media Gol (Ultime 3):** {goals_diff:+.2f}
            - **Partita in Campo Neutro:** {neutral_text}
            """)
            
            # Visualizzazione delle metriche percentuali in 3 colonne distintive
            if is_knockout:
                c1, c2 = st.columns(2)
                c1.metric(label=f"Vittoria {squadra_A}", value=f"{p_A*100:.1f}%")
                c2.metric(label=f"Vittoria {squadra_B}", value=f"{p_B*100:.1f}%")
                
                chart_data = pd.DataFrame({
                    'Esito': [f'Vittoria {squadra_A}', f'Vittoria {squadra_B}'],
                    'Probabilità': [p_A, p_B]
                })
                sort_order = [f'Vittoria {squadra_A}', f'Vittoria {squadra_B}']
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(label=f"Vittoria {squadra_A}", value=f"{p_A*100:.1f}%")
                c2.metric(label="Pareggio", value=f"{p_X*100:.1f}%")
                c3.metric(label=f"Vittoria {squadra_B}", value=f"{p_B*100:.1f}%")
                
                chart_data = pd.DataFrame({
                    'Esito': [f'Vittoria {squadra_A}', 'Pareggio', f'Vittoria {squadra_B}'],
                    'Probabilità': [p_A, p_X, p_B]
                })
                sort_order = [f'Vittoria {squadra_A}', 'Pareggio', f'Vittoria {squadra_B}']
            
            # Generazione del grafico 
            grafico = alt.Chart(chart_data).mark_bar(color='#1f77b4').encode(
                x=alt.X(
                    'Esito', 
                    sort=sort_order, 
                    axis=alt.Axis(labelAngle=0, title=None, labelLimit=0) 
                ), 
                y=alt.Y(
                    'Probabilità', 
                    axis=alt.Axis(format='%', title='Probabilità Stimata')
                ) 
            ).properties(height=350) 
            
            st.altair_chart(grafico, use_container_width=True)

# =====================================================================
# MODALITÀ 2: SIMULAZIONE TORNEO
# =====================================================================
elif modalita == "🏆🥇 Simulazione Torneo (Monte Carlo)":
    st.header("Simulazione Torneo Globale")
    
    n_simulazioni = st.slider("Numero di Mondiali da simulare: ", min_value = 100, max_value=5000, value=1000, step=100)

    st.markdown(f"Il modello simulerà l'intero mondiale {n_simulazioni} volte, valutando probabilità, fluttuazioni di forma, scontri diretti e il formato a 48 squadre.")
    
    if st.button("Avvia Oracolo", type="primary"):

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        vittorie_mondiali = []
        
        # Simulazione Monte Carlo
        for i in range(n_simulazioni):
            if i % 10 == 0 or i == n_simulazioni - 1:
                progress_bar.progress((i + 1) / n_simulazioni)
                status_text.text(f"Simulazione {i} di {n_simulazioni} in corso...")
                
            campione = simulate_tournament_phase(modelli_dict, live_elo, live_avg_goals)
            vittorie_mondiali.append(campione)
            
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ {n_simulazioni} mondiali simulati con successo!")
        
        # --- CALCOLO RISULTATI ---
        conteggio_vittorie = Counter(vittorie_mondiali)
        classifica_probabilita = [(squadra, (vittorie / n_simulazioni) * 100) for squadra, vittorie in conteggio_vittorie.items()]
        classifica_probabilita.sort(key=lambda x: x[1], reverse=True)
        
        # Estrazione della Top 15 per visualizzarla
        top_15 = classifica_probabilita[:15]
        
        # --- VISUALIZZAZIONE GRAFICA ---
        st.subheader("La Top 15 delle Favorite")
        
        df_risultati = pd.DataFrame(top_15, columns=['Nazionale', 'Probabilità di Vittoria (%)'])
        
        # Grafico a barre orizzontali Altair
        grafico_torneo = alt.Chart(df_risultati).mark_bar(cornerRadiusEnd = 4).encode(
            x=alt.X('Probabilità di Vittoria (%):Q', title = 'Probabilità Stimata', axis = alt.Axis(format='.1f')),
            y=alt.Y('Nazionale:N', sort = '-x', title = None),
            color=alt.Color('Probabilità di Vittoria (%):Q', scale = alt.Scale(scheme = 'viridis'), legend = None),
            tooltip=['Nazionale', alt.Tooltip('Probabilità di Vittoria (%):Q', format = '.2f')]
        ).properties(height = 500)
        
        # Aggiunta delle etichette testuali alla fine di ogni barra
        text_labels = grafico_torneo.mark_text(
            align = 'left', baseline = 'middle', dx = 3, fontWeight = 'bold', color = 'white'
        ).encode(
            text = alt.Text('Probabilità di Vittoria (%):Q', format = '.1f')
        )
        
        st.altair_chart(grafico_torneo + text_labels, use_container_width = True)
        
        
