import streamlit as st
import pandas as pd
import numpy as np

from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

# --------------------------------------------------
# CONFIGURATION & STYLE
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT Expert",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Corrélation Modulation Haute Précision")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_moteur_rpm, micro_ajustement_hz):
    """
    Rapports stricts : Moteur -> Réducteur 1:246 -> Poulie 15d -> Courroie 126d -> Poulie 50d
    + Suivi de l'harmonique 4X d'engrènement
    """
    f_moteur = (vitesse_moteur_rpm / 60.0) + micro_ajustement_hz
    vitesse_moteur_corrigee_rpm = f_moteur * 60.0
    
    f_poulie_15 = f_moteur / 246.0
    f_engrenement = f_poulie_15 * 15.0
    f_courroie = f_engrenement / 126.0
    f_poulie_50 = f_poulie_15 * (15.0 / 50.0)
    vitesse_sortie_rpm = f_poulie_50 * 60.0
    
    f_engrenement_4x = f_engrenement * 4.0
    
    return {
        "Rotation Moteur": f_moteur,
        "Rotation Poulie 15d": f_poulie_15,
        "Engrènement (15d/50d)": f_engrenement,
        "Harmonique Engrènement 4X": f_engrenement_4x,
        "Défilement Courroie": f_courroie,
        "Rotation Sortie (50d)": f_poulie_50
    }, vitesse_sortie_rpm, vitesse_moteur_corrigee_rpm

# --------------------------------------------------
# FONCTIONS FFT
# --------------------------------------------------
def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)
    x = x - np.mean(x)
    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre
    dt = np.mean(np.diff(t))
    N = len(x)
    fft = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)
    return freq, fft

def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin = cible - tolerance
    fmax = cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask):
        return float(np.max(amp[mask]))
    else:
        idx = np.argmin(np.abs(freq - cible))
        return float(amp[idx])

def energie_totale(amp):
    return float(np.sum(amp**2))

def energie_bande(freq, amp, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    return float(np.sum(amp[mask]**2))

def entropie_spectrale(amp):
    p = amp**2
    if np.sum(p) == 0:
        return 0.0
    p = p / np.sum(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def calcul_indicateurs(freq, amp, cible_freq):
    A_cible = amplitude_bande_max(freq, amp, cible_freq, tolerance=0.1)
    Etotal = energie_totale(amp)
    H = entropie_spectrale(amp)
    E05 = energie_bande(freq, amp, 0, 5)
    E1020 = energie_bande(freq, amp, 10, 20)

    IDM3 = 0.0
    if Etotal > 0:
        IDM3 = (A_cible**2 / Etotal) * H

    if IDM3 < 0.5:
        statut = "🟢 Bon"
    elif IDM3 < 1.5:
        statut = "🟡 À surveiller"
    else:
        statut = "🔴 Alarme"

    return {
        "Amp Cible (Bande)": A_cible,
        "Etotal": Etotal,
        "Entropie": H,
        "E0_5": E05,
        "E10_20": E1020,
        "IDM3": IDM3,
        "Statut": statut
    }

# --------------------------------------------------
# INITIALISATION DU SESSION STATE (MÉMOIRE)
# --------------------------------------------------
if "micro_hz" not in st.session_state:
    st.session_state.micro_hz = 0.000

# --------------------------------------------------
# BARRE LATÉRALE
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Vitesse Nominale")
vitesse_moteur_slider = st.sidebar.slider(
    "Vitesse Moteur théorique (tr/min) :", 
    min_value=400.0, max_value=2000.0, value=820.0, step=1.0
)

# Application de la correction
freqs_meca, tr_min_sortie, tr_min_moteur_reel = calculer_frequences_theoriques(vitesse_moteur_slider, st.session_state.micro_hz)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 État du Calage")
st.sidebar.metric("Correction active (Hz)", f"{st.session_state.micro_hz:+.3f} Hz")
st.sidebar.metric("Moteur Réel Corrigé", f"{tr_min_moteur_reel:.2f} tr/min")
st.sidebar.metric("Sortie Réelle Corrigée", f"{tr_min_sortie:.3f} tr/min")

if st.sidebar.button("🔄 Réinitialiser le calage à 0 Hz"):
    st.session_state.micro_hz = 0.000
    st.rerun()

st.sidebar.markdown("---")
notes_text = st.sidebar.text_area(
    "📝 Scores de défaut réels (Feedback Terrain) :", 
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67", 
    height=120,
    help="Entrez ici les sévérités constatées pour corréler automatiquement vos indicateurs avec le terrain."
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data = {}

    f_cible_suivi = freqs_meca["Rotation Moteur"]

    # Traitement complet des données
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns):
                continue

            freq, amp = calcul_fft(df)
            
            # 1. KPIs Généraux
            indic = calcul_indicateurs(freq, amp, f_cible_suivi)
            indic["Ensemble"] = feuille
            
            # 2. KPIs par pièce
            for nom_elem, f_elem in freqs_meca.items():
                # Fenêtre serrée pour l'arbre de sortie très lent (0.016 Hz)
                tol = 0.005 if "Sortie" in nom_elem else 0.08
                indic[f"Amp_{nom_elem}"] = amplitude_bande_max(freq, amp, f_elem, tolerance=tol)
            
            # 3. Indicateurs croisés de modulation
            indic["IDM_Modulation"] = indic["Amp_Rotation Moteur"] * indic["Amp_Rotation Sortie (50d)"]
            
            # NOUVEL INDICATEUR PHYSIQUE CIBLE : Produit de l'arbre lent (~0.016Hz) et de l'H4X (~3.68Hz)
            indic["IDM_Modulation_4X"] = indic["Amp_Rotation Sortie (50d)"] * indic["Amp_Harmonique Engrènement 4X"]
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur onglet {feuille} : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        
        # --- PARSING ET INTEGRATION DU DEFAUT REEL ---
        notes = {}
        if notes_text:
            for ligne in notes_text.splitlines():
                if "=" in ligne:
                    nom, valeur = ligne.split("=")
                    try: notes[nom.strip()] = float(valeur.strip())
                    except: pass
        
        resultats["Defaut_Réel"] = resultats["Ensemble"].map(notes)
        
        # Organisation du tableau d'affichage expert
        colonnes_affichage = [
            "Ensemble", "Statut", "Defaut_Réel", "IDM_Modulation_4X", "IDM3", "IDM_Modulation", 
            "Amp_Rotation Sortie (50d)", "Amp_Harmonique Engrènement 4X", "Amp_Rotation Moteur",
            "Amp_Engrènement (15d/50d)", "Amp_Rotation Poulie 15d", "Amp_Défilement Courroie"
        ]
        
        # Tri initial par pertinence vis-à-vis de l'indicateur ciblé
        resultats_triés = resultats.sort_values("IDM_Modulation_4X", ascending=False)

        # ------------------------------------------
        # INTERACTION & CORRÉLATION DE NOTRE NOUVEL INDICATEUR
        # ------------------------------------------
        st.subheader("📋 État de santé exhaustif du parc de Motoréducteurs")
        
        df_valid_corr = resultats_triés.dropna(subset=["Defaut_Réel"])
        
        if len(df_valid_corr) >= 2:
            # Calcul des corrélations de Pearson avec les données terrain
            corr_mod4x = df_valid_corr["IDM_Modulation_4X"].corr(df_valid_corr["Defaut_Réel"])
            corr_mod = df_valid_corr["IDM_Modulation"].corr(df_valid_corr["Defaut_Réel"])
            
            # Affichage des KPIs de corrélation
            c_c1, c_c2, c_c3 = st.columns(3)
            
            # Remplacement par le KPI demandé : Corrélation de la modulation 4X (0.016Hz * 3.68Hz)
            c_c1.metric(
                label="📉 Corrélation Mod. 4X [0.016Hz × 3.68Hz] / Terrain", 
                value=f"{corr_mod4x:.3f}", 
                delta="Cible Validée" if corr_mod4x > 0.8 else "Modérée" if corr_mod4x > 0.5 else "Ajuster le calage"
            )
            c_c2.metric(
                label="📉 Corrélation Mod. Moteur [Moteur × Sortie] / Terrain", 
                value=f"{corr_mod:.3f}"
            )
            c_c3.markdown("<div style='padding-top:10px; font-size:13px; color:#888;'><i>Note technique : Une corrélation élevée sur la Modulation 4X confirme un problème géométrique ou un défaut de denture cyclique sur l'arbre basse vitesse.</i></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Renseignez au moins 2 machines valides dans 'Scores de défaut réels' (ex: ASM21A=2.44) pour débloquer le calcul de corrélation dynamique.")

        # Affichage du Tableau de bord mis à jour
        st.dataframe(resultats_triés[colonnes_affichage], use_container_width=True, hide_index=True)

        # ------------------------------------------
        # COMMANDES DE RECALAGE
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Calage Fin & Analyse du Spectre Vibratoire")
        
        ensemble = st.selectbox("Sélectionner une machine à analyser en détail :", resultats_triés["Ensemble"])
        
        st.write("**🕹️ Commandes de recalage instantané au millième de Hz :**")
        c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns([1, 1, 2, 1, 1])
        
        if c_btn1.button("⏪ - 0.010 Hz"):
            st.session_state.micro_hz -= 0.010
            st.rerun()
        if c_btn2.button("◀️ - 0.001 Hz"):
            st.session_state.micro_hz -= 0.001
            st.rerun()
        with c_btn3:
            st.markdown(f"<h4 style='text-align: center; color: #19D3F3;'>Décalage : {st.session_state.micro_hz:+.3f} Hz</h4>", unsafe_allow_html=True)
        if c_btn4.button("▶️ + 0.001 Hz"):
            st.session_state.micro_hz += 0.001
            st.rerun()
        if c_btn5.button("⏩ + 0.010 Hz"):
            st.session_state.micro_hz += 0.010
            st.rerun()

        # Données spécifiques de la machine sélectionnée
        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
        ligne_machine = resultats[resultats["Ensemble"] == ensemble].iloc[0]

        # Affichage des compteurs cinématiques
        st.write("**Amplitudes lues aux repères actuels :**")
        cols_f = st.columns(6)
        noms_p = [
            "Rotation Moteur", "Rotation Poulie 15d", 
            "Engrènement (15d/50d)", "Harmonique Engrènement 4X", 
            "Défilement Courroie", "Rotation Sortie (50d)"
        ]
        for idx, nom in enumerate(noms_p):
            cols_f[idx].metric(label=f"{nom} ({freqs_meca[nom]:.3f} Hz)", value=f"{ligne_machine[f'Amp_{nom}']:.4f} V")

        # Trace du Graphique FFT
        fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT — {ensemble}")
        fig.update_xaxes(range=[0, 20])
        
        couleurs = {
            "Rotation Moteur": "#EF553B", 
            "Rotation Poulie 15d": "#00CC96", 
            "Engrènement (15d/50d)": "#AB63FA", 
            "Harmonique Engrènement 4X": "#FFD700", 
            "Défilement Courroie": "#19D3F3", 
            "Rotation Sortie (50d)": "#FFA15A"
        }
        
        for nom, f_val in freqs_meca.items():
            if f_val <= 20:
                fig.add_vline(x=f_val, line_dash="dash", line_color=couleurs[nom], annotation_text=nom)
        
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # APPRENTISSAGE IA
        # ------------------------------------------
        modele_df = resultats_triés.dropna(subset=["Defaut_Réel"])

        if len(modele_df) >= 5:
            st.markdown("---")
            st.subheader("🤖 Apprentissage IA (Toutes Caractéristiques)")
            
            # Intégration de la nouvelle feature de modulation fine dans l'entraînement du Random Forest
            features = ["Amp Cible (Bande)", "Entropie", "E0_5", "E10_20", "IDM3", "IDM_Modulation", "IDM_Modulation_4X"]
            X = modele_df[features]
            y = modele_df["Defaut_Réel"]

            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X, y)

            resultats_triés["Prédiction IA"] = model.predict(resultats_triés[features])

            st.dataframe(
                resultats_triés[["Ensemble", "Defaut_Réel", "Prédiction IA", "IDM3", "IDM_Modulation_4X"]].dropna(subset=["Defaut_Réel"]),
                hide_index=True, use_container_width=True
            )

        # ------------------------------------------
        # EXPORT TOTAL
        # ------------------------------------------
        st.markdown("---")
        sortie = BytesIO()
        with pd.ExcelWriter(sortie, engine="openpyxl") as writer:
            resultats_triés.to_excel(writer, index=False, sheet_name="Synthese_Totale")
        st.download_button(label="📥 Télécharger le registre complet (.xlsx)", data=sortie.getvalue(), file_name="Registre_Vibratoire_Total.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("👋 Données prêtes pour l'évaluation de la corrélation par modulation 4X. Chargez votre fichier Excel.")
