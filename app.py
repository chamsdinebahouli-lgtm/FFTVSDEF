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
    page_title="Analyse FFT & Cinématique",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Diagnostic Cinématique")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_poulie_15_rpm):
    """
    Calcule les fréquences de la chaîne cinématique basées sur la vitesse
    de rotation de la petite poulie de 15 dents.
    """
    # 1. Fréquence de rotation de la petite poulie (Hz)
    f_poulie_15 = vitesse_poulie_15_rpm / 60.0
    
    # 2. Fréquence d'engrènement de la poulie (choc des dents sur la courroie)
    f_engrenement_poulie = f_poulie_15 * 15
    
    # 3. Fréquence de défilement complet de la courroie (126 dents)
    f_courroie = f_engrenement_poulie / 126
    
    # 4. Fréquence de rotation du moteur en amont du réducteur (1:246)
    f_moteur = f_poulie_15 * 246
    
    return {
        "Rotation Poulie 15d": f_poulie_15,
        "Engrènement 15d": f_engrenement_poulie,
        "Défilement Courroie": f_courroie,
        "Rotation Moteur": f_moteur
    }

# --------------------------------------------------
# FONCTIONS FFT AMÉLIORÉES
# --------------------------------------------------
def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)

    # Suppression de la composante continue
    x = x - np.mean(x)

    # Application d'une fenêtre de Hanning (Évite les fuites spectrales)
    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre

    dt = np.mean(np.diff(t))
    N = len(x)

    # Calcul FFT avec correction du gain du fenêtrage
    fft = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)

    return freq, fft

def amplitude_bande_max(freq, amp, cible, tolerance=0.45):
    """Cherche le pic maximum dans une bande autour de la cible (ex: 13.66 Hz)"""
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
    # Utilisation de la fréquence cible dynamique définie par l'utilisateur
    A_cible = amplitude_bande_max(freq, amp, cible_freq, tolerance=0.45)
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
# BARRE LATÉRALE (SIDEBAR) : CONFIGURATION
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration & Données")

uploaded_file = st.sidebar.file_uploader(
    "1. Importer le fichier Excel (.xlsx)",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Paramètres Mécaniques")

# Curseur pour ajuster la vitesse estimée de la poulie 15 dents
vitesse_estimee_poulie = st.sidebar.slider(
    "Vitesse de la poulie 15d (tr/min) :", 
    min_value=5.0, max_value=120.0, value=54.64, step=0.05
)

# Calcul dynamique des fréquences théoriques pour l'affichage
freqs_meca = calculer_frequences_theoriques(vitesse_estimee_poulie)
f_engrenement_defaut = freqs_meca["Engrènement 15d"]

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Retour d'expérience Terrain")
notes_text = st.sidebar.text_area(
    "Coller les scores de défaut réels :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67",
    height=120
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE / TRAITEMENT
# --------------------------------------------------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data = {}

    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns):
                continue

            freq, amp = calcul_fft(df)
            # On passe l'engrènement calculé comme fréquence cible pour l'IDM3
            indic = calcul_indicateurs(freq, amp, f_engrenement_defaut)
            indic["Ensemble"] = feuille
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur sur l'onglet {feuille} : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        colonnes = ["Ensemble", "Statut", "IDM3", "Amp Cible (Bande)", "Etotal", "Entropie"]
        resultats = resultats.sort_values("IDM3", ascending=False)

        # ------------------------------------------
        # AFFICHAGE : SYNTHÈSE DU PARC
        # ------------------------------------------
        st.subheader("📋 État de santé du parc de Motoréducteurs")
        
        moteurs_critiques = len(resultats[resultats["IDM3"] >= 1.5])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Machines analysées", len(resultats))
        col2.metric("En Alarme 🔴", moteurs_critiques, delta=-moteurs_critiques, delta_color="inverse")
        col3.metric("Fréquence Cible Suivie", f"{f_engrenement_defaut:.2f} Hz")

        st.dataframe(
            resultats[colonnes],
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # AFFICHAGE : FRÉQUENCES CALCULÉES
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Analyse Spectrale & Diagnostic de Panne")
        
        st.write("**Fréquences cinématiques calculées (Hz) correspondantes :**")
        cols_f = st.columns(len(freqs_meca))
        for i, (nom, f_val) in enumerate(freqs_meca.items()):
            cols_f[i].metric(nom, f"{f_val:.2f} Hz")

        ensemble = st.selectbox(
            "Sélectionner une machine pour visualiser son spectre FFT :",
            resultats["Ensemble"]
        )

        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})

        fig = px.line(
            fft_df, x="Fréquence (Hz)", y="Amplitude",
            title=f"Spectre FFT — {ensemble} (Axe limité à 0-40 Hz pour observation)"
        )
        fig.update_xaxes(range=[0, 40])
        
        # Styles de couleurs distincts pour chaque élément mécanique
        couleurs = {
            "Rotation Poulie 15d": "#00CC96",    # Vert
            "Engrènement 15d": "#EF553B",        # Rouge
            "Défilement Courroie": "#AB63FA",    # Violet
            "Rotation Moteur": "#19D3F3"          # Bleu ciel
        }
        
        # Tracer les lignes repères cinématiques sur la FFT
        for nom, f_val in freqs_meca.items():
            if f_val <= 40:  # On ne l'affiche que si visible sur le graphique
                fig.add_vline(
                    x=f_val, 
                    line_dash="dash", 
                    line_color=couleurs[nom],
                    annotation_text=nom, 
                    annotation_position="top right"
                )
        
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # COUCHE INTELLIGENCE ARTIFICIELLE
        # ------------------------------------------
        if notes_text:
            notes = {}
            for ligne in notes_text.splitlines():
                if "=" in ligne:
                    nom, valeur = ligne.split("=")
                    try:
                        notes[nom.strip()] = float(valeur.strip())
                    except:
                        pass

            resultats["Defaut_Réel"] = resultats["Ensemble"].map(notes)
            modele_df = resultats.dropna(subset=["Defaut_Réel"])

            if len(modele_df) >= 5:
                st.markdown("---")
                st.subheader("🤖 Module Prédictif Corrélation (Random Forest)")
                
                features = ["Amp Cible (Bande)", "Entropie", "E0_5", "E10_20", "IDM3"]
                X = modele_df[features]
                y = modele_df["Defaut_Réel"]

                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X, y)

                resultats["Prédiction IA"] = model.predict(resultats[features])
                corr = resultats["IDM3"].corr(resultats["Defaut_Réel"])

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Corrélation IDM3 / Terrain", f"{corr:.3f}")
                    st.caption("Plus la corrélation est proche de 1.000, plus le curseur cinématique est sur la bonne vitesse réelle.")
                
                with c2:
                    st.dataframe(
                        resultats[["Ensemble", "Defaut_Réel", "Prédiction IA", "IDM3"]].dropna(subset=["Defaut_Réel"]),
                        hide_index=True, use_container_width=True
                    )

        # ------------------------------------------
        # BOUTON D'EXPORTATION
        # ------------------------------------------
        st.markdown("---")
        sortie = BytesIO()
        with pd.ExcelWriter(sortie, engine="openpyxl") as writer:
            resultats.to_excel(writer, index=False, sheet_name="Synthese_Maintenance")

        st.download_button(
            label="📥 Télécharger le rapport d'analyse (.xlsx)",
            data=sortie.getvalue(),
            file_name="Rapport_Analyse_Vibratoire.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👋 Prêt pour l'analyse. Chargez un fichier Excel (.xlsx) contenant vos colonnes 'ms' et 'V' depuis le volet de gauche.")
