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
    page_title="Analyse FFT Motoréducteur v2",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Dashboard Industriel")

# --------------------------------------------------
# FONCTIONS FFT AMÉLIORÉES
# --------------------------------------------------
def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)

    # 1. Suppression de la composante continue
    x = x - np.mean(x)

    # 2. Application d'une fenêtre de Hanning (Évite les fuites spectrales)
    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre

    dt = np.mean(np.diff(t))
    N = len(x)

    # 3. Calcul FFT (Correction du gain due au fenêtrage : * 2 / somme(fenetre))
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
        # Repli si la fréquence est hors plage
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

def calcul_indicateurs(freq, amp):
    # Utilisation de la recherche par bande (13.21 Hz à 14.11 Hz)
    A1366 = amplitude_bande_max(freq, amp, 13.66, tolerance=0.45)
    Etotal = energie_totale(amp)
    H = entropie_spectrale(amp)
    E05 = energie_bande(freq, amp, 0, 5)
    E1020 = energie_bande(freq, amp, 10, 20)

    IDM3 = 0.0
    if Etotal > 0:
        IDM3 = (A1366**2 / Etotal) * H

    # Détermination du statut de criticité pour le visuel
    if IDM3 < 0.5:
        statut = "🟢 Bon"
    elif IDM3 < 1.5:
        statut = "🟡 À surveiller"
    else:
        statut = "🔴 Alarme"

    return {
        "A13.66 (Max Bande)": A1366,
        "Etotal": Etotal,
        "Entropie": H,
        "E0_5": E05,
        "E10_20": E1020,
        "IDM3": IDM3,
        "Statut": statut
    }

# --------------------------------------------------
# BARRE LATÉRALE (SIDEBAR) : CONFIGURATION & ENTRÉES
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration & Données")

uploaded_file = st.sidebar.file_uploader(
    "1. Importer le fichier Excel (.xlsx)",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Retour d'expérience Terrain")
notes_text = st.sidebar.text_area(
    "Coller les scores de défaut réels :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67",
    height=150
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
            indic = calcul_indicateurs(freq, amp)
            indic["Ensemble"] = feuille
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur sur l'onglet {feuille} : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        # Réorganisation des colonnes pour la lisibilité
        colonnes = ["Ensemble", "Statut", "IDM3", "A13.66 (Max Bande)", "Etotal", "Entropie"]
        resultats = resultats.sort_values("IDM3", ascending=False)

        # ------------------------------------------
        # AFFICHAGE : CLASSEMENT PAR CRITICITÉ
        # ------------------------------------------
        st.subheader("📋 État de santé du parc de Motoréducteurs")
        
        moteurs_critiques = len(resultats[resultats["IDM3"] >= 1.5])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total machines analysées", len(resultats))
        col2.metric("Machines en Alarme 🔴", moteurs_critiques, delta=-moteurs_critiques, delta_color="inverse")
        col3.metric("Score IDM3 Maximum", f"{resultats['IDM3'].max():.2f}")

        st.dataframe(
            resultats[colonnes],
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # AFFICHAGE : GRAPHIQUES FFT INTERACTIFS
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Analyse Spectrale Détaillée")
        
        ensemble = st.selectbox(
            "Sélectionner une machine pour voir son spectre FFT :",
            resultats["Ensemble"]
        )

        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})

        fig = px.line(
            fft_df, x="Fréquence (Hz)", y="Amplitude",
            title=f"Spectre FFT — {ensemble} (Zoom fenêtré 0-25 Hz)"
        )
        fig.update_xaxes(range=[0, 25])
        
        # AJUSTEMENT ICI : Utilisation de annotation_text pour éviter le bug de version Plotly
        fig.add_vrect(
            x0=13.21, 
            x1=14.11, 
            line_width=0, 
            fillcolor="rgba(255, 0, 0, 0.1)", 
            annotation_text="Zone défaut (13.66 Hz)",
            annotation_position="top left"
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
                st.subheader("🤖 Module Prédictif (Random Forest)")
                
                features = ["A13.66 (Max Bande)", "Entropie", "E0_5", "E10_20", "IDM3"]
                X = modele_df[features]
                y = modele_df["Defaut_Réel"]

                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X, y)

                resultats["Prédiction IA"] = model.predict(resultats[features])
                corr = resultats["IDM3"].corr(resultats["Defaut_Réel"])

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Corrélation IDM3 / Terrain", f"{corr:.3f}")
                    st.caption("Une valeur proche de 1 indique que l'indicateur IDM3 est extrêmement fiable pour ce type de défaut.")
                
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
            label="📥 Télécharger le rapport complet d'analyse (.xlsx)",
            data=sortie.getvalue(),
            file_name="Rapport_Analyse_Vibratoire.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👋 Bienvenue ! Veuillez charger un fichier Excel contenant vos données vibratoires temporelles (colonnes 'ms' et 'V') dans le panneau latéral gauche pour démarrer l'analyse.")
