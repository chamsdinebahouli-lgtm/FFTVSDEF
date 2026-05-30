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
    page_title="Analyse FFT Graphique Interactif",
    layout="wide"
)

st.title("Analyse FFT — Recalage par Clic Direct sur Graphique")
st.markdown("### Cliquez directement sur le vrai pic moteur du graphique pour tout aligner")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE AVEC MICRO-AJUSTEMENT
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_moteur_rpm, micro_ajustement_hz):
    """
    Calcule toutes les fréquences cinématiques.
    Configuration : Moteur -> Réducteur 1:246 -> Poulie 15d -> Courroie 126d -> Poulie 50d
    """
    f_moteur = (vitesse_moteur_rpm / 60.0) + micro_ajustement_hz
    vitesse_moteur_corrigee_rpm = f_moteur * 60.0
    
    f_poulie_15 = f_moteur / 246.0
    f_engrenement = f_poulie_15 * 15.0
    f_courroie = f_engrenement / 126.0
    f_poulie_50 = f_poulie_15 * (15.0 / 50.0)
    vitesse_sortie_rpm = f_poulie_50 * 60.0
    
    return {
        "Rotation Moteur": f_moteur,
        "Rotation Poulie 15d": f_poulie_15,
        "Engrènement (15d/50d)": f_engrenement,
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

def amplitude_bande_max(freq, amp, cible, tolerance=0.08):
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
    A_cible = amplitude_bande_max(freq, amp, cible_freq, tolerance=0.08)
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
# GESTION DES ÉTATS (SESSION STATE)
# --------------------------------------------------
if "micro_hz" not in st.session_state:
    st.session_state.micro_hz = 0.000

# --------------------------------------------------
# BARRE LATÉRALE
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Calage Nominal Initial")
vitesse_moteur_slider = st.sidebar.slider(
    "Vitesse Moteur théorique (tr/min) :", 
    min_value=400.0, max_value=2000.0, value=820.0, step=1.0
)

# Bouton de réinitialisation du calage graphique
if st.sidebar.button("🔄 Réinitialiser le recalage à 0 Hz"):
    st.session_state.micro_hz = 0.000
    st.rerun()

# Calcul initial des fréquences théoriques pures (sans micro-ajustement) pour le calcul de décalage au clic
f_moteur_theorique_pure = vitesse_moteur_slider / 60.0

# Application du micro-ajustement actuel
freqs_meca, tr_min_sortie, tr_min_moteur_reel = calculer_frequences_theoriques(vitesse_moteur_slider, st.session_state.micro_hz)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Correction active :** `{st.session_state.micro_hz:+.3f} Hz`")
st.sidebar.metric("Moteur Recalé", f"{tr_min_moteur_reel:.2f} tr/min")
st.sidebar.metric("Sortie Recalée", f"{tr_min_sortie:.3f} tr/min")

notes_text = st.sidebar.text_area("Scores de défaut réels :", value="ASM21A=2.44\nASM21B=2.74", height=60)

# --------------------------------------------------
# LOGIQUE PRINCIPALE DYNAMIQUE
# --------------------------------------------------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data = {}

    f_cible_suivi = freqs_meca["Rotation Moteur"]

    # Traitement des feuilles avec prise en compte DYNAMIQUE des nouvelles fréquences ajustées
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns):
                continue

            freq, amp = calcul_fft(df)
            indic = calcul_indicateurs(freq, amp, f_cible_suivi)
            indic["Ensemble"] = feuille
            
            for nom_elem, f_elem in freqs_meca.items():
                tol = 0.01 if "Sortie" in nom_elem else 0.05
                indic[f"Amp_{nom_elem}"] = amplitude_bande_max(freq, amp, f_elem, tolerance=tol)
            
            # Recalcul dynamique immédiat de ton indicateur fétiche
            indic["IDM_Modulation"] = indic["Amp_Rotation Moteur"] * indic["Amp_Rotation Sortie (50d)"]
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        
        # 1. TABLEAU DE BORD EXHAUSTIF DYNAMIQUE
        st.subheader("📋 Indicateurs mis à jour en temps réel")
        colonnes_affichage = [
            "Ensemble", "Statut", "IDM3", "IDM_Modulation", 
            "Amp_Rotation Moteur", "Amp_Rotation Sortie (50d)", "Etotal", "Entropie"
        ]
        resultats_triés = resultats.sort_values("IDM3", ascending=False)
        st.dataframe(resultats_triés[colonnes_affichage], use_container_width=True, hide_index=True)

        # ------------------------------------------
        # GRAPHIC INTERACTIF BIDIRECTONNEL
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Graphique de Calage Tactique")
        st.info("🎯 **Mode d'emploi :** Regarde où se trouve le pic réel de ton moteur. **Clique une fois sur ce pic dans le graphique**. L'application va instantanément aspirer cette fréquence et caler toutes les lignes dessus !")

        ensemble = st.selectbox("Sélectionner la machine à calibrer :", resultats_triés["Ensemble"])
        
        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
        ligne_machine = resultats[resultats["Ensemble"] == ensemble].iloc[0]

        # Création du graphique Plotly standard
        fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT — {ensemble}")
        fig.update_xaxes(range=[0, 20])
        
        couleurs = {"Rotation Moteur": "#EF553B", "Rotation Poulie 15d": "#00CC96", "Engrènement (15d/50d)": "#AB63FA", "Défilement Courroie": "#19D3F3", "Rotation Sortie (50d)": "#FFA15A"}
        
        for nom, f_val in freqs_meca.items():
            if f_val <= 20:
                fig.add_vline(x=f_val, line_dash="dash", line_color=couleurs[nom], annotation_text=f"{nom} ({f_val:.3f} Hz)")

        # --- CAPTURE DU CLIC SUR LE GRAPHique (Magie Streamlit 1.30+) ---
        # On active l'écoute des clics de souris sur les données du graphique
        evenement_clic = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        # Si l'utilisateur clique sur un point du graphique
        if evenement_clic and "selection" in evenement_clic and "points" in evenement_clic["selection"]:
            points = evenement_clic["selection"]["points"]
            if len(points) > 0:
                # On extrait la fréquence précise (l'axe X) là où l'opérateur a cliqué
                frequence_cliquee = points[0]["x"]
                
                # Le but est de faire correspondre la Fréquence Moteur Théorique sur ce point cliqué.
                # Calcul de la nouvelle correction (Micro-ajustement)
                nouvelle_correction = frequence_cliquee - f_moteur_theorique_pure
                
                # Sauvegarde en mémoire et rechargement dynamique
                st.session_state.micro_hz = float(nouvelle_correction)
                st.toast(f"🎯 Calage réussi sur {frequence_cliquee:.3f} Hz ! Recalcul global en cours...", icon="🚀")
                st.rerun()

        # Affichage des résultats individuels raffinés sous le graphique
        st.write("**Amplitudes lues après ton calage par clic :**")
        cols_f = st.columns(len(freqs_meca))
        for i, (nom, f_val) in enumerate(freqs_meca.items()):
            cols_f[i].metric(label=f"{nom}", value=f"{ligne_machine[f'Amp_{nom}']:.4f} V")

        # ------------------------------------------
        # EXPORT DATA
        # ------------------------------------------
        st.markdown("---")
        sortie = BytesIO()
        with pd.ExcelWriter(sortie, engine="openpyxl") as writer:
            resultats_triés.to_excel(writer, index=False, sheet_name="Synthese_Clic_Dynamique")
        st.download_button(label="📥 Télécharger le rapport ajusté par clic (.xlsx)", data=sortie.getvalue(), file_name="Rapport_FFT_Clic_Dynamique.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("👋 Importez votre fichier Excel. Vous pourrez ensuite cliquer sur le graphique pour ajuster instantanément les calculs.")
