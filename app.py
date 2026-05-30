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
    page_title="Analyse FFT Commandes Chirurgicales",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Ajustement Synchrone au Millième")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE AVEC MICRO-AJUSTEMENT
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_moteur_rpm, micro_ajustement_hz):
    """
    Calcule toutes les fréquences de la chaîne cinématique à partir de la vitesse moteur
    et applique un micro-ajustement chirurgical au millième en Hz sur la fréquence moteur.
    Configuration : Moteur -> Réducteur 1:246 -> Poulie 15d -> Courroie 126d -> Poulie 50d
    """
    # 1. Fréquence de rotation du moteur (Hz) + Correction ultra-fine
    f_moteur = (vitesse_moteur_rpm / 60.0) + micro_ajustement_hz
    
    # Vitesse RPM réelle corrigée
    vitesse_moteur_corrigee_rpm = f_moteur * 60.0
    
    # 2. Rotation en sortie de réducteur (Arbre poulie 15 dents)
    f_poulie_15 = f_moteur / 246.0
    
    # 3. Fréquence d'engrènement des poulies (choc des dents)
    f_engrenement = f_poulie_15 * 15.0
    
    # 4. Fréquence de défilement de la courroie (126 dents)
    f_courroie = f_engrenement / 126.0
    
    # 5. Vitesse de rotation de la grande poulie finale (50 dents)
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
# BARRE LATÉRALE (SIDEBAR) - SYNCHRONISÉE SANS ERREUR
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration & Données")

uploaded_file = st.sidebar.file_uploader(
    "1. Importer le fichier Excel (.xlsx)",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Calage Cinématique Synchrone")

# 1. Ajustement grossier (RPM)
vitesse_moteur_slider = st.sidebar.slider(
    "1. Vitesse Moteur nominale (tr/min) :", 
    min_value=400.0, max_value=2000.0, value=820.0, step=1.0
)

st.sidebar.markdown("**2. Recalage micrométrique (Hz) :**")

# Initialisation de la variable centrale si elle n'existe pas encore
if "micro_hz" not in st.session_state:
    st.session_state.micro_hz = 0.000

# Callbacks de synchronisation réciproque
def update_depuis_input():
    st.session_state.micro_hz = st.session_state.input_micro

def update_depuis_slider():
    st.session_state.micro_hz = st.session_state.slider_micro

# Commande A : Boutons +/- numériques (Pas de 0.001 Hz)
st.sidebar.number_input(
    "Ajuster par boutons (+/-) :",
    min_value=-2.000,
    max_value=2.000,
    step=0.001,
    format="%.3f",
    key="input_micro",
    value=st.session_state.micro_hz,
    on_change=update_depuis_input
)

# Commande B : Glissière graphique synchronisée
st.sidebar.slider(
    "Ajuster par glissière :",
    min_value=-2.000,
    max_value=2.000,
    step=0.001,
    format="%.3f",
    key="slider_micro",
    value=st.session_state.micro_hz,
    on_change=update_depuis_slider
)

# Récupération de la valeur finale stabilisée
micro_ajustement = st.session_state.micro_hz

# Calcul des fréquences cinématiques basées sur cette valeur unique
freqs_meca, tr_min_sortie, tr_min_moteur_reel = calculer_frequences_theoriques(vitesse_moteur_slider, micro_ajustement)

st.sidebar.markdown("**Vitesses résultantes :**")
st.sidebar.metric("Moteur Réel Corrigé", f"{tr_min_moteur_reel:.2f} tr/min")
st.sidebar.metric("Sortie Réelle Corrigée", f"{tr_min_sortie:.3f} tr/min")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Retour d'expérience Terrain")
notes_text = st.sidebar.text_area(
    "Coller les scores de défaut réels :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67",
    height=80
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data = {}

    f_cible_suivi = freqs_meca["Rotation Moteur"]

    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns):
                continue

            freq, amp = calcul_fft(df)
            
            # Calcul avec cible recalée au millième
            indic = calcul_indicateurs(freq, amp, f_cible_suivi)
            indic["Ensemble"] = feuille
            
            # Extraction des amplitudes basées sur les positions chirurgicales
            for nom_elem, f_elem in freqs_meca.items():
                tol = 0.01 if "Sortie" in nom_elem else 0.08  
                indic[f"Amp_{nom_elem}"] = amplitude_bande_max(freq, amp, f_elem, tolerance=tol)
            
            indic["IDM_Modulation"] = indic["Amp_Rotation Moteur"] * indic["Amp_Rotation Sortie (50d)"]
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur sur l'onglet {feuille} : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        
        colonnes_affichage = [
            "Ensemble", "Statut", "IDM3", "IDM_Modulation", 
            "Etotal", "Entropie", "E0_5", "E10_20",
            "Amp_Rotation Moteur", "Amp_Rotation Poulie 15d", 
            "Amp_Engrènement (15d/50d)", "Amp_Défilement Courroie", "Amp_Rotation Sortie (50d)"
        ]
        
        resultats_triés = resultats.sort_values("IDM3", ascending=False)

        # ------------------------------------------
        # TABLES & METRICS GLOBALES
        # ------------------------------------------
        st.subheader("📋 État de santé du parc (Fréquences Recalées au Millième)")
        
        moteurs_critiques = len(resultats_triés[resultats_triés["IDM3"] >= 1.5])
        col1, col2, col3 = st.columns(3)
        col1.metric("Machines analysées", len(resultats_triés))
        col2.metric("En Alarme 🔴", moteurs_critiques, delta=-moteurs_critiques, delta_color="inverse")
        col3.metric("Ligne Moteur Calée sur", f"{f_cible_suivi:.3f} Hz")

        st.dataframe(resultats_triés[colonnes_affichage], use_container_width=True, hide_index=True)

        # ------------------------------------------
        # FOCUS ET GRAPHIQUE FFT
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Zoom & Calage Millimétrique des Composants")
        st.caption("💡 **Astuce de contrôle :** Utilisez les boutons haut/bas de la boîte numérique à gauche pour ajuster pas à pas de pile 0.001 Hz.")
        
        ensemble = st.selectbox(
            "Sélectionner une machine pour aligner les spectres :",
            resultats_triés["Ensemble"]
        )

        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})

        ligne_machine = resultats[resultats["Ensemble"] == ensemble].iloc[0]
        
        # Affichage des amplitudes lues (au millième de Hz près)
        st.write("**Amplitudes extraites au sommet des repères calculés :**")
        cols_f = st.columns(len(freqs_meca))
        for i, (nom, f_val) in enumerate(freqs_meca.items()):
            cols_f[i].metric(label=f"{nom} ({f_val:.3f} Hz)", value=f"{ligne_machine[f'Amp_{nom}']:.4f} V")

        # Graphique Plotly
        fig = px.line(
            fft_df, x="Fréquence (Hz)", y="Amplitude",
            title=f"Spectre FFT Haute Résolution — {ensemble}"
        )
        fig.update_xaxes(range=[0, 20])
        
        couleurs = {
            "Rotation Moteur": "#EF553B",        
            "Rotation Poulie 15d": "#00CC96",    
            "Engrènement (15d/50d)": "#AB63FA",  
            "Défilement Courroie": "#19D3F3",    
            "Rotation Sortie (50d)": "#FFA15A"   
        }
        
        for nom, f_val in freqs_meca.items():
            if f_val <= 20:
                fig.add_vline(
                    x=f_val, 
                    line_dash="dash", 
                    line_color=couleurs[nom],
                    annotation_text=f"{nom} ({f_val:.3f} Hz)", 
                    annotation_position="top right"
                )
        
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # COUCHE IA
        # ------------------------------------------
        if notes_text:
            notes = {}
            for ligne in notes_text.splitlines():
                if "=" in ligne:
                    nom, valeur = ligne.split("=")
                    try: notes[nom.strip()] = float(valeur.strip())
                    except: pass

            resultats_triés["Defaut_Réel"] = resultats_triés["Ensemble"].map(notes)
            modele_df = resultats_triés.dropna(subset=["Defaut_Réel"])

            if len(modele_df) >= 5:
                st.markdown("---")
                st.subheader("🤖 Apprentissage IA Optimisé (Fréquences Alignées)")
                
                features = ["Amp Cible (Bande)", "Entropie", "E0_5", "E10_20", "IDM3", "IDM_Modulation"]
                X = modele_df[features]
                y = modele_df["Defaut_Réel"]

                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X, y)

                resultats_triés["Prédiction IA"] = model.predict(resultats_triés[features])
                corr = resultats_triés["IDM3"].corr(resultats_triés["Defaut_Réel"])

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Corrélation de Précision", f"{corr:.3f}")
                with c2:
                    st.dataframe(
                        resultats_triés[["Ensemble", "Defaut_Réel", "Prédiction IA", "IDM3"]].dropna(subset=["Defaut_Réel"]),
                        hide_index=True, use_container_width=True
                    )

        # ------------------------------------------
        # EXPORT
        # ------------------------------------------
        st.markdown("---")
        sortie = BytesIO()
        with pd.ExcelWriter(sortie, engine="openpyxl") as writer:
            resultats_triés.to_excel(writer, index=False, sheet_name="Synthese_Millieme")

        st.download_button(
            label="📥 Télécharger le rapport haute précision (.xlsx)",
            data=sortie.getvalue(),
            file_name="Analyse_Vibratoire_Millieme.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👋 Prêt pour l'analyse au millième. Chargez votre fichier Excel.")
