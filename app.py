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
    page_title="Analyse FFT & Indicateur Croisé",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Indicateur de Modulation")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE 
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_moteur_rpm):
    """
    Calcule toutes les fréquences de la chaîne cinématique à partir de la vitesse moteur.
    Configuration : Moteur -> Réducteur 1:246 -> Poulie 15d -> Courroie 126d -> Poulie 50d
    """
    # 1. Fréquence de rotation du moteur (Hz)
    f_moteur = vitesse_moteur_rpm / 60.0
    
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
    }, vitesse_sortie_rpm

# --------------------------------------------------
# FONCTIONS FFT 
# --------------------------------------------------
def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)

    # Suppression composante continue
    x = x - np.mean(x)

    # Fenêtre de Hanning
    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre

    dt = np.mean(np.diff(t))
    N = len(x)

    fft = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)

    return freq, fft

def amplitude_bande_max(freq, amp, cible, tolerance=0.45):
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
# BARRE LATÉRALE (SIDEBAR)
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration & Données")

uploaded_file = st.sidebar.file_uploader(
    "1. Importer le fichier Excel (.xlsx)",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Analyse Cinématique")

vitesse_moteur_slider = st.sidebar.slider(
    "Ajuster Vitesse Moteur (tr/min) :", 
    min_value=400.0, max_value=2000.0, value=820.0, step=1.0
)

freqs_meca, tr_min_sortie = calculer_frequences_theoriques(vitesse_moteur_slider)

st.sidebar.metric("Vitesse calculée en BAM (Sortie)", f"{tr_min_sortie:.2f} tr/min")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Retour d'expérience Terrain")
notes_text = st.sidebar.text_area(
    "Coller les scores de défaut réels :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67",
    height=100
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
            indic = calcul_indicateurs(freq, amp, f_cible_suivi)
            indic["Ensemble"] = feuille
            
            # Extraction des amplitudes individuelles de la chaîne
            for nom_elem, f_elem in freqs_meca.items():
                # Note : On réduit la tolérance à 0.15 pour la basse fréquence (sortie) pour éviter de capter le bruit continu
                tol = 0.05 if "Sortie" in nom_elem else 0.3
                indic[f"Amp_{nom_elem}"] = amplitude_bande_max(freq, amp, f_elem, tolerance=tol)
            
            # --- AJOUT DU NOUVEL INDICATEUR CROISÉ ---
            # Multiplication Amplitude Moteur * Amplitude Sortie (1 tr/min)
            indic["IDM_Modulation"] = indic["Amp_Rotation Moteur"] * indic["Amp_Rotation Sortie (50d)"]
            
            resultats.append(indic)
            fft_data[feuille] = (freq, amp)
        except Exception as e:
            st.sidebar.error(f"Erreur sur l'onglet {feuille} : {e}")

    if len(resultats) > 0:
        resultats = pd.DataFrame(resultats)
        
        # Intégration de l'IDM_Modulation dans le tableau principal de synthèse
        colonnes_synthese = ["Ensemble", "Statut", "IDM3", "IDM_Modulation", "Amp_Rotation Moteur", "Amp_Rotation Sortie (50d)"]
        # On trie ici par votre nouvel indicateur pour mettre en évidence les modulations suspectes
        resultats_triés = resultats.sort_values("IDM_Modulation", ascending=False)

        # ------------------------------------------
        # TABLES & METRICS
        # ------------------------------------------
        st.subheader("📋 État de santé du parc de Motoréducteurs")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Machines analysées", len(resultats_triés))
        col2.metric("IDM_Modulation Max trouvé", f"{resultats_triés['IDM_Modulation'].max():.5f}")
        col3.metric("Fréquence Moteur (Vitesse)", f"{f_cible_suivi:.2f} Hz")

        st.write("**Classement par niveau d'Indicateur de Modulation (Moteur × Sortie) :**")
        st.dataframe(resultats_triés[colonnes_synthese], use_container_width=True, hide_index=True)

        # ------------------------------------------
        # GRAPHIQUE FFT & EXTRACTION DES AMPLITUDES
        # ------------------------------------------
        st.markdown("---")
        st.subheader("📊 Focus Spectre & Indicateurs par Composant")
        
        ensemble = st.selectbox(
            "Sélectionner une machine pour le détail :",
            resultats_triés["Ensemble"]
        )

        freq, amp = fft_data[ensemble]
        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})

        ligne_machine = resultats[resultats["Ensemble"] == ensemble].iloc[0]
        
        # Affichage des métriques de la machine sélectionnée
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        c_kpi1.metric("Amp. Moteur (A)", f"{ligne_machine['Amp_Rotation Moteur']:.4f} V")
        c_kpi2.metric("Amp. Sortie ~1tr/min (B)", f"{ligne_machine['Amp_Rotation Sortie (50d)']:.4f} V")
        c_kpi3.metric("Indicateur Croisé (A × B)", f"{ligne_machine['IDM_Modulation']:.5f}", delta="Modulation")

        # Graphique Plotly
        fig = px.line(
            fft_df, x="Fréquence (Hz)", y="Amplitude",
            title=f"Spectre FFT — {ensemble}"
        )
        fig.update_xaxes(range=[0, 20])
        
        couleurs = {
            "Rotation Moteur": "#EF553B",        # Rouge
            "Rotation Poulie 15d": "#00CC96",    # Vert
            "Engrènement (15d/50d)": "#AB63FA",  # Violet
            "Défilement Courroie": "#19D3F3",    # Bleu
            "Rotation Sortie (50d)": "#FFA15A"   # Orange
        }
        
        for nom, f_val in freqs_meca.items():
            if f_val <= 20:
                fig.add_vline(
                    x=f_val, 
                    line_dash="dash", 
                    line_color=couleurs[nom],
                    annotation_text=f"{nom} ({ligne_machine[f'Amp_{nom}']:.3f}V)", 
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
                st.subheader("🤖 Module IA Predictive (Random Forest)")
                
                # Ajout du nouvel indicateur croisé dans les features d'apprentissage de l'IA
                features = ["Amp Cible (Bande)", "Entropie", "IDM3", "IDM_Modulation"]
                X = modele_df[features]
                y = modele_df["Defaut_Réel"]

                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X, y)

                resultats_triés["Prédiction IA"] = model.predict(resultats_triés[features])
                corr = resultats_triés["IDM_Modulation"].corr(resultats_triés["Defaut_Réel"])

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Corrélation Nouvelle Variable / Terrain", f"{corr:.3f}")
                with c2:
                    st.dataframe(
                        resultats_triés[["Ensemble", "Defaut_Réel", "Prédiction IA", "IDM_Modulation", "IDM3"]].dropna(subset=["Defaut_Réel"]),
                        hide_index=True, use_container_width=True
                    )

        # ------------------------------------------
        # EXPORT
        # ------------------------------------------
        st.markdown("---")
        sortie = BytesIO()
        with pd.ExcelWriter(sortie, engine="openpyxl") as writer:
            resultats_triés.to_excel(writer, index=False, sheet_name="Synthese_Modulation")

        st.download_button(
            label="📥 Télécharger le rapport (.xlsx)",
            data=sortie.getvalue(),
            file_name="Rapport_Modulation_Vibratoire.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👋 En attente de votre fichier Excel pour calculer l'indicateur croisé (Moteur × Sortie).")
