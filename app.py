import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
import plotly.express as px

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT — Multi-Étages",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Surveillance Multi-Étages (Vitesse Fixe)")

# --------------------------------------------------
# FONCTIONS CINÉMATIQUE
# --------------------------------------------------
def build_chain_from_fixed_sortie(f_sortie, ratio_reducteur, dents_primaire, dents_secondaire, dents_courroie):
    """
    Calcule toute la chaîne cinématique à partir d'une fréquence de sortie FIXE.
    Intègre désormais l'Engrènement du Dernier Étage constaté à ~3.67 Hz.
    """
    f_poulie_prim = f_sortie * (dents_secondaire / dents_primaire)
    f_moteur      = f_poulie_prim * ratio_reducteur
    f_engrenement = f_poulie_prim * dents_primaire
    f_courroie    = f_engrenement / dents_courroie
    f_4x          = f_engrenement * 4.0
    
    # Estimation théorique du nombre de dents associé au pic constaté à 3.67 Hz
    z_dernier_etage = round(3.67 / f_sortie)  # ~229 dents
    f_engrenement_sortie = f_sortie * z_dernier_etage

    return {
        "Rotation Sortie (Arbre Lent)": f_sortie,
        "Rotation Poulie Pri. (Interm.)": f_poulie_prim,
        "Engrènement 1er Étage":        f_engrenement,
        "Harmonique Engrènement 4X (État B)": f_4x,
        "Engrènement Dernier Étage (État C)": f_engrenement_sortie,
        "Défilement Courroie":          f_courroie,
        "Rotation Moteur (Arbre Rapide)": f_moteur,
    }

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
    fft_amp = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)
    return freq, fft_amp

# --------------------------------------------------
# RECHERCHE DE PICS
# --------------------------------------------------
def find_peak_near(freq, amp, target_hz, tolerance_pct=0.02):
    tol = target_hz * tolerance_pct
    fmin, fmax = target_hz - tol, target_hz + tol
    mask = (freq >= fmin) & (freq <= fmax)
    if not np.any(mask):
        idx = np.argmin(np.abs(freq - target_hz))
        return float(freq[idx]), float(amp[idx]), False
    sub_amp = amp[mask]
    sub_freq = freq[mask]
    idx_max = np.argmax(sub_amp)
    return float(sub_freq[idx_max]), float(sub_amp[idx_max]), True

def process_with_fixed_vitesse(freq, amp, f_sortie_fixe, machine_cfg, tolerance_pct=0.02, n_harmonics=3):
    chain = build_chain_from_fixed_sortie(
        f_sortie_fixe,
        machine_cfg["ratio_reducteur"],
        machine_cfg["dents_primaire"],
        machine_cfg["dents_secondaire"],
        machine_cfg["dents_courroie"],
    )

    identification = {}
    for nom, f_theoric in chain.items():
        f_found, a_found, found = find_peak_near(freq, amp, f_theoric, tolerance_pct)
        
        harmoniques = []
        for h in range(2, n_harmonics + 1):
            fh = f_theoric * h
            fh_found, ah_found, hfound = find_peak_near(freq, amp, fh, tolerance_pct)
            if hfound:
                harmoniques.append({"ordre": h, "f_trouvee": fh_found, "amplitude": ah_found})

        identification[nom] = {
            "f_theorique":  f_theoric,
            "f_trouvee":    f_found if found else f_theoric,
            "amplitude":    a_found if found else 0.0,
            "found":        found,
            "harmoniques":  harmoniques,
        }
    return identification

# --------------------------------------------------
# INDICATEURS MULTI-CIBLES
# --------------------------------------------------
def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin, fmax = cible - tolerance, cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask): return float(np.max(amp[mask]))
    return float(amp[np.argmin(np.abs(freq - cible))])

def calcul_idm3_specifique(freq, amp, f_cible):
    A_cible = amplitude_bande_max(freq, amp, f_cible, tolerance=0.1)
    Etotal   = float(np.sum(amp**2))
    
    # Entropie spectrale
    p = amp**2
    if np.sum(p) > 0:
        p = p / np.sum(p)
        H = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    else:
        H = 0.0
        
    valeur_brute = (A_cible**2 / Etotal) * H if Etotal > 0 else 0.0
    return 5.0 - valeur_brute

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Fréquence de Sortie Fixe")
f_sortie_imposee = st.sidebar.number_input("Fréquence de sortie (Hz) :", min_value=0.001, max_value=1.0, value=0.01600, format="%.5f")
st.sidebar.caption(f"Équivaut à : **{f_sortie_imposee * 60:.3f} RPM**")

tolerance_pct = st.sidebar.slider("Tolérance fenêtrage (%) :", 0.5, 5.0, 2.0, 0.5) / 100.0

# Configuration machine simplifiée par défaut
machine_cfg = {"ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126}

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
COLORS_MAP = {
    "Rotation Sortie (Arbre Lent)":       "#1D9E75",
    "Rotation Poulie Pri. (Interm.)":     "#7F77DD",
    "Engrènement 1er Étage":              "#BA7517",
    "Harmonique Engrènement 4X (État B)": "#E24B4A",
    "Engrènement Dernier Étage (État C)": "#185FA5",
    "Défilement Courroie":                "#4A90E2",
    "Rotation Moteur (Arbre Rapide)":     "#D85A30",
}

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data, calage_data = {}, {}

    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): continue

            freq, amp = calcul_fft(df)
            fft_data[feuille] = (freq, amp)

            # Identification aux fréquences verrouillées
            ident = process_with_fixed_vitesse(freq, amp, f_sortie_imposee, machine_cfg, tolerance_pct)
            calage_data[feuille] = ident

            # Extraction des deux cibles majeures
            f_cible_B = ident["Harmonique Engrènement 4X (État B)"]["f_trouvee"]
            f_cible_C = ident["Engrènement Dernier Étage (État C)"]["f_trouvee"]

            # Calcul des IDM3 distincts
            idm3_B = calcul_idm3_specifique(freq, amp, f_cible_B)
            idm3_C = calcul_idm3_specifique(freq, amp, f_cible_C)

            resultats.append({
                "Ensemble": feuille,
                "IDM3_État_B (3.20Hz)": idm3_B,
                "IDM3_État_C (3.67Hz)": idm3_C,
                "Amp_Sortie": ident["Rotation Sortie (Arbre Lent)"]["amplitude"],
                "Amp_État_B (4X)": ident["Harmonique Engrènement 4X (État B)"]["amplitude"],
                "Amp_État_C (Dernier Étage)": ident["Engrènement Dernier Étage (État C)"]["amplitude"],
            })
        except Exception as e:
            st.sidebar.error(f"Erreur onglet {feuille} : {e}")

    if resultats:
        resultats_df = pd.DataFrame(resultats)

        # 1. TABLEAU DE SYNTHÈSE
        st.subheader("📋 Tableau de bord de santé multi-étages")
        st.dataframe(resultats_df, use_container_width=True, hide_index=True)

        # 2. COMPARATIF PAR MACHINE
        st.markdown("---")
        st.subheader("🎯 Analyse détaillée par composant")
        tabs = st.tabs([r["Ensemble"] for _, r in resultats_df.iterrows()])

        for tab, (_, row) in zip(tabs, resultats_df.iterrows()):
            with tab:
                feuille = row["Ensemble"]
                ident = calage_data[feuille]

                # Métriques d'alerte ciblées
                cb, cc = st.columns(2)
                cb.metric("IDM3 — Premier Étage (Entrée)", f"{row['IDM3_État_B (3.20Hz)']:.3f}", 
                          delta="Alerte Matage" if row['Amp_État_B (4X)'] > 0.15 else "Normal")
                cc.metric("IDM3 — Dernier Étage (Sortie)", f"{row['IDM3_État_C (3.67Hz)']:.3f}", 
                          delta="Alerte Surcharge / Usure" if row['Amp_État_C (Dernier Étage)'] > 0.15 else "Normal")

                # Tableau des fréquences
                rows_display = []
                for nom_elem, res in ident.items():
                    rows_display.append({
                        "Composante": nom_elem,
                        "f théorique (Hz)": f"{res['f_theorique']:.5f}",
                        "f mesurée proche (Hz)": f"{res['f_trouvee']:.5f}" if res["found"] else "—",
                        "Amplitude (V)": f"{res['amplitude']:.4f}",
                        "Statut": "🟢 Détecté" if res["found"] else "⚪ Bruit de fond"
                    })
                st.dataframe(pd.DataFrame(rows_display), use_container_width=True, hide_index=True)

        # 3. GRAPHES SPECTRES AVEC DEUX LIGNES GUIDES
        st.markdown("---")
        st.subheader("📊 Visualisation interactive des spectres")
        ensemble = st.selectbox("Sélectionner la machine à tracer :", resultats_df["Ensemble"].tolist())

        if ensemble in fft_data:
            freq, amp = fft_data[ensemble]
            ident = calage_data[ensemble]

            fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
            fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT — {ensemble}")
            fig.update_xaxes(range=[0, 15]) # Focus 0-15 Hz là où tout se passe

            for nom_elem, res in ident.items():
                f_val = res["f_theorique"]
                if f_val > 15: continue
                col = COLORS_MAP.get(nom_elem, "#888888")
                fig.add_vline(x=f_val, line_dash="solid" if res["found"] else "dot", line_color=col, 
                              annotation_text=nom_elem, annotation_font_color=col)
            st.plotly_chart(fig, use_container_width=True)
