import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
import plotly.express as px

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT Motoréducteur",
    layout="wide"
)

st.title("Analyse FFT Avancée — Surveillance Multi-Étages à Vitesse Verrouillée")

# --------------------------------------------------
# FONCTIONS CINÉMATIQUE
# --------------------------------------------------
def build_chain_from_fixed_sortie(f_sortie, ratio_reducteur, dents_primaire, dents_secondaire, dents_courroie):
    """
    Calcule toute la chaîne cinématique à partir d'une fréquence de sortie FIXE.
    Intègre l'Engrènement du Premier Étage, l'État Cible B (4X) et le Dernier Étage (~3.67 Hz).
    """
    f_poulie_prim = f_sortie * (dents_secondaire / dents_primaire)
    f_moteur      = f_poulie_prim * ratio_reducteur
    f_engrenement = f_poulie_prim * dents_primaire
    f_courroie    = f_engrenement / dents_courroie
    f_4x          = f_engrenement * 4.0
    
    # Nombre de dents estimé pour le dernier étage donnant le pic constaté à ~3.67 Hz
    z_dernier_etage = round(3.67 / f_sortie)  # ~229 dents
    f_engrenement_sortie = f_sortie * z_dernier_etage

    return {
        "Rotation Sortie":                  f_sortie,
        "Rotation Poulie Primaire":         f_poulie_prim,
        "Engrènement 1er Étage":            f_engrenement,
        "Harmonique Engrènement 4X":        f_4x,
        "Engrènement Dernier Étage":        f_engrenement_sortie,
        "Défilement Courroie":              f_courroie,
        "Rotation Moteur":                  f_moteur,
    }

# --------------------------------------------------
# FONCTIONS FFT
# --------------------------------------------------
def calcul_fft(df):
    """FFT standard avec fenêtre de Hanning."""
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

def signal_info(df):
    """Retourne les métriques de qualité du signal."""
    N = len(df)
    dt = 0.020  # 20 ms fixe
    T = N * dt
    fs = 1.0 / dt
    resolution_brute = 1.0 / T
    return {
        "N": N,
        "dt_ms": dt * 1000,
        "T_s": T,
        "fs_hz": fs,
        "resolution_hz": resolution_brute,
    }

# --------------------------------------------------
# ANALYSE ET RECHERCHE DE PICS
# --------------------------------------------------
def find_peak_near(freq, amp, target_hz, tolerance_pct=0.02):
    """Cherche le pic dominant dans la bande de tolérance autour de la cible."""
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

def confidence_score(ecart_pct, amplitude, amp_max):
    if amp_max <= 0: return 0
    ecart_score = max(0.0, 1.0 - (ecart_pct / 2.0))
    amp_score   = min(1.0, amplitude / amp_max)
    return int((0.6 * ecart_score + 0.4 * amp_score) * 100)

def process_with_fixed_vitesse(freq, amp, f_sortie_fixe, machine_cfg, tolerance_pct=0.02, n_harmonics=4):
    amp_max = np.max(amp)
    chain = build_chain_from_fixed_sortie(
        f_sortie_fixe,
        machine_cfg["ratio_reducteur"],
        machine_cfg["dents_primaire"],
        machine_cfg["dents_secondaire"],
        machine_cfg["dents_courroie"],
    )

    identification = {}
    for nom, f_theoric in chain.items():
        is_sortie = (nom == "Rotation Sortie")
        f_found, a_found, found = find_peak_near(freq, amp, f_theoric, tolerance_pct)
        ecart_pct = abs(f_found - f_theoric) / f_theoric * 100 if (found and not is_sortie) else 0.0
        conf = confidence_score(ecart_pct, a_found, amp_max) if found else 0

        # Recherche d'harmoniques
        harmoniques = []
        for h in range(2, n_harmonics + 1):
            fh = f_theoric * h
            fh_found, ah_found, hfound = find_peak_near(freq, amp, fh, tolerance_pct)
            if hfound:
                harmoniques.append({"ordre": h, "f_theorique": fh, "f_trouvee": fh_found, "amplitude": ah_found})

        identification[nom] = {
            "f_theorique":  f_theoric,
            "f_trouvee":    f_found if (found or is_sortie) else f_theoric,
            "amplitude":    a_found if (found or is_sortie) else 0.0,
            "found":        found,
            "ecart_pct":    ecart_pct if not is_sortie else 0.0,
            "confiance":    conf if not is_sortie else 100,
            "harmoniques":  harmoniques,
            "fixe":         is_sortie
        }
    return identification

# --------------------------------------------------
# INDICATEURS MATHÉMATIQUES
# --------------------------------------------------
def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin, fmax = cible - tolerance, cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask): return float(np.max(amp[mask]))
    return float(amp[np.argmin(np.abs(freq - cible))])

def matrix_energie_totale(amp):
    return float(np.sum(amp**2))

def energie_bande(freq, amp, fmin, fmax):
    return float(np.sum(amp[(freq >= fmin) & (freq <= fmax)]**2))

def entropie_spectrale(amp):
    p = amp**2
    if np.sum(p) == 0: return 0.0
    p = p / np.sum(p)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))

def calcul_indicateurs_specifiques(freq, amp, f_cible):
    A_cible = amplitude_bande_max(freq, amp, f_cible, tolerance=0.1)
    Etotal   = matrix_energie_totale(amp)
    H        = entropie_spectrale(amp)
    E05      = energie_bande(freq, amp, 0, 5)
    E1020    = energie_bande(freq, amp, 10, 20)
    valeur_brute = (A_cible**2 / Etotal) * H if Etotal > 0 else 0.0
    IDM3 = 5.0 - valeur_brute
    return {"A_cible": A_cible, "Etotal": Etotal, "Entropie": H, "E0_5": E05, "E10_20": E1020, "IDM3": IDM3}

# --------------------------------------------------
# STOCKAGE MACHINE (SESSION STATE)
# --------------------------------------------------
if "machines" not in st.session_state:
    st.session_state.machines = [
        {"nom": "ASM21A", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
        {"nom": "ASM21B", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
    ]

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("🛠️ Réglages Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Fréquence de Sortie Fixe")
vitesse_mode = st.sidebar.radio("Saisie de la vitesse via :", ["Fréquence (Hz)", "Vitesse (RPM)"])
if vitesse_mode == "Fréquence (Hz)":
    f_sortie_imposee = st.sidebar.number_input("Fréquence de sortie (Hz) :", min_value=0.001, max_value=2.0, value=0.01600, format="%.5f")
    st.sidebar.caption(f"Équivaut à : **{f_sortie_imposee*60:.3f} RPM**")
else:
    rpm_impose = st.sidebar.number_input("Vitesse de sortie (RPM) :", min_value=0.01, max_value=120.0, value=0.96, format="%.2f")
    f_sortie_imposee = rpm_impose / 60.0
    st.sidebar.caption(f"Équivaut à : **{f_sortie_imposee:.5f} Hz**")

tolerance_pct = st.sidebar.slider("Tolérance de recherche (%) :", 0.5, 5.0, 2.0, 0.5) / 100.0
n_harmonics = st.sidebar.slider("Nombre d'harmoniques suivies :", 1, 8, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cinématiques")
with st.sidebar.expander("Gérer les paramètres machines"):
    nb_machines = st.number_input("Nb Machines :", min_value=1, max_value=10, value=len(st.session_state.machines))
    while len(st.session_state.machines) < nb_machines:
        st.session_state.machines.append({"nom": f"Machine{len(st.session_state.machines)+1}", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126})
    st.session_state.machines = st.session_state.machines[:nb_machines]

    for i, m in enumerate(st.session_state.machines):
        st.markdown(f"**Machine {i+1}**")
        m["nom"] = st.text_input("Nom", value=m["nom"], key=f"n_{i}")
        m["ratio_reducteur"] = st.number_input("Ratio Réducteur", value=m["ratio_reducteur"], min_value=1, key=f"r_{i}")
        m["dents_primaire"] = st.number_input("Dents Prim.", value=m["dents_primaire"], min_value=1, key=f"dp_{i}")
        m["dents_secondaire"] = st.number_input("Dents Sec.", value=m["dents_secondaire"], min_value=1, key=f"ds_{i}")
        m["dents_courroie"] = st.number_input("Dents Courroie", value=m["dents_courroie"], min_value=1, key=f"dc_{i}")

st.sidebar.markdown("---")
notes_text = st.sidebar.text_area("📝 Données d'expertise terrain (Nom=Score) :", value="ASM21A=2.44\nASM21B=2.74", height=80)

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
COLORS_MAP = {
    "Rotation Sortie":                  "#1D9E75",
    "Rotation Poulie Primaire":         "#7F77DD",
    "Engrènement 1er Étage":            "#BA7517",
    "Harmonique Engrènement 4X":        "#E24B4A",
    "Engrènement Dernier Étage":        "#185FA5",
    "Défilement Courroie":              "#4A90E2",
    "Rotation Moteur":                  "#D85A30",
}

def conf_badge(conf, found, is_fixe=False):
    if is_fixe: return "🔒 IMPOSÉ"
    if not found: return "⚪ Non trouvé"
    return f"🟢 {conf}%" if conf >= 70 else f"🟡 {conf}%" if conf >= 40 else f"🔴 {conf}%"

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data, calage_data = {}, {}

    # Parsing du feedback terrain
    notes = {}
    for line in notes_text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try: notes[k.strip()] = float(v.strip())
            except ValueError: pass

    # Onglet de diagnostic signal
    for feuille in xls.sheet_names:
        try:
            _df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if {"ms", "V"}.issubset(_df.columns):
                info = signal_info(_df)
                st.subheader("📡 Diagnostic de Résolution Spectrale")
                ri1, ri2, ri3 = st.columns(3)
                ri1.metric("Points Mesurés", f"{info['N']:,}")
                ri2.metric("Durée d'Acquisition", f"{info['T_s']:.1f} s ({info['T_s']/60:.1f} min)")
                ri3.metric("Pas Fréquentiel (Δf)", f"{info['resolution_hz']:.5f} Hz")
                st.success(f"✅ Vitesse verrouillée à {f_sortie_imposee:.5f} Hz — Analyse multi-étages stabilisée.")
                st.markdown("---")
                break
        except Exception: pass

    # Boucle de traitement
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): continue

            freq, amp = calcul_fft(df)
            fft_data[feuille] = (freq, amp)

            # Identification config machine
            m_cfg = st.session_state.machines[0]
            for m in st.session_state.machines:
                if m["nom"].lower() in feuille.lower() or feuille.lower() in m["nom"].lower():
                    m_cfg = m
                    break

            ident = process_with_fixed_vitesse(freq, amp, f_sortie_imposee, m_cfg, tolerance_pct, n_harmonics)
            calage_data[feuille] = ident

            # Calculs énergétiques séparés pour l'État B (3.20 Hz) et l'État C (3.67 Hz)
            f_B = ident["Harmonique Engrènement 4X"]["f_trouvee"]
            f_C = ident["Engrènement Dernier Étage"]["f_trouvee"]

            ind_B = calcul_indicateurs_specifiques(freq, amp, f_B)
            ind_C = calcul_indicateurs_specifiques(freq, amp, f_C)

            # Génération de la ligne d'indicateurs globaux
            res_row = {
                "Ensemble": feuille,
                "Defaut_Réel": notes.get(feuille, np.nan),
                "IDM3_État_B (3.20Hz)": ind_B["IDM3"],
                "IDM3_État_C (3.67Hz)": ind_C["IDM3"],
                "Etotal": ind_B["Etotal"],
                "Entropie": ind_B["Entropie"],
                "E0_5": ind_B["E0_5"],
                "E10_20": ind_B["E10_20"],
                "Amp_Rotation_Sortie": ident["Rotation Sortie"]["amplitude"],
                "Amp_État_B_4X": ident["Harmonique Engrènement 4X"]["amplitude"],
                "Amp_État_C_Dernier": ident["Engrènement Dernier Étage"]["amplitude"],
                "Amp_Rotation_Moteur": ident["Rotation Moteur"]["amplitude"],
            }
            # Modulations croisées sécurisées
            res_row["ID_Modulation_Moteur_Sortie"] = res_row["Amp_Rotation_Moteur"] * res_row["Amp_Rotation_Sortie"]
            res_row["IDM_Modulation_4X_Sortie"]   = res_row["Amp_État_B_4X"] * res_row["Amp_Rotation_Sortie"]

            resultats.append(res_row)
        except Exception as e:
            st.sidebar.error(f"Erreur sur l'onglet {feuille} : {e}")

    if not resultats:
        st.warning("Aucun onglet valide trouvé.")
        st.stop()

    resultats_df = pd.DataFrame(resultats)

    # --------------------------------------------------
    # SEUILLAGE STATISTIQUE DYNAMIQUE (SIGMA)
    # --------------------------------------------------
    st.subheader("📋 Synthèse de Santé Générale et Écarts Statistiques")
    df_valid = resultats_df.dropna(subset=["Defaut_Réel"])

    # On calibre les seuils dynamiques sur l'IDM3 de l'État B (notre historique de référence)
    if len(df_valid) >= 3:
        X = df_valid["Defaut_Réel"].values
        Y = df_valid["IDM3_État_B (3.20Hz)"].values
        m_lin, b_lin = np.polyfit(X, Y, 1)
        residus = Y - (m_lin * X + b_lin)
        sigma = np.std(residus) if np.std(residus) > 0 else 1.0

        statuts = []
        for _, r in resultats_df.iterrows():
            if not pd.isna(r["Defaut_Réel"]):
                val_attendue = m_lin * r["Defaut_Réel"] + b_lin
                ecart = r["IDM3_État_B (3.20Hz)"] - val_attendue
                if ecart <= 1.0 * sigma: statuts.append("🟢 Conforme à la tendance")
                elif ecart <= 2.0 * sigma: statuts.append("🟡 Écart Modéré (Hors Tendance)")
                else: statuts.append("🔴 Alarme Critique (Dérive Forte)")
            else:
                statuts.append("🟢 Bon" if r["IDM3_État_B (3.20Hz)"] < 3.5 else "🟡 À surveiller" if r["IDM3_État_B (3.20Hz)"] < 4.5 else "🔴 Alarme")
        resultats_df["Statut_Tendance"] = statuts

        # Métriques de corrélations globales
        corr_B = df_valid["IDM3_État_B (3.20Hz)"].corr(df_valid["Defaut_Réel"])
        corr_C = df_valid["IDM3_État_C (3.67Hz)"].corr(df_valid["Defaut_Réel"])
        corr_mod = df_valid["IDM_Modulation_4X_Sortie"].corr(df_valid["Defaut_Réel"])

        c_cor1, c_cor2, c_cor3 = st.columns(3)
        c_cor1.metric("R² Corrélation État B (3.20Hz)", f"{corr_B:.3f}", delta="Modèle σ Actif")
        c_cor2.metric("R² Corrélation État C (3.67Hz)", f"{corr_C:.3f}")
        c_cor3.metric("R² Corrélation Modulation 4X/Sortie", f"{corr_mod:.3f}")
    else:
        resultats_df["Statut_Tendance"] = resultats_df["IDM3_État_B (3.20Hz)"].apply(lambda x: "🟢 Bon" if x < 3.5 else "🟡 À surveiller" if x < 4.5 else "🔴 Alarme")
        st.info("💡 Ajoutez au moins 3 valeurs de 'Défaut Réel' dans la barre latérale pour activer le calcul de dérive par Sigma.")

    # Affichage de la table maîtresse ordonnée
    cols_order = [
        "Ensemble", "Statut_Tendance", "Defaut_Réel", "IDM3_État_B (3.20Hz)", "IDM3_État_C (3.67Hz)",
        "IDM_Modulation_4X_Sortie", "Amp_Rotation_Sortie", "Amp_État_B_4X", "Amp_État_C_Dernier", "Etotal"
    ]
    st.dataframe(resultats_df.sort_values("IDM3_État_B (3.20Hz)", ascending=False)[cols_order], use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # DETAIL PAR MACHINE (ONGLETS PICS & HARMONIQUES)
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Cinématique Interne & Identification des Harmoniques")
    tabs = st.tabs([r["Ensemble"] for _, r in resultats_df.iterrows()])

    for tab, (_, row) in zip(tabs, resultats_df.iterrows()):
        with tab:
            f_name = row["Ensemble"]
            ident = calage_data[f_name]

            # Sous-indicateurs d'énergie
            e1, e2, e3 = st.columns(3)
            e1.metric("Énergie Basse Fréquence [0–5 Hz]", f"{row['E0_5']:.4f} V²")
            e2.metric("Énergie Moyenne Fréquence [10–20 Hz]", f"{row['E10_20']:.4f} V²")
            e3.metric("Entropie du Spectre (Bruit)", f"{row['Entropie']:.3f}")

            # Structure du tableau harmonique
            rows_table = []
            for nom_elem, res in ident.items():
                harm_str = ", ".join([f"×{h['ordre']}@{h['f_trouvee']:.2f}Hz" for h in res["harmoniques"]]) or "—"
                rows_table.append({
                    "Composante":               nom_elem,
                    "f cible Verrouillée (Hz)": f"{res['f_theorique']:.5f}",
                    "f Réelle Détectée (Hz)":   f"{res['f_trouvee']:.5f}",
                    "Amplitude Pic (V)":        f"{res['amplitude']:.4f}",
                    "Écart Relatif (%)":        f"{res['ecart_pct']:.3f} %",
                    "Score Ancrage":            conf_badge(res["confiance"], res["found"], res["fixe"]),
                    "Harmoniques Validées":     harm_str
                })
            st.dataframe(pd.DataFrame(rows_table), use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # GRAPHES SPECTRES INTERACTIFS
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Visualisation des Spectres avec Lignes de Référence Multi-Étages")
    selected_machine = st.selectbox("Sélectionner l'ensemble à visualiser :", resultats_df["Ensemble"].tolist())

    if selected_machine in fft_data and selected_machine in calage_data:
        freq, amp = fft_data[selected_machine]
        ident     = calage_data[selected_machine]

        plot_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
        fig = px.line(plot_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT complet — {selected_machine}")
        fig.update_xaxes(range=[0, 16])  # Focus utile pour nos composantes

        for nom_elem, res in ident.items():
            f_pos = res["f_theorique"]
            if f_pos > 16: continue
            col = COLORS_MAP.get(nom_elem, "#888888")
            fig.add_vline(
                x=f_pos, 
                line_dash="solid" if res["found"] else "dot", 
                line_color=col, 
                annotation_text=nom_elem, 
                annotation_font_color=col
            )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # ANALYSE DE PREMIER NIVEAU ET ALERTES
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Diagnostics de Premier Niveau Générés par l'IA")
    
    for _, r in resultats_df.iterrows():
        alerts = []
        name = r["Ensemble"]

        if r["Amp_État_B_4X"] > 0.15:
            alerts.append(f"🔴 **Alerte Premier Étage (3.20 Hz) :** Énergie critique détectée sur l'État Cible B. Suspicion forte de matage des dentures d'entrée.")
        if r["Amp_État_C_Dernier"] > 0.15:
            alerts.append(f"🔴 **Alerte Dernier Étage (3.67 Hz) :** Augmentation critique de l'engrènement de sortie. Risque d'usure sous fort couple ou de défaut de pitting.")
        if r["IDM_Modulation_4X_Sortie"] > 0.05:
            alerts.append(f"🟡 **Alerte Modulation Croisée :** Fort couplage entre l'État B et l'arbre de sortie. Risque d'excentricité, de voilage ou de faux-rond sur le grand pignon.")

        if alerts:
            with st.expander(f"⚠️ {name} — {len(alerts)} anomalie(s) détectée(s)"):
                for a in alerts: st.write(a)
        else:
            st.success(f"✅ {name} — Comportement vibratoire nominal sur l'ensemble des étages verrouillés.")

    # --------------------------------------------------
    # LEXIQUE PÉDAGOGIQUE RÉINTÉGRÉ
    # --------------------------------------------------
    st.markdown("---")
    with st.expander("💡 Lexique Mécanique — Rôle Physique des Fréquences Surveillées"):
        st.markdown("""
        | Composante Cinématique | Fréquence Cible ($f_{\text{sortie}} = 0.016\text{ Hz}$) | Symptômes Mécaniques en cas de Hausse du Pic |
        | :--- | :--- | :--- |
        | **Rotation Sortie** | $0,01600\text{ Hz}$ | Balourd, désalignement ou excentricité sur l'arbre lent (récepteur). |
        | **Défilement Courroie** | $0,00635\text{ Hz}$ | Défaut d'aspect sur la courroie, hernie locale ou perte de tension. |
        | **Rotation Poulie Primaire** | $0,05333\text{ Hz}$ | Défaut de fixation, usure de clavette ou faux-rond de la poulie intermédiaire. |
        | **Engrènement 1er Étage** | $0,80000\text{ Hz}$ | Usure normale ou manque de lubrification sur le premier train de pignons. |
        | 🎯 **Harmonique 4X (État B)** | **$3,20000\text{ Hz}$** | **Matage sévère, défaut d'engrènement ou choc cyclique en entrée (Prioritaire).** |
        | 🎯 **Engrènement Sortie (État C)** | **$3,664\text{ Hz} \pm \text{tol}$** | **Usure par fatigue (pitting) ou surcharge de couple sur le dernier engrenage.** |
        | **Rotation Moteur** | $13,12000\text{ Hz}$ | Balourd du rotor moteur, défaut électrique ou désalignement de l'accouplement rapide. |
        """)
