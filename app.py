import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT — Calage Automatique",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Calage Automatique par Ancrage Sortie")

# --------------------------------------------------
# FONCTIONS CINÉMATIQUE
# --------------------------------------------------
def build_chain_from_sortie(f_sortie, ratio_reducteur, dents_primaire, dents_secondaire, dents_courroie):
    """
    Remonte toute la chaîne cinématique depuis la fréquence de sortie mesurée.
    Variable d'ancrage : f_sortie (la plus robuste, ~1 rpm).
    """
    f_poulie_prim = f_sortie * (dents_secondaire / dents_primaire)
    f_moteur      = f_poulie_prim * ratio_reducteur
    f_engrenement = f_poulie_prim * dents_primaire
    f_courroie    = f_engrenement / dents_courroie
    f_4x          = f_engrenement * 4.0

    return {
        "Rotation Sortie":          f_sortie,
        "Rotation Poulie Primaire": f_poulie_prim,
        "Engrènement":              f_engrenement,
        "Harmonique Engrènement 4X": f_4x,
        "Défilement Courroie":      f_courroie,
        "Rotation Moteur":          f_moteur,
    }

def sortie_theorique(ratio_reducteur, dents_primaire, dents_secondaire):
    """Fréquence de sortie théorique pour 1 tour/min moteur de référence."""
    return (1.0 / 60.0) / ratio_reducteur * (dents_primaire / dents_secondaire)

# --------------------------------------------------
# FONCTIONS FFT
# --------------------------------------------------
def calcul_fft(df):
    """
    FFT standard avec fenêtre de Hanning.
    fs = 50 Hz (échantillonnage 20 ms), résolution = 1/T.
    """
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

def calcul_fft_zeropad_anchor(df, zone_max_hz=0.025, pad_factor=32):
    """
    FFT zéro-paddée x pad_factor pour la zone d'ancrage uniquement.
    Permet d'interpoler artificiellement le spectre pour trouver le sommet du pic lent.
    """
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)
    x = x - np.mean(x)
    N = len(x)
    dt = float(np.mean(np.diff(t)))

    fenetre = np.hanning(N)
    x_fen = x * fenetre

    N_pad = N * pad_factor
    x_pad = np.zeros(N_pad)
    x_pad[:N] = x_fen

    fft_pad = np.abs(rfft(x_pad)) * (2.0 / np.sum(fenetre))
    freq_pad = rfftfreq(N_pad, d=dt)

    mask = freq_pad <= (zone_max_hz * 1.5)
    return freq_pad[mask], fft_pad[mask]

def signal_info(df):
    """Retourne les métriques clés du signal pour diagnostic de résolution."""
    N = len(df)
    dt = 0.020  # 20 ms fixe
    T = N * dt
    fs = 1.0 / dt
    resolution_brute = 1.0 / T
    f_sortie_theorique = 1.0 / 60.0  # 1 rpm en Hz
    bins_par_tour = f_sortie_theorique / resolution_brute
    return {
        "N": N,
        "dt_ms": dt * 1000,
        "T_s": T,
        "fs_hz": fs,
        "resolution_hz": resolution_brute,
        "bins_pour_sortie": bins_par_tour,
    }

# --------------------------------------------------
# CALAGE AUTOMATIQUE
# --------------------------------------------------
def find_anchor_peak(freq, amp, zone_max_hz=0.025):
    """Cherche le pic dominant dans la zone basse fréquence."""
    mask = (freq >= 0.003) & (freq <= zone_max_hz)
    if not np.any(mask):
        return None, None
    sub_amp = amp[mask]
    sub_freq = freq[mask]
    idx_max = np.argmax(sub_amp)
    return float(sub_freq[idx_max]), float(sub_amp[idx_max])

def find_peak_near(freq, amp, target_hz, tolerance_pct=0.02):
    """Cherche le pic le plus fort dans la bande de tolérance autour de la cible."""
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
    """Score composite : 60% précision fréquentielle + 40% amplitude relative."""
    if amp_max <= 0:
        return 0
    ecart_score = max(0.0, 1.0 - (ecart_pct / 2.0))
    amp_score   = min(1.0, amplitude / amp_max)
    return int((0.6 * ecart_score + 0.4 * amp_score) * 100)

def auto_calibrate(freq, amp, machine_cfg, tolerance_pct=0.02, anchor_zone_hz=0.05, n_harmonics=4):
    """Calage automatique complet pour une machine depuis l'ancrage sortie."""
    amp_max = np.max(amp)

    f_anchor, a_anchor = find_anchor_peak(freq, amp, zone_max_hz=anchor_zone_hz)
    if f_anchor is None:
        return {"error": f"Aucun pic trouvé dans la zone d'ancrage [0–{anchor_zone_hz} Hz]"}

    chain = build_chain_from_sortie(
        f_anchor,
        machine_cfg["ratio_reducteur"],
        machine_cfg["dents_primaire"],
        machine_cfg["dents_secondaire"],
        machine_cfg["dents_courroie"],
    )

    identification = {}
    for nom, f_theoric in chain.items():
        f_found, a_found, found = find_peak_near(freq, amp, f_theoric, tolerance_pct)
        ecart_pct = abs(f_found - f_theoric) / f_theoric * 100 if found else None
        conf = confidence_score(ecart_pct, a_found, amp_max) if found else 0

        # Harmoniques
        harmoniques = []
        for h in range(2, n_harmonics + 1):
            fh = f_theoric * h
            fh_found, ah_found, hfound = find_peak_near(freq, amp, fh, tolerance_pct)
            if hfound:
                harmoniques.append({"ordre": h, "f_theorique": fh, "f_trouvee": fh_found, "amplitude": ah_found})

        identification[nom] = {
            "f_theorique":  f_theoric,
            "f_trouvee":    f_found,
            "amplitude":    a_found,
            "found":        found,
            "ecart_pct":    ecart_pct,
            "confiance":    conf,
            "harmoniques":  harmoniques,
            "ancre":        nom == "Rotation Sortie",
        }

    return {
        "ancre_hz":      f_anchor,
        "ancre_rpm":     f_anchor * 60.0,
        "ancre_amp":     a_anchor,
        "identification": identification,
    }

# --------------------------------------------------
# INDICATEURS MATHÉMATIQUES
# --------------------------------------------------
def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin, fmax = cible - tolerance, cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask):
        return float(np.max(amp[mask]))
    return float(amp[np.argmin(np.abs(freq - cible))])

def matrix_energie_totale(amp):
    return float(np.sum(amp**2))

def energie_bande(freq, amp, fmin, fmax):
    return float(np.sum(amp[(freq >= fmin) & (freq <= fmax)]**2))

def entropie_spectrale(amp):
    p = amp**2
    if np.sum(p) == 0:
        return 0.0
    p = p / np.sum(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def calcul_indicateurs(freq, amp, f_cible):
    A_cible = amplitude_bande_max(freq, amp, f_cible, tolerance=0.1)
    Etotal   = matrix_energie_totale(amp)
    H        = entropie_spectrale(amp)
    E05      = energie_bande(freq, amp, 0, 5)
    E1020    = energie_bande(freq, amp, 10, 20)
    valeur_brute = (A_cible**2 / Etotal) * H if Etotal > 0 else 0.0
    IDM3 = 5.0 - valeur_brute
    return {
        "Amp Cible (Bande)": A_cible,
        "Etotal": Etotal,
        "Entropie": H,
        "E0_5": E05,
        "E10_20": E1020,
        "IDM3": IDM3,
    }

# --------------------------------------------------
# SESSION STATE (MÉMOIRE PARAMÈTRES)
# --------------------------------------------------
if "machines" not in st.session_state:
    st.session_state.machines = [
        {"nom": "ASM21A", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
        {"nom": "ASM21B", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
    ]

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Paramètres du calage automatique")

anchor_zone = st.sidebar.slider(
    "Zone d'ancrage sortie (Hz max) :",
    min_value=0.005, max_value=0.10, value=0.025, step=0.001,
    help="Plage basse fréquence où chercher le pic de sortie."
)
st.sidebar.caption(f"Zone active : 0.001 – {anchor_zone:.3f} Hz  |  soit 0.06 – {anchor_zone*60:.2f} rpm")

tolerance_pct = st.sidebar.slider(
    "Tolérance identification (%) :",
    min_value=0.5, max_value=5.0, value=2.0, step=0.5
) / 100.0

n_harmonics = st.sidebar.slider("Harmoniques à chercher :", 1, 8, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cinématiques des machines")

with st.sidebar.expander("Configurer les machines", expanded=False):
    nb_machines = st.number_input("Nombre de machines :", min_value=1, max_value=10, value=len(st.session_state.machines), step=1)

    while len(st.session_state.machines) < nb_machines:
        st.session_state.machines.append(
            {"nom": f"Machine{len(st.session_state.machines)+1}", "ratio_reducteur": 246,
             "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126}
        )
    st.session_state.machines = st.session_state.machines[:nb_machines]

    for i, m in enumerate(st.session_state.machines):
        st.markdown(f"**Machine {i+1}**")
        m["nom"]               = st.text_input("Nom",                    value=m["nom"],               key=f"nom_{i}")
        m["ratio_reducteur"]   = st.number_input("Rapport réducteur (1:N)", value=m["ratio_reducteur"],   min_value=1, step=1,   key=f"rr_{i}")
        m["dents_primaire"]    = st.number_input("Dents poulie primaire",   value=m["dents_primaire"],    min_value=1, step=1,   key=f"dp_{i}")
        m["dents_secondaire"]  = st.number_input("Dents poulie secondaire", value=m["dents_secondaire"],  min_value=1, step=1,   key=f"ds_{i}")
        m["dents_courroie"]    = st.number_input("Dents courroie",          value=m["dents_courroie"],    min_value=1, step=1,   key=f"dc_{i}")
        f_th = sortie_theorique(m["ratio_reducteur"], m["dents_primaire"], m["dents_secondaire"])
        st.caption(f"Sortie théorique : {f_th:.5f} Hz ({f_th*60:.4f} rpm)")
        if i < nb_machines - 1: st.markdown("---")

st.sidebar.markdown("---")
notes_text = st.sidebar.text_area(
    "📝 Scores de défaut réels (Feedback terrain) :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67", height=100
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE & GRAPHICS DICTIONARIES
# --------------------------------------------------
COLORS_MAP = {
    "Rotation Sortie":          "#1D9E75",
    "Rotation Poulie Primaire":  "#7F77DD",
    "Engrènement":               "#BA7517",
    "Harmonique Engrènement 4X": "#E24B4A",
    "Défilement Courroie":       "#185FA5",
    "Rotation Moteur":           "#D85A30",
}

def conf_badge(conf, found):
    if not found: return "⚪ Non trouvé"
    if conf >= 70: return f"🟢 {conf}%"
    if conf >= 40: return f"🟡 {conf}%"
    return f"🔴 {conf}%"

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data, fft_zp_data, calage_data = {}, {}, {}

    # Parsing notes terrain
    notes = {}
    for ligne in notes_text.splitlines():
        if "=" in ligne:
            nom, val = ligne.split("=", 1)
            try: notes[nom.strip()] = float(val.strip())
            except ValueError: pass

    # ── Panneau diagnostic résolution initiale ──
    first_valid_df = None
    for feuille in xls.sheet_names:
        try:
            _df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if {"ms", "V"}.issubset(_df.columns):
                first_valid_df = _df
                break
        except Exception: pass

    if first_valid_df is not None:
        info = signal_info(first_valid_df)
        st.subheader("📡 Diagnostic de résolution spectrale")
        ri1, ri2, ri3, ri4 = st.columns(4)
        ri1.metric("Points / acquisition", f"{info['N']:,}")
        ri2.metric("Durée signal", f"{info['T_s']:.0f} s  ({info['T_s']/60:.1f} min)")
        ri3.metric("Résolution brute FFT", f"{info['resolution_hz']:.5f} Hz")
        ri4.metric("Bins couvrant 1 rpm (0.0167 Hz)", f"{info['bins_pour_sortie']:.1f}")

        if info["bins_pour_sortie"] < 3:
            st.warning(f"⚠️ Résolution brute insuffisante pour isoler le pic à 0.0167 Hz. **Zero-padding x32 activé**.")
        else:
            st.success(f"✅ Résolution physique suffisante ({info['bins_pour_sortie']:.1f} bins autour de 0.0167 Hz)")
        st.markdown("---")

    # Boucle sur les onglets du fichier Excel
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): continue

            # Spectres FFT Standard & Zéro-Paddé
            freq, amp = calcul_fft(df)
            fft_data[feuille] = (freq, amp)

            freq_zp, amp_zp = calcul_fft_zeropad_anchor(df, zone_max_hz=anchor_zone, pad_factor=32)
            fft_zp_data[feuille] = (freq_zp, amp_zp)

            # Identification du modèle cinématique cible
            machine_cfg = st.session_state.machines[0]
            for m in st.session_state.machines:
                if m["nom"].lower() in feuille.lower() or feuille.lower() in m["nom"].lower():
                    machine_cfg = m
                    break

            # Calage via ancre sur spectre Zéro-Paddé
            calage = auto_calibrate(freq_zp, amp_zp, machine_cfg, tolerance_pct, anchor_zone, n_harmonics)
            
            if "error" in calage:
                st.sidebar.warning(f"{feuille} : {calage['error']}")
                continue

            # Identification complémentaire sur spectre classique
            f_anchor = calage["ancre_hz"]
            chain = build_chain_from_sortie(f_anchor, machine_cfg["ratio_reducteur"], machine_cfg["dents_primaire"], machine_cfg["dents_secondaire"], machine_cfg["dents_courroie"])
            amp_max = float(np.max(amp))

            for nom, f_th in chain.items():
                if nom == "Rotation Sortie": continue
                f_found, a_found, found = find_peak_near(freq, amp, f_th, tolerance_pct)
                ecart_pct = abs(f_found - f_th) / f_th * 100 if found else None
                conf = confidence_score(ecart_pct, a_found, amp_max) if found else 0
                
                harmoniques = []
                for h in range(2, n_harmonics + 1):
                    fh = f_th * h
                    fh_f, ah_f, hf = find_peak_near(freq, amp, fh, tolerance_pct)
                    if hf: harmoniques.append({"ordre": h, "f_theorique": fh, "f_trouvee": fh_f, "amplitude": ah_f})
                
                calage["identification"][nom] = {
                    "f_theorique": f_th, "f_trouvee": f_found, "amplitude": a_found,
                    "found": found, "ecart_pct": ecart_pct, "confiance": conf, "harmoniques": harmoniques, "ancre": False,
                }

            calage_data[feuille] = calage
            ident = calage["identification"]

            # CORRECTION : Calcul des indicateurs focalisé sur la fréquence 4X trouvée (ou théorique)
            h4x_info = ident["Harmonique Engrènement 4X"]
            f_cible_idm3 = h4x_info["f_trouvee"] if h4x_info["found"] else h4x_info["f_theorique"]

            indic = calcul_indicateurs(freq, amp, f_cible_idm3)
            indic["Ensemble"]    = feuille
            indic["Defaut_Réel"] = notes.get(feuille, np.nan)

            for nom_elem, res_elem in ident.items():
                indic[f"Amp_{nom_elem}"] = res_elem["amplitude"]

            indic["Ancre_Hz"]  = calage["ancre_hz"]
            indic["Ancre_RPM"] = calage["ancre_rpm"]

            # Sécurisation des valeurs d'amplitude pour les calculs de modulations
            amp_moteur = ident["Rotation Moteur"]["amplitude"] if ident["Rotation Moteur"]["found"] else 0.0
            amp_sortie = ident["Rotation Sortie"]["amplitude"] if ident["Rotation Sortie"]["found"] else 0.0
            amp_4x     = ident["Harmonique Engrènement 4X"]["amplitude"] if ident["Harmonique Engrènement 4X"]["found"] else 0.0

            indic["ID_Modulation"]     = amp_moteur * amp_sortie
            indic["IDM_Modulation_4X"] = amp_sortie * amp_4x

            resultats.append(indic)
        except Exception as e:
            st.sidebar.error(f"Erreur traitement onglet {feuille} : {e}")

    if len(resultats) == 0:
        st.warning("Aucune donnée exploitable extraite. Vérifiez la forme de vos onglets Excel.")
        st.stop()

    resultats_df = pd.DataFrame(resultats)

    # --------------------------------------------------
    # DISPLAY TABS : COMPONENT IDENTIFICATION
    # --------------------------------------------------
    st.subheader("🎯 Résultats du calage automatique par machine")
    tabs = st.tabs([r["Ensemble"] for _, r in resultats_df.iterrows()])

    for tab, (_, row) in zip(tabs, resultats_df.iterrows()):
        with tab:
            feuille = row["Ensemble"]
            calage  = calage_data.get(feuille, {})
            if "error" in calage:
                st.error(calage["error"])
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ancre Sortie Détectée", f"{calage['ancre_hz']:.5f} Hz")
            c2.metric("Vitesse de Sortie Réelle", f"{calage['ancre_rpm']:.4f} rpm")
            c3.metric("Amplitude Ancre", f"{calage['ancre_amp']:.4f} V")
            identified = sum(1 for v in calage["identification"].values() if v["found"])
            c4.metric("Composantes Calées", f"{identified}/{len(calage['identification'])}")

            st.markdown("**Tableau d'identification harmonique**")
            rows_ident = []
            for nom_elem, res in calage["identification"].items():
                harm_str = ", ".join([f"×{h['ordre']}@{h['f_trouvee']:.2f}Hz" for h in res["harmoniques"]]) or "—"
                rows_ident.append({
                    "Composante":          nom_elem,
                    "f théorique (Hz)":    f"{res['f_theorique']:.5f}",
                    "f mesurée (Hz)":      f"{res['f_trouvee']:.5f}" if res["found"] else "—",
                    "Amplitude (V)":       f"{res['amplitude']:.4f}" if res["found"] else "—",
                    "Écart (%)":           f"{res['ecart_pct']:.3f}" if res["ecart_pct"] is not None else "—",
                    "Score Confiance":     conf_badge(res["confiance"], res["found"]),
                    "Harmoniques Validées": harm_str
                })
            st.dataframe(pd.DataFrame(rows_ident), hide_index=True, use_container_width=True)

    # --------------------------------------------------
    # SEUILLAGE DYNAMIQUE STATISTIQUE (SIGMA)
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 État de santé général — Seuils Statistiques Dynamiques")
    df_valid = resultats_df.dropna(subset=["Defaut_Réel"])

    if len(df_valid) >= 3:
        X_meca = df_valid["Defaut_Réel"].values
        Y_idm3 = df_valid["IDM3"].values
        m_lin, b_lin = np.polyfit(X_meca, Y_idm3, 1)
        residus = Y_idm3 - (m_lin * X_meca + b_lin)
        sigma = np.std(residus) if np.std(residus) > 0 else 1.0

        statuts = []
        for _, row in resultats_df.iterrows():
            if not pd.isna(row["Defaut_Réel"]):
                val_att = m_lin * row["Defaut_Réel"] + b_lin
                ecart   = row["IDM3"] - val_att
                if ecart <= 1.0 * sigma: statuts.append("🟢 Conforme")
                elif ecart <= 2.0 * sigma: statuts.append("🟡 Écart Modéré")
                else: statuts.append("🔴 Alarme (Hors Tendance)")
            else:
                if row["IDM3"] < 3.5: statuts.append("🟢 Bon (Fixe)")
                elif row["IDM3"] < 4.5: statuts.append("🟡 À surveiller (Fixe)")
                else: statuts.append("🔴 Alarme (Fixe)")
        resultats_df["Statut"] = statuts

        if len(df_valid) >= 2:
            corr_mod4x = df_valid["IDM_Modulation_4X"].corr(df_valid["Defaut_Réel"])
            corr_idm3  = df_valid["IDM3"].corr(df_valid["Defaut_Réel"])
            corr_mod   = df_valid["ID_Modulation"].corr(df_valid["Defaut_Réel"])
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Corrélation Mod. 4X (État B)", f"{corr_mod4x:.3f}")
            cc2.metric("Corrélation IDM3 (Énergie Globale)", f"{corr_idm3:.3f}", delta="Modèle σ Actif")
            cc3.metric("Corrélation Mod. Moteur", f"{corr_mod:.3f}")
    else:
        resultats_df["Statut"] = resultats_df["IDM3"].apply(lambda x: "🟢 Bon" if x < 3.5 else "🟡 À surveiller" if x < 4.5 else "🔴 Alarme")

    colonnes_affichage = [
        "Ensemble", "Statut", "Defaut_Réel", "Ancre_Hz", "Ancre_RPM", "IDM3", "IDM_Modulation_4X", "ID_Modulation",
        "Amp_Rotation Sortie", "Amp_Harmonique Engrènement 4X", "Amp_Rotation Moteur"
    ]
    colonnes_ok = [c for c in colonnes_affichage if c in resultats_df.columns]
    st.dataframe(resultats_df.sort_values("IDM3", ascending=False)[colonnes_ok], use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # PLOTLY GRAPHICS : SPECTRES ET ZOOM ANCRAGE
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Visualisation des spectres FFT")
    ensemble = st.selectbox("Sélectionner une machine à analyser :", resultats_df.sort_values("IDM3", ascending=False)["Ensemble"].tolist())

    if ensemble in fft_data and ensemble in calage_data:
        freq, amp = fft_data[ensemble]
        calage    = calage_data[ensemble]

        if "error" not in calage:
            # Graphes 1 : Spectre global 0-20 Hz
            fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
            fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT complet — {ensemble}")
            fig.update_xaxes(range=[0, 20])

            for nom_elem, res in calage["identification"].items():
                f_val = res["f_trouvee"] if res["found"] else res["f_theorique"]
                if f_val > 20: continue
                col = COLORS_MAP.get(nom_elem, "#888888")
                fig.add_vline(x=f_val, line_dash="solid" if res["found"] else "dot", line_color=col, annotation_text=nom_elem, annotation_font_color=col)
            st.plotly_chart(fig, use_container_width=True)

            # Graphes 2 : Zoom Zéro-Paddé zone d'ancrage
            st.markdown("**Focus Zone Basse Fréquence (Zéro-padding ×32 / Interpolation spectrale à ~0.0001 Hz)**")
            if ensemble in fft_zp_data:
                freq_zp, amp_zp = fft_zp_data[ensemble]
                f_anc = calage["ancre_hz"]
                zp_df = pd.DataFrame({"Fréquence (Hz)": freq_zp, "Amplitude": amp_zp})
                fig_zp = px.line(zp_df, x="Fréquence (Hz)", y="Amplitude", title=f"Zone d'Ancrage de l'Arbre Lent — {ensemble}", color_discrete_sequence=["#1D9E75"])
                fig_zp.update_xaxes(range=[0.003, anchor_zone])
                fig_zp.add_vline(x=f_anc, line_color="#1D9E75", line_width=2.5, annotation_text=f"Ancre Réelle: {f_anc:.5f} Hz")
                st.plotly_chart(fig_zp, use_container_width=True)

    # --------------------------------------------------
    # DIAGNOSTICS DE PREMIER NIVEAU AUTOMATIQUES
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Diagnostics Automatiques de l'Installation")

    for f_name, c_data in calage_data.items():
        if "error" in c_data: continue
        id_c = c_data["identification"]
        alerts = []

        if id_c.get("Harmonique Engrènement 4X", {}).get("amplitude", 0) > 0.15:
            alerts.append("🔴 **Harmonique 4X (État Cible B) critique :** Risque sévère de matage des dentures ou désalignement marqué.")
        if id_c.get("Défilement Courroie", {}).get("amplitude", 0) > 0.05:
            alerts.append("🟡 **Activité Courroie :** Énergie détectée sur le défilement. Inspecter une hernie ou une perte de tension.")
        if id_c.get("Rotation Sortie", {}).get("confiance", 100) < 60:
            alerts.append("🟡 **Ancrage Incertain :** Le pic de l'arbre lent manque d'émergence. Rallongez le temps de mesure si possible.")

        if alerts:
            with st.expander(f"⚠️ {f_name} — {len(alerts)} alerte(s) détectée(s)"):
                for a in alerts: st.write(a)
        else:
            st.success(f"✅ {f_name} — Aucun défaut mécanique flagrant identifié.")

    # --------------------------------------------------
    # LEXIQUE PÉDAGOGIQUE COMPACT
    # --------------------------------------------------
    with st.expander("💡 Aide à l'interprétation — Signification Physique des Composantes"):
        st.markdown("""
        | Composante Cinématique | Origine Mécanique dans la Machine | Causes d'Émergence d'un Pic Vibratoire |
        | :--- | :--- | :--- |
        | **Rotation Sortie** | Vitesse de l'arbre lent (~1 rpm). **Point d'ancrage de l'application.** | Balourd sur le récepteur ou l'organe entraîné en bout de ligne. |
        | **Poulie Primaire** | Arbre intermédiaire du premier étage de réduction. | Fixation lâche ou défaut d'excentricité de la poulie. |
        | **Engrènement** | Fréquence de choc naturelle du contact dent contre dent. | Usure normale ou manque de lubrification des flancs de denture. |
        | **Harmonique Engrènement 4X** | Point de contrôle de l'**État Cible B**. | **Matage, usure sévère des engrenages ou contrainte géométrique.** |
        | **Défilement Courroie** | Cycle complet de la courroie de transmission. | Présence d'une fêlure, hernie localisée ou mauvaise tension. |
        | **Rotation Moteur** | Vitesse de rotation de l'arbre d'entrée électrique. | Problème d'alignement d'accouplement ou balourd moteur standard. |
        """)
