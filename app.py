import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
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
    FFT standard avec fenetre de Hanning.
    fs = 50 Hz (echantillonnage 20 ms), resolution = 1/T.
    Pour 15 000 pts : resolution brute = 1/300 s = 0.0033 Hz.
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
    FFT zero-paddee x pad_factor pour la zone d'ancrage uniquement.

    Contexte : fs=50 Hz, T~300 s -> resolution brute=0.0033 Hz.
    Le pic de sortie a 0.0167 Hz est couvert par 1-2 bins seulement.
    Le zero-padding x32 interpole le spectre a ~0.0001 Hz de resolution
    apparente, permettant de lire la frequence du pic avec une precision
    de +/-0.0002 Hz (+/-1.2% a 0.0167 Hz).

    Note : ce n'est pas de la vraie resolution physique (limitee par T),
    mais une interpolation spectrale suffisante pour l'ancrage.
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

    # Retourne uniquement la zone d'ancrage pour economiser la memoire
    mask = freq_pad <= (zone_max_hz * 1.5)
    return freq_pad[mask], fft_pad[mask]

def signal_info(df):
    """Retourne les metriques cles du signal pour diagnostic de resolution."""
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
    """
    Cherche le pic dominant dans la zone [0.003, zone_max_hz].
    1 rpm = 0.01667 Hz.
    Borne basse a 0.003 Hz pour eviter le DC residuel.
    Utiliser le spectre zero-padde pour la precision.
    """
    mask = (freq >= 0.003) & (freq <= zone_max_hz)
    if not np.any(mask):
        return None, None
    sub_amp = amp[mask]
    sub_freq = freq[mask]
    idx_max = np.argmax(sub_amp)
    return float(sub_freq[idx_max]), float(sub_amp[idx_max])

def find_peak_near(freq, amp, target_hz, tolerance_pct=0.02):
    """
    Cherche le pic le plus fort dans la bande ± tolerance_pct autour de target_hz.
    Retourne (fréquence trouvée, amplitude, trouvé).
    """
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
    """
    Calage automatique complet pour une machine.
    1. Ancrage sur le pic dominant en zone basse fréquence (sortie ~1 rpm)
    2. Remontée de la chaîne cinématique par les rapports
    3. Identification de chaque composante dans le spectre réel
    Retourne un dict de résultats par composante.
    """
    amp_max = np.max(amp)

    # Étape 1 : ancrage
    f_anchor, a_anchor = find_anchor_peak(freq, amp, zone_max_hz=anchor_zone_hz)
    if f_anchor is None:
        return {"error": f"Aucun pic trouvé dans la zone d'ancrage [0–{anchor_zone_hz} Hz]"}

    # Étape 2 : chaîne cinématique depuis l'ancre
    chain = build_chain_from_sortie(
        f_anchor,
        machine_cfg["ratio_reducteur"],
        machine_cfg["dents_primaire"],
        machine_cfg["dents_secondaire"],
        machine_cfg["dents_courroie"],
    )

    # Étape 3 : identification dans le spectre
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
# INDICATEURS (inchangés + IDM3 recalculé sur ancre réelle)
# --------------------------------------------------
def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin, fmax = cible - tolerance, cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask):
        return float(np.max(amp[mask]))
    return float(amp[np.argmin(np.abs(freq - cible))])

def energie_totale(amp):
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
    Etotal   = energie_totale(amp)
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
# SESSION STATE
# --------------------------------------------------
if "machines" not in st.session_state:
    st.session_state.machines = [
        {"nom": "ASM21A", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
        {"nom": "ASM21B", "ratio_reducteur": 246, "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126},
    ]

# --------------------------------------------------
# BARRE LATÉRALE
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")

# --- Fichier ---
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")

# --- Paramètres de calage automatique ---
st.sidebar.subheader("🎯 Paramètres du calage automatique")

anchor_zone = st.sidebar.slider(
    "Zone d'ancrage sortie (Hz max) :",
    min_value=0.005, max_value=0.10, value=0.025, step=0.001,
    help="Plage basse fréquence où chercher le pic de sortie. 1 rpm = 0.0167 Hz — la zone couvre [0.001, valeur] Hz"
)
st.sidebar.caption(f"Zone active : 0.001 – {anchor_zone:.3f} Hz  |  soit 0.06 – {anchor_zone*60:.2f} rpm")
tolerance_pct = st.sidebar.slider(
    "Tolérance identification (%) :",
    min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    help="Écart max entre fréquence théorique et pic mesuré"
) / 100.0

n_harmonics = st.sidebar.slider("Harmoniques à chercher :", 1, 8, 4)

st.sidebar.markdown("---")

# --- Cinématiques ---
st.sidebar.subheader("⚙️ Cinématiques des machines")

with st.sidebar.expander("Configurer les machines", expanded=True):
    nb_machines = st.number_input("Nombre de machines :", min_value=1, max_value=10, value=len(st.session_state.machines), step=1)

    while len(st.session_state.machines) < nb_machines:
        st.session_state.machines.append(
            {"nom": f"Machine{len(st.session_state.machines)+1}", "ratio_reducteur": 246,
             "dents_primaire": 15, "dents_secondaire": 50, "dents_courroie": 126}
        )
    st.session_state.machines = st.session_state.machines[:nb_machines]

    for i, m in enumerate(st.session_state.machines):
        st.markdown(f"**Machine {i+1}**")
        m["nom"]               = st.text_input("Nom",                   value=m["nom"],               key=f"nom_{i}")
        m["ratio_reducteur"]   = st.number_input("Rapport réducteur (1:N)", value=m["ratio_reducteur"],   min_value=1,  step=1,   key=f"rr_{i}")
        m["dents_primaire"]    = st.number_input("Dents poulie primaire",   value=m["dents_primaire"],    min_value=1,  step=1,   key=f"dp_{i}")
        m["dents_secondaire"]  = st.number_input("Dents poulie secondaire", value=m["dents_secondaire"],  min_value=1,  step=1,   key=f"ds_{i}")
        m["dents_courroie"]    = st.number_input("Dents courroie",          value=m["dents_courroie"],    min_value=1,  step=1,   key=f"dc_{i}")
        f_th = sortie_theorique(m["ratio_reducteur"], m["dents_primaire"], m["dents_secondaire"])
        st.caption(f"Sortie théorique : {f_th:.5f} Hz ({f_th*60:.4f} rpm)")
        if i < nb_machines - 1:
            st.markdown("---")

st.sidebar.markdown("---")

# --- Feedback terrain ---
notes_text = st.sidebar.text_area(
    "📝 Scores de défaut réels (Feedback terrain) :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67",
    height=100
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
COLORS_MAP = {
    "Rotation Sortie":           "#1D9E75",
    "Rotation Poulie Primaire":  "#7F77DD",
    "Engrènement":               "#BA7517",
    "Harmonique Engrènement 4X": "#E24B4A",
    "Défilement Courroie":       "#185FA5",
    "Rotation Moteur":           "#D85A30",
}

BADGE_CONF = {
    "high": "🟢",
    "mid":  "🟡",
    "low":  "🔴",
    "none": "⚪",
}

def conf_badge(conf, found):
    if not found:
        return "⚪ Non trouvé"
    if conf >= 70:
        return f"🟢 {conf}%"
    if conf >= 40:
        return f"🟡 {conf}%"
    return f"🔴 {conf}%"

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data = {}
    fft_zp_data = {}   # spectres zero-paddes (zone ancrage)
    calage_data = {}
    sig_info_data = {}

    # Parsing notes terrain
    notes = {}
    for ligne in notes_text.splitlines():
        if "=" in ligne:
            nom, val = ligne.split("=", 1)
            try:
                notes[nom.strip()] = float(val.strip())
            except ValueError:
                pass

    # ── Panneau diagnostic resolution (premiere feuille valide) ──
    first_valid_df = None
    for feuille in xls.sheet_names:
        try:
            _df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if {"ms", "V"}.issubset(_df.columns):
                first_valid_df = _df
                break
        except Exception:
            pass

    if first_valid_df is not None:
        info = signal_info(first_valid_df)
        st.subheader("📡 Diagnostic de resolution spectrale")
        ri1, ri2, ri3, ri4 = st.columns(4)
        ri1.metric("Points / acquisition", f"{info['N']:,}")
        ri2.metric("Duree signal", f"{info['T_s']:.0f} s  ({info['T_s']/60:.1f} min)")
        ri3.metric("Resolution brute FFT", f"{info['resolution_hz']:.5f} Hz")
        ri4.metric("Bins couvrant 1 rpm (0.0167 Hz)", f"{info['bins_pour_sortie']:.1f}")

        if info["bins_pour_sortie"] < 3:
            st.warning(
                f"⚠️ Resolution brute ({info['resolution_hz']:.4f} Hz) insuffisante pour isoler le pic a 0.0167 Hz "
                f"({info['bins_pour_sortie']:.1f} bins). "
                f"**Zero-padding x32 active** : resolution interpolee = "
                f"{info['resolution_hz']/32:.5f} Hz → precision ancrage ≈ ±{info['resolution_hz']/32/0.0167*100:.1f}%"
            )
        else:
            st.success(f"✅ Resolution suffisante ({info['bins_pour_sortie']:.1f} bins autour de 0.0167 Hz)")
        st.markdown("---")

    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns):
                continue

            # FFT standard (spectre complet pour affichage)
            freq, amp = calcul_fft(df)
            fft_data[feuille] = (freq, amp)

            # FFT zero-paddee x32 (zone ancrage uniquement)
            freq_zp, amp_zp = calcul_fft_zeropad_anchor(df, zone_max_hz=anchor_zone, pad_factor=32)
            fft_zp_data[feuille] = (freq_zp, amp_zp)

            # Trouver la cinematique correspondante
            machine_cfg = st.session_state.machines[0]
            for m in st.session_state.machines:
                if m["nom"].lower() in feuille.lower() or feuille.lower() in m["nom"].lower():
                    machine_cfg = m
                    break

            # ── CALAGE AUTOMATIQUE sur spectre zero-padde pour l'ancrage ──
            calage = auto_calibrate(
                freq_zp, amp_zp, machine_cfg,
                tolerance_pct=tolerance_pct,
                anchor_zone_hz=anchor_zone,
                n_harmonics=n_harmonics,
            )
            # Identification des autres composantes sur le spectre standard
            if "error" not in calage:
                f_anchor = calage["ancre_hz"]
                chain = build_chain_from_sortie(
                    f_anchor,
                    machine_cfg["ratio_reducteur"],
                    machine_cfg["dents_primaire"],
                    machine_cfg["dents_secondaire"],
                    machine_cfg["dents_courroie"],
                )
                amp_max = float(np.max(amp))
                for nom, f_th in chain.items():
                    if nom == "Rotation Sortie":
                        continue  # deja trouve par zero-padding
                    f_found, a_found, found = find_peak_near(freq, amp, f_th, tolerance_pct)
                    ecart_pct = abs(f_found - f_th) / f_th * 100 if found else None
                    conf = confidence_score(ecart_pct, a_found, amp_max) if found else 0
                    harmoniques = []
                    for h in range(2, n_harmonics + 1):
                        fh = f_th * h
                        fh_f, ah_f, hf = find_peak_near(freq, amp, fh, tolerance_pct)
                        if hf:
                            harmoniques.append({"ordre": h, "f_theorique": fh, "f_trouvee": fh_f, "amplitude": ah_f})
                    calage["identification"][nom] = {
                        "f_theorique": f_th,
                        "f_trouvee":   f_found,
                        "amplitude":   a_found,
                        "found":       found,
                        "ecart_pct":   ecart_pct,
                        "confiance":   conf,
                        "harmoniques": harmoniques,
                        "ancre":       False,
                    }

            calage_data[feuille] = calage

            if "error" in calage:
                st.sidebar.warning(f"{feuille} : {calage['error']}")
                continue

            ident = calage["identification"]

            # Indicateurs globaux (ancre réelle)
            f_cible = calage["ancre_hz"]
            indic = calcul_indicateurs(freq, amp, f_cible)
            indic["Ensemble"]    = feuille
            indic["Defaut_Réel"] = notes.get(feuille, np.nan)

            # Amplitudes par composante (fréquences calées automatiquement)
            for nom_elem, res_elem in ident.items():
                indic[f"Amp_{nom_elem}"] = res_elem["amplitude"]

            # Ancre mesurée
            indic["Ancre_Hz"]  = calage["ancre_hz"]
            indic["Ancre_RPM"] = calage["ancre_rpm"]

            # Modulations
            indic["ID_Modulation"]    = ident["Rotation Moteur"]["amplitude"]   * ident["Rotation Sortie"]["amplitude"]
            indic["IDM_Modulation_4X"]= ident["Rotation Sortie"]["amplitude"]   * ident["Harmonique Engrènement 4X"]["amplitude"]

            resultats.append(indic)

        except Exception as e:
            st.sidebar.error(f"Erreur onglet {feuille} : {e}")

    if len(resultats) == 0:
        st.warning("Aucune feuille valide (colonnes 'ms' et 'V' requises).")
        st.stop()

    resultats_df = pd.DataFrame(resultats)

    # --------------------------------------------------
    # TABLEAU DE CALAGE AUTOMATIQUE
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
            c1.metric("Ancre Sortie", f"{calage['ancre_hz']:.5f} Hz")
            c2.metric("Ancre RPM",    f"{calage['ancre_rpm']:.4f} rpm")
            c3.metric("Amplitude ancre", f"{calage['ancre_amp']:.4f} u.")
            identified = sum(1 for v in calage["identification"].values() if v["found"])
            c4.metric("Pics identifiés", f"{identified}/{len(calage['identification'])}")

            st.markdown("**Identification des composantes**")
            rows_ident = []
            for nom_elem, res in calage["identification"].items():
                harm_str = ", ".join([f"×{h['ordre']} @ {h['f_trouvee']:.3f}Hz" for h in res["harmoniques"]]) or "—"
                rows_ident.append({
                    "Composante":          nom_elem,
                    "f théorique (Hz)":    f"{res['f_theorique']:.5f}",
                    "f pic trouvé (Hz)":   f"{res['f_trouvee']:.5f}" if res["found"] else "—",
                    "Amplitude (u.)":      f"{res['amplitude']:.4f}" if res["found"] else "—",
                    "Écart (%)":           f"{res['ecart_pct']:.3f}" if res["ecart_pct"] is not None else "—",
                    "Confiance":           conf_badge(res["confiance"], res["found"]),
                    "Harmoniques trouvées": harm_str,
                })
            st.dataframe(pd.DataFrame(rows_ident), hide_index=True, use_container_width=True)

    # --------------------------------------------------
    # SEUILLAGE DYNAMIQUE (inchangé, sur ancre réelle)
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 État de santé — Seuils Statistiques Dynamiques")

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
                if ecart <= 1.0 * sigma:
                    statuts.append("🟢 Conforme")
                elif ecart <= 2.0 * sigma:
                    statuts.append("🟡 Écart Modéré")
                else:
                    statuts.append("🔴 Alarme (Hors Tendance)")
            else:
                if row["IDM3"] < 3.5:
                    statuts.append("🟢 Bon (Fixe)")
                elif row["IDM3"] < 4.5:
                    statuts.append("🟡 À surveiller (Fixe)")
                else:
                    statuts.append("🔴 Alarme (Fixe)")
        resultats_df["Statut"] = statuts

        if len(df_valid) >= 2:
            corr_mod4x = df_valid["IDM_Modulation_4X"].corr(df_valid["Defaut_Réel"])
            corr_idm3  = df_valid["IDM3"].corr(df_valid["Defaut_Réel"])
            corr_mod   = df_valid["ID_Modulation"].corr(df_valid["Defaut_Réel"])
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Corrélation Mod. 4X",         f"{corr_mod4x:.3f}")
            cc2.metric("Corrélation IDM3 Linéarisé",  f"{corr_idm3:.3f}", delta="Seuils σ actifs")
            cc3.metric("Corrélation Mod. Moteur",      f"{corr_mod:.3f}")
    else:
        resultats_df["Statut"] = resultats_df["IDM3"].apply(
            lambda x: "🟢 Bon" if x < 3.5 else "🟡 À surveiller" if x < 4.5 else "🔴 Alarme"
        )

    colonnes_affichage = [
        "Ensemble", "Statut", "Defaut_Réel", "Ancre_Hz", "Ancre_RPM",
        "IDM3", "IDM_Modulation_4X", "ID_Modulation",
        "Amp_Rotation Sortie", "Amp_Harmonique Engrènement 4X",
        "Amp_Rotation Moteur", "Amp_Engrènement",
        "Amp_Rotation Poulie Primaire", "Amp_Défilement Courroie",
    ]
    colonnes_ok = [c for c in colonnes_affichage if c in resultats_df.columns]
    st.dataframe(
        resultats_df.sort_values("IDM3", ascending=False)[colonnes_ok],
        use_container_width=True, hide_index=True
    )

    # --------------------------------------------------
    # GRAPHIQUE FFT AVEC MARQUEURS CALÉS
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Spectre FFT — Fréquences identifiées automatiquement")

    ensemble = st.selectbox(
        "Sélectionner une machine à analyser :",
        resultats_df.sort_values("IDM3", ascending=False)["Ensemble"].tolist()
    )

    if ensemble in fft_data and ensemble in calage_data:
        freq, amp = fft_data[ensemble]
        calage    = calage_data[ensemble]

        if "error" not in calage:

            # ── Graphique 1 : spectre complet 0–20 Hz ──
            fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
            fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude",
                          title=f"Spectre FFT complet — {ensemble}")
            fig.update_xaxes(range=[0, 20])

            for nom_elem, res in calage["identification"].items():
                f_val = res["f_trouvee"] if res["found"] else res["f_theorique"]
                if f_val > 20:
                    continue
                col = COLORS_MAP.get(nom_elem, "#888888")
                line_dash = "solid" if res["found"] else "dot"
                fig.add_vline(
                    x=f_val, line_dash=line_dash, line_color=col,
                    annotation_text=nom_elem, annotation_font_color=col,
                )
                for h in res["harmoniques"]:
                    if h["f_trouvee"] <= 20:
                        fig.add_vline(
                            x=h["f_trouvee"], line_dash="dash",
                            line_color=col, line_width=0.8,
                            annotation_text=f"x{h['ordre']}",
                            annotation_font_color=col, annotation_font_size=10,
                        )

            fig.add_vline(
                x=calage["ancre_hz"], line_dash="solid",
                line_color="#1D9E75", line_width=2.5,
                annotation_text=f"Ancre {calage['ancre_hz']:.5f}Hz",
                annotation_font_color="#1D9E75",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Graphique 2 : zoom zone ancrage (spectre zero-padde) ──
            st.markdown("**Zoom zone d'ancrage — spectre zero-paddé ×32 (résolution interpolée ~0.0001 Hz)**")
            st.caption(
                f"Resolution brute : {1/300:.5f} Hz  →  Zero-padding ×32  →  "
                f"Resolution interpolee : {1/300/32:.6f} Hz  |  "
                f"Pic ancre : **{calage['ancre_hz']:.6f} Hz  =  {calage['ancre_rpm']:.5f} rpm**"
            )

            if ensemble in fft_zp_data:
                freq_zp, amp_zp = fft_zp_data[ensemble]
                # Zoom sur ±50% autour de l'ancre
                f_anc = calage["ancre_hz"]
                zoom_min = max(0.003, f_anc * 0.3)
                zoom_max = min(anchor_zone, f_anc * 2.5)

                zp_df = pd.DataFrame({"Fréquence (Hz)": freq_zp, "Amplitude": amp_zp})
                fig_zp = px.line(
                    zp_df, x="Fréquence (Hz)", y="Amplitude",
                    title=f"Zone ancrage (zero-padde x32) — {ensemble}",
                    color_discrete_sequence=["#1D9E75"],
                )
                fig_zp.update_xaxes(range=[zoom_min, zoom_max])
                fig_zp.add_vline(
                    x=f_anc, line_dash="solid", line_color="#1D9E75", line_width=2,
                    annotation_text=f"Ancre {f_anc:.6f} Hz ({f_anc*60:.5f} rpm)",
                    annotation_font_color="#1D9E75",
                )
                # Marquer aussi la valeur theorique
                _mcfg = next(
                    (m for m in st.session_state.machines
                     if m["nom"].lower() in ensemble.lower()
                     or ensemble.lower() in m["nom"].lower()),
                    st.session_state.machines[0]
                )
                f_th_sortie = sortie_theorique(
                    _mcfg["ratio_reducteur"],
                    _mcfg["dents_primaire"],
                    _mcfg["dents_secondaire"],
                )
                fig_zp.add_vline(
                    x=f_th_sortie, line_dash="dash", line_color="#BA7517", line_width=1.5,
                    annotation_text=f"Theorique {f_th_sortie:.6f} Hz",
                    annotation_font_color="#BA7517",
                )
                ecart_th = (f_anc - f_th_sortie) / f_th_sortie * 100
                st.plotly_chart(fig_zp, use_container_width=True)
                st.info(
                    f"Ecart pic mesuré / théorique : **{ecart_th:+.3f}%**  "
                    f"({(f_anc - f_th_sortie)*1000:.4f} mHz)  —  "
                    f"Vitesse réelle : **{f_anc*60:.5f} rpm** vs théorique {f_th_sortie*60:.5f} rpm"
                )

            # Tableau résumé des amplitudes lues
            st.write("**Amplitudes lues aux fréquences calées :**")
            cols_f = st.columns(len(calage["identification"]))
            for col_el, (nom_elem, res) in zip(cols_f, calage["identification"].items()):
                col_el.metric(
                    label=f"{nom_elem[:18]}\n({res['f_trouvee']:.3f} Hz)",
                    value=f"{res['amplitude']:.4f} V",
                    delta=f"Conf. {res['confiance']}%" if res["found"] else "Non trouvé",
                )

    # --------------------------------------------------
    # DIAGNOSTICS AUTOMATIQUES
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Diagnostics automatiques")

    for feuille, calage in calage_data.items():
        if "error" in calage:
            continue
        ident = calage["identification"]
        diags = []

        eng4x  = ident.get("Harmonique Engrènement 4X", {})
        moteur = ident.get("Rotation Moteur", {})
        sortie = ident.get("Rotation Sortie", {})
        courr  = ident.get("Défilement Courroie", {})

        if eng4x.get("found") and eng4x.get("amplitude", 0) > 0.15:
            diags.append(("🔴", f"Harmonique 4X élevée ({eng4x['amplitude']:.3f} u.) → risque matage de denture ou désalignement"))
        if moteur.get("found") and moteur.get("confiance", 100) < 40:
            diags.append(("🟡", f"Pic moteur faible (confiance {moteur['confiance']}%) → vérifier vitesse nominale ou glissement"))
        if sortie.get("confiance", 100) < 60:
            diags.append(("🟡", f"Ancrage sortie incertain (confiance {sortie.get('confiance',0)}%) → allonger le signal ou vérifier la zone d'ancrage"))
        if courr.get("found") and courr.get("amplitude", 0) > 0.05:
            diags.append(("🟡", f"Courroie visible ({courr['amplitude']:.3f} u.) → inspecter hernie ou usure localisée"))

        if diags:
            with st.expander(f"⚠️ {feuille} — {len(diags)} alerte(s)"):
                for icon, msg in diags:
                    st.write(f"{icon} {msg}")
        else:
            st.success(f"✅ {feuille} — Aucune anomalie détectée")

    # --------------------------------------------------
    # LEXIQUE
    # --------------------------------------------------
    with st.expander("💡 Aide à l'interprétation — Fréquences et leur signification physique"):
        st.markdown("""
| Composante | Signification | Anomalie si amplitude élevée |
|---|---|---|
| **Rotation Sortie** | Arbre de sortie lent (~1 rpm) — **ancre du calage** | Balourd sur l'organe entraîné |
| **Poulie Primaire** | Premier étage après réducteur | Défaut de fixation ou d'équilibrage |
| **Engrènement** | Contact dents à dents | Usure des flancs, jeu excessif |
| **Harmonique 4X** | 4ème harmonique d'engrènement | **Matage de denture, désalignement critique** |
| **Défilement Courroie** | Fréquence propre de la courroie | Hernie, fêlure, tension incorrecte |
| **Rotation Moteur** | Vitesse de l'arbre moteur | Balourd moteur, désalignement d'accouplement |

> ⚠️ **Zone 16 Hz** : énergie stable non liée à l'engrènement → anomalie magnétique/électrique (barres de rotor) ou frottement large bande (roulement endommagé, perte d'huile).
        """)

    # --------------------------------------------------
    # APPRENTISSAGE IA
    # --------------------------------------------------
    modele_df = resultats_df.dropna(subset=["Defaut_Réel"])
    if len(modele_df) >= 5:
        st.markdown("---")
        st.subheader("🤖 Apprentissage IA")
        features = ["Amp Cible (Bande)", "Entropie", "E0_5", "E10_20", "IDM3", "ID_Modulation", "IDM_Modulation_4X"]
        features_ok = [f for f in features if f in modele_df.columns]
        X = modele_df[features_ok]
        y = modele_df["Defaut_Réel"]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        resultats_df["Prédiction IA"] = model.predict(resultats_df[features_ok])
        st.dataframe(
            resultats_df[["Ensemble", "Defaut_Réel", "Prédiction IA", "IDM3", "IDM_Modulation_4X"]].dropna(subset=["Defaut_Réel"]),
            hide_index=True, use_container_width=True
        )
        st.caption("⚠️ Modèle indicatif — fiabilité limitée avec peu d'échantillons. Interprétation à croiser avec l'analyse fréquentielle.")

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------
    st.markdown("---")
    sortie_bytes = BytesIO()
    with pd.ExcelWriter(sortie_bytes, engine="openpyxl") as writer:
        resultats_df.to_excel(writer, index=False, sheet_name="Synthese_Totale")

        # Feuille de calage détaillée
        rows_cal = []
        for feuille, calage in calage_data.items():
            if "error" in calage:
                continue
            for nom_elem, res in calage["identification"].items():
                rows_cal.append({
                    "Machine":         feuille,
                    "Composante":      nom_elem,
                    "f_theorique_Hz":  res["f_theorique"],
                    "f_trouvee_Hz":    res["f_trouvee"] if res["found"] else None,
                    "amplitude_u":     res["amplitude"] if res["found"] else None,
                    "ecart_pct":       res["ecart_pct"],
                    "confiance_pct":   res["confiance"],
                    "ancre":           res["ancre"],
                })
        pd.DataFrame(rows_cal).to_excel(writer, index=False, sheet_name="Calage_Auto_Detail")

    st.download_button(
        label="📥 Télécharger le registre complet (.xlsx)",
        data=sortie_bytes.getvalue(),
        file_name="Registre_Vibratoire_Calage_Auto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("""
    👋 **Interface prête — Calage automatique activé**

    Le système va :
    1. **Ancrer** sur le pic de sortie à **~0.0167 Hz (1 rpm)** dans la zone [0.005 – 0.025 Hz]
    2. **Remonter** toute la chaîne cinématique par les rapports configurés
    3. **Identifier** chaque fréquence dans le spectre réel avec un score de confiance
    4. **Diagnostiquer** automatiquement les anomalies (4X, courroie, moteur)

    > ℹ️ La résolution fréquentielle doit être ≤ 0.001 Hz pour détecter le pic à 0.0167 Hz.
    > Cela nécessite un signal d'au moins **1000 s** (≈ 17 min) à fréquence d'échantillonnage suffisante.

    Configurez vos cinématiques dans la barre latérale, puis importez votre fichier Excel.
    """)
