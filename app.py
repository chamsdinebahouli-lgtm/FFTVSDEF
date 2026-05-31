import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
import plotly.express as px

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT — Fréquence Fixe",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Calage sur Fréquence de Sortie Fixe")

# --------------------------------------------------
# FONCTIONS CINÉMATIQUE
# --------------------------------------------------
def build_chain_from_fixed_sortie(f_sortie, ratio_reducteur, dents_primaire, dents_secondaire, dents_courroie):
    """
    Calcule toute la chaîne cinématique à partir d'une fréquence de sortie FIXE et imposée.
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

def signal_info(df):
    """Retourne les métriques clés du signal pour diagnostic."""
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
    if amp_max <= 0: return 0
    ecart_score = max(0.0, 1.0 - (ecart_pct / 2.0))
    amp_score   = min(1.0, amplitude / amp_max)
    return int((0.6 * ecart_score + 0.4 * amp_score) * 100)

def process_with_fixed_vitesse(freq, amp, f_sortie_fixe, machine_cfg, tolerance_pct=0.02, n_harmonics=4):
    """Calcule le calage cinématique basé uniquement sur la vitesse imposée."""
    amp_max = np.max(amp)

    # Construction de la chaîne théorique stricte
    chain = build_chain_from_fixed_sortie(
        f_sortie_fixe,
        machine_cfg["ratio_reducteur"],
        machine_cfg["dents_primaire"],
        machine_cfg["dents_secondaire"],
        machine_cfg["dents_courroie"],
    )

    identification = {}
    for nom, f_theoric in chain.items():
        # On cherche s'il y a un pic réel proche de cette harmonique théorique
        f_found, a_found, found = find_peak_near(freq, amp, f_theoric, tolerance_pct)
        ecart_pct = abs(f_found - f_theoric) / f_theoric * 100 if found else None
        conf = confidence_score(ecart_pct, a_found, amp_max) if found else 0

        # Harmoniques secondaires
        harmoniques = []
        for h in range(2, n_harmonics + 1):
            fh = f_theoric * h
            fh_found, ah_found, hfound = find_peak_near(freq, amp, fh, tolerance_pct)
            if hfound:
                harmoniques.append({"ordre": h, "f_theorique": fh, "f_trouvee": fh_found, "amplitude": ah_found})

        identification[nom] = {
            "f_theorique":  f_theoric,
            "f_trouvee":    f_found if found else f_theoric, # Valeur mesurée ou fallback théorique
            "amplitude":    a_found if found else 0.0,
            "found":        found,
            "ecart_pct":    ecart_pct,
            "confiance":    conf,
            "harmoniques":  harmoniques,
            "fixe":         nom == "Rotation Sortie"
        }

    return identification

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
    if np.sum(p) == 0: return 0.0
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
st.sidebar.subheader("🎯 Imposition de la Vitesse Réelle")

# Sélection par l'utilisateur de la vitesse de sortie fixe
vitesse_mode = st.sidebar.radio("Définir la vitesse de sortie via :", ["Fréquence (Hz)", "Rotation (RPM)"])
if vitesse_mode == "Fréquence (Hz)":
    f_sortie_imposee = st.sidebar.number_input("Fréquence de sortie fixe (Hz) :", min_value=0.001, max_value=5.000, value=0.01666, format="%.5f")
    rpm_equivalent = f_sortie_imposee * 60.0
    st.sidebar.caption(f"Équivaut à : **{rpm_equivalent:.3f} RPM**")
else:
    rpm_impose = st.sidebar.number_input("Vitesse de sortie fixe (RPM) :", min_value=0.01, max_value=300.0, value=1.00, format="%.2f")
    f_sortie_imposee = rpm_impose / 60.0
    st.sidebar.caption(f"Équivaut à : **{f_sortie_imposee:.5f} Hz**")

tolerance_pct = st.sidebar.slider(
    "Tolérance recherche pics (%) :",
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
        if i < nb_machines - 1: st.markdown("---")

st.sidebar.markdown("---")
notes_text = st.sidebar.text_area(
    "📝 Scores de défaut réels (Feedback terrain) :",
    value="ASM21A=2.44\nASM21B=2.74\nASM22A=1.67", height=100
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE
# --------------------------------------------------
COLORS_MAP = {
    "Rotation Sortie":          "#1D9E75",
    "Rotation Poulie Primaire":  "#7F77DD",
    "Engrènement":               "#BA7517",
    "Harmonique Engrènement 4X": "#E24B4A",
    "Défilement Courroie":       "#185FA5",
    "Rotation Moteur":           "#D85A30",
}

def conf_badge(conf, found, is_fixe=False):
    if is_fixe: return "🔒 IMPOSÉ (100%)"
    if not found: return "⚪ Non trouvé"
    if conf >= 70: return f"🟢 {conf}%"
    if conf >= 40: return f"🟡 {conf}%"
    return f"🔴 {conf}%"

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    fft_data, calage_data = {}, {}

    # Parsing notes terrain
    notes = {}
    for ligne in notes_text.splitlines():
        if "=" in ligne:
            nom, val = ligne.split("=", 1)
            try: notes[nom.strip()] = float(val.strip())
            except ValueError: pass

    # Onglet diagnostic de résolution
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
        st.subheader("📡 Diagnostic du Signal Temporel")
        ri1, ri2, ri3 = st.columns(3)
        ri1.metric("Points / acquisition", f"{info['N']:,}")
        ri2.metric("Durée du signal", f"{info['T_s']:.1f} s  ({info['T_s']/60:.1f} min)")
        ri3.metric("Résolution brute FFT", f"{info['resolution_hz']:.5f} Hz")
        st.info(f"💡 Vitesse verrouillée par l'utilisateur à **{f_sortie_imposee:.5f} Hz** ({f_sortie_imposee*60:.3f} RPM). Plus aucun risque de dérive basse fréquence.")
        st.markdown("---")

    # Boucle de traitement des onglets
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): continue

            # Calcul de la FFT standard
            freq, amp = calcul_fft(df)
            fft_data[feuille] = (freq, amp)

            # Identification de la configuration machine associée
            machine_cfg = st.session_state.machines[0]
            for m in st.session_state.machines:
                if m["nom"].lower() in feuille.lower() or feuille.lower() in m["nom"].lower():
                    machine_cfg = m
                    break

            # Calcul cinématique complet sur base fixe
            ident = process_with_fixed_vitesse(freq, amp, f_sortie_imposee, machine_cfg, tolerance_pct, n_harmonics)
            calage_data[feuille] = ident

            # Calcul des indicateurs focalisés sur la fréquence 4X calculée
            h4x_info = ident["Harmonique Engrènement 4X"]
            f_cible_idm3 = h4x_info["f_trouvee"]

            indic = calcul_indicateurs(freq, amp, f_cible_idm3)
            indic["Ensemble"]    = feuille
            indic["Defaut_Réel"] = notes.get(feuille, np.nan)

            for nom_elem, res_elem in ident.items():
                indic[f"Amp_{nom_elem}"] = res_elem["amplitude"]

            # Sécurisation du calcul des modulations
            amp_moteur = ident["Rotation Moteur"]["amplitude"]
            amp_sortie = ident["Rotation Sortie"]["amplitude"]
            amp_4x     = ident["Harmonique Engrènement 4X"]["amplitude"]

            indic["ID_Modulation"]     = amp_moteur * amp_sortie
            indic["IDM_Modulation_4X"] = amp_sortie * amp_4x

            resultats.append(indic)
        except Exception as e:
            st.sidebar.error(f"Erreur traitement onglet {feuille} : {e}")

    if len(resultats) == 0:
        st.warning("Aucune donnée exploitable extraite.")
        st.stop()

    resultats_df = pd.DataFrame(resultats)

    # --------------------------------------------------
    # AFFICHAGE DES ONGLETS PAR MACHINE
    # --------------------------------------------------
    st.subheader("🎯 Suivi des Composantes (Fréquences Fixées)")
    tabs = st.tabs([r["Ensemble"] for _, r in resultats_df.iterrows()])

    for tab, (_, row) in zip(tabs, resultats_df.iterrows()):
        with tab:
            feuille = row["Ensemble"]
            ident = calage_data.get(feuille, {})

            c1, c2, c3 = st.columns(3)
            c1.metric("Vitesse Sortie Imposée", f"{f_sortie_imposee:.5f} Hz")
            c2.metric("Vitesse Sortie (RPM)", f"{f_sortie_imposee*60:.3f} rpm")
            identified = sum(1 for v in ident.values() if v["found"])
            c3.metric("Pics réels corrélés", f"{identified}/{len(ident)}")

            st.markdown("**Tableau des composants cinématiques**")
            rows_ident = []
            for nom_elem, res in ident.items():
                harm_str = ", ".join([f"×{h['ordre']}@{h['f_trouvee']:.2f}Hz" for h in res["harmoniques"]]) or "—"
                is_sortie = (nom_elem == "Rotation Sortie")
                rows_ident.append({
                    "Composante":          nom_elem,
                    "f cible calculée (Hz)": f"{res['f_theorique']:.5f}",
                    "f trouvé au plus proche (Hz)": f"{res['f_trouvee']:.5f}" if (res["found"] or is_sortie) else "—",
                    "Amplitude du pic (V)":  f"{res['amplitude']:.4f}" if (res["found"] or is_sortie) else "—",
                    "Écart freq (%)":       f"{res['ecart_pct']:.3f}" if res["ecart_pct"] is not None else "0.000",
                    "État d'ancrage":       conf_badge(res["confiance"], res["found"], is_sortie),
                    "Harmoniques Validées": harm_str
                })
            st.dataframe(pd.DataFrame(rows_ident), hide_index=True, use_container_width=True)

    # --------------------------------------------------
    # TABLEAU GÉNÉRAL ET SEUILLAGE
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📋 État de santé global — Analyse Statistique")
    df_valid = resultats_df.dropna(subset=["Defaut_Réel"])

    if len(df_valid) >= 3:
        X_meca = df_valid["Defaut_Réel"].values
        Y_idm3 = df_valid["IDM3"].values
        m_lin, b_lin = np.polyfit(X_meca, Y_idm3, 1)
        sigma = np.std(Y_idm3 - (m_lin * X_meca + b_lin)) or 1.0

        statuts = []
        for _, row in resultats_df.iterrows():
            if not pd.isna(row["Defaut_Réel"]):
                val_att = m_lin * row["Defaut_Réel"] + b_lin
                ecart = row["IDM3"] - val_att
                if ecart <= 1.0 * sigma: statuts.append("🟢 Conforme")
                elif ecart <= 2.0 * sigma: statuts.append("🟡 Écart Modéré")
                else: statuts.append("🔴 Alarme (Hors Tendance)")
            else:
                statuts.append("🟢 Bon" if row["IDM3"] < 3.5 else "🟡 À surveiller" if row["IDM3"] < 4.5 else "🔴 Alarme")
        resultats_df["Statut"] = statuts
    else:
        resultats_df["Statut"] = resultats_df["IDM3"].apply(lambda x: "🟢 Bon" if x < 3.5 else "🟡 À surveiller" if x < 4.5 else "🔴 Alarme")

    colonnes_affichage = [
        "Ensemble", "Statut", "Defaut_Réel", "IDM3", "IDM_Modulation_4X", "ID_Modulation",
        "Amp_Rotation Sortie", "Amp_Harmonique Engrènement 4X", "Amp_Rotation Moteur"
    ]
    colonnes_ok = [c for c in colonnes_affichage if c in resultats_df.columns]
    st.dataframe(resultats_df.sort_values("IDM3", ascending=False)[colonnes_ok], use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # GRAPHES SPECTRES
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Visualisation des spectres FFT (Lignes repères fixes)")
    ensemble = st.selectbox("Sélectionner une machine à analyser :", resultats_df["Ensemble"].tolist())

    if ensemble in fft_data and ensemble in calage_data:
        freq, amp = fft_data[ensemble]
        ident     = calage_data[ensemble]

        fft_df = pd.DataFrame({"Fréquence (Hz)": freq, "Amplitude": amp})
        fig = px.line(fft_df, x="Fréquence (Hz)", y="Amplitude", title=f"Spectre FFT — {ensemble}")
        fig.update_xaxes(range=[0, 20])

        for nom_elem, res in ident.items():
            f_val = res["f_theorique"] # On affiche la ligne théorique exacte fixée
            if f_val > 20: continue
            col = COLORS_MAP.get(nom_elem, "#888888")
            fig.add_vline(x=f_val, line_dash="solid" if res["found"] else "dot", line_color=col, annotation_text=nom_elem, annotation_font_color=col)
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # DIAGNOSTICS AUTOMATIQUES
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🔎 Diagnostics Automatiques")
    for f_name, id_c in calage_data.items():
        alerts = []
        if id_c.get("Harmonique Engrènement 4X", {}).get("amplitude", 0) > 0.15:
            alerts.append("🔴 **Harmonique 4X (État Cible B) critique :** Énergie très élevée détectée sur la zone cible.")
        if id_c.get("Défilement Courroie", {}).get("amplitude", 0) > 0.05:
            alerts.append("🟡 **Activité Courroie :** Émergence de la fréquence courroie (usure ou tension lâche).")

        if alerts:
            with st.expander(f"⚠️ {f_name} — {len(alerts)} alerte(s) détectée(s)"):
                for a in alerts: st.write(a)
        else:
            st.success(f"✅ {f_name} — Aucun défaut critique détecté aux fréquences verrouillées.")
