"""
app.py
------
Analyseur Électrique DC & Diagnostique Vibratoire - Application Streamlit (fichier unique).
Intégration d'indicateurs avancés : RMS, Peak, Peak-to-Peak, Crest Factor, Kurtosis, Skewness, THD, SNR, Énergie, Offset DC.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# LOGIQUE MÉTIER : calculs électriques DC, statistiques avancées et FFT
# =============================================================================


class ModeFFT(str, Enum):
    """Méthode de calcul de la FFT."""

    ANCIEN = "ancien"  # Sans fenêtre, lecture au point fréquentiel le plus proche
    NOUVEAU = "nouveau"  # Fenêtre de Hanning, recherche du pic local ±3 %

    @classmethod
    def from_label_ui(cls, label: str) -> "ModeFFT":
        return cls.ANCIEN if label.startswith("Ancien") else cls.NOUVEAU


class UniteTemps(str, Enum):
    """Unité de la colonne temps dans le fichier source."""

    MILLISECONDES = "ms"
    SECONDES = "s"


@dataclass(frozen=True)
class ResultatFFT:
    """Résultat d'un calcul FFT sur un signal."""

    freq: np.ndarray
    amplitude: np.ndarray
    dt: float
    resolution_hz: float
    n_points: int


class DonneesInsuffisantesError(ValueError):
    """Levée quand un signal est trop court ou dégénéré pour être analysé."""


class OngletInvalideError(ValueError):
    """Levée quand un onglet Excel n'a pas le format attendu (temps, signal)."""


def detecter_dt(temps: np.ndarray, unite: UniteTemps = UniteTemps.MILLISECONDES) -> float:
    """Détecte le pas d'échantillonnage (dt) en secondes à partir d'une colonne temps."""
    diffs = np.diff(temps.astype(float))
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        raise DonneesInsuffisantesError(
            "Impossible de déterminer le pas d'échantillonnage : "
            "la colonne temps ne contient pas d'intervalles positifs."
        )
    facteur = 1000.0 if unite == UniteTemps.MILLISECONDES else 1.0
    return float(np.median(diffs_pos) / facteur)


def calculer_metriques_avancees(signal: np.ndarray, resultat_fft: ResultatFFT) -> dict[str, float]:
    """Calcule l'ensemble des indicateurs statistiques et électriques avancés."""
    x = np.asarray(signal, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    metriques_par_defaut = {
        "Offset DC (V)": 0.0,
        "RMS Total (V)": 0.0,
        "RMS AC (V)": 0.0,
        "Peak (V)": 0.0,
        "Peak-to-Peak (V)": 0.0,
        "Crest Factor": 1.0,
        "Kurtosis": 3.0,
        "Skewness": 0.0,
        "THD (%)": 0.0,
        "SNR (dB)": 0.0,
        "Énergie AC": 0.0,
    }

    if n == 0:
        return metriques_par_defaut

    try:
        dc_val = float(np.mean(x))
        rms_total = float(np.sqrt(np.mean(x**2)))
        
        x_ac = x - dc_val
        rms_ac = float(np.sqrt(np.mean(x_ac**2)))
        energie_ac = float(np.sum(x_ac**2))

        peak_val = float(np.max(np.abs(x_ac)))
        min_val, max_val = float(np.min(x)), float(np.max(x))
        peak_to_peak = max_val - min_val

        crest_factor = float(peak_val / rms_ac) if rms_ac > 1e-9 else 1.0

        kurt = float(kurtosis(x, fisher=False, bias=False)) if n > 3 else 3.0
        skw = float(skew(x, bias=False)) if n > 2 else 0.0

        freq, amp = resultat_fft.freq, resultat_fft.amplitude
        energie_totale_spec = float(np.sum(amp**2))
        
        if energie_totale_spec > 1e-9:
            bruit_estime = float(np.median(amp)) if len(amp) > 0 else 0.0
            signal_utile_estime = float(np.max(amp))
            snr = float(20 * np.log10(signal_utile_estime / bruit_estime)) if bruit_estime > 1e-9 else 0.0
            thd = float((np.sqrt(max(0, energie_totale_spec - signal_utile_estime**2)) / (signal_utile_estime + 1e-9)) * 100.0)
        else:
            snr = 0.0
            thd = 0.0

        return {
            "Offset DC (V)": dc_val,
            "RMS Total (V)": rms_total,
            "RMS AC (V)": rms_ac,
            "Peak (V)": peak_val,
            "Peak-to-Peak (V)": peak_to_peak,
            "Crest Factor": crest_factor,
            "Kurtosis": kurt,
            "Skewness": skw,
            "THD (%)": thd,
            "SNR (dB)": snr,
            "Énergie AC": energie_ac,
        }
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques avancées : {e}")
        return metriques_par_defaut


def calculer_fft(signal: np.ndarray, dt: float, mode: ModeFFT) -> ResultatFFT:
    """Calcule le spectre d'amplitude (FFT) d'un signal temporel."""
    x = np.asarray(signal, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 2:
        raise DonneesInsuffisantesError(f"Signal trop court pour un calcul FFT (n={n} points valides).")
    if dt <= 0:
        raise DonneesInsuffisantesError(f"Pas d'échantillonnage invalide (dt={dt}).")

    x_centre = x - np.mean(x)

    if mode == ModeFFT.ANCIEN:
        amplitude = np.abs(rfft(x_centre)) * (2.0 / n)
    else:
        fenetre = np.hanning(n)
        amplitude = np.abs(rfft(x_centre * fenetre)) * (2.0 / np.sum(fenetre))

    freq = rfftfreq(n, d=dt)
    resolution_hz = 1.0 / (n * dt)

    return ResultatFFT(freq=freq, amplitude=amplitude, dt=dt, resolution_hz=resolution_hz, n_points=n)


def extraire_amplitude(
    resultat: ResultatFFT,
    freq_cible: float,
    mode: ModeFFT,
    tolerance_relative: float = 0.03,
) -> float:
    """Extrait l'amplitude d'ondulation à une fréquence cible donnée."""
    freq, amp = resultat.freq, resultat.amplitude

    if mode == ModeFFT.ANCIEN:
        idx = int(np.argmin(np.abs(freq - freq_cible)))
        return float(amp[idx])

    f_min = freq_cible * (1 - tolerance_relative)
    f_max = freq_cible * (1 + tolerance_relative)
    mask = (freq >= f_min) & (freq <= f_max)

    if np.any(mask):
        return float(np.max(amp[mask]))

    idx = int(np.argmin(np.abs(freq - freq_cible)))
    return float(amp[idx])


def resolution_suffisante(
    resultat: ResultatFFT, freq_cible: float, tolerance_relative: float = 0.03
) -> bool:
    """Vérifie que la résolution fréquentielle permet de distinguer une fréquence cible."""
    largeur_fenetre = 2 * freq_cible * tolerance_relative
    return largeur_fenetre >= 2 * resultat.resolution_hz


def analyser_systeme(
    df: pd.DataFrame,
    col_temps: str,
    col_signal: str,
    mode: ModeFFT,
    freqs_cibles: dict[str, float],
    unite_temps: UniteTemps = UniteTemps.MILLISECONDES,
) -> dict:
    """Pipeline complet pour un système (un onglet Excel)."""
    dt = detecter_dt(df[col_temps].values, unite=unite_temps)
    signal_brut = df[col_signal].values

    resultat_fft = calculer_fft(signal_brut, dt=dt, mode=mode)
    metriques = calculer_metriques_avancees(signal_brut, resultat_fft)

    amplitudes_cibles: dict[str, float] = {}
    alertes_resolution: list[str] = []
    tolerance_relative = 0.03

    for nom_composant, f_cible in freqs_cibles.items():
        amplitudes_cibles[nom_composant] = extraire_amplitude(resultat_fft, f_cible, mode, tolerance_relative)
        if mode == ModeFFT.NOUVEAU and not resolution_suffisante(resultat_fft, f_cible, tolerance_relative):
            alertes_resolution.append(nom_composant)

    return {
        **metriques,
        "cibles": amplitudes_cibles,
        "dt": resultat_fft.dt,
        "resolution_hz": resultat_fft.resolution_hz,
        "n_points": resultat_fft.n_points,
        "alertes_resolution": alertes_resolution,
    }


def lire_onglet(xls: pd.ExcelFile, nom_onglet: str) -> pd.DataFrame:
    """Lit un onglet et vérifie qu'il contient au minimum deux colonnes exploitables."""
    df = pd.read_excel(xls, sheet_name=nom_onglet)
    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 2:
        raise OngletInvalideError(f"L'onglet '{nom_onglet}' contient moins de 2 colonnes exploitables.")

    col_temps = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[col_temps]):
        raise OngletInvalideError(
            f"L'onglet '{nom_onglet}' : la première colonne ('{col_temps}') n'est pas numérique."
        )

    return df


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

FREQS_CIBLES = {
    "Rotation 1 tr/min (0.0167 Hz)": 0.016759,
    "1er étage réducteur (3.68 Hz)": 3.68,
    "Dernier étage réducteur (12.33 Hz)": 12.33,
    "Moteur / Commutation (13.67 Hz)": 13.67,
}

st.set_page_config(page_title="Diagnostic Électromécanique DC & FFT", layout="wide")
st.title("⚡ Analyseur Avancé : Couplage Électrique & Vibratoire (1 tr/min)")
st.write("Suivi multi-indicateurs professionnels : Kurtosis, Facteur de Crête, THD, RMS et Analyse Spectrale.")

st.sidebar.header("⚙️ Paramètres")

mode_label = st.sidebar.radio(
    "Méthode FFT :",
    ["Ancien Mode (Sans fenêtre, point fixe)", "Nouveau Mode (Hanning, pic local ±3%)"],
)
mode_calcul = ModeFFT.from_label_ui(mode_label)

unite_label = st.sidebar.radio(
    "Unité de la colonne temps :",
    ["Millisecondes (ms)", "Secondes (s)"],
)
unite_temps = (
    UniteTemps.MILLISECONDES if unite_label.startswith("Millisecondes") else UniteTemps.SECONDES
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Indicateurs Statistiques & Seuils")
afficher_moyenne = st.sidebar.checkbox("Afficher la Ligne de Moyenne du lot", value=True)
afficher_seuil = st.sidebar.checkbox(
    "Afficher le Seuil d'Alerte (Moyenne + 1σ)",
    value=True,
    help="Repère les machines présentant des déviations statistiques anormales par rapport au parc.",
)

uploaded_file = st.sidebar.file_uploader("Importer le fichier Excel (.xlsx)", type=["xlsx", "xls"])


@st.cache_data(show_spinner=False)
def _analyser_fichier(file_bytes: bytes, mode_value: str, unite_value: str):
    mode = ModeFFT(mode_value)
    unite = UniteTemps(unite_value)

    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    resultats = []
    erreurs = []

    for nom_onglet in xls.sheet_names:
        try:
            df = lire_onglet(xls, nom_onglet)
            res = analyser_systeme(
                df=df,
                col_temps=df.columns[0],
                col_signal=df.columns[1],
                mode=mode,
                freqs_cibles=FREQS_CIBLES,
                unite_temps=unite,
            )
            resultats.append({"nom": nom_onglet, **res})
        except Exception as exc:
            erreurs.append(f"Onglet '{nom_onglet}' : {exc}")

    return resultats, erreurs


if uploaded_file is not None:
    resultats, erreurs = _analyser_fichier(
        uploaded_file.getvalue(), mode_calcul.value, unite_temps.value
    )

    if erreurs:
        with st.expander(f"⚠️ {len(erreurs)} onglet(s) ignoré(s)", expanded=False):
            for err in erreurs:
                st.warning(err)

    if not resultats:
        st.error("Aucun onglet exploitable trouvé.")
        st.stop()

    # Construction du tableau synthétique complet sécurisée
    lignes = []
    for r in resultats:
        ligne = {
            "Système / Machine": r.get("nom", "Inconnu"),
            "Offset DC (V)": round(r.get("Offset DC (V)", 0.0), 4),
            "RMS Total (V)": round(r.get("RMS Total (V)", 0.0), 4),
            "RMS AC (V)": round(r.get("RMS AC (V)", 0.0), 4),
            "Peak (V)": round(r.get("Peak (V)", 0.0), 4),
            "Peak-to-Peak (V)": round(r.get("Peak-to-Peak (V)", 0.0), 4),
            "Crest Factor": round(r.get("Crest Factor", 1.0), 3),
            "Kurtosis": round(r.get("Kurtosis", 3.0), 3),
            "Skewness": round(r.get("Skewness", 0.0), 3),
            "THD (%)": round(r.get("THD (%)", 0.0), 2),
            "SNR (dB)": round(r.get("SNR (dB)", 0.0), 2),
        }
        for comp, val in r.get("cibles", {}).items():
            ligne[comp] = round(val, 5)
        lignes.append(ligne)

    df_res = pd.DataFrame(lignes)

    st.subheader("📋 Tableau Synthétique - Indicateurs Électromécaniques Avancés")
    st.dataframe(df_res, use_container_width=True)

    metriques_disponibles = [
        "Kurtosis",
        "Crest Factor",
        "RMS AC (V)",
        "Peak-to-Peak (V)",
        "THD (%)",
        "SNR (dB)",
        "RMS Total (V)",
        "Offset DC (V)",
    ]

    metriques_existantes = [m for m in metriques_disponibles if m in df_res.columns]
    cols_cibles = list(FREQS_CIBLES.keys())

    # Fusion propre en n'utilisant que le nom de la machine comme identifiant
    df_melted = pd.melt(
        df_res,
        id_vars=["Système / Machine"],
        value_vars=cols_cibles,
        var_name="Composante / Fréquence Cible",
        value_name="Amplitude Spectrale (V)",
    )
    # Ré-associe proprement les métriques de base pour chaque machine
    df_melted = df_melted.merge(df_res[["Système / Machine"] + metriques_existantes], on="Système / Machine", how="left")

    st.markdown("---")
    st.subheader("📈 Visualisation & Diagnostic Dynamique")

    col_gauche, col_droite = st.columns(2)
    with col_gauche:
        metrique_maitresse = st.selectbox(
            "Indicateur principal à analyser / classer :",
            options=metriques_existantes if metriques_existantes else ["Système / Machine"],
            index=0,
            help="Sélectionnez un indicateur sensible aux chocs ou aux ondulations pour identifier les machines atypiques.",
        )
    with col_droite:
        sens_tri = st.radio("Ordre de classement :", ["Du plus faible au plus fort", "Du plus fort au plus faible"], horizontal=True)

    ascending_flag = True if sens_tri.startswith("Du plus faible") else False
    ordre_systemes = df_res.sort_values(by=metrique_maitresse, ascending=ascending_flag)["Système / Machine"].tolist()

    tab1, tab2, tab3 = st.tabs(
        [
            f"📊 Classement Global ({metrique_maitresse})",
            "📶 Spectre des Fréquences Cibles (1 tr/min, etc.)",
            "🔍 Focus par Composant Mécanique",
        ]
    )

    with tab1:
        st.markdown(f"#### Classement du parc selon l'indicateur : **{metrique_maitresse}**")
        fig_global = px.bar(
            df_res,
            x="Système / Machine",
            y=metrique_maitresse,
            title=f"Classement des machines par {metrique_maitresse}",
            text_auto=".2f",
            category_orders={"Système / Machine": ordre_systemes},
            color=metrique_maitresse,
            color_continuous_scale="Viridis",
        )

        val_moy = df_res[metrique_maitresse].mean()
        val_std = df_res[metrique_maitresse].std()

        if afficher_moyenne:
            fig_global.add_hline(
                y=val_moy,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Moyenne: {val_moy:.2f}",
                annotation_position="bottom right",
            )
        if afficher_seuil and not pd.isna(val_std):
            seuil_alerte = val_moy + val_std
            fig_global.add_hline(
                y=seuil_alerte,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Alerte (Moy+σ): {seuil_alerte:.2f}",
                annotation_position="top right",
            )

        st.plotly_chart(fig_global, use_container_width=True)

    with tab2:
        st.markdown("#### Amplitudes spectrales par machine (Empilées)")
        fig_stacked = px.bar(
            df_melted,
            x="Système / Machine",
            y="Amplitude Spectrale (V)",
            color="Composante / Fréquence Cible",
            title="Contribution des Fréquences Cibles (dont 1 tr/min)",
            barmode="stack",
            category_orders={"Système / Machine": ordre_systemes},
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    with tab3:
        st.markdown("#### Analyse Ciblée par Composante Fréquentielle")
        composant_selectionne = st.selectbox(
            "Sélectionner la fréquence/composante à inspecter :",
            options=["Toutes les composantes"] + cols_cibles,
        )

        if composant_selectionne == "Toutes les composantes":
            df_filtre = df_melted
            titre_f = "Toutes les fréquences cibles"
        else:
            df_filtre = df_melted[df_melted["Composante / Fréquence Cible"] == composant_selectionne]
            df_filtre = df_filtre.sort_values(by="Amplitude Spectrale (V)", ascending=ascending_flag)
            titre_f = f"Zoom sur : {composant_selectionne}"

        fig_single = px.bar(
            df_filtre,
            x="Système / Machine",
            y="Amplitude Spectrale (V)",
            color="Composante / Fréquence Cible" if composant_selectionne == "Toutes les composantes" else None,
            title=titre_f,
            text_auto=".4f" if composant_selectionne != "Toutes les composantes" else False,
            barmode="group",
        )
        st.plotly_chart(fig_single, use_container_width=True)

else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale pour lancer l'analyse avancée.")
