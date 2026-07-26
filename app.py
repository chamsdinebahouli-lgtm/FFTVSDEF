"""
app.py
------
Analyseur Électrique DC & FFT - Application Streamlit (fichier unique).
Intégration d'indicateurs professionnels (DC, RMS, Taux d'ondulation, Facteur de crête).
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# LOGIQUE MÉTIER : calculs électriques DC, FFT et métriques d'ondulation
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


def calculer_metriques_dc(signal: np.ndarray) -> dict[str, float]:
    """Calcule les indicateurs électriques professionnels pour un signal DC."""
    x = np.asarray(signal, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n == 0:
        return {"DC (V)": 0.0, "RMS Total (V)": 0.0, "RMS AC (V)": 0.0, "Taux d'ondulation (%)": 0.0, "Facteur de Crête": 1.0}

    # 1. Composante DC (Moyenne arithmétique)
    dc_val = float(np.mean(x))

    # 2. Valeur RMS Totale (Efficace globale)
    rms_total = float(np.sqrt(np.mean(x**2)))

    # 3. Composante alternative (AC / Ondulations)
    x_ac = x - dc_val
    rms_ac = float(np.sqrt(np.mean(x_ac**2)))

    # 4. Taux d'ondulation (Ripple Factor en %) : (RMS_ac / |DC|) * 100
    if abs(dc_val) > 1e-9:
        taux_ondulation = float((rms_ac / abs(dc_val)) * 100.0)
    else:
        taux_ondulation = 0.0

    # 5. Facteur de Crête (Crest Factor) : Pic AC / RMS AC
    peak_ac = float(np.max(np.abs(x_ac)))
    if rms_ac > 1e-9:
        facteur_crete = float(peak_ac / rms_ac)
    else:
        facteur_crete = 1.0

    return {
        "DC (V)": dc_val,
        "RMS Total (V)": rms_total,
        "RMS AC (V)": rms_ac,
        "Taux d'ondulation (%)": taux_ondulation,
        "Facteur de Crête": facteur_crete,
    }


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
    """Pipeline complet pour un système (un onglet Excel) incluant DC et FFT."""
    dt = detecter_dt(df[col_temps].values, unite=unite_temps)
    signal_brut = df[col_signal].values

    # 1. Calcul des indicateurs globaux DC & Qualité d'onde
    metriques_dc = calculer_metriques_dc(signal_brut)

    # 2. Calcul FFT pour le suivi des harmoniques/ondulations ciblées
    resultat = calculer_fft(signal_brut, dt=dt, mode=mode)

    amplitudes_ondulations: dict[str, float] = {}
    alertes_resolution: list[str] = []
    tolerance_relative = 0.03

    for nom_composant, f_cible in freqs_cibles.items():
        amplitudes_ondulations[nom_composant] = extraire_amplitude(resultat, f_cible, mode, tolerance_relative)
        if mode == ModeFFT.NOUVEAU and not resolution_suffisante(resultat, f_cible, tolerance_relative):
            alertes_resolution.append(nom_composant)

    return {
        **metriques_dc,
        "ondulations": amplitudes_ondulations,
        "dt": resultat.dt,
        "resolution_hz": resultat.resolution_hz,
        "n_points": resultat.n_points,
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
    "Ondulation 1 (0.0167 Hz)": 0.016759,
    "Ondulation 2 (3.68 Hz)": 3.68,
    "Ondulation 3 (12.33 Hz)": 12.33,
    "Commutation (13.67 Hz)": 13.67,
}

st.set_page_config(page_title="Analyseur Électrique DC & Ondulations", layout="wide")
st.title("⚡ Analyseur Électrique DC & Qualité du Signal")
st.write("Suivi professionnel : Composante DC, Taux d'ondulation (Ripple), RMS et Analyse Spectrale.")

st.sidebar.header("⚙️ Paramètres")

mode_label = st.sidebar.radio(
    "Méthode FFT (Ondulations) :",
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
afficher_moyenne = st.sidebar.checkbox("Afficher la Ligne de Moyenne", value=True)
afficher_seuil = st.sidebar.checkbox(
    "Afficher le Seuil d'Alerte Lot (Moyenne + 1σ)",
    value=True,
    help="Repère les machines présentant un taux d'ondulation ou un RMS anormalement élevé par rapport au lot.",
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

    lignes = []
    for r in resultats:
        ligne = {
            "Système / Ligne DC": r["nom"],
            "DC (V)": round(r["DC (V)"], 4),
            "RMS Total (V)": round(r["RMS Total (V)"], 4),
            "RMS AC (V)": round(r["RMS AC (V)"], 4),
            "Taux d'ondulation (%)": round(r["Taux d'ondulation (%)"], 3),
            "Facteur de Crête": round(r["Facteur de Crête"], 3),
        }
        for comp, val in r["ondulations"].items():
            ligne[comp] = round(val, 5)
        lignes.append(ligne)

    df_res = pd.DataFrame(lignes)

    st.subheader("📋 Tableau Synthétique - Indicateurs Électriques DC")
    st.dataframe(df_res, use_container_width=True)

    cols_ondulations = list(FREQS_CIBLES.keys())
    df_melted = df_res.melt(
        id_vars=["Système / Ligne DC", "DC (V)", "RMS Total (V)", "Taux d'ondulation (%)", "Facteur de Crête"],
        value_vars=cols_ondulations,
        var_name="Fréquence / Ondulation",
        value_name="Amplitude AC (V)",
    )

    st.markdown("---")
    st.subheader("📈 Visualisation Dynamique & Qualité Électrique")

    metrique_maitresse = st.selectbox(
        "Métrique principale pour le classement dynamique du parc :",
        ["Taux d'ondulation (%)", "RMS Total (V)", "DC (V)", "Facteur de Crête"],
        index=0,
        help="Permet de trier automatiquement les machines de la plus saine à la plus perturbée.",
    )

    ordre_systemes = df_res.sort_values(by=metrique_maitresse, ascending=True)["Système / Ligne DC"].tolist()

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Vue Globale (Métrique Principale)",
            "📶 Spectre des Ondulations",
            "🔍 Focus Composante / Fréquence",
        ]
    )

    with tab1:
        st.markdown(f"#### Classement du parc selon : {metrique_maitresse} (Du plus faible au plus fort)")
        fig_global = px.bar(
            df_res,
            x="Système / Ligne DC",
            y=metrique_maitresse,
            title=f"Classement par {metrique_maitresse}",
            text_auto=".2f",
            category_orders={"Système / Ligne DC": ordre_systemes},
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
            fig_global.add_hline(
                y=val_moy + val_std,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Alerte (Moy+σ): {val_moy + val_std:.2f}",
                annotation_position="top right",
            )

        st.plotly_chart(fig_global, use_container_width=True)

    with tab2:
        st.markdown("#### Contribution des harmoniques / ondulations par ligne")
        fig_stacked = px.bar(
            df_melted,
            x="Système / Ligne DC",
            y="Amplitude AC (V)",
            color="Fréquence / Ondulation",
            title="Amplitudes des Ondulations AC par Ligne",
            barmode="stack",
            category_orders={"Système / Ligne DC": ordre_systemes},
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    with tab3:
        st.markdown("#### Analyse Ciblée par Fréquence d'Ondulation")
        composant_selectionne = st.selectbox(
            "Sélectionner l'ondulation ou la fréquence à inspecter :",
            options=["Tous les composants"] + cols_ondulations,
        )

        if composant_selectionne == "Tous les composants":
            df_filtre = df_melted
            titre_f = "Toutes les ondulations confondues"
        else:
            df_filtre = df_melted[df_melted["Fréquence / Ondulation"] == composant_selectionne]
            df_filtre = df_filtre.sort_values(by="Amplitude AC (V)", ascending=True)
            titre_f = f"Zoom sur : {composant_selectionne}"

        fig_single = px.bar(
            df_filtre,
            x="Système / Ligne DC",
            y="Amplitude AC (V)",
            color="Fréquence / Ondulation" if composant_selectionne == "Tous les composants" else None,
            title=titre_f,
            text_auto=".4f" if composant_selectionne != "Tous les composants" else False,
            barmode="group",
        )
        st.plotly_chart(fig_single, use_container_width=True)

else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale pour lancer l'analyse DC.")
