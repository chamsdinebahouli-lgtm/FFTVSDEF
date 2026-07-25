"""
app.py
------
Analyseur Vibratoire FFT - Application Streamlit (fichier unique).

Toute la logique (calcul FFT, lecture Excel, interface) est regroupée ici
volontairement pour simplifier le déploiement : aucun import local, donc
aucun risque d'erreur "ModuleNotFoundError" liée à l'organisation des
fichiers sur GitHub / Streamlit Cloud.
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
# LOGIQUE MÉTIER : calcul FFT et extraction d'amplitudes
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


def calculer_fft(signal: np.ndarray, dt: float, mode: ModeFFT) -> ResultatFFT:
    """Calcule le spectre d'amplitude (FFT) d'un signal temporel."""
    x = np.asarray(signal, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 2:
        raise DonneesInsuffisantesError(
            f"Signal trop court pour un calcul FFT (n={n} points valides)."
        )
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

    return ResultatFFT(
        freq=freq, amplitude=amplitude, dt=dt, resolution_hz=resolution_hz, n_points=n
    )


def extraire_amplitude(
    resultat: ResultatFFT,
    freq_cible: float,
    mode: ModeFFT,
    tolerance_relative: float = 0.03,
) -> float:
    """Extrait l'amplitude vibratoire à une fréquence cible donnée."""
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
    resultat = calculer_fft(df[col_signal].values, dt=dt, mode=mode)

    amplitudes: dict[str, float] = {}
    alertes_resolution: list[str] = []

    for nom_composant, f_cible in freqs_cibles.items():
        amplitudes[nom_composant] = extraire_amplitude(resultat, f_cible, mode)
        if mode == ModeFFT.NOUVEAU and not resolution_suffisante(resultat, f_cible):
            alertes_resolution.append(nom_composant)

    valeurs = list(amplitudes.values())

    return {
        "amplitudes": amplitudes,
        "somme": float(np.sum(valeurs)),
        "produit": float(np.prod(valeurs)),
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
        raise OngletInvalideError(
            f"L'onglet '{nom_onglet}' contient moins de 2 colonnes exploitables."
        )

    col_temps = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[col_temps]):
        raise OngletInvalideError(
            f"L'onglet '{nom_onglet}' : la première colonne "
            f"('{col_temps}') n'est pas numérique, impossible de l'utiliser "
            "comme axe temps."
        )

    return df


def charger_systemes(uploaded_file) -> tuple[list[dict], list[str]]:
    """Parcourt tous les onglets d'un fichier Excel, isole les onglets valides des invalides."""
    xls = pd.ExcelFile(uploaded_file)
    systemes: list[dict] = []
    erreurs: list[str] = []

    for nom_onglet in xls.sheet_names:
        try:
            df = lire_onglet(xls, nom_onglet)
            systemes.append(
                {
                    "nom": nom_onglet,
                    "df": df,
                    "col_temps": df.columns[0],
                    "col_signal": df.columns[1],
                }
            )
        except OngletInvalideError as exc:
            logger.warning("Onglet ignoré : %s", exc)
            erreurs.append(str(exc))
        except Exception as exc:
            logger.exception("Erreur inattendue sur l'onglet '%s'", nom_onglet)
            erreurs.append(f"Onglet '{nom_onglet}' : erreur inattendue ({exc}).")

    return systemes, erreurs


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

FREQS_CIBLES = {
    "Porte (0.0167 Hz)": 0.016759,
    "1er étage réducteur (3.68 Hz)": 3.68,
    "Dernier étage réducteur (12.33 Hz)": 12.33,
    "Moteur (13.67 Hz)": 13.67,
}

st.set_page_config(page_title="Analyse Vibratoire FFT & Stats", layout="wide")
st.title("📊 Analyseur Vibratoire FFT - Visualisation Dynamique & Statistiques")
st.write("Calcul FFT, classement dynamique et indicateurs statistiques de maintenance.")

st.sidebar.header("⚙️ Paramètres")

mode_label = st.sidebar.radio(
    "Méthode FFT :",
    ["Ancien Mode (Sans fenêtre, point fixe)", "Nouveau Mode (Hanning, pic local ±3%)"],
)
mode_calcul = ModeFFT.from_label_ui(mode_label)

unite_label = st.sidebar.radio(
    "Unité de la colonne temps :",
    ["Millisecondes (ms)", "Secondes (s)"],
    help="Vérifie cette unité si les fréquences extraites semblent incohérentes : "
    "une erreur d'unité fausse tout l'axe fréquentiel d'un facteur 1000.",
)
unite_temps = (
    UniteTemps.MILLISECONDES if unite_label.startswith("Millisecondes") else UniteTemps.SECONDES
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Indicateurs Statistiques")
afficher_moyenne = st.sidebar.checkbox("Afficher la Ligne de Moyenne", value=True)
afficher_seuil = st.sidebar.checkbox(
    "Afficher les Machines Atypiques du Lot (Moyenne + 1 Écart-type)",
    value=True,
    help="Ce seuil est calculé sur les machines du fichier importé, pas sur une "
    "baseline saine de référence. Il signale une dispersion relative au sein du "
    "lot, pas un seuil de maintenance absolu.",
)

uploaded_file = st.sidebar.file_uploader("Importer le fichier Excel (.xlsx)", type=["xlsx", "xls"])


@st.cache_data(show_spinner=False)
def _analyser_fichier(file_bytes: bytes, mode_value: str, unite_value: str):
    mode = ModeFFT(mode_value)
    unite = UniteTemps(unite_value)

    systemes, erreurs_lecture = charger_systemes(io.BytesIO(file_bytes))

    resultats = []
    erreurs_calcul = []
    for systeme in systemes:
        try:
            res = analyser_systeme(
                df=systeme["df"],
                col_temps=systeme["col_temps"],
                col_signal=systeme["col_signal"],
                mode=mode,
                freqs_cibles=FREQS_CIBLES,
                unite_temps=unite,
            )
            resultats.append({"nom": systeme["nom"], **res})
        except DonneesInsuffisantesError as exc:
            erreurs_calcul.append(f"Onglet '{systeme['nom']}' : {exc}")

    return resultats, erreurs_lecture + erreurs_calcul


if uploaded_file is not None:
    resultats, erreurs = _analyser_fichier(
        uploaded_file.getvalue(), mode_calcul.value, unite_temps.value
    )

    if erreurs:
        with st.expander(f"⚠️ {len(erreurs)} onglet(s) ignoré(s) — détails", expanded=False):
            for err in erreurs:
                st.warning(err)

    if not resultats:
        st.error("Aucun onglet exploitable n'a été trouvé dans ce fichier.")
        st.stop()

    alertes_globales = {
        r["nom"]: r["alertes_resolution"] for r in resultats if r["alertes_resolution"]
    }
    if alertes_globales:
        with st.expander("⚠️ Résolution fréquentielle insuffisante sur certains organes", expanded=False):
            st.write(
                "Pour les systèmes ci-dessous, la résolution FFT est plus grossière "
                "que la fenêtre de recherche ±3 % : le pic local retombe en pratique "
                "sur le bin le plus proche, comme en mode Ancien."
            )
            for nom, composants in alertes_globales.items():
                st.write(f"- **{nom}** : {', '.join(composants)}")

    lignes = []
    for r in resultats:
        ligne = {"Système": r["nom"], **{k: round(v, 6) for k, v in r["amplitudes"].items()}}
        ligne["Somme (V)"] = round(r["somme"], 6)
        ligne["Produit"] = f"{r['produit']:.2e}"
        ligne["dt (s)"] = f"{r['dt']:.6f}"
        ligne["Résolution (Hz)"] = f"{r['resolution_hz']:.5f}"
        lignes.append(ligne)

    df_res = pd.DataFrame(lignes)

    st.subheader("📋 Tableau Synthétique des Amplitudes")
    st.dataframe(df_res, use_container_width=True)
    st.caption(
        "Les colonnes dt et Résolution sont fournies pour diagnostic : "
        "vérifiez-les si un résultat semble incohérent."
    )

    cols_composants = list(FREQS_CIBLES.keys())
    df_melted = df_res.melt(
        id_vars=["Système"],
        value_vars=cols_composants,
        var_name="Composant Kinématique",
        value_name="Amplitude (V)",
    )

    st.markdown("---")
    st.subheader("📈 Visualisation Dynamique & Indicateurs Statistiques")

    ordre_systemes_global = df_res.sort_values(by="Somme (V)", ascending=True)["Système"].tolist()

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Barres Empilées (Ordre par Cumul)",
            "📶 Barres Groupées (Ordre par Cumul)",
            "🔍 Focus Organe (Rangement Dynamique)",
        ]
    )

    with tab1:
        st.markdown("#### Contribution de chaque organe (Tri par Cumul Croissant)")
        fig_stacked = px.bar(
            df_melted,
            x="Système",
            y="Amplitude (V)",
            color="Composant Kinématique",
            title="Amplitude Cumulée par Machine",
            barmode="stack",
            text_auto=".3f",
            category_orders={"Système": ordre_systemes_global},
        )

        moy_somme = df_res["Somme (V)"].mean()
        std_somme = df_res["Somme (V)"].std()

        if afficher_moyenne:
            fig_stacked.add_hline(
                y=moy_somme,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Moyenne globale: {moy_somme:.4f}V",
                annotation_position="bottom right",
            )
        if afficher_seuil and not pd.isna(std_somme):
            fig_stacked.add_hline(
                y=moy_somme + std_somme,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Atypique (Moy+σ): {moy_somme + std_somme:.4f}V",
                annotation_position="top right",
            )

        fig_stacked.update_layout(
            xaxis_title="Système / Machine",
            yaxis_title="Amplitude Cumulée (V)",
            legend_title="Organe Mécanique",
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    with tab2:
        st.markdown("#### Comparaison de tous les organes (Tri par Niveau Global)")
        fig_grouped = px.bar(
            df_melted,
            x="Système",
            y="Amplitude (V)",
            color="Composant Kinématique",
            title="Amplitudes Vibratoires par Composant",
            barmode="group",
            category_orders={"Système": ordre_systemes_global},
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    with tab3:
        st.markdown("#### Focus Dynamique par Organe Mécanique")

        composants_disponibles = ["Tous les composants"] + cols_composants
        composant_selectionne = st.selectbox(
            "Sélectionner le module à analyser :",
            options=composants_disponibles,
            index=0,
        )

        if composant_selectionne == "Tous les composants":
            df_filtre = df_melted
            ordre_dynamique = ordre_systemes_global
            titre_graph = "Comparaison Globale - Classée par Cumul Total"
            valeurs_stats = df_res["Somme (V)"]
        else:
            df_filtre = df_melted[df_melted["Composant Kinématique"] == composant_selectionne]
            ordre_dynamique = df_filtre.sort_values(by="Amplitude (V)", ascending=True)[
                "Système"
            ].tolist()
            titre_graph = f"Classement du plus faible au plus fort : {composant_selectionne}"
            valeurs_stats = df_filtre["Amplitude (V)"]

        fig_single = px.bar(
            df_filtre,
            x="Système",
            y="Amplitude (V)",
            color="Composant Kinématique" if composant_selectionne == "Tous les composants" else None,
            title=titre_graph,
            text_auto=".4f" if composant_selectionne != "Tous les composants" else False,
            barmode="group",
            category_orders={"Système": ordre_dynamique},
        )

        if len(valeurs_stats) > 0:
            moy_comp = valeurs_stats.mean()
            std_comp = valeurs_stats.std()

            if afficher_moyenne:
                fig_single.add_hline(
                    y=moy_comp,
                    line_dash="dash",
                    line_color="blue",
                    annotation_text=f"Moyenne: {moy_comp:.4f}V",
                    annotation_position="bottom right",
                )
            if afficher_seuil and not pd.isna(std_comp):
                fig_single.add_hline(
                    y=moy_comp + std_comp,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Atypique (Moy+σ): {moy_comp + std_comp:.4f}V",
                    annotation_position="top right",
                )

        fig_single.update_layout(
            xaxis_title="Système / Machine",
            yaxis_title="Amplitude Vibratoire (V)",
            showlegend=(composant_selectionne == "Tous les composants"),
        )
        st.plotly_chart(fig_single, use_container_width=True)

else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale.")
