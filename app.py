#!/usr/bin/env bash
# Script de generation du projet Analyseur Vibratoire FFT
# Usage : bash generate_project.sh
set -e

mkdir -p vib_analyzer/vibration_analysis vib_analyzer/tests
cd vib_analyzer

mkdir -p $(dirname .gitignore)
cat > .gitignore << 'PYEOF__GITIGNORE'
__pycache__/
*.pyc
venv/
.venv/
*.egg-info/
.pytest_cache/
.streamlit/
PYEOF__GITIGNORE

mkdir -p $(dirname requirements.txt)
cat > requirements.txt << 'PYEOF_REQUIREMENTS_TXT'
streamlit>=1.32
pandas>=2.0
numpy>=1.24
scipy>=1.10
plotly>=5.18
openpyxl>=3.1
pytest>=7.4
PYEOF_REQUIREMENTS_TXT

mkdir -p $(dirname pytest.ini)
cat > pytest.ini << 'PYEOF_PYTEST_INI'
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v
PYEOF_PYTEST_INI

mkdir -p $(dirname README.md)
cat > README.md << 'PYEOF_README_MD'
# Analyseur Vibratoire FFT

Application Streamlit d'analyse vibratoire par FFT pour la maintenance
prédictive : extraction d'amplitudes à des fréquences cinématiques cibles
(porte, réducteur, moteur), visualisation comparative entre machines, et
indicateurs statistiques de dispersion.

## Structure du projet

```
vib_analyzer/
├── app.py                          # Interface Streamlit (affichage uniquement)
├── vibration_analysis/
│   ├── fft_core.py                 # Calcul FFT, extraction d'amplitude (logique pure, testée)
│   └── io_utils.py                 # Lecture et validation des fichiers Excel
├── tests/
│   └── test_fft_core.py            # Tests unitaires sur signaux synthétiques
├── requirements.txt
└── README.md
```

La logique métier (`vibration_analysis/`) est indépendante de Streamlit :
elle peut être testée, réutilisée en script batch, ou appelée depuis un
notebook sans dépendre de l'interface web.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

## Lancer les tests

```bash
pytest tests/ -v
```

## Format du fichier Excel attendu

Chaque onglet représente un système/machine, avec au minimum deux colonnes :

| Colonne 1 (temps) | Colonne 2 (signal) |
|---|---|
| numérique, croissant | vibration mesurée |

- La colonne temps est interprétée en **millisecondes par défaut**
  (réglable dans la barre latérale : Millisecondes / Secondes). Une erreur
  d'unité fausse tout l'axe fréquentiel d'un facteur 1000 — vérifiez la
  colonne "dt (s)" du tableau de résultats si un résultat semble incohérent.
- Les onglets ne respectant pas ce format (colonne temps non numérique,
  moins de 2 colonnes) sont automatiquement ignorés et signalés dans
  l'interface, sans bloquer l'analyse des autres onglets.

## Méthodes de calcul FFT

- **Ancien mode** : FFT sans fenêtre, lecture au point fréquentiel discret
  le plus proche de la fréquence cible.
- **Nouveau mode** : fenêtre de Hanning (réduit la fuite spectrale) +
  recherche du maximum local dans une fenêtre de ±3 % autour de la
  fréquence cible (tolère un léger décalage réel de la fréquence
  cinématique). Ce mode capte le pic le plus énergétique de la zone :
  en présence de bruit, il peut différer légèrement de la composante
  cinématique pure — voir la colonne de diagnostic "Résolution (Hz)".

## Limites connues

- Les seuils statistiques ("Moyenne + écart-type") sont calculés sur
  l'échantillon de machines du fichier importé, pas sur une baseline
  saine de référence : ils indiquent des machines atypiques *au sein
  du lot analysé*, pas un seuil de maintenance absolu.
- Pour les basses fréquences cibles (ex. 0.0167 Hz), un enregistrement
  trop court peut donner une résolution FFT insuffisante pour que la
  recherche de pic local (±3 %) apporte un bénéfice réel par rapport
  au mode Ancien — l'application avertit dans ce cas.
PYEOF_README_MD

mkdir -p $(dirname vibration_analysis/__init__.py)
cat > vibration_analysis/__init__.py << 'PYEOF_VIBRATION_ANALYSIS___INIT___PY'
"""Package d'analyse vibratoire par FFT."""

from .fft_core import (
    DonneesInsuffisantesError,
    ModeFFT,
    ResultatFFT,
    UniteTemps,
    analyser_systeme,
    calculer_fft,
    detecter_dt,
    extraire_amplitude,
    resolution_suffisante,
)
from .io_utils import OngletInvalideError, charger_systemes, lire_onglet

__all__ = [
    "DonneesInsuffisantesError",
    "ModeFFT",
    "ResultatFFT",
    "UniteTemps",
    "analyser_systeme",
    "calculer_fft",
    "detecter_dt",
    "extraire_amplitude",
    "resolution_suffisante",
    "OngletInvalideError",
    "charger_systemes",
    "lire_onglet",
]
PYEOF_VIBRATION_ANALYSIS___INIT___PY

mkdir -p $(dirname vibration_analysis/fft_core.py)
cat > vibration_analysis/fft_core.py << 'PYEOF_VIBRATION_ANALYSIS_FFT_CORE_PY'
"""
fft_core.py
-----------
Logique de calcul FFT et d'extraction d'amplitudes vibratoires.

Ce module est volontairement indépendant de Streamlit : toutes les fonctions
sont pures (entrées -> sorties), ce qui les rend testables unitairement et
réutilisables hors de l'application web (script batch, notebook, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


class ModeFFT(str, Enum):
    """Méthode de calcul de la FFT."""

    ANCIEN = "ancien"  # Sans fenêtre, lecture au point fréquentiel le plus proche
    NOUVEAU = "nouveau"  # Fenêtre de Hanning, recherche du pic local ±3 %

    @classmethod
    def from_label_ui(cls, label: str) -> "ModeFFT":
        """Convertit le libellé affiché dans l'UI Streamlit vers l'enum."""
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
    dt: float  # pas d'échantillonnage effectif, en secondes
    resolution_hz: float  # résolution fréquentielle = 1 / (N * dt)
    n_points: int


class DonneesInsuffisantesError(ValueError):
    """Levée quand un signal est trop court ou dégénéré pour être analysé."""


def detecter_dt(
    temps: np.ndarray, unite: UniteTemps = UniteTemps.MILLISECONDES
) -> float:
    """
    Détecte le pas d'échantillonnage (dt) en secondes à partir d'une colonne temps.

    Args:
        temps: valeurs temporelles brutes, dans l'unité indiquée par `unite`.
        unite: unité des valeurs de `temps` (millisecondes par défaut, comme
            dans les exports source historiques).

    Returns:
        Le pas d'échantillonnage médian, en secondes.

    Raises:
        DonneesInsuffisantesError: si moins de 2 points valides sont disponibles.
    """
    diffs = np.diff(temps.astype(float))
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        raise DonneesInsuffisantesError(
            "Impossible de déterminer le pas d'échantillonnage : "
            "la colonne temps ne contient pas d'intervalles positifs."
        )
    facteur = 1000.0 if unite == UniteTemps.MILLISECONDES else 1.0
    return float(np.median(diffs_pos) / facteur)


def calculer_fft(
    signal: np.ndarray,
    dt: float,
    mode: ModeFFT,
) -> ResultatFFT:
    """
    Calcule le spectre d'amplitude (FFT) d'un signal temporel.

    Args:
        signal: valeurs du signal (vibration), centré automatiquement sur sa moyenne.
        dt: pas d'échantillonnage en secondes.
        mode: ANCIEN (sans fenêtre) ou NOUVEAU (fenêtre de Hanning).

    Returns:
        Un ResultatFFT contenant les fréquences, amplitudes et métadonnées.

    Raises:
        DonneesInsuffisantesError: si le signal contient moins de 2 points
            ou uniquement des NaN.
    """
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
    """
    Extrait l'amplitude vibratoire à une fréquence cible donnée.

    En mode ANCIEN : lit l'amplitude au bin fréquentiel le plus proche.
    En mode NOUVEAU : cherche le maximum local dans une fenêtre
        [freq_cible * (1 - tolerance), freq_cible * (1 + tolerance)].
        Ceci capte le pic le plus énergétique de la zone, qui peut différer
        de la composante cinématique pure en présence de bruit ou de raies
        parasites proches — comportement voulu pour tolérer un léger
        décalage de fréquence, mais à garder en tête lors de l'interprétation.

    Args:
        resultat: sortie de `calculer_fft`.
        freq_cible: fréquence cinématique recherchée, en Hz.
        mode: méthode d'extraction (doit correspondre au mode utilisé pour la FFT).
        tolerance_relative: demi-largeur relative de la fenêtre de recherche
            en mode NOUVEAU (0.03 = ±3 %).

    Returns:
        L'amplitude extraite (même unité que le signal d'origine).
    """
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
    """
    Vérifie que la résolution fréquentielle permet de distinguer une fréquence
    cible dans sa fenêtre de tolérance (mode NOUVEAU).

    Si la résolution FFT est plus grossière que la largeur de la fenêtre de
    recherche, la recherche de "pic local" ne fait en pratique que retomber
    sur le bin le plus proche, comme en mode ANCIEN — ce qui peut induire
    l'utilisateur en erreur sur la précision réelle du résultat.

    Args:
        resultat: sortie de `calculer_fft`.
        freq_cible: fréquence cinématique en Hz.
        tolerance_relative: demi-largeur relative de la fenêtre (0.03 = ±3 %).

    Returns:
        True si au moins 2 bins FFT tombent dans la fenêtre de tolérance.
    """
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
    """
    Pipeline complet pour un système (un onglet Excel) : FFT + extraction
    de toutes les fréquences cibles + indicateurs agrégés.

    Args:
        df: DataFrame contenant au moins les colonnes temps et signal.
        col_temps: nom de la colonne temps.
        col_signal: nom de la colonne signal.
        mode: méthode de calcul FFT.
        freqs_cibles: dict {nom_composant: fréquence_hz}.
        unite_temps: unité de la colonne temps.

    Returns:
        Un dict avec les amplitudes par composant, la somme, le produit,
        et des métadonnées de diagnostic (dt, résolution, alertes).

    Raises:
        DonneesInsuffisantesError: propagée depuis `detecter_dt` / `calculer_fft`.
    """
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
PYEOF_VIBRATION_ANALYSIS_FFT_CORE_PY

mkdir -p $(dirname vibration_analysis/io_utils.py)
cat > vibration_analysis/io_utils.py << 'PYEOF_VIBRATION_ANALYSIS_IO_UTILS_PY'
"""
io_utils.py
-----------
Lecture et validation des fichiers Excel sources.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class OngletInvalideError(ValueError):
    """Levée quand un onglet Excel n'a pas le format attendu (temps, signal)."""


def lire_onglet(xls: pd.ExcelFile, nom_onglet: str) -> pd.DataFrame:
    """
    Lit un onglet et vérifie qu'il contient au minimum deux colonnes exploitables.

    Args:
        xls: fichier Excel ouvert (pd.ExcelFile).
        nom_onglet: nom de l'onglet à lire.

    Returns:
        Le DataFrame de l'onglet, colonnes renommées (espaces superflus retirés).

    Raises:
        OngletInvalideError: si l'onglet a moins de 2 colonnes exploitables,
            ou si la colonne temps n'est pas numérique.
    """
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
    """
    Parcourt tous les onglets d'un fichier Excel et isole les onglets valides
    des onglets en erreur, sans faire échouer l'ensemble du traitement.

    Args:
        uploaded_file: objet fichier (ex: retourné par st.file_uploader).

    Returns:
        Un tuple (systemes, erreurs) où :
        - systemes: liste de dicts {"nom": str, "df": DataFrame,
            "col_temps": str, "col_signal": str}
        - erreurs: liste de messages d'erreur, un par onglet en échec.
    """
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
        except Exception as exc:  # sécurité : un onglet corrompu ne bloque pas les autres
            logger.exception("Erreur inattendue sur l'onglet '%s'", nom_onglet)
            erreurs.append(f"Onglet '{nom_onglet}' : erreur inattendue ({exc}).")

    return systemes, erreurs
PYEOF_VIBRATION_ANALYSIS_IO_UTILS_PY

mkdir -p $(dirname app.py)
cat > app.py << 'PYEOF_APP_PY'
"""
app.py
------
Interface Streamlit de l'analyseur vibratoire FFT.

Ce fichier ne contient que la logique d'affichage : tout le calcul est
délégué au package `vibration_analysis` (voir fft_core.py / io_utils.py),
qui est testé indépendamment (cf. tests/).
"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from vibration_analysis import (
    DonneesInsuffisantesError,
    ModeFFT,
    UniteTemps,
    analyser_systeme,
    charger_systemes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Fréquences d'intérêt et leurs organes associés ---
FREQS_CIBLES = {
    "Porte (0.0167 Hz)": 0.016759,
    "1er étage réducteur (3.68 Hz)": 3.68,
    "Dernier étage réducteur (12.33 Hz)": 12.33,
    "Moteur (13.67 Hz)": 13.67,
}

st.set_page_config(page_title="Analyse Vibratoire FFT & Stats", layout="wide")
st.title("📊 Analyseur Vibratoire FFT - Visualisation Dynamique & Statistiques")
st.write("Calcul FFT, classement dynamique et indicateurs statistiques de maintenance.")

# --- SIDEBAR - CONFIGURATION ---
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
    """
    Wrapper cachable : Streamlit ne peut pas hasher un mode_calcul/unite_temps
    enum directement dans tous les cas, donc on passe des primitives et on
    reconstruit les enums à l'intérieur.
    """
    import io

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

    # Alerte résolution fréquentielle
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

    # --- Construction du tableau de résultats ---
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
PYEOF_APP_PY

mkdir -p $(dirname tests/__init__.py)
cat > tests/__init__.py << 'PYEOF_TESTS___INIT___PY'

PYEOF_TESTS___INIT___PY

mkdir -p $(dirname tests/test_fft_core.py)
cat > tests/test_fft_core.py << 'PYEOF_TESTS_TEST_FFT_CORE_PY'
"""
Tests unitaires du module fft_core.

Stratégie : on génère des signaux synthétiques (sinusoïdes de fréquence
connue) et on vérifie que le pipeline FFT retrouve bien l'amplitude et la
fréquence attendues, à une tolérance numérique près.
"""

import numpy as np
import pandas as pd
import pytest

from vibration_analysis.fft_core import (
    DonneesInsuffisantesError,
    ModeFFT,
    UniteTemps,
    analyser_systeme,
    calculer_fft,
    detecter_dt,
    extraire_amplitude,
    resolution_suffisante,
)


# --- Fixtures : signal synthétique ---


def generer_signal_sinusoidal(freq_hz: float, amplitude: float, duree_s: float, fs_hz: float):
    """Génère un signal sinus pur échantillonné, pour tester la détection de pic."""
    n = int(duree_s * fs_hz)
    t = np.arange(n) / fs_hz
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return t, signal


# --- Tests detecter_dt ---


class TestDetecterDt:
    def test_dt_millisecondes(self):
        temps_ms = np.array([0, 10, 20, 30, 40])  # pas de 10 ms
        dt = detecter_dt(temps_ms, unite=UniteTemps.MILLISECONDES)
        assert dt == pytest.approx(0.01)

    def test_dt_secondes(self):
        temps_s = np.array([0, 0.01, 0.02, 0.03])
        dt = detecter_dt(temps_s, unite=UniteTemps.SECONDES)
        assert dt == pytest.approx(0.01)

    def test_leve_erreur_si_temps_constant(self):
        temps = np.array([5, 5, 5, 5])
        with pytest.raises(DonneesInsuffisantesError):
            detecter_dt(temps)

    def test_ignore_les_decrements(self):
        # simule une ligne aberrante où le temps repart en arrière
        temps_ms = np.array([0, 10, 20, 5, 30, 40])
        dt = detecter_dt(temps_ms, unite=UniteTemps.MILLISECONDES)
        assert dt == pytest.approx(0.01)


# --- Tests calculer_fft ---


class TestCalculerFFT:
    def test_retrouve_amplitude_signal_pur(self):
        freq_cible, amp_cible = 10.0, 2.0
        _, signal = generer_signal_sinusoidal(freq_cible, amp_cible, duree_s=10, fs_hz=200)
        dt = 1 / 200

        resultat = calculer_fft(signal, dt=dt, mode=ModeFFT.ANCIEN)
        idx_pic = np.argmax(resultat.amplitude)

        assert resultat.freq[idx_pic] == pytest.approx(freq_cible, abs=0.2)
        assert resultat.amplitude[idx_pic] == pytest.approx(amp_cible, rel=0.05)

    def test_hanning_reduit_fuite_spectrale_vs_ancien(self):
        # Fréquence volontairement "off-bin" pour révéler la fuite spectrale
        freq_cible = 10.37
        _, signal = generer_signal_sinusoidal(freq_cible, 1.0, duree_s=10, fs_hz=200)
        dt = 1 / 200

        res_ancien = calculer_fft(signal, dt=dt, mode=ModeFFT.ANCIEN)
        res_hanning = calculer_fft(signal, dt=dt, mode=ModeFFT.NOUVEAU)

        # Les deux doivent repérer le pic proche de la fréquence cible
        idx_ancien = np.argmax(res_ancien.amplitude)
        idx_hanning = np.argmax(res_hanning.amplitude)
        assert res_ancien.freq[idx_ancien] == pytest.approx(freq_cible, abs=0.3)
        assert res_hanning.freq[idx_hanning] == pytest.approx(freq_cible, abs=0.3)

    def test_leve_erreur_si_signal_trop_court(self):
        with pytest.raises(DonneesInsuffisantesError):
            calculer_fft(np.array([1.0]), dt=0.01, mode=ModeFFT.ANCIEN)

    def test_leve_erreur_si_dt_invalide(self):
        with pytest.raises(DonneesInsuffisantesError):
            calculer_fft(np.array([1.0, 2.0, 3.0]), dt=0, mode=ModeFFT.ANCIEN)

    def test_ignore_les_nan(self):
        signal = np.array([1.0, np.nan, 2.0, np.nan, 1.0, 3.0, 1.0, 2.0])
        resultat = calculer_fft(signal, dt=0.01, mode=ModeFFT.ANCIEN)
        assert resultat.n_points == 6


# --- Tests extraire_amplitude ---


class TestExtraireAmplitude:
    def test_mode_ancien_prend_le_bin_le_plus_proche(self):
        freq_cible, amp_cible = 5.0, 3.0
        _, signal = generer_signal_sinusoidal(freq_cible, amp_cible, duree_s=20, fs_hz=100)
        resultat = calculer_fft(signal, dt=1 / 100, mode=ModeFFT.ANCIEN)

        amp_extraite = extraire_amplitude(resultat, freq_cible, mode=ModeFFT.ANCIEN)
        assert amp_extraite == pytest.approx(amp_cible, rel=0.05)

    def test_mode_nouveau_trouve_pic_decale(self):
        # Le pic réel est légèrement décalé de la fréquence "cible" nominale
        freq_reelle, freq_cible_nominale, amp_cible = 5.05, 5.0, 3.0
        _, signal = generer_signal_sinusoidal(freq_reelle, amp_cible, duree_s=20, fs_hz=100)
        resultat = calculer_fft(signal, dt=1 / 100, mode=ModeFFT.NOUVEAU)

        amp_extraite = extraire_amplitude(resultat, freq_cible_nominale, mode=ModeFFT.NOUVEAU)
        assert amp_extraite == pytest.approx(amp_cible, rel=0.1)

    def test_mode_nouveau_fallback_si_hors_fenetre(self):
        # Fréquence cible très éloignée du signal : ne doit pas planter,
        # doit retomber sur le bin le plus proche (fallback).
        _, signal = generer_signal_sinusoidal(5.0, 1.0, duree_s=20, fs_hz=100)
        resultat = calculer_fft(signal, dt=1 / 100, mode=ModeFFT.NOUVEAU)

        # Ne doit pas lever d'exception
        amp = extraire_amplitude(resultat, freq_cible=0.001, mode=ModeFFT.NOUVEAU)
        assert amp >= 0


# --- Tests resolution_suffisante ---


class TestResolutionSuffisante:
    def test_resolution_insuffisante_sur_basse_frequence_courte_duree(self):
        # 5 secondes d'enregistrement -> résolution ~0.2 Hz, bien trop grossière
        # pour distinguer 0.0167 Hz ± 3%.
        _, signal = generer_signal_sinusoidal(1.0, 1.0, duree_s=5, fs_hz=100)
        resultat = calculer_fft(signal, dt=1 / 100, mode=ModeFFT.NOUVEAU)

        assert resolution_suffisante(resultat, freq_cible=0.016759) is False

    def test_resolution_suffisante_avec_longue_duree(self):
        # Enregistrement long -> résolution fine, suffisante même pour 0.0167 Hz
        _, signal = generer_signal_sinusoidal(1.0, 1.0, duree_s=3600, fs_hz=10)
        resultat = calculer_fft(signal, dt=1 / 10, mode=ModeFFT.NOUVEAU)

        assert resolution_suffisante(resultat, freq_cible=0.016759) is True


# --- Test d'intégration : analyser_systeme ---


class TestAnalyserSysteme:
    def test_pipeline_complet_sur_dataframe(self):
        freq_cible, amp_cible = 13.67, 1.5  # correspond au "Moteur" dans FREQS_CIBLES
        t, signal = generer_signal_sinusoidal(freq_cible, amp_cible, duree_s=30, fs_hz=200)
        df = pd.DataFrame({"Temps": t * 1000, "Signal": signal})  # temps en ms

        resultat = analyser_systeme(
            df=df,
            col_temps="Temps",
            col_signal="Signal",
            mode=ModeFFT.NOUVEAU,
            freqs_cibles={"Moteur": freq_cible},
            unite_temps=UniteTemps.MILLISECONDES,
        )

        assert resultat["amplitudes"]["Moteur"] == pytest.approx(amp_cible, rel=0.1)
        assert resultat["somme"] == pytest.approx(amp_cible, rel=0.1)
        assert resultat["alertes_resolution"] == []
PYEOF_TESTS_TEST_FFT_CORE_PY

echo "Projet genere avec succes dans ./vib_analyzer"
echo "Prochaines etapes :"
echo "  cd vib_analyzer"
echo "  python -m venv venv && source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  pytest tests/ -v"
echo "  streamlit run app.py"
echo ""
echo "Pour initialiser git :"
echo "  git init && git add . && git commit -m 'Initial commit'"
