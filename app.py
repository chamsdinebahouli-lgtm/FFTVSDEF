"""
app.py
------
Analyseur Électrique DC & Diagnostique Vibratoire - Application Streamlit (fichier unique).
Somme des amplitudes calculée strictement par la somme des 4 fréquences ciblées.
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

# Importation optionnelle de ReportLab pour les PDF professionnels
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

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
    """Calcule l'ensemble des indicateurs statistiques et électriques."""
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

    # Somme stricte des 4 amplitudes ciblées
    somme_amp_cibles = float(sum(amplitudes_cibles.values()))

    return {
        **metriques,
        "cibles": amplitudes_cibles,
        "Somme des amplitudes": somme_amp_cibles,
        "dt": resultat_fft.dt,
        "resolution_hz": resultat_fft.resolution_hz,
        "n_points": resultat_fft.n_points,
        "alertes_resolution": alertes_resolution,
    }


def lire_onglet(xls: pd.ExcelFile, nom_onglet: str) -> pd.DataFrame:
    """Lit un onglet et vérifie qu'il contient au moins deux colonnes exploitables."""
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


def generer_pdf_rapport_complet(df_res: pd.DataFrame, resultats_bruts: list[dict], metrique_maitresse: str) -> bytes:
    """Génère un rapport PDF global et détaillé pour tout le parc."""
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab non disponible.")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=20,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=5
    )

    # PAGE 1 : SYNTHÈSE GLOBALE DU PARC
    elements.append(Paragraph("Rapport Global de Diagnostic Électromécanique & Vibratoire", title_style))
    elements.append(Paragraph("Synthèse multicritère de l'ensemble du parc machine", subtitle_style))

    elements.append(Paragraph("1. Vue d'ensemble du parc", heading_style))
    n_machines = len(df_res)
    elements.append(Paragraph(f"Nombre total de systèmes / machines analysés : <b>{n_machines}</b>", body_style))
    elements.append(Paragraph(f"Indicateur maître de classement : <b>{metrique_maitresse}</b>", body_style))

    if metrique_maitresse in df_res.columns:
        val_moy = df_res[metrique_maitresse].mean()
        val_max = df_res[metrique_maitresse].max()
        machine_max = df_res.loc[df_res[metrique_maitresse].idxmax(), "Système / Machine"] if n_machines > 0 else "N/A"
        elements.append(Paragraph(f"• Moyenne du parc pour {metrique_maitresse} : <b>{val_moy:.4f}</b>", body_style))
        elements.append(Paragraph(f"• Valeur maximale : <b>{val_max:.4f}</b> (Machine critique : <b>{machine_max}</b>)", body_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("2. Tableau Récapitulatif Global", heading_style))

    cols_a_afficher = ["Système / Machine", "Offset DC (V)", "RMS Total (V)", "Crest Factor", "Kurtosis", "THD (%)", "Somme des amplitudes"]
    cols_pdf = [c for c in cols_a_afficher if c in df_res.columns]

    data_table = [cols_pdf]
    for _, row in df_res.iterrows():
        data_table.append([str(row[c]) for c in cols_pdf])

    t = Table(data_table, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F3F4F6")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
    ]))
    elements.append(t)

    # FICHES DÉTAILLÉES PAR MACHINE / ONGLET
    for r in resultats_bruts:
        elements.append(PageBreak())
        nom_machine = r.get("nom", "Inconnu")
        elements.append(Paragraph(f"Fiche Détaillée Machine : {nom_machine}", title_style))
        elements.append(Paragraph("Paramètres physiques, électriques et spectrales extraits", subtitle_style))

        elements.append(Paragraph("Indicateurs Électriques & Statistiques", heading_style))
        fiche_metriques = [
            ["Indicateur", "Valeur"],
            ["Offset DC (V)", f"{r.get('Offset DC (V)', 0):.4f}"],
            ["RMS Total (V)", f"{r.get('RMS Total (V)', 0):.4f}"],
            ["RMS AC (V)", f"{r.get('RMS AC (V)', 0):.4f}"],
            ["Peak (V)", f"{r.get('Peak (V)', 0):.4f}"],
            ["Peak-to-Peak (V)", f"{r.get('Peak-to-Peak (V)', 0):.4f}"],
            ["Facteur de Crête (Crest Factor)", f"{r.get('Crest Factor', 0):.3f}"],
            ["Kurtosis", f"{r.get('Kurtosis', 0):.3f}"],
            ["Skewness", f"{r.get('Skewness', 0):.3f}"],
            ["THD (%)", f"{r.get('THD (%)', 0):.2f}%"],
            ["SNR (dB)", f"{r.get('SNR (dB)', 0):.2f} dB"],
            ["Somme des amplitudes (4 cibles)", f"{r.get('Somme des amplitudes', 0):.4f}"],
        ]
        t_met = Table(fiche_metriques, repeatRows=1)
        t_met.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4B5563")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))
        elements.append(t_met)

        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Amplitudes Spectrales par Composante Cible", heading_style))
        cibles_data = [["Composante / Fréquence Cible", "Amplitude (V)"]]
        for comp, val in r.get("cibles", {}).items():
            cibles_data.append([comp, f"{val:.5f} V"])

        t_cib = Table(cibles_data, repeatRows=1)
        t_cib.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))
        elements.append(t_cib)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generer_html_rapport_complet(df_res: pd.DataFrame, resultats_bruts: list[dict], metrique_maitresse: str) -> str:
    """Génère un rapport HTML complet multi-machines imprimable en PDF."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Rapport Global de Diagnostic</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; color: #1F2937; }}
            h1 {{ color: #1E3A8A; text-align: center; }}
            h2 {{ color: #1E3A8A; border-bottom: 2px solid #1E3A8A; padding-bottom: 5px; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; font-size: 11px; }}
            th, td {{ border: 1px solid #D1D5DB; padding: 6px; text-align: center; }}
            th {{ background-color: #1E3A8A; color: white; }}
            tr:nth-child(even) {{ background-color: #F3F4F6; }}
            .summary {{ background: #EFF6FF; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <h1>Rapport Global de Diagnostic Électromécanique & Vibratoire</h1>
        <div class="summary">
            <h2>Vue d'Ensemble du Parc</h2>
            <p>Nombre total de systèmes analysés : <b>{len(df_res)}</b></p>
            <p>Indicateur maître : <b>{metrique_maitresse}</b></p>
        </div>
        <h2>Tableau Récapitulatif Global</h2>
        {df_res.to_html(index=False, classes='table')}
    """

    for r in resultats_bruts:
        nom_machine = r.get("nom", "Inconnu")
        html += f"""
        <div class="page-break"></div>
        <h1>Fiche Détaillée Machine : {nom_machine}</h1>
        <h2>Indicateurs Électriques & Statistiques</h2>
        <table>
            <tr><th>Indicateur</th><th>Valeur</th></tr>
            <tr><td>Offset DC (V)</td><td>{r.get('Offset DC (V)', 0):.4f}</td></tr>
            <tr><td>RMS Total (V)</td><td>{r.get('RMS Total (V)', 0):.4f}</td></tr>
            <tr><td>RMS AC (V)</td><td>{r.get('RMS AC (V)', 0):.4f}</td></tr>
            <tr><td>Peak (V)</td><td>{r.get('Peak (V)', 0):.4f}</td></tr>
            <tr><td>Peak-to-Peak (V)</td><td>{r.get('Peak-to-Peak (V)', 0):.4f}</td></tr>
            <tr><td>Facteur de Crête</td><td>{r.get('Crest Factor', 0):.3f}</td></tr>
            <tr><td>Kurtosis</td><td>{r.get('Kurtosis', 0):.3f}</td></tr>
            <tr><td>Skewness</td><td>{r.get('Skewness', 0):.3f}</td></tr>
            <tr><td>THD (%)</td><td>{r.get('THD (%)', 0):.2f}%</td></tr>
            <tr><td>SNR (dB)</td><td>{r.get('SNR (dB)', 0):.2f} dB</td></tr>
            <tr><td>Somme des amplitudes (4 cibles)</td><td>{r.get('Somme des amplitudes', 0):.4f}</td></tr>
        </table>
        <h2>Amplitudes Spectrales par Composante</h2>
        <table>
            <tr><th>Composante / Fréquence Cible</th><th>Amplitude (V)</th></tr>
        """
        for comp, val in r.get("cibles", {}).items():
            html += f"<tr><td>{comp}</td><td>{val:.5f} V</td></tr>"
        html += "</table>"

    html += "</body></html>"
    return html


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

FREQS_CIBLES = {
    "Rotation 1 tr/min door (0.0167 Hz)": 0.016759,
    "1er étage réducteur (3.68 Hz)": 3.68,
    "Dernier étage réducteur (12.33 Hz)": 12.33,
    "Moteur / Commutation (13.67 Hz)": 13.67,
}

st.set_page_config(page_title="Diagnostic Électromécanique DC & FFT", layout="wide")
st.title("⚡ Analyseur Avancé : Couplage Électrique & Vibratoire (1 tr/min)")
st.write("Suivi multi-indicateurs professionnels : Kurtosis, Facteur de Crête, THD, Somme des amplitudes ciblées et Analyse Spectrale.")

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
            "Somme des amplitudes": round(r.get("Somme des amplitudes", 0.0), 4),
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
        "Somme des amplitudes",
        "RMS Total (V)",
        "Offset DC (V)",
    ]

    metriques_existantes = [m for m in metriques_disponibles if m in df_res.columns]
    cols_cibles = [col for col in FREQS_CIBLES.keys() if col in df_res.columns]

    if cols_cibles:
        df_melted = pd.melt(
            df_res,
            id_vars=["Système / Machine"],
            value_vars=cols_cibles,
            var_name="Composante / Fréquence Cible",
            value_name="Amplitude Spectrale (V)",
        )
        df_melted = df_melted.merge(df_res[["Système / Machine"] + metriques_existantes], on="Système / Machine", how="left")
    else:
        df_melted = pd.DataFrame(columns=["Système / Machine", "Composante / Fréquence Cible", "Amplitude Spectrale (V)"] + metriques_existantes)

    st.markdown("---")
    st.subheader("📈 Visualisation & Diagnostic Dynamique")

    col_gauche, col_droite = st.columns(2)
    with col_gauche:
        metrique_maitresse = st.selectbox(
            "Indicateur principal à analyser / classer :",
            options=metriques_existantes if metriques_existantes else ["Système / Machine"],
            index=0,
            help="Sélectionnez un indicateur sensible pour identifier les machines atypiques.",
        )
    with col_droite:
        sens_tri = st.radio("Ordre de classement :", ["Du plus faible au plus fort", "Du plus fort au plus faible"], horizontal=True)

    ascending_flag = True if sens_tri.startswith("Du plus faible") else False
    ordre_systemes = df_res.sort_values(by=metrique_maitresse, ascending=ascending_flag)["Système / Machine"].tolist() if metrique_maitresse in df_res.columns else []

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
        if not df_melted.empty:
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
        else:
            st.info("Aucune donnée spectrale disponible pour le graphique empilé.")

    with tab3:
        st.markdown("#### Analyse Ciblée par Composante Fréquentielle")
        composant_selectionne = st.selectbox(
            "Sélectionner la fréquence/composante à inspecter :",
            options=["Toutes les composantes"] + cols_cibles,
        )

        if not df_melted.empty:
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
            st.info("Aucune donnée disponible pour l'analyse ciblée.")

    # =========================================================================
    # EXPORTS : CSV & RAPPORT GLOBAL & DÉTAILLÉ
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 Exportation des Résultats & Rapports Complets")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger le tableau synthétique (CSV)",
            data=csv_bytes,
            file_name="synthese_indicateurs_electromecaniques.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_exp2:
        if HAS_REPORTLAB:
            try:
                pdf_bytes = generer_pdf_rapport_complet(df_res, resultats, metrique_maitresse)
                st.download_button(
                    label="📄 Télécharger le Rapport Global Complet (PDF)",
                    data=pdf_bytes,
                    file_name="rapport_global_diagnostic_parc.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Erreur de génération PDF : {e}")
        else:
            html_content = generer_html_rapport_complet(df_res, resultats, metrique_maitresse)
            st.download_button(
                label="📄 Télécharger le Rapport Global Complet (HTML / Imprimable PDF)",
                data=html_content.encode("utf-8"),
                file_name="rapport_global_diagnostic_parc.html",
                mime="text/html",
                use_container_width=True,
                help="Ouvrez ce fichier dans votre navigateur puis faites Ctrl+P -> Enregistrer au format PDF pour obtenir le rapport complet de tout le parc.",
            )

else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale pour lancer l'analyse avancée.")
