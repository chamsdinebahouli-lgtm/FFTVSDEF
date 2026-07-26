"""
app.py
------
Analyseur Électrique DC & Diagnostic Vibratoire - Application Streamlit (fichier unique).

Le signal source est une tension moteur DC (pas un capteur de vibration) :
les amplitudes spectrales sont normalisées par la composante continue (DC)
du signal pour rester comparables entre machines quel que soit le gain de
la chaîne d'acquisition. Certains indicateurs (THD, SNR) sont calculés de
façon simplifiée et explicitement labellisés "indicatif" : ce ne sont PAS
les définitions normées (IEC 61000 pour le THD notamment) — voir les
docstrings de `calculer_metriques_avancees` pour le détail.
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
from scipy.signal import find_peaks
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
# LOGIQUE MÉTIER : FFT, normalisation DC, statistiques électriques
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
    dc: float  # composante continue (valeur moyenne) du signal brut, avant centrage


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
        raise DonneesInsuffisantesError(f"Signal trop court pour un calcul FFT (n={n} points valides).")
    if dt <= 0:
        raise DonneesInsuffisantesError(f"Pas d'échantillonnage invalide (dt={dt}).")

    dc = float(np.mean(x))
    x_centre = x - dc

    if mode == ModeFFT.ANCIEN:
        amplitude = np.abs(rfft(x_centre)) * (2.0 / n)
    else:
        fenetre = np.hanning(n)
        amplitude = np.abs(rfft(x_centre * fenetre)) * (2.0 / np.sum(fenetre))

    freq = rfftfreq(n, d=dt)
    resolution_hz = 1.0 / (n * dt)

    return ResultatFFT(
        freq=freq, amplitude=amplitude, dt=dt, resolution_hz=resolution_hz, n_points=n, dc=dc
    )


def amplitude_relative_pct(amplitude: float, resultat: ResultatFFT) -> float:
    """
    Exprime une amplitude en pourcentage de la composante DC du signal
    (taux de modulation), pour rendre les mesures comparables entre systèmes
    indépendamment du gain de la chaîne d'acquisition.

    Retourne 0.0 si la composante DC est nulle ou quasi nulle (évite une
    division par zéro / valeur aberrante).
    """
    if abs(resultat.dc) < 1e-9:
        return 0.0
    return float(amplitude / abs(resultat.dc) * 100.0)


def calculer_metriques_avancees(signal: np.ndarray, resultat_fft: ResultatFFT) -> dict[str, float]:
    """
    Calcule les indicateurs statistiques et électriques globaux du signal.

    ATTENTION - "Distorsion Spectrale" et "Ratio Pic/Bruit" :
    Ces deux indicateurs sont calculés de façon simplifiée à partir du
    spectre complet (pic maximum vs médiane / reste de l'énergie), PAS selon
    les définitions normées habituelles (le THD au sens IEC 61000 se calcule
    à partir de l'énergie des harmoniques de la fondamentale uniquement, et
    le SNR nécessite une séparation signal/bruit fiable). Ils sont donc
    explicitement suffixés "(indicatif)" et ne doivent pas être comparés à
    des seuils issus de fiches techniques utilisant les vraies définitions.

    Kurtosis et Skewness sont des moments statistiques centrés (invariants à
    l'offset DC par construction), donc valides même calculés sur le signal
    brut. Le Kurtosis est un indicateur reconnu en analyse vibratoire pour
    détecter des chocs/impacts (défauts de roulement typiquement) — sa
    valeur de référence usuelle (~3 pour un signal proche gaussien) reste
    indicative ici puisque le signal est une tension moteur, pas une
    vibration mécanique directe.
    """
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
        "Distorsion Spectrale (%) - indicatif": 0.0,
        "Ratio Pic/Bruit (dB) - indicatif": 0.0,
    }

    if n == 0:
        return metriques_par_defaut

    try:
        dc_val = resultat_fft.dc
        rms_total = float(np.sqrt(np.mean(x**2)))

        x_ac = x - dc_val
        rms_ac = float(np.sqrt(np.mean(x_ac**2)))

        peak_val = float(np.max(np.abs(x_ac)))
        min_val, max_val = float(np.min(x)), float(np.max(x))
        peak_to_peak = max_val - min_val

        crest_factor = float(peak_val / rms_ac) if rms_ac > 1e-9 else 1.0

        kurt = float(kurtosis(x, fisher=False, bias=False)) if n > 3 else 3.0
        skw = float(skew(x, bias=False)) if n > 2 else 0.0

        freq, amp = resultat_fft.freq, resultat_fft.amplitude
        energie_totale_spec = float(np.sum(amp**2))

        if energie_totale_spec > 1e-9 and len(amp) > 0:
            bruit_estime = float(np.median(amp))
            signal_utile_estime = float(np.max(amp))
            snr = float(20 * np.log10(signal_utile_estime / bruit_estime)) if bruit_estime > 1e-9 else 0.0
            thd = float(
                (np.sqrt(max(0, energie_totale_spec - signal_utile_estime**2)) / (signal_utile_estime + 1e-9))
                * 100.0
            )
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
            "Distorsion Spectrale (%) - indicatif": thd,
            "Ratio Pic/Bruit (dB) - indicatif": snr,
        }
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques avancées : {e}")
        return metriques_par_defaut


def extraire_amplitude(
    resultat: ResultatFFT,
    freq_cible: float,
    mode: ModeFFT,
    tolerance_relative: float = 0.03,
) -> float:
    """Extrait l'amplitude à une fréquence cible donnée."""
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


def identifier_frequence_cible_proche(
    freq_detectee: float, freqs_cibles: dict[str, float], tolerance_relative: float = 0.05
) -> tuple[str | None, float | None]:
    """
    Rapproche une fréquence détectée automatiquement d'une fréquence cible
    connue (Porte, Réducteur, Moteur...), si elle est suffisamment proche.

    Contrairement à un nommage codé en dur par plage de fréquence, cette
    fonction s'appuie sur les fréquences cinématiques réelles du système
    (déjà définies dans FREQS_CIBLES), donc reste cohérente avec le reste
    de l'application.

    Returns:
        (nom_composant, écart_relatif) si une correspondance est trouvée
        dans la tolérance, sinon (None, None).
    """
    meilleur_nom, meilleur_ecart = None, None
    for nom, f_cible in freqs_cibles.items():
        if f_cible <= 0:
            continue
        ecart_relatif = abs(freq_detectee - f_cible) / f_cible
        if ecart_relatif <= tolerance_relative and (meilleur_ecart is None or ecart_relatif < meilleur_ecart):
            meilleur_nom, meilleur_ecart = nom, ecart_relatif
    return meilleur_nom, meilleur_ecart


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

    amplitudes_v: dict[str, float] = {}
    amplitudes_pct_dc: dict[str, float] = {}
    alertes_resolution: list[str] = []
    tolerance_relative = 0.03

    for nom_composant, f_cible in freqs_cibles.items():
        amp = extraire_amplitude(resultat_fft, f_cible, mode, tolerance_relative)
        amplitudes_v[nom_composant] = amp
        amplitudes_pct_dc[nom_composant] = amplitude_relative_pct(amp, resultat_fft)
        if mode == ModeFFT.NOUVEAU and not resolution_suffisante(resultat_fft, f_cible, tolerance_relative):
            alertes_resolution.append(nom_composant)

    return {
        **metriques,
        "cibles_v": amplitudes_v,
        "cibles_pct_dc": amplitudes_pct_dc,
        # Somme des amplitudes des composantes cibles. Additionne des
        # fréquences correspondant à des mécanismes de défaut différents,
        # donc pas un indicateur de santé mécanique validé scientifiquement.
        # Conservée à la demande explicite de l'utilisateur pour son suivi
        # personnel (tendance globale au fil des imports) : NE PAS
        # interpréter comme un diagnostic, chaque composant individuel reste
        # la référence pour un diagnostic mécanique.
        "Somme des amplitudes (indicatif)": float(sum(amplitudes_pct_dc.values())),
        "dc": resultat_fft.dc,
        "dt": resultat_fft.dt,
        "resolution_hz": resultat_fft.resolution_hz,
        "n_points": resultat_fft.n_points,
        "alertes_resolution": alertes_resolution,
        # Spectre complet conservé pour la détection automatique de pics
        # (onglet dédié) : évite de refaire un import/calcul FFT séparé.
        "spectre_freq": resultat_fft.freq,
        "spectre_amplitude_v": resultat_fft.amplitude,
        "spectre_amplitude_pct_dc": (
            resultat_fft.amplitude / abs(resultat_fft.dc) * 100.0
            if abs(resultat_fft.dc) > 1e-9
            else np.zeros_like(resultat_fft.amplitude)
        ),
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


# =============================================================================
# GÉNÉRATION DE RAPPORTS (PDF / HTML)
# =============================================================================


def generer_pdf_rapport_complet(
    df_res: pd.DataFrame, resultats_bruts: list[dict], metrique_maitresse: str, unite_amplitude: str
) -> bytes:
    """Génère un rapport PDF global et détaillé pour tout le parc."""
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab non disponible.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"], fontSize=18,
        textColor=colors.HexColor("#1E3A8A"), spaceAfter=12, alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#4B5563"), spaceAfter=20, alignment=1,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#1E3A8A"), spaceBefore=12, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#1F2937"), spaceAfter=5,
    )
    caveat_style = ParagraphStyle(
        "CaveatStyle", parent=styles["Normal"], fontSize=7.5,
        textColor=colors.HexColor("#B45309"), spaceAfter=10, alignment=0,
    )

    elements.append(Paragraph("Rapport Global de Diagnostic Électromécanique", title_style))
    elements.append(Paragraph("Synthèse multicritère de l'ensemble du parc machine", subtitle_style))
    elements.append(Paragraph(
        "⚠ Signal source : tension moteur DC (pas un capteur de vibration calibré). "
        "Les indicateurs 'Distorsion Spectrale' et 'Ratio Pic/Bruit' sont des indicateurs "
        "indicatifs maison, PAS les définitions normées (THD IEC 61000, SNR classique). "
        "La 'Somme des amplitudes' n'est pas un indicateur de santé mécanique global.",
        caveat_style,
    ))

    elements.append(Paragraph("1. Vue d'ensemble du parc", heading_style))
    n_machines = len(df_res)
    elements.append(Paragraph(f"Nombre total de systèmes / machines analysés : <b>{n_machines}</b>", body_style))
    elements.append(Paragraph(f"Indicateur maître de classement : <b>{metrique_maitresse}</b>", body_style))

    if metrique_maitresse in df_res.columns and n_machines > 0:
        val_moy = df_res[metrique_maitresse].mean()
        val_max = df_res[metrique_maitresse].max()
        machine_max = df_res.loc[df_res[metrique_maitresse].idxmax(), "Système / Machine"]
        elements.append(Paragraph(f"• Moyenne du parc pour {metrique_maitresse} : <b>{val_moy:.4f}</b>", body_style))
        elements.append(Paragraph(f"• Valeur maximale : <b>{val_max:.4f}</b> (Machine : <b>{machine_max}</b>)", body_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("2. Tableau Récapitulatif Global", heading_style))

    cols_a_afficher = [
        "Système / Machine", "Offset DC (V)", "RMS Total (V)", "Crest Factor",
        "Kurtosis", "Somme des amplitudes (indicatif)",
    ]
    cols_pdf = [c for c in cols_a_afficher if c in df_res.columns]

    data_table = [cols_pdf]
    for _, row in df_res.iterrows():
        data_table.append([str(row[c]) for c in cols_pdf])

    t = Table(data_table, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F4F6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
    ]))
    elements.append(t)

    for r in resultats_bruts:
        elements.append(PageBreak())
        nom_machine = r.get("nom", "Inconnu")
        elements.append(Paragraph(f"Fiche Détaillée Machine : {nom_machine}", title_style))
        elements.append(Paragraph("Paramètres électriques et spectraux extraits", subtitle_style))

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
            ["Distorsion Spectrale (%) - indicatif", f"{r.get('Distorsion Spectrale (%) - indicatif', 0):.2f}%"],
            ["Ratio Pic/Bruit (dB) - indicatif", f"{r.get('Ratio Pic/Bruit (dB) - indicatif', 0):.2f} dB"],
            ["Somme des amplitudes (indicatif)", f"{r.get('Somme des amplitudes (indicatif)', 0):.4f}"],
        ]
        t_met = Table(fiche_metriques, repeatRows=1)
        t_met.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B5563")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ]))
        elements.append(t_met)

        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Amplitudes Spectrales par Composante Cible ({unite_amplitude})", heading_style))
        cle_cibles = "cibles_pct_dc" if unite_amplitude == "% DC" else "cibles_v"
        cibles_data = [["Composante / Fréquence Cible", f"Amplitude ({unite_amplitude})"]]
        for comp, val in r.get(cle_cibles, {}).items():
            cibles_data.append([comp, f"{val:.5f}"])

        t_cib = Table(cibles_data, repeatRows=1)
        t_cib.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ]))
        elements.append(t_cib)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generer_html_rapport_complet(
    df_res: pd.DataFrame, resultats_bruts: list[dict], metrique_maitresse: str, unite_amplitude: str
) -> str:
    """Génère un rapport HTML complet multi-machines imprimable en PDF."""
    cle_cibles = "cibles_pct_dc" if unite_amplitude == "% DC" else "cibles_v"

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
            .caveat {{ background: #FFFBEB; border-left: 4px solid #B45309; padding: 10px; font-size: 12px; margin-bottom: 15px; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <h1>Rapport Global de Diagnostic Électromécanique</h1>
        <div class="caveat">
            ⚠ Signal source : tension moteur DC (pas un capteur de vibration calibré).
            "Distorsion Spectrale" et "Ratio Pic/Bruit" sont des indicateurs indicatifs
            maison, pas les définitions normées (THD IEC 61000, SNR classique).
            La "Somme des amplitudes" n'est pas un indicateur de santé mécanique global.
        </div>
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
            <tr><td>Distorsion Spectrale (%) - indicatif</td><td>{r.get('Distorsion Spectrale (%) - indicatif', 0):.2f}%</td></tr>
            <tr><td>Ratio Pic/Bruit (dB) - indicatif</td><td>{r.get('Ratio Pic/Bruit (dB) - indicatif', 0):.2f} dB</td></tr>
            <tr><td>Somme des amplitudes (indicatif)</td><td>{r.get('Somme des amplitudes (indicatif)', 0):.4f}</td></tr>
        </table>
        <h2>Amplitudes Spectrales par Composante ({unite_amplitude})</h2>
        <table>
            <tr><th>Composante / Fréquence Cible</th><th>Amplitude ({unite_amplitude})</th></tr>
        """
        for comp, val in r.get(cle_cibles, {}).items():
            html += f"<tr><td>{comp}</td><td>{val:.5f}</td></tr>"
        html += "</table>"

    html += "</body></html>"
    return html


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
st.title("⚡ Analyseur Avancé : Signal Électrique DC & Diagnostic Spectral")
st.write(
    "Suivi multi-indicateurs (Kurtosis, Facteur de Crête, indicateurs spectraux) "
    "à partir d'une tension moteur DC. Amplitudes normalisées par la composante "
    "continue pour rester comparables entre machines."
)

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
st.sidebar.header("📐 Normalisation")
st.sidebar.caption(
    "Le signal est une tension moteur DC : l'amplitude brute en Volts dépend "
    "du gain de la chaîne d'acquisition. La normalisation par la composante "
    "continue (DC) donne un taux de modulation indépendant de ce gain — "
    "recommandé pour comparer plusieurs machines entre elles."
)
normaliser_dc = st.sidebar.checkbox("Normaliser par la composante DC (% de modulation)", value=True)
unite_amplitude = "% DC" if normaliser_dc else "V"

st.sidebar.markdown("---")
st.sidebar.header("📊 Indicateurs Statistiques")
afficher_moyenne = st.sidebar.checkbox("Afficher la Ligne de Moyenne du lot", value=True)
afficher_seuil = st.sidebar.checkbox(
    "Afficher les Machines Atypiques du Lot (Moyenne + 1σ)",
    value=True,
    help="Ce repère est calculé sur les machines du fichier importé, pas sur une "
    "baseline saine de référence : il signale une dispersion relative au sein "
    "du lot, pas un seuil de maintenance absolu.",
)

uploaded_file = st.sidebar.file_uploader("Importer le fichier Excel (.xlsx)", type=["xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.header("🔗 Journal de Défectivité (optionnel)")
st.sidebar.caption(
    "Constats terrain (défauts observés hors de cette application), pour "
    "tenter des corrélations avec les indicateurs calculés. Renseigne un "
    "niveau par système dans le tableau principal, puis exporte le journal "
    "pour le réimporter la prochaine fois."
)
fichier_defectivite = st.sidebar.file_uploader(
    "Importer un journal existant (CSV)", type=["csv"], key="import_defectivite"
)


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
                df=df, col_temps=df.columns[0], col_signal=df.columns[1],
                mode=mode, freqs_cibles=FREQS_CIBLES, unite_temps=unite,
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

    alertes_globales = {r["nom"]: r["alertes_resolution"] for r in resultats if r["alertes_resolution"]}
    if alertes_globales:
        with st.expander("⚠️ Résolution fréquentielle insuffisante sur certains organes", expanded=False):
            for nom, composants in alertes_globales.items():
                st.write(f"- **{nom}** : {', '.join(composants)}")

    cle_cibles = "cibles_pct_dc" if normaliser_dc else "cibles_v"

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
            "Distorsion Spectrale (%) - indicatif": round(r.get("Distorsion Spectrale (%) - indicatif", 0.0), 2),
            "Ratio Pic/Bruit (dB) - indicatif": round(r.get("Ratio Pic/Bruit (dB) - indicatif", 0.0), 2),
            "Somme des amplitudes (indicatif)": round(r.get("Somme des amplitudes (indicatif)", 0.0), 4),
        }
        amplitudes_cibles = r.get(cle_cibles, {})
        for comp, val in amplitudes_cibles.items():
            ligne[comp] = round(val, 5)

        lignes.append(ligne)

    df_res = pd.DataFrame(lignes)

    # -------------------------------------------------------------------
    # DÉFECTIVITÉ : constats terrain saisis manuellement, pour tenter des
    # corrélations avec les indicateurs calculés. Stockée en texte (pas en
    # nombre) pendant la saisie pour accepter aussi bien "1.33" que "1,33"
    # sans piège de conversion ni arrondi d'affichage.
    # -------------------------------------------------------------------
    if fichier_defectivite is not None:
        df_defect_importe = None
        derniere_erreur = None
        for kwargs_lecture in ({"sep": ",", "decimal": "."}, {"sep": ";", "decimal": ","}):
            try:
                fichier_defectivite.seek(0)
                candidat = pd.read_csv(fichier_defectivite, dtype=str, **kwargs_lecture)
                if "Niveau de défectivité" in candidat.columns and "Système / Machine" in candidat.columns:
                    df_defect_importe = candidat
                    break
            except Exception as exc:
                derniere_erreur = exc
        if df_defect_importe is None:
            st.sidebar.warning(
                "Journal de défectivité illisible : vérifie que les colonnes "
                "s'appellent exactement 'Système / Machine' et 'Niveau de "
                f"défectivité'. Erreur technique : {derniere_erreur}"
            )
            df_defect_importe = pd.DataFrame(columns=["Système / Machine", "Niveau de défectivité"])
    else:
        df_defect_importe = pd.DataFrame(columns=["Système / Machine", "Niveau de défectivité"])

    df_res = df_res.merge(df_defect_importe, on="Système / Machine", how="left")
    df_res["Niveau de défectivité"] = df_res["Niveau de défectivité"].fillna("0").astype(str)

    st.subheader("📋 Tableau Synthétique - Indicateurs Électromécaniques")
    if normaliser_dc:
        st.caption(
            "Amplitudes par composante exprimées en % de la composante DC (taux "
            "de modulation), comparables entre machines même avec des chaînes "
            "d'acquisition différentes."
        )
    else:
        st.caption(
            "⚠️ Amplitudes par composante en Volts bruts : dépendent du gain de "
            "la chaîne d'acquisition, à ne comparer qu'entre mesures faites avec "
            "exactement le même matériel."
        )
    st.caption(
        "💡 La colonne **'Niveau de défectivité'** est éditable directement "
        "ci-dessous (double-clic sur une cellule) : renseigne tes constats "
        "terrain avec point OU virgule (ex: 1.3295615 ou 1,3295615), les "
        "deux fonctionnent. Le reste du tableau est calculé automatiquement "
        "et non modifiable."
    )
    colonnes_non_editables = [c for c in df_res.columns if c != "Niveau de défectivité"]
    df_res = st.data_editor(
        df_res,
        use_container_width=True,
        hide_index=True,
        disabled=colonnes_non_editables,
        column_config={
            "Niveau de défectivité": st.column_config.TextColumn(
                help="Constat terrain, pas calculé par l'application. Point ou virgule acceptés.",
            ),
        },
        key="editeur_tableau_principal",
    )

    # Conversion texte -> numérique pour tous les usages en aval (graphiques,
    # corrélation, export). Accepte point et virgule comme séparateur décimal.
    df_res["Niveau de défectivité"] = pd.to_numeric(
        df_res["Niveau de défectivité"].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)

    csv_defect_bytes = df_res[["Système / Machine", "Niveau de défectivité"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Télécharger le journal de défectivité seul (CSV) — pour le réimporter la prochaine fois",
        data=csv_defect_bytes,
        file_name="journal_defectivite.csv",
        mime="text/csv",
    )

    st.caption(
        "'Distorsion Spectrale' et 'Ratio Pic/Bruit' sont des indicateurs "
        "indicatifs maison (voir info-bulle du code), pas les définitions "
        "normées THD/SNR. La somme des amplitudes sert uniquement à ordonner "
        "visuellement les machines, ce n'est pas un indicateur de santé "
        "mécanique global — chaque composant doit être évalué individuellement."
    )

    metriques_disponibles = [
        "Kurtosis", "Crest Factor", "RMS AC (V)", "Peak-to-Peak (V)",
        "Distorsion Spectrale (%) - indicatif", "Ratio Pic/Bruit (dB) - indicatif",
        "Somme des amplitudes (indicatif)", "RMS Total (V)", "Offset DC (V)",
    ]
    metriques_existantes = [m for m in metriques_disponibles if m in df_res.columns]
    cols_cibles = [col for col in FREQS_CIBLES.keys() if col in df_res.columns]

    if cols_cibles:
        df_melted = pd.melt(
            df_res, id_vars=["Système / Machine"], value_vars=cols_cibles,
            var_name="Composante / Fréquence Cible", value_name=f"Amplitude ({unite_amplitude})",
        )
        df_melted = df_melted.merge(
            df_res[["Système / Machine"] + metriques_existantes], on="Système / Machine", how="left"
        )
    else:
        df_melted = pd.DataFrame(
            columns=["Système / Machine", "Composante / Fréquence Cible", f"Amplitude ({unite_amplitude})"]
            + metriques_existantes
        )

    st.markdown("---")
    st.subheader("📈 Visualisation & Diagnostic Dynamique")

    col_gauche, col_droite = st.columns(2)
    with col_gauche:
        metrique_maitresse = st.selectbox(
            "Indicateur principal à analyser / classer :",
            options=metriques_existantes if metriques_existantes else ["Système / Machine"],
            index=0,
            help="Sélectionnez un indicateur pour identifier les machines atypiques du lot.",
        )
    with col_droite:
        sens_tri = st.radio("Ordre de classement :", ["Du plus faible au plus fort", "Du plus fort au plus faible"], horizontal=True)

    ascending_flag = sens_tri.startswith("Du plus faible")
    ordre_systemes = (
        df_res.sort_values(by=metrique_maitresse, ascending=ascending_flag)["Système / Machine"].tolist()
        if metrique_maitresse in df_res.columns else []
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"📊 Classement Global ({metrique_maitresse})",
        "📶 Spectre des Fréquences Cibles",
        "🔍 Focus par Composant Mécanique",
        "🔗 Corrélation avec Défectivité",
        "🔎 Détection Auto de Pics",
    ])

    with tab1:
        st.markdown(f"#### Classement du parc selon l'indicateur : **{metrique_maitresse}**")
        fig_global = px.bar(
            df_res, x="Système / Machine", y=metrique_maitresse,
            title=f"Classement des machines par {metrique_maitresse}",
            text_auto=".2f", category_orders={"Système / Machine": ordre_systemes},
            color=metrique_maitresse, color_continuous_scale="Viridis",
        )

        val_moy = df_res[metrique_maitresse].mean()
        val_std = df_res[metrique_maitresse].std()

        if afficher_moyenne:
            fig_global.add_hline(
                y=val_moy, line_dash="dash", line_color="blue",
                annotation_text=f"Moyenne du lot: {val_moy:.2f}", annotation_position="bottom right",
            )
        if afficher_seuil and not pd.isna(val_std):
            fig_global.add_hline(
                y=val_moy + val_std, line_dash="dot", line_color="orange",
                annotation_text=f"Atypique du lot (Moy+σ): {val_moy + val_std:.2f}",
                annotation_position="top right",
            )

        st.plotly_chart(fig_global, use_container_width=True)

    with tab2:
        st.markdown("#### Amplitudes spectrales par machine (Empilées)")
        st.caption("Empilement à but de lecture visuelle : la hauteur cumulée n'est pas un indicateur de sévérité globale.")
        if not df_melted.empty:
            fig_stacked = px.bar(
                df_melted, x="Système / Machine", y=f"Amplitude ({unite_amplitude})",
                color="Composante / Fréquence Cible", title="Contribution des Fréquences Cibles",
                barmode="stack", category_orders={"Système / Machine": ordre_systemes},
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.info("Aucune donnée spectrale disponible pour le graphique empilé.")

    with tab3:
        st.markdown("#### Analyse Ciblée par Composante Fréquentielle")
        composant_selectionne = st.selectbox(
            "Sélectionner la fréquence/composante à inspecter :", options=cols_cibles,
        )

        if not df_melted.empty and composant_selectionne:
            df_filtre = df_melted[df_melted["Composante / Fréquence Cible"] == composant_selectionne]
            df_filtre = df_filtre.sort_values(by=f"Amplitude ({unite_amplitude})", ascending=ascending_flag)

            fig_single = px.bar(
                df_filtre, x="Système / Machine", y=f"Amplitude ({unite_amplitude})",
                title=f"Zoom sur : {composant_selectionne}", text_auto=".4f",
            )

            valeurs_stats = df_filtre[f"Amplitude ({unite_amplitude})"]
            if len(valeurs_stats) > 0:
                moy_comp, std_comp = valeurs_stats.mean(), valeurs_stats.std()
                if afficher_moyenne:
                    fig_single.add_hline(
                        y=moy_comp, line_dash="dash", line_color="blue",
                        annotation_text=f"Moyenne du lot: {moy_comp:.4f}", annotation_position="bottom right",
                    )
                if afficher_seuil and not pd.isna(std_comp):
                    fig_single.add_hline(
                        y=moy_comp + std_comp, line_dash="dot", line_color="orange",
                        annotation_text=f"Atypique du lot (Moy+σ): {moy_comp + std_comp:.4f}",
                        annotation_position="top right",
                    )

            st.plotly_chart(fig_single, use_container_width=True)
        else:
            st.info("Aucune donnée disponible pour l'analyse ciblée.")

    with tab4:
        st.markdown("#### Corrélation entre les indicateurs calculés et tes constats de défectivité")
        st.caption(
            "Objectif : repérer quel(s) indicateur(s) varient le plus avec le "
            "niveau de défaut que tu mesures sur le terrain. Deux coefficients "
            "affichés (-1 à +1, proche de 0 = pas de lien) : **Pearson** suppose "
            "une relation linéaire et est sensible à l'échelle exacte des "
            "valeurs ; **Spearman** ne regarde que l'ordre des valeurs (relation "
            "monotone, pas forcément linéaire) et est plus robuste si tu n'es "
            "pas sûr de la forme de la relation. À interpréter avec prudence "
            "avec peu de machines (les deux deviennent instables en dessous "
            "d'une dizaine de points)."
        )

        if df_res["Niveau de défectivité"].nunique() <= 1:
            st.info(
                "Renseigne des niveaux de défectivité différents d'au moins deux "
                "systèmes dans le journal ci-dessus pour activer l'analyse de "
                "corrélation."
            )
        else:
            colonnes_indicateurs = [
                c for c in df_res.columns
                if c not in ("Système / Machine", "Niveau de défectivité")
                and pd.api.types.is_numeric_dtype(df_res[c])
            ]

            matrice_pearson = df_res[colonnes_indicateurs + ["Niveau de défectivité"]].corr(
                method="pearson", numeric_only=True
            )
            matrice_spearman = df_res[colonnes_indicateurs + ["Niveau de défectivité"]].corr(
                method="spearman", numeric_only=True
            )

            if "Niveau de défectivité" not in matrice_pearson.columns:
                st.warning(
                    "Impossible de calculer la corrélation : la colonne 'Niveau de "
                    "défectivité' n'a pas pu être traitée comme numérique. "
                    "Réessaie après avoir vérifié les valeurs saisies dans le "
                    "journal ci-dessus (uniquement des nombres)."
                )
                correlations = pd.Series(dtype=float)
            else:
                pearson_s = matrice_pearson["Niveau de défectivité"].drop("Niveau de défectivité")
                spearman_s = matrice_spearman["Niveau de défectivité"].drop("Niveau de défectivité")
                correlations = pearson_s.dropna().sort_values(key=lambda s: s.abs(), ascending=False)

            if correlations.empty:
                if "Niveau de défectivité" in matrice_pearson.columns:
                    st.info("Pas assez de variation dans les données pour calculer une corrélation exploitable.")
            else:
                df_corr = pd.DataFrame({
                    "Indicateur": correlations.index,
                    "Pearson (linéaire)": correlations.values,
                    "Spearman (monotone)": spearman_s.reindex(correlations.index).values,
                })
                df_corr_melted = df_corr.melt(
                    id_vars="Indicateur", value_vars=["Pearson (linéaire)", "Spearman (monotone)"],
                    var_name="Méthode", value_name="Coefficient de corrélation",
                )

                fig_corr = px.bar(
                    df_corr_melted, x="Coefficient de corrélation", y="Indicateur",
                    color="Méthode", barmode="group",
                    orientation="h", title="Indicateurs classés par force de corrélation (tri sur Pearson)",
                    range_x=[-1, 1],
                )
                fig_corr.update_layout(yaxis={"categoryorder": "array", "categoryarray": correlations.index[::-1].tolist()})
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("##### Nuage de points — inspection détaillée d'un indicateur")
                indicateur_focus = st.selectbox(
                    "Indicateur à inspecter :", options=df_corr["Indicateur"].tolist(),
                )
                try:
                    fig_scatter = px.scatter(
                        df_res, x="Niveau de défectivité", y=indicateur_focus,
                        text="Système / Machine", title=f"{indicateur_focus} vs Niveau de défectivité",
                        trendline="ols",
                    )
                except Exception:
                    # Repli si statsmodels n'est pas installé (trendline indisponible)
                    fig_scatter = px.scatter(
                        df_res, x="Niveau de défectivité", y=indicateur_focus,
                        text="Système / Machine", title=f"{indicateur_focus} vs Niveau de défectivité",
                    )
                fig_scatter.update_traces(textposition="top center")
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.caption(
                    "La droite de tendance (régression linéaire) est indicative : "
                    "avec peu de machines, ne pas sur-interpréter sa pente."
                )

    with tab5:
        st.markdown("#### Détection automatique de pics — un système à la fois")
        st.caption(
            "Complément aux 4 fréquences cibles suivies par ailleurs : repère "
            "d'éventuels pics significatifs à d'autres fréquences, que tu "
            "n'aurais pas anticipées. Analyse un système à la fois (le spectre "
            "complet n'a de sens que pour une machine précise)."
        )

        noms_systemes = [r["nom"] for r in resultats]
        systeme_focus = st.selectbox("Système à inspecter :", options=noms_systemes, key="select_pics_auto")
        r_focus = next(r for r in resultats if r["nom"] == systeme_focus)

        freq_spectre = r_focus["spectre_freq"]
        amp_spectre = r_focus["spectre_amplitude_pct_dc"] if normaliser_dc else r_focus["spectre_amplitude_v"]

        col_seuil, col_dist = st.columns(2)
        with col_seuil:
            seuil_pic_abs = st.number_input(
                f"Seuil minimal des pics ({unite_amplitude})",
                min_value=0.0, value=0.0, step=0.01,
                help="Amplitude absolue minimale pour qu'un pic soit retenu, dans "
                "l'unité actuellement sélectionnée. Contrairement à un seuil en "
                "% du maximum du spectre, une valeur absolue reste cohérente "
                "d'une machine à l'autre.",
            )
        with col_dist:
            separation_min_hz = st.number_input(
                "Séparation minimale entre deux pics (Hz)",
                min_value=0.0, value=max(r_focus["resolution_hz"] * 3, 0.01), step=0.01,
                help="Exprimée en Hz (pas en nombre d'échantillons), pour rester "
                "cohérente même si les machines ont des durées d'enregistrement "
                "différentes.",
            )

        distance_echantillons = max(1, int(round(separation_min_hz / r_focus["resolution_hz"])))
        hauteur_min = seuil_pic_abs if seuil_pic_abs > 0 else None
        indices_pics, _ = find_peaks(amp_spectre, height=hauteur_min, distance=distance_echantillons)

        if len(indices_pics) == 0:
            st.info("Aucun pic détecté avec ces réglages. Baisse le seuil minimal si besoin.")
        else:
            lignes_pics = []
            for idx in indices_pics:
                f_val = float(freq_spectre[idx])
                a_val = float(amp_spectre[idx])
                nom_proche, ecart = identifier_frequence_cible_proche(f_val, FREQS_CIBLES)
                correspondance = (
                    f"≈ {nom_proche} (écart {ecart * 100:.1f}%)" if nom_proche else "Aucune fréquence cible connue proche"
                )
                lignes_pics.append({
                    "Fréquence (Hz)": round(f_val, 5),
                    f"Amplitude ({unite_amplitude})": round(a_val, 5),
                    "Correspondance": correspondance,
                })

            df_pics = pd.DataFrame(lignes_pics).sort_values(
                by=f"Amplitude ({unite_amplitude})", ascending=False
            ).reset_index(drop=True)

            st.dataframe(df_pics, use_container_width=True, hide_index=True)
            st.caption(
                f"Somme des pics détectés et affichés ci-dessus (indicatif, "
                f"distinct de la 'Somme des amplitudes' des 4 fréquences "
                f"cibles) : **{df_pics[f'Amplitude ({unite_amplitude})'].sum():.4f} {unite_amplitude}**"
            )

            fig_spectre = px.line(
                x=freq_spectre, y=amp_spectre,
                labels={"x": "Fréquence (Hz)", "y": f"Amplitude ({unite_amplitude})"},
                title=f"Spectre complet — {systeme_focus}",
            )
            fig_spectre.add_scatter(
                x=df_pics["Fréquence (Hz)"], y=df_pics[f"Amplitude ({unite_amplitude})"],
                mode="markers", marker=dict(color="red", size=9, symbol="x"),
                name="Pics détectés",
            )
            for _, ligne_cible in pd.DataFrame(
                {"nom": list(FREQS_CIBLES.keys()), "freq": list(FREQS_CIBLES.values())}
            ).iterrows():
                fig_spectre.add_vline(
                    x=ligne_cible["freq"], line_dash="dot", line_color="gray", opacity=0.5,
                )
            st.plotly_chart(fig_spectre, use_container_width=True)
            st.caption("Lignes verticales grises : les 4 fréquences cibles suivies par ailleurs, pour repère visuel.")

    st.markdown("---")
    st.subheader("📥 Exportation des Résultats & Rapports Complets")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger le tableau synthétique (CSV)", data=csv_bytes,
            file_name="synthese_indicateurs_electromecaniques.csv", mime="text/csv",
            use_container_width=True,
        )

    with col_exp2:
        if HAS_REPORTLAB:
            try:
                pdf_bytes = generer_pdf_rapport_complet(df_res, resultats, metrique_maitresse, unite_amplitude)
                st.download_button(
                    "📄 Télécharger le Rapport Global Complet (PDF)", data=pdf_bytes,
                    file_name="rapport_global_diagnostic_parc.pdf", mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Erreur de génération PDF : {e}")
        else:
            html_content = generer_html_rapport_complet(df_res, resultats, metrique_maitresse, unite_amplitude)
            st.download_button(
                "📄 Télécharger le Rapport Global Complet (HTML / Imprimable PDF)",
                data=html_content.encode("utf-8"),
                file_name="rapport_global_diagnostic_parc.html", mime="text/html",
                use_container_width=True,
                help="Ouvrez ce fichier dans votre navigateur puis Ctrl+P -> Enregistrer au format PDF.",
            )

else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale pour lancer l'analyse.")
