import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse FFT Corrigée", layout="wide")

st.title("📊 Analyseur FFT de Signal")
st.write(
    "Chargez votre fichier Excel pour calculer les spectre FFT avec correction automatique du temps."
)


# --- FONCTIONS DE CALCUL FFT ---
def calculer_fft_signal(df, col_temps, col_signal, dt_force=None):
    x = df[col_signal].values.astype(float)
    N = len(x)

    if N == 0:
        raise ValueError("Le tableau de données est vide.")

    if dt_force is not None and dt_force > 0:
        dt = float(dt_force)
    else:
        # Correction des saut arrières/reset d'horloge
        diffs = np.diff(df[col_temps].values.astype(float))
        diffs_positives = diffs[diffs > 0]

        if len(diffs_positives) == 0:
            raise ValueError(
                "Impossible de calculer dt depuis la colonne temps."
            )

        dt_ms = np.median(diffs_positives)
        dt = dt_ms / 1000.0  # Conversion ms -> s

    # Centrage et fenêtrage Hanning
    x_centered = x - np.mean(x)
    fenetre = np.hanning(N)
    x_fen = x_centered * fenetre

    # Calcul FFT
    somme_fenetre = np.sum(fenetre)
    fft_amp = np.abs(rfft(x_fen)) * (2.0 / somme_fenetre)
    freq = rfftfreq(N, d=dt)

    return freq, fft_amp, dt


def obtenir_amplitude_a_frequence(freq, amp, target_hz, tol_pct=0.03):
    tol_hz = target_hz * tol_pct
    fmin, fmax = max(0.0, target_hz - tol_hz), target_hz + tol_hz
    mask = (freq >= fmin) & (freq <= fmax)

    if np.any(mask):
        sub_indices = np.where(mask)[0]
        max_sub_idx = sub_indices[np.argmax(amp[mask])]
        return freq[max_sub_idx], amp[max_sub_idx]
    else:
        idx = np.argmin(np.abs(freq - target_hz))
        return freq[idx], amp[idx]


# --- INTERFACE STREAMLIT ---

# 1. Barre latérale : upload du fichier
st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier Excel (.xlsx)", type=["xlsx", "xls"]
)

# Configuration des fréquences cibles à analyser
freqs_cibles = {
    "Amplitude 0,016759 Hz": 0.016759,
    "Amplitude 3,68 Hz": 3.68,
    "Amplitude 12,33 Hz": 12.33,
    "Amplitude 13,67 Hz": 13.67,
}

if uploaded_file is not None:
    try:
        # Lecture du fichier uploadé directement en mémoire
        xls = pd.ExcelFile(uploaded_file)
        resultats = []

        # Barre de progression Streamlit
        progress_bar = st.progress(0)
        num_sheets = len(xls.sheet_names)

        for i, nom_onglet in enumerate(xls.sheet_names):
            df = pd.read_excel(xls, sheet_name=nom_onglet)
            df.columns = [str(c).strip() for c in df.columns]

            if len(df.columns) < 2:
                continue

            col_temps = df.columns[0]
            col_signal = df.columns[1]

            # Calcul FFT
            freq, amp, dt = calculer_fft_signal(
                df, col_temps=col_temps, col_signal=col_signal
            )

            ligne_res = {
                "Système / Onglet": nom_onglet,
                "Nb Points": len(df),
                "dt (s)": round(dt, 4),
                "Durée Totale (s)": round(len(df) * dt, 2),
            }

            for label, f_cible in freqs_cibles.items():
                _, amp_val = obtenir_amplitude_a_frequence(freq, amp, f_cible)
                ligne_res[label] = round(amp_val, 6)

            resultats.append(ligne_res)
            progress_bar.progress((i + 1) / num_sheets)

        progress_bar.empty()

        # Affichage du tableau interactif sous Streamlit
        df_resultats = pd.DataFrame(resultats)

        st.subheader("📋 Synthèse des amplitudes FFT calculées")
        st.dataframe(df_resultats, use_container_width=True)

        # Option de téléchargement au format CSV ou Excel
        csv = df_resultats.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Télécharger les résultats (CSV)",
            data=csv,
            file_name="resultats_fft.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("👈 Veuillez importer un fichier Excel dans le panneau de gauche.")
