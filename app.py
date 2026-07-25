import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import streamlit as st

st.set_page_config(
    page_title="Analyse FFT - Mode Ancien Calcul", layout="wide"
)

st.title("📊 Analyseur FFT (Mode Ancien Calcul)")
st.write(
    "Calcul FFT direct sans fenêtrage et extraction de la fréquence fixe la plus proche."
)


# --- FONCTION FFT REPRODUISANT L'ANCIEN CALCUL ---
def calculer_fft_ancien_mode(df, col_temps, col_signal):
    x = df[col_signal].values.astype(float)
    N = len(x)

    if N == 0:
        raise ValueError("Le tableau de données est vide.")

    # Estimation du pas de temps dt
    diffs = np.diff(df[col_temps].values.astype(float))
    diffs_pos = diffs[diffs > 0]
    dt = np.median(diffs_pos) / 1000.0 if len(diffs_pos) > 0 else 0.01

    # Centrage du signal sans fenêtre
    x_centered = x - np.mean(x)

    # FFT Brute (Rectangle)
    fft_amp = np.abs(rfft(x_centered)) * (2.0 / N)
    freq = rfftfreq(N, d=dt)

    return freq, fft_amp, dt


def obtenir_amplitude_exacte(freq, amp, target_hz):
    # Index de la fréquence théorique la plus proche (point fixe exact)
    idx = np.argmin(np.abs(freq - target_hz))
    return freq[idx], amp[idx]


# --- INTERFACE STREAMLIT ---
st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier Excel (.xlsx)", type=["xlsx", "xls"]
)

freqs_cibles = {
    "Amplitude 0,0167 Hz": 0.016759,
    "Amplitude 3,68 Hz": 3.68,
    "Amplitude 12,33 Hz": 12.33,
    "Amplitude 13,67 Hz": 13.67,
}

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        resultats = []

        progress_bar = st.progress(0)
        num_sheets = len(xls.sheet_names)

        for i, nom_onglet in enumerate(xls.sheet_names):
            df = pd.read_excel(xls, sheet_name=nom_onglet)
            df.columns = [str(c).strip() for c in df.columns]

            if len(df.columns) < 2:
                continue

            col_temps = df.columns[0]
            col_signal = df.columns[1]

            freq, amp, dt = calculer_fft_ancien_mode(
                df, col_temps=col_temps, col_signal=col_signal
            )

            ligne_res = {
                "Système / Onglet": nom_onglet,
            }

            amplitudes_extraites = []
            for label, f_cible in freqs_cibles.items():
                _, amp_val = obtenir_amplitude_exacte(freq, amp, f_cible)
                ligne_res[label] = round(amp_val, 6)
                amplitudes_extraites.append(amp_val)

            produit_amp = np.prod(amplitudes_extraites)
            somme_amp = np.sum(amplitudes_extraites)

            ligne_res["Produit des Fréquences"] = f"{produit_amp:.2e}"
            ligne_res["Somme des Fréquences"] = round(somme_amp, 6)

            resultats.append(ligne_res)
            progress_bar.progress((i + 1) / num_sheets)

        progress_bar.empty()

        df_resultats = pd.DataFrame(resultats)
        st.subheader("📋 Synthèse des résultats (Conforme aux anciens calculs)")
        st.dataframe(df_resultats, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
