import numpy as np
import pandas as pd
import plotly.express as px
from scipy.fft import rfft, rfftfreq
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Analyse Vibratoire FFT & Graphiques", layout="wide"
)

st.title("📊 Analyseur Vibratoire FFT & Visualisation par Composant")
st.write(
    "Calcul FFT et décomposition par organe mécanique (Porte, Réducteur, Moteur)."
)

# --- SIDEBAR - CONFIGURATION ---
st.sidebar.header("⚙️ Paramètres")
mode_calcul = st.sidebar.radio(
    "Méthode FFT :",
    [
        "Ancien Mode (Sans fenêtre, point fixe)",
        "Nouveau Mode (Hanning, pic local ±3%)",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Importer le fichier Excel (.xlsx)", type=["xlsx", "xls"]
)

# Fréquences d'intérêt et leurs organes associés
freqs_cibles = {
    "Porte (0.0167 Hz)": 0.016759,
    "1er étage réducteur (3.68 Hz)": 3.68,
    "Dernier étage réducteur (12.33 Hz)": 12.33,
    "Moteur (13.67 Hz)": 13.67,
}


# --- FONCTIONS DE CALCUL FFT ---
def calculer_fft(df, col_temps, col_signal, mode):
    x = df[col_signal].values.astype(float)
    N = len(x)

    diffs = np.diff(df[col_temps].values.astype(float))
    diffs_pos = diffs[diffs > 0]
    dt = np.median(diffs_pos) / 1000.0 if len(diffs_pos) > 0 else 0.01

    x_centered = x - np.mean(x)

    if mode.startswith("Ancien"):
        # Sans fenêtrage (Rectangulaire)
        fft_amp = np.abs(rfft(x_centered)) * (2.0 / N)
    else:
        # Fenêtre de Hanning
        fenetre = np.hanning(N)
        fft_amp = np.abs(rfft(x_centered * fenetre)) * (2.0 / np.sum(fenetre))

    freq = rfftfreq(N, d=dt)
    return freq, fft_amp


def extraire_amplitude(freq, amp, f_cible, mode):
    if mode.startswith("Ancien"):
        # Index de la fréquence théorique la plus proche
        idx = np.argmin(np.abs(freq - f_cible))
        return amp[idx]
    else:
        # Recherche du maximum dans la plage ±3%
        f_min, f_max = f_cible * 0.97, f_cible * 1.03
        mask = (freq >= f_min) & (freq <= f_max)
        if np.any(mask):
            return np.max(amp[mask])
        else:
            idx = np.argmin(np.abs(freq - f_cible))
            return amp[idx]


# --- TRAITEMENT DU FICHIER EXCEL ---
if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        data_rows = []

        for nom_onglet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=nom_onglet)
            df.columns = [str(c).strip() for c in df.columns]

            if len(df.columns) < 2:
                continue

            col_t, col_sig = df.columns[0], df.columns[1]
            freq, amp = calculer_fft(df, col_t, col_sig, mode_calcul)

            row = {"Système": nom_onglet}
            amps = []
            for comp, f_c in freqs_cibles.items():
                val = extraire_amplitude(freq, amp, f_c, mode_calcul)
                row[comp] = round(val, 6)
                amps.append(val)

            row["Somme (V)"] = round(np.sum(amps), 6)
            row["Produit"] = f"{np.prod(amps):.2e}"
            data_rows.append(row)

        df_res = pd.DataFrame(data_rows)

        # 1. Tableau de résultats
        st.subheader("📋 Tableau Synthétique des Amplitudes")
        st.dataframe(df_res, use_container_width=True)

        # Transformation des données pour la visualisation
        cols_composants = list(freqs_cibles.keys())
        df_melted = df_res.melt(
            id_vars=["Système"],
            value_vars=cols_composants,
            var_name="Composant Kinématique",
            value_name="Amplitude (V)",
        )

        st.markdown("---")
        st.subheader("📈 Représentation Graphique des Vibrations")

        # 2. Onglets de visualisation Plotly
        tab1, tab2 = st.tabs(
            [
                "📊 Barres Empilées (Niveau Cumulé)",
                "📶 Barres Groupées (Par Organe)",
            ]
        )

        with tab1:
            st.markdown(
                "#### Contribution de chaque organe à l'amplitude globale"
            )
            fig_stacked = px.bar(
                df_melted,
                x="Système",
                y="Amplitude (V)",
                color="Composant Kinématique",
                title="Amplitude Vibratoire Cumulée par Machine",
                barmode="stack",
                text_auto=".3f",
            )
            fig_stacked.update_layout(
                xaxis_title="Système / Machine",
                yaxis_title="Amplitude Cumulée (V)",
                legend_title="Organe Mécanique",
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

        with tab2:
            st.markdown(
                "#### Comparaison de l'état vibratoire par type d'organe"
            )
            fig_grouped = px.bar(
                df_melted,
                x="Système",
                y="Amplitude (V)",
                color="Composant Kinématique",
                title="Amplitudes Vibratoires par Composant",
                barmode="group",
            )
            fig_grouped.update_layout(
                xaxis_title="Système / Machine",
                yaxis_title="Amplitude (V)",
                legend_title="Organe Mécanique",
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("👈 Veuillez importer votre fichier Excel dans la barre latérale.")
