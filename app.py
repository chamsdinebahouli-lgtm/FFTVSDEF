
import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Analyse FFT Motoréducteur", layout="wide")

# ---------- Fonctions ----------

def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)
    x = x - np.mean(x)

    dt = np.mean(np.diff(t))
    N = len(x)

    amp = np.abs(rfft(x))
    freq = rfftfreq(N, d=dt)

    return freq, amp

def amplitude_freq(freq, amp, cible):
    idx = np.argmin(np.abs(freq - cible))
    return float(amp[idx])

def energie_totale(amp):
    return float(np.sum(amp**2))

def entropie_spectrale(amp):
    p = amp**2
    p = p / np.sum(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def energie_bande(freq, amp, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    return float(np.sum(amp[mask]**2))

def extraire_indicateurs(freq, amp):
    A1366 = amplitude_freq(freq, amp, 13.66)
    Etotal = energie_totale(amp)
    H = entropie_spectrale(amp)
    E05 = energie_bande(freq, amp, 0, 5)
    E1020 = energie_bande(freq, amp, 10, 20)

    IDM3 = (A1366**2 / Etotal) * H if Etotal > 0 else 0

    return {
        "A13.66": A1366,
        "Etotal": Etotal,
        "Entropie": H,
        "E0_5": E05,
        "E10_20": E1020,
        "IDM3": IDM3
    }

# ---------- Interface ----------

st.title("Analyse FFT Motoréducteur - V3")

uploaded = st.file_uploader("Importer un fichier Excel multi-onglets", type=["xlsx"])

if uploaded:

    xls = pd.ExcelFile(uploaded)

    resultats = []
    fft_data = {}

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded, sheet_name=sheet)

            if not {"ms", "V"}.issubset(df.columns):
                continue

            freq, amp = calcul_fft(df)

            indic = extraire_indicateurs(freq, amp)
            indic["Ensemble"] = sheet

            resultats.append(indic)
            fft_data[sheet] = (freq, amp)

        except Exception as e:
            st.warning(f"Erreur sur {sheet}: {e}")

    if resultats:

        resultats = pd.DataFrame(resultats)

        st.subheader("Résultats FFT")
        st.dataframe(resultats, use_container_width=True)

        st.subheader("FFT")
        choix = st.selectbox("Ensemble", resultats["Ensemble"])

        freq, amp = fft_data[choix]

        fft_df = pd.DataFrame({
            "Fréquence": freq,
            "Amplitude": amp
        })

        fig = px.line(
            fft_df,
            x="Fréquence",
            y="Amplitude",
            title=f"FFT - {choix}"
        )
        fig.update_xaxes(range=[0, 25])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Notes terrain (optionnel)")

        notes_txt = st.text_area(
            "Format : ASM21A=2.44 (une ligne par ensemble)",
            height=150
        )

        if notes_txt.strip():

            notes = {}

            for ligne in notes_txt.splitlines():
                if "=" in ligne:
                    k, v = ligne.split("=")
                    notes[k.strip()] = float(v.strip())

            resultats["Defaut"] = resultats["Ensemble"].map(notes)

            st.dataframe(resultats, use_container_width=True)

            modele_df = resultats.dropna(subset=["Defaut"]).copy()

            if len(modele_df) >= 5:

                features = [
                    "A13.66",
                    "Entropie",
                    "E0_5",
                    "E10_20",
                    "IDM3"
                ]

                X = modele_df[features]
                y = modele_df["Defaut"]

                model = RandomForestRegressor(
                    n_estimators=300,
                    random_state=42
                )

                model.fit(X, y)

                resultats["Prediction"] = model.predict(
                    resultats[features]
                )

                corr = resultats["IDM3"].corr(
                    resultats["Defaut"]
                )

                st.metric(
                    "Corrélation IDM3 / Défaut",
                    f"{corr:.3f}"
                )

                st.subheader("Prédictions")

                st.dataframe(
                    resultats[
                        ["Ensemble",
                         "Defaut",
                         "Prediction",
                         "IDM3"]
                    ],
                    use_container_width=True
                )

                imp = pd.DataFrame({
                    "Variable": features,
                    "Importance": model.feature_importances_
                }).sort_values(
                    "Importance",
                    ascending=False
                )

                st.subheader("Importance des variables")

                fig2 = px.bar(
                    imp,
                    x="Variable",
                    y="Importance"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        sortie = BytesIO()

        with pd.ExcelWriter(
            sortie,
            engine="openpyxl"
        ) as writer:

            resultats.to_excel(
                writer,
                index=False,
                sheet_name="Resultats"
            )

        st.download_button(
            "Télécharger les résultats",
            sortie.getvalue(),
            file_name="Resultats_FFT.xlsx"
        )
