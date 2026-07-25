import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

# --------------------------------------------------
# CONFIGURATION DE L'APPLICATION
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT - Extracteur Multi-Fréquences",
    layout="wide"
)

st.title("Extraction d'Amplitudes Spécifiques et Corrélations")

# --------------------------------------------------
# FONCTIONS FFT & SIGNAL
# --------------------------------------------------
def calcul_fft(df):
    """FFT standard avec fenêtre de Hanning."""
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)
    x = x - np.mean(x)
    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre
    dt = np.mean(np.diff(t))
    N = len(x)
    fft_amp = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)
    return freq, fft_amp

def extract_amp(freq, amp, target_hz, tol_pct=0.05):
    """Extrait l'amplitude maximale dans une bande de tolérance en % autour de la cible."""
    # Calcul de la fenêtre de tolérance dynamique
    tol_hz = target_hz * tol_pct
    fmin, fmax = target_hz - tol_hz, target_hz + tol_hz
    
    # Sécurité absolue : on empêche de descendre à 0 Hz (composante continue)
    fmin = max(0.005, fmin) 

    mask = (freq >= fmin) & (freq <= fmax)
    
    if np.any(mask):
        # On renvoie le pic maximum trouvé DANS la fenêtre
        return float(np.max(amp[mask]))
    
    # Sécurité si aucun point ne tombe dans la fenêtre (cas de très basse résolution)
    idx = np.argmin(np.abs(freq - target_hz))
    return float(amp[idx])

# --------------------------------------------------
# INTERFACE BARRE LATÉRALE (SIDEBAR)
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Recherche Intelligente des Pics")
# On utilise maintenant un pourcentage (ex: 2% = 0.02)
tol_recherche_pct = st.sidebar.slider(
    "Tolérance de recherche autour de la cible (%) :", 
    min_value=0.5, max_value=10.0, value=3.0, step=0.5,
    help="Le script cherchera le pic d'amplitude maximum dans cette plage. Ex: Pour 13.67 Hz à 3%, il cherche entre 13.26 Hz et 14.08 Hz."
) / 100.0

st.sidebar.markdown("---")
notes_text = st.sidebar.text_area(
    "📝 Mean SUMOFDEF (1 year) :", 
    value="A21A=1.3295615\nA21B=1.3798294\nA22A=1.1538701\nA22B=1.1731472\nA32A=1.2950152\nA42A=1.3890785\nA71A=1.7692308\nA71B=1.9425951", 
    height=200,
    help="Renseignez ici vos valeurs de défaut au format NomMachine=Valeur"
)

# --------------------------------------------------
# LOGIQUE PRINCIPALE DE CALCUL
# --------------------------------------------------
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    resultats = []
    
    notes = {line.split("=")[0].strip(): float(line.split("=")[1].strip()) for line in notes_text.splitlines() if "=" in line}

    FREQS_CIBLES = {
        "Amplitude à 0,0167 Hz": 0.0167,
        "Amplitude à 3,68 Hz (V)": 3.68,
        "Amplitude à 12,33 Hz": 12.33,
        "Amplitude à 13,67 Hz": 13.67
    }

    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): 
                continue

            freq, amp = calcul_fft(df)

            res_row = {
                "Système": feuille,
                "Mean SUMOFDEF (1 year)": notes.get(feuille, np.nan)
            }

            somme_freqs = 0.0
            produit_freqs = 1.0
            
            for col_name, f_cible in FREQS_CIBLES.items():
                # Appel de la fonction avec la tolérance en pourcentage
                amp_val = extract_amp(freq, amp, f_cible, tol_pct=tol_recherche_pct)
                res_row[col_name] = amp_val
                somme_freqs += amp_val
                produit_freqs *= amp_val

            res_row["Produit des Fréquences"] = produit_freqs
            res_row["Somme des Fréquences"] = somme_freqs
            
            resultats.append(res_row)

        except Exception as e:
            st.sidebar.error(f"Erreur d'analyse sur l'onglet {feuille} : {e}")

    if resultats:
        df_res = pd.DataFrame(resultats)
        df_valid = df_res.dropna(subset=["Mean SUMOFDEF (1 year)"])
        
        corr_row = {
            "Système": "Coef. de corrélation DEF vs..",
            "Mean SUMOFDEF (1 year)": "" 
        }
        
        colonnes_a_correler = list(FREQS_CIBLES.keys()) + ["Produit des Fréquences", "Somme des Fréquences"]
        
        if len(df_valid) >= 2:
            for col in colonnes_a_correler:
                corr = df_valid["Mean SUMOFDEF (1 year)"].corr(df_valid[col])
                if pd.notna(corr):
                    corr_row[col] = f"{corr * 100:.2f}%"
                else:
                    corr_row[col] = "N/A"
        else:
            for col in colonnes_a_correler:
                corr_row[col] = "N/A"

        df_display = df_res.copy()
        
        df_display["Produit des Fréquences"] = df_display["Produit des Fréquences"].apply(lambda x: f"{x:.2E}")
        for col in FREQS_CIBLES.keys():
            df_display[col] = df_display[col].apply(lambda x: f"{x:.5f}")
        df_display["Somme des Fréquences"] = df_display["Somme des Fréquences"].apply(lambda x: f"{x:.5f}")

        df_display = pd.concat([df_display, pd.DataFrame([corr_row])], ignore_index=True)

        st.subheader("📊 Tableau de Synthèse et Corrélations (Recherche sur Plages Flexibles)")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        csv = df_display.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Télécharger le tableau en CSV",
            data=csv,
            file_name='resultats_fft_amplitudes.csv',
            mime='text/csv',
        )
else:
    st.info("Veuillez importer un fichier Excel depuis la barre latérale pour lancer l'analyse.")
