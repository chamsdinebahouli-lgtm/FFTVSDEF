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

def extract_amp(freq, amp, target_hz, tol_hz=0.05):
    """Extrait l'amplitude maximale dans une bande de tolérance autour de la cible."""
    fmin, fmax = target_hz - tol_hz, target_hz + tol_hz
    mask = (freq >= fmin) & (freq <= fmax)
    if np.any(mask):
        return float(np.max(amp[mask]))
    # Si rien n'est trouvé dans la bande, on prend le point le plus proche
    idx = np.argmin(np.abs(freq - target_hz))
    return float(amp[idx])

# --------------------------------------------------
# INTERFACE BARRE LATÉRALE (SIDEBAR)
# --------------------------------------------------
st.sidebar.header("🛠️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Importer le fichier Excel (.xlsx)", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Tolérance de recherche")
tol_recherche = st.sidebar.number_input("Bande de tolérance (Hz) :", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

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
    
    # Parsing des notes terrain (Défaut Réel)
    notes = {line.split("=")[0].strip(): float(line.split("=")[1].strip()) for line in notes_text.splitlines() if "=" in line}

    # Fréquences cibles définies dans votre tableau
    FREQS_CIBLES = {
        "Amplitude à 0,0167 Hz": 0.0167,
        "Amplitude à 3,68 Hz (V)": 3.68,
        "Amplitude à 12,33 Hz": 12.33,
        "Amplitude à 13,67 Hz": 13.67
    }

    # 1. Extraction des données par onglet
    for feuille in xls.sheet_names:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=feuille)
            if not {"ms", "V"}.issubset(df.columns): 
                continue

            freq, amp = calcul_fft(df)

            # Création de la ligne de résultat
            res_row = {
                "Système": feuille,
                "Mean SUMOFDEF (1 year)": notes.get(feuille, np.nan)
            }

            # Extraction dynamique pour chaque fréquence cible
            somme_freqs = 0.0
            produit_freqs = 1.0
            
            for col_name, f_cible in FREQS_CIBLES.items():
                amp_val = extract_amp(freq, amp, f_cible, tol_hz=tol_recherche)
                res_row[col_name] = amp_val
                somme_freqs += amp_val
                produit_freqs *= amp_val

            res_row["Produit des Fréquences"] = produit_freqs
            res_row["Somme des Fréquences"] = somme_freqs
            
            resultats.append(res_row)

        except Exception as e:
            st.sidebar.error(f"Erreur d'analyse sur l'onglet {feuille} : {e}")

    # 2. Construction du DataFrame final et calcul des corrélations
    if resultats:
        df_res = pd.DataFrame(resultats)
        
        # Isoler les lignes avec un 'Mean SUMOFDEF' valide pour calculer la corrélation
        df_valid = df_res.dropna(subset=["Mean SUMOFDEF (1 year)"])
        
        # Préparation de la ligne de corrélation
        corr_row = {
            "Système": "Coef. de corrélation DEF vs..",
            "Mean SUMOFDEF (1 year)": "" # Laissé vide pour cette ligne
        }
        
        colonnes_a_correler = list(FREQS_CIBLES.keys()) + ["Produit des Fréquences", "Somme des Fréquences"]
        
        if len(df_valid) >= 2:
            for col in colonnes_a_correler:
                # Calcul de la corrélation de Pearson
                corr = df_valid["Mean SUMOFDEF (1 year)"].corr(df_valid[col])
                if pd.notna(corr):
                    corr_row[col] = f"{corr * 100:.2f}%"
                else:
                    corr_row[col] = "N/A"
        else:
            for col in colonnes_a_correler:
                corr_row[col] = "N/A"

        # Formater les colonnes numériques avant d'ajouter la ligne de corrélation (pour la beauté de l'affichage)
        df_display = df_res.copy()
        
        # Arrondir et formater en notation scientifique pour le produit
        df_display["Produit des Fréquences"] = df_display["Produit des Fréquences"].apply(lambda x: f"{x:.2E}")
        for col in FREQS_CIBLES.keys():
            df_display[col] = df_display[col].apply(lambda x: f"{x:.5f}")
        df_display["Somme des Fréquences"] = df_display["Somme des Fréquences"].apply(lambda x: f"{x:.5f}")

        # Ajouter la ligne de corrélation à la fin
        df_display = pd.concat([df_display, pd.DataFrame([corr_row])], ignore_index=True)

        # 3. Affichage
        st.subheader("📊 Tableau de Synthèse et Corrélations")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Option pour télécharger les données en CSV
        csv = df_display.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Télécharger le tableau en CSV",
            data=csv,
            file_name='resultats_fft_amplitudes.csv',
            mime='text/csv',
        )
else:
    st.info("Veuillez importer un fichier Excel depuis la barre latérale pour lancer l'analyse.")
