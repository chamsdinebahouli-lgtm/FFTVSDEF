import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hann

# Configuration de la page
st.set_page_config(
    page_title="Analyse Électrique DC & Vibratoire",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Analyse Électrique DC & Diagnostic Vibratoire")
st.markdown("Application de traitement des signaux moteurs, analyse spectrale et calcul de la somme des amplitudes.")

# ==========================================
# 1. BARRE LATÉRALE - PARAMÈTRES
# ==========================================
st.sidebar.header("1. Paramètres d'analyse")
methode_fft = st.sidebar.selectbox("Méthode FFT", ["Nouveau Mode (Hanning + Pic)", "Ancien Mode (Brut)"])
unite_temps = st.sidebar.selectbox("Unité de temps de la 1ère colonne", ["Secondes (s)", "Millisecondes (ms)"])
normalisation_dc = st.sidebar.checkbox("Normalisation par rapport au DC (% DC)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("2. Détection Automatique de Pics")
activer_pics_auto = st.sidebar.checkbox("Activer la détection automatique", value=True)
seuil_pic = st.sidebar.slider("Seuil minimal du pic (% du max)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)

# ==========================================
# 2. IMPORTATION DU FICHIER EXCEL
# ==========================================
st.markdown("### 📂 Importation des Données")
fichier_upload = st.file_uploader("Téléchargez votre fichier Excel multi-onglets (.xlsx)", type=["xlsx", "xls"])

if fichier_upload is not None:
    try:
        # Lecture de tous les onglets du fichier Excel
        excel_file = pd.ExcelFile(fichier_upload)
        sheet_names = excel_file.sheet_names
        
        st.success(f"Fichier chargé avec succès ! {len(sheet_names)} onglet(s) détecté(s).")
        
        # Sélection de l'onglet à analyser
        onglet_selectionne = st.selectbox("Sélectionner la machine / l'onglet à analyser :", sheet_names)
        
        # Chargement des données de l'onglet
        df_machine = pd.read_excel(excel_file, sheet_name=onglet_selectionne)
        
        if df_machine.shape[1] < 2:
            st.error("L'onglet sélectionné doit contenir au moins 2 colonnes (Temps et Signal).")
        else:
            # Extraction du temps et du signal
            t = df_machine.iloc[:, 0].values
            signal = df_machine.iloc[:, 1].values
            
            # Conversion du temps si nécessaire
            if unite_temps == "Millisecondes (ms)":
                t = t / 1000.0  # Conversion en secondes
                
            dt = np.mean(np.diff(t))
            fs = 1.0 / dt if dt > 0 else 1.0
            
            # --- CALCUL DE LA COMPOSANTE CONTINUE (DC) ---
            dc_offset = np.mean(signal)
            
            # --- CALCUL DE LA FFT ---
            n = len(signal)
            signal_AC = signal - dc_offset # Retrait du DC pour le spectre alternatif
            
            if "Hanning" in methode_fft:
                fenetre = hann(n)
                signal_AC_fenetre = signal_AC * fenetre
                fft_vals = np.fft.rfft(signal_AC_fenetre)
                freqs = np.fft.rfftfreq(n, d=dt)
                # Correction d'amplitude liée à la fenêtre de Hanning
                amplitudes = (2.0 / np.sum(fenetre)) * np.abs(fft_vals)
            else:
                fft_vals = np.fft.rfft(signal_AC)
                freqs = np.fft.rfftfreq(n, d=dt)
                amplitudes = (2.0 / n) * np.abs(fft_vals)
                
            # Normalisation en % DC si demandé
            if normalisation_dc and dc_offset != 0:
                amplitudes_aff = (amplitudes / abs(dc_offset)) * 100.0
                unite_amp = "% DC"
            else:
                amplitudes_aff = amplitudes
                unite_amp = "Amplitude"

            # ==========================================
            # 3. DÉTECTION DES PICS & COURROIE (0.06 Hz)
            # ==========================================
            tableau_pics_data = []
            
            if activer_pics_auto:
                # Recherche automatique des pics basée sur le seuil
                seuil_absolu = (seuil_pic / 100.0) * np.max(amplitudes_aff) if len(amplitudes_aff) > 0 else 0
                peaks, _ = find_peaks(amplitudes_aff, height=seuil_absolu, distance=5)
                
                for p in peaks:
                    f_val = freqs[p]
                    amp_val = amplitudes_aff[p]
                    
                    # Nommage intelligent si on détecte une basse fréquence (ex: Courroie à ~0.06 Hz)
                    if 0.04 <= f_val <= 0.08:
                        nom_pic = f"Pic auto : {f_val:.2f} Hz (Courroie d'entraînement)"
                    else:
                        nom_pic = f"Pic auto : {f_val:.2f} Hz"
                        
                    tableau_pics_data.append({
                        "Nom du Pic": nom_pic,
                        "Fréquence (Hz)": round(f_val, 3),
                        f"Amplitude ({unite_amp})": round(amp_val, 3),
                        "Inclure dans la Somme": True # Coché par défaut, inclut la courroie
                    })
            
            # Transformation en DataFrame interactif
            df_pics_detectes = pd.DataFrame(tableau_pics_data)
            
            st.markdown("### 📊 Résultats et Analyse Spectrale")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Offset DC (Moyenne)", f"{dc_offset:.3f}")
            with col2:
                # Calcul de la Somme des Amplitudes en tenant compte de la sélection de l'utilisateur
                if not df_pics_detectes.empty and "Inclure dans la Somme" in df_pics_detectes.columns:
                    pics_retenus = df_pics_detectes[df_pics_detectes["Inclure dans la Somme"] == True]
                    somme_amplitudes = pics_retenus[f"Amplitude ({unite_amp})"].sum()
                else:
                    somme_amplitudes = 0.0
                    
                st.metric("Somme des Amplitudes Sélectionnées", f"{somme_amplitudes:.3f} {unite_amp}")

            # ==========================================
            # 4. TABLEAU INTERACTIF DES PICS (Éditable)
            # ==========================================
            st.markdown("#### Gestion des pics détectés (Modifiez l'inclusion si besoin)")
            if not df_pics_detectes.empty:
                df_editable = st.data_editor(df_pics_detectes, use_container_width=True, hide_index=True)
                
                # Recalcul dynamique après modification du tableau par l'utilisateur
                pics_finaux = df_editable[df_editable["Inclure dans la Somme"] == True]
                somme_finale = pics_finaux[f"Amplitude ({unite_amp})"].sum()
                st.info(f"💡 Somme actualisée des amplitudes incluses : **{somme_finale:.3f} {unite_amp}** (Inclut la courroie si cochée)")
            else:
                st.warning("Aucun pic détecté avec les seuils actuels. Baissez le seuil dans la barre latérale.")

            # ==========================================
            # 5. VISUALISATION GRAPHIQUE (FFT)
            # ==========================================
            st.markdown("### 📈 Spectre de Fréquence")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freqs, amplitudes_aff, color='b', lw=1.2, label='Spectre')
            
            if not df_pics_detectes.empty:
                ax.scatter(df_editable["Fréquence (Hz)"], df_editable[f"Amplitude ({unite_amp})"], color='r', zorder=5, label='Pics détectés')
                
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_ylabel(f"Amplitude ({unite_amp})")
            ax.set_title(f"Spectre FFT - {onglet_selectionne}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f-f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez importer un fichier Excel pour démarrer l'analyse.")
