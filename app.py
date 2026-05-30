import streamlit as st
import pandas as pd
import numpy as np

from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import BytesIO

# --------------------------------------------------
# CONFIGURATION & STYLE
# --------------------------------------------------
st.set_page_config(
    page_title="Analyse FFT Commandes Chirurgicales",
    layout="wide"
)

st.title("Analyse FFT Motoréducteur — Ajustement Synchrone au Millième")

# --------------------------------------------------
# FONCTION DE CALCUL CINÉMATIQUE AVEC MICRO-AJUSTEMENT
# --------------------------------------------------
def calculer_frequences_theoriques(vitesse_moteur_rpm, micro_ajustement_hz):
    """
    Calcule toutes les fréquences de la chaîne cinématique à partir de la vitesse moteur
    et applique un micro-ajustement chirurgical au millième en Hz sur la fréquence moteur.
    Configuration : Moteur -> Réducteur 1:246 -> Poulie 15d -> Courroie 126d -> Poulie 50d
    """
    # 1. Fréquence de rotation du moteur (Hz) + Correction ultra-fine
    f_moteur = (vitesse_moteur_rpm / 60.0) + micro_ajustement_hz
    
    # Vitesse RPM réelle corrigée
    vitesse_moteur_corrigee_rpm = f_moteur * 60.0
    
    # 2. Rotation en sortie de réducteur (Arbre poulie 15 dents)
    f_poulie_15 = f_moteur / 246.0
    
    # 3. Fréquence d'engrènement des poulies (choc des dents)
    f_engrenement = f_poulie_15 * 15.0
    
    # 4. Fréquence de défilement de la courroie (126 dents)
    f_courroie = f_engrenement / 126.0
    
    # 5. Vitesse de rotation de la grande poulie finale (50 dents)
    f_poulie_50 = f_poulie_15 * (15.0 / 50.0)
    vitesse_sortie_rpm = f_poulie_50 * 60.0
    
    return {
        "Rotation Moteur": f_moteur,
        "Rotation Poulie 15d": f_poulie_15,
        "Engrènement (15d/50d)": f_engrenement,
        "Défilement Courroie": f_courroie,
        "Rotation Sortie (50d)": f_poulie_50
    }, vitesse_sortie_rpm, vitesse_moteur_corrigee_rpm

# --------------------------------------------------
# FONCTIONS FFT 
# --------------------------------------------------
def calcul_fft(df):
    t = df["ms"].values / 1000.0
    x = df["V"].values.astype(float)

    x = x - np.mean(x)

    fenetre = np.hanning(len(x))
    x_fenetre = x * fenetre

    dt = np.mean(np.diff(t))
    N = len(x)

    fft = np.abs(rfft(x_fenetre)) * (2.0 / np.sum(fenetre))
    freq = rfftfreq(N, d=dt)

    return freq, fft

def amplitude_bande_max(freq, amp, cible, tolerance=0.1):
    fmin = cible - tolerance
    fmax = cible + tolerance
    mask = (freq >= fmin) & (freq <= fmax)
    
    if np.any(mask):
        return float(np.max(amp[mask]))
    else:
        idx = np.argmin(np.abs(freq - cible))
        return float(amp[idx])

def energie_totale(amp):
    return float(np.sum(amp**2))

def energie_bande(freq, amp, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    return float(np.sum(amp[mask]**2))

def entropie_spectrale(amp):
    p = amp**2
    if np.sum(p) == 0:
        return 0.0
    p = p / np.sum(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def calcul_indicateurs(freq, amp, cible_freq):
    A_cible = amplitude_bande_max(freq, amp, cible_freq, tolerance=0.1)
    Etotal = energie_totale(amp)
    H = entropie_spectrale(amp)
    E05 = energie_bande(freq, amp, 0, 5)
    E1020 = energie_bande(freq, amp, 10, 20)

    IDM3 = 0.0
    if Etotal > 0:
        IDM3 = (A_cible**2 / Etotal) * H

    if IDM3 < 0.5:
        statut = "🟢 Bon"
    elif IDM3 < 1.5:
        statut = "🟡 À surveiller"
    else:
        statut = "🔴 Alarme"

    return {
        "Amp Cible (
