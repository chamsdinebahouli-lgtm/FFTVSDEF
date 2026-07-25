import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


def calculer_fft_signal(df, col_temps="ms", col_signal="V", dt_force=None):
    """Calcule la FFT d'un signal en corrigeant automatiquement les réinitialisations

    de l'axe temporel (ex: horloge/colonne 'ms' qui boucle).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données du signal.
    col_temps : str
        Nom de la colonne du temps (ex: "ms").
    col_signal : str
        Nom de la colonne du signal/tension (ex: "V").
    dt_force : float, optional
        Pas de temps en secondes si connu à l'avance (ex: 0.020 pour 20 ms).
        Si None, il est calculé automatiquement sur la médiane des deltas positifs.

    Returns:
    --------
    freq : numpy.ndarray
        Vecteur des fréquences en Hz.
    fft_amp : numpy.ndarray
        Amplitudes FFT associées (en V).
    dt : float
        Pas d'échantillonnage réel retenu (en secondes).
    """
    # 1. Extraction et nettoyage des données du signal
    x = df[col_signal].values.astype(float)
    N = len(x)

    if N == 0:
        raise ValueError("Le tableau de données est vide.")

    # 2. Détermination du pas d'échantillonnage dt (en secondes)
    if dt_force is not None:
        dt = float(dt_force)
    else:
        # Calcul des deltas entre lignes consécutives
        diffs = np.diff(df[col_temps].values.astype(float))

        # Filtrage : on ne garde que les deltas strictement positifs
        # pour éliminer les sauts arrières lors des réinitialisations de l'horloge
        diffs_positives = diffs[diffs > 0]

        if len(diffs_positives) == 0:
            raise ValueError(
                "Impossible de déterminer le pas de temps depuis la colonne temporel."
            )

        dt_ms = np.median(diffs_positives)  # Pas de temps médian (ex: 20 ms)
        dt = dt_ms / 1000.0  # Conversion ms -> secondes

    # 3. Traitement du signal : suppression de la composante continue (offset DC)
    x_centered = x - np.mean(x)

    # 4. Fenêtrage Hanning d'apodisation pour éviter le fuite spectrale
    fenetre = np.hanning(N)
    x_fen = x_centered * fenetre

    # 5. Calcul de la FFT réelle
    # Correction de l'amplitude tenant compte de la puissance de la fenêtre
    somme_fenetre = np.sum(fenetre)
    fft_amp = np.abs(rfft(x_fen)) * (2.0 / somme_fenetre)

    # 6. Vecteur des fréquences réelles
    freq = rfftfreq(N, d=dt)

    return freq, fft_amp, dt


def obtenir_amplitude_a_frequence(freq, amp, target_hz, tol_pct=0.03):
    """Recherche la valeur d'amplitude maximale d'un pic autour d'une fréquence cible.

    Parameters:
    -----------
    freq : numpy.ndarray
        Vecteur des fréquences en Hz.
    amp : numpy.ndarray
        Vecteur des amplitudes.
    target_hz : float
        Fréquence cible recherchée (ex: 0.016759 Hz).
    tol_pct : float
        Plage de tolérance relative (ex: 0.03 = ±3%).

    Returns:
    --------
    dict: Dictionnaire contenant la fréquence cible, la fréquence réelle la plus proche
          et l'amplitude mesurée.
    """
    tol_hz = target_hz * tol_pct
    fmin, fmax = max(0.0, target_hz - tol_hz), target_hz + tol_hz

    mask = (freq >= fmin) & (freq <= fmax)

    if np.any(mask):
        sub_indices = np.where(mask)[0]
        max_sub_idx = sub_indices[np.argmax(amp[mask])]
        f_trouvee = freq[max_sub_idx]
        amp_trouvee = amp[max_sub_idx]
    else:
        idx = np.argmin(np.abs(freq - target_hz))
        f_trouvee = freq[idx]
        amp_trouvee = amp[idx]

    return {
        "target_hz": target_hz,
        "freq_reelle_hz": f_trouvee,
        "amplitude_V": amp_trouvee,
    }


# ==============================================================================
# EXECUTION DU SCRIPT SUR LE FICHIER EXCEL
# ==============================================================================
if __name__ == "__main__":
    nom_fichier = "FFT_Décalage plaque_nettoyé.xlsx"

    # Liste des fréquences cibles à rechercher dans chaque onglet
    freqs_cibles = {
        "Amplitude 0,016759 Hz": 0.016759,
        "Amplitude 3,68 Hz": 3.68,
        "Amplitude 12,33 Hz": 12.33,
        "Amplitude 13,67 Hz": 13.67,
    }

    try:
        xls = pd.ExcelFile(nom_fichier)
        resultats = []

        for nom_onglet in xls.sheet_names:
            df = pd.read_excel(nom_fichier, sheet_name=nom_onglet)

            # Nettoyage des espaces éventuels dans le nom des colonnes
            df.columns = [str(c).strip() for c in df.columns]

            if len(df.columns) < 2:
                continue

            col_temps = df.columns[0]  # Première colonne (ex: 'ms')
            col_signal = df.columns[1]  # Deuxième colonne (ex: 'V')

            # Calcul de la FFT avec correction automatique du temps
            freq, amp, dt = calculer_fft_signal(
                df, col_temps=col_temps, col_signal=col_signal
            )

            ligne_res = {
                "Système / Onglet": nom_onglet,
                "Nb Points": len(df),
                "dt (s)": dt,
                "Durée Totale (s)": round(len(df) * dt, 2),
            }

            # Extrait l'amplitude pour chaque fréquence souhaitée
            for label, f_cible in freqs_cibles.items():
                info = obtenir_amplitude_a_frequence(freq, amp, f_cible)
                ligne_res[label] = round(info["amplitude_V"], 6)

            resultats.append(ligne_res)

        # Affichage synthétique des résultats sous forme de DataFrame
        df_resultats = pd.DataFrame(resultats)
        print("\n--- RÉSULTATS DE L'ANALYSE FFT CORRIGÉE ---")
        print(df_resultats.to_string(index=False))

        # Optionnel : Exporter le tableau propre vers un fichier Excel
        # df_resultats.to_excel("Resultats_FFT_Corriges.xlsx", index=False)

    except FileNotFoundError:
        print(
            f"Erreur : Le fichier '{nom_fichier}' n'a pas été trouvé dans le dossier."
        )
