"""
EEG preprocessing module
Applies band-pass, notch filters, baseline correction, trimming, and z-scoring
to each trial in the EEG dataframe.

Public function:
    preprocess_eeg_df(df)  -> returns dataframe with new column "signal_clean"
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd

# Dataset-defined EEG sampling rate
FS = 250  


# -------------------------------------------------------------------
# Filtering utilities
# -------------------------------------------------------------------
def bandpass(epoch, low=8, high=90, fs=FS, order=4):
    """
    Apply 8–90 Hz bandpass filter.
    epoch: (n_channels, n_samples)
    """
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, epoch, axis=-1)


def notch(epoch, freq=50, fs=FS, q=30):
    """
    Apply notch filter to remove powerline noise at `freq` Hz.
    """
    nyq = fs / 2
    w0 = freq / nyq
    b, a = iirnotch(w0, Q=q)
    return filtfilt(b, a, epoch, axis=-1)


# -------------------------------------------------------------------
# Full single-epoch preprocessing pipeline
# -------------------------------------------------------------------
def preprocess_epoch(epoch_raw):
    """
    Full preprocessing for a single trial.
    
    Input:
        epoch_raw : np.ndarray of shape (8, 710)

    Output:
        epoch_clean : np.ndarray of shape (8, 500)
    """
    # Ensure float64 precision for filters
    epoch = epoch_raw.astype(np.float64)

    # 1) 8–90 Hz bandpass (keeps SSVEP harmonics, removes drift/noise)
    epoch = bandpass(epoch, low=8, high=90, fs=FS, order=4)

    # 2) 50 Hz + 100 Hz notch (mains noise + harmonic)
    epoch = notch(epoch, freq=50, fs=FS, q=30)
    epoch = notch(epoch, freq=100, fs=FS, q=30)

    # 3) Baseline correction (first 125 samples = -0.5 to 0s)
    baseline = epoch[:, :125].mean(axis=-1, keepdims=True)
    epoch = epoch - baseline

    # 4) Trim to the SSVEP response window (160:660 → 2 seconds)
    epoch = epoch[:, 160:660]  # shape (8, 500)

    # 5) Per-epoch & per-channel z-score normalisation
    mean = epoch.mean(axis=-1, keepdims=True)
    std = epoch.std(axis=-1, keepdims=True) + 1e-8
    epoch = (epoch - mean) / std

    return epoch.astype(np.float32)


# -------------------------------------------------------------------
# Public API matching EMG style
# -------------------------------------------------------------------
def preprocess_eeg_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to every row in the EEG DataFrame.

    Required columns:
        df["signal"] = np.ndarray of shape (8, 710)

    Adds:
        df["signal_clean"] = np.ndarray of shape (8, 500)
    """
    if "signal" not in df.columns:
        raise ValueError("EEG DataFrame must contain a 'signal' column.")

    cleaned = []
    for idx, raw in enumerate(df["signal"]):
        try:
            cleaned.append(preprocess_epoch(raw))
        except Exception as exc:
            raise RuntimeError(f"Error preprocessing row {idx}: {exc}")

    df = df.copy()
    df["signal_clean"] = cleaned
    return df
