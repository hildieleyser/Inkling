"""
EEG preprocessing utilities.

Applies:
- band-pass filtering (8–90 Hz)
- notch filtering (50, 100 Hz)
- baseline correction
- trimming to stable SSVEP window
- per-channel z-score normalization

All training, evaluation, and inference MUST call these functions.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple

# Dataset-defined EEG sampling rate
FS = 250


# ---------------------------------------------------------------------
# Filtering utilities
# ---------------------------------------------------------------------
def bandpass(epoch: np.ndarray, low: float = 8, high: float = 90,
             fs: int = FS, order: int = 4) -> np.ndarray:
    """
    Band-pass filter.
    epoch: (n_channels, n_samples)
    """
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, epoch, axis=-1)


def notch(epoch: np.ndarray, freq: float = 50,
          fs: int = FS, q: int = 30) -> np.ndarray:
    """
    Notch filter (e.g. line noise).
    """
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q=q)
    return filtfilt(b, a, epoch, axis=-1)


# ---------------------------------------------------------------------
# Epoch-level preprocessing (CANONICAL)
# ---------------------------------------------------------------------
def preprocess_epoch(epoch_raw: np.ndarray) -> np.ndarray:
    """
    Full preprocessing for a single EEG trial.

    Input:
        epoch_raw: np.ndarray, shape (8, 710)

    Output:
        epoch_clean: np.ndarray, shape (8, 500), dtype float32
    """
    # Ensure numerical stability for filters
    epoch = epoch_raw.astype(np.float64)

    # 1) Band-pass (SSVEP fundamentals + harmonics)
    epoch = bandpass(epoch, 8, 90, FS)

    # 2) Notch powerline noise
    epoch = notch(epoch, 50, FS)
    epoch = notch(epoch, 100, FS)

    # 3) Baseline correction (pre-stimulus: first 0.5 s)
    baseline = epoch[:, :125].mean(axis=-1, keepdims=True)
    epoch = epoch - baseline

    # 4) Trim to stable SSVEP response window
    # 160:660 → ~0.64s to 2.64s (2.0 s total)
    epoch = epoch[:, 160:660]

    # 5) Per-channel z-score normalization
    mean = epoch.mean(axis=-1, keepdims=True)
    std = epoch.std(axis=-1, keepdims=True) + 1e-8
    epoch = (epoch - mean) / std

    return epoch.astype(np.float32)


# ---------------------------------------------------------------------
# Batch preprocessing (numpy-friendly for Vertex AI)
# ---------------------------------------------------------------------
def preprocess_eeg_dataset(X: np.ndarray) -> np.ndarray:
    """
    Preprocess a batch of EEG epochs.

    Input:
        X: np.ndarray, shape (N, 8, 710)

    Output:
        X_clean: np.ndarray, shape (N, 8, 500)
    """
    return np.stack([preprocess_epoch(epoch) for epoch in X])


# ---------------------------------------------------------------------
# DataFrame adapter (for notebook / analysis workflows)
# ---------------------------------------------------------------------
def preprocess_eeg_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to each row of a DataFrame.

    Required columns:
        df["signal"] : np.ndarray, shape (8, 710)

    Adds:
        df["signal_clean"] : np.ndarray, shape (8, 500)
    """
    if "signal" not in df.columns:
        raise ValueError("EEG DataFrame must contain a 'signal' column.")

    cleaned = []
    for idx, raw in enumerate(df["signal"]):
        try:
            cleaned.append(preprocess_epoch(raw))
        except Exception as exc:
            raise RuntimeError(f"Preprocessing failed at row {idx}: {exc}")

    out = df.copy()
    out["signal_clean"] = cleaned
    return out
