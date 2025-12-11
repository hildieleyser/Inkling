# zaki_EMG_packages/preprocess.py

from typing import Sequence
import numpy as np

# Channels selected from ranking step (0–15 indices)
BEST_CHANNELS = (3, 10, 12, 1)

# Fixed time length used in training (from X_train shape)
INPUT_LENGTH = 9940  # update if your training length changes


def fix_length(signal: np.ndarray, target_length: int = INPUT_LENGTH) -> np.ndarray:
    """
    Ensure the EMG has exactly `target_length` time samples.

    Input:
        signal: (C, N_time)
    Output:
        (C, target_length)
    """
    channels, length = signal.shape

    if length == target_length:
        return signal

    if length > target_length:
        # crop
        return signal[:, :target_length]

    # pad with zeros at the end
    pad_width = target_length - length
    pad = np.zeros((channels, pad_width), dtype=signal.dtype)
    return np.concatenate([signal, pad], axis=1)


def select_channels(signal: np.ndarray, channels: Sequence[int] = BEST_CHANNELS) -> np.ndarray:
    """
    Pick only the desired channels.

    Input:
        signal: (16, N_time)
    Output:
        (len(channels), N_time)
    """
    return signal[list(channels), :]


def prepare_for_model(
    signal_16ch: np.ndarray,
    scaler: object,
    channels: Sequence[int] = BEST_CHANNELS,
    target_length: int = INPUT_LENGTH,
) -> np.ndarray:
    """
    Preprocess EMG from the original 16-channel recording:

    - select best 4 channels
    - fix length
    - flatten + scale with StandardScaler
    - reshape to (1, time, channels) for Conv1D

    Input:
        signal_16ch: (16, N_time)
    Output:
        x: (1, time, 4)
    """
    # 1) (16, N) -> (4, N)
    sig_4 = select_channels(signal_16ch, channels=channels)

    # 2) (4, N) -> (4, target_length)
    sig_4 = fix_length(sig_4, target_length=target_length)

    # 3) (4, T) -> (T, 4)
    sig_4_T = sig_4.T

    # 4) Flatten for scaler: (1, T*4)
    flat = sig_4_T.reshape(1, -1)

    # 5) Scale
    flat_scaled = scaler.transform(flat)

    # 6) Back to (1, T, 4)
    time_steps = target_length
    x = flat_scaled.reshape(1, time_steps, len(channels)).astype("float32")
    return x


def prepare_live_4ch(
    signal_4ch: np.ndarray,
    scaler: object,
    target_length: int = INPUT_LENGTH,
) -> np.ndarray:
    """
    Preprocess live EMG from 4 electrodes.

    Input:
        signal_4ch: (4, N_time) raw EMG
    Output:
        x: (1, time, 4) ready for model
    """
    if signal_4ch.ndim != 2:
        raise ValueError(f"Expected 2D array (4, N_time), got {signal_4ch.shape}")
    if signal_4ch.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got {signal_4ch.shape[0]}")

    # (4, N) -> (4, target_length)
    sig = fix_length(signal_4ch, target_length=target_length)

    # (4, T) -> (T, 4)
    sig_T = sig.T

    # flatten for scaler: (1, T*4)
    flat = sig_T.reshape(1, -1)

    # scale
    flat_scaled = scaler.transform(flat)

    # back to (1, T, 4)
    x = flat_scaled.reshape(1, target_length, 4).astype("float32")
    return x
