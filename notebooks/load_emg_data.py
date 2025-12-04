import os
import json
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import storage

def parse_signal_cell(cell: str | np.ndarray | list | tuple) -> np.ndarray:
    """Parse a corrupted/loosely formatted signal into a (16, N) float32 array."""
    # If already array-like, trust and reshape as needed.
    if isinstance(cell, (np.ndarray, list, tuple)):
        arr = np.asarray(cell, dtype=np.float32)
        if arr.shape == (16, 10000):
            return arr
        if arr.shape == (10000, 16):
            return arr.T
        if arr.ndim == 1 and arr.size == 160000:
            return arr.reshape(16, 10000)
        if arr.ndim == 2 and arr.shape[0] == 16:
            return arr

    raw = str(cell)
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    if not tokens:
        raise ValueError(f"No numeric tokens found in signal cell: {raw[:120]}")

    numbers = np.asarray([float(t) for t in tokens], dtype=np.float32)
    if numbers.size < 16:
        raise ValueError(f"Too few numeric tokens ({numbers.size}); sample: {raw[:120]}")

    usable = (numbers.size // 16) * 16
    numbers = numbers[:usable]
    channel_len = usable // 16
    if channel_len == 0:
        raise ValueError(f"Not enough data to form 16 channels; tokens={numbers.size}")

    arr = numbers.reshape(16, channel_len)
    return arr


def fix_length(arr: np.ndarray, target_len: int = 10000) -> np.ndarray:
    """Trim or pad signals to exactly (16, target_len)."""
    if arr.ndim != 2 or arr.shape[0] != 16:
        raise ValueError(f"Signal must have shape (16, N); got {arr.shape}")

    current_len = arr.shape[1]
    if current_len == target_len:
        return arr
    if current_len < target_len:
        pad_width = target_len - current_len
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
    return arr[:, :target_len]


def get_emg_data():

    bucket_name = "inkling-ssvep-emg"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_path = "EMG-nature/Clean_df/emg_trial_level_df.csv"

    print(f"Accessing blob: {blob_path}")
    blob = bucket.blob(blob_path)

    if not blob.exists(client=client):
        raise FileNotFoundError(f"Blob {blob_path} not found in bucket {bucket_name}")

    data_bytes = blob.download_as_bytes()
    print(f"Downloaded {len(data_bytes)} bytes from GCS")

    emg_df = pd.read_csv(BytesIO(data_bytes))
    print(f"Loaded DataFrame with shape: {emg_df.shape}")

    emg_df["signal"] = emg_df["signal"].apply(parse_signal_cell)
    emg_df["signal_fixed"] = emg_df["signal"].apply(fix_length)
    emg_df = emg_df.drop(columns = "Unnamed: 0")
    return emg_df

