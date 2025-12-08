import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from google.cloud import storage


# -------------------------------------------------------------------
# Helper: Download a .mat and return the 5-D EEG array
# -------------------------------------------------------------------
def _load_eeg_mat(blob) -> np.ndarray:
    """Download and parse a single EEG subject .mat file."""
    bytes_data = blob.download_as_bytes()
    mat = sio.loadmat(BytesIO(bytes_data))

    if "data" not in mat:
        raise KeyError(f"'data' variable not found in {blob.name}")

    data = mat["data"]  # shape should be (8, 710, 2, 10, 12)
    expected = (8, 710, 2, 10, 12)

    if data.shape != expected:
        raise ValueError(f"Unexpected EEG shape {data.shape}, expected {expected}")

    return data.astype(np.float32)


# -------------------------------------------------------------------
# Helper: Convert 5D EEG array → clean epoch-level rows
# -------------------------------------------------------------------
def _explode_subject(subject_id: str, data: np.ndarray) -> pd.DataFrame:
    """Convert a subject's 5D EEG array into 240 tidy rows."""
    rows = []
    electrode_map = {0: "wet", 1: "dry"}

    # Loop dimensions: electrode × block × target
    for e_idx in range(2):
        for b_idx in range(10):
            for t_idx in range(12):
                epoch_signal = data[:, :, e_idx, b_idx, t_idx]  # (8, 710)

                rows.append(
                    {
                        "subject": subject_id,
                        "electrode": electrode_map[e_idx],
                        "block": b_idx + 1,
                        "target": t_idx + 1,
                        "signal": epoch_signal,
                    }
                )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# PUBLIC FUNCTION (exactly like get_emg_data)
# -------------------------------------------------------------------
def get_eeg_data():
    """
    Load ALL EEG subjects from GCS and return a tidy DataFrame.

    Columns:
        subject (str)
        electrode ('wet'/'dry')
        block (1–10)
        target (1–12)
        signal: np.ndarray of shape (8, 710)
    """

    bucket_name = "inkling-ssvep-emg"
    prefix = "Wearable SSVEP Dataset/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)


    mat_files = []
    for blob in bucket.list_blobs(prefix=prefix):
        name = os.path.basename(blob.name)

        # Only subject files like S001.mat
        if not name.endswith(".mat"):
            continue
        if not re.match(r"S\d{3}\.mat$", name):
            continue

        mat_files.append(blob)

    if not mat_files:
        raise RuntimeError("No subject .mat files found in bucket.")

    print(f"Found {len(mat_files)} subjects.")

    # Build full dataset
    all_subject_rows = []

    for blob in mat_files:
        subject_id = os.path.splitext(os.path.basename(blob.name))[0]
        print(f"Loading {subject_id}...")

        # Load and explode into rows
        data = _load_eeg_mat(blob)
        df_subj = _explode_subject(subject_id, data)

        all_subject_rows.append(df_subj)

    full_df = pd.concat(all_subject_rows, ignore_index=True)

    return full_df
