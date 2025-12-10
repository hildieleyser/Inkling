# api_eeg/fast.py

from pathlib import Path
import io
import os
import sys
import importlib
import threading
import time
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from scipy.io import loadmat

# ------------------------------------------------------------
# Locate and import your ssvep package (EEGNet + preprocess)
# ------------------------------------------------------------

HERE = Path(__file__).resolve().parent          # .../inkling/api_eeg
PROJECT_ROOT = HERE.parent                      # .../inkling

# OpenBCI recordings folder:
# e.g. /Users/<user>/Documents/OpenBCI_GUI/Recordings
RECORDINGS_ROOT = Path.home() / "Documents" / "OpenBCI_GUI" / "Recordings"

EEG_CSV_GLOB = "BrainFlow-RAW_*.csv"  # matches OpenBCI BrainFlow raw exports

def _find_ssvep_package(start: Path) -> Tuple[Path, Path]:
    for root, dirs, files in os.walk(start):
        if "ssvep" in dirs:
            pkg_dir = Path(root) / "ssvep"
            if (pkg_dir / "__init__.py").exists():
                return pkg_dir, Path(root)
    raise RuntimeError("Could not find 'ssvep' package under project root.")


def _import_ssvep():
    pkg_dir, parent = _find_ssvep_package(PROJECT_ROOT)

    parent_str = str(parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

    try:
        ssvep = importlib.import_module("ssvep")
        data_mod = importlib.import_module("ssvep.data")
    except ImportError as e:
        raise RuntimeError(f"Found ssvep at {pkg_dir}, but could not import: {e}")

    EEGNet = getattr(ssvep, "EEGNet")
    preprocess_epoch = getattr(data_mod, "preprocess_epoch")
    return EEGNet, preprocess_epoch


EEGNet, preprocess_epoch = _import_ssvep()

# ------------------------------------------------------------
# Config + model load
# ------------------------------------------------------------

MODEL_PATH = HERE / "eegnet_tuned.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training
N_CHANS = 8
N_SAMPLES = 500
N_CLASSES = 12

app = FastAPI(title="EEG SSVEP Inference API")


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = EEGNet(n_chans=N_CHANS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[EEG] Model loaded.")
    return model


model = load_model()

# ------------------------------------------------------------
# .mat → single epoch helper (unchanged)
# ------------------------------------------------------------


def _extract_epoch_from_mat(
    contents: bytes,
    block_idx: int = 0,
    trial_idx: int = 0,
    target_idx: int = 0,
) -> np.ndarray:
    """
    Return a single epoch as (channels, time).
    Tries 'data' first (Wearable SSVEP-style), then falls back to
    scanning arrays for something with 8 channels and >= 660 samples.
    """
    bio = io.BytesIO(contents)
    mat = loadmat(bio)

    # Preferred: mat["data"] with shape like (ch, time, block, trial, target)
    if "data" in mat and isinstance(mat["data"], np.ndarray):
        data = np.array(mat["data"])
        if data.ndim >= 3 and data.shape[0] == N_CHANS:
            if data.ndim >= 5:  # (ch, time, block, trial, target)
                n_blocks, n_trials, n_targets = data.shape[2:5]
                if not (0 <= block_idx < n_blocks):
                    raise ValueError(f"block_idx out of range (0..{n_blocks-1})")
                if not (0 <= trial_idx < n_trials):
                    raise ValueError(f"trial_idx out of range (0..{n_trials-1})")
                if not (0 <= target_idx < n_targets):
                    raise ValueError(f"target_idx out of range (0..{n_targets-1})")
                epoch = data[:, :, block_idx, trial_idx, target_idx]

            elif data.ndim == 4:  # (ch, time, block, epoch)
                n_blocks, n_epochs = data.shape[2:4]
                if not (0 <= block_idx < n_blocks):
                    raise ValueError(f"block_idx out of range (0..{n_blocks-1})")
                if not (0 <= target_idx < n_epochs):
                    raise ValueError(f"target_idx out of range (0..{n_epochs-1})")
                epoch = data[:, :, block_idx, target_idx]

            elif data.ndim == 3:  # (ch, time, epoch)
                n_epochs = data.shape[2]
                if not (0 <= target_idx < n_epochs):
                    raise ValueError(f"target_idx out of range (0..{n_epochs-1})")
                epoch = data[:, :, target_idx]

            else:
                epoch = None

            if epoch is not None:
                epoch = np.squeeze(epoch)
                if epoch.ndim != 2:
                    raise ValueError("Selected epoch is not 2D.")
                if epoch.shape[0] != N_CHANS:
                    epoch = epoch.T
                if epoch.shape[0] != N_CHANS:
                    raise ValueError(f"Epoch does not have {N_CHANS} channels.")
                if epoch.shape[1] < 660:
                    raise ValueError("Epoch has too few samples (need >= 660).")
                return epoch.astype(np.float64)

    # Fallback: search for any 2D array that looks like (8, >=660) or (>=660, 8)
    def _try_epoch(arr: np.ndarray) -> np.ndarray | None:
        a = np.squeeze(np.array(arr))
        while a.ndim > 2:
            a = np.squeeze(a[..., 0])
        if a.ndim != 2:
            return None
        if a.shape[0] == N_CHANS:
            e = a
        elif a.shape[1] == N_CHANS:
            e = a.T
        else:
            return None
        if e.shape[1] < 660:
            return None
        return e.astype(np.float64)

    for key in ["epoch", "EEG", "eeg", "X", "signal"]:
        if key in mat and isinstance(mat[key], np.ndarray):
            candidate = _try_epoch(mat[key])
            if candidate is not None:
                return candidate

    for v in mat.values():
        if isinstance(v, np.ndarray):
            candidate = _try_epoch(v)
            if candidate is not None:
                return candidate

    raise ValueError("No suitable EEG array found in .mat file.")


# ------------------------------------------------------------
# Shared BrainFlow CSV → prediction helper
# ------------------------------------------------------------


def _predict_from_brainflow_df(df: pd.DataFrame, filename: str) -> Dict[str, object]:
    """
    Take a BrainFlow RAW dataframe and run it through
    preprocess_epoch + EEGNet, returning a prediction dict.
    Raises ValueError on problems.
    """
    if df.shape[1] < 2 + N_CHANS:
        raise ValueError(
            f"CSV has only {df.shape[1]} columns; expected at least {2 + N_CHANS}"
        )

    # Typical BrainFlow RAW:
    #   0: sample index / counter
    #   1: board meta / reserved
    #   2..9: EEG channels (8 channels)  <-- we use these
    data = df.iloc[:, 2:2 + N_CHANS].to_numpy(dtype="float64")  # (T, 8)
    epoch_raw = data.T  # (8, T)

    if epoch_raw.shape[1] < 660:
        raise ValueError(
            f"Not enough samples in CSV ({epoch_raw.shape[1]}); need >= 660"
        )

    epoch_pp = preprocess_epoch(epoch_raw)  # (8, time)

    if epoch_pp.ndim != 2 or epoch_pp.shape[0] != N_CHANS:
        raise ValueError(
            f"Unexpected preprocessed shape {epoch_pp.shape}; expected ({N_CHANS}, T)"
        )

    x = torch.from_numpy(epoch_pp.astype(np.float32))[None, None, ...].to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())

    return {
        "filename": filename,
        "prediction": pred_class,
    }


# ------------------------------------------------------------
# Endpoints: health + original predict
# ------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    block_idx: int = Form(0),
    trial_idx: int = Form(0),
    target_idx: int = Form(0),
):
    """
    Classify one EEG epoch chosen by (block_idx, trial_idx, target_idx).
    These are read from multipart form fields so they work with your
    Streamlit front-end (which sends them in `data={...}`).
    """
    if not file.filename.endswith(".mat"):
        raise HTTPException(status_code=400, detail="Only .mat files are supported")

    contents = await file.read()

    print(f"[EEG] indices: block={block_idx}, trial={trial_idx}, target={target_idx}")

    try:
        epoch_raw = _extract_epoch_from_mat(
            contents,
            block_idx=block_idx,
            trial_idx=trial_idx,
            target_idx=target_idx,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading .mat: {e}")

    try:
        epoch_pp = preprocess_epoch(epoch_raw)  # (ch, time)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing EEG: {e}")

    if epoch_pp.ndim != 2 or epoch_pp.shape[0] != N_CHANS:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected preprocessed shape {epoch_pp.shape}; expected ({N_CHANS}, T)",
        )

    x = torch.from_numpy(epoch_pp.astype(np.float32))[None, None, ...].to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())

    return {
        "filename": file.filename,
        "block_idx": block_idx,
        "trial_idx": trial_idx,
        "target_idx": target_idx,
        "prediction": pred_class,
    }


# ------------------------------------------------------------
# Live EEG prediction: upload + auto-watcher
# ------------------------------------------------------------

EEG_LABELS: Dict[int, str] = {i: f"class_{i}" for i in range(N_CLASSES)}

LAST_AUTO_PREDICTION: Optional[Dict[str, object]] = None


@app.post("/predict_eeg_live")
async def predict_eeg_live(file: UploadFile = File(...)):
    """
    Predict EEG target (12-way) from an uploaded BrainFlow RAW CSV.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")

    raw_bytes = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep="\t", header=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    print(f"[EEG CSV upload] shape={df.shape}")

    try:
        result = _predict_from_brainflow_df(df, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["label"] = EEG_LABELS.get(result["prediction"], "unknown")
    return result


@app.get("/last_auto_prediction")
async def last_auto_prediction():
    """
    Get the last automatic prediction made from the recordings watcher.
    """
    if LAST_AUTO_PREDICTION is None:
        raise HTTPException(status_code=404, detail="No automatic prediction yet.")
    return LAST_AUTO_PREDICTION


# ------------------------------------------------------------
# Background watcher: automatically predict latest OpenBCI CSV
# ------------------------------------------------------------


def _watch_recordings_folder():
    """
    Background thread: poll the OpenBCI recordings folder for new BrainFlow-RAW_*.csv
    When a new one appears (after 'Stop Recording'), run prediction automatically.
    """
    global LAST_AUTO_PREDICTION

    seen: set[Path] = set()

    while True:
        try:
            if not RECORDINGS_ROOT.exists():
                time.sleep(2.0)
                continue

            # Find all matching BrainFlow CSVs
            csv_files = list(RECORDINGS_ROOT.rglob(EEG_CSV_GLOB))
            if not csv_files:
                time.sleep(2.0)
                continue

            latest_path: Path = max(csv_files, key=lambda p: p.stat().st_mtime)

            if latest_path not in seen:
                # Wait a bit to ensure file is fully written
                time.sleep(1.0)
                print(f"[AUTO EEG] New CSV detected: {latest_path}")

                try:
                    with latest_path.open("rb") as f:
                        raw_bytes = f.read()
                    df = pd.read_csv(io.BytesIO(raw_bytes), sep="\t", header=None)
                    print(f"[AUTO EEG] CSV shape: {df.shape}")

                    result = _predict_from_brainflow_df(df, filename=str(latest_path))
                    result["label"] = EEG_LABELS.get(result["prediction"], "unknown")
                    LAST_AUTO_PREDICTION = result

                    print(f"[AUTO EEG] Prediction: {result}", flush=True)
                    seen.add(latest_path)
                except Exception as e:
                    print(f"[AUTO EEG] Error processing {latest_path}: {e}", flush=True)

        except Exception as e:
            print(f"[AUTO EEG] Watcher error: {e}", flush=True)

        time.sleep(2.0)


@app.on_event("startup")
def _start_watcher():
    """
    Launch the recordings watcher thread when FastAPI starts.
    """
    t = threading.Thread(target=_watch_recordings_folder, daemon=True)
    t.start()
    print(f"[AUTO EEG] Watcher started on {RECORDINGS_ROOT}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast:app", host="0.0.0.0", port=8000, reload=True)
