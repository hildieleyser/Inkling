from pathlib import Path
import io
import os
import sys
import importlib
from typing import Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from scipy.io import loadmat

# ============================================================
# PATH SETUP & DYNAMIC IMPORT OF YOUR REAL 'ssvep' PACKAGE
# ============================================================

HERE = Path(__file__).resolve().parent          # .../inkling/api_eeg
PROJECT_ROOT = HERE.parent                      # .../inkling  (repo root)


def _find_ssvep_package(start: Path) -> Tuple[Path, Path]:
    """
    Search under `start` for a directory called 'ssvep' that looks like a package.
    Returns (package_dir, package_parent).
    Raises RuntimeError if not found.
    """
    for root, dirs, files in os.walk(start):
        if "ssvep" in dirs:
            pkg_dir = Path(root) / "ssvep"
            if (pkg_dir / "__init__.py").exists():
                return pkg_dir, Path(root)
    raise RuntimeError(
        f"Could not find 'ssvep' package under {start}. "
        "Make sure your project contains a 'ssvep' folder with an __init__.py."
    )


def _import_ssvep():
    """
    Locate the ssvep package, add its parent to sys.path, and import:
      - ssvep.EEGNet
      - ssvep.data.preprocess_epoch
    """
    pkg_dir, parent = _find_ssvep_package(PROJECT_ROOT)

    # Add the parent directory so `import ssvep` works
    parent_str = str(parent)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)

    try:
        ssvep = importlib.import_module("ssvep")
        data_mod = importlib.import_module("ssvep.data")
    except ImportError as e:
        raise RuntimeError(
            f"Found ssvep package at {pkg_dir}, but could not import it: {e}"
        )

    EEGNet = getattr(ssvep, "EEGNet")
    preprocess_epoch = getattr(data_mod, "preprocess_epoch")

    return EEGNet, preprocess_epoch


# Do the dynamic import once at module import time
EEGNet, preprocess_epoch = _import_ssvep()

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = HERE / "eegnet_tuned.pth"   # model is in api_eeg/
print(f"[fast.py] Loading model from: {MODEL_PATH}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These MUST match what you used in train.py / predict.py
N_CHANS = 8
N_SAMPLES = 500
N_CLASSES = 12

app = FastAPI(title="EEG SSVEP Inference API")

# ============================================================
# MODEL LOADING (USING YOUR REAL EEGNET)
# ============================================================


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = EEGNet(n_chans=N_CHANS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[fast.py] Model loaded and set to eval mode.")
    return model


model = load_model()

# ============================================================
# .MAT HANDLING
# ============================================================


def _extract_epoch_from_mat(
    contents: bytes,
    block_idx: int = 0,
    trial_idx: int = 0,
    target_idx: int = 0,
) -> np.ndarray:
    """
    Load .mat from raw bytes and return a 2D (channels, time) array
    suitable for preprocess_epoch.

    For typical Wearable SSVEP Dataset-style files with:
        data.shape == (8, 710, n_blocks, n_trials, n_targets)
    we take:
        epoch = data[:, :, block_idx, trial_idx, target_idx]  -> (8, 710)

    Fallback:
      - Accept arrays of any ndim
      - Squeeze singleton dims
      - If ndim > 2, progressively slice the first index until 2D
      - Ensure one dimension == N_CHANS (8) → channel axis
      - Transpose if needed so final shape is (N_CHANS, time)
      - Require at least 660 samples in the time dimension
    """
    bio = io.BytesIO(contents)
    mat = loadmat(bio)

    # ---- 1) Preferred path: explicit 'data' with multi-epoch structure ----
    if "data" in mat and isinstance(mat["data"], np.ndarray):
        data = np.array(mat["data"])
        # Expected: (n_chans, n_time, [n_block, n_trial, n_target, ...])
        if data.ndim >= 3 and data.shape[0] == N_CHANS:
            # If 5D: (ch, time, block, trial, target)
            if data.ndim >= 5:
                n_blocks = data.shape[2]
                n_trials = data.shape[3]
                n_targets = data.shape[4]

                if not (0 <= block_idx < n_blocks):
                    raise ValueError(
                        f"block_idx out of range: {block_idx} (valid 0..{n_blocks-1})"
                    )
                if not (0 <= trial_idx < n_trials):
                    raise ValueError(
                        f"trial_idx out of range: {trial_idx} (valid 0..{n_trials-1})"
                    )
                if not (0 <= target_idx < n_targets):
                    raise ValueError(
                        f"target_idx out of range: {target_idx} (valid 0..{n_targets-1})"
                    )

                epoch = data[:, :, block_idx, trial_idx, target_idx]  # (8, 710, ...)
                epoch = np.squeeze(epoch)

            # If 4D: (ch, time, block, epoch) – use block_idx and target_idx
            elif data.ndim == 4:
                n_blocks = data.shape[2]
                n_epochs = data.shape[3]
                if not (0 <= block_idx < n_blocks):
                    raise ValueError(
                        f"block_idx out of range: {block_idx} (valid 0..{n_blocks-1})"
                    )
                if not (0 <= target_idx < n_epochs):
                    raise ValueError(
                        f"target_idx out of range: {target_idx} (valid 0..{n_epochs-1})"
                    )
                epoch = data[:, :, block_idx, target_idx]
                epoch = np.squeeze(epoch)

            # If 3D: (ch, time, epoch) – use target_idx as epoch index
            elif data.ndim == 3:
                n_epochs = data.shape[2]
                if not (0 <= target_idx < n_epochs):
                    raise ValueError(
                        f"target_idx out of range: {target_idx} (valid 0..{n_epochs-1})"
                    )
                epoch = data[:, :, target_idx]
                epoch = np.squeeze(epoch)

            else:
                epoch = None

            if epoch is not None and epoch.ndim == 2:
                if epoch.shape[0] != N_CHANS:
                    epoch = epoch.T
                if epoch.shape[0] != N_CHANS:
                    raise ValueError(
                        f"Selected epoch does not have {N_CHANS} channels: {epoch.shape}"
                    )
                if epoch.shape[1] < 660:
                    raise ValueError(
                        f"Selected epoch does not have enough samples (got {epoch.shape[1]}, need >=660)."
                    )
                return epoch.astype(np.float64)

    # ---- 2) Fallback heuristic for any other .mat layout ----

    def _try_make_epoch(arr: np.ndarray) -> np.ndarray | None:
        a = np.array(arr)

        # Squeeze singleton dims
        a = np.squeeze(a)

        # If still >2D, keep taking the first slice along extra dims
        # until we get 2D or less
        while a.ndim > 2:
            a = a[..., 0]
            a = np.squeeze(a)

        if a.ndim != 2:
            return None

        # Make channels dimension first
        if a.shape[0] == N_CHANS:
            epoch_ = a
        elif a.shape[1] == N_CHANS:
            epoch_ = a.T
        else:
            return None

        # Need enough timepoints to safely do epoch[:, 160:660]
        if epoch_.shape[1] < 660:  # your pipeline crops 160:660 → 500
            return None

        return epoch_.astype(np.float64)

    # 2a) Try nice key names first
    for key in ["epoch", "EEG", "eeg", "X", "signal"]:
        if key in mat and isinstance(mat[key], np.ndarray):
            candidate = _try_make_epoch(mat[key])
            if candidate is not None:
                return candidate

    # 2b) Fallback: scan all arrays in the .mat
    for v in mat.values():
        if isinstance(v, np.ndarray):
            candidate = _try_make_epoch(v)
            if candidate is not None:
                return candidate

    raise ValueError("No suitable EEG array (8 channels, >=660 samples) found in .mat file")

# ============================================================
# API ENDPOINTS
# ============================================================


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    block_idx: int = Query(0, description="Block index (e.g. 0..1)"),
    trial_idx: int = Query(0, description="Trial index (e.g. 0..9)"),
    target_idx: int = Query(0, description="Target index (e.g. 0..11)"),
):
    """
    Upload a .mat file containing multiple epochs and choose which one to classify.

    For typical data arrays with shape (8, 710, n_blocks, n_trials, n_targets),
    you can select:
      - block_idx: which block/session
      - trial_idx: which trial within that block/target
      - target_idx: which SSVEP stimulus/target

    Uses your existing preprocess_epoch() and EEGNet model.
    """
    if not file.filename.endswith(".mat"):
        raise HTTPException(status_code=400, detail="Only .mat files are supported")

    contents = await file.read()

    # 1. Extract raw epoch from .mat
    try:
        epoch_raw = _extract_epoch_from_mat(
            contents,
            block_idx=block_idx,
            trial_idx=trial_idx,
            target_idx=target_idx,
        )   # shape (channels, time)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading .mat: {e}")

    # 2. Preprocess using your existing pipeline
    try:
        epoch_pp = preprocess_epoch(epoch_raw)          # shape (channels, time)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing EEG: {e}")

    if epoch_pp.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Expected preprocessed epoch to be 2D, got {epoch_pp.shape}",
        )

    chans, time = epoch_pp.shape
    if chans != N_CHANS:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_CHANS} channels after preprocessing, got {chans}",
        )

    # 3. Forward pass
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


# Optional: run with `python fast.py` directly (bypasses uvicorn command)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast:app", host="0.0.0.0", port=8000, reload=True)
