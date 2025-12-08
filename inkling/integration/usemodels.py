"""
Integration sketch that wires the actual EEG/EMG model helpers into the
functional speller pipeline.

The flow:
1) Run EEG stage 1 (12 targets) -> softmax probabilities -> propose_stage1_eeg.
2) Run EMG -> yes/no probability -> handle_emg_confirmation.
3) If stage 1 accepted, run EEG stage 2 (3 targets inside chosen panel) -> propose_stage2_eeg.
4) Run EMG again -> handle_emg_confirmation -> letter appended to state["text"].

Notes:
- EEG model (`ssvep-eegnet/predict.py`) by default returns an argmax index; we
  apply softmax to its logits to get a length-12 probability vector.
- EMG model (`emg-model/emg/predict.py`) returns y_prob (sigmoid) and y_pred;
  we use y_prob as the "yes" probability and 1 - y_prob as "no".
- Replace the dummy `run_stage2_eeg` with your real 3-target stage 2 model.
"""

from importlib import util as import_util
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

# Make the model helper modules importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SSVEP_DIR = PROJECT_ROOT / "inkling" / "ssvep-eegnet"
EMG_DIR = PROJECT_ROOT / "inkling" / "emg-model" / "emg"
SSVEP_PREDICT = SSVEP_DIR / "predict.py"
EMG_PREDICT = EMG_DIR / "predict.py"
EMG_MODEL_DEF = EMG_DIR / "model.py"


def _load_module(path: Path, name: str):
    spec = import_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


ssvep_predict = _load_module(SSVEP_PREDICT, "ssvep_predict")
emg_predict = _load_module(EMG_PREDICT, "emg_predict")
emg_model_def = _load_module(EMG_MODEL_DEF, "emg_model_def")

load_ssvep_model = ssvep_predict.load_model
predict_emg = emg_predict.predict_emg
build_emg_model = emg_model_def.build_emg_model

# Allow inkling_speller import from project root
import sys

sys.path.append(str(PROJECT_ROOT))

from inkling_speller import (
    handle_emg_confirmation,
    init_speller_state,
    propose_stage1_eeg,
    propose_stage2_eeg,
    speller_status,
)


# ---- EEG helpers ------------------------------------------------------------

def run_stage1_eeg(model, epoch_8x500: np.ndarray) -> Sequence[float]:
    """
    Convert EEGNet logits to a length-12 probability vector via softmax.
    """
    assert epoch_8x500.shape == (8, 500)
    with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.from_numpy(epoch_8x500[None, None, ...]).float().to(device)
        logits = model(x)  # shape (1, 12)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs.tolist()


def run_stage2_eeg(panel_index: int) -> Sequence[float]:
    """
    Placeholder for the 3-target stage 2 EEG model.
    Replace with your real model; must return length-3 probabilities.
    """
    del panel_index  # unused in placeholder
    return [0.7, 0.2, 0.1]  # dummy probs; replace with real call


# ---- EMG helpers ------------------------------------------------------------

def load_emg_model(weights_path: Path, input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = build_emg_model(input_shape)
    model.load_weights(weights_path)
    return model


def run_emg(model: tf.keras.Model, batch: np.ndarray) -> Tuple[float, float]:
    """
    Returns (yes_prob, no_prob) from the sigmoid output.
    """
    y_prob, _ = predict_emg(model, batch, threshold=0.5)
    yes = float(y_prob.squeeze())
    return yes, 1.0 - yes


# ---- Main integration loop --------------------------------------------------

def main():
    # Load models
    ssvep_model = load_ssvep_model(SSVEP_DIR / "eegnet_tuned.pth")
    # TODO: set correct input shape and weights path for your EMG model
    emg_model = load_emg_model(EMG_DIR / "emg_weights.h5", input_shape=(10000, 16))

    state = init_speller_state()

    while True:
        # 1) Stage 1 EEG over 12 targets
        epoch = np.zeros((8, 500), dtype=np.float32)  # TODO: replace with live data
        s1_probs = run_stage1_eeg(ssvep_model, epoch)
        cand1 = propose_stage1_eeg(state, s1_probs)
        if not cand1:
            continue

        # 2) EMG confirmation for stage 1
        emg_batch = np.zeros((1, 10000, 16), dtype=np.float32)  # TODO: replace with live data
        emg_yes, emg_no = run_emg(emg_model, emg_batch)
        if handle_emg_confirmation(state, emg_yes, emg_no) != "accepted_stage1":
            continue

        # 3) Stage 2 EEG over 3 targets in the chosen panel
        s2_probs = run_stage2_eeg(panel_index=cand1["index"])
        cand2 = propose_stage2_eeg(state, s2_probs)
        if not cand2:
            continue

        # 4) EMG confirmation for the letter
        emg_yes, emg_no = run_emg(emg_model, emg_batch)
        result = handle_emg_confirmation(state, emg_yes, emg_no)
        if result == "accepted_letter":
            print("Typed buffer:", state["text"])
            print("Status:", speller_status(state))


if __name__ == "__main__":
    main()
