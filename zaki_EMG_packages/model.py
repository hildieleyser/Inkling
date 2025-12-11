# zaki_EMG_packages/model.py

from pathlib import Path

import joblib
from tensorflow.keras.models import load_model

# This file lives in: <PROJECT_ROOT>/zaki_EMG_packages/model.py
# We want:            <PROJECT_ROOT>/artifacts_emg/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_emg"

MODEL_PATH = ARTIFACTS_DIR / "emg_cnn_4ch.h5"
SCALER_PATH = ARTIFACTS_DIR / "emg_scaler.pkl"

_model = None
_scaler = None


def get_model():
    """Lazy-load Keras model from artifacts_emg."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def get_scaler():
    """Lazy-load StandardScaler from artifacts_emg."""
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler
