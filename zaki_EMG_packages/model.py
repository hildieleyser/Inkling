# zaki_EMG_packages/model.py

from pathlib import Path
from tensorflow.keras.models import load_model

PACKAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "emg_model.h5"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model
