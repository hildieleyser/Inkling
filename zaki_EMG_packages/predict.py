# zaki_EMG_packages/predict.py

from pathlib import Path
from typing import Union

from .extract import load_emg
from .preprocess import fix_length, reshape_for_model
from .model import get_model


def predict_from_hdf5(path: Union[str, Path], dataset_name="0") -> int:
    path = Path(path)
    signal = load_emg(path, dataset_name=dataset_name)
    signal = fix_length(signal, target_length=10000)
    x = reshape_for_model(signal)
    model = get_model()
    proba = model.predict(x)[0][0]
    return int(proba >= 0.5)
