# zaki_EMG_packages/predict.py

from pathlib import Path
from typing import Union

import numpy as np

from .extract import load_emg
from .preprocess import prepare_for_model, INPUT_LENGTH, BEST_CHANNELS
from .model import get_model, get_scaler


def predict_from_hdf5(path: Union[str, Path], dataset_name: str = "0") -> int:
    """
    High-level helper:
    - load EMG trial from HDF5
    - preprocess to match training (4 best channels, scaling, fixed length)
    - run CNN to get P(power)
    - threshold at 0.5 -> 0 = rest, 1 = power

    Args:
        path: path to .hdf5 file
        dataset_name: key inside HDF5 file (e.g. "0", "1", ...)

    Returns:
        int: 0 (rest) or 1 (power)
    """
    path = Path(path)

    # 1) load raw EMG (16, N_time)
    signal_16ch = load_emg(path, dataset_name=dataset_name)

    # 2) prepare for model: (1, time, channels)
    scaler = get_scaler()
    x = prepare_for_model(
        signal_16ch,
        scaler=scaler,
        channels=BEST_CHANNELS,
        target_length=INPUT_LENGTH,
    )

    # 3) predict
    model = get_model()
    proba = model.predict(x)[0, 0]  # scalar probability

    # 4) convert to 0/1
    return int(proba >= 0.5)
