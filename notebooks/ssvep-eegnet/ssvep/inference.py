# ssvep/inference.py
import os
import numpy as np
import torch

from ssvep import EEGNet          # from your __init__.py
from ssvep.data import preprocess_epoch
from ssvep.config import FREQ_PER_TARGET  # list/array of 12 frequencies

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "eegnet_tuned.pth")


class EEGNetService:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> EEGNet:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Model matches training config
        model = EEGNet(n_chans=8, n_samples=500, n_classes=12)
        state = torch.load(self.model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model

    def predict_raw_epoch(self, epoch_raw: np.ndarray):
        """
        epoch_raw: np.ndarray of shape (8, 710)
        Returns: dict with predicted index, frequency, and probabilities.
        """
        assert epoch_raw.shape == (8, 710)

        # Preprocess to (8, 500)
        epoch_clean = preprocess_epoch(epoch_raw)

        # Prepare tensor: (1, 1, C, T)
        x = torch.from_numpy(epoch_clean[None, None, ...]).float().to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_freq = float(FREQ_PER_TARGET[pred_idx])

        return {
            "class_index": pred_idx,
            "frequency_hz": pred_freq,
            "probabilities": probs.tolist(),
        }
