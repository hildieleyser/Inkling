# predict.py
import numpy as np
import torch

from ssvep import EEGNet
from ssvep.data import preprocess_epoch
from ssvep.config import FREQ_PER_TARGET  # optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "eegnet_tuned.pth"
TEST_DATA_PATH = "X_test.npy"


def load_model(model_path=MODEL_PATH):
    model = EEGNet(n_chans=8, n_samples=500, n_classes=12)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def predict_epoch(model, epoch):
    """
    epoch: np.ndarray of shape (8, 500), preprocessed.
    """
    assert epoch.shape == (8, 500)
    with torch.no_grad():
        x = torch.from_numpy(epoch[None, None, ...]).float().to(DEVICE)
        pred = int(model(x).argmax(1).cpu().item())
    return pred


def predict_raw_epoch(model, epoch_raw):
    """
    epoch_raw: np.ndarray of shape (8, 710)
    """
    epoch = preprocess_epoch(epoch_raw)
    return predict_epoch(model, epoch)


def main():
    model = load_model()

    # Example: test on held-out epoch
    X_test_raw = np.load(TEST_DATA_PATH)   # RAW or preprocessed depending on what you saved
    epoch_raw = X_test_raw[0]

    pred = predict_raw_epoch(model, epoch_raw)

    print(f"Predicted class index: {pred}")
    print(f"Frequency: {FREQ_PER_TARGET[pred]:.2f} Hz")


if __name__ == "__main__":
    main()
