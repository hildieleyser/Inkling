# predict.py
import numpy as np
import torch

from ssvep import EEGNet
from ssvep.config import FREQ_PER_TARGET  # if you want to show Hz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = EEGNet(n_chans=8, n_samples=500, n_classes=12)
    model.load_state_dict(torch.load("eegnet_tuned.pth", map_location=DEVICE))
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

def main():
    # example: load some test epoch
    X_test = np.load("X_test.npy")
    epoch = X_test[0]
    model = load_model()
    pred = predict_epoch(model, epoch)
    print(f"Predicted target index: {pred}")
    print(f"Frequency: {FREQ_PER_TARGET[pred]:.2f} Hz")

if __name__ == "__main__":
    main()


