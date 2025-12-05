# evaluate.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from ssvep import EEGNet
from ssvep.data import preprocess_eeg_dataset

# Load RAW test set
X_test_raw = np.load("X_test.npy")   # (N_test, 8, 710)
y_test = np.load("y_test.npy")

# Preprocess
X_test = preprocess_eeg_dataset(X_test_raw)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load test set saved by train.py
    X_test = np.load("X_test.npy")  # (N_test, 8, 500)
    y_test = np.load("y_test.npy")  # (N_test,)

    # Load model
    model = EEGNet(n_chans=X_test.shape[1], n_samples=X_test.shape[2],
                   n_classes=len(np.unique(y_test)))
    model.load_state_dict(torch.load("eegnet_tuned.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    X_test_t = torch.from_numpy(np.expand_dims(X_test, 1)).float().to(DEVICE)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), 64):
            logits = model(X_test_t[i:i+64])
            preds.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(cm)

if __name__ == "__main__":
    main()
