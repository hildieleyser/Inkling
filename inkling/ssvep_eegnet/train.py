# train.py
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from google.cloud import storage

from ssvep import EEGNet
from ssvep.data import preprocess_eeg_dataset

DEVICE = torch.device("cpu")  # CPU only


EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
PATIENCE = 6


def download_from_gcs(bucket_name, blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def upload_to_gcs(bucket_name, local_path, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def main(args):
    bucket = "inkling-ssvep-emg"
    data_prefix = args.data_prefix.rstrip("/")      # e.g. ssvep
    out_prefix = args.out_prefix.rstrip("/")        # e.g. ssvep/models

    os.makedirs("/tmp/data", exist_ok=True)
    os.makedirs("/tmp/out", exist_ok=True)

    print("Downloading X_dry.npy and y_dry.npy...")
    download_from_gcs(bucket, f"{data_prefix}/X_dry.npy", "/tmp/data/X_dry.npy")
    download_from_gcs(bucket, f"{data_prefix}/y_dry.npy", "/tmp/data/y_dry.npy")

    # Load RAW EEG (N, 8, 710)
    X_raw = np.load("/tmp/data/X_dry.npy")    # (N, 8, 710)
    y = np.load("/tmp/data/y_dry.npy")    # (N,)
    print("Loaded raw X:", X_raw.shape, "y:", y.shape)
    
    # Preprocess ONCE, before splitting
    X = preprocess_eeg_dataset(X_raw)         # (N, 8, 500)
    print("Preprocessed X:", X.shape)

    # ----------------------
    # Outer split: trainval vs test
    # ----------------------
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2025)
    trainval_idx, test_idx = next(sss_outer.split(X, y))
    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    np.save("/tmp/out/X_test.npy", X_test)
    np.save("/tmp/out/y_test.npy", y_test)

    # ----------------------
    # Inner split: train vs val
    # ----------------------
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2024)
    tr_idx, va_idx = next(sss_inner.split(X_trainval, y_trainval))

    X_tr = np.expand_dims(X_trainval[tr_idx], 1)
    y_tr = y_trainval[tr_idx]
    X_va = np.expand_dims(X_trainval[va_idx], 1)
    y_va = y_trainval[va_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).long()),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----------------------
    # Build model
    # ----------------------
    model = EEGNet(
        n_chans=X.shape[1],
        n_samples=X.shape[2],
        n_classes=len(np.unique(y)),
        dropout=DROPOUT
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    wait = 0
    ckpt_local = "/tmp/out/eegnet_tuned.pth"

    # ----------------------
    # Training loop
    # ----------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.numel()

        val_acc = correct / total
        print(f"Epoch {epoch:02d} | val acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            wait = 0
            torch.save(model.state_dict(), ckpt_local)
            print(f"  Saved best model to {ckpt_local}")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    print(f"BEST VAL ACCURACY: {best_val:.4f}")

    # ----------------------
    # Upload outputs to GCS
    # ----------------------
    print("Uploading outputs to GCS...")

    upload_to_gcs(bucket, ckpt_local, f"{out_prefix}/eegnet_tuned.pth")
    upload_to_gcs(bucket, "/tmp/out/X_test.npy", f"{out_prefix}/X_test.npy")
    upload_to_gcs(bucket, "/tmp/out/y_test.npy", f"{out_prefix}/y_test.npy")

    print("DONE training & uploading.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", default="ssvep")
    parser.add_argument("--out_prefix", default="ssvep/models")
    args = parser.parse_args()
    main(args)
