# prepare_dry_numpy.py
import numpy as np
import pandas as pd

from ssvep.load_data import get_eeg_data          # loads all subjects from GCS
from ssvep.data import preprocess_eeg_df     # adds signal_clean (8, 500)


def main():
    print("Loading full EEG dataset from GCS via get_eeg_data()...")
    df = get_eeg_data()  # columns: subject, electrode, block, target, signal (8,710)

    print("Filtering to dry electrode only...")
    df_dry = df[df["electrode"] == "dry"].reset_index(drop=True)
    print("Dry rows:", len(df_dry))

    print("Preprocessing epochs (bandpass, notch, trim, z-score)...")
    df_dry = preprocess_eeg_df(df_dry)  # adds "signal_clean" (8,500)

    print("Stacking into X_dry (N, 8, 500) and y_dry (N,)...")
    X_dry = np.stack(df_dry["signal_clean"].to_numpy(), axis=0)   # (N, 8, 500)
    # targets are 1..12 in df, convert to 0..11 for classification
    y_dry = df_dry["target"].to_numpy().astype(int) - 1          # (N,)

    print("X_dry shape:", X_dry.shape, "y_dry shape:", y_dry.shape)

    print("Saving locally...")
    np.save("X_dry.npy", X_dry)
    np.save("y_dry.npy", y_dry)
    print("Done. Files: X_dry.npy, y_dry.npy")


if __name__ == "__main__":
    main()
