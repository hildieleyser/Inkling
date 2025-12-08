import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1) Filter and build X, y
# ---------------------------------------------------------------------
def build_Xy_two_class(emg_df, signal_col="signal_fixed", label_col="grasp"):
    """
    Filters dataset to only grasp 1 (power) and grasp 5 (open),
    then maps labels to 0 and 1.

    Returns:
        X: (n_trials, 16, 10000)
        y: (n_trials,) with values {0,1}
    """

    # --- Filter classes ---
    emg_df = emg_df[emg_df[label_col].isin([1, 5])].copy()

    # --- Build X ---
    X = np.stack(emg_df[signal_col].values).astype(np.float32)

    # --- Build y and remap ---
    label_map = {5: 0, 1: 1}   # power → 1, open → 0
    y = emg_df[label_col].map(label_map).astype(int).values

    return X, y


# ---------------------------------------------------------------------
# 2) Train/test split
# ---------------------------------------------------------------------
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ---------------------------------------------------------------------
# 3) Compute normalization stats from training data only
# ---------------------------------------------------------------------
def compute_normalization_stats(X_train):
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1,16,1)
    train_std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return train_mean, train_std


# ---------------------------------------------------------------------
# 4) Apply normalization
# ---------------------------------------------------------------------
def apply_normalization(X, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------
# 5) Final FULL pipeline
# ---------------------------------------------------------------------
def preprocess_emg_fulltrial(emg_df):
    """
    Full preprocessing for binary EMG classification:
    - Filter to grasps 1 & 5
    - Build X,y
    - Train/test split
    - Channel-wise z-score normalization
    - Return normalized tensors ready for Conv1D
    """

    # Build filtered dataset
    X, y = build_Xy_two_class(emg_df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compute stats on training ONLY
    mean, std = compute_normalization_stats(X_train)

    # Normalize
    X_train_norm = apply_normalization(X_train, mean, std)
    X_test_norm  = apply_normalization(X_test, mean, std)

    # Reshape for Conv1D → (n_samples, time, channels)
    X_train_norm = np.transpose(X_train_norm, (0, 2, 1))  # (N,10000,16)
    X_test_norm  = np.transpose(X_test_norm,  (0, 2, 1))

    return X_train_norm, X_test_norm, y_train, y_test
