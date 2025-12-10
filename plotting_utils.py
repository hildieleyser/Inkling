import numpy as np
import matplotlib.pyplot as plt

def plot_preprocessed_epoch(epoch_pp: np.ndarray, fs: float):
    n_channels, n_samples = epoch_pp.shape
    time_pp = np.arange(n_samples) / fs

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Preprocessed Signal (all channels)")

    for ch_idx in range(n_channels):
        ax.plot(time_pp, epoch_pp[ch_idx, :] + ch_idx * 5, label=f"Ch {ch_idx+1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z-scored Amplitude + offset")
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    return fig

def plot_raw_epoch(epoch_raw: np.ndarray, fs: float):
    n_channels, n_samples = epoch_raw.shape
    time_raw = np.arange(n_samples) / fs

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Raw Signal (all channels)")

    for ch_idx in range(n_channels):
        ax.plot(time_raw, epoch_raw[ch_idx, :] + ch_idx * 5, label=f"Ch {ch_idx+1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude + offset")
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    return fig

def plot_class_probabilities(probs: np.ndarray, freq_per_target: np.ndarray):
    n_classes = len(probs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_classes), probs, tick_label=[f"{freq:.1f} Hz" for freq in freq_per_target])
    ax.set_title("Class Probabilities")
    ax.set_xlabel("Target Frequency")
    ax.set_ylabel("Probability")
    fig.tight_layout()
    return fig
