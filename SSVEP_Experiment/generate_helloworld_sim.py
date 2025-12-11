import numpy as np

# Your true SSVEP frequencies from the speller
FREQS_MAIN = [
    9.25, 11.25, 13.25,
    9.75, 11.75, 13.75,
    10.25, 12.25, 14.25,
    10.75, 12.75, 14.75
]

def generate_sim_dataset(
    outfile="eeg_emg_helloworld.npz",
    n_trials=200,
    fs_eeg=250,
    fs_emg=1000,
    eeg_len=710,
    emg_len=10000
):
    """
    Generates *strong*, *clean*, *FFT-detectable* SSVEP signals
    that match EXACTLY what the speller's _freq_logits_from_signal()
    is expecting.
    """

    eeg = []
    labels = []

    emg = []
    emg_labels = []

    for i in range(n_trials):

        # ----------------------------
        # 1. CHOOSE A LABEL 0–11
        # ----------------------------
        label = i % 12   # sequential → predictable
        # OR use random: label = np.random.randint(0, 12)

        labels.append(label)

        # ----------------------------
        # 2. EEG SYNTHESIS
        # ----------------------------
        t = np.arange(eeg_len) / fs_eeg
        freq = FREQS_MAIN[label]

        # STRONG sinusoid so FFT will ALWAYS detect the right target:
        sig = (
            1.8 * np.sin(2 * np.pi * freq * t)     # strong SSVEP
            + 0.2 * np.random.randn(eeg_len)       # small noise
        )

        eeg.append(sig.astype(np.float32))

        # ----------------------------
        # 3. EMG SYNTHESIS (simple YES/NO)
        # ----------------------------
        yes = np.random.rand() > 0.5
        emg_labels.append(int(yes))

        base = 0.05 * np.random.randn(emg_len)

        if yes:
            burst = np.zeros(emg_len)
            burst[2000:3000] = 1.0 + 0.5 * np.random.randn(1000)
            sig_emg = base + burst
        else:
            sig_emg = base

        emg.append(sig_emg.astype(np.float32))

    eeg = np.stack(eeg)
    emg = np.stack(emg)
    labels = np.array(labels, dtype=np.int64)
    emg_labels = np.array(emg_labels, dtype=np.int64)

    # ----------------------------
    # SAVE NPZ
    # ----------------------------
    np.savez(
        outfile,
        eeg=eeg,
        labels=labels,
        fs_eeg=float(fs_eeg),
        emg=emg,
        emg_labels=emg_labels,
        fs_emg=float(fs_emg)
    )

    print("Saved simulation →", outfile)
    print("EEG:", eeg.shape)
    print("EMG:", emg.shape)
    print("Labels:", labels[:20])


if __name__ == "__main__":
    generate_sim_dataset()
