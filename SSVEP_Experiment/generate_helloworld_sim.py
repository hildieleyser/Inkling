"""
Generate deterministic simulated EEG/EMG data that spells "HELLO WORLD".

Outputs:
  - eeg_emg_helloworld.npz (combined EEG+EMG for SIM_DATA_PATH)
  - eeg_helloworld.npz (EEG-only)
  - emg_helloworld.npz (EMG-only)
"""
import numpy as np
from simulated_data import (
    synth_ssvep,
    synth_emg,
    FREQS_MAIN,
    PHASES_MAIN,
)

# Sequence of (main-key class id, letter class id) pairs to spell "HELLO WORLD"
# Keypad class ids follow KEYPAD_LABELS order; letter class ids are global 12-class IDs.
HELLO_WORLD_EEG_LABELS = [
    3, 9,   # H  (key 4 -> class 3; letter H -> class 9)
    2, 5,   # E  (key 3 -> class 2; letter E -> class 5)
    4, 2,   # L  (key 5 -> class 4; letter L -> class 2)
    4, 2,   # L
    5, 7,   # O  (key 6 -> class 5; letter O -> class 7)
    10, 1,  # space (key 0 -> class 10; space -> class 1)
    8, 6,   # W  (key 9 -> class 8; letter W -> class 6)
    5, 7,   # O
    6, 11,  # R  (key 7 -> class 6; letter R -> class 11)
    4, 2,   # L
    2, 4,   # D  (key 3 -> class 2; letter D -> class 4)
]

# EMG confirmations: 1 (confirm) for every step (key + letter)
HELLO_WORLD_EMG_LABELS = [1] * len(HELLO_WORLD_EEG_LABELS)


def generate_helloworld(
    fs_eeg=250,
    eeg_duration=1.5,
    eeg_noise_sd=0.3,
    fs_emg=1000,
    emg_duration=1.0,
    emg_noise_sd=0.3,
    emg_burst_amp=3.0,
):
    eeg = np.stack([
        synth_ssvep(
            FREQS_MAIN[label],
            PHASES_MAIN[label],
            fs=fs_eeg,
            duration=eeg_duration,
            noise_sd=eeg_noise_sd,
        )
        for label in HELLO_WORLD_EEG_LABELS
    ])

    emg = np.stack([
        synth_emg(
            fs=fs_emg,
            duration=emg_duration,
            want_burst=bool(flag),
            burst_amp=emg_burst_amp,
            noise_sd=emg_noise_sd,
        )
        for flag in HELLO_WORLD_EMG_LABELS
    ])

    return {
        "eeg": eeg,
        "labels": np.array(HELLO_WORLD_EEG_LABELS, dtype=np.int32),
        "fs_eeg": fs_eeg,
        "emg": emg,
        "emg_labels": np.array(HELLO_WORLD_EMG_LABELS, dtype=np.int8),
        "fs_emg": fs_emg,
    }


def main():
    data = generate_helloworld()
    np.savez("eeg_emg_helloworld.npz", **data)
    np.savez("eeg_helloworld.npz", eeg=data["eeg"], labels=data["labels"], fs_eeg=data["fs_eeg"])
    np.savez("emg_helloworld.npz", emg=data["emg"], emg_labels=data["emg_labels"], fs_emg=data["fs_emg"])
    print("Wrote eeg_emg_helloworld.npz, eeg_helloworld.npz, emg_helloworld.npz")
    print("EEG shape:", data["eeg"].shape, "labels len:", len(data["labels"]))
    print("EMG shape:", data["emg"].shape, "emg_labels len:", len(data["emg_labels"]))


if __name__ == "__main__":
    main()
