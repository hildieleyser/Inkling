"""
Simulated EEG/EMG generator for quick model testing.

Produces:
  - eeg:   (n_trials, n_samples) synthetic SSVEP waveforms (single channel)
  - labels:(n_trials,) class ids 0..11 matching the freqs below
  - emg:   (n_trials, emg_samples) EMG-like bursts for confirm/reject
  - fs_eeg, fs_emg: sampling rates

Default freqs/phases mirror the 12-target keypad mapping in 12_targets.py.
"""
import numpy as np

# 12-target freqs/phases (copied to avoid importing PsychoPy-heavy module)
FREQS_MAIN = [
    8.0, 8.5, 9.0, 9.5,
    10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 13.0, 13.5,
]

PHASES_MAIN = [
    0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
    0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
    0.0, np.pi / 2, np.pi, 3 * np.pi / 2,
]


def synth_ssvep(freq, phase, fs=250, duration=1.5, noise_sd=0.35, harmonics=(1, 2)):
    """Single-channel SSVEP-like signal with a couple harmonics and Gaussian noise."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    sig = np.zeros_like(t)
    for h in harmonics:
        sig += (1.0 / len(harmonics)) * np.sin(2 * np.pi * h * freq * t + phase)
    noise = np.random.normal(scale=noise_sd, size=n)
    return sig + noise


def synth_emg(fs=1000, duration=1.0, want_burst=True, burst_amp=3.0, noise_sd=0.3):
    """Rudimentary EMG-like waveform with optional burst segment."""
    n = int(duration * fs)
    sig = np.random.normal(scale=noise_sd, size=n)
    if want_burst:
        start = np.random.randint(0, n - int(0.15 * fs))
        end = start + int(0.1 * fs)
        sig[start:end] += np.random.normal(scale=burst_amp, size=end - start)
    return sig


def generate_dataset(
    n_trials=240,
    fs_eeg=250,
    eeg_duration=1.5,
    eeg_noise_sd=0.35,
    fs_emg=1000,
    emg_duration=1.0,
    emg_burst_prob=0.7,
):
    """Generate synthetic EEG/EMG arrays and labels."""
    labels = np.random.randint(0, len(FREQS_MAIN), size=n_trials)
    emg_labels = np.random.rand(n_trials) < emg_burst_prob

    eeg = np.stack([
        synth_ssvep(
            FREQS_MAIN[label],
            PHASES_MAIN[label],
            fs=fs_eeg,
            duration=eeg_duration,
            noise_sd=eeg_noise_sd,
        )
        for label in labels
    ])

    emg = np.stack([
        synth_emg(fs=fs_emg, duration=emg_duration, want_burst=bool(flag))
        for flag in emg_labels
    ])

    return {
        "eeg": eeg,
        "labels": labels,
        "fs_eeg": fs_eeg,
        "fs_emg": fs_emg,
        "emg": emg,
        "emg_labels": emg_labels.astype(np.int8),
    }


def save_dataset(path="simulated_eeg_emg.npz", **kwargs):
    data = generate_dataset(**kwargs)
    np.savez(path, **data)
    return path, data


if __name__ == "__main__":
    path, data = save_dataset()
    print(f"Wrote {path}")
    print({k: v.shape if hasattr(v, 'shape') else v for k, v in data.items()})
