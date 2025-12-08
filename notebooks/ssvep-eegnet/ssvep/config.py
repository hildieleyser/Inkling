# ssvep/config.py
import numpy as np

# ----- EEG Data Parameters -----

FS = 250          # sampling rate (Hz)
N_HARMONICS = 5   # harmonics for refs

# ----- Stimulation mapping from stimulation_information.pdf -----
FREQ_PER_TARGET = np.array([
    9.25,   # target 1  → '1'
    11.25,  # target 2  → '2'
    13.25,  # target 3  → '3'
    9.75,   # target 4  → '4'
    11.75,  # target 5  → '5'
    13.75,  # target 6  → '6'
    10.25,  # target 7  → '7'
    12.25,  # target 8  → '8'
    14.25,  # target 9  → '9'
    10.75,  # target 10 → '0'
    12.75,  # target 11 → '*'
    14.75   # target 12 → '#'
], dtype=float)

PHASE_PER_TARGET = np.array([
    0.0 * np.pi,  # 1
    0.0 * np.pi,  # 2
    0.0 * np.pi,  # 3
    0.5 * np.pi,  # 4
    0.5 * np.pi,  # 5
    0.5 * np.pi,  # 6
    1.0 * np.pi,  # 7
    1.0 * np.pi,  # 8
    1.0 * np.pi,  # 9
    1.5 * np.pi,  # 10
    1.5 * np.pi,  # 11
    1.5 * np.pi   # 12
], dtype=float)
