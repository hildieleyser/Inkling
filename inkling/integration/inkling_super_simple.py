import numpy as np

LETTER_GROUPS = [
    ["A", "B", "C"],   # panel 0
    ["D", "E", "F"],   # panel 1
    ["G", "H", "I"],   # panel 2
    ["J", "K", "L"],   # panel 3
    ["M", "N", "O"],   # panel 4
    ["P", "Q", "R"],   # panel 5
    ["S", "T", "U"],   # panel 6
    ["V", "W", "X"],   # panel 7
    ["Y", "Z", " "],   # panel 8
    [".", ",", "?"],   # panel 9
    ["BACK", "-", "_"],# panel 10
    ["OK", "NO", "!"], # panel 11
]


def eeg_argmax(probs):
    """Return index of the largest probability."""
    return int(np.argmax(probs))


def emg_yes(prob, threshold=0.5):
    """Return True if EMG 'yes' probability crosses threshold."""
    return prob >= threshold


def select_letter_once(
    eeg12_probs,
    emg_stage1_yes_prob,
    eeg3_probs,
    emg_stage2_yes_prob,
    emg_threshold=0.5,
    letter_groups=LETTER_GROUPS,
):
    """
    One full cycle:
      - Pick panel from 12 EEG probs
      - Confirm with EMG
      - Pick letter from 3 EEG probs (in that panel)
      - Confirm with EMG

    Returns:
      selected_letter (str or None)
    """

    # Stage 1: choose panel
    panel_idx = eeg_argmax(eeg12_probs)

    # Confirm panel with EMG
    if not emg_yes(emg_stage1_yes_prob, emg_threshold):
        return None

    # Stage 2: choose letter within panel
    letter_idx = eeg_argmax(eeg3_probs)
    letter = letter_groups[panel_idx][letter_idx]

    # Confirm letter with EMG
    if not emg_yes(emg_stage2_yes_prob, emg_threshold):
        return None

    return letter


def update_text(text, letter):
    """Append letter to text; handle BACK specially."""
    if letter is None:
        return text

    if letter == "BACK":
        return text[:-1]

    return text + letter
