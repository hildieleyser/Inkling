import numpy as np

LETTER_GROUPS = [
    ["A", "B", "C"],
    ["D", "E", "F"],
    ["G", "H", "I"],
    ["J", "K", "L"],
    ["M", "N", "O"],
    ["P", "Q", "R"],
    ["S", "T", "U"],
    ["V", "W", "X"],
    ["Y", "Z", " "],
    [".", ",", "?"],
    ["BACK", "-", "_"],
    ["OK", "NO", "!"],
]


def eeg_argmax(probs):
    return int(np.argmax(probs))


def emg_yes(prob, threshold=0.5):
    return prob >= threshold


def select_letter_once(eeg12_probs, emg_stage1_yes_prob, eeg3_probs, emg_stage2_yes_prob,
                       emg_threshold=0.5, letter_groups=LETTER_GROUPS):

    # Stage 1: pick panel from 12 EEG probabilities
    panel_idx = eeg_argmax(eeg12_probs)

    # EMG confirmation
    if not emg_yes(emg_stage1_yes_prob, emg_threshold):
        return None

    # Stage 2: pick letter from 3 EEG probabilities inside selected panel
    letter_idx = eeg_argmax(eeg3_probs)
    letter = letter_groups[panel_idx][letter_idx]

    # EMG confirmation
    if not emg_yes(emg_stage2_yes_prob, emg_threshold):
        return None

    return letter


def update_text(text, letter):
    if letter is None:
        return text
    if letter == "BACK":
        return text[:-1]
    return text + letter
