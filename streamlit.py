# ============================================
# NOKIA KEYPAD + EEG / EMG LETTER SELECTION
# ============================================

# --- Main keypad layout (EEG model 1: classes 0–11) ---
KEYPAD_LABELS = [
    "1", "2", "3",
    "4", "5", "6",
    "7", "8", "9",
    "*", "0", "#"
]

# -----------------------------------------------------
# LETTER MAP
# -----------------------------------------------------
# Which letters belong to each key
# (Empty = no letter panel, immediate selection)
LETTER_MAP = {
    "1": [],
    "2": ["A", "B", "C"],
    "3": ["D", "E", "F"],
    "4": ["G", "H", "I"],
    "5": ["J", "K", "L"],
    "6": ["M", "N", "O"],
    "7": ["P", "Q", "R", "S"],
    "8": ["T", "U", "V"],
    "9": ["W", "X", "Y", "Z"],
    "0": [" "],          # space handled in letter panel
    "*": [],
    "#": [],
}

# -----------------------------------------------------
# LETTER CLASS IDS (LOCAL INDICES FOR LETTER EEG MODEL)
# -----------------------------------------------------
# These are LOCAL indices returned by the LETTER EEG model.
#
# 0 = key itself
# 1.. = letters
# Special case:
#   key "0" → [ "0", " ", "<BACK>" ]
#
LETTER_CLASS_IDS = {
    "1": [0],
    "*": [0],
    "#": [0],

    "2": [0, 1, 2, 3],          # 2 + A,B,C
    "3": [0, 1, 2, 3],          # 3 + D,E,F
    "4": [0, 1, 2, 3],          # 4 + G,H,I
    "5": [0, 1, 2, 3],          # 5 + J,K,L
    "6": [0, 1, 2, 3],          # 6 + M,N,O
    "7": [0, 1, 2, 3, 4],       # 7 + P,Q,R,S
    "8": [0, 1, 2, 3],          # 8 + T,U,V
    "9": [0, 1, 2, 3, 4],       # 9 + W,X,Y,Z

    "0": [0, 1, 2],             # 0, SPACE, BACK
}

# -----------------------------------------------------
# LETTER PANEL CONSTRUCTION
# -----------------------------------------------------
def build_letter_panel(key: str) -> list[str]:
    """
    Returns the list of selectable items for the given key.
    This list aligns exactly with LETTER_CLASS_IDS[key].
    """
    if key == "0":
        return ["0", " ", "<BACK>"]

    if key in LETTER_MAP and LETTER_MAP[key]:
        return [key] + LETTER_MAP[key]

    # keys with no letters (1, *, #)
    return [key]

# -----------------------------------------------------
# LETTER EEG SELECTION
# -----------------------------------------------------
def select_letter_from_eeg(key: str, predicted_index: int) -> str:
    """
    Given a key and the local index predicted by the letter EEG model,
    return the selected character.
    """
    panel_items = build_letter_panel(key)
    allowed = LETTER_CLASS_IDS[key]

    if predicted_index not in allowed:
        raise ValueError(
            f"Invalid letter EEG index {predicted_index} for key {key}"
        )

    return panel_items[predicted_index]

# -----------------------------------------------------
# EXAMPLE FLOW (LOGIC ONLY)
# -----------------------------------------------------
def example_usage():
    # EEG model 1 predicts class 3 → keypad key "4"
    eeg_key_class = 3
    key = KEYPAD_LABELS[eeg_key_class]      # "4"

    # Build the letter panel
    panel = build_letter_panel(key)          # ["4", "G", "H", "I"]

    # EEG model 2 predicts local index (e.g. 2)
    eeg_letter_index = 2

    # Select letter
    char = select_letter_from_eeg(key, eeg_letter_index)
    return char  # "H"
