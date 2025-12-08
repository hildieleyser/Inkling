from fastapi import FastAPI
import numpy as np

from inkling.integration.inkling_simple import select_letter_once, update_text

app = FastAPI()

# ---------------------------------------------------------
# A simple global text buffer (like your typing interface)
# ---------------------------------------------------------
TEXT = ""

@app.get("/")
def root():
    return {"message": "it is working"}

@app.get("/status")
def status():
    """Check current text."""
    return {"text": TEXT}


@app.post("/step")
def step(payload: dict):
    """
    The simplest EEG + EMG integration endpoint.

    Expected JSON body:
    {
        "eeg12": [12 floats],
        "eeg3": [3 floats],
        "emg1": float,
        "emg2": float
    }
    """

    global TEXT

    # -----------------------------
    # Load numbers from request
    # -----------------------------
    eeg12 = np.array(payload["eeg12"], dtype=float)   # shape (12,)
    eeg3  = np.array(payload["eeg3"], dtype=float)    # shape (3,)
    emg1  = float(payload["emg1"])                    # stage 1 confirmation
    emg2  = float(payload["emg2"])                    # stage 2 confirmation

    # -----------------------------
    # Core logic: choose a letter
    # -----------------------------
    letter = select_letter_once(
        eeg12_probs=eeg12,
        emg_stage1_yes_prob=emg1,
        eeg3_probs=eeg3,
        emg_stage2_yes_prob=emg2,
    )

    # -----------------------------
    # Update text interface
    # -----------------------------
    TEXT = update_text(TEXT, letter)

    return {
        "selected_letter": letter,
        "current_text": TEXT
    }
