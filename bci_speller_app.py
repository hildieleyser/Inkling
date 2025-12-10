# bci_speller_app.py
# Streamlit UI for EEG + EMG Nokia-style speller.

import streamlit as st
import requests
from io import BytesIO

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

EEG_API_URL = "http://localhost:8000/predict"
EMG_API_URL = "http://localhost:8001/predict_file"


def rerun():
    """Streamlit rerun helper (works on old/new versions)."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# -------------------------------------------------------------------
# KEYPAD + LETTER MAPPING
# -------------------------------------------------------------------

KEYPAD_LABELS = [
    "1", "2", "3",
    "4", "5", "6",
    "7", "8", "9",
    "*", "0", "#",
]

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
    "0": [" "],
    "*": [],
    "#": [],
}

# global 12-class IDs active in each key’s letter panel
LETTER_CLASS_IDS = {
    "1": [0],
    "*": [0],
    "#": [0],

    "0": [0, 1, 2],           # 0, space, back

    "2": [0, 1, 2, 3],        # 2 + A,B,C
    "3": [4, 5, 6, 7],        # 3 + D,E,F
    "4": [8, 9, 10, 11],      # 4 + G,H,I
    "5": [0, 1, 2, 4],        # 5 + J,K,L
    "6": [5, 6, 7, 8],        # 6 + M,N,O
    "7": [9, 10, 11, 0, 1],   # 7 + P,Q,R,S
    "8": [2, 3, 4, 5],        # 8 + T,U,V
    "9": [6, 7, 8, 9, 10],    # 9 + W,X,Y,Z
}


def build_letter_panel(key: str) -> list[str]:
    """Options shown after a key is confirmed."""
    if key == "0":
        return ["0", " ", "<BACK>"]
    letters = LETTER_MAP.get(key, [])
    return [key] + letters if letters else [key]


def pretty_item(x: str) -> str:
    if x == " ":
        return "<SPACE>"
    if x == "<BACK>":
        return "←BACK"
    return x


# -------------------------------------------------------------------
# API CALLS
# -------------------------------------------------------------------

def call_eeg(eeg_bytes: bytes) -> int | None:
    """Call EEG FastAPI /predict, using indices from session_state."""
    files = {"file": ("eeg.mat", BytesIO(eeg_bytes), "application/octet-stream")}
    data = {
        "block_idx": st.session_state.eeg_block_idx,
        "trial_idx": st.session_state.eeg_trial_idx,
        "target_idx": st.session_state.eeg_target_idx,
    }

    try:
        resp = requests.post(EEG_API_URL, files=files, data=data, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"EEG API error: {e}")
        return None

    out = resp.json()
    pred = out.get("prediction")
    if pred is None:
        st.error(f"EEG API response missing 'prediction': {out}")
        return None
    return int(pred)


def call_emg(emg_bytes: bytes) -> int | None:
    """Call EMG FastAPI /predict_file, using dataset string from session_state."""
    files = {"file": ("emg.hdf5", BytesIO(emg_bytes), "application/octet-stream")}
    data = {"dataset": st.session_state.emg_dataset}

    try:
        resp = requests.post(EMG_API_URL, files=files, data=data, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"EMG API error: {e}")
        return None

    out = resp.json()
    result = out.get("result")
    if result is None:
        st.error(f"EMG API response missing 'result': {out}")
        return None
    return int(result)


# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------

def init_state():
    defaults = {
        "typed_text": "",
        "phase": "idle",          # idle | key_eeg | key_emg | char_eeg | char_emg
        "current_key": None,
        "current_panel": None,    # list[str]
        "candidate_char": None,
        "message": "",
        "eeg_file_bytes": None,
        "emg_file_bytes": None,
        "eeg_block_idx": 0,
        "eeg_trial_idx": 0,
        "eeg_target_idx": 0,
        "emg_dataset": "0",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_char(msg: str = "Ready for next character."):
    st.session_state.phase = "idle"
    st.session_state.current_key = None
    st.session_state.current_panel = None
    st.session_state.candidate_char = None
    st.session_state.message = msg


def advance_eeg_target():
    """Move to the next target index for the next EEG call."""
    st.session_state.eeg_target_idx += 1


def advance_emg_dataset():
    """Increment dataset name if it is numeric (for stepping through EMG segments)."""
    try:
        d = int(st.session_state.emg_dataset)
        st.session_state.emg_dataset = str(d + 1)
    except ValueError:
        pass


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

def sidebar():
    st.sidebar.header("Inputs")

    eeg_file = st.sidebar.file_uploader("EEG .mat file", type=["mat"])
    if eeg_file is not None:
        st.session_state.eeg_file_bytes = eeg_file.read()

    emg_file = st.sidebar.file_uploader("EMG .hdf5 file", type=["hdf5"])
    if emg_file is not None:
        st.session_state.emg_file_bytes = emg_file.read()

    st.sidebar.subheader("EEG indices")
    st.session_state.eeg_block_idx = int(
        st.sidebar.number_input("block_idx", 0, value=st.session_state.eeg_block_idx)
    )
    st.session_state.eeg_trial_idx = int(
        st.sidebar.number_input("trial_idx", 0, value=st.session_state.eeg_trial_idx)
    )
    st.session_state.eeg_target_idx = int(
        st.sidebar.number_input("target_idx", 0, value=st.session_state.eeg_target_idx)
    )

    st.sidebar.subheader("EMG dataset")
    st.session_state.emg_dataset = st.sidebar.text_input(
        "dataset", value=st.session_state.emg_dataset
    )

    if st.sidebar.button("Reset indices"):
        st.session_state.eeg_block_idx = 0
        st.session_state.eeg_trial_idx = 0
        st.session_state.eeg_target_idx = 0
        st.session_state.emg_dataset = "0"
        rerun()


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="EEG+EMG Nokia Speller", layout="centered")
    init_state()
    sidebar()

    st.title("📱 EEG + EMG Nokia-style BCI Speller")
    st.caption("EEG selects keys & letters; EMG confirms YES/NO.")

    # Typed text
    st.markdown("### Typed output")
    st.text_area("Output", st.session_state.typed_text, height=80, disabled=True)

    st.write("---")
    st.write(f"**State:** `{st.session_state.phase}`")
    if st.session_state.message:
        st.info(st.session_state.message)

    # ---------------- IDLE ----------------
    if st.session_state.phase == "idle":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start next character"):
                if st.session_state.eeg_file_bytes is None:
                    st.error("Upload an EEG .mat file first.")
                else:
                    st.session_state.phase = "key_eeg"
                    st.session_state.message = "Running EEG to select a key..."
                    rerun()
        with col2:
            if st.button("Clear text"):
                st.session_state.typed_text = ""
                reset_char("Text cleared.")
                rerun()

    # ---------------- KEY EEG ----------------
    if st.session_state.phase == "key_eeg":
        if st.button("EEG → pick key"):
            if st.session_state.eeg_file_bytes is None:
                st.error("No EEG file uploaded.")
            else:
                pred = call_eeg(st.session_state.eeg_file_bytes)
                advance_eeg_target()
                if pred is None:
                    st.stop()
                if not 0 <= pred < len(KEYPAD_LABELS):
                    st.error(f"EEG key prediction out of range: {pred}")
                    st.stop()

                key = KEYPAD_LABELS[pred]
                st.session_state.current_key = key
                st.session_state.phase = "key_emg"
                st.session_state.message = f"EEG suggests key `{key}`. EMG to confirm?"
                rerun()

    if st.session_state.current_key:
        st.markdown(f"**Key candidate:** `{st.session_state.current_key}`")

    # ---------------- KEY EMG ----------------
    if st.session_state.phase == "key_emg":
        col1, col2 = st.columns(2)

        with col1:
            if st.button("EMG → confirm key"):
                if st.session_state.emg_file_bytes is None:
                    st.error("No EMG file uploaded.")
                else:
                    res = call_emg(st.session_state.emg_file_bytes)
                    advance_emg_dataset()
                    if res is None:
                        st.stop()

                    key = st.session_state.current_key
                    if res == 1:  # YES to key
                        letters = LETTER_MAP.get(key, [])
                        # If key has no letters (except 0) we just append it
                        if (not letters and key != "0") or key in ["1", "*", "#"]:
                            st.session_state.typed_text += key
                            reset_char(f"Key `{key}` confirmed.")
                        else:
                            st.session_state.current_panel = build_letter_panel(key)
                            st.session_state.phase = "char_eeg"
                            opts = ", ".join(
                                pretty_item(x) for x in st.session_state.current_panel
                            )
                            st.session_state.message = (
                                f"Key `{key}` confirmed. EEG now selects from: {opts}"
                            )
                    else:        # NO to key
                        reset_char("Key rejected. Start again.")
                    rerun()

        with col2:
            if st.button("Reject key"):
                reset_char("Key rejected.")
                rerun()

    # ---------------- PANEL DISPLAY ----------------
    if st.session_state.current_panel:
        st.markdown("**Letter panel:**")
        st.write(", ".join(pretty_item(x) for x in st.session_state.current_panel))

    # ---------------- CHAR EEG ----------------
    if st.session_state.phase == "char_eeg":
        if st.button("EEG → pick letter"):
            if st.session_state.eeg_file_bytes is None:
                st.error("No EEG file uploaded.")
                st.stop()

            panel = st.session_state.current_panel
            key = st.session_state.current_key
            if not panel or not key:
                st.error("Panel or key missing.")
                st.stop()

            pred = call_eeg(st.session_state.eeg_file_bytes)
            advance_eeg_target()
            if pred is None:
                st.stop()

            active_ids = LETTER_CLASS_IDS.get(key, [])[: len(panel)]
            if pred not in active_ids:
                st.error(f"EEG predicted {pred}, not in active IDs {active_ids}.")
                st.stop()

            idx = active_ids.index(pred)
            candidate = panel[idx]
            st.session_state.candidate_char = candidate
            st.session_state.phase = "char_emg"
            st.session_state.message = (
                f"EEG suggests `{pretty_item(candidate)}`. EMG to confirm?"
            )
            rerun()

    if st.session_state.candidate_char:
        st.markdown(
            f"**Character candidate:** `{pretty_item(st.session_state.candidate_char)}`"
        )

    # ---------------- CHAR EMG ----------------
    if st.session_state.phase == "char_emg":
        col1, col2 = st.columns(2)

        with col1:
            if st.button("EMG → confirm letter"):
                if st.session_state.emg_file_bytes is None:
                    st.error("No EMG file uploaded.")
                else:
                    res = call_emg(st.session_state.emg_file_bytes)
                    advance_emg_dataset()
                    if res is None:
                        st.stop()

                    cand = st.session_state.candidate_char
                    if res == 1:  # YES
                        if cand == "<BACK>":
                            st.session_state.typed_text = st.session_state.typed_text[:-1]
                        else:
                            st.session_state.typed_text += cand
                        reset_char(f"Character `{pretty_item(cand)}` confirmed.")
                    else:         # NO
                        st.session_state.candidate_char = None
                        st.session_state.phase = "char_eeg"
                        st.session_state.message = (
                            "Character rejected. EEG will pick another."
                        )
                    rerun()

        with col2:
            if st.button("Reject letter"):
                st.session_state.candidate_char = None
                st.session_state.phase = "char_eeg"
                st.session_state.message = "Character rejected. Try again."
                rerun()


if __name__ == "__main__":
    main()
