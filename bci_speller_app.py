# bci_speller_app.py
#
# Streamlit UI for EEG+EMG Nokia-style speller.
#
# - EEG FastAPI (12-class) at EEG_API_URL: /predict
#   returns { "prediction": 0..11 }
# - EMG FastAPI (binary) at EMG_API_URL: /predict_file
#   returns { "result": 0 or 1 }  (0=NO, 1=YES)
#
# Main state machine phases:
#   idle      → waiting to start a character
#   key_eeg   → EEG picks keypad key
#   key_emg   → EMG confirms/rejects key
#   char_eeg  → EEG picks letter/space/back within the key’s panel
#   char_emg  → EMG confirms/rejects that character

import streamlit as st
import requests
from io import BytesIO

# ============================================================
# CONFIG
# ============================================================

EEG_API_URL = "http://localhost:8000/predict"
EMG_API_URL = "http://localhost:8001/predict_file"

# Wrapper to support both new and older Streamlit versions
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ============================================================
# KEYPAD + LETTER MAPPING
# ============================================================

KEYPAD_LABELS = [
    "1", "2", "3",
    "4", "5", "6",
    "7", "8", "9",
    "*", "0", "#"
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
    "0": [" "],      # space handled specially in the panel
    "*": [],
    "#": [],
}

# For letter panels we restrict to specific 12-class IDs used during
# stimulation (same idea as your PsychoPy LETTER_CLASS_IDS).
#
# These are the global 12-class indices (0..11) that are active in each panel.
LETTER_CLASS_IDS = {
    "1": [0],                 # no letters, but included for completeness
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


# ============================================================
# PANEL / SELECTION HELPERS
# ============================================================

def build_letter_panel(key: str) -> list[str]:
    """
    Return the panel items for a given key.
    Must align in order with the LETTER_CLASS_IDS[key] subset we use.
    """
    if key == "0":
        # [0, " ", "<BACK>"]
        return ["0", " ", "<BACK>"]

    letters = LETTER_MAP.get(key, [])
    if letters:
        return [key] + letters

    # keys with no letters (1, *, #)
    return [key]


def pretty_item(x: str) -> str:
    if x == " ":
        return "<SPACE>"
    if x == "<BACK>":
        return "←BACK"
    return x


# ============================================================
# EEG / EMG API CALLS
# ============================================================

def call_eeg_api(eeg_bytes: bytes, block_idx: int, trial_idx: int, target_idx: int) -> int | None:
    """
    Call EEG FastAPI: /predict
    Expects JSON with key 'prediction' ∈ {0..11}.
    """
    files = {
        "file": ("eeg.mat", BytesIO(eeg_bytes), "application/octet-stream")
    }
    data = {
        "block_idx": block_idx,
        "trial_idx": trial_idx,
        "target_idx": target_idx,
    }

    try:
        resp = requests.post(EEG_API_URL, files=files, data=data, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"EEG API error: {e}")
        return None

    out = resp.json()
    pred = out.get("prediction", None)
    if pred is None:
        st.error(f"EEG API response missing 'prediction': {out}")
        return None
    return int(pred)


def call_emg_api(emg_bytes: bytes, dataset_name: str = "0") -> int | None:
    """
    Call EMG FastAPI: /predict_file
    Expects JSON with 'result' ∈ {0,1}.
    """
    files = {
        "file": ("emg.hdf5", BytesIO(emg_bytes), "application/octet-stream")
    }
    data = {"dataset": dataset_name}

    try:
        resp = requests.post(EMG_API_URL, files=files, data=data, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"EMG API error: {e}")
        return None

    out = resp.json()
    result = out.get("result", None)
    if result is None:
        st.error(f"EMG API response missing 'result': {out}")
        return None
    return int(result)


# ============================================================
# STATE MANAGEMENT
# ============================================================

def init_state():
    defaults = {
        "typed_text": "",
        "phase": "idle",              # idle | key_eeg | key_emg | char_eeg | char_emg
        "current_key": None,
        "current_panel_items": None,  # list[str]
        "current_candidate_char": None,
        "last_message": "",
        "eeg_file_bytes": None,
        "emg_file_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_for_next_char():
    st.session_state.phase = "idle"
    st.session_state.current_key = None
    st.session_state.current_panel_items = None
    st.session_state.current_candidate_char = None
    st.session_state.last_message = "Ready for next character."


# ============================================================
# UI HELPERS
# ============================================================

def show_typed_text():
    st.markdown("### Typed output")
    st.text_area("Output", st.session_state.typed_text, height=80, disabled=True)


def sidebar_inputs():
    st.sidebar.header("EEG / EMG inputs")

    eeg_file = st.sidebar.file_uploader("EEG .mat file", type=["mat"])
    if eeg_file is not None:
        st.session_state.eeg_file_bytes = eeg_file.read()

    emg_file = st.sidebar.file_uploader("EMG .hdf5 file", type=["hdf5"])
    if emg_file is not None:
        st.session_state.emg_file_bytes = emg_file.read()

    st.sidebar.markdown("**EEG epoch indices** (demo/testing)")
    block_idx = st.sidebar.number_input("block_idx", min_value=0, value=0, step=1)
    trial_idx = st.sidebar.number_input("trial_idx", min_value=0, value=0, step=1)
    target_idx = st.sidebar.number_input("target_idx", min_value=0, value=0, step=1)

    dataset_name = st.sidebar.text_input("EMG dataset name", value="0")

    return {
        "block_idx": int(block_idx),
        "trial_idx": int(trial_idx),
        "target_idx": int(target_idx),
        "dataset": dataset_name,
    }


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="EEG+EMG Nokia Speller", layout="centered")
    init_state()

    st.title("📱 EEG + EMG Nokia-style BCI Speller")
    st.caption("EEG selects keys & letters; EMG confirms YES/NO.")

    params = sidebar_inputs()

    show_typed_text()
    st.write("---")

    st.write(f"**State:** `{st.session_state.phase}`")
    if st.session_state.last_message:
        st.info(st.session_state.last_message)

    # ---------------------- IDLE ----------------------
    if st.session_state.phase == "idle":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start selecting next character (EEG key)"):
                if st.session_state.eeg_file_bytes is None:
                    st.error("Upload an EEG .mat file first (sidebar).")
                else:
                    st.session_state.phase = "key_eeg"
                    st.session_state.last_message = "Running EEG model to select a keypad key..."
                    rerun()
        with col2:
            if st.button("Clear all text"):
                st.session_state.typed_text = ""
                reset_for_next_char()
                rerun()

    # ---------------------- KEY EEG ----------------------
    if st.session_state.phase == "key_eeg":
        if st.button("Run EEG → predict key"):
            if st.session_state.eeg_file_bytes is None:
                st.error("No EEG file uploaded.")
            else:
                pred_class = call_eeg_api(
                    st.session_state.eeg_file_bytes,
                    params["block_idx"],
                    params["trial_idx"],
                    params["target_idx"],
                )
                if pred_class is None:
                    st.stop()

                if pred_class < 0 or pred_class >= len(KEYPAD_LABELS):
                    st.error(f"EEG key prediction out of range: {pred_class}")
                    st.stop()

                key = KEYPAD_LABELS[pred_class]
                st.session_state.current_key = key
                st.session_state.phase = "key_emg"
                st.session_state.last_message = f"EEG suggests key `{key}`. Use EMG to confirm?"
                rerun()

    if st.session_state.current_key is not None:
        st.markdown(f"**Current key candidate:** `{st.session_state.current_key}`")

    # ---------------------- KEY EMG ----------------------
    if st.session_state.phase == "key_emg":
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Run EMG confirm (key)"):
                if st.session_state.emg_file_bytes is None:
                    st.error("Upload an EMG .hdf5 file first (sidebar).")
                else:
                    emg_result = call_emg_api(
                        st.session_state.emg_file_bytes,
                        dataset_name=params["dataset"],
                    )
                    if emg_result is None:
                        st.stop()

                    key = st.session_state.current_key

                    if emg_result == 1:
                        # YES to key
                        st.session_state.last_message = f"EMG confirmed key `{key}`."

                        letters = LETTER_MAP.get(key, [])
                        # If no letters and not 0 → just append key directly
                        if (not letters and key != "0") or key in ["1", "*", "#"]:
                            st.session_state.typed_text += key
                            reset_for_next_char()
                        else:
                            # Need a letter panel (key has letters or is 0)
                            panel = build_letter_panel(key)
                            st.session_state.current_panel_items = panel
                            st.session_state.phase = "char_eeg"
                            st.session_state.last_message = (
                                f"Key `{key}` confirmed. EEG now selects from: "
                                + ", ".join(pretty_item(x) for x in panel)
                            )
                    else:
                        # NO to key → restart character
                        st.session_state.last_message = "EMG rejected key. Restarting key selection."
                        reset_for_next_char()

                    rerun()

        with col2:
            if st.button("Reject & restart character"):
                reset_for_next_char()
                rerun()

        with col3:
            if st.button("Cancel"):
                reset_for_next_char()
                rerun()

    # Show letter panel if present
    if st.session_state.current_panel_items is not None:
        st.markdown("**Letter panel options:**")
        st.write(", ".join(pretty_item(x) for x in st.session_state.current_panel_items))

    # ---------------------- CHAR EEG ----------------------
    if st.session_state.phase == "char_eeg":
        if st.button("Run EEG → predict letter/option"):
            if st.session_state.eeg_file_bytes is None:
                st.error("No EEG file uploaded.")
                st.stop()

            panel = st.session_state.current_panel_items
            key = st.session_state.current_key

            if panel is None or key is None:
                st.error("Letter panel or key missing.")
                st.stop()

            pred_class = call_eeg_api(
                st.session_state.eeg_file_bytes,
                params["block_idx"],
                params["trial_idx"],
                params["target_idx"],
            )
            if pred_class is None:
                st.stop()

            allowed_global_ids = LETTER_CLASS_IDS.get(key, [])
            if not allowed_global_ids:
                st.error(f"No LETTER_CLASS_IDS defined for key {key}.")
                st.stop()

            # Only consider the first len(panel) IDs as active items
            active_ids = allowed_global_ids[:len(panel)]
            if pred_class not in active_ids:
                st.error(
                    f"EEG predicted global class {pred_class}, "
                    f"which is not active in this panel {active_ids}."
                )
                st.stop()

            idx = active_ids.index(pred_class)
            candidate = panel[idx]

            st.session_state.current_candidate_char = candidate
            st.session_state.phase = "char_emg"
            st.session_state.last_message = (
                f"EEG suggests `{pretty_item(candidate)}`. Use EMG to confirm?"
            )
            rerun()

    if st.session_state.current_candidate_char is not None:
        st.markdown(
            f"**Character candidate:** `{pretty_item(st.session_state.current_candidate_char)}`"
        )

    # ---------------------- CHAR EMG ----------------------
    if st.session_state.phase == "char_emg":
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Run EMG confirm (letter)"):
                if st.session_state.emg_file_bytes is None:
                    st.error("Upload an EMG .hdf5 file first (sidebar).")
                else:
                    emg_result = call_emg_api(
                        st.session_state.emg_file_bytes,
                        dataset_name=params["dataset"],
                    )
                    if emg_result is None:
                        st.stop()

                    candidate = st.session_state.current_candidate_char

                    if emg_result == 1:
                        # YES to candidate
                        if candidate == "<BACK>":
                            st.session_state.typed_text = st.session_state.typed_text[:-1]
                        else:
                            st.session_state.typed_text += candidate

                        st.session_state.last_message = (
                            f"EMG confirmed `{pretty_item(candidate)}`."
                        )
                        reset_for_next_char()
                    else:
                        # NO to candidate → re-run char EEG within same key
                        st.session_state.current_candidate_char = None
                        st.session_state.phase = "char_eeg"
                        st.session_state.last_message = (
                            "EMG rejected character. Re-running EEG for this key."
                        )
                    rerun()

        with col2:
            if st.button("Cancel this character"):
                reset_for_next_char()
                rerun()

        with col3:
            if st.button("Clear all text & restart"):
                st.session_state.typed_text = ""
                reset_for_next_char()
                rerun()


if __name__ == "__main__":
    main()
