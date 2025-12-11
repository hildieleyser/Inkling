from psychopy import visual, core, event
import numpy as np
import os
import io
import tempfile
import requests  # for HTTP calls
import h5py
from scipy.io import savemat
import requests  # for HTTP calls

###########################################
# CONFIG
###########################################
DEBUG_KEYBOARD = False          # keyboard-only control
USE_SIM_DATA = True             # drive EEG/EMG from simulated data (.npz)
USE_REAL_MODELS_ON_SIM = False   # if True: NPZ -> FastAPI -> EEGNet/EMG model

DEBUG_KEYBOARD = True       # keyboard-only control
USE_SIM_DATA = False        # drive EEG/EMG from simulated data when not in DEBUG
SIM_DATA_PATH = "eeg_emg_helloworld.npz"
EMG_BURST_THRESH = 0.5      # envelope/percentile cutoff for sim EMG confirm
STIM_TIME_KEY = 5.0         # seconds of flicker before querying main keypad EEG
STIM_TIME_LETTER = 5.0      # seconds of flicker before querying letter EEG
FONT_NAME = "Helvetica"

# Debug overlay: show real flicker frequencies above each box
SHOW_DEBUG_OVERLAY_DEFAULT = True   # initial state
_debug_overlay_on = SHOW_DEBUG_OVERLAY_DEFAULT

# FastAPI servers
EEG_API_URL = "http://127.0.0.1:8000/predict"        # from api_eeg/fast.py
EMG_API_URL = "http://127.0.0.1:8001/predict_file"   # from api_emg

# File lists for offline testing (YOU fill these)
EEG_FILES = [
    r"/Users/rayanhasan/code/hildieleyser/Inkling/S001.mat",  # example
]
EMG_FILES = [
    r"/Users/rayanhasan/code/hildieleyser/Inkling/EMG Data Participant 1 Day 1 Block 1.hdf5",
]

_EEG_FILE_IDX = 0
_EMG_FILE_IDX = 0
_EEG_TARGET_IDX = 0

# --- Universal Box Size (SAME AS 12-TARGET KEYPAD) ---
BOX_W = 0.42
BOX_H = 0.30

# =====================================================
# TRUE SSVEP FREQUENCIES & PHASES (FROM PAPER TABLE)
# Order matches KEYPAD_LABELS: 1,2,3,4,5,6,7,8,9,*,0,#
# =====================================================
FREQS_MAIN = [
    9.25, 11.25, 13.25,   # 1,2,3  (phase 0·π)
    9.75, 11.75, 13.75,   # 4,5,6  (phase 0.5·π)
    10.25, 12.25, 14.25,  # 7,8,9  (phase 1·π)
    10.75, 12.75, 14.75   # *,0,#  (phase 1.5·π)
]

PHASES_MAIN = [
    0.0,           0.0,           0.0,           # 1,2,3
    0.5 * np.pi,   0.5 * np.pi,   0.5 * np.pi,   # 4,5,6
    1.0 * np.pi,   1.0 * np.pi,   1.0 * np.pi,   # 7,8,9
    1.5 * np.pi,   1.5 * np.pi,   1.5 * np.pi    # *,0,#
]

# --- Keypad labels (row-wise layout) ---
KEYPAD_LABELS = [
    "1","2","3",
    "4","5","6",
    "7","8","9",
    "*","0","#"
]

LETTER_MAP = {
    "2": ["A","B","C"],
    "3": ["D","E","F"],
    "4": ["G","H","I"],
    "5": ["J","K","L"],
    "6": ["M","N","O"],
    "7": ["P","Q","R","S"],
    "8": ["T","U","V"],
    "9": ["W","X","Y","Z"]
}

# Which 12-class indices are used for each letter panel
# (indices refer to FREQS_MAIN / PHASES_MAIN)
LETTER_CLASS_IDS = {
    "0": [0, 1, 2],            # 0, space, back   (we'll map meaning in UI)
    "2": [0, 1, 2, 3],         # key + A,B,C
    "3": [4, 5, 6, 7],         # key + D,E,F
    "4": [8, 9, 10, 11],       # key + G,H,I
    "5": [0, 1, 2, 4],         # key + J,K,L
    "6": [5, 6, 7, 8],         # key + M,N,O
    "7": [9, 10, 11, 0, 1],    # key + P,Q,R,S
    "8": [2, 3, 4, 5],         # key + T,U,V
    "9": [6, 7, 8, 9, 10],     # key + W,X,Y,Z
}

# EEG confidence thresholds
EEG_KEY_THRESH = 0.8
EEG_LETTER_THRESH = 0.8

# Simulated data state
_SIM_DATA = None
_SIM_EEG_IDX = 0
_SIM_EMG_IDX = 0

###########################################
# WINDOW
###########################################
win = visual.Window(
    size=[0, 0],
    fullscr=True,
    color=[0.2, 0.2, 0.2],
    units="norm"
)

###########################################
# SIM HELPERS
###########################################
def _next_eeg_file():
    global _EEG_FILE_IDX
    if not EEG_FILES:
        raise RuntimeError("EEG_FILES list is empty. Add your .mat paths.")
    path = EEG_FILES[_EEG_FILE_IDX % len(EEG_FILES)]
    _EEG_FILE_IDX += 1
    return path

def _next_emg_file():
    global _EMG_FILE_IDX
    if not EMG_FILES:
        raise RuntimeError("EMG_FILES list is empty. Add your .hdf5 paths.")
    path = EMG_FILES[_EMG_FILE_IDX % len(EMG_FILES)]
    _EMG_FILE_IDX += 1
    return path

def _next_eeg_params():
    global _EEG_TARGET_IDX
    params = {
        "block_idx": 0,
        "trial_idx": 0,
        "target_idx": _EEG_TARGET_IDX % 12,
    }
    _EEG_TARGET_IDX += 1
    return params

def _load_sim_data():
    global _SIM_DATA
    if _SIM_DATA is not None:
        return _SIM_DATA
    candidates = [
        SIM_DATA_PATH,
        os.path.join(os.path.dirname(__file__), SIM_DATA_PATH),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path)
            _SIM_DATA = {k: data[k] for k in data.files}
            return _SIM_DATA
    raise FileNotFoundError(f"Simulated data not found. Tried: {candidates}")

def _next_sim_eeg():
    global _SIM_EEG_IDX
    data = _load_sim_data()
    n = _SIM_DATA["labels"].shape[0]
    idx = _SIM_EEG_IDX % n
    _SIM_EEG_IDX += 1
    return _SIM_DATA["eeg"][idx], int(_SIM_DATA["labels"][idx]), float(_SIM_DATA["fs_eeg"])

def _next_sim_emg():
    global _SIM_EMG_IDX
    data = _load_sim_data()
    n = _SIM_DATA["emg"].shape[0]
    idx = _SIM_EMG_IDX % n
    _SIM_EMG_IDX += 1
    emg_label = _SIM_DATA.get("emg_labels", None)
    label = bool(emg_label[idx]) if emg_label is not None else None
    return _SIM_DATA["emg"][idx], float(_SIM_DATA["fs_emg"]), label

def _freq_logits_from_signal(sig, fs):
    f = np.fft.rfft(sig)
    hz = np.fft.rfftfreq(len(sig), 1 / fs)
    mags = []
    for target in FREQS_MAIN:
        bin_idx = np.argmin(np.abs(hz - target))
        mags.append(np.abs(f[bin_idx]))
    mags = np.array(mags)
    return np.log(mags + 1e-6)

###########################################
# EEG / EMG PREDICTION HELPERS
###########################################
def _get_eeg_logits_from_sim_via_model():
    """
    NPZ → (8 x time) epoch -> in-memory .mat → EEG FastAPI → EEGNet logits.
    """
    sig, true_label, fs = _next_sim_eeg()
    sig = np.asarray(sig, dtype=np.float32)

    # Tile into 8 channels to satisfy EEGNet input (8, time)
    N_CHANS = 8
    epoch_raw = np.tile(sig, (N_CHANS, 1))

    # Save to in-memory .mat with key "epoch"
    mat_buf = io.BytesIO()
    savemat(mat_buf, {"epoch": epoch_raw})
    mat_buf.seek(0)

    files = {
        "file": ("sim_epoch.mat", mat_buf, "application/octet-stream")
    }
    params = {"block_idx": 0, "trial_idx": 0, "target_idx": 0}

    try:
        resp = requests.post(EEG_API_URL, files=files, params=params, timeout=10.0)
        resp.raise_for_status()
        out = resp.json()
        pred_class = int(out["prediction"])   # 0..11
        print(f"[SIM-API-EEG] model_pred={pred_class} (true={true_label})")
    except Exception as e:
        print(f"[SIM-API-EEG ERROR] {e}")
        # Fallback to classic FFT decoding using true_label hint
        logits = _freq_logits_from_signal(sig, fs)
        logits[true_label] += 1.0
        return logits

    logits = np.full(12, -5.0, dtype=float)
    logits[pred_class] = 5.0
    return logits

def get_eeg_logits():
    """
    Master EEG dispatch:
      - DEBUG keyboard (only when no sim)
      - SIM + REAL MODEL via API
      - SIM + classic FFT
      - Real .mat via API
    """
    # Keyboard-only mode (for quick testing without sim)
    if DEBUG_KEYBOARD and not USE_SIM_DATA:
        return np.zeros(12, dtype=float)

    # SIMULATION PATHS
    if USE_SIM_DATA:
        if USE_REAL_MODELS_ON_SIM:
            return _get_eeg_logits_from_sim_via_model()
        else:
            # Old classic sim: FFT-based decoder
            sig, true_label, fs = _next_sim_eeg()
            logits = _freq_logits_from_signal(sig, fs)
            logits[true_label] += 1.0
            print(f"[SIM-CLASSIC-EEG] true={true_label}")
            return logits

    # REAL DATA via FastAPI
    mat_path = _next_eeg_file()
    try:
        with open(mat_path, "rb") as f:
            files = {"file": (os.path.basename(mat_path), f, "application/octet-stream")}
            params = _next_eeg_params()
            resp = requests.post(EEG_API_URL, files=files, params=params, timeout=10.0)

        resp.raise_for_status()
        data = resp.json()
        pred_class = int(data["prediction"])   # 0..11
        print(f"[REAL-EEG] file={mat_path} params={params} -> pred {pred_class}")

    except Exception as e:
        print(f"[REAL-EEG API error for file {mat_path}: {e}")
        return np.zeros(12, dtype=float)

    logits = np.full(12, -5.0, dtype=float)
    logits[pred_class] = 5.0
    return logits

def _eeg_predict_from_subset(active_ids, thresh=None):
    logits = np.array(get_eeg_logits())   # (12,)
    logits = logits[active_ids]           # (K,)

    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()

    k = int(np.argmax(probs))
    conf = float(probs[k])

    if (thresh is not None) and (conf < thresh):
        print(f"[EEG] low confidence {conf:.2f}, returning None")
        return None
    return k

def eeg_predict_key():
    # Keyboard override only when not using sim
    use_keyboard = DEBUG_KEYBOARD and not USE_SIM_DATA
    if use_keyboard:
        keys = event.getKeys()
        for k in keys:
            if k in ["1","2","3","4","5","6","7","8","9","0"]:
                return KEYPAD_LABELS.index(k)
            if k == "num_multiply":
                return KEYPAD_LABELS.index("*")
            if k == "num_add":
                return KEYPAD_LABELS.index("#")
        return None
    active_ids = list(range(12))
    return _eeg_predict_from_subset(active_ids, thresh=EEG_KEY_THRESH)

def eeg_predict_letter(key, n_items):
    use_keyboard = DEBUG_KEYBOARD and not USE_SIM_DATA
    if use_keyboard:
        mapping = ["a","b","c","d","e"]
        keys = event.getKeys()
        for i, k in enumerate(mapping[:n_items]):
            if k in keys:
                return i
        return None

    active_ids = LETTER_CLASS_IDS[key][:n_items]
    return _eeg_predict_from_subset(active_ids, thresh=EEG_LETTER_THRESH)

def emg_confirm():
    """
    EMG decision:
      - keyboard fallback (y/n) when DEBUG and not sim
      - SIM + real EMG model via FastAPI (USE_REAL_MODELS_ON_SIM)
      - SIM + envelope threshold (classic)
      - Real file via EMG FastAPI
    """
    # Keyboard override
    if DEBUG_KEYBOARD and not USE_SIM_DATA:
        keys = event.getKeys()
        if "y" in keys:
            return True
        if "n" in keys:
            return False
        return None

    # SIM + real EMG model
    if USE_SIM_DATA and USE_REAL_MODELS_ON_SIM:
        sig, fs, true_label = _next_sim_emg()
        tmp_path = None
        try:
            # Create temporary HDF5 with dataset "0"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
                tmp_path = tmp.name

            with h5py.File(tmp_path, "w") as hf:
                hf.create_dataset("0", data=sig.astype(np.float32))

            with open(tmp_path, "rb") as f:
                files = {"file": ("sim_emg.hdf5", f, "application/octet-stream")}
                resp = requests.post(
                    EMG_API_URL,
                    files=files,
                    data={"dataset": "0"},
                    timeout=10.0
                )
            resp.raise_for_status()
            out = resp.json()
            pred = int(out["result"])
            print(f"[SIM-API-EMG] model_pred={pred} true={true_label}")
            return bool(pred)
        except Exception as e:
            print(f"[SIM-API-EMG ERROR] {e}")
            # Fallback to classic decision
            env = np.percentile(np.abs(sig), 95)
            decision = env > EMG_BURST_THRESH
            return decision
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # SIM + classic envelope threshold
    if USE_SIM_DATA and not USE_REAL_MODELS_ON_SIM:
        sig, fs, true_label = _next_sim_emg()
        env = np.percentile(np.abs(sig), 95)
        decision = env > EMG_BURST_THRESH
        print(f"[SIM-CLASSIC-EMG] decision={decision} true={true_label}")
        return decision

    # REAL EMG FILE via API
    h5_path = _next_emg_file()
    try:
        with open(h5_path, "rb") as f:
            files = {"file": (os.path.basename(h5_path), f, "application/octet-stream")}
            resp = requests.post(
                EMG_API_URL,
                files=files,
                data={"dataset": "0"},
                timeout=10.0
            )
        resp.raise_for_status()
        out = resp.json()
        result = int(out["result"])
        print(f"[REAL-EMG] file={h5_path} -> {result}")
        return bool(result)
    except Exception as e:
        print(f"[REAL-EMG API error: {e}]")
        return None

###########################################
# UI Helper
###########################################
def draw_text_buffer(buffer_text):
    visual.TextStim(
        win, text=buffer_text, pos=(0,0.85),
        height=0.06, color="white", font=FONT_NAME
    ).draw()

###########################################
# MAIN KEYPAD
###########################################
def build_main_keypad():
    rows, cols = 4, 3
    x_positions = np.linspace(-0.75, 0.75, cols)
    y_positions = np.linspace(0.55, -0.55, rows)

    targets, labels, freq_overlays = [], [], []
    idx = 0

    for r in range(rows):
        for c in range(cols):
            pos = (x_positions[c], y_positions[r])
            key = KEYPAD_LABELS[idx]

            box = visual.Rect(
                win,
                width=BOX_W, height=BOX_H,
                fillColor=[0,0,0],
                lineColor=[1,1,1],
                pos=pos, lineWidth=2
            )

            txt = key if key not in LETTER_MAP else key + "\n" + "".join(LETTER_MAP[key])

            label = visual.TextStim(
                win,
                text=txt,
                pos=pos,
                color="white",
                height=0.10,
                alignText="center",
                font=FONT_NAME
            )

            # Debug frequency overlay (slightly above box)
            freq_label = visual.TextStim(
                win,
                text=f"{FREQS_MAIN[idx]:.2f} Hz",
                pos=(pos[0], pos[1] + BOX_H/2 + 0.05),
                color="yellow",
                height=0.04,
                font=FONT_NAME
            )

            targets.append(box)
            labels.append(label)
            freq_overlays.append(freq_label)
            idx += 1

    return targets, labels, freq_overlays

def _handle_debug_toggle():
    """Toggle overlay with 'd' key globally."""
    global _debug_overlay_on
    keys = event.getKeys()
    if "d" in keys:
        _debug_overlay_on = not _debug_overlay_on

def run_main_keypad(buffer_text):
    targets, labels, freq_overlays = build_main_keypad()
    clock = core.Clock()

    while True:
        t = clock.getTime()

        # allow toggling overlay
        _handle_debug_toggle()

        # hard escape
        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)

        for i in range(12):
            lum = np.clip(
                0.4 + 0.5 * np.sin(2*np.pi*FREQS_MAIN[i]*t + PHASES_MAIN[i]),
                0, 1
            )
            targets[i].fillColor = [lum,lum,lum]
            targets[i].draw()
            labels[i].draw()
            if _debug_overlay_on:
                freq_overlays[i].draw()

        win.flip()

        if t > STIM_TIME_KEY:
            idx = eeg_predict_key()
            if idx is not None:
                return KEYPAD_LABELS[idx]

###########################################
# EMG CONFIRMATION (KEY)
###########################################
def run_emg_confirm_key(key, buffer_text):
    mode_str = (
        "SIM + API models" if (USE_SIM_DATA and USE_REAL_MODELS_ON_SIM)
        else "SIM classic" if USE_SIM_DATA
        else "REAL data"
    )

    msg = visual.TextStim(
        win,
        text=(
            f"Selected key:\n\n{key}\n\n"
            f"EMG gesture to CONFIRM.\n\n"
            f"Mode: {mode_str}\n"
            "Debug: press 'd' to toggle freq overlay."
        ),
        height=0.06,
        color="white",
        font=FONT_NAME,
        wrapWidth=1.6
    )

    timer = core.Clock()
    while timer.getTime() < 2.0:
        _handle_debug_toggle()
        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

    while True:
        _handle_debug_toggle()
        draw_text_buffer(buffer_text); msg.draw(); win.flip()
        d = emg_confirm()
        if d is not None:
            return d

###########################################
# LETTER PANEL
###########################################
def run_letter_panel(key, buffer_text):

    # Build items
    if key == "0":
        panel_items = ["0", " ", "<BACK"]
    else:
        panel_items = [key] + LETTER_MAP[key]

    n = len(panel_items)
    class_ids = LETTER_CLASS_IDS[key][:n]

    # Layout rules
    if key == "0":  # 3 items → one row
        positions = [(-0.6,0.0),(0.0,0.0),(0.6,0.0)]
    else:
        if n == 5:  # number top, 4 letters in 2×2
            positions = [
                (0,0.55),
                (-0.45,0.05),(0.45,0.05),
                (-0.45,-0.35),(0.45,-0.35)
            ]
        elif n == 4:  # number top + 3 letters
            positions = [
                (0,0.55),
                (-0.55,-0.05),(0,-0.05),(0.55,-0.05)
            ]
        elif n == 3:  # number top + 2 letters
            positions = [
                (0,0.55),
                (-0.45,-0.05),(0.45,-0.05)
            ]
        else:
            raise ValueError("Unexpected panel size.")

    boxes, texts, freq_overlays = [], [], []
    for i, item in enumerate(panel_items):
        pos = positions[i]
        lbl = "<SPACE>" if item == " " else ("←" if item=="<BACK" else item)

        box = visual.Rect(
            win,width=BOX_W,height=BOX_H,
            fillColor=[0,0,0],lineColor=[1,1,1],
            pos=pos,lineWidth=2
        )

        txt = visual.TextStim(
            win,text=lbl,pos=pos,
            color="white",height=0.12,font=FONT_NAME
        )

        cls_id = class_ids[i]
        freq_lbl = visual.TextStim(
            win,
            text=f"{FREQS_MAIN[cls_id]:.2f} Hz",
            pos=(pos[0], pos[1] + BOX_H/2 + 0.05),
            color="yellow",
            height=0.04,
            font=FONT_NAME
        )

        boxes.append(box)
        texts.append(txt)
        freq_overlays.append(freq_lbl)

    instr = visual.TextStim(
        win,
        text=f"Select item for key {key}\n(press 'd' to toggle freq overlay)",
        pos=(0, 0.85),
        color="white",
        height=0.05,
        font=FONT_NAME
    )

    clock = core.Clock()
    while True:
        t = clock.getTime()

        _handle_debug_toggle()

        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)
        instr.draw()

        for i in range(n):
            cls_id = class_ids[i]
            freq = FREQS_MAIN[cls_id]
            phase = PHASES_MAIN[cls_id]
            lum = np.clip(0.4 + 0.5*np.sin(2*np.pi*freq*t + phase),0,1)
            boxes[i].fillColor = [lum,lum,lum]
            boxes[i].draw()
            texts[i].draw()
            if _debug_overlay_on:
                freq_overlays[i].draw()

        win.flip()

        if t > STIM_TIME_LETTER:
            idx = eeg_predict_letter(key, n_items=n)
            if idx is not None:
                return panel_items[idx]

###########################################
# EMG CONFIRMATION (LETTER)
###########################################
def run_emg_confirm_letter(letter, buffer_text):
    msg = visual.TextStim(
        win,
        text=f"Confirm:\n\n{letter}\n\n(press 'd' to toggle freq overlay)",
        height=0.06,
        color="white",
        font=FONT_NAME,
        wrapWidth=1.6
    )

    timer = core.Clock()
    while timer.getTime() < 2.0:
        _handle_debug_toggle()
        draw_text_buffer(buffer_text); msg.draw(); win.flip()

    while True:
        _handle_debug_toggle()
        draw_text_buffer(buffer_text); msg.draw(); win.flip()
        d = emg_confirm()
        if d is not None:
            return d

###########################################
# MAIN LOOP
###########################################
def main():
    global _debug_overlay_on
    buffer_text = ""

    mode_desc = (
        "SIM data → FastAPI models (EEGNet + EMG CNN)"
        if (USE_SIM_DATA and USE_REAL_MODELS_ON_SIM)
        else "SIM data → classic FFT/envelope"
        if USE_SIM_DATA
        else "REAL .mat/.hdf5 data via APIs"
    )

    intro = visual.TextStim(
        win,
        text=(
            "EEG-EMG Hybrid Speller\n\n"
            "Real SSVEP frequencies (9.25–14.75 Hz) & phases from paper.\n"
            "EEG: look at a flickering key.\n"
            "EMG: gesture to confirm.\n\n"
            f"Current mode: {mode_desc}\n\n"
            "Controls:\n"
            "  - SPACE: start\n"
            "  - d: toggle frequency overlay\n"
            "  - ESC: quit\n"
        ),
        height=0.05,
        color="white",
        font=FONT_NAME,
        wrapWidth=1.8
    )
    intro.draw(); win.flip()
    keys = event.waitKeys(keyList=["space","escape"])
    if "escape" in keys:
        win.close()
        core.quit()

    _debug_overlay_on = SHOW_DEBUG_OVERLAY_DEFAULT

    while True:
        key = run_main_keypad(buffer_text)

        if not run_emg_confirm_key(key, buffer_text):
            continue

        if key not in LETTER_MAP and key != "0":
            buffer_text += key
            print("Typed:", buffer_text)
            continue

        choice = run_letter_panel(key, buffer_text)

        if run_emg_confirm_letter(choice, buffer_text):
            if choice == "<BACK":
                buffer_text = buffer_text[:-1]
            else:
                buffer_text += choice
            print("Typed:", buffer_text)


if __name__ == "__main__":
    main()
