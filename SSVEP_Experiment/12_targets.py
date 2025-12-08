from psychopy import visual, core, event
import numpy as np
import os
import requests  # for HTTP calls

###########################################
# CONFIG
###########################################
DEBUG_KEYBOARD = True       # keyboard-only control
USE_SIM_DATA = False         # drive EEG/EMG from simulated data when not in DEBUG
SIM_DATA_PATH = "eeg_emg_helloworld.npz"
EMG_BURST_THRESH = 0.5       # envelope/percentile cutoff for sim EMG confirm
STIM_TIME_KEY = 5.0          # seconds of flicker before querying main keypad EEG
STIM_TIME_LETTER = 5.0       # seconds of flicker before querying letter EEG
FONT_NAME = "Helvetica"

# FastAPI servers
EEG_API_URL = "http://127.0.0.1:8000/predict"        # from fast.py
EMG_API_URL = "http://127.0.0.1:8001/predict_file"   # from EMG API

# File lists for offline testing (YOU fill these)
EEG_FILES = [
    r"/Users/hildieleyser/code/hildieleyser/Inkling/S001.mat",  # example if this is your file
]
EMG_FILES = [
    r"/Users/hildieleyser/code/hildieleyser/Inkling/EMG Data Participant 1 Day 1 Block 1.hdf5",
    # r"/absolute/path/to/your_emg_file2.hdf5",
]

_EEG_FILE_IDX = 0
_EMG_FILE_IDX = 0
_EEG_TARGET_IDX = 0

# --- Universal Box Size (SAME AS 12-TARGET KEYPAD) ---
BOX_W = 0.42
BOX_H = 0.30

# --- Main 12-target SSVEP frequencies ---
FREQS_MAIN = [
    8.0, 8.5, 9.0, 9.5,
    10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 13.0, 13.5
]

PHASES_MAIN = [
    0.0, np.pi/2, np.pi, 3*np.pi/2,
    0.0, np.pi/2, np.pi, 3*np.pi/2,
    0.0, np.pi/2, np.pi, 3*np.pi/2
]

# --- Keypad labels ---
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
# (these indices refer to FREQS_MAIN / PHASES_MAIN)
LETTER_CLASS_IDS = {
    "0": [0, 1, 2],            # 0, space, back
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

def _next_eeg_file():
    """
    Cycle through EEG_FILES and return one path each time.
    """
    global _EEG_FILE_IDX
    if not EEG_FILES:
        raise RuntimeError("EEG_FILES list is empty. Add your .mat paths.")
    path = EEG_FILES[_EEG_FILE_IDX % len(EEG_FILES)]
    _EEG_FILE_IDX += 1
    return path


def _next_emg_file():
    """
    Cycle through EMG_FILES and return one path each time.
    """
    global _EMG_FILE_IDX
    if not EMG_FILES:
        raise RuntimeError("EMG_FILES list is empty. Add your .hdf5 paths.")
    path = EMG_FILES[_EMG_FILE_IDX % len(EMG_FILES)]
    _EMG_FILE_IDX += 1
    return path


def _next_eeg_params():
    """
    Provide block/trial/target indices for EEG API calls.
    Currently cycles target_idx 0..11 while keeping block/trial at 0.
    """
    global _EEG_TARGET_IDX
    params = {
        "block_idx": 0,
        "trial_idx": 0,
        "target_idx": _EEG_TARGET_IDX % 12,
    }
    _EEG_TARGET_IDX += 1
    return params

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
# EEG / EMG STUBS
###########################################
def _load_sim_data():
    global _SIM_DATA
    if _SIM_DATA is not None:
        return _SIM_DATA
    # try CWD then script-relative
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
    _SIM_DATA = {k: data[k] for k in data.files}
    return _SIM_DATA


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
    """
    Very simple PSD peak picker: use FFT magnitude at each target freq as logits.
    """
    f = np.fft.rfft(sig)
    hz = np.fft.rfftfreq(len(sig), 1 / fs)
    mags = []
    for target in FREQS_MAIN:
        bin_idx = np.argmin(np.abs(hz - target))
        mags.append(np.abs(f[bin_idx]))
    mags = np.array(mags)
    # avoid zeros, scale to logits
    return np.log(mags + 1e-6)


def get_eeg_logits():
    """
    For file-based FastAPI integration:
    - Upload one .mat file to EEG API `/predict`
    - Read the 0..11 class prediction
    - Convert that into a 'logit' vector (big value on predicted class)
    """
    # If you want to keep the NPZ simulator for dev, you can leave this:
    if USE_SIM_DATA and not DEBUG_KEYBOARD:
        sig, label, fs = _next_sim_eeg()
        logits = _freq_logits_from_signal(sig, fs)
        logits[label] += 1.0
        return logits

    if DEBUG_KEYBOARD:
        # Not really used in this function any more when DEBUG_KEYBOARD=False
        return np.zeros(12, dtype=float)

    # ------------ REAL EEG VIA FASTAPI + FILE UPLOAD ------------
    mat_path = _next_eeg_file()
    try:
        with open(mat_path, "rb") as f:
            files = {"file": (os.path.basename(mat_path), f, "application/octet-stream")}
            # Cycle through targets so you don't always read the same epoch
            params = _next_eeg_params()
            resp = requests.post(EEG_API_URL, files=files, params=params, timeout=10.0)

        resp.raise_for_status()
        data = resp.json()
        pred_class = int(data["prediction"])   # 0..11 from your EEG API
        print(f"EEG API: {mat_path} params={params} -> pred {pred_class}")

    except Exception as e:
        print(f"EEG API error for file {mat_path}: {e}")
        # Return neutral logits (no strong decision)
        return np.zeros(12, dtype=float)

    # Convert class index into pseudo-logits:
    # big positive for predicted class, negative for others
    logits = np.full(12, -5.0, dtype=float)
    logits[pred_class] = 5.0
    return logits


def _eeg_predict_from_subset(active_ids, thresh=None):
    """
    active_ids: list of 12-class indices that are currently on screen.
    Returns local_index (0..K-1) or None if not confident.
    """
    logits = np.array(get_eeg_logits())   # shape (12,)
    logits = logits[active_ids]           # shape (K,)

    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()

    k = int(np.argmax(probs))
    conf = float(probs[k])

    if (thresh is not None) and (conf < thresh):
        return None
    return k


def eeg_predict_key():
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
        for i,k in enumerate(mapping[:n_items]):
            if k in keys:
                return i
        return None
    active_ids = LETTER_CLASS_IDS[key][:n_items]
    return _eeg_predict_from_subset(active_ids, thresh=EEG_LETTER_THRESH)

def emg_confirm():
    use_keyboard = DEBUG_KEYBOARD and not USE_SIM_DATA
    if use_keyboard:
        keys = event.getKeys()
        if "y" in keys:
            return True
        if "n" in keys:
            return False
        return None

    if USE_SIM_DATA:
        # keep your existing simulated EMG behaviour if you want
        sig, fs, label = _next_sim_emg()
        env = np.percentile(np.abs(sig), 95)
        decision = bool(env > EMG_BURST_THRESH)
        if label is not None:
            return bool(label)
        return decision

    # ------------ REAL EMG VIA FASTAPI + FILE UPLOAD ------------
    h5_path = _next_emg_file()

    try:
        with open(h5_path, "rb") as f:
            files = {"file": (os.path.basename(h5_path), f, "application/octet-stream")}
            data = {"dataset": "0"}   # change if you use a different dataset name
            resp = requests.post(EMG_API_URL, files=files, data=data, timeout=10.0)

        resp.raise_for_status()
        out = resp.json()
        # Your EMG API returns {"filename": ..., "result": 0/1, "gesture": "..."}
        result = int(out["result"])

    except Exception as e:
        print(f"EMG API error for file {h5_path}: {e}")
        return None   # so the UI keeps polling

    # 0 = NO, 1 = YES
    return bool(result)

###########################################
# UI Helper
###########################################
def draw_text_buffer(buffer_text):
    visual.TextStim(
        win,text=buffer_text,pos=(0,0.85),
        height=0.06,color="white",font=FONT_NAME
    ).draw()

###########################################
# MAIN KEYPAD
###########################################
def build_main_keypad():
    rows, cols = 4, 3
    x_positions = np.linspace(-0.75, 0.75, cols)
    y_positions = np.linspace(0.55, -0.55, rows)

    targets, labels = [], []
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

            targets.append(box)
            labels.append(label)
            idx += 1

    return targets, labels

def run_main_keypad(buffer_text):
    targets, labels = build_main_keypad()
    clock = core.Clock()

    while True:
        t = clock.getTime()
        draw_text_buffer(buffer_text)

        for i in range(12):
            lum = np.clip(0.4 + 0.5*np.sin(2*np.pi*FREQS_MAIN[i]*t + PHASES_MAIN[i]),0,1)
            targets[i].fillColor = [lum,lum,lum]
            targets[i].draw()
            labels[i].draw()

        win.flip()

        if t > STIM_TIME_KEY:
            idx = eeg_predict_key()
            if idx is not None:
                return KEYPAD_LABELS[idx]

###########################################
# EMG CONFIRMATION (KEY)
###########################################
def run_emg_confirm_key(key, buffer_text):
    msg = visual.TextStim(
        win,text=f"Selected key:\n\n{key}\n\nEMG gesture to confirm.",
        height=0.08,color="white",font=FONT_NAME
    )

    timer = core.Clock()
    while timer.getTime() < 2.0:
        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

    while True:
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
        if n == 5:  # number on top, 4 letters in 2×2 grid
            positions = [
                (0,0.55),
                (-0.45,0.05),(0.45,0.05),
                (-0.45,-0.35),(0.45,-0.35)
            ]

        elif n == 4:  # number on top + 3 letters
            positions = [
                (0,0.55),
                (-0.55,-0.05),(0,-0.05),(0.55,-0.05)
            ]

        elif n == 3:  # number on top + 2 letters
            positions = [
                (0,0.55),
                (-0.45,-0.05),(0.45,-0.05)
            ]

        else:
            raise ValueError("Unexpected panel size.")

    # Build visuals (same size as main keypad)
    boxes, texts = [], []
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

        boxes.append(box)
        texts.append(txt)

    # Flicker loop
    clock = core.Clock()
    while True:
        t = clock.getTime()
        draw_text_buffer(buffer_text)

        for i in range(n):
            cls_id = class_ids[i]
            freq = FREQS_MAIN[cls_id]
            phase = PHASES_MAIN[cls_id]
            lum = np.clip(0.4 + 0.5*np.sin(2*np.pi*freq*t + phase),0,1)
            boxes[i].fillColor = [lum,lum,lum]
            boxes[i].draw()
            texts[i].draw()

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
        win,text=f"Confirm:\n\n{letter}",
        height=0.08,color="white",font=FONT_NAME
    )

    timer = core.Clock()
    while timer.getTime() < 2.0:
        draw_text_buffer(buffer_text); msg.draw(); win.flip()

    while True:
        draw_text_buffer(buffer_text); msg.draw(); win.flip()
        d = emg_confirm()
        if d is not None:
            return d

###########################################
# MAIN LOOP
###########################################
def main():
    buffer_text = ""

    intro = visual.TextStim(
        win,text="EEG-EMG Hybrid Speller\nPress SPACE to begin.",
        height=0.05,color="white",font=FONT_NAME
    )
    intro.draw(); win.flip()
    event.waitKeys(keyList=["space"])

    while True:
        key = run_main_keypad(buffer_text)

        if not run_emg_confirm_key(key, buffer_text):
            continue

        if key not in LETTER_MAP and key != "0":
            buffer_text += key
            print("Typed:", buffer_text)
            continue

        # Letter panel
        choice = run_letter_panel(key, buffer_text)

        if run_emg_confirm_letter(choice, buffer_text):
            if choice == "<BACK":
                buffer_text = buffer_text[:-1]
            else:
                buffer_text += choice

            print("Typed:", buffer_text)


if __name__ == "__main__":
    main()
