from psychopy import visual, core, event
import numpy as np

###########################################
# CONFIG
###########################################
DEBUG_KEYBOARD = True
FONT_NAME = "Helvetica"

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

# --- Letter panel SSVEP frequencies ---
FREQS_LETTER = [8.0, 9.0, 10.0, 11.0, 12.0]
PHASES_LETTER = [0.0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4]

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
def eeg_predict_key():
    if DEBUG_KEYBOARD:
        keys = event.getKeys()
        for k in keys:
            if k in ["1","2","3","4","5","6","7","8","9","0"]:
                return KEYPAD_LABELS.index(k)
            if k == "num_multiply":
                return KEYPAD_LABELS.index("*")
            if k == "num_add":
                return KEYPAD_LABELS.index("#")
        return None
    return None

def eeg_predict_letter(n_items):
    if DEBUG_KEYBOARD:
        mapping = ["a","b","c","d","e"]
        keys = event.getKeys()
        for i,k in enumerate(mapping[:n_items]):
            if k in keys:
                return i
        return None
    return None

def emg_confirm():
    if DEBUG_KEYBOARD:
        keys = event.getKeys()
        if "y" in keys: return True
        if "n" in keys: return False
        return None
    return None

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

        if t > 1.0:
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
    freqs = FREQS_LETTER[:n]
    phases = PHASES_LETTER[:n]

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
            lum = np.clip(0.4 + 0.5*np.sin(2*np.pi*freqs[i]*t + phases[i]),0,1)
            boxes[i].fillColor = [lum,lum,lum]
            boxes[i].draw()
            texts[i].draw()

        win.flip()

        if t > 1.0:
            idx = eeg_predict_letter(n)
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
