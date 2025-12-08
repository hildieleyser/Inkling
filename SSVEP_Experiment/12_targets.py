from psychopy import visual, core, event
import numpy as np
import string

###########################################
# CONFIG
###########################################
DEBUG_KEYBOARD = True   # Set False when using real EEG/EMG

# --- SSVEP frequencies and phases for 12 keys ---
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

# For letter panels we’ll just use the first 4 frequencies again
FREQS_LETTER = [8.0, 9.0, 10.0, 11.0]
PHASES_LETTER = [0.0, np.pi/2, np.pi, 3*np.pi/2]

# --- Nokia keypad mapping ---
KEYPAD_LABELS = [
    "1", "2", "3",
    "4", "5", "6",
    "7", "8", "9",
    "*", "0", "#"
]

LETTER_MAP = {
    "2": ["A", "B", "C"],
    "3": ["D", "E", "F"],
    "4": ["G", "H", "I"],
    "5": ["J", "K", "L"],
    "6": ["M", "N", "O"],
    "7": ["P", "Q", "R", "S"],
    "8": ["T", "U", "V"],
    "9": ["W", "X", "Y", "Z"],
}

# --- PsychoPy window ---
FONT_NAME = "Helvetica"

win = visual.Window(
    size=[0, 0],
    fullscr=True,
    color=[0.2, 0.2, 0.2],
    units="norm"
)

###########################################
# STUBS: EEG & EMG MODEL CALLS
###########################################

def eeg_predict_key(prob_vector=None):
    """
    Return index 0..11 of selected main-keypad target.

    Replace this with:
        - call to your SSVEP model
        - return np.argmax(predictions) for 12 classes

    For now (DEBUG):
      - use keyboard: keys 1..9, 0, *, # to fake EEG
    """
    if DEBUG_KEYBOARD:
        keys = event.getKeys()
        for k in keys:
            if k in ["1","2","3","4","5","6","7","8","9","0"]:
                label = k
            elif k == "num_multiply":
                label = "*"
            elif k == "num_add":
                label = "#"
            else:
                continue
            return KEYPAD_LABELS.index(label)
        return None

    # TODO: plug in your real EEG model here
    # example:
    # idx = np.argmax(model.predict(...))
    # return int(idx)

    return None


def eeg_predict_letter(n_letters):
    """
    Return index 0..(n_letters-1) for letter panel.

    For now in DEBUG:
      - keys a,b,c,d map to 0,1,2,3
    """
    if DEBUG_KEYBOARD:
        mapping = ["a", "b", "c", "d"]
        keys = event.getKeys()
        for i, k in enumerate(mapping[:n_letters]):
            if k in keys:
                return i
        return None

    # TODO: plug in real SSVEP letter-panel model
    return None


def emg_confirm():
    """
    EMG confirmation:
      return True  -> confirm
      return False -> reject

    In DEBUG:
      'y' -> confirm, 'n' -> reject
    """
    if DEBUG_KEYBOARD:
        keys = event.getKeys()
        if "y" in keys:
            return True
        if "n" in keys:
            return False
        return None

    # TODO: call your EMG classifier here
    # e.g. prob = emg_model.predict(...)
    # return bool(prob > 0.5)

    return None


###########################################
# UI HELPERS
###########################################

def draw_text_buffer(buffer_text):
    buf = visual.TextStim(
        win,
        text=buffer_text,
        pos=(0, 0.85),
        height=0.06,
        color="white",
        font=FONT_NAME
    )
    buf.draw()


###########################################
# 1. MAIN 12-KEY SSVEP KEYPAD
###########################################

def build_main_keypad():
    rows, cols = 4, 3
    x_positions = np.linspace(-0.75, 0.75, cols)
    y_positions = np.linspace(0.55, -0.55, rows)

    targets = []
    labels = []

    idx = 0
    for r in range(rows):
        for c in range(cols):
            pos = (x_positions[c], y_positions[r])
            key = KEYPAD_LABELS[idx]

            box = visual.Rect(
                win,
                width=0.42,
                height=0.30,
                fillColor=[0, 0, 0],
                lineColor=[1, 1, 1],
                pos=pos,
                lineWidth=2
            )

            # number + letters (e.g. "5\nJKL")
            if key in LETTER_MAP:
                text_str = key + "\n" + "".join(LETTER_MAP[key])
            else:
                text_str = key

            txt = visual.TextStim(
                win,
                text=text_str,
                pos=pos,
                color="white",
                height=0.10,
                font=FONT_NAME,
                alignText="center"
            )

            targets.append(box)
            labels.append(txt)
            idx += 1

    return targets, labels


def run_main_keypad(buffer_text):
    """
    Show main 12-key keypad with flicker and return chosen key label.
    """
    targets, texts = build_main_keypad()
    clock = core.Clock()

    while True:
        t = clock.getTime()

        # quit
        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)

        for i in range(12):
            lum = 0.4 + 0.5 * np.sin(2 * np.pi * FREQS_MAIN[i] * t + PHASES_MAIN[i])
            lum = np.clip(lum, 0.0, 1.0)
            targets[i].fillColor = [lum, lum, lum]
            targets[i].draw()
            texts[i].draw()

        win.flip()

        # EEG prediction query
        if t > 1.0:  # at least 1 second of stimulation
            idx = eeg_predict_key()
            if idx is not None:
                return KEYPAD_LABELS[idx]


###########################################
# 2. EMG CONFIRMATION FOR KEY
###########################################

def run_emg_confirm_key(key, buffer_text):
    """
    Show EMG confirmation screen for a selected key.
    Returns True/False.
    """
    msg = visual.TextStim(
        win,
        text=f"Selected key:\n\n{key}\n\n"
             f"EMG: gesture = CONFIRM, still = CANCEL\n"
             f"(DEBUG: 'y' yes, 'n' no)",
        height=0.08,
        color="white",
        wrapWidth=1.6,
        font=FONT_NAME
    )

    timer = core.Clock()
    window_s = 2.0  # show for 2s before checking EMG

    while timer.getTime() < window_s:
        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

    # Now poll EMG until we get a decision
    while True:
        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

        decision = emg_confirm()
        if decision is not None:
            return decision


###########################################
# 3. LETTER-PANEL SSVEP FOR A GIVEN KEY
###########################################

def run_letter_panel(key, buffer_text):
    """
    Show SSVEP letter panel for a given key (2-9).
    Returns selected letter (e.g. 'K').
    """
    letters = LETTER_MAP[key]
    n = len(letters)

    # positions in a row
    x_positions = np.linspace(-0.6, 0.6, n)

    boxes = []
    texts = []
    for i, letter in enumerate(letters):
        pos = (x_positions[i], 0.0)

        box = visual.Rect(
            win,
            width=0.35,
            height=0.35,
            fillColor=[0, 0, 0],
            lineColor=[1, 1, 1],
            pos=pos,
            lineWidth=2
        )
        txt = visual.TextStim(
            win,
            text=letter,
            pos=pos,
            color="white",
            height=0.18,
            font=FONT_NAME
        )
        boxes.append(box)
        texts.append(txt)

    instr = visual.TextStim(
        win,
        text=f"Select a letter for key {key}",
        pos=(0, 0.7),
        color="white",
        height=0.07,
        font=FONT_NAME
    )

    clock = core.Clock()

    while True:
        t = clock.getTime()

        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)
        instr.draw()

        for i in range(n):
            freq = FREQS_LETTER[i]
            phase = PHASES_LETTER[i]
            lum = 0.4 + 0.5 * np.sin(2 * np.pi * freq * t + phase)
            lum = np.clip(lum, 0.0, 1.0)
            boxes[i].fillColor = [lum, lum, lum]
            boxes[i].draw()
            texts[i].draw()

        win.flip()

        if t > 1.0:
            idx = eeg_predict_letter(n_letters=n)
            if idx is not None:
                return letters[idx]


###########################################
# 4. EMG FINAL CONFIRMATION FOR LETTER
###########################################

def run_emg_confirm_letter(letter, buffer_text):
    msg = visual.TextStim(
        win,
        text=f"Confirm letter:\n\n{letter}\n\n"
             f"EMG: gesture = CONFIRM, still = CANCEL\n"
             f"(DEBUG: 'y' yes, 'n' no)",
        height=0.08,
        color="white",
        wrapWidth=1.6,
        font=FONT_NAME
    )

    timer = core.Clock()
    window_s = 2.0

    while timer.getTime() < window_s:
        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

    while True:
        if "escape" in event.getKeys():
            win.close()
            core.quit()

        draw_text_buffer(buffer_text)
        msg.draw()
        win.flip()

        decision = emg_confirm()
        if decision is not None:
            return decision


###########################################
# 5. MASTER LOOP
###########################################

def main():
    buffer_text = ""  # typed text

    # Intro screen
    intro = visual.TextStim(
        win,
        text="Hybrid EEG–EMG Speller\n\n"
             "EEG: Look at a flashing key\n"
             "EMG: Gesture to confirm\n\n"
             "DEBUG:\n"
             "  Main keypad: number keys / numpad * = '*', numpad + = '#'\n"
             "  Letter panel: a,b,c,d\n"
             "  EMG: y = confirm, n = cancel\n\n"
             "Press SPACE to begin.",
        height=0.05,
        wrapWidth=1.6,
        color="white",
        font=FONT_NAME
    )
    intro.draw()
    win.flip()
    event.waitKeys(keyList=["space", "escape"])
    if "escape" in event.getKeys():
        win.close()
        core.quit()

    while True:
        # --- Step 1: main EEG keypad ---
        key = run_main_keypad(buffer_text)

        # --- Step 2: EMG confirm key ---
        if not run_emg_confirm_key(key, buffer_text):
            # rejected → restart
            continue

        # If key has no letters (1,0,*,#) -> directly append
        if key not in LETTER_MAP:
            buffer_text += key
            print("Typed:", buffer_text)
            continue

        # --- Step 3: letter panel EEG ---
        letter = run_letter_panel(key, buffer_text)

        # --- Step 4: EMG final confirm ---
        if run_emg_confirm_letter(letter, buffer_text):
            buffer_text += letter
            print("Typed:", buffer_text)
        else:
            # letter cancelled
            continue


if __name__ == "__main__":
    main()
