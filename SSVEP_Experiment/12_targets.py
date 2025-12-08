from SSVEP_Experiment.psychopy import visual, core, event
import numpy as np

# ==============================
# 1. Stimulation parameters
# ==============================
stim_info = [
    {"label": "1", "freq": 9.25,  "phase": 0},
    {"label": "2", "freq": 11.25, "phase": 0},
    {"label": "3", "freq": 13.25, "phase": 0},
    {"label": "4", "freq": 9.75,  "phase": 0.5},
    {"label": "5", "freq": 11.75, "phase": 0.5},
    {"label": "6", "freq": 13.75, "phase": 0.5},
    {"label": "7", "freq": 10.25, "phase": 1},
    {"label": "8", "freq": 12.25, "phase": 1},
    {"label": "9", "freq": 14.25, "phase": 1},
    {"label": "0", "freq": 10.75, "phase": 1.5},
    {"label": "*", "freq": 12.75, "phase": 1.5},
    {"label": "#", "freq": 14.75, "phase": 1.5},
]

stim_duration = 5  # seconds per trial

# ==============================
# 2. Window
# ==============================
win = visual.Window(size=[1920, 1080], color=[0, 0, 0], fullscr=False)
refresh_rate = win.getActualFrameRate()
if refresh_rate is None:
    refresh_rate = 60
print("Refresh rate:", refresh_rate)

# ==============================
# 3. Build 12 stimuli
# ==============================
stimuli = []
positions = [
    (-0.6, 0.4), (-0.2, 0.4), (0.2, 0.4), (0.6, 0.4),
    (-0.6, 0.0), (-0.2, 0.0), (0.2, 0.0), (0.6, 0.0),
    (-0.6,-0.4), (-0.2,-0.4), (0.2,-0.4), (0.6,-0.4),
]

for info, pos in zip(stim_info, positions):
    stim = visual.Rect(
        win,
        width=0.25,
        height=0.25,
        fillColor=[1, 1, 1],
        lineColor=[1, 1, 1],
        pos=pos
    )
    info["stim"] = stim

# ==============================
# 4. Stimulation loop
# ==============================
timer = core.Clock()
frame = 0

while timer.getTime() < stim_duration:
    t = frame / refresh_rate  # current time in seconds

    for info in stim_info:
        freq = info["freq"]
        phase_pi = info["phase"] * np.pi  # convert to radians
        stim = info["stim"]

        # Sinusoidal luminance modulation (0 → white, -1 → black)
        lum = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase_pi)
        stim.fillColor = [lum, lum, lum]

        stim.draw()

    win.flip()
    frame += 1

# Cleanup
win.close()
core.quit()
