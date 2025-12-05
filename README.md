# Inkling

## Two-stage EEG+EMG speller controller

`inkling_speller.py` contains a small, dependency-light state machine that fuses:
- Stage 1 SSVEP prediction over 12 targets (panels)
- EMG confirmation/rejection for that panel
- Stage 2 SSVEP prediction over 3 targets inside the chosen panel (letters/tokens)
- EMG confirmation/rejection to commit the token into a text buffer

### Quick start
```python
from inkling_speller import TwoStageSpeller

# Instantiate with default 12x3 layout (A-Z plus a few control tokens)
speller = TwoStageSpeller()

# Stage 1: feed SSVEP probabilities over 12 targets
cand1 = speller.propose_stage1(stage1_probs)  # returns a Candidate if over threshold
speller.handle_emg(emg_yes_prob)              # confirm/reject panel (stage 1)

# Stage 2: feed SSVEP probabilities over 3 targets within that panel
cand2 = speller.propose_stage2(stage2_probs)
speller.handle_emg(emg_yes_prob)              # confirm/reject token (stage 2)

print(speller.text)  # accumulated text buffer
```

The EMG handler accepts an `emg_yes_prob` (and optional `emg_no_prob`). It uses
`emg_yes_threshold` / `emg_no_threshold` to decide when to accept/reject; otherwise
it returns `"pending"` to keep waiting.

### Custom layouts and thresholds
- Provide your own 12x3 layout of strings to `TwoStageSpeller(layout=...)`.
- Adjust SSVEP confidence thresholds with `stage1_threshold` / `stage2_threshold`.
- Adjust EMG thresholds with `emg_yes_threshold` / `emg_no_threshold`.

See `demo_run()` at the bottom of `inkling_speller.py` for a tiny console example.

## Streamlit demo UI

`streamlit_app.py` provides a simple, polished UI to drive the speller with model outputs or simulated probabilities.

Run it locally:
```bash
streamlit run streamlit_app.py
```

- Stage 1: choose the 12-target EEG prediction and confidence, then EMG confirm/reject.
- Stage 2: once a panel is confirmed, pick the 3-target EEG prediction for the letter and EMG confirm to append to the text buffer.
- Sidebar has a one-click auto-play to see the flow without hardware.
