from inkling_simple import select_letter_once, update_text
from usemodels import run_stage1_eeg, run_stage2_eeg, run_emg

text = ""

# EEG stage 1
eeg12_probs = run_stage1_eeg(eeg_model, epoch_8x500)

# EMG stage 1
emg1 = run_emg(emg_model, emg_signal)

# EEG stage 2
eeg3_probs = run_stage2_eeg(eeg_model, epoch_3targets)

# EMG stage 2
emg2 = run_emg(emg_model, emg_signal)

letter = select_letter_once(eeg12_probs, emg1, eeg3_probs, emg2)
text = update_text(text, letter)

print(text)
