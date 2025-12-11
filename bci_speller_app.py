
# bci_speller_streamlit.py
# Streamlit UI for Inkling: hybrid EEG + EMG speller

import time
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st


# ============================================================
# GLOBAL CONFIG AND STYLING
# ============================================================

EEG_API_URL = "http://localhost:8000/predict"
EEG_LAST_PRED_URL = "http://localhost:8000/last_auto_prediction"
EMG_API_URL = "http://localhost:8001/predict_file"
FS = 250  # demo sampling rate for EEG animation


def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def inject_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lekton:wght@400;700&display=swap');

    :root {
        --ink-primary:   #2A4F98;   /* deep blue */
        --ink-cream:     #ECE2CD;   /* warm sand */
        --ink-brown:     #3F1110;   /* dark brown */
        --ink-red:       #860100;   /* deep red */

        --ink-shadow-soft: 0 8px 20px rgba(0,0,0,0.08);
        --ink-shadow-strong: 0 18px 48px rgba(0,0,0,0.18);
    }

    /* GLOBAL APP BACKGROUND -------------------------------------------- */

    .stApp {
        background: linear-gradient(
            140deg,
            var(--ink-cream) 0%,
            #f7f2e6 40%,
            var(--ink-primary) 130%
        );
        color: #0a0f14;
        font-family: "Gilroy", "Inter", sans-serif;
    }

    .block-container {
        width: min(1200px, 95vw);
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        margin: 0 auto;
        z-index: 1;
        position: relative;
    }

    /* TYPOGRAPHY -------------------------------------------------------- */

    h1, h2, h3, h4 {
        font-family: "Gilroy", "Inter", sans-serif !important;
        color: #0a0f14 !important;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    h1 { font-size: clamp(2.6rem, 4vw, 3.2rem); }
    h2 { font-size: clamp(1.8rem, 3vw, 2.2rem); }
    h3 { font-size: 1.25rem; }

    p, li {
        font-family: "Gilroy", "Inter", sans-serif;
        color: #0a0f14;
        font-size: 1rem;
    }

    /* HERO --------------------------------------------------------------- */

    .inkling-hero {
        margin-top: 2rem;
        margin-bottom: 2.4rem;
        padding: 2.2rem;
        background: linear-gradient(
            120deg,
            var(--ink-primary),
            #3d6ac0,
            var(--ink-red)
        );
        color: #ffffff;
        border-radius: 24px;
        box-shadow: var(--ink-shadow-strong);
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .inkling-hero-title {
        font-size: clamp(2.8rem, 4.4vw, 3.6rem);
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .inkling-hero-kicker {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.3em;
        opacity: 0.9;
    }

    .inkling-hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 600px;
        opacity: 0.95;
    }

    .inkling-hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.4rem;
    }

    .inkling-pill {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        padding: 0.28rem 1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.45);
        font-size: 0.75rem;
    }

    /* SPLASH CALLOUT ---------------------------------------------------- */

    .inkling-splash {
        background: transparent;
        border: none;
        padding: 0.4rem 0;
        border-radius: 0;
        box-shadow: none;
        margin-bottom: 0.9rem;
    }

    .inkling-splash-title {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.28em;
        font-size: 0.78rem;
        color: var(--ink-primary);
        margin-bottom: 0.25rem;
    }

    /* CARDS -------------------------------------------------------------- */

    .inkling-card,
    .inkling-team-card {
        background: linear-gradient(145deg, rgba(236,226,205,0.85), rgba(213,199,180,0.75));
        border-radius: 18px;
        padding: 1.7rem 2rem;
        border: 1px solid rgba(10,15,20,0.12);
        box-shadow: 0 14px 30px rgba(0,0,0,0.1);
        margin-top: 1.2rem;
        transition: all 140ms ease;
    }

    .inkling-card:hover,
    .inkling-team-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--ink-shadow-strong);
        border-color: rgba(42,79,152,0.35);
    }

    /* BUTTONS ------------------------------------------------------------ */

    .stButton>button {
        background: var(--ink-primary) !important;
        color: #ffffff !important;
        border-radius: 999px;
        padding: 0.6rem 1.8rem;
        text-transform: uppercase;
        font-family: "Lekton", monospace;
        letter-spacing: 0.22em;
        font-size: 0.78rem;
        border: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transition: all 140ms ease;
    }

    .stButton>button:hover {
        background: var(--ink-red) !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.28);
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# VISUAL DEMOS AND EXPLAINERS
# ============================================================

def render_ssvep_demo():
    st.markdown("#### SSVEP illustration")

    with st.container():
        col_ctrl, col_plot = st.columns([1, 2])

        with col_ctrl:
            freq = st.slider("Flicker frequency (Hz)", min_value=5, max_value=20, value=12)
            duration = st.slider("Window length (seconds)", 1.0, 3.0, 1.0, step=0.5)
            fs = 250
            noise_level = st.slider(
                "Noise level", min_value=0.0, max_value=1.0, value=0.3, step=0.1
            )

            t = np.arange(0, duration, 1.0 / fs)
            signal = np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(len(t))

        with col_plot:
            st.markdown("Time domain")
            fig_time, ax_time = plt.subplots()
            ax_time.plot(t, signal)
            ax_time.set_xlabel("Time (s)")
            ax_time.set_ylabel("Amplitude")
            ax_time.set_xlim(0, min(duration, 1.0))
            st.pyplot(fig_time)

            st.markdown("Frequency domain")
            n = len(signal)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            spectrum = np.abs(np.fft.rfft(signal)) ** 2

            fig_freq, ax_freq = plt.subplots()
            ax_freq.plot(freqs, spectrum)
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("Power")
            ax_freq.set_xlim(0, 40)
            st.pyplot(fig_freq)

    st.markdown(
        "The peak in the spectrum approximates the flicker frequency the signal is locked to."
    )


def render_state_machine_diagram():
    st.markdown("#### Control state machine")

    dot = r"""
digraph {
    rankdir=LR;
    node [shape=box];

    idle      [label="idle\nwaiting for next character"];
    key_eeg   [label="key_eeg\nEEG proposes keypad key"];
    key_emg   [label="key_emg\nEMG confirms or rejects key"];
    char_eeg  [label="char_eeg\nEEG proposes letter"];
    char_emg  [label="char_emg\nEMG confirms or rejects letter"];

    idle     -> key_eeg   [label="start"];
    key_eeg  -> key_emg   [label="EEG key"];
    key_emg  -> char_eeg  [label="confirm"];
    key_emg  -> idle      [label="reject"];

    char_eeg -> char_emg  [label="EEG letter"];
    char_emg -> idle      [label="confirm or cancel"];
    char_emg -> char_eeg  [label="reject"];
}
"""
    st.graphviz_chart(dot)
    st.markdown(
        "Each transition corresponds to a combination of EEG evidence and EMG decision."
    )


def render_inkling_explainer():
    st.markdown("## How Inkling works")

    st.markdown(
        """
Inkling is a hybrid brain–muscle speller.

EEG is used to infer which visual pattern the user is attending to.
EMG is used to decide what to do with that option: confirm, delete, cancel, or ignore.

The aim is to allow typing with only brief, deliberate muscle bursts.
"""
    )

    st.divider()
    st.markdown("### Brain signals: EEG and SSVEP")

    tab_overview, tab_model = st.tabs(
        ["Flicker and visual response", "What the EEG model computes"]
    )

    with tab_overview:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
On the display, twelve blocks flicker with distinct frequencies and phases.
Each block contains three letters, producing a 36-letter layout.

When a user fixates one block, the visual cortex oscillates at that block’s flicker pattern.
This is a steady-state visual evoked potential, measurable over occipital and parietal electrodes.
"""
            )

        with col2:
            st.markdown(
                """
In the first stage, the system attempts to identify which of the twelve flickers dominates the response.

In the second stage, only the three letters inside the chosen block matter.
They then flicker in three distinct ways, allowing the EEG model to separate them.
"""
            )

    with tab_model:
        col3, col4 = st.columns([2, 1])

        with col3:
            st.markdown(
                """
The EEG model receives short windows of multi-channel occipital and parietal data.

A convolutional network outputs:
twelve probabilities during block selection,
and three probabilities during letter selection.

The highest-probability option is treated as the current candidate when confidence is sufficient.
"""
            )

        with col4:
            st.markdown(
                """
Conceptual flow:

Input: EEG time series
Feature extraction: temporal and spatial filters
Output: probability distribution over the active flicker set
"""
            )

    st.markdown(
        """
Inkling is not reading thoughts.
It is identifying which external flicker pattern the visual cortex is currently phase-locked to.
"""
    )

    st.markdown("### SSVEP demo")
    with st.container():
        st.markdown('<div class="inkling-demo-frame">', unsafe_allow_html=True)
        render_ssvep_demo()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Muscle signals: EMG and intention bursts")

    tab_emg_overview, tab_emg_mapping = st.tabs(
        ["Detecting bursts", "Mapping bursts to actions"]
    )

    with tab_emg_overview:
        col5, col6 = st.columns(2)

        with col5:
            st.markdown(
                """
Many individuals with severe motor impairment retain small, controllable contractions in at least one muscle.

Inkling records surface EMG from such a site:
two electrodes over the muscle, rectified signal, smoothed amplitude envelope.
"""
            )

        with col6:
            st.markdown(
                """
A simple classifier distinguishes:
rest, single burst, and double burst.

Single bursts are short, intentional twitches.
Double bursts are two twitches close together.
This requires minimal muscle strength, only reliability.
"""
            )

    with tab_emg_mapping:
        st.markdown(
            """
EMG states are translated into discrete control signals:

Single burst → confirm the current candidate
Double burst → delete the last character
Optional extra pattern → cancel and return
No burst → hold state

EMG does not select letters.
It confirms, rejects, or cancels what the EEG system proposes.
"""
        )

    st.divider()
    st.markdown("### Control loop: EEG proposals and EMG decisions")

    col7, col8 = st.columns(2)

    with col7:
        st.markdown(
            """
Block stage

1. EEG estimates which of twelve blocks is attended.
2. The interface highlights that block.
3. An EMG confirm locks it in and moves the system to the letter stage.
"""
        )

        st.markdown(
            """
Letter stage

1. EEG estimates which of the three letters is attended.
2. The interface highlights that letter.
3. EMG:
   • confirm → append letter to text
   • delete → remove the previous letter
   • cancel → return to block selection
"""
        )

    with col8:
        st.markdown(
            """
Simplified logic:

while typing:
&nbsp;&nbsp;&nbsp;&nbsp;eeg_choice = argmax p(option | EEG)
&nbsp;&nbsp;&nbsp;&nbsp;if confidence is high: highlight(eeg_choice)

&nbsp;&nbsp;&nbsp;&nbsp;emg = classify(EMG)
&nbsp;&nbsp;&nbsp;&nbsp;if emg == CONFIRM: commit(eeg_choice)
&nbsp;&nbsp;&nbsp;&nbsp;elif emg == DELETE: delete_last()
&nbsp;&nbsp;&nbsp;&nbsp;elif emg == CANCEL: go_back()

No action is committed without both a reliable EEG signal and an explicit EMG confirmation.
"""
        )

    render_state_machine_diagram()

    st.divider()
    st.markdown("### Why combine EEG and EMG")

    col9, col10 = st.columns(2)

    with col9:
        st.markdown(
            """
EEG alone

Can reach high information rates,
but requires constant visual focus on flickering targets
and can become tiring over long periods.
"""
        )

    with col10:
        st.markdown(
            """
EMG alone

Works well with strong, stable contractions.
Becomes difficult with tremor, dystonia, or very weak muscles.
Continuous EMG control can be tiring.
"""
        )

    st.markdown(
        """
The hybrid design separates roles.

EEG manages continuous selection of block and letter.
EMG performs short, decisive actions.

This division reduces fatigue and is designed for people who have very limited,
but reliable, motor output.
"""
    )


# ============================================================
# TEAM
# ============================================================

def render_meet_the_team():
    st.markdown("## Meet the team")

    st.markdown(
        """
Inkling is being developed by a small interdisciplinary group with backgrounds in neuroscience,
engineering, data science, and applied machine learning.
"""
    )

    st.divider()

    # Hildelith
    st.markdown('<div class="inkling-team-card">', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-role">Project leader</div>', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-name">Hildelith Frances Leyser</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="inkling-team-body">
I am a neuroscience PhD student at McGill University. At the RIKEN Center for Brain Science
and the Montreal Neurological Institute I work on decision-making and sensory-motor
datasets and I build real-time analysis pipelines. I have practical experience with EEG, EMG,
eye tracking, kinematic tracking, and autonomic measures. I have designed and run
behavioural and neural experiments with both humans and non-human primates. I
regularly produce clear documentation and reproducible code for students and
collaborators.

This background prepares me to lead the design and implementation of the
hybrid EEG and EMG speller and to release it in a way that others can understand
and use. I have also developed neurotechnology prototypes during research
hackathons, including an EEG synchrony system, an EMG-driven regulation game, and a
VR-based cognitive task. These projects required hardware integration, Python signal
processing, and rapid system building.

As the senior advisor for the McGill Biomechanics society I helped build an EMG-based
wearable prototype for a Parkinson's exoskeleton in collaboration with clinics
during 2023–2024 and I am a clinic volunteer for Parkinson's awareness.
These issues are close to my heart and I have worked with them in the lab, in the clinic,
and in my family.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Tanya
    st.markdown('<div class="inkling-team-card">', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-role">Team member</div>', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-name">Tanya Saha Gupta</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="inkling-team-contact">Contact: tanyasahagupta@gmail.com · +44 7800 648047</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="inkling-team-body">
I have an interdisciplinary background spanning biological science, quantitative analysis,
and applied machine learning. I hold a degree in Biochemistry from Imperial College
London, where my academic work required rigorous engagement with human physiology,
signal interpretation, and experimental data.

Professionally, I worked in an analytical role in investment banking, building and
stress-testing quantitative models under real-world constraints, before founding and
engineering a retail-technology startup where I developed a Shopify app.
Most recently, I have been deepening my technical training in machine
learning and statistical modelling.

This combination of physiological understanding, quantitative reasoning, and practical
machine learning implementation motivates my interest in working with EEG and EMG
signals using BCI headgear.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Zaki
    st.markdown('<div class="inkling-team-card">', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-role">Team member</div>', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-name">Zaki Baalwaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="inkling-team-contact">Contact: +44 7454 812223 · zaki_b98@hotmail.co.uk</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="inkling-team-body">
I hold a master in Engineering and I am currently an Assistant Engineer with experience
primarily in the structures team for Project Centre, where I contribute to the design,
analysis, and maintenance of highway structures, ensuring safety and compliance with
industry standards.

I also have a background in data science and applied machine learning, with
experience working on signal-processing projects that involve cleaning, transforming,
and modelling complex time-series data. I am proficient in Python, using libraries such as
NumPy, SciPy, MNE, and scikit-learn for statistical analysis and feature extraction, and I
have worked with biometrics-related datasets that required careful noise
reduction and interpretation.

My academic and project experience has given me an understanding of neural data
characteristics, experimental design, and ethical handling of sensitive information.
Access to the BCI headgear would enable me to apply these skills to real-world neural
signals and to extend both the project and my technical expertise.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Rayan
    st.markdown('<div class="inkling-team-card">', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-role">Team member</div>', unsafe_allow_html=True)
    st.markdown('<div class="inkling-team-name">Rayan Hasan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="inkling-team-contact">Contact: +1 365 292 5250 · rayan@dada.com.pk</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="inkling-team-body">
I completed my undergraduate degree at the University of Toronto in Organizational
Management and Human Geography, a program that strengthened my analytical, research,
and problem-solving abilities.

Previously, I worked with Nestlé on a project focused on digitizing record-keeping systems
for small dairy farms, which exposed me to technology adoption, data organization, and
field-level operational challenges. I am also completing the Le Wagon Data Science and
Deep Learning Bootcamp, where I have developed practical skills in machine learning,
signal processing, and model development.

These academic and technical experiences support my interest in working with BCI
headgear and ensure that I can approach its use with responsibility, rigour, and a strong
understanding of data-driven methodologies.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# KEYPAD + LETTER MAPPING (WITH FREQUENCIES)
# ============================================================

KEYPAD_LABELS = [
    "1", "2", "3",
    "4", "5", "6",
    "7", "8", "9",
    "*", "0", "#",
]

LETTER_MAP = {
    "1": [],
    "2": ["A (9.25Hz)", "B (11.25Hz)", "C (13.25Hz)"],
    "3": ["D (9.25Hz)", "E (11.25Hz)", "F (13.25Hz)"],
    "4": ["G (9.25Hz)", "H (11.25Hz)", "I (13.25Hz)"],
    "5": ["J (9.25Hz)", "K (11.25Hz)", "L (13.25Hz)"],
    "6": ["M (9.25Hz)", "N (11.25Hz)", "O (13.25Hz)"],
    "7": ["P (9.25Hz)", "Q (11.25Hz)", "R (13.25Hz)", "S (9.75Hz)"],
    "8": ["T (9.25Hz)", "U (11.25Hz)", "V (13.25Hz)"],
    "9": ["W (9.25Hz)", "X (11.25Hz)", "Y (13.25Hz)", "Z (9.75Hz)"],
    "0": [" (9.25Hz)"],
    "*": [],
    "#": [],
}

LETTER_CLASS_IDS = {
    "1": [0],
    "*": [0],
    "#": [0],
    "0": [0, 1, 2],          # 0, space, back
    "2": [0, 1, 2, 3],       # 2, A, B, C
    "3": [0, 1, 2, 3],       # 3, D, E, F
    "4": [0, 1, 2, 3],       # 4, G, H, I
    "5": [0, 1, 2, 3],       # 5, J, K, L
    "6": [0, 1, 2, 3],       # 6, M, N, O
    "7": [0, 1, 2, 3, 4],    # 7, P, Q, R, S
    "8": [0, 1, 2, 3],       # 8, T, U, V
    "9": [0, 1, 2, 3, 4],    # 9, W, X, Y, Z
}


# ============================================================
# PANEL / SELECTION HELPERS
# ============================================================

def build_letter_panel(key: str) -> list[str]:
    if key == "0":
        return ["0"] + LETTER_MAP["0"] + ["<BACK>"]

    letters = LETTER_MAP.get(key, [])
    if letters:
        return [key] + letters

    return [key]


def extract_char_from_token(token: str) -> str:
    """
    Take 'H (11.25Hz)' -> 'H'
    Take ' (9.25Hz)'  -> ' '
    Keep '<BACK>' as-is.
    """
    if token == "<BACK>":
        return "<BACK>"
    if token and token[0] == " ":
        return " "
    return token[0]


def pretty_item(x: str) -> str:
    if x == " ":
        return "<SPACE>"
    if x == "<BACK>":
        return "←BACK"
    return x


# ============================================================
# EEG / EMG API CALLS
# ============================================================

def call_eeg_api(eeg_bytes: bytes, block_idx: int, trial_idx: int, target_idx: int):
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


def call_emg_api(emg_bytes: bytes, dataset_name: str = "0"):
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


def get_last_eeg_prediction():
    try:
        resp = requests.get(EEG_LAST_PRED_URL, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"EEG API error: {e}")
        return None

    out = resp.json()
    pred = out.get("prediction")
    if pred is None:
        st.error(f"EEG API response missing 'prediction': {out}")
        return None

    return out


# ============================================================
# STATE MANAGEMENT
# ============================================================

def init_state():
    defaults = {
        "typed_text": "",
        "phase": "idle",
        "current_key": None,
        "current_panel_items": None,
        "current_candidate_char": None,
        "last_message": "",
        "eeg_file_bytes": None,
        "emg_file_bytes": None,
        "last_prediction": None,
        "last_prediction_payload": None,
        "block_idx": 0,
        "trial_idx": 0,
        "target_idx": 0,
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
    st.markdown("### Output text")
    st.text_area("Typed text", st.session_state.typed_text, height=90, disabled=True)


# ============================================================
# SPELLER UI (LANDING TAB)
# ============================================================

def render_speller_ui():
    col_main, col_side = st.columns([1.65, 1], gap="large")

    with col_main:
        st.markdown('<div class="inkling-card">', unsafe_allow_html=True)
        st.markdown("### Decode my thoughts")
        st.markdown(
            "Fetch the latest automatic EEG prediction from the backend watcher "
            "(`/last_auto_prediction`). Keep your EEG server running so it can populate this."
        )

        if st.button("Decode my thoughts"):
            result = get_last_eeg_prediction()
            if result is not None:
                st.session_state.last_prediction = result.get("prediction")
                st.session_state.last_prediction_payload = result
                st.success(f"EEG model predicted letter class {st.session_state.last_prediction}.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="inkling-card">', unsafe_allow_html=True)
        st.markdown("### Prediction status")

        if st.session_state.last_prediction is not None:
            label = None
            payload = st.session_state.last_prediction_payload or {}
            label = payload.get("label")
            st.markdown(
                f"""
                <div class="selection-card">
                    <div class="selection-card-label">Predicted letter class</div>
                    <div class="selection-card-main">{st.session_state.last_prediction}</div>
                    <div class="selection-card-sub">From /last_auto_prediction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if label is not None:
                st.markdown(
                    f"""
                    <div style="
                        margin-top: 10px;
                        font-family: 'Lekton', monospace;
                        font-size: 0.78rem;
                        letter-spacing: 0.14em;
                        text-transform: uppercase;
                        color: rgba(11,32,39,0.75);
                    ">
                        Label: {label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <p style="color: rgba(11,32,39,0.7);">
                    No prediction yet. Start the EEG server with its watcher and tap
                    <em>Decode my thoughts</em> to pull the latest result.
                </p>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DEMO TAB (SIMULATED EEG + EMG)
# ============================================================

def animate_epoch(epoch_pp: np.ndarray, fs: float, title: str):
    n_channels, n_samples = epoch_pp.shape

    max_duration = 2.0
    max_samples = int(fs * max_duration)
    n_samples = min(n_samples, max_samples)

    n_frames = 60
    frame_indices = np.linspace(1, n_samples, n_frames, dtype=int)

    st.markdown(f"##### {title}")
    plot_placeholder = st.empty()

    sleep_per_frame = max_duration / n_frames

    for t_idx in frame_indices:
        data = epoch_pp[:, :t_idx]
        time_axis = np.arange(t_idx) / fs

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Preprocessed EEG – eight channels")
        for ch_idx in range(n_channels):
            ax.plot(time_axis, data[ch_idx, :] + ch_idx * 5, label=f"Ch {ch_idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z-scored amplitude + offset")
        ax.legend(loc="upper right", ncol=4)
        fig.tight_layout()

        plot_placeholder.pyplot(fig)
        time.sleep(sleep_per_frame)


def show_selection_card(label: str, value: str):
    main_text = value
    freq_text = None

    if "(" in value and "Hz" in value:
        left, right = value.split("(", 1)
        main_text = left.strip()
        freq_text = right.strip(" )")

    if main_text in ("", " "):
        main_text = "SPACE"

    freq_html = ""
    if freq_text is not None:
        freq_html = f"""
            <div style="
                margin-top: 4px;
                font-family: 'Lekton', monospace;
                font-size: 0.75rem;
                letter-spacing: 0.22em;
                text-transform: uppercase;
                color: #9CA3AF;
            ">
                {freq_text}
            </div>
        """

    st.markdown(
        f"""
        <div style="
            margin-top: 18px;
            padding: 22px;
            border-radius: 24px;
            background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #020617 100%);
            text-align: center;
            border: 1px solid rgba(148,163,184,0.6);
            box-shadow: 0 18px 40px rgba(15,23,42,0.7);
        ">
            <div style="
                font-family: 'Lekton', monospace;
                color: #9CA3AF;
                font-size: 0.75rem;
                letter-spacing: 0.35em;
                text-transform: uppercase;
            ">
                {label}
            </div>
            <div style="
                font-family: 'Space Grotesk', system-ui, sans-serif;
                color: #F9FAFB;
                font-size: 4.0rem;
                font-weight: 700;
                margin-top: 8px;
                letter-spacing: 0.24em;
            ">
                {main_text}
            </div>
            {freq_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_emg_confirmation(text: str):
    st.markdown(
        f"""
        <div style="
            margin-top: 10px;
            padding: 0.7rem 1.4rem;
            border-radius: 999px;
            background: rgba(71,21,18,0.08);
            border: 1px solid rgba(71,21,18,0.35);
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            font-family: 'Lekton', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #471512;
        ">
            <span style="
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: #16A34A;
                box-shadow: 0 0 0 4px rgba(22,163,74,0.25);
            "></span>
            EMG agrees · {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _find_key_for_letter(letter: str) -> Optional[str]:
    for k, tokens in LETTER_MAP.items():
        for tok in tokens:
            if tok and tok[0] == letter:
                return k
    return None


def render_demo_tab():
    st.markdown('<div class="inkling-card">', unsafe_allow_html=True)
    st.markdown("### Demo: live EEG + EMG decoding (simulated)")

    st.markdown(
        """
This tab replays a single simulated run of the speller.

You’ll see:

- multi-channel preprocessed EEG epochs
- a proposed keypad key
- a short EMG “yes” burst
- then a refined letter choice

The decoded text builds up character by character, as it would in a real session.
"""
    )

    if st.button("Start demo run"):
        hidden_word = "PLATYPUS"
        typed = ""
        typed_placeholder = st.empty()

        for letter in hidden_word:
            key = _find_key_for_letter(letter)
            if key is None:
                continue

            epoch_key = np.random.randn(8, 500)
            animate_epoch(epoch_key, FS, "EEG window – keypad selection")
            show_selection_card("Key selected", key)
            time.sleep(0.6)
            show_emg_confirmation("key")
            st.markdown("---")

            epoch_letter = np.random.randn(8, 500)
            animate_epoch(epoch_letter, FS, "EEG window – letter selection")

            display_token = None
            for tok in LETTER_MAP.get(key, []):
                if tok and tok[0] == letter:
                    display_token = tok
                    break

            if display_token is None:
                display_token = letter

            show_selection_card("Letter selected", display_token)
            time.sleep(0.6)
            show_emg_confirmation("letter")

            typed += letter
            typed_placeholder.markdown(
                f"""
                <div style="
                    margin-top: 18px;
                    font-family: 'Lekton', monospace;
                    font-size: 0.85rem;
                    letter-spacing: 0.22em;
                    text-transform: uppercase;
                    color: #471512;
                ">
                    Decoded so far ·
                    <span style="
                        font-family: 'Space Grotesk', system-ui, sans-serif;
                        font-size: 1.6rem;
                        letter-spacing: 0.18em;
                        margin-left: 0.6rem;
                    ">{typed}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN ENTRY POINT WITH TABS
# ============================================================

def main():
    st.set_page_config(page_title="Inkling – EEG + EMG Speller", layout="wide")
    inject_css()
    init_state()
    # Animated neural backdrop
    # st.markdown('<div class="inkling-neural"></div>', unsafe_allow_html=True)
    # st.markdown('<div class="inkling-neural"></div>', unsafe_allow_html=True)
    # st.markdown('<div class="inkling-neural"></div>', unsafe_allow_html=True)

    # Hero section
    st.markdown(
        """
        <div class="inkling-hero">
            <div>
                <div class="inkling-hero-title">
                    Inkling
                </div>
                <div class="inkling-hero-kicker">
                    Hybrid EEG & EMG communication interface
                </div>
                <p class="inkling-hero-subtitle">
                    A minimal-movement speller that fuses visual brain rhythms with tiny muscle bursts,
                    designed for contexts where conventional typing and pointing are no longer possible.
                </p>
                <div class="inkling-hero-pills">
                    <div class="inkling-pill">SSVEP decoding</div>
                    <div class="inkling-pill">Intent bursts</div>
                    <div class="inkling-pill">Real-time control</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_landing, tab_how, tab_demo, tab_team = st.tabs(
        ["Landing", "How it works", "Demo: EEG + EMG flow", "Meet the team"]
    )

    with tab_landing:
        st.markdown(
            """
Inkling is an experimental interface that combines brain signals and small muscle
contractions to allow typing when standard keyboards and pointing devices are not usable.
"""
        )
        st.markdown(
            """
            <div class="inkling-splash">
                <div class="inkling-splash-title">Live session pulse</div>
                <div class="inkling-splash-body">
                    Watch the latest EEG class roll in and keep the backend running to feed this UI.
                    The layout stretches on desktop while staying touch-friendly on mobile.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_speller_ui()

    with tab_how:
        st.markdown(
            """
            <div class="inkling-splash">
                <div class="inkling-splash-title">Signal story</div>
                <div class="inkling-splash-body">
                    Explore the visual rhythms, control loop, and why EEG + EMG make a stable pair.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_inkling_explainer()

    with tab_demo:
        st.markdown(
            """
            <div class="inkling-splash">
                <div class="inkling-splash-title">Simulated run</div>
                <div class="inkling-splash-body">
                    A guided decode of "PLATYPUS" with animated epochs, highlights, and EMG confirms.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_demo_tab()

    with tab_team:
        st.markdown(
            """
            <div class="inkling-splash">
                <div class="inkling-splash-title">People behind the signals</div>
                <div class="inkling-splash-body">
                    Meet the crew, see the roles, and pick up contacts for collaborations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_meet_the_team()


if __name__ == "__main__":
    main()
