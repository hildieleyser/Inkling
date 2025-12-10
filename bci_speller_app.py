# bci_speller_streamlit.py
# Streamlit UI for Inkling: hybrid EEG + EMG speller

import streamlit as st
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================
# GLOBAL CONFIG AND STYLING
# ============================================================

EEG_API_URL = "http://localhost:8000/predict"
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
    /* ------------------------------------------------------------
       FONT IMPORTS + STACKS
       ------------------------------------------------------------ */

    /* Lekton (labels / technical) */
    @import url('https://fonts.googleapis.com/css2?family=Lekton:wght@400;700&display=swap');
    /* Aileron (main sans – closest web analogue) */
    @import url('https://fonts.googleapis.com/css2?family=Aileron:wght@300;400;500;600;800&display=swap');
    /* Script fallback for Bourgiono Rastelenio */
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    :root {
        /* Refined palette */
        --inkling-bg-cream: #F4EBDD;
        --inkling-bg-cream-soft: #F7F0E4;
        --inkling-panel: #FBF7EF;
        --inkling-ink: #261612;
        --inkling-blue: #143D7A;
        --inkling-blue-soft: #2458A7;
        --inkling-oxblood: #471512;
        --inkling-oxblood-soft: #6B2620;
        --inkling-line-soft: rgba(38,22,18,0.10);
        --inkling-line-strong: rgba(38,22,18,0.22);
        --inkling-pill-bg: rgba(244,235,221,0.08);
    }

    /* ------------------------------------------------------------
       GLOBAL LAYOUT & BACKGROUND
       ------------------------------------------------------------ */

    body {
        margin: 0;
        padding: 0;
        background:
            radial-gradient(circle at 0% 0%, rgba(255,255,255,0.45), transparent 55%),
            radial-gradient(circle at 100% 100%, rgba(255,255,255,0.35), transparent 55%),
            linear-gradient(135deg, #F4EBDD 0%, #EFE3D4 45%, #F6EFE5 100%);
        color: var(--inkling-ink);
        font-family: "Aileron", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    /* Main Streamlit content wrapper */
    .block-container {
        max-width: 980px;
        padding-top: 1.75rem;
        padding-bottom: 3rem;
    }

    /* ------------------------------------------------------------
       TYPOGRAPHY
       ------------------------------------------------------------ */

    h1, h2, h3, h4 {
        font-family: "Heading now 61-68", "Aileron", system-ui, sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 800;
        color: var(--inkling-oxblood) !important;
        margin-bottom: 0.2rem;
    }

    h1 {
        font-size: clamp(3.4rem, 6.4vw, 4.6rem);  /* ~61–68 px range */
        line-height: 1.02;
    }

    h2 {
        font-size: clamp(2.2rem, 4.4vw, 3rem);
        line-height: 1.06;
    }

    h3, h4 {
        font-size: 1.2rem;
        line-height: 1.12;
    }

    p, li {
        font-family: "Aileron", system-ui, sans-serif;
        font-size: 0.95rem;
        line-height: 1.65;
        color: var(--inkling-ink);
    }

    .section-label {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.3em;
        color: var(--inkling-blue-soft);
        font-size: 0.7rem;
        margin-bottom: -0.1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
    }

    .section-label::before {
        content: "";
        display: inline-block;
        width: 26px;
        height: 1px;
        background: var(--inkling-blue-soft);
        opacity: 0.7;
    }

    .inkling-script {
        font-family: "Bourgiono Rastelenio", "Pacifico", cursive;
        font-size: 1.7rem;
        letter-spacing: 0.06em;
        color: var(--inkling-oxblood-soft);
    }

    /* ------------------------------------------------------------
       TABS
       ------------------------------------------------------------ */

    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
        border-bottom: 1px solid var(--inkling-line-soft);
        margin-top: 0.8rem;
        padding-bottom: 0.1rem;
    }

    .stTabs [data-baseweb="tab"] {
        position: relative;
        color: rgba(38,22,18,0.7) !important;
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.28em;
        padding-bottom: 0.7rem;
        transition: color 160ms ease-out;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--inkling-oxblood-soft) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--inkling-blue-soft) !important;
    }

    .stTabs [aria-selected="true"]::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0.1rem;
        width: 56%;
        height: 2px;
        background: linear-gradient(90deg, var(--inkling-blue-soft), var(--inkling-oxblood-soft));
    }

    /* ------------------------------------------------------------
       HERO BAND
       ------------------------------------------------------------ */

    .inkling-hero {
        margin-top: 0.5rem;
        margin-bottom: 2.3rem;
        padding: 1.9rem 2.2rem;
        background: radial-gradient(circle at 0% 0%, rgba(244,235,221,0.08), transparent 55%),
                    radial-gradient(circle at 100% 100%, rgba(255,255,255,0.18), transparent 55%),
                    linear-gradient(120deg, #163D7B 0%, #12315F 55%, #192442 100%);
        color: #FDF8EE;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 26px 50px rgba(4,6,18,0.55);
        display: grid;
        grid-template-columns: minmax(0, 2.3fr) minmax(0, 1.4fr);
        gap: 1.8rem;
        align-items: center;
    }

    @media (max-width: 900px) {
        .inkling-hero {
            grid-template-columns: 1fr;
            padding: 1.6rem 1.5rem;
        }
    }

    .inkling-hero-kicker {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.35em;
        font-size: 0.68rem;
        color: rgba(250,245,234,0.9);
        margin-bottom: 0.55rem;
        opacity: 0.9;
    }

    .inkling-hero-title {
        font-family: "Heading now 61-68", "Aileron", system-ui, sans-serif;
        text-transform: uppercase;
        font-weight: 800;
        font-size: clamp(3.5rem, 6.6vw, 4.9rem);
        letter-spacing: 0.22em;
        color: #FDF8EE;
        margin-bottom: 0.3rem;
    }

    .inkling-hero-subtitle {
        font-family: "Aileron", system-ui, sans-serif;
        font-size: 0.98rem;
        line-height: 1.7;
        color: rgba(252,248,240,0.9);
        max-width: 560px;
        margin-bottom: 0.9rem;
    }

    .inkling-hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.35rem;
    }

    .inkling-pill {
        font-family: "Lekton", monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(250,245,234,0.6);
        background: var(--inkling-pill-bg);
        color: rgba(250,245,234,0.92);
        backdrop-filter: blur(4px);
    }

    .inkling-hero::after {
        content: "BCI Speller";
        justify-self: flex-end;
        align-self: flex-start;
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.3em;
        font-size: 0.62rem;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(250,245,234,0.3);
        background: rgba(3,6,20,0.4);
        color: rgba(250,245,234,0.9);
    }

    /* ------------------------------------------------------------
       CARDS (PANELS & TEAM)
       ------------------------------------------------------------ */

    .inkling-card,
    .inkling-team-card {
        background: var(--inkling-panel);
        border-radius: 18px;
        padding: 1.7rem 2rem;
        border: 1px solid var(--inkling-line-soft);
        box-shadow:
            0 10px 26px rgba(0,0,0,0.06),
            0 1px 0 rgba(255,255,255,0.9) inset;
        margin-top: 1.3rem;
        transition:
            transform 200ms ease-out,
            box-shadow 200ms ease-out,
            border-color 200ms ease-out,
            background-color 200ms ease-out;
    }

    .inkling-card:hover,
    .inkling-team-card:hover {
        transform: translateY(-3px);
        box-shadow:
            0 16px 40px rgba(0,0,0,0.16),
            0 1px 0 rgba(255,255,255,0.9) inset;
        border-color: var(--inkling-line-strong);
        background-color: var(--inkling-bg-cream-soft);
    }

    .inkling-team-role {
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.3em;
        font-size: 0.72rem;
        color: var(--inkling-blue-soft);
        margin-bottom: 0.25rem;
    }

    .inkling-team-name {
        font-family: "Heading now 61-68", "Aileron", system-ui, sans-serif;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.18em;
        font-size: 1.25rem;
        margin-bottom: 0.2rem;
        color: var(--inkling-oxblood);
    }

    .inkling-team-contact {
        font-family: "Lekton", monospace;
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(38,22,18,0.7);
        margin-bottom: 0.7rem;
    }

    .inkling-team-body {
        font-family: "Aileron", system-ui, sans-serif;
        font-size: 0.93rem;
        line-height: 1.7;
        color: var(--inkling-ink);
    }

    /* ------------------------------------------------------------
       BUTTONS
       ------------------------------------------------------------ */

    .stButton>button {
        background: linear-gradient(120deg, var(--inkling-blue) 0%, var(--inkling-blue-soft) 50%, var(--inkling-oxblood-soft) 100%) !important;
        color: #FDF8EE !important;
        border-radius: 999px;
        padding: 0.45rem 1.7rem;
        font-family: "Lekton", monospace;
        text-transform: uppercase;
        letter-spacing: 0.28em;
        font-size: 0.72rem;
        border: none;
        box-shadow: 0 10px 22px rgba(0,0,0,0.25);
        transition:
            transform 120ms ease-out,
            box-shadow 120ms ease-out,
            filter 120ms ease-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 16px 30px rgba(0,0,0,0.32);
    }

    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.28);
    }

    /* ------------------------------------------------------------
       TEXT INPUTS / TEXT AREAS
       ------------------------------------------------------------ */

    textarea,
    .stTextInput>div>div>input {
        background: var(--inkling-bg-cream-soft) !important;
        border-radius: 14px !important;
        border: 1px solid var(--inkling-line-soft) !important;
        color: var(--inkling-ink) !important;
        font-family: "Aileron", system-ui, sans-serif !important;
        font-size: 0.9rem !important;
    }

    textarea:focus,
    .stTextInput>div>div>input:focus {
        outline: none !important;
        border-color: var(--inkling-blue-soft) !important;
        box-shadow: 0 0 0 1px rgba(36,88,167,0.35) !important;
    }

    /* ------------------------------------------------------------
       INFO / STATUS ELEMENTS
       ------------------------------------------------------------ */

    .stAlert {
        border-radius: 14px !important;
        border: 1px solid var(--inkling-line-soft) !important;
    }

    .stAlert>div {
        background-color: var(--inkling-bg-cream-soft) !important;
        color: var(--inkling-oxblood-soft) !important;
        font-family: "Aileron", system-ui, sans-serif;
        font-size: 0.9rem;
    }

    hr {
        border: none;
        border-top: 1px solid var(--inkling-line-soft);
        margin: 1.1rem 0;
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

            with st.container():
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

            with st.container():
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
   confirm → append letter to text
   delete → remove the previous letter
   cancel → return to block selection
"""
        )

    with col8:
        st.markdown(
            """
Simplified logic:

while typing:
    eeg_choice = argmax p(option | EEG)
    if confidence is high:
        highlight(eeg_choice)

    emg = classify(EMG)
    if emg == CONFIRM:
        commit(eeg_choice)
    elif emg == DELETE:
        delete_last()
    elif emg == CANCEL:
        go_back()

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
# KEYPAD + LETTER MAPPING
# ============================================================

# Stage 1: EEG classes 0–11 map to these 12 keypad entries
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

# Stage 2: local class indices (0..N-1) per key
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
        return ["0", " ", "<BACK>"]

    letters = LETTER_MAP.get(key, [])
    if letters:
        return [key] + letters

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


def sidebar_inputs():
    st.sidebar.header("EEG and EMG inputs")

    eeg_file = st.sidebar.file_uploader("EEG .mat file", type=["mat"])
    if eeg_file is not None:
        st.session_state.eeg_file_bytes = eeg_file.read()

    emg_file = st.sidebar.file_uploader("EMG .hdf5 file", type=["hdf5"])
    if emg_file is not None:
        st.session_state.emg_file_bytes = emg_file.read()

    st.sidebar.markdown("EEG epoch indices")
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
# SPELLER UI (LANDING TAB)
# ============================================================

def render_speller_ui():
    params = sidebar_inputs()

    st.markdown('<div class="inkling-card">', unsafe_allow_html=True)
    st.markdown("### Run the speller")
    show_typed_text()
    st.write("---")

    st.write(f"State: {st.session_state.phase}")
    if st.session_state.last_message:
        st.info(st.session_state.last_message)

    # idle
    if st.session_state.phase == "idle":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start next character (EEG)"):
                if st.session_state.eeg_file_bytes is None:
                    st.error("Upload an EEG .mat file first.")
                else:
                    st.session_state.phase = "key_eeg"
                    st.session_state.last_message = "Running EEG model to select a key."
                    rerun()
        with col2:
            if st.button("Clear text"):
                st.session_state.typed_text = ""
                reset_for_next_char()
                rerun()

    # key EEG
    if st.session_state.phase == "key_eeg":
        if st.button("Run EEG to predict key"):
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
                st.session_state.last_message = (
                    f"EEG suggests key {key}. Awaiting EMG confirmation."
                )
                rerun()

    if st.session_state.current_key is not None:
        st.markdown(f"Current key candidate: {st.session_state.current_key}")

    # key EMG
    if st.session_state.phase == "key_emg":
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("EMG confirm key"):
                if st.session_state.emg_file_bytes is None:
                    st.error("Upload an EMG .hdf5 file first.")
                else:
                    emg_result = call_emg_api(
                        st.session_state.emg_file_bytes,
                        dataset_name=params["dataset"],
                    )
                    if emg_result is None:
                        st.stop()

                    key = st.session_state.current_key

                    if emg_result == 1:
                        st.session_state.last_message = f"EMG confirmed key {key}."

                        letters = LETTER_MAP.get(key, [])
                        if (not letters and key != "0") or key in ["1", "*", "#"]:
                            st.session_state.typed_text += key
                            reset_for_next_char()
                        else:
                            panel = build_letter_panel(key)
                            st.session_state.current_panel_items = panel
                            st.session_state.phase = "char_eeg"
                            st.session_state.last_message = (
                                f"Key {key} confirmed. EEG now selecting from: "
                                + ", ".join(pretty_item(x) for x in panel)
                            )
                    else:
                        st.session_state.last_message = "EMG rejected key. Restarting."
                        reset_for_next_char()

                    rerun()

        with col2:
            if st.button("Reject and restart key"):
                reset_for_next_char()
                rerun()

        with col3:
            if st.button("Cancel"):
                reset_for_next_char()
                rerun()

    if st.session_state.current_panel_items is not None:
        st.markdown("Letter panel:")
        st.write(", ".join(pretty_item(x) for x in st.session_state.current_panel_items))

    # char EEG
    if st.session_state.phase == "char_eeg":
        if st.button("Run EEG to predict letter"):
            if st.session_state.eeg_file_bytes is None:
                st.error("No EEG file uploaded.")
                st.stop()

            panel = st.session_state.current_panel_items
            key = st.session_state.current_key

            if panel is None or key is None:
                st.error("Panel or key missing.")
                st.stop()

            pred_class = call_eeg_api(
                st.session_state.eeg_file_bytes,
                params["block_idx"],
                params["trial_idx"],
                params["target_idx"],
            )
            if pred_class is None:
                st.stop()

            allowed_ids = LETTER_CLASS_IDS.get(key, [])
            if not allowed_ids:
                st.error(f"No LETTER_CLASS_IDS defined for key {key}.")
                st.stop()

            if pred_class not in allowed_ids:
                st.error(
                    f"EEG predicted local class {pred_class}, "
                    f"which is not valid for key {key} (allowed {allowed_ids})."
                )
                st.stop()

            if pred_class >= len(panel):
                st.error(
                    f"EEG predicted local class {pred_class}, "
                    f"but panel only has {len(panel)} options."
                )
                st.stop()

            candidate = panel[pred_class]

            st.session_state.current_candidate_char = candidate
            st.session_state.phase = "char_emg"
            st.session_state.last_message = (
                f"EEG suggests {pretty_item(candidate)}. Awaiting EMG confirmation."
            )
            rerun()

    if st.session_state.current_candidate_char is not None:
        st.markdown(
            f"Character candidate: {pretty_item(st.session_state.current_candidate_char)}"
        )

    # char EMG
    if st.session_state.phase == "char_emg":
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("EMG confirm letter"):
                if st.session_state.emg_file_bytes is None:
                    st.error("Upload an EMG .hdf5 file first.")
                else:
                    emg_result = call_emg_api(
                        st.session_state.emg_file_bytes,
                        dataset_name=params["dataset"],
                    )
                    if emg_result is None:
                        st.stop()

                    candidate = st.session_state.current_candidate_char

                    if emg_result == 1:
                        if candidate == "<BACK>":
                            st.session_state.typed_text = (
                                st.session_state.typed_text[:-1]
                            )
                        else:
                            st.session_state.typed_text += candidate

                        st.session_state.last_message = (
                            f"EMG confirmed {pretty_item(candidate)}"
                        )
                        reset_for_next_char()

                    else:
                        st.session_state.current_candidate_char = None
                        st.session_state.phase = "char_eeg"
                        st.session_state.last_message = (
                            "EMG rejected letter. Re-running EEG for this key."
                        )

                    rerun()

        with col2:
            if st.button("Cancel letter"):
                reset_for_next_char()
                rerun()

        with col3:
            if st.button("Clear all text"):
                st.session_state.typed_text = ""
                reset_for_next_char()
                rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# COOL VIDEO ANIMATION TAB – EEG + EMG FLOW (DUMMY)
# ============================================================

def animate_epoch(epoch_pp: np.ndarray, fs: float, title: str):
    """
    Animate preprocessed EEG (all channels) over ~5 seconds.
    """
    n_channels, n_samples = epoch_pp.shape
    n_frames = min(500, n_samples)

    st.markdown(f"##### {title}")
    plot_placeholder = st.empty()

    for t_idx in range(1, n_frames + 1):
        data = epoch_pp[:, :t_idx]
        time_axis = np.arange(t_idx) / fs

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Preprocessed EEG – 8 nodes")
        for ch_idx in range(n_channels):
            ax.plot(time_axis, data[ch_idx, :] + ch_idx * 5, label=f"Ch {ch_idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z-scored amplitude + offset")
        ax.legend(loc="upper right", ncol=4)
        fig.tight_layout()

        plot_placeholder.pyplot(fig)
        time.sleep(0.01)


def show_selection_card(label: str, value: str):
    """
    Show a big branded card displaying the selected key/letter.
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
                font-family: 'Heading now 61-68', 'Aileron', system-ui, sans-serif;
                color: #F9FAFB;
                font-size: 4.2rem;
                font-weight: 800;
                margin-top: 8px;
                letter-spacing: 0.24em;
            ">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_emg_confirmation(text: str):
    """
    Show an EMG confirmation chip/card.
    """
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


def render_demo_tab():
    """
    Sequential animation with dummy data:
    1. EEG for key '4' (dummy), then show 4
    2. EMG confirms
    3. EEG for letter 'H' (dummy), then show H
    4. EMG confirms
    """
    st.markdown('<div class="inkling-card">', unsafe_allow_html=True)
    st.markdown("### Demo: EEG + EMG flow (dummy data)")

    st.markdown(
        """
This sequence shows the *intended* interaction visually:

1. Preprocessed EEG for the **number 4** on the Nokia keypad
2. EMG confirmation that the key selection is correct
3. Preprocessed EEG for the **letter H** within that key
4. EMG confirmation that the letter selection is correct
"""
    )

    if st.button("Play demo sequence"):
        epoch_key4 = np.random.randn(8, 500)
        animate_epoch(epoch_key4, FS, "EEG · selecting keypad number 4")
        show_selection_card("Key selected", "4")
        time.sleep(1.0)
        show_emg_confirmation("key 4")

        st.markdown("---")

        epoch_H = np.random.randn(8, 500)
        animate_epoch(epoch_H, FS, "EEG · selecting letter H")
        show_selection_card("Letter selected", "H")
        time.sleep(1.0)
        show_emg_confirmation("letter H")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN ENTRY POINT WITH TABS
# ============================================================

def main():
    st.set_page_config(page_title="Inkling – EEG + EMG Speller", layout="centered")
    inject_css()
    init_state()

    # Hero section
    st.markdown(
        """
        <div class="inkling-hero">
            <div class="inkling-hero-kicker">
                Hybrid EEG & EMG communication interface
            </div>
            <div class="inkling-hero-title">
                Inkling
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
        render_speller_ui()

    with tab_how:
        render_inkling_explainer()

    with tab_demo:
        render_demo_tab()

    with tab_team:
        render_meet_the_team()


if __name__ == "__main__":
    main()

