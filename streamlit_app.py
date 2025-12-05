"""
Interactive Streamlit UI for the two-stage EEG+EMG speller.

This demo wraps the TwoStageSpeller state machine with a simple, attractive UI
so you can feed model outputs (or simulated probabilities) and watch text build
up in real time.
"""

from __future__ import annotations

from typing import List, Sequence

import streamlit as st

from inkling_speller import DEFAULT_LAYOUT, TwoStageSpeller


st.set_page_config(
    page_title="Inkling Speller",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helpers -----------------------------------------------------------------

def make_probs(n: int, idx: int, confidence: float) -> List[float]:
    """Construct a probability array with one dominant target."""
    if not 0 <= idx < n:
        raise ValueError("idx out of range")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be within [0, 1]")
    if n == 1:
        return [1.0]
    remaining = max(0.0, 1.0 - confidence)
    filler = remaining / (n - 1)
    return [confidence if i == idx else filler for i in range(n)]


def ensure_session_state() -> TwoStageSpeller:
    """Initialize or fetch the speller in Streamlit session state."""
    if "speller" not in st.session_state:
        st.session_state.speller = TwoStageSpeller()
    return st.session_state.speller


def reset_session() -> None:
    st.session_state.speller = TwoStageSpeller()
    st.session_state.stage1_probs = [1 / 12] * 12
    st.session_state.stage2_probs = [1 / 3] * 3


# --- Styling -----------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
        --ink-bg: radial-gradient(1200px at 20% 20%, rgba(80,120,255,0.15), transparent),
                   radial-gradient(1000px at 80% 0%, rgba(255,120,200,0.12), transparent),
                   linear-gradient(135deg, #0f172a, #0b1023 45%, #0c1227);
        --ink-card: rgba(255, 255, 255, 0.05);
        --ink-accent: #7dd3fc;
        --ink-yes: #34d399;
        --ink-no: #f87171;
        --ink-text: #e2e8f0;
        --ink-muted: #94a3b8;
        --ink-border: rgba(255,255,255,0.08);
    }
    .stApp {
        background: var(--ink-bg);
        color: var(--ink-text);
        font-family: 'Space Grotesk', 'Helvetica Neue', 'Avenir Next', sans-serif;
    }
    .ink-card {
        background: var(--ink-card);
        border: 1px solid var(--ink-border);
        padding: 1rem 1.2rem;
        border-radius: 16px;
        backdrop-filter: blur(6px);
    }
    .ink-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
    }
    .ink-panel {
        border: 1px solid var(--ink-border);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        background: rgba(255,255,255,0.02);
    }
    .ink-panel h4 {
        margin: 0;
        font-size: 15px;
        letter-spacing: 0.04em;
        color: var(--ink-accent);
    }
    .ink-panel .tokens {
        margin-top: 6px;
        font-size: 13px;
        color: var(--ink-muted);
    }
    .ink-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        background: var(--ink-card);
        border: 1px solid var(--ink-border);
    }
    .ink-textarea {
        width: 100%;
        min-height: 140px;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid var(--ink-border);
        background: rgba(0,0,0,0.25);
        color: var(--ink-text);
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- UI ----------------------------------------------------------------------

speller = ensure_session_state()

header_col, status_col = st.columns([3, 2], gap="large")
with header_col:
    st.markdown(
        "<h1 style='margin-bottom:0.2em'>Inkling Speller</h1>"
        "<p style='color:#cbd5e1;margin-top:0'>Two-stage SSVEP + EMG fusion for hands-free typing.</p>",
        unsafe_allow_html=True,
    )
with status_col:
    st.button("Reset session", on_click=reset_session)

typed_col, control_col = st.columns([2, 3], gap="large")

with typed_col:
    st.markdown("#### Text buffer")
    placeholder = (
        speller.text if speller.text else '<span style="color:#64748b">Start selecting to type…</span>'
    )
    st.markdown(
        f"<div class='ink-card'><div class='ink-textarea'>{placeholder}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Layout map")
    grid = ""
    for idx, panel in enumerate(DEFAULT_LAYOUT):
        tokens = " • ".join(panel)
        grid += f"<div class='ink-panel'><h4>Panel {idx + 1:02d}</h4><div class='tokens'>{tokens}</div></div>"
    st.markdown(f"<div class='ink-card'><div class='ink-grid'>{grid}</div></div>", unsafe_allow_html=True)

with control_col:
    st.markdown("#### Stage 1 · 12-target EEG")
    s1_col_a, s1_col_b = st.columns([2, 1])
    with s1_col_a:
        s1_choice = st.selectbox(
            "EEG top target (panel)",
            options=list(range(12)),
            format_func=lambda i: f"Panel {i + 1:02d}",
        )
        s1_conf = st.slider("EEG confidence", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
    with s1_col_b:
        st.write("")
        if st.button("Propose stage 1", use_container_width=True):
            probs = make_probs(12, s1_choice, s1_conf)
            speller.propose_stage1(probs)
            st.success(f"Stage 1 proposed: Panel {s1_choice + 1} @ {s1_conf:.2f}")

    st.markdown("#### EMG confirmation")
    emg_yes = st.slider("EMG yes probability", 0.0, 1.0, 0.82, 0.01)
    emg_no = st.slider("EMG no probability (optional)", 0.0, 1.0, 0.18, 0.01)
    if st.button("Handle EMG", use_container_width=True):
        decision = speller.handle_emg(emg_yes, emg_no)
        st.info(f"EMG decision: {decision}")

    if speller.stage1_candidate:
        panel_idx = speller.stage1_candidate.target_index
        panel_tokens: Sequence[str] = DEFAULT_LAYOUT[panel_idx]
        st.markdown(f"#### Stage 2 · 3-target EEG (Panel {panel_idx + 1:02d})")
        s2_choice = st.selectbox(
            "EEG top token",
            options=list(range(3)),
            format_func=lambda i: f"{panel_tokens[i]} (slot {i + 1})",
            key="s2_choice",
        )
        s2_conf = st.slider("EEG confidence (stage 2)", 0.1, 0.95, 0.75, 0.05, key="s2_conf")
        if st.button("Propose stage 2", use_container_width=True):
            probs = make_probs(3, s2_choice, s2_conf)
            speller.propose_stage2(probs)
            st.success(f"Stage 2 proposed: {panel_tokens[s2_choice]} @ {s2_conf:.2f}")
    else:
        st.markdown(
            "<div class='ink-card' style='margin-top:14px; color:#94a3b8;'>"
            "Propose and confirm a stage 1 panel to unlock stage 2 targets."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Live state")
    status = speller.status()
    st.markdown(
        f"""
        <div class='ink-card' style='display:flex;flex-direction:column;gap:6px;'>
            <span class='ink-badge'>waiting_for_emg: {status["waiting_for_emg"]}</span>
            <span class='ink-badge'>stage1_candidate: {status["stage1_candidate"]}</span>
            <span class='ink-badge'>stage2_candidate: {status["stage2_candidate"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.sidebar.markdown("### Quick demo")
if st.sidebar.button("Auto-play a letter"):
    # Simulate a deterministic path: panel 3 -> token 1 (letter H)
    speller.propose_stage1(make_probs(12, 2, 0.72))
    speller.handle_emg(0.91)
    speller.propose_stage2(make_probs(3, 1, 0.8))
    speller.handle_emg(0.9)
    st.sidebar.success("Added a letter via simulated EEG+EMG.")

st.sidebar.caption("Tip: feed your real model probabilities into the Stage 1/2 controls and EMG handler to mimic live usage.")
