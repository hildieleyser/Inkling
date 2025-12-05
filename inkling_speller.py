"""
Simple controller to fuse two-stage SSVEP target predictions with EMG confirmations
and build text for a typing interface.

The flow matches the described UX:
1) EEG stage 1 picks one of 12 targets (e.g., a macro key that opens a 3-letter panel).
2) EMG confirms or rejects that target.
3) EEG stage 2 picks one of 3 letters within the chosen panel.
4) EMG confirms or rejects the letter; on confirmation, the letter is appended to the text buffer.

The controller below does not run models itself; it expects upstream SSVEP/EMG
models to provide probabilities. It keeps a small state machine that you can
poll from your acquisition loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_LAYOUT: Tuple[Tuple[str, ...], ...] = (
    ("A", "B", "C"),
    ("D", "E", "F"),
    ("G", "H", "I"),
    ("J", "K", "L"),
    ("M", "N", "O"),
    ("P", "Q", "R"),
    ("S", "T", "U"),
    ("V", "W", "X"),
    ("Y", "Z", " "),
    ("BACKSPACE", ".", ","),
    ("?", "!", "'"),
    ("ENTER", "-", "_"),
)  # 12 panels, 3 tokens each


@dataclass
class Candidate:
    target_index: int
    confidence: float
    probabilities: Sequence[float] = field(repr=False)
    stage: int = 1  # 1 or 2


class TwoStageSpeller:
    """
    Two-stage speller controller driven by SSVEP predictions and EMG confirmations.

    Usage pattern:
        speller = TwoStageSpeller()
        speller.propose_stage1(eeg_probs_stage1)
        speller.handle_emg(emg_yes_prob)  # confirm or reject stage 1
        speller.propose_stage2(eeg_probs_stage2)
        speller.handle_emg(emg_yes_prob)  # confirm or reject letter
        typed_text = speller.text
    """

    def __init__(
        self,
        layout: Sequence[Sequence[str]] = DEFAULT_LAYOUT,
        stage1_threshold: float = 0.45,
        stage2_threshold: float = 0.45,
        emg_yes_threshold: float = 0.6,
        emg_no_threshold: float = 0.35,
    ) -> None:
        self._validate_layout(layout)
        self.layout: Tuple[Tuple[str, ...], ...] = tuple(tuple(row) for row in layout)
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.emg_yes_threshold = emg_yes_threshold
        self.emg_no_threshold = emg_no_threshold

        self.text: str = ""
        self.stage1_candidate: Optional[Candidate] = None
        self.stage2_candidate: Optional[Candidate] = None
        self.waiting_for_emg_stage: Optional[int] = None  # None, 1, or 2

    @staticmethod
    def _validate_layout(layout: Sequence[Sequence[str]]) -> None:
        if len(layout) != 12:
            raise ValueError("layout must contain exactly 12 panels")
        for row in layout:
            if len(row) != 3:
                raise ValueError("each panel must contain exactly 3 tokens")

    @staticmethod
    def _top_choice(probs: Sequence[float]) -> Tuple[int, float]:
        if len(probs) == 0:
            raise ValueError("probability array cannot be empty")
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return int(best_idx), float(probs[best_idx])

    def propose_stage1(self, probs: Sequence[float]) -> Optional[Candidate]:
        """Register a stage 1 EEG proposal. Returns the candidate if it passes threshold."""
        idx, conf = self._top_choice(probs)
        if conf < self.stage1_threshold:
            return None

        self.stage1_candidate = Candidate(target_index=idx, confidence=conf, probabilities=probs, stage=1)
        self.waiting_for_emg_stage = 1
        return self.stage1_candidate

    def propose_stage2(self, probs: Sequence[float]) -> Optional[Candidate]:
        """Register a stage 2 EEG proposal (only valid after stage 1 confirmed)."""
        if self.stage1_candidate is None:
            raise RuntimeError("stage 1 must be confirmed before proposing stage 2")

        idx, conf = self._top_choice(probs)
        if conf < self.stage2_threshold:
            return None

        self.stage2_candidate = Candidate(target_index=idx, confidence=conf, probabilities=probs, stage=2)
        self.waiting_for_emg_stage = 2
        return self.stage2_candidate

    def handle_emg(self, emg_yes_prob: float, emg_no_prob: Optional[float] = None) -> str:
        """
        Interpret an EMG model output and update state.

        Returns one of: "pending" (no decision), "rejected", "accepted_stage1",
        "accepted_letter".
        """
        if self.waiting_for_emg_stage is None:
            return "pending"

        no_score = emg_no_prob if emg_no_prob is not None else 1.0 - emg_yes_prob
        if emg_yes_prob >= self.emg_yes_threshold:
            decision = "yes"
        elif no_score >= self.emg_no_threshold:
            decision = "no"
        else:
            return "pending"

        if self.waiting_for_emg_stage == 1:
            return self._handle_stage1_emg(decision)
        return self._handle_stage2_emg(decision)

    def _handle_stage1_emg(self, decision: str) -> str:
        if decision == "yes" and self.stage1_candidate:
            self.waiting_for_emg_stage = None
            return "accepted_stage1"

        self._reset_stage1()
        return "rejected"

    def _handle_stage2_emg(self, decision: str) -> str:
        if decision == "yes" and self.stage1_candidate and self.stage2_candidate:
            token = self.layout[self.stage1_candidate.target_index][self.stage2_candidate.target_index]
            self._apply_token(token)
            self._reset_stage2(drop_stage1=True)
            return "accepted_letter"

        self._reset_stage2()
        return "rejected"

    def _reset_stage1(self) -> None:
        self.stage1_candidate = None
        self.waiting_for_emg_stage = None
        self.stage2_candidate = None

    def _reset_stage2(self, drop_stage1: bool = False) -> None:
        self.stage2_candidate = None
        self.waiting_for_emg_stage = None
        if drop_stage1:
            self.stage1_candidate = None

    def _apply_token(self, token: str) -> None:
        if token == "BACKSPACE":
            self.text = self.text[:-1]
        elif token == "ENTER":
            self.text += "\n"
        else:
            self.text += token

    def status(self) -> Dict[str, Optional[str]]:
        """Return a snapshot for UI/debugging."""
        return {
            "text": self.text,
            "stage1_candidate": self._candidate_str(self.stage1_candidate),
            "stage2_candidate": self._candidate_str(self.stage2_candidate),
            "waiting_for_emg": self.waiting_for_emg_stage,
        }

    @staticmethod
    def _candidate_str(candidate: Optional[Candidate]) -> Optional[str]:
        if candidate is None:
            return None
        return f"stage{candidate.stage}:idx{candidate.target_index}@{candidate.confidence:.2f}"


def demo_run() -> None:
    """Tiny console demo using mock probabilities."""
    speller = TwoStageSpeller()
    print("[demo] starting text buffer is empty")

    # Stage 1: 12 targets, choose index 2 with 0.71 prob
    speller.propose_stage1([0.02, 0.03, 0.71, 0.04, 0.05, 0.03, 0.02, 0.03, 0.03, 0.02, 0.01, 0.01])
    speller.handle_emg(emg_yes_prob=0.9)  # confirm panel 3 (G/H/I)

    # Stage 2: 3 targets within panel 3, choose index 1 (H) with 0.8 prob
    speller.propose_stage2([0.1, 0.8, 0.1])
    speller.handle_emg(emg_yes_prob=0.85)  # confirm letter H

    # Another letter: pick panel 9 (Y/Z/space) and choose space
    speller.propose_stage1([0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.75, 0.04, 0.02, 0.01])
    speller.handle_emg(emg_yes_prob=0.88)
    speller.propose_stage2([0.2, 0.2, 0.6])
    speller.handle_emg(emg_yes_prob=0.92)

    print(f"[demo] final text buffer: {repr(speller.text)}")


if __name__ == "__main__":
    demo_run()
