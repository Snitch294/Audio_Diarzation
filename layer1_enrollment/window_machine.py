"""
SPOVNOB — Module 2 (Layer 1): window_machine.py
================================================

The E_window capture state machine: a pure, deterministic FSM over
per-frame observations. No models, no I/O — fully stdlib-self-testable.

States: IDLE -> ACTIVE -> PLOSIVE_BUFFER -> (ACTIVE | emit window).

Implements: Audio_Diarization.md — "`E_window` — definition, capture,
and parameters" (start/stop policy, plosive buffer, Early Stop Rule,
no arbitrary truncation, head-yaw suspension) and the Layer 1
pseudocode "E_window capture loop (frame-driven)".

Rules encoded here, exactly:
  - START: state IDLE, target present, MAR not suspended, smoothed MAR
    (5-frame causal EMA, pre-seeded) > MAR_on, AND Silero speech within
    +/- vad_tol of the frame PTS  ->  T_start = frame PTS.
  - ACTIVE -> PLOSIVE: smoothed MAR < MAR_off starts the plosive timer
    (deadline = pts + plosive_ms). Target DISAPPEARANCE while ACTIVE is
    treated the same way; the EMA is decayed to None for the duration of
    the absence (NOT frozen) so a stale pre-absence value cannot drive a
    spurious resume/restart when the target returns — it reseeds to the
    returning frame's raw MAR (review-flagged decision: the document does
    not specify the absence case; the plosive-buffer semantics are the
    conservative deterministic choice).
  - PLOSIVE -> ACTIVE: target's smoothed MAR rises above MAR_on before
    the deadline (buffer cancelled; window continues).
  - Early Stop Rule: while the buffer runs, if the interviewer is
    visible and the interviewer's smoothed MAR rises above MAR_on, the
    window ends IMMEDIATELY at the current PTS (buffer discarded).
  - Clean expiry: T_stop = the timer's deadline PTS (not the frame PTS
    at which expiry is noticed).
  - Yaw suspension: while suspended, the EMA is frozen, no transitions
    fire, and a running plosive deadline is EXTENDED by the suspended
    wall time (deadline += pts - prev_pts per suspended frame), i.e.
    the timer pauses while the head is turned (review-flagged).
  - End of video: an open window is finalized at the last frame PTS.
  - EMAs persist across windows (they smooth a physical signal, not
    window state), but each decays to None while its subject is ABSENT
    from frame and reseeds on return, so an absence gap never leaves a
    stale value that could mis-fire a transition on re-entry. (Yaw
    suspension still FREEZES the target EMA: a brief head-turn must not
    lose the smoothed value, whereas a true disappearance must.)

All PTS values are integer milliseconds (Rule 6).
CUDA determinism dependencies: none.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .geometry import CausalEMA
from .params import EnrollmentParams

IDLE = "IDLE"
ACTIVE = "ACTIVE"
PLOSIVE = "PLOSIVE_BUFFER"

END_PLOSIVE_EXPIRY = "plosive_expiry"
END_INTERVIEWER_INTERJECTION = "interviewer_interjection"
END_OF_VIDEO = "end_of_video"


@dataclass
class FrameObs:
    """One frame's observation, as seen by the machine. Built by the
    orchestrator from vision output + Silero segment lookups."""

    pts_ms: int
    target_present: bool
    target_mar: Optional[float]          # raw MAR; None if not computable
    target_suspended: bool               # head yaw > limit (geometry rule)
    interviewer_present: bool
    interviewer_mar: Optional[float]
    vad_speech: bool                     # Silero speech within +/- vad_tol


@dataclass
class CandidateWindow:
    """An emitted E_window candidate (pre-gate). Stats cover the frames
    observed between T_start and T_stop, for Gate A and audit traces."""

    t_start_ms: int
    t_stop_ms: int
    end_reason: str
    frames: int = 0
    interviewer_present_frames: int = 0
    interviewer_closed_frames: int = 0
    mar_trace: List[Tuple[int, Optional[float]]] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.t_stop_ms - self.t_start_ms


class WindowMachine:
    def __init__(self, params: EnrollmentParams) -> None:
        self.params = params
        self.state = IDLE
        self.target_ema = CausalEMA(params.ema_span)
        self.interviewer_ema = CausalEMA(params.ema_span)
        self._t_start: int = 0
        self._deadline: int = 0
        self._prev_pts: Optional[int] = None
        self._stats_frames = 0
        self._stats_int_present = 0
        self._stats_int_closed = 0
        self._trace: List[Tuple[int, Optional[float]]] = []

    # -- internals -------------------------------------------------------------

    def _reset_window(self) -> None:
        self.state = IDLE
        self._stats_frames = 0
        self._stats_int_present = 0
        self._stats_int_closed = 0
        self._trace = []

    def _emit(self, t_stop_ms: int, reason: str) -> CandidateWindow:
        window = CandidateWindow(
            t_start_ms=self._t_start,
            t_stop_ms=t_stop_ms,
            end_reason=reason,
            frames=self._stats_frames,
            interviewer_present_frames=self._stats_int_present,
            interviewer_closed_frames=self._stats_int_closed,
            mar_trace=self._trace,
        )
        self._reset_window()
        return window

    def _accumulate(self, obs: FrameObs, ema_value: Optional[float]) -> None:
        self._stats_frames += 1
        self._trace.append((obs.pts_ms, None if obs.target_suspended else ema_value))
        if obs.interviewer_present:
            self._stats_int_present += 1
            int_ema = self.interviewer_ema.value
            if int_ema is not None and int_ema < self.params.mar_off:
                self._stats_int_closed += 1

    # -- public API --------------------------------------------------------------

    def step(self, obs: FrameObs) -> Optional[CandidateWindow]:
        """Feed one frame observation; returns a finished CandidateWindow
        when one ends at this frame, else None."""
        params = self.params
        emitted: Optional[CandidateWindow] = None

        # Interviewer EMA updates on every frame the interviewer is seen.
        # When the interviewer is ABSENT the EMA decays to None so a stale
        # pre-gap value cannot survive the absence and false-trigger the
        # Early Stop Rule on re-entry; it reseeds to the returning frame's
        # raw MAR. A present-but-MAR-unavailable frame is transient, not an
        # absence, so the EMA is left untouched there.
        if obs.interviewer_present and obs.interviewer_mar is not None:
            self.interviewer_ema.update(obs.interviewer_mar)
        elif not obs.interviewer_present:
            self.interviewer_ema.value = None

        # Target EMA: FROZEN while suspended (a brief head-turn must not lose
        # the smoothed value), but DECAYED to None while the target is ABSENT
        # so a stale pre-gap value cannot drive a spurious start/resume on
        # re-entry — it reseeds to the returning frame's raw MAR instead.
        if (
            obs.target_present
            and not obs.target_suspended
            and obs.target_mar is not None
        ):
            self.target_ema.update(obs.target_mar)
        elif not obs.target_present:
            self.target_ema.value = None
        ema = self.target_ema.value

        if self.state == PLOSIVE:
            if obs.target_suspended:
                # Timer pauses while the head is turned away.
                if self._prev_pts is not None:
                    self._deadline += obs.pts_ms - self._prev_pts
                self._accumulate(obs, ema)
            elif obs.pts_ms >= self._deadline:
                # Clean expiry: T_stop is the deadline PTS itself; the
                # current frame is outside the window.
                emitted = self._emit(self._deadline, END_PLOSIVE_EXPIRY)
            elif (
                obs.interviewer_present
                and self.interviewer_ema.value is not None
                and self.interviewer_ema.value > params.mar_on
            ):
                # Early Stop Rule: interjection ends the window NOW.
                emitted = self._emit(obs.pts_ms, END_INTERVIEWER_INTERJECTION)
            elif (
                obs.target_present
                and ema is not None
                and ema > params.mar_on
                and obs.vad_speech
            ):
                # Bench-corrected MAR (2026-06-12): the normalized outer-lip
                # formula sits in a narrow band (~0.10 closed to ~0.25 open,
                # resting ~0.13), so smoothed MAR alone cannot reliably tell a
                # resting-but-open mouth during silence from actual speech.
                # Silero VAD is therefore required alongside MAR > mar_on
                # (0.15) to resume, so the plosive buffer expires through
                # silence instead of immediately re-activating.
                self.state = ACTIVE
                self._accumulate(obs, ema)
            else:
                self._accumulate(obs, ema)

        elif self.state == ACTIVE:
            if obs.target_suspended:
                self._accumulate(obs, ema)   # frozen: wait for head return
            elif not obs.target_present or (
                ema is not None and ema < params.mar_off
            ) or not obs.vad_speech:
                # Bench-corrected MAR (2026-06-12): a relaxed/listening mouth
                # rests around ~0.13, ABOVE mar_off (0.10), so MAR rarely
                # falls below mar_off during ordinary silence (only fully
                # pressed-shut lips reach ~0.00). Silero VAD is the primary
                # close trigger so a window cannot span a non-speech segment
                # while the lips happen to stay open; MAR < mar_off remains a
                # secondary close condition.
                self.state = PLOSIVE
                self._deadline = obs.pts_ms + params.plosive_ms
                self._accumulate(obs, ema)
            else:
                self._accumulate(obs, ema)

        # IDLE (possibly entered this same frame after an emit): start check.
        if self.state == IDLE:
            if (
                obs.target_present
                and not obs.target_suspended
                and ema is not None
                and ema > params.mar_on
                and obs.vad_speech
            ):
                self.state = ACTIVE
                self._t_start = obs.pts_ms
                self._accumulate(obs, ema)

        self._prev_pts = obs.pts_ms
        return emitted

    def finalize(self, last_pts_ms: int) -> Optional[CandidateWindow]:
        """End of video: close any open window at the last frame PTS."""
        if self.state in (ACTIVE, PLOSIVE):
            return self._emit(last_pts_ms, END_OF_VIDEO)
        return None
