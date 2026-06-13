"""
SPOVNOB — Module 2 (Layer 1): selftest.py
==========================================

Stdlib-only self-test over every pure submodule (standing policy: zero
pip installs, no torch, no cv2, no numpy, no GPU). The window-machine
tests derive EMA-dependent expectations from a tiny reference EMA loop
so they assert machine SEMANTICS (deadline arithmetic, early stop,
suspension pause) rather than float rounding accidents.

CUDA determinism dependencies: none.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from session_manifest import canonical_json, validate_time_fields

from . import encoding, gates, geometry, quality
from .enrollment import Layer1Error, load_clicks
from .params import EnrollmentParams
from .window_machine import (
    END_INTERVIEWER_INTERJECTION,
    END_OF_VIDEO,
    END_PLOSIVE_EXPIRY,
    FrameObs,
    WindowMachine,
)

P = EnrollmentParams()


def _obs(
    pts: int,
    mar: Optional[float] = None,
    present: bool = True,
    vad: bool = True,
    suspended: bool = False,
    int_present: bool = False,
    int_mar: Optional[float] = None,
) -> FrameObs:
    return FrameObs(
        pts_ms=pts, target_present=present, target_mar=mar,
        target_suspended=suspended, interviewer_present=int_present,
        interviewer_mar=int_mar, vad_speech=vad,
    )


def _reference_ema(values: List[float], span: int = 5) -> List[float]:
    alpha = 2.0 / (span + 1)
    state: Optional[float] = None
    out: List[float] = []
    for value in values:
        state = value if state is None else alpha * value + (1 - alpha) * state
        out.append(state)
    return out


def _drive(machine: WindowMachine, observations: List[FrameObs]):
    windows = []
    for one in observations:
        emitted = machine.step(one)
        if emitted is not None:
            windows.append(emitted)
    return windows


def _test_geometry() -> None:
    landmarks = [(0.0, 0.0)] * 106
    for index, point in zip(P.upper_inner_lip, [(0, 0), (10, 0), (20, 0)]):
        landmarks[index] = (float(point[0]), float(point[1]))
    for index, point in zip(P.lower_inner_lip, [(0, 12), (10, 12), (20, 12)]):
        landmarks[index] = (float(point[0]), float(point[1]))
    # bench-corrected width pair (52, 61) is disjoint from the lip-arc
    # indices and must be placed explicitly: distance 12 -> MAR exactly 1.0
    landmarks[P.mouth_width_pair[0]] = (-6.0, 6.0)
    landmarks[P.mouth_width_pair[1]] = (6.0, 6.0)
    assert geometry.compute_mar(landmarks, P) == 1.0
    assert geometry.compute_mar(landmarks[:50], P) is None
    assert geometry.compute_mar([(5.0, 5.0)] * 106, P) is None  # degenerate

    assert not geometry.yaw_suspends_mar(None, P)
    assert geometry.yaw_suspends_mar(36.0, P)
    assert geometry.yaw_suspends_mar(-40.0, P)
    assert not geometry.yaw_suspends_mar(35.0, P)

    ema = geometry.CausalEMA(span=5)
    assert ema.update(0.3) == 0.3          # pre-seeded, no warm-up
    assert abs(ema.update(0.9) - 0.5) < 1e-9


def _test_window_machine() -> None:
    step = 10  # ms grid

    def zeros_after(speech_frames: int, total_frames: int) -> List[float]:
        return [0.9] * speech_frames + [0.0] * (total_frames - speech_frames)

    # Where does the plosive buffer start for 5 speech frames then zeros?
    mars = zeros_after(5, 80)
    ema_track = _reference_ema(mars)
    plosive_index = next(
        i for i in range(5, len(mars)) if ema_track[i] < P.mar_off
    )
    plosive_pts = plosive_index * step
    expected_stop = plosive_pts + P.plosive_ms

    # A — clean plosive expiry: T_stop is the deadline PTS itself.
    machine = WindowMachine(P)
    windows = _drive(machine, [_obs(i * step, m) for i, m in enumerate(mars)])
    assert len(windows) == 1, windows
    assert (windows[0].t_start_ms, windows[0].t_stop_ms,
            windows[0].end_reason) == (0, expected_stop, END_PLOSIVE_EXPIRY)

    # B — plosive resume: lips reopen inside the buffer -> one window,
    # closed only at end of video.
    resume_index = plosive_index + 3
    mars_b = list(mars)
    for i in range(resume_index, len(mars_b)):
        mars_b[i] = 0.9
    machine = WindowMachine(P)
    obs_b = [_obs(i * step, m) for i, m in enumerate(mars_b)]
    windows = _drive(machine, obs_b)
    tail = machine.finalize(obs_b[-1].pts_ms)
    assert windows == [] and tail is not None
    assert tail.t_start_ms == 0 and tail.end_reason == END_OF_VIDEO
    assert tail.t_stop_ms == obs_b[-1].pts_ms

    # C — Early Stop Rule: interviewer lips open during the buffer ends
    # the window immediately at the current PTS.
    interject_pts = plosive_pts + step
    machine = WindowMachine(P)
    obs_c = []
    for i, m in enumerate(mars):
        if i * step == interject_pts:
            obs_c.append(_obs(i * step, m, int_present=True, int_mar=0.9))
        else:
            obs_c.append(_obs(i * step, m))
    windows = _drive(machine, obs_c)
    assert len(windows) == 1
    assert (windows[0].t_stop_ms, windows[0].end_reason) == (
        interject_pts, END_INTERVIEWER_INTERJECTION)

    # D — VAD gates the start trigger: T_start is the first VAD-true frame.
    machine = WindowMachine(P)
    obs_d = [_obs(0, 0.9, vad=False), _obs(10, 0.9, vad=False),
             _obs(20, 0.9, vad=False), _obs(30, 0.9, vad=True),
             _obs(40, 0.9)]
    _drive(machine, obs_d)
    tail = machine.finalize(40)
    assert tail is not None and tail.t_start_ms == 30

    # E — yaw suspension pauses the plosive timer (deadline extends by
    # the suspended wall time).
    suspended_frames = 11
    machine = WindowMachine(P)
    obs_e = [_obs(i * step, m) for i, m in enumerate(mars[:plosive_index + 2])]
    next_pts = (plosive_index + 2) * step
    for _ in range(suspended_frames):
        obs_e.append(_obs(next_pts, None, suspended=True))
        next_pts += step
    while next_pts <= expected_stop + suspended_frames * step + 5 * step:
        obs_e.append(_obs(next_pts, 0.0))
        next_pts += step
    windows = _drive(machine, obs_e)
    assert len(windows) == 1
    assert windows[0].t_stop_ms == expected_stop + suspended_frames * step
    assert windows[0].end_reason == END_PLOSIVE_EXPIRY

    # F — target disappearance behaves like the plosive buffer (frozen EMA).
    machine = WindowMachine(P)
    obs_f = [_obs(i * step, 0.9) for i in range(5)]
    next_pts = 5 * step
    while next_pts <= 5 * step + P.plosive_ms + 5 * step:
        obs_f.append(_obs(next_pts, None, present=False))
        next_pts += step
    windows = _drive(machine, obs_f)
    assert len(windows) == 1
    assert windows[0].t_stop_ms == 5 * step + P.plosive_ms

    # Interviewer stats: visible-and-closed frames are counted for Gate A.
    # (interviewer EMA must be strictly below mar_off=0.10 to count as
    # closed; 0.05 sits in the bench-validated pressed-shut range.)
    machine = WindowMachine(P)
    obs_g = [_obs(i * step, 0.9, int_present=True, int_mar=0.05)
             for i in range(10)]
    _drive(machine, obs_g)
    tail = machine.finalize(90)
    assert tail.interviewer_present_frames == 10
    assert tail.interviewer_closed_frames == 10


def _test_gates() -> None:
    common = dict(interviewer_present_frames=10, interviewer_closed_frames=9,
                  params=P)
    ok = gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.80, sim_anti=0.20,
        anti_available=True, **common)
    assert ok.accepted and ok.failed_gate is None

    assert gates.evaluate_triple_gate(
        vad_coverage=0.3, sim_seed=0.80, sim_anti=0.20,
        anti_available=True, **common).failed_gate == "A"
    assert gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.80, sim_anti=0.20, anti_available=True,
        interviewer_present_frames=10, interviewer_closed_frames=5,
        params=P).failed_gate == "A"
    assert gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.60, sim_anti=0.20,
        anti_available=True, **common).failed_gate == "B"
    assert gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.80, sim_anti=0.60,
        anti_available=True, **common).failed_gate == "C"

    loose = EnrollmentParams(threshold_anti=0.70)
    margin_fail = gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.75, sim_anti=0.65, anti_available=True,
        interviewer_present_frames=0, interviewer_closed_frames=0,
        params=loose)
    assert margin_fail.failed_gate == "C"
    assert margin_fail.detail["reason"] == "margin_too_small"

    skipped = gates.evaluate_triple_gate(
        vad_coverage=0.8, sim_seed=0.80, sim_anti=None,
        anti_available=False, **common)
    assert skipped.accepted and skipped.detail["anti_applied"] is False

    assert gates.mtrap_discard(0.65, P) and not gates.mtrap_discard(0.55, P)
    assert gates.contamination_level(0.30, P) == gates.CONTAM_OK
    assert gates.contamination_level(0.50, P) == gates.CONTAM_WARNING
    assert gates.contamination_level(0.70, P) == gates.CONTAM_HALT

    e1, e2 = [1.0, 0.0], [0.0, 1.0]
    assert gates.pairwise_cosine_variance([e1, e1]) == 0.0
    assert abs(gates.pairwise_cosine_variance([e1, e1, e2]) - 2.0 / 9.0) < 1e-9

    segments = [(100, 200), (300, 400)]
    assert gates.segment_overlap_ms(150, 350, segments) == 100
    assert gates.vad_near(90, segments, 50)
    assert gates.vad_near(240, segments, 50)   # within 50 of either segment
    assert not gates.vad_near(49, segments, 50)
    assert not gates.vad_near(500, segments, 50)


def _test_quality() -> None:
    assert quality.assess_quality(50000, 0.01, True, P) == quality.STRONG
    assert quality.assess_quality(50000, 0.10, True, P) == quality.MARGINAL
    assert quality.assess_quality(30000, 0.01, True, P) == quality.MARGINAL
    assert quality.assess_quality(10000, 0.01, True, P) == quality.INSUFFICIENT
    assert quality.assess_quality(50000, 0.01, False, P) == quality.MARGINAL
    assert quality.assess_quality(65000, 0.01, False, P) == quality.STRONG


def _test_encoding_pure() -> None:
    assert encoding.l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert abs(encoding.cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-12
    assert encoding.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    mixed = encoding.duration_weighted_mean(
        [[1.0, 0.0], [0.0, 1.0]], [1000, 3000])
    assert mixed == [0.25, 0.75]

    assert encoding.plan_chunks(45000, 60000, 2000) == [(0, 45000)]
    assert encoding.plan_chunks(130000, 60000, 2000) == [
        (0, 60000), (58000, 118000), (116000, 130000)]
    assert encoding.plan_chunks(121000, 60000, 2000) == [
        (0, 60000), (58000, 118000), (116000, 121000)]

    pcm = bytes(64000)                      # 32000 samples = 2 s
    sliced = encoding.pcm_slice(pcm, 23, 32000, 1023, 1523)
    assert len(sliced) == 16000             # 500 ms * 16 samples * 2 bytes
    clamped = encoding.pcm_slice(pcm, 23, 32000, 0, 50)
    assert len(clamped) == 27 * 16 * 2      # clamped at data start


def _test_params_and_clicks() -> None:
    payload = P.manifest_payload()
    validate_time_fields(payload)           # every *_ms key is an int
    canonical_json(payload)                 # serializable (tuples -> lists)

    with tempfile.TemporaryDirectory() as tmp:
        clicks_path = Path(tmp) / "clicks.json"
        clicks_path.write_text(json.dumps({
            "speaking_click": {"file_index": 0, "pts_ms": 41250,
                               "x": 812, "y": 440},
            "anti_click": {"file_index": 0, "pts_ms": 95000,
                           "x": 300, "y": 400},
        }), encoding="utf-8")
        clicks = load_clicks(clicks_path)
        assert clicks.speaking.pts_ms == 41250 and clicks.anti is not None

        clicks_path.write_text(json.dumps({
            "speaking_click": {"file_index": 0, "pts_ms": 41250.5,
                               "x": 1, "y": 1},
        }), encoding="utf-8")
        try:
            load_clicks(clicks_path)
            raise AssertionError("float pts_ms accepted")
        except Layer1Error:
            pass


def run() -> int:
    assert "torch" not in sys.modules, "torch imported at module level"
    _test_geometry()
    _test_window_machine()
    _test_gates()
    _test_quality()
    _test_encoding_pure()
    _test_params_and_clicks()
    for forbidden in ("torch", "cv2", "numpy", "onnxruntime", "insightface",
                      "ultralytics"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"
    print("layer1_enrollment stdlib self-test OK — "
          "no torch, no cv2, no numpy, no GPU, no pip")
    return 0
