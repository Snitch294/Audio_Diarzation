"""
SPOVNOB — Module 2: layer1_enrollment (package)
================================================

Layer:      Layer 1 — Visual-Anchored Enrollment.
            Imports only Module 0a (session_manifest), Module 0b
            (environment_gate) and Module 1 (layer0_preprocessor).

Purpose:    Produce the frozen Target Enrollment Profile (E_composite)
            and interviewer anti-profile (E_anti) from operator-anchored,
            triple-validated, visually-confirmed speech windows. Zero
            training — frozen models, deterministic arithmetic only.

Inputs:     SessionManifest · BatchAudio (Layer 0) · ResidentModels
            (environment gate) · operator clicks JSON · EnrollmentParams
Outputs:    EnrollmentResult (frozen E_composite + pools + quality
            history) · per-window WAV/JSON artifacts · manifest entries
            for every vector, discard, warning, quality state, freeze.

Package layout (one logical module, per the approved package ruling):
    params.py          parameter table (doc + implementation defaults)
    errors.py          Layer1Error / Layer1ReclickError
    geometry.py        MAR, causal EMA, yaw suspension          [pure]
    window_machine.py  E_window capture state machine           [pure]
    gates.py           Triple Gate, M-Trap, variance, VAD lookup [pure]
    quality.py         STRONG/MARGINAL/INSUFFICIENT states      [pure]
    encoding.py        pooling/cosine/chunking [pure] + ECAPA (torch lazy)
    vision.py          frame PTS, YOLO/InsightFace scan (cv2/GPU lazy)
    enrollment.py      orchestrator (sequential, canonical file order)
    selftest.py        stdlib-only self-test over all pure modules

Implements (Audio_Diarization.md): the entire "Layer 1 — Finalized
Architecture Notes" section — see each submodule's docstring for its
exact subsection mapping, and enrollment.py for the guardrail map.

CUDA determinism dependencies: the four environment_gate constants for
ECAPA/YOLO forward passes; pure arithmetic uses math.fsum (correctly
rounded, order-independent). All timestamps integer PTS milliseconds.

Self-test:  python3 -m layer1_enrollment --selftest   (zero pip installs)
Run:        python3 -m layer1_enrollment --run --videos ... --clicks ...
                --work-dir ... --model-store ... --manifest ...
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

from .enrollment import (
    AntiClick,
    EnrollmentResult,
    OperatorClicks,
    PoolEntry,
    SpeakingClick,
    load_clicks,
    run_layer1,
)
from .errors import Layer1Error, Layer1ReclickError
from .params import EnrollmentParams

__all__ = [
    "AntiClick",
    "EnrollmentParams",
    "EnrollmentResult",
    "Layer1Error",
    "Layer1ReclickError",
    "OperatorClicks",
    "PoolEntry",
    "SpeakingClick",
    "load_clicks",
    "run_layer1",
]
