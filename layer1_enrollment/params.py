"""
SPOVNOB — Module 2 (Layer 1): params.py
========================================

The complete Layer 1 parameter set: every value from the document's
"Enrollment parameter table" plus the implementation-defined defaults
introduced by this module (each one review-flagged in the module 2
delivery notes). The full set is written to the manifest at Layer 1
init, so any change is part of the chain of custody.

Implements: Audio_Diarization.md — "Enrollment parameter table",
"MAR Definitions & Suggested parameters".
CUDA determinism dependencies: none (pure data).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class EnrollmentParams:
    # --- document parameter table (Layer 1) ----------------------------------
    face_reid_threshold: float = 0.40    # ArcFace cosine: keep target lock
    reid_warning_floor: float = 0.50     # guardrail 6: running-mean warning
    mar_on: float = 0.55                 # hysteresis: lips clearly open
    mar_off: float = 0.40                # hysteresis: lips closing
    plosive_ms: int = 500                # keep window open through closures
    vad_tol_ms: int = 50                 # Silero/PTS alignment tolerance
    min_enroll_len_ms: int = 2000        # discard shorter candidates
    seed_min_ms: int = 3000              # E_seed minimum (supersedes the
                                         # legacy "3-8s" remnant; no maximum)
    int_lips_closed_frac: float = 0.80   # Gate A: interviewer visually silent
    threshold_target: float = 0.70       # Gate B: sim(window, E_seed)
    threshold_anti: float = 0.50         # Gate C: sim(window, E_anti) ceiling
    margin_minimum: float = 0.15         # Gate C: seed-vs-anti margin
    mtrap_sim_max: float = 0.60          # Track B M-Trap guard
    anti_contam_warning: float = 0.45    # sim(E_composite, E_anti) warning
    anti_contam_halt: float = 0.60       # sim(E_composite, E_anti) halt
    pool_var_warning: float = 0.05       # intra-pool variance increase warning

    # --- geometry / vision ----------------------------------------------------
    yaw_max_degrees: float = 35.0        # suspend MAR beyond this head yaw
    ema_span: int = 5                    # 5-frame causal EMA (pre-seeded)
    upper_inner_lip: Tuple[int, ...] = (52, 53, 54)   # InsightFace 2d106det
    lower_inner_lip: Tuple[int, ...] = (61, 62, 63)   # InsightFace 2d106det
    mouth_width_pair: Tuple[int, int] = (52, 61)
    # ^ VALIDATE-ON-BENCH: these are the document's exact indices. The width
    #   pair (52, 61) duplicates the vertical inner-lip pair, which would
    #   make MAR ~constant; the indices can only be confirmed against real
    #   2d106det output on the Ubuntu bench. They are parameters (manifest-
    #   logged), so correcting them is a recorded operator change, not code.
    insightface_min_det_score: float = 0.50  # guardrail 5: below = not detected
    yolo_min_conf: float = 0.30          # person-present gate for InsightFace
    silence_stride: int = 1              # >1 enables the optional visual-scan
                                         # efficiency rule (System Environment)

    # --- ECAPA encoding (Single-Pass rule, Rev 3) ------------------------------
    encode_max_ms: int = 60000           # 60s sanity cap: one pass below this
    encode_overlap_ms: int = 2000        # chunk overlap above the cap

    # --- implementation-defined defaults (review-flagged) ----------------------
    gate_a_vad_min_coverage: float = 0.50   # "Silero confirms speech" quantified
    trackb_window_ms: int = 2000            # doc: 2000ms context window
    trackb_min_spacing_ms: int = 2000       # dedupe: one candidate per 2s
    click_overlap_max_frac: float = 0.20    # guardrail 1 visual overlap proxy
    strong_ms: int = 45000                  # STRONG: >= 45s verified
    strong_ms_no_anti: int = 60000          # NO_ANTI_PROFILE escalation: 60s
    marginal_ms: int = 20000                # MARGINAL floor: 20s
    variance_high: float = 0.05             # "high variance" for quality states

    def manifest_payload(self) -> Dict[str, Any]:
        return asdict(self)
