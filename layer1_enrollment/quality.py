"""
SPOVNOB — Module 2 (Layer 1): quality.py
=========================================

Pure progressive enrollment quality states.

Implements: Audio_Diarization.md — "Progressive Enrollment Quality
States" and the NO_ANTI_PROFILE escalation rule ("escalate the variance
gate thresholds — require more cumulative seconds to promote to STRONG",
quantified as strong_ms 45s -> 60s when no anti-profile exists;
review-flagged implementation default).

State rules (evaluated on the cumulative pool after each video):
  STRONG       — verified >= strong_ms (60s when no anti-profile)
                 AND pool variance <= variance_high
  MARGINAL     — verified >= marginal_ms (covers the document's
                 "20-45s OR high variance" branch: meeting the seconds
                 bar with high variance lands here, not STRONG)
  INSUFFICIENT — below marginal_ms
CRITICAL FAILURE is a batch-end condition (all videos done, still
INSUFFICIENT) and is enforced by the orchestrator, not here.

CUDA determinism dependencies: none.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

from .params import EnrollmentParams

STRONG = "STRONG"
MARGINAL = "MARGINAL"
INSUFFICIENT = "INSUFFICIENT"


def assess_quality(
    verified_ms: int,
    pool_variance: float,
    anti_available: bool,
    params: EnrollmentParams,
) -> str:
    strong_ms = params.strong_ms if anti_available else params.strong_ms_no_anti
    if verified_ms >= strong_ms and pool_variance <= params.variance_high:
        return STRONG
    if verified_ms >= params.marginal_ms:
        return MARGINAL
    return INSUFFICIENT
