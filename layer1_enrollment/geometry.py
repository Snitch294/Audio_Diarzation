"""
SPOVNOB — Module 2 (Layer 1): geometry.py
==========================================

Pure lip/pose geometry: normalized Mouth Aspect Ratio from InsightFace
2d106det landmarks, the 5-frame causal EMA (pre-seeded with the first
value to kill warm-up artifacts), and the head-yaw suspension rule.

Implements: Audio_Diarization.md — "MAR Definitions & Suggested
parameters" (MAR normalization, explicit landmark indices, head pose
yaw filter, EMA smoothing with pre-seed).
CUDA determinism dependencies: none (pure stdlib math).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import math
from typing import Optional, Sequence, Tuple

from .params import EnrollmentParams

Point = Tuple[float, float]


def euclidean(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def compute_mar(
    landmarks: Sequence[Point], params: EnrollmentParams
) -> Optional[float]:
    """Normalized MAR: mean vertical inner-lip distance divided by the
    mouth width (face-scale invariant). Returns None if the landmark
    array is too short or the width is degenerate."""
    needed = max(
        *params.upper_inner_lip, *params.lower_inner_lip, *params.mouth_width_pair
    )
    if len(landmarks) <= needed:
        return None
    vertical_gaps = [
        euclidean(landmarks[u], landmarks[l])
        for u, l in zip(params.upper_inner_lip, params.lower_inner_lip)
    ]
    vertical = math.fsum(vertical_gaps) / len(vertical_gaps)
    width = euclidean(
        landmarks[params.mouth_width_pair[0]],
        landmarks[params.mouth_width_pair[1]],
    )
    if width <= 1e-6:
        return None
    return vertical / width


def yaw_suspends_mar(yaw_degrees: Optional[float], params: EnrollmentParams) -> bool:
    """Head Pose Yaw Filter: beyond ±yaw_max_degrees the lips project into
    a foreshortened geometry, so MAR checking is suspended entirely.
    Unknown yaw (pose model gave nothing) does NOT suspend — that case is
    logged once per video by the orchestrator instead (review-flagged)."""
    return yaw_degrees is not None and abs(yaw_degrees) > params.yaw_max_degrees


class CausalEMA:
    """5-frame causal EMA, alpha = 2/(span+1). Pre-seeded with the first
    observed value so there is no warm-up artifact (the document's
    'pre-seed the EMA buffer with frame 0's MAR' rule)."""

    def __init__(self, span: int) -> None:
        self.alpha = 2.0 / (span + 1)
        self.value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = float(sample)
        else:
            self.value = self.alpha * float(sample) + (1.0 - self.alpha) * self.value
        return self.value
