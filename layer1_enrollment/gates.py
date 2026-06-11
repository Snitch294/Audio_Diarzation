"""
SPOVNOB — Module 2 (Layer 1): gates.py
=======================================

Pure gate logic: the Triple Validation Gate (A/B/C), the M-Trap guard,
contamination levels for sim(E_composite, E_anti), pairwise-cosine pool
variance, and the Silero segment lookup helpers used by Gate A and the
window machine's start trigger. No models, no I/O — fully self-testable.

Implements: Audio_Diarization.md — "The Triple Validation Gate",
"M-Trap Guard", "`threshold_target`, `E_anti` capture, and how `E_anti`
is used" (sanity checks), "Intra-Pool Variance Check".

Gate definitions (evaluated in order A -> B -> C; first failure wins):
  Gate A: Silero confirms speech (window VAD coverage >=
          gate_a_vad_min_coverage — implementation-defined quantification,
          review-flagged) AND, when the interviewer was visible during
          the window, their lips were closed for > int_lips_closed_frac
          of the frames in which they were visible.
  Gate B: cosine_sim(window, E_seed) >= threshold_target.
  Gate C: only when an anti-profile exists —
          cosine_sim(window, E_anti) <= threshold_anti AND
          (sim_seed - sim_anti) >= margin_minimum.
          Missing E_anti: Gate C is skipped, never failed (the document's
          fail-open robustness rule; the candidate records anti_applied=False).

CUDA determinism dependencies: none (math.fsum-based pure arithmetic).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .encoding import cosine
from .params import EnrollmentParams

CONTAM_OK = "OK"
CONTAM_WARNING = "WARNING"
CONTAM_HALT = "HALT"


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    failed_gate: Optional[str]          # None | "A" | "B" | "C"
    detail: Dict[str, Any]


def evaluate_triple_gate(
    *,
    vad_coverage: float,
    interviewer_present_frames: int,
    interviewer_closed_frames: int,
    sim_seed: float,
    sim_anti: Optional[float],
    anti_available: bool,
    params: EnrollmentParams,
) -> GateResult:
    closed_frac = (
        interviewer_closed_frames / interviewer_present_frames
        if interviewer_present_frames > 0
        else None
    )
    detail: Dict[str, Any] = {
        "vad_coverage": vad_coverage,
        "interviewer_closed_frac": closed_frac,
        "sim_seed": sim_seed,
        "sim_anti": sim_anti,
        "anti_applied": anti_available,
    }

    # Gate A — visual + VAD contamination check
    if vad_coverage < params.gate_a_vad_min_coverage:
        return GateResult(False, "A", {**detail, "reason": "vad_coverage_low"})
    if closed_frac is not None and closed_frac < params.int_lips_closed_frac:
        return GateResult(False, "A", {**detail, "reason": "interviewer_lips_open"})

    # Gate B — similarity to the verified seed
    if sim_seed < params.threshold_target:
        return GateResult(False, "B", {**detail, "reason": "low_sim_to_seed"})

    # Gate C — anti-profile rejection + ambiguity margin
    if anti_available:
        if sim_anti is None:
            return GateResult(False, "C", {**detail, "reason": "sim_anti_missing"})
        if sim_anti > params.threshold_anti:
            return GateResult(False, "C", {**detail, "reason": "high_sim_to_anti"})
        if (sim_seed - sim_anti) < params.margin_minimum:
            return GateResult(False, "C", {**detail, "reason": "margin_too_small"})

    return GateResult(True, None, detail)


def mtrap_discard(sim_to_seed: float, params: EnrollmentParams) -> bool:
    """M-Trap Guard: a Track B anti candidate too similar to E_seed is
    likely the target making a lips-closed phoneme — discard silently
    (the caller logs the discard)."""
    return sim_to_seed > params.mtrap_sim_max


def contamination_level(similarity: float, params: EnrollmentParams) -> str:
    """sim(E_composite, E_anti) severity (guardrail 8 / sanity checks)."""
    if similarity > params.anti_contam_halt:
        return CONTAM_HALT
    if similarity > params.anti_contam_warning:
        return CONTAM_WARNING
    return CONTAM_OK


def pairwise_cosine_variance(vectors: Sequence[Sequence[float]]) -> float:
    """Population variance of all pairwise cosine similarities in a pool.
    Fewer than two vectors -> 0.0. Deterministic (fsum reductions,
    fixed i<j pair order)."""
    n = len(vectors)
    if n < 2:
        return 0.0
    sims: List[float] = [
        cosine(vectors[i], vectors[j]) for i in range(n) for j in range(i + 1, n)
    ]
    mean = math.fsum(sims) / len(sims)
    return math.fsum((s - mean) ** 2 for s in sims) / len(sims)


# --- Silero segment lookups (shared by Gate A and the window machine) ---------

def segment_overlap_ms(
    start_ms: int, stop_ms: int, segments: Sequence[Tuple[int, int]]
) -> int:
    """Total overlap between [start_ms, stop_ms) and the segment list."""
    return sum(
        max(0, min(stop_ms, seg_end) - max(start_ms, seg_start))
        for seg_start, seg_end in segments
    )


def vad_near(
    pts_ms: int, segments: Sequence[Tuple[int, int]], tol_ms: int
) -> bool:
    """True if Silero marks speech within +/- tol_ms of pts_ms
    (the document's `Silero.vad_near(pts, vad_tol)` predicate)."""
    return any(
        seg_start - tol_ms <= pts_ms <= seg_end + tol_ms
        for seg_start, seg_end in segments
    )
