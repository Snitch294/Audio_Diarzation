"""
SPOVNOB — Module 3: layer2_tracker.py
======================================

Layer:      Layer 2 — Pure ECAPA Sliding Window Scanning.
            Imports Modules 0a/0b/1/2 only (session_manifest,
            environment_gate, layer0_preprocessor, layer1_enrollment).

Purpose:    The single authoritative target-tracking pass: scan the raw
            16kHz audio with the frozen E_composite_final / E_anti via
            pure cosine similarity, calibrate thresholds per session,
            median-pool overlapping window scores into 1-second blocks,
            tier them, refine HIGH run edges (trim-only), and emit
            PTS-stamped raw blocks for Layer 3 plus the SHA-256-hashed
            authoritative output document.

Inputs:     SessionManifest · BatchAudio (Layer 0, RAM-preloaded) ·
            ResidentModels (gate) · EnrollmentResult (Layer 1, frozen) ·
            Layer2Params · work dir
Outputs:    Layer2Result (per-file HIGH runs with refined edges + full
            block tier map + ratio/drift diagnostics + calibration) ·
            <work_dir>/layer2/layer2_output.json (canonical, hashed) ·
            per-file worker logs merged into the manifest in canonical
            order · manifest entries: init, sanity, calibration,
            warnings, output hash.

Implements (Audio_Diarization.md, Layer 2 — Rev 3):
            "Decision Record" inputs/outputs · "Score semantics" (raw
            cosine S_*, never probabilities) · "Deterministic Threshold
            Calibration" · "Step-by-step execution model" · "How the
            sliding window works" (Silero skip rule) · "How a window
            gets scored" (median pooling) · "Edge-Trim Boundary
            Refinement" · "Confidence tiers" · "Activity ratio check" ·
            "Single authoritative pass" (preview optional, default OFF) ·
            "What Layer 2 does when E_anti is missing" · Execution Flow
            Steps 1-9 · "Layer 2 Output Block Format" · System
            Environment "Canonical Manifest Merge Rule".

CUDA determinism dependencies:
            All four environment_gate constants; fixed inference batch
            ECAPA_BATCH_WINDOWS = 256 (final partial batches are padded
            by repeating the last window — pad outputs discarded — so
            every forward pass has the identical shape); float32; pure
            reductions via math.fsum / statistics.median.

Determinism / sequencing notes:
            - GPU scoring runs SEQUENTIALLY in canonical file order
              (2-3 batches per file makes fan-out pointless and a shared
              resident model is not provably race-free across threads),
              but every per-file record is written through a WorkerLog
              and merged via the Canonical Manifest Merge Rule, so the
              manifest is identical if scoring is ever parallelized at
              process level.
            - Time is integer PTS milliseconds everywhere; block grid is
              data-relative 1s boundaries converted to local PTS via
              audio_start_pts_ms and to global via file_offset_ms.

Self-test:  python3 layer2_tracker.py --selftest   (stdlib only: the
            full flow runs against injected synthetic scorers — zero
            pip installs, no torch, no GPU).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from layer0_preprocessor import BatchAudio, FileAudio
from layer1_enrollment.encoding import (
    cosine,
    duration_weighted_mean,
    l2_normalize,
    pcm_slice,
)
from layer1_enrollment.enrollment import EnrollmentResult
from layer1_enrollment.gates import segment_overlap_ms
from session_manifest import (
    Operation,
    SessionManifest,
    WorkerLog,
    canonical_json,
    merge_worker_logs,
    sha256_of_obj,
)

OP_INIT = "layer2_init"
OP_BLOCK = "layer2_block"
OP_EDGE = "layer2_edge_trim"
OP_FILE = "layer2_file_summary"

TIER_HIGH = "HIGH"
TIER_MEDIUM = "MEDIUM"
TIER_SUB = "SUB_THRESHOLD"
TIER_REJECT = "REJECT"
TIER_SKIPPED = "SKIPPED_NONSPEECH"

CAL_DERIVED = "DERIVED"
CAL_DERIVED_NO_ANTI = "DERIVED_NO_ANTI"
CAL_FALLBACK = "FALLBACK_DEFAULTS"

# A scorer maps window spans (data-relative ms) to (S_target,
# S_interviewer-or-None). Injectable so the self-test drives the whole
# flow without torch; production uses _ecapa_scorer below.
Scorer = Callable[[Sequence[Tuple[int, int]]], List[Tuple[float, Optional[float]]]]


class Layer2Error(RuntimeError):
    """Unrecoverable Layer 2 failure (blocking halt already recorded)."""


@dataclass(frozen=True)
class Layer2Params:
    # --- architectural constants (Design Decision Summary) -------------------
    window_ms: int = 5000                  # row 6 — never tunable
    hop_ms: int = 1000                     # row 7 — never tunable
    block_ms: int = 1000
    silero_skip_floor: float = 0.20        # row 8 — Silero skip rule
    evidence_floor: float = 0.20           # row 3 — sub-threshold log floor
    edge_fine_window_ms: int = 2000        # row 13 — edge-trim window
    edge_fine_hop_ms: int = 250            # row 13 — edge-trim hop
    edge_scan_span_ms: int = 2000          # fine scan reaches +/- this far
    edge_min_fine_window_ms: int = 1000    # clamped fine windows below this
                                           # are skipped (file-bound edges)
    edge_max_trim_ms: int = 750            # trim beyond this -> demote block
    # --- calibration ------------------------------------------------------------
    genuine_quantile: float = 0.10
    impostor_safety_margin: float = 0.05
    theta_clamp_low: float = 0.45
    theta_clamp_high: float = 0.75
    theta_med_step: float = 0.15
    theta_med_floor: float = 0.30
    min_calibration_windows: int = 10
    fallback_theta_high: float = 0.60
    fallback_theta_med: float = 0.40
    no_anti_theta_floor: float = 0.55
    margin_minimum: float = 0.15           # row 4 — NOT calibrated
    # --- sanity / diagnostics ---------------------------------------------------
    anti_contam_warning: float = 0.45
    anti_contam_halt: float = 0.60
    ratio_normal: float = 0.25
    ratio_low: float = 0.10
    drift_window_ms: int = 30000
    drift_delta: float = 0.10

    def manifest_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    theta_high: float
    theta_med: float
    kind: str                              # DERIVED / DERIVED_NO_ANTI / FALLBACK
    record: Dict[str, Any]
    calibration_ref: str
    overlap_warning: bool


@dataclass
class HighRun:
    start_local_ms: int
    end_local_ms: int
    blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FileTrack:
    file_index: int
    source_file: str
    high_runs: List[HighRun]
    tier_counts: Dict[str, int]
    high_ms: int
    silero_ms: int
    ratio: Optional[float]
    ratio_level: str
    unattributed_speech_ms: int
    high_scores: List[float]               # S_target medians of HIGH blocks
    # Full per-block tier map (local PTS), added for Layer 3's gap
    # dominance guard (Module 4 amendment — additive, no behavior change).
    blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Layer2Result:
    calibration: Calibration
    files: List[FileTrack]
    no_anti_profile: bool
    authoritative: bool
    output_path: str
    output_sha256: str


# =============================================================================
# Pure calibration machinery (stdlib-only; self-tested)
# =============================================================================

def quantile_sorted(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (deterministic,
    documented quantile definition)."""
    if not sorted_values:
        raise ValueError("empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(position)
    frac = position - low
    if low + 1 >= len(sorted_values):
        return float(sorted_values[-1])
    return sorted_values[low] + frac * (sorted_values[low + 1] - sorted_values[low])


def loo_scores(
    vectors: Sequence[Sequence[float]], durations_ms: Sequence[int]
) -> List[float]:
    """Leave-one-out genuine scores: each pool vector against the
    duration-weighted mean of the others (the same arithmetic the
    enrollment pool itself uses). A pool with fewer than two vectors has
    no leave-one-out counterpart and yields no scores — the caller's
    minimum-data rule then routes calibration to FALLBACK_DEFAULTS (a
    single >=20s seed window can legitimately reach Layer 2 alone)."""
    if len(vectors) < 2:
        return []
    scores: List[float] = []
    for index in range(len(vectors)):
        rest = [v for k, v in enumerate(vectors) if k != index]
        rest_durations = [d for k, d in enumerate(durations_ms) if k != index]
        pooled = l2_normalize(duration_weighted_mean(rest, rest_durations))
        scores.append(cosine(vectors[index], pooled))
    return scores


def derive_thresholds(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    anti_available: bool,
    enrollment_ref: str,
    params: Layer2Params,
) -> Calibration:
    """The document's derivation, verbatim, as pure sorted arithmetic."""
    genuine_sorted = sorted(genuine_scores)
    impostor_sorted = sorted(impostor_scores)
    record: Dict[str, Any] = {
        "method": "loo_duration_weighted_vs_anti_pool",
        "n_enrollment": len(genuine_sorted),
        "n_impostor": len(impostor_sorted),
        "genuine_scores_sorted": list(genuine_sorted),
        "impostor_scores_sorted": list(impostor_sorted),
        "genuine_quantile": params.genuine_quantile,
        "margin_minimum": params.margin_minimum,
        "evidence_floor": params.evidence_floor,
        "enrollment_ref": enrollment_ref,
    }
    overlap_warning = False

    if len(genuine_sorted) < params.min_calibration_windows:
        theta_high = params.fallback_theta_high
        theta_med = params.fallback_theta_med
        kind = CAL_FALLBACK
        record.update({
            "reason": "insufficient_enrollment_windows",
            "required": params.min_calibration_windows,
        })
    else:
        q_genuine = quantile_sorted(genuine_sorted, params.genuine_quantile)
        record["q10_genuine"] = q_genuine
        if anti_available and impostor_sorted:
            raw = max(
                q_genuine, impostor_sorted[-1] + params.impostor_safety_margin
            )
            record["max_impostor"] = impostor_sorted[-1]
            kind = CAL_DERIVED
        else:
            raw = max(q_genuine, params.no_anti_theta_floor)
            kind = CAL_DERIVED_NO_ANTI
        record["theta_high_raw"] = raw
        theta_high = min(max(raw, params.theta_clamp_low), params.theta_clamp_high)
        record["clamped_low"] = raw < params.theta_clamp_low
        record["clamped_high"] = raw > params.theta_clamp_high
        if record["clamped_high"]:
            overlap_warning = True          # CALIBRATION_OVERLAP condition
        theta_med = max(theta_high - params.theta_med_step, params.theta_med_floor)

    record.update({
        "theta_high": theta_high,
        "theta_med": theta_med,
        "calibration": kind,
        "calibration_overlap": overlap_warning,
    })
    return Calibration(
        theta_high=theta_high,
        theta_med=theta_med,
        kind=kind,
        record=record,
        calibration_ref=sha256_of_obj(record),
        overlap_warning=overlap_warning,
    )


# =============================================================================
# Pure windowing / tiering machinery (stdlib-only; self-tested)
# =============================================================================

def plan_windows(
    duration_ms: int, window_ms: int, hop_ms: int
) -> List[Tuple[int, int]]:
    """Data-relative coarse window spans. Files shorter than one window
    yield a single full-file window."""
    if duration_ms <= window_ms:
        return [(0, duration_ms)] if duration_ms > 0 else []
    spans: List[Tuple[int, int]] = []
    start = 0
    while start + window_ms <= duration_ms:
        spans.append((start, start + window_ms))
        start += hop_ms
    return spans


def tier_block(
    s_target: float,
    s_interviewer: Optional[float],
    calibration: Calibration,
    anti_available: bool,
    params: Layer2Params,
) -> Tuple[str, bool]:
    """Tier one 1s block. Returns (tier, margin_failed). A block above
    theta_high that fails the dual-target margin rule is demoted to
    MEDIUM with margin_failed=True (review-flagged decision)."""
    if s_target > calibration.theta_high:
        if not anti_available:
            return TIER_HIGH, False
        if s_interviewer is not None and (
            s_target - s_interviewer
        ) > params.margin_minimum:
            return TIER_HIGH, False
        return TIER_MEDIUM, True
    if s_target > calibration.theta_med:
        return TIER_MEDIUM, False
    if s_target >= params.evidence_floor:
        return TIER_SUB, False
    return TIER_REJECT, False


def median_pool_blocks(
    duration_ms: int,
    window_spans: Sequence[Tuple[int, int]],
    window_scores: Sequence[Tuple[float, Optional[float]]],
    params: Layer2Params,
) -> List[Dict[str, Any]]:
    """Map overlapping window scores onto the 1s block grid and take the
    median per block (statistics.median: even counts average the middle
    pair — fixed, documented rule). Blocks with no covering scored
    window get tier SKIPPED_NONSPEECH."""
    block_count = duration_ms // params.block_ms
    per_block_target: List[List[float]] = [[] for _ in range(block_count)]
    per_block_interviewer: List[List[float]] = [[] for _ in range(block_count)]
    for (w_start, w_end), (s_target, s_interviewer) in zip(
        window_spans, window_scores
    ):
        first_block = (w_start + params.block_ms - 1) // params.block_ms
        last_block = w_end // params.block_ms      # exclusive
        for block in range(first_block, min(last_block, block_count)):
            per_block_target[block].append(s_target)
            if s_interviewer is not None:
                per_block_interviewer[block].append(s_interviewer)
    blocks: List[Dict[str, Any]] = []
    for index in range(block_count):
        if per_block_target[index]:
            blocks.append({
                "block_index": index,
                "evaluations": len(per_block_target[index]),
                "s_target_median": statistics.median(per_block_target[index]),
                "s_interviewer_median": (
                    statistics.median(per_block_interviewer[index])
                    if per_block_interviewer[index] else None
                ),
            })
        else:
            blocks.append({
                "block_index": index,
                "evaluations": 0,
                "s_target_median": None,
                "s_interviewer_median": None,
            })
    return blocks


def find_high_runs(tiers: Sequence[str]) -> List[Tuple[int, int]]:
    """Maximal runs of contiguous HIGH blocks as (first, last_exclusive)."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, tier in enumerate(tiers):
        if tier == TIER_HIGH and start is None:
            start = index
        elif tier != TIER_HIGH and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(tiers)))
    return runs


def _fine_positions(edge_ms: int, params: Layer2Params) -> List[int]:
    span, hop = params.edge_scan_span_ms, params.edge_fine_hop_ms
    return list(range(edge_ms - span, edge_ms + span + 1, hop))


def refine_run_edges(
    run_start_ms: int,
    run_end_ms: int,
    duration_ms: int,
    scorer: Scorer,
    calibration: Calibration,
    anti_available: bool,
    params: Layer2Params,
) -> Dict[str, Any]:
    """Edge-Trim Boundary Refinement for one HIGH run (trim-only).

    Leading edge: fine windows [p, p+2000ms] at every 250ms position p in
    [edge-2000, edge+2000] (clamped; clamped windows shorter than
    edge_min_fine_window_ms are skipped). The new start is the SMALLEST
    p >= run_start whose fine window passes the HIGH criteria. Trailing
    edge mirrored with windows [p-2000, p] and the LARGEST p <= run_end.
    A required trim beyond edge_max_trim_ms (or no passing position)
    demotes exactly one edge 1s block to MEDIUM — logged, no recursion
    (review-flagged decision). The run can never grow."""

    def _passes(score: Tuple[float, Optional[float]]) -> bool:
        s_target, s_interviewer = score
        if s_target <= calibration.theta_high:
            return False
        if anti_available and s_interviewer is not None:
            return (s_target - s_interviewer) > params.margin_minimum
        return True

    def _scan(positions: List[int], leading: bool) -> Tuple[
        Dict[int, Optional[bool]], List[Dict[str, Any]]
    ]:
        spans: List[Tuple[int, int]] = []
        usable: List[int] = []
        pass_map: Dict[int, Optional[bool]] = {}
        for p in positions:
            if leading:
                span = (max(0, p), min(duration_ms, p + params.edge_fine_window_ms))
            else:
                span = (max(0, p - params.edge_fine_window_ms), min(duration_ms, p))
            if span[1] - span[0] < params.edge_min_fine_window_ms:
                pass_map[p] = None          # skipped: too clamped at bounds
                continue
            spans.append(span)
            usable.append(p)
        scores = scorer(spans) if spans else []
        trace: List[Dict[str, Any]] = []
        for p, span, score in zip(usable, spans, scores):
            pass_map[p] = _passes(score)
            trace.append({
                "position_ms": p, "span_start_ms": span[0],
                "span_end_ms": span[1], "s_target": score[0],
                "s_interviewer": score[1], "passes": pass_map[p],
            })
        return pass_map, trace

    result: Dict[str, Any] = {
        "run_start_ms": run_start_ms, "run_end_ms": run_end_ms,
        "leading_trim_ms": 0, "trailing_trim_ms": 0,
        "leading_demoted_block": False, "trailing_demoted_block": False,
    }
    new_start, new_end = run_start_ms, run_end_ms

    # Leading edge
    lead_map, lead_trace = _scan(_fine_positions(run_start_ms, params), True)
    result["leading_trace"] = lead_trace
    inward = [p for p in sorted(lead_map) if p >= run_start_ms]
    passing = next((p for p in inward if lead_map[p] is True), None)
    if passing is None or passing - run_start_ms > params.edge_max_trim_ms:
        new_start = run_start_ms + params.block_ms
        result["leading_demoted_block"] = True
    else:
        new_start = passing
        result["leading_trim_ms"] = passing - run_start_ms

    # Trailing edge (skip if the demotion consumed the whole run)
    if new_start < run_end_ms:
        trail_map, trail_trace = _scan(_fine_positions(run_end_ms, params), False)
        result["trailing_trace"] = trail_trace
        inward = [p for p in sorted(trail_map, reverse=True) if p <= run_end_ms]
        passing = next((p for p in inward if trail_map[p] is True), None)
        if passing is None or run_end_ms - passing > params.edge_max_trim_ms:
            if run_end_ms - params.block_ms > new_start:
                new_end = run_end_ms - params.block_ms
                result["trailing_demoted_block"] = True
        else:
            new_end = passing
            result["trailing_trim_ms"] = run_end_ms - passing

    result["new_start_ms"] = new_start
    result["new_end_ms"] = new_end
    result["survives"] = new_start < new_end
    return result


def activity_ratio_level(ratio: Optional[float], params: Layer2Params) -> str:
    if ratio is None:
        return "NO_SPEECH"
    if ratio > params.ratio_normal:
        return "NORMAL"
    if ratio >= params.ratio_low:
        return "LOW_ADVISORY"
    return "NEAR_ZERO_ALERT"


def drift_notice(
    previous_mean: Optional[float],
    current_first_mean: Optional[float],
    params: Layer2Params,
) -> Optional[Dict[str, float]]:
    """Cross-video drift: current file's mean S_target over its first
    drift_window_ms of HIGH activity vs the previous file's mean over
    ALL its HIGH blocks (review-flagged quantification)."""
    if previous_mean is None or current_first_mean is None:
        return None
    if previous_mean - current_first_mean > params.drift_delta:
        return {"previous_mean": previous_mean,
                "current_first30s_mean": current_first_mean,
                "delta": previous_mean - current_first_mean}
    return None


# =============================================================================
# Per-file tracking (scorer-driven; the self-test runs this without torch)
# =============================================================================

def track_file(
    file_audio: FileAudio,
    scorer: Scorer,
    calibration: Calibration,
    anti_available: bool,
    params: Layer2Params,
    worker_log: WorkerLog,
) -> FileTrack:
    duration = file_audio.duration_ms
    start_pts = file_audio.audio_start_pts_ms
    segments_rel = [
        (s - start_pts, e - start_pts)
        for s, e in file_audio.silero_segments_local_ms
    ]

    # Step 3 — coarse windows + Silero skip rule
    all_spans = plan_windows(duration, params.window_ms, params.hop_ms)
    scored_spans: List[Tuple[int, int]] = []
    skipped_spans: List[Tuple[int, int]] = []
    for span in all_spans:
        overlap = segment_overlap_ms(span[0], span[1], segments_rel)
        if (span[1] - span[0]) > 0 and (
            overlap / (span[1] - span[0])
        ) >= params.silero_skip_floor:
            scored_spans.append(span)
        else:
            skipped_spans.append(span)
    window_scores = scorer(scored_spans) if scored_spans else []
    for span in skipped_spans:
        worker_log.append(
            "layer2_window_skipped",
            {"reason": "SKIPPED_NONSPEECH",
             "start_local_ms": start_pts + span[0],
             "end_local_ms": start_pts + span[1]},
            start_ms=start_pts + span[0],
        )

    # Step 4 — median pooling onto the 1s block grid
    blocks = median_pool_blocks(duration, scored_spans, window_scores, params)

    # Step 5 — tiering + margin rule
    tiers: List[str] = []
    tier_counts = {t: 0 for t in
                   (TIER_HIGH, TIER_MEDIUM, TIER_SUB, TIER_REJECT, TIER_SKIPPED)}
    for block in blocks:
        if block["evaluations"] == 0:
            tier, margin_failed = TIER_SKIPPED, False
        else:
            tier, margin_failed = tier_block(
                block["s_target_median"], block["s_interviewer_median"],
                calibration, anti_available, params,
            )
        block["tier"] = tier
        block["margin_failed"] = margin_failed
        tiers.append(tier)
        tier_counts[tier] += 1
        block_start = start_pts + block["block_index"] * params.block_ms
        worker_log.append(OP_BLOCK, {
            "tier": tier,
            "start_local_ms": block_start,
            "end_local_ms": block_start + params.block_ms,
            "s_target_median": block["s_target_median"],
            "s_interviewer_median": block["s_interviewer_median"],
            "evaluations": block["evaluations"],
            "margin_failed": margin_failed,
            "no_anti_profile": not anti_available,
        }, start_ms=block_start)

    # Step 6 — edge-trim refinement of maximal HIGH runs
    high_runs: List[HighRun] = []
    for first, last in find_high_runs(tiers):
        run_start = first * params.block_ms
        run_end = last * params.block_ms
        refinement = refine_run_edges(
            run_start, run_end, duration, scorer, calibration,
            anti_available, params,
        )
        worker_log.append(OP_EDGE, {
            **{k: v for k, v in refinement.items()
               if k not in ("leading_trace", "trailing_trace")},
            "leading_trace": refinement.get("leading_trace"),
            "trailing_trace": refinement.get("trailing_trace"),
            "start_local_ms": start_pts + run_start,
        }, start_ms=start_pts + run_start)
        for demoted, edge_block in (
            (refinement["leading_demoted_block"], first),
            (refinement["trailing_demoted_block"], last - 1),
        ):
            if demoted:
                blocks[edge_block]["tier"] = TIER_MEDIUM
                blocks[edge_block]["demoted_by_edge_trim"] = True
                tier_counts[TIER_HIGH] -= 1
                tier_counts[TIER_MEDIUM] += 1
        if not refinement["survives"]:
            continue
        new_start, new_end = refinement["new_start_ms"], refinement["new_end_ms"]
        run = HighRun(
            start_local_ms=start_pts + new_start,
            end_local_ms=start_pts + new_end,
        )
        block_first = new_start // params.block_ms
        block_last = (new_end + params.block_ms - 1) // params.block_ms
        for index in range(block_first, block_last):
            if blocks[index]["tier"] != TIER_HIGH:
                continue
            b_start = max(index * params.block_ms, new_start)
            b_end = min((index + 1) * params.block_ms, new_end)
            run.blocks.append({
                "source_file": file_audio.source_path,
                "start_ms": start_pts + b_start,
                "end_ms": start_pts + b_end,
                "start_global_ms": file_audio.to_global_ms(start_pts + b_start),
                "end_global_ms": file_audio.to_global_ms(start_pts + b_end),
                "duration_ms": b_end - b_start,
                "confidence_tier": TIER_HIGH,
                "S_target_median": blocks[index]["s_target_median"],
                "S_interviewer_median": blocks[index]["s_interviewer_median"],
                "edge_trim": {
                    "leading_trim_ms": (
                        refinement["leading_trim_ms"]
                        if b_start == new_start else 0),
                    "trailing_trim_ms": (
                        refinement["trailing_trim_ms"]
                        if b_end == new_end else 0),
                },
                "no_anti_profile": not anti_available,
            })
        high_runs.append(run)

    # Step 7 — activity ratio
    high_ms = sum(r.end_local_ms - r.start_local_ms for r in high_runs)
    silero_ms = sum(e - s for s, e in segments_rel)
    ratio = (high_ms / silero_ms) if silero_ms > 0 else None
    level = activity_ratio_level(ratio, params)
    high_scores = [
        b["S_target_median"] for r in high_runs for b in r.blocks
    ]
    worker_log.append(OP_FILE, {
        "high_activity_ms": high_ms,
        "silero_speech_ms": silero_ms,
        "unattributed_speech_ms": max(0, silero_ms - high_ms),
        "activity_ratio": ratio,
        "ratio_level": level,
        "tier_counts": tier_counts,
        "windows_scored": len(scored_spans),
        "windows_skipped": len(skipped_spans),
    })
    return FileTrack(
        file_index=file_audio.file_index,
        source_file=file_audio.source_path,
        high_runs=high_runs,
        tier_counts=tier_counts,
        high_ms=high_ms,
        silero_ms=silero_ms,
        ratio=ratio,
        ratio_level=level,
        unattributed_speech_ms=max(0, silero_ms - high_ms),
        high_scores=high_scores,
        blocks=[
            {
                "start_local_ms": start_pts + b["block_index"] * params.block_ms,
                "end_local_ms": start_pts + (b["block_index"] + 1) * params.block_ms,
                "tier": b["tier"],
                "s_target_median": b["s_target_median"],
                "s_interviewer_median": b["s_interviewer_median"],
                "margin_failed": b.get("margin_failed", False),
            }
            for b in blocks
        ],
    )


# =============================================================================
# Production scorer (torch lazy; fixed 256-window batches, repeat-padded)
# =============================================================================

def _ecapa_scorer(
    models: Any,
    file_audio: FileAudio,
    e_composite: Sequence[float],
    e_anti: Optional[Sequence[float]],
) -> Scorer:
    def scorer(
        spans: Sequence[Tuple[int, int]]
    ) -> List[Tuple[float, Optional[float]]]:
        import torch

        batch_size = environment_gate.ECAPA_BATCH_WINDOWS
        results: List[Tuple[float, Optional[float]]] = [None] * len(spans)  # type: ignore
        # Uniform tensor shapes per forward pass: group spans by length,
        # ascending; original order restored via index bookkeeping.
        by_length: Dict[int, List[int]] = {}
        for index, (start, end) in enumerate(spans):
            by_length.setdefault(end - start, []).append(index)
        for length_ms in sorted(by_length):
            indices = by_length[length_ms]
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start: batch_start + batch_size]
                tensors = []
                for span_index in batch_indices:
                    start, end = spans[span_index]
                    pcm = pcm_slice(
                        file_audio.pcm, 0, file_audio.num_samples,
                        start, end,
                    )
                    tensors.append(
                        torch.frombuffer(bytearray(pcm), dtype=torch.int16)
                        .to(torch.float32) / 32768.0
                    )
                pad = batch_size - len(tensors)
                if pad and len(indices) > batch_size:
                    # Fixed batch shape: repeat the last window; padded
                    # outputs are discarded below.
                    tensors.extend([tensors[-1]] * pad)
                stacked = torch.stack(tensors)
                with torch.no_grad():
                    embeddings = models.ecapa.encode_batch(stacked)
                flat = embeddings.reshape(len(tensors), -1).to(torch.float32)
                for row, span_index in enumerate(batch_indices):
                    vector = l2_normalize(flat[row].cpu().tolist())
                    s_target = cosine(vector, e_composite)
                    s_interviewer = (
                        cosine(vector, e_anti) if e_anti is not None else None
                    )
                    results[span_index] = (s_target, s_interviewer)
        return results

    return scorer


# =============================================================================
# Layer 2 entrypoint
# =============================================================================

def run_layer2(
    manifest: SessionManifest,
    batch: BatchAudio,
    models: Any,
    enrollment: EnrollmentResult,
    work_dir: Path | str,
    params: Layer2Params = Layer2Params(),
    authoritative: bool = True,
    scorer_factory: Optional[Callable[[FileAudio], Scorer]] = None,
) -> Layer2Result:
    """The single authoritative pass (Steps 1-9). ``scorer_factory`` is
    injectable for the self-test; production defaults to the resident
    ECAPA model."""
    layer2_dir = Path(work_dir) / "layer2"
    layer2_dir.mkdir(parents=True, exist_ok=True)
    anti_available = enrollment.e_anti is not None

    manifest.append(OP_INIT, {
        "layer": 2,
        "authoritative": authoritative,
        "params": params.manifest_payload(),
        "enrollment_ref": enrollment.e_composite_sha256,
        "no_anti_profile": not anti_available,
        "ecapa_batch_windows": environment_gate.ECAPA_BATCH_WINDOWS,
        **({} if authoritative else
           {"superseded_by": "authoritative_pass", "note": "preview pass"}),
    })

    # Step 1 — E_composite sanity check
    if anti_available:
        contamination = cosine(enrollment.e_composite, enrollment.e_anti)
        if contamination > params.anti_contam_halt:
            manifest.append(Operation.BLOCKING_HALT, {
                "reason": "layer2_enrollment_contamination_critical",
                "sim_composite_anti": contamination,
            })
            raise Layer2Error("compromised enrollment — re-run Layer 1")
        if contamination > params.anti_contam_warning:
            manifest.append(Operation.WARNING, {
                "warning": "layer2_enrollment_contamination",
                "sim_composite_anti": contamination,
            })
    else:
        manifest.append(Operation.WARNING, {
            "warning": "sanity_check_unavailable_no_anti_profile",
        })

    # Step 2 — deterministic threshold calibration
    calibration = derive_thresholds(
        genuine_scores=loo_scores(
            [entry.vector for entry in enrollment.pool],
            [entry.duration_ms for entry in enrollment.pool],
        ),
        impostor_scores=[
            cosine(entry.vector, enrollment.e_composite)
            for entry in enrollment.anti_pool
        ],
        anti_available=anti_available,
        enrollment_ref=enrollment.e_composite_sha256,
        params=params,
    )
    manifest.append(Operation.CALIBRATION, calibration.record)
    if calibration.overlap_warning:
        manifest.append(Operation.WARNING, {
            "warning": "CALIBRATION_OVERLAP",
            "note": "genuine/impostor distributions overlap; margin rule "
                    "is the primary discriminator",
        })

    # Steps 3-8 — sequential per-file tracking through worker logs
    files: List[FileTrack] = []
    worker_paths: List[Path] = []
    previous_high_mean: Optional[float] = None
    for file_audio in batch.files:
        worker_path = layer2_dir / f"worker_{file_audio.file_index:03d}.jsonl"
        scorer = (
            scorer_factory(file_audio) if scorer_factory is not None
            else _ecapa_scorer(
                models, file_audio, enrollment.e_composite, enrollment.e_anti,
            )
        )
        with WorkerLog(worker_path, file_audio.file_index) as worker_log:
            track = track_file(
                file_audio, scorer, calibration, anti_available,
                params, worker_log,
            )
            # Step 8 — cross-video drift (first 30s of HIGH activity)
            first_scores: List[float] = []
            accumulated = 0
            for run in track.high_runs:
                for block in run.blocks:
                    if accumulated >= params.drift_window_ms:
                        break
                    first_scores.append(block["S_target_median"])
                    accumulated += block["duration_ms"]
            current_mean = (
                sum(first_scores) / len(first_scores) if first_scores else None
            )
            notice = drift_notice(previous_high_mean, current_mean, params)
            if notice is not None:
                worker_log.append(Operation.DRIFT_NOTICE, {
                    "warning": "cross_video_vocal_drift", **notice,
                })
            previous_high_mean = (
                sum(track.high_scores) / len(track.high_scores)
                if track.high_scores else previous_high_mean
            )
        worker_paths.append(worker_path)
        files.append(track)

    # Canonical Manifest Merge Rule — single writer, sorted records
    merge_worker_logs(manifest, worker_paths)

    # Step 9 — authoritative output document + SHA-256
    output_doc = {
        "schema": "spovnob-layer2-output-v1",
        "authoritative": authoritative,
        "enrollment_ref": enrollment.e_composite_sha256,
        "calibration_ref": calibration.calibration_ref,
        "thresholds_used": {
            "theta_high": calibration.theta_high,
            "theta_med": calibration.theta_med,
            "margin_minimum": params.margin_minimum,
            "evidence_floor": params.evidence_floor,
            "calibration": calibration.kind,
            "operator_modified": False,
        },
        "no_anti_profile": not anti_available,
        "files": [
            {
                "file_index": track.file_index,
                "source_file": track.source_file,
                "tier_counts": track.tier_counts,
                "high_activity_ms": track.high_ms,
                "silero_speech_ms": track.silero_ms,
                "unattributed_speech_ms": track.unattributed_speech_ms,
                "activity_ratio": track.ratio,
                "ratio_level": track.ratio_level,
                "high_runs": [
                    {
                        "start_local_ms": run.start_local_ms,
                        "end_local_ms": run.end_local_ms,
                        "blocks": run.blocks,
                    }
                    for run in track.high_runs
                ],
            }
            for track in files
        ],
    }
    output_sha = sha256_of_obj(output_doc)
    output_path = layer2_dir / "layer2_output.json"
    output_path.write_text(canonical_json(output_doc) + "\n", encoding="utf-8")
    manifest.append(Operation.OUTPUT_HASH, {
        "layer": 2,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "authoritative": authoritative,
    })
    for track in files:
        if track.ratio_level == "NEAR_ZERO_ALERT":
            manifest.append(Operation.WARNING, {
                "warning": "near_zero_activity_manual_review",
                "file_index": track.file_index,
                "activity_ratio": track.ratio,
            })
    return Layer2Result(
        calibration=calibration,
        files=files,
        no_anti_profile=not anti_available,
        authoritative=authoritative,
        output_path=str(output_path),
        output_sha256=output_sha,
    )


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB Layer 2 tracker (Module 3)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip, no torch, no GPU)")
    mode.add_argument("--run", action="store_true",
                      help="run Layers 0+1+2 on a batch (Ubuntu box)")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--clicks", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--preview", action="store_true",
                        help="mark this as a non-authoritative preview pass")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_stdlib()

    for required in ("videos", "clicks", "work_dir", "model_store", "manifest"):
        if getattr(args, required) is None:
            parser.error(f"--run requires --{required.replace('_', '-')}")

    from layer0_preprocessor import preprocess_batch
    from layer1_enrollment import load_clicks, run_layer1

    clicks = load_clicks(args.clicks)
    with SessionManifest(args.manifest, operator_id=args.operator) as manifest:
        models = environment_gate.run_gate(args.model_store, manifest)
        batch = preprocess_batch(manifest, args.videos, args.work_dir,
                                 models.silero)
        enrollment = run_layer1(manifest, batch, models, clicks, args.work_dir)
        result = run_layer2(manifest, batch, models, enrollment,
                            args.work_dir, authoritative=not args.preview)
    total_high = sum(track.high_ms for track in result.files)
    print(
        f"layer2 complete — theta_high={result.calibration.theta_high:.3f} "
        f"({result.calibration.kind}), {total_high} ms HIGH activity, "
        f"output sha256 {result.output_sha256[:12]}…"
    )
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch/GPU)
# =============================================================================

def _fake_file(duration_ms: int, segments: List[Tuple[int, int]],
               start_pts: int = 0, index: int = 0) -> FileAudio:
    return FileAudio(
        file_index=index, source_path=f"video_{index:02d}.mp4", wav_path="",
        source_sha256="", wav_sha256="", num_samples=duration_ms * 16,
        duration_ms=duration_ms, audio_start_pts_ms=start_pts,
        audio_start_missing=False, vfr_suspected=False,
        silero_segments_local_ms=[(start_pts + s, start_pts + e)
                                  for s, e in segments],
    )


def _selftest_stdlib() -> int:
    import tempfile

    assert "torch" not in sys.modules, "torch imported at module level"
    params = Layer2Params()

    # 1. quantiles
    values = [float(v) for v in range(1, 11)]
    assert abs(quantile_sorted(values, 0.10) - 1.9) < 1e-12
    assert quantile_sorted(values, 0.0) == 1.0
    assert quantile_sorted(values, 1.0) == 10.0
    assert quantile_sorted([0.5], 0.10) == 0.5

    # 2. LOO genuine scores
    e1, e2 = [1.0, 0.0], [0.0, 1.0]
    assert all(abs(s - 1.0) < 1e-12
               for s in loo_scores([e1, e1, e1], [1000, 1000, 1000]))
    mixed = loo_scores([e1, e1, e2], [1000, 1000, 1000])
    assert abs(mixed[0] - (0.5 ** 0.5)) < 1e-9     # cos(e1, norm(e1+e2))
    assert loo_scores([e1], [3000]) == []          # seed-only pool: no LOO
    assert loo_scores([], []) == []

    # 3. threshold derivation
    genuine = [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76]
    cal = derive_thresholds(genuine, [0.20, 0.30], True, "ref", params)
    expected_q10 = quantile_sorted(sorted(genuine), 0.10)
    assert cal.kind == CAL_DERIVED
    assert abs(cal.theta_high - max(expected_q10, 0.35)) < 1e-12
    assert abs(cal.theta_med - (cal.theta_high - 0.15)) < 1e-12

    low = derive_thresholds([0.30] * 10, [0.20], True, "ref", params)
    assert low.theta_high == params.theta_clamp_low and not low.overlap_warning
    high = derive_thresholds([0.90] * 10, [0.85], True, "ref", params)
    assert high.theta_high == params.theta_clamp_high and high.overlap_warning
    no_anti = derive_thresholds([0.50] * 10, [], False, "ref", params)
    assert no_anti.kind == CAL_DERIVED_NO_ANTI
    assert abs(no_anti.theta_high - 0.55) < 1e-12
    fallback = derive_thresholds([0.9] * 5, [0.2], True, "ref", params)
    assert fallback.kind == CAL_FALLBACK
    assert fallback.theta_high == 0.60 and fallback.theta_med == 0.40
    seed_only = derive_thresholds([], [0.2], True, "ref", params)
    assert seed_only.kind == CAL_FALLBACK          # empty LOO -> fallback
    assert derive_thresholds(genuine, [0.20, 0.30], True, "ref", params
                             ).calibration_ref == cal.calibration_ref

    # 4. window planning
    assert plan_windows(10000, 5000, 1000) == [
        (0, 5000), (1000, 6000), (2000, 7000), (3000, 8000),
        (4000, 9000), (5000, 10000)]
    assert plan_windows(4000, 5000, 1000) == [(0, 4000)]
    assert plan_windows(5000, 5000, 1000) == [(0, 5000)]

    # 5. tiering boundaries (theta_high=0.6, theta_med=0.45)
    test_cal = derive_thresholds([0.62] * 10, [0.50], True, "ref", params)
    assert abs(test_cal.theta_high - 0.62) < 1e-12

    def _tier(s_t, s_i, anti=True):
        return tier_block(s_t, s_i, test_cal, anti, params)

    assert _tier(0.80, 0.20) == (TIER_HIGH, False)
    assert _tier(0.80, 0.70) == (TIER_MEDIUM, True)     # margin failed
    assert _tier(0.50, 0.20) == (TIER_MEDIUM, False)
    assert _tier(0.30, 0.20) == (TIER_SUB, False)
    assert _tier(0.10, 0.20) == (TIER_REJECT, False)
    assert _tier(0.80, None, anti=False) == (TIER_HIGH, False)

    # 6-8. full flow over a synthetic 20s file: coarse HIGH run = blocks
    # 7..11 (hand-verified median coverage), then edge-trim scenarios.
    duration = 20000
    high_starts = {5000, 6000, 7000, 8000, 9000}

    def make_scorer(fine_fail: set) -> Scorer:
        def scorer(spans):
            out = []
            for start, end in spans:
                if end - start == params.edge_fine_window_ms:
                    score = 0.1 if (start, end) in fine_fail else 0.9
                else:                       # coarse 5s windows
                    score = 0.9 if start in high_starts else 0.1
                out.append((score, 0.2))
            return out
        return scorer

    cal6 = derive_thresholds([0.62] * 10, [0.30], True, "ref", params)

    def run_track(fine_fail: set, tmp: Path, tag: str) -> FileTrack:
        file_audio = _fake_file(duration, [(0, duration)])
        with WorkerLog(tmp / f"w_{tag}.jsonl", 0) as worker_log:
            return track_file(file_audio, make_scorer(fine_fail), cal6,
                              True, params, worker_log)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # (a) no fine failures: run [7000,12000), no trims
        track = run_track(set(), tmp, "a")
        assert len(track.high_runs) == 1
        run = track.high_runs[0]
        assert (run.start_local_ms, run.end_local_ms) == (7000, 12000)
        assert len(run.blocks) == 5 and track.high_ms == 5000
        assert track.ratio_level == "LOW_ADVISORY"      # 5000/20000 = 0.25
        assert track.unattributed_speech_ms == 15000

        # (b) leading fine window at 7000 fails -> 250ms trim
        track = run_track({(7000, 9000)}, tmp, "b")
        run = track.high_runs[0]
        assert (run.start_local_ms, run.end_local_ms) == (7250, 12000)
        assert run.blocks[0]["duration_ms"] == 750
        assert run.blocks[0]["edge_trim"]["leading_trim_ms"] == 250
        assert run.blocks[-1]["edge_trim"]["trailing_trim_ms"] == 0
        assert track.high_ms == 4750

        # (c) all near-edge leading windows fail -> first block demoted
        fails = {(7000, 9000), (7250, 9250), (7500, 9500), (7750, 9750)}
        track = run_track(fails, tmp, "c")
        run = track.high_runs[0]
        assert (run.start_local_ms, run.end_local_ms) == (8000, 12000)
        assert track.tier_counts[TIER_HIGH] == 4        # one demoted
        assert track.high_ms == 4000

    # 9. activity ratio levels
    assert activity_ratio_level(0.30, params) == "NORMAL"
    assert activity_ratio_level(0.15, params) == "LOW_ADVISORY"
    assert activity_ratio_level(0.05, params) == "NEAR_ZERO_ALERT"
    assert activity_ratio_level(None, params) == "NO_SPEECH"

    # 10. drift
    assert drift_notice(0.80, 0.65, params) is not None
    assert drift_notice(0.80, 0.75, params) is None
    assert drift_notice(None, 0.75, params) is None

    # 11. end-to-end run_layer2 with injected scorers: deterministic
    # output hash + canonical worker-log merge into a verified manifest.
    pool_vectors = [[1.0, 0.0]] * 10
    pool = [type("E", (), {"vector": v, "duration_ms": 3000})()
            for v in pool_vectors]
    anti = [type("E", (), {"vector": [0.0, 1.0], "duration_ms": 2000})()]
    enrollment = EnrollmentResult(
        f_target=[1.0], e_seed=[1.0, 0.0], e_composite=[1.0, 0.0],
        e_composite_sha256="e" * 64, e_anti=[0.0, 1.0],
        e_anti_sha256="a" * 64, no_anti_profile=False, pool=pool,
        anti_pool=anti, total_verified_ms=30000, quality_history=[],
    )
    batch = BatchAudio(files=[
        _fake_file(duration, [(0, duration)], index=0),
        _fake_file(duration, [(0, duration)], index=1),
    ])

    def factory(_file_audio: FileAudio) -> Scorer:
        return make_scorer(set())

    hashes = []
    for attempt in range(2):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with SessionManifest(tmp / "m.jsonl") as manifest:
                result = run_layer2(
                    manifest, batch, models=None, enrollment=enrollment,
                    work_dir=tmp, params=params, scorer_factory=factory,
                )
            entries = SessionManifest.verify_chain(tmp / "m.jsonl")
            hashes.append(result.output_sha256)
            ops = [e["operation"] for e in entries]
            assert Operation.CALIBRATION in ops
            assert Operation.OUTPUT_HASH in ops
            merged = [e["payload"] for e in entries
                      if e["operation"] == OP_FILE]
            assert [m["file_index"] for m in merged] == [0, 1]
    assert hashes[0] == hashes[1], "output hash not deterministic"

    for forbidden in ("torch", "cv2", "numpy"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"
    print("layer2_tracker stdlib self-test OK — full flow exercised with "
          "injected scorers; no torch, no GPU, no pip")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
