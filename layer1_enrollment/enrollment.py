"""
SPOVNOB — Module 2 (Layer 1): enrollment.py
============================================

The Layer 1 orchestrator: operator clicks, identity anchoring, the
dual-track visual confirmation loop (Track A enrollment / Track B
anti-profile auto-collection / Track C anti-profile click), the Triple
Validation Gate, the cumulative pool, quality states, and the freeze.

Inputs:     open SessionManifest · BatchAudio (Layer 0) · ResidentModels
            (gate) · OperatorClicks (JSON) · EnrollmentParams · work dir
Outputs:    EnrollmentResult — frozen E_composite (+SHA-256), E_anti,
            both pools with durations (Layer 2 calibration needs them),
            quality history, no_anti_profile flag. Every vector creation,
            discard, warning, and state change is a manifest entry; every
            accepted window's WAV slice + JSON sidecar (PTS range, MAR
            trace) is persisted under <work_dir>/enroll/.

Implements (Audio_Diarization.md, Layer 1): "Operator Input — Minimal
and Validated" · "E_seed capture strategy" · "Anti-Profile Dual-Track
Strategy" + M-Trap · "The Triple Validation Gate" · "Progressive
Enrollment Quality States" + NO_ANTI escalation · "Cumulative Pool" ·
"Persistence, audit, and manifest rules" · "The 9 Guardrails" ·
"Cross-File Behavior" (anchor propagation, gap entries are Layer 0's).

Guardrail coverage map:
  1 Speaking-click overlap  -> visual proxy: non-target lips-open frac of
                               the seed window must be <= click_overlap_max_frac
  2 Speaking-click duration -> seed window >= seed_min_ms (3s; supersedes
                               the legacy "3-8s" remnant — no maximum)
  3 Anti-click identity     -> clicked face must NOT match F_target
  4 M-Trap on Track B       -> gates.mtrap_discard, logged discard
  5 InsightFace confidence  -> vision.py det_score filter
  6 Low detection quality   -> running mean ReID sim < reid_warning_floor
  7 Separation margin       -> Gate C margin_minimum
  8 Acoustic similarity     -> contamination_level WARNING / HALT
  9 Critical failure        -> batch end, still INSUFFICIENT -> blocking halt

Sequencing: STRICTLY SEQUENTIAL in canonical file order (System
Environment rule — the cumulative pool is order-dependent by design).

CUDA determinism dependencies: inherits the four constants via
environment_gate; ECAPA windows encoded batch-of-1 (variable length),
vision batches fixed by the gate constants.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import json
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layer0_preprocessor import SAMPLE_RATE, BatchAudio, FileAudio
from session_manifest import (
    Operation,
    SessionManifest,
    canonical_json,
    sha256_of_file,
    sha256_of_obj,
)

from . import vision
from .encoding import (
    cosine,
    duration_weighted_mean,
    encode_window,
    l2_normalize,
    pcm_slice,
)
from .errors import Layer1Error, Layer1ReclickError
from .gates import (
    CONTAM_HALT,
    CONTAM_WARNING,
    contamination_level,
    evaluate_triple_gate,
    mtrap_discard,
    pairwise_cosine_variance,
    segment_overlap_ms,
    vad_near,
)
from .geometry import yaw_suspends_mar
from .params import EnrollmentParams
from .quality import INSUFFICIENT, MARGINAL, assess_quality
from .vision import FaceObs, FrameFaces
from .window_machine import CandidateWindow, FrameObs, WindowMachine

OP_INIT = "layer1_init"
OP_SEED = "layer1_seed"
OP_SCAN = "layer1_video_scan"
OP_QUALITY = "layer1_quality"
OP_FREEZE = "layer1_freeze"

KIND_SEED = "seed"
KIND_TRACK_A = "track_a_window"
KIND_ANTI_B = "anti_track_b"
KIND_ANTI_C = "anti_track_c"

CLICK_MATCH_MAX_GAP_MS = 200   # nearest analyzed frame must be this close


# =============================================================================
# Operator click input
# =============================================================================

@dataclass(frozen=True)
class SpeakingClick:
    file_index: int
    pts_ms: int
    x: float
    y: float


@dataclass(frozen=True)
class AntiClick:
    file_index: int
    pts_ms: int
    x: float
    y: float


@dataclass(frozen=True)
class OperatorClicks:
    speaking: SpeakingClick
    anti: Optional[AntiClick] = None


def load_clicks(path: Path | str) -> OperatorClicks:
    """Parse the operator clicks JSON file:
    {"speaking_click": {"file_index": 0, "pts_ms": 41250, "x": 812, "y": 440},
     "anti_click":     {"file_index": 0, "pts_ms": 95000, "x": 300, "y": 400}}
    ("anti_click" is optional — Track C is optional by design.)"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def _click(block: Dict[str, Any], cls: type) -> Any:
        for key in ("file_index", "pts_ms"):
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise Layer1Error(f"clicks JSON: {key} must be an integer")
        return cls(
            file_index=block["file_index"], pts_ms=block["pts_ms"],
            x=float(block["x"]), y=float(block["y"]),
        )

    if "speaking_click" not in data:
        raise Layer1Error("clicks JSON: speaking_click is mandatory")
    speaking = _click(data["speaking_click"], SpeakingClick)
    anti = _click(data["anti_click"], AntiClick) if data.get("anti_click") else None
    return OperatorClicks(speaking=speaking, anti=anti)


# =============================================================================
# Result structures
# =============================================================================

@dataclass
class PoolEntry:
    vector: List[float]                  # L2-normalized ECAPA d-vector
    duration_ms: int
    kind: str
    file_index: int
    t_start_local_ms: int
    t_stop_local_ms: int


@dataclass
class EnrollmentResult:
    f_target: List[float]
    e_seed: List[float]
    e_composite: List[float]
    e_composite_sha256: str
    e_anti: Optional[List[float]]
    e_anti_sha256: Optional[str]
    no_anti_profile: bool
    pool: List[PoolEntry]
    anti_pool: List[PoolEntry]
    total_verified_ms: int
    quality_history: List[Dict[str, Any]]


# =============================================================================
# Internal helpers
# =============================================================================

def _mean_embedding(embeddings: Sequence[Sequence[float]]) -> List[float]:
    dims = len(embeddings[0])
    count = len(embeddings)
    return l2_normalize(
        [math.fsum(e[d] for e in embeddings) / count for d in range(dims)]
    )


def _face_at_click(
    frames: List[FrameFaces], pts_ms: int, x: float, y: float
) -> Tuple[FaceObs, int]:
    """The face the operator clicked: nearest analyzed frame with faces
    (within CLICK_MATCH_MAX_GAP_MS), preferring a bbox containing the
    point, else the face whose center is closest to it."""
    candidates = [f for f in frames if f.faces]
    if not candidates:
        raise Layer1ReclickError("no detected faces anywhere near the click")
    frame = min(candidates, key=lambda f: abs(f.pts_ms - pts_ms))
    if abs(frame.pts_ms - pts_ms) > CLICK_MATCH_MAX_GAP_MS:
        raise Layer1ReclickError(
            f"no analyzed frame with faces within {CLICK_MATCH_MAX_GAP_MS}ms "
            f"of click pts {pts_ms}"
        )
    containing = [face for face in frame.faces if face.contains_point(x, y)]
    if containing:
        face = max(containing, key=lambda f: f.det_score)
    else:
        face = min(
            frame.faces,
            key=lambda f: (f.center()[0] - x) ** 2 + (f.center()[1] - y) ** 2,
        )
    return face, frame.pts_ms


def _match_face(
    faces: Sequence[FaceObs], anchor: Sequence[float], threshold: float
) -> Tuple[Optional[FaceObs], float]:
    """Best face matching the anchor embedding at or above threshold."""
    best, best_sim = None, -1.0
    for face in faces:
        sim = cosine(face.embedding, anchor)
        if sim > best_sim:
            best, best_sim = face, sim
    if best is not None and best_sim >= threshold:
        return best, best_sim
    return None, best_sim


@dataclass
class _ObsBundle:
    obs: List[FrameObs]
    reid_sims: List[float]
    trackb_centers: List[int]
    any_pose: bool


def _build_obs(
    frames: List[FrameFaces],
    f_target: Sequence[float],
    f_interviewer: Optional[Sequence[float]],
    file_audio: FileAudio,
    params: EnrollmentParams,
) -> _ObsBundle:
    """Assign identities per frame and assemble the window machine's
    observation stream + Track B candidate centers (a non-target face,
    lips closed by raw MAR, Silero energy at the same PTS)."""
    segments = file_audio.silero_segments_local_ms
    obs: List[FrameObs] = []
    reid_sims: List[float] = []
    trackb_centers: List[int] = []
    any_pose = False

    for frame in frames:
        target, sim = _match_face(frame.faces, f_target, params.face_reid_threshold)
        if target is not None:
            reid_sims.append(sim)
        others = [f for f in frame.faces if f is not target]
        if f_interviewer is not None:
            interviewer, _ = _match_face(
                others, f_interviewer, params.face_reid_threshold
            )
        else:
            interviewer = max(others, key=lambda f: f.det_score) if others else None
        if any(face.yaw_degrees is not None for face in frame.faces):
            any_pose = True

        speech_near = vad_near(frame.pts_ms, segments, params.vad_tol_ms)
        obs.append(FrameObs(
            pts_ms=frame.pts_ms,
            target_present=target is not None,
            target_mar=target.mar if target is not None else None,
            target_suspended=(
                target is not None
                and yaw_suspends_mar(target.yaw_degrees, params)
            ),
            interviewer_present=interviewer is not None,
            interviewer_mar=interviewer.mar if interviewer is not None else None,
            vad_speech=speech_near,
        ))
        # Track B trigger (document pseudocode): non-target face, lips
        # closed (raw MAR — review-flagged simplification), audio energy.
        if (
            interviewer is not None
            and interviewer.mar is not None
            and interviewer.mar < params.mar_off
            and speech_near
        ):
            trackb_centers.append(frame.pts_ms)
    return _ObsBundle(obs, reid_sims, trackb_centers, any_pose)


def _run_machine(
    obs: Sequence[FrameObs], params: EnrollmentParams
) -> List[CandidateWindow]:
    machine = WindowMachine(params)
    windows: List[CandidateWindow] = []
    for one in obs:
        emitted = machine.step(one)
        if emitted is not None:
            windows.append(emitted)
    if obs:
        tail = machine.finalize(obs[-1].pts_ms)
        if tail is not None:
            windows.append(tail)
    return windows


def _clamp_local(file_audio: FileAudio, pts_ms: int) -> int:
    low = file_audio.audio_start_pts_ms
    high = file_audio.audio_start_pts_ms + file_audio.duration_ms
    return min(max(pts_ms, low), high)


def _persist_window(
    enroll_dir: Path,
    file_audio: FileAudio,
    kind: str,
    t_start_ms: int,
    t_stop_ms: int,
    mar_trace: Optional[List[Tuple[int, Optional[float]]]],
    end_reason: Optional[str],
) -> Dict[str, Any]:
    """Write the raw WAV slice + JSON sidecar for one enrollment window
    (the document's persistence rule). Returns hashes/paths for the
    manifest entry."""
    stem = f"{kind}_{file_audio.file_index:03d}_{t_start_ms}_{t_stop_ms}"
    wav_path = enroll_dir / f"{stem}.wav"
    pcm = pcm_slice(
        file_audio.pcm, file_audio.audio_start_pts_ms, file_audio.num_samples,
        t_start_ms, t_stop_ms,
    )
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    record = {
        "kind": kind,
        "file_index": file_audio.file_index,
        "source_file": file_audio.source_path,
        "t_start_local_ms": t_start_ms,
        "t_stop_local_ms": t_stop_ms,
        "t_start_global_ms": file_audio.to_global_ms(t_start_ms),
        "t_stop_global_ms": file_audio.to_global_ms(t_stop_ms),
        "duration_ms": t_stop_ms - t_start_ms,
        "end_reason": end_reason,
        "mar_trace": (
            [[pts, value] for pts, value in mar_trace] if mar_trace else None
        ),
    }
    record_path = enroll_dir / f"{stem}.json"
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    return {
        "wav_path": str(wav_path),
        "wav_sha256": sha256_of_file(wav_path),
        "record_path": str(record_path),
        "record_sha256": sha256_of_file(record_path),
    }


def _vector_payload(
    kind: str,
    file_audio: FileAudio,
    t_start_ms: int,
    t_stop_ms: int,
    vector: Sequence[float],
    persist: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "file_index": file_audio.file_index,
        "source_file": file_audio.source_path,
        "t_start_local_ms": t_start_ms,
        "t_stop_local_ms": t_stop_ms,
        "t_start_global_ms": file_audio.to_global_ms(t_start_ms),
        "t_stop_global_ms": file_audio.to_global_ms(t_stop_ms),
        "duration_ms": t_stop_ms - t_start_ms,
        "vector_dim": len(vector),
        "vector_sha256": sha256_of_obj(list(vector)),
        **persist,
        **extra,
    }


def _reclick(
    manifest: SessionManifest, reason: str, detail: Dict[str, Any]
) -> None:
    manifest.append(Operation.WARNING,
                    {"warning": "reclick_required", "reason": reason, **detail})
    raise Layer1ReclickError(f"re-click required: {reason} — {detail}")


# =============================================================================
# Layer 1 entrypoint
# =============================================================================

def run_layer1(
    manifest: SessionManifest,
    batch: BatchAudio,
    models: Any,
    clicks: OperatorClicks,
    work_dir: Path | str,
    params: EnrollmentParams = EnrollmentParams(),
) -> EnrollmentResult:
    enroll_dir = Path(work_dir) / "enroll"
    enroll_dir.mkdir(parents=True, exist_ok=True)

    manifest.append(OP_INIT, {
        "layer": 1,
        "params": params.manifest_payload(),
        "speaking_click": {
            "file_index": clicks.speaking.file_index,
            "pts_ms": clicks.speaking.pts_ms,
        },
        "anti_click_present": clicks.anti is not None,
    })

    first_index = batch.files[0].file_index
    if clicks.speaking.file_index != first_index:
        manifest.append(Operation.BLOCKING_HALT, {
            "reason": "speaking_click_not_on_first_video",
            "click_file_index": clicks.speaking.file_index,
            "first_file_index": first_index,
        })
        raise Layer1Error("speaking click must be on the first video (Video 1)")
    if clicks.anti is not None and clicks.anti.file_index != first_index:
        manifest.append(Operation.BLOCKING_HALT, {
            "reason": "anti_click_not_on_first_video",
            "click_file_index": clicks.anti.file_index,
        })
        raise Layer1Error("anti-profile click must be on the first video")

    f_target: Optional[List[float]] = None
    f_interviewer: Optional[List[float]] = None
    e_seed: Optional[List[float]] = None
    seed_span: Optional[Tuple[int, int, int]] = None    # file_index, t0, t1
    pool: List[PoolEntry] = []
    anti_pool: List[PoolEntry] = []
    e_anti: Optional[List[float]] = None
    e_composite: Optional[List[float]] = None
    prev_anti_variance: Optional[float] = None
    quality_history: List[Dict[str, Any]] = []
    verified_ms = 0
    pool_variance = 0.0

    def _encode(file_audio: FileAudio, t0: int, t1: int) -> Tuple[List[float], int]:
        return encode_window(
            models.ecapa, file_audio.pcm, file_audio.audio_start_pts_ms,
            file_audio.num_samples, t0, t1, params,
        )

    def _recompute_anti() -> None:
        nonlocal e_anti
        if anti_pool:
            e_anti = l2_normalize(duration_weighted_mean(
                [entry.vector for entry in anti_pool],
                [entry.duration_ms for entry in anti_pool],
            ))

    def _recompute_composite() -> None:
        nonlocal e_composite, verified_ms, pool_variance
        e_composite = l2_normalize(duration_weighted_mean(
            [entry.vector for entry in pool],
            [entry.duration_ms for entry in pool],
        ))
        verified_ms = sum(entry.duration_ms for entry in pool)
        pool_variance = pairwise_cosine_variance(
            [entry.vector for entry in pool]
        )

    def _accept(
        file_audio: FileAudio, kind: str, t0: int, t1: int,
        vector: List[float], duration_ms: int,
        mar_trace: Optional[List[Tuple[int, Optional[float]]]],
        end_reason: Optional[str], extra: Dict[str, Any],
        into_anti: bool,
    ) -> None:
        persist = _persist_window(
            enroll_dir, file_audio, kind, t0, t1, mar_trace, end_reason,
        )
        entry = PoolEntry(
            vector=vector, duration_ms=duration_ms, kind=kind,
            file_index=file_audio.file_index,
            t_start_local_ms=t0, t_stop_local_ms=t1,
        )
        (anti_pool if into_anti else pool).append(entry)
        manifest.append(
            Operation.ENROLLMENT_VECTOR,
            _vector_payload(kind, file_audio, t0, t1, vector, persist, extra),
        )

    # =========================================================================
    # Sequential per-video loop (canonical file order — architectural)
    # =========================================================================
    for file_audio in batch.files:
        frame_pts = vision.video_frame_pts_ms(file_audio.source_path)
        frames, scan_stats = vision.scan_video(
            models, file_audio.source_path, frame_pts,
            file_audio.silero_segments_local_ms, params,
        )
        manifest.append(OP_SCAN, {"file_index": file_audio.file_index, **scan_stats})
        if scan_stats["pts_mismatch"]:
            manifest.append(Operation.WARNING, {
                "warning": "frame_pts_mismatch",
                "file_index": file_audio.file_index,
                "decoded_frames": scan_stats["decoded_frames"],
                "listed_pts": scan_stats["listed_pts"],
            })

        # --- Video 1 only: clicks -> F_target, E_seed, optional Track C ----
        if f_target is None:
            try:
                anchor, _ = _face_at_click(
                    frames, clicks.speaking.pts_ms,
                    clicks.speaking.x, clicks.speaking.y,
                )
            except Layer1ReclickError as exc:
                _reclick(manifest, "no_face_at_speaking_click",
                         {"pts_ms": clicks.speaking.pts_ms,
                          "detail": str(exc)})
            anchor_obs = _build_obs(
                frames, anchor.embedding, None, file_audio, params,
            )
            seed_window = next(
                (w for w in _run_machine(anchor_obs.obs, params)
                 if w.t_start_ms <= clicks.speaking.pts_ms <= w.t_stop_ms),
                None,
            )
            if seed_window is None:
                _reclick(manifest, "click_outside_speaking_window",
                         {"pts_ms": clicks.speaking.pts_ms})
            if seed_window.duration_ms < params.seed_min_ms:       # guardrail 2
                _reclick(manifest, "seed_too_short", {
                    "duration_ms": seed_window.duration_ms,
                    "seed_min_ms": params.seed_min_ms,
                })
            if seed_window.interviewer_present_frames > 0:          # guardrail 1
                open_frac = 1.0 - (
                    seed_window.interviewer_closed_frames
                    / seed_window.interviewer_present_frames
                )
                if open_frac > params.click_overlap_max_frac:
                    _reclick(manifest, "overlap_at_speaking_click",
                             {"non_target_lips_open_frac": open_frac})

            # Refined biometric lock: mean embedding over the seed span.
            matched = [
                face.embedding
                for frame in frames
                if seed_window.t_start_ms <= frame.pts_ms <= seed_window.t_stop_ms
                for face, sim in [_match_face(
                    frame.faces, anchor.embedding, params.face_reid_threshold)]
                if face is not None
            ]
            f_target = _mean_embedding(matched) if matched else list(anchor.embedding)

            seed_t0 = _clamp_local(file_audio, seed_window.t_start_ms)
            seed_t1 = _clamp_local(file_audio, seed_window.t_stop_ms)
            e_seed, seed_duration = _encode(file_audio, seed_t0, seed_t1)
            seed_span = (file_audio.file_index, seed_t0, seed_t1)
            _accept(
                file_audio, KIND_SEED, seed_t0, seed_t1, e_seed, seed_duration,
                seed_window.mar_trace, seed_window.end_reason,
                {"operator_verified": True}, into_anti=False,
            )
            manifest.append(OP_SEED, {
                "file_index": file_audio.file_index,
                "t_start_local_ms": seed_t0,
                "t_stop_local_ms": seed_t1,
                "duration_ms": seed_duration,
                "vector_sha256": sha256_of_obj(e_seed),
            })

            # Track C — operator anti-profile click (optional, prioritized)
            if clicks.anti is not None:
                try:
                    anti_face, anti_pts = _face_at_click(
                        frames, clicks.anti.pts_ms,
                        clicks.anti.x, clicks.anti.y,
                    )
                except Layer1ReclickError as exc:
                    _reclick(manifest, "no_face_at_anti_click",
                             {"pts_ms": clicks.anti.pts_ms,
                              "detail": str(exc)})
                sim_to_target = cosine(anti_face.embedding, f_target)
                if sim_to_target >= params.face_reid_threshold:     # guardrail 3
                    _reclick(manifest, "anti_click_matches_target",
                             {"sim_to_target": sim_to_target})
                click_frame = min(frames, key=lambda f: abs(f.pts_ms - anti_pts))
                target_face, _ = _match_face(
                    click_frame.faces, f_target, params.face_reid_threshold,
                )
                if (
                    target_face is not None
                    and target_face.mar is not None
                    and target_face.mar >= params.mar_on
                ):
                    # Bench-corrected MAR (2026-06-12): the normalized
                    # outer-lip-contour formula ranges ~0.10 (lips closed) to
                    # ~0.25 (open), with resting/listening ~0.13. Use mar_on
                    # (0.15) as the "clearly speaking" threshold so a target
                    # who is merely listening (MAR below mar_on) does not
                    # false-trigger this guardrail; only an open mouth
                    # (MAR >= mar_on) flags the target as possibly speaking.
                    _reclick(manifest, "target_lips_open_at_anti_click",
                             {"target_mar": target_face.mar})
                if not vad_near(anti_pts, file_audio.silero_segments_local_ms,
                                params.vad_tol_ms):
                    _reclick(manifest, "no_audio_energy_at_anti_click",
                             {"pts_ms": anti_pts})
                half = params.trackb_window_ms // 2
                anti_t0 = _clamp_local(file_audio, anti_pts - half)
                anti_t1 = _clamp_local(file_audio, anti_pts + half)
                anti_vector, anti_duration = _encode(file_audio, anti_t0, anti_t1)
                f_interviewer = list(anti_face.embedding)
                _accept(
                    file_audio, KIND_ANTI_C, anti_t0, anti_t1,
                    anti_vector, anti_duration, None, None,
                    {"operator_verified": True,
                     "sim_to_target_face": sim_to_target},
                    into_anti=True,
                )

        # --- Production observation stream for this video --------------------
        bundle = _build_obs(frames, f_target, f_interviewer, file_audio, params)
        if not bundle.any_pose:
            manifest.append(Operation.WARNING, {
                "warning": "pose_unavailable",
                "file_index": file_audio.file_index,
                "note": "yaw suspension inactive for this video",
            })
        if bundle.reid_sims:
            reid_mean = math.fsum(bundle.reid_sims) / len(bundle.reid_sims)
            if reid_mean < params.reid_warning_floor:               # guardrail 6
                manifest.append(Operation.WARNING, {
                    "warning": "low_detection_quality",
                    "file_index": file_audio.file_index,
                    "reid_running_mean": reid_mean,
                    "floor": params.reid_warning_floor,
                })

        # --- Track A: candidate windows through the Triple Gate --------------
        gate_c_failures: List[Dict[str, Any]] = []
        for window in _run_machine(bundle.obs, params):
            t0 = _clamp_local(file_audio, window.t_start_ms)
            t1 = _clamp_local(file_audio, window.t_stop_ms)
            base = {
                "kind": KIND_TRACK_A,
                "file_index": file_audio.file_index,
                "t_start_local_ms": t0,
                "t_stop_local_ms": t1,
                "duration_ms": t1 - t0,
                "end_reason": window.end_reason,
            }
            if (
                seed_span is not None
                and file_audio.file_index == seed_span[0]
                and not (t1 <= seed_span[1] or t0 >= seed_span[2])
            ):
                manifest.append(Operation.ENROLLMENT_DISCARD,
                                {**base, "reason": "seed_overlap"})
                continue
            if (t1 - t0) < params.min_enroll_len_ms:
                manifest.append(Operation.ENROLLMENT_DISCARD,
                                {**base, "reason": "below_min_enroll_len"})
                continue
            coverage = (
                segment_overlap_ms(t0, t1, file_audio.silero_segments_local_ms)
                / (t1 - t0)
            )
            vector, duration = _encode(file_audio, t0, t1)
            sim_seed = cosine(vector, e_seed)
            anti_available = e_anti is not None
            sim_anti = cosine(vector, e_anti) if anti_available else None
            result = evaluate_triple_gate(
                vad_coverage=coverage,
                interviewer_present_frames=window.interviewer_present_frames,
                interviewer_closed_frames=window.interviewer_closed_frames,
                sim_seed=sim_seed,
                sim_anti=sim_anti,
                anti_available=anti_available,
                params=params,
            )
            if result.accepted:
                _accept(file_audio, KIND_TRACK_A, t0, t1, vector, duration,
                        window.mar_trace, window.end_reason,
                        result.detail, into_anti=False)
            else:
                manifest.append(Operation.ENROLLMENT_DISCARD, {
                    **base, "failed_gate": result.failed_gate, **result.detail,
                })
                if result.failed_gate == "C":
                    gate_c_failures.append({
                        "window": window, "t0": t0, "t1": t1,
                        "vector": vector, "duration": duration,
                        "coverage": coverage, "sim_seed": sim_seed,
                    })

        # --- Track B: automatic anti-profile collection -----------------------
        last_center: Optional[int] = None
        for center in bundle.trackb_centers:
            if (
                last_center is not None
                and center - last_center < params.trackb_min_spacing_ms
            ):
                continue
            last_center = center
            half = params.trackb_window_ms // 2
            t0 = _clamp_local(file_audio, center - half)
            t1 = _clamp_local(file_audio, center + half)
            if (t1 - t0) < half:        # too clipped at a file edge
                continue
            vector, duration = _encode(file_audio, t0, t1)
            sim_to_seed = cosine(vector, e_seed)
            if mtrap_discard(sim_to_seed, params):                  # guardrail 4
                manifest.append(Operation.ENROLLMENT_DISCARD, {
                    "kind": KIND_ANTI_B,
                    "file_index": file_audio.file_index,
                    "t_start_local_ms": t0, "t_stop_local_ms": t1,
                    "duration_ms": t1 - t0,
                    "reason": "mtrap_high_sim_to_seed",
                    "sim_to_seed": sim_to_seed,
                })
                continue
            _accept(file_audio, KIND_ANTI_B, t0, t1, vector, duration,
                    None, None, {"sim_to_seed": sim_to_seed}, into_anti=True)

        # --- Pool maintenance, sanity checks, quality state -------------------
        if anti_pool:
            _recompute_anti()
            anti_variance = pairwise_cosine_variance(
                [entry.vector for entry in anti_pool]
            )
            if (
                prev_anti_variance is not None
                and anti_variance - prev_anti_variance > params.pool_var_warning
            ):
                manifest.append(Operation.WARNING, {
                    "warning": "anti_pool_variance_increase",
                    "file_index": file_audio.file_index,
                    "previous_variance": prev_anti_variance,
                    "variance": anti_variance,
                })
            prev_anti_variance = anti_variance

        _recompute_composite()
        if e_anti is not None:                                       # guardrail 8
            contamination = cosine(e_composite, e_anti)
            level = contamination_level(contamination, params)
            if level == CONTAM_WARNING:
                manifest.append(Operation.WARNING, {
                    "warning": "enrollment_contamination",
                    "sim_composite_anti": contamination,
                    "file_index": file_audio.file_index,
                })
            elif level == CONTAM_HALT:
                manifest.append(Operation.BLOCKING_HALT, {
                    "reason": "enrollment_contamination_critical",
                    "sim_composite_anti": contamination,
                    "file_index": file_audio.file_index,
                })
                raise Layer1Error(
                    "critical enrollment contamination — re-run Layer 1"
                )

        state = assess_quality(
            verified_ms, pool_variance, bool(anti_pool), params,
        )

        # MARGINAL second pass: re-evaluate this video's Gate-C discards
        # against the grown anti-profile ("second pass with improved anchor").
        second_pass_accepted = 0
        if state == MARGINAL and gate_c_failures and e_anti is not None:
            for failed in gate_c_failures:
                sim_anti = cosine(failed["vector"], e_anti)
                retry = evaluate_triple_gate(
                    vad_coverage=failed["coverage"],
                    interviewer_present_frames=(
                        failed["window"].interviewer_present_frames),
                    interviewer_closed_frames=(
                        failed["window"].interviewer_closed_frames),
                    sim_seed=failed["sim_seed"],
                    sim_anti=sim_anti,
                    anti_available=True,
                    params=params,
                )
                if retry.accepted:
                    _accept(file_audio, KIND_TRACK_A,
                            failed["t0"], failed["t1"],
                            failed["vector"], failed["duration"],
                            failed["window"].mar_trace,
                            failed["window"].end_reason,
                            {**retry.detail, "second_pass": True},
                            into_anti=False)
                    second_pass_accepted += 1
            if second_pass_accepted:
                _recompute_composite()
                state = assess_quality(
                    verified_ms, pool_variance, bool(anti_pool), params,
                )

        quality_entry = {
            "file_index": file_audio.file_index,
            "state": state,
            "verified_ms": verified_ms,
            "pool_size": len(pool),
            "pool_variance": pool_variance,
            "anti_pool_size": len(anti_pool),
            "no_anti_profile": not anti_pool,
            "second_pass_accepted": second_pass_accepted,
        }
        quality_history.append(quality_entry)
        manifest.append(OP_QUALITY, quality_entry)

    # =========================================================================
    # Batch end: critical failure, variance gate, freeze
    # =========================================================================
    if verified_ms < params.marginal_ms or quality_history[-1]["state"] == INSUFFICIENT:
        manifest.append(Operation.BLOCKING_HALT, {                  # guardrail 9
            "reason": "critical_enrollment_failure",
            "verified_ms": verified_ms,
            "required_ms": params.marginal_ms,
        })
        raise Layer1Error(
            "CRITICAL ENROLLMENT FAILURE — all videos processed, enrollment "
            "still INSUFFICIENT; operator intervention required"
        )
    if pool_variance > params.variance_high:
        manifest.append(Operation.WARNING, {
            "warning": "high_pool_variance_operator_review",
            "pool_variance": pool_variance,
            "note": "human decides; windows are flagged, not auto-discarded",
        })

    e_composite_sha = sha256_of_obj(e_composite)
    e_anti_sha = sha256_of_obj(e_anti) if e_anti is not None else None
    manifest.append(OP_FREEZE, {
        "e_composite_sha256": e_composite_sha,
        "e_anti_sha256": e_anti_sha,
        "e_anti_pool_sha256": sha256_of_obj(
            [entry.vector for entry in anti_pool]
        ) if anti_pool else None,
        "pool_size": len(pool),
        "anti_pool_size": len(anti_pool),
        "total_verified_ms": verified_ms,
        "pool_variance": pool_variance,
        "no_anti_profile": not anti_pool,
        "note": "E_composite is FROZEN — never modified after this entry",
    })
    return EnrollmentResult(
        f_target=list(f_target),
        e_seed=list(e_seed),
        e_composite=list(e_composite),
        e_composite_sha256=e_composite_sha,
        e_anti=list(e_anti) if e_anti is not None else None,
        e_anti_sha256=e_anti_sha,
        no_anti_profile=not anti_pool,
        pool=pool,
        anti_pool=anti_pool,
        total_verified_ms=verified_ms,
        quality_history=quality_history,
    )
