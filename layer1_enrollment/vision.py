"""
SPOVNOB — Module 2 (Layer 1): vision.py
========================================

The GPU-facing visual scan: per-frame PTS via ffprobe packet timestamps,
sequential OpenCV decode, batched YOLOv8 person gating, and InsightFace
faces (detection score filter, ArcFace embedding, 2d106det MAR, head
yaw). cv2 / ultralytics / insightface are imported lazily inside the
functions — this module's import is stdlib-safe, but its functions run
only on the Ubuntu bench with the resident models from the gate.

Implements: Audio_Diarization.md — Layer 1 mermaid nodes "Video Frame
Stream", "YOLOv8" (person detection, empty frames skipped), and
"InsightFace" (biometric measurements; min confidence enforced —
guardrail 5); System Environment "Parallel Execution Model" optional
visual-scan efficiency rule (silence stride); PTS mandate (frame times
come from container packet PTS, sorted to presentation order — frame
INDICES are never used as time anywhere).

Determinism notes:
  - Frame PTS list: ffprobe video packet pts_time values, converted with
    the same Decimal/ROUND_HALF_EVEN rule as Layer 0, sorted ascending
    (packets arrive in decode order; presentation order = sorted PTS).
    OpenCV decodes in presentation order; frame i pairs with pts[i].
    A count mismatch is reported to the caller for a WARNING entry and
    pairing stops at the shorter length.
  - YOLO runs on fixed batches of VISUAL_BATCH_FRAMES (environment_gate
    architectural constant); frames without a person detection skip
    InsightFace entirely (document rule).
  - Silence stride (params.silence_stride > 1): frames outside Silero
    speech (+/- vad_tol) are analyzed only every Nth frame; frames
    inside speech are always analyzed. Depends only on input -> still
    deterministic. Default stride 1 = rule off.

CUDA determinism dependencies: inherits the four constants from
environment_gate for YOLO; InsightFace runs under ONNXRuntime
CUDAExecutionProvider (FLAG 2 guard proved it live at startup).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layer0_preprocessor import decimal_seconds_to_ms

from .errors import Layer1Error
from .gates import vad_near
from .geometry import compute_mar
from .params import EnrollmentParams


@dataclass
class FaceObs:
    """One detected face in one frame (det_score already filtered by
    guardrail 5 — low-confidence faces never reach this object)."""

    bbox: Tuple[float, float, float, float]      # x1, y1, x2, y2
    det_score: float
    embedding: List[float]                       # ArcFace, L2-normalized
    mar: Optional[float]                         # raw normalized MAR
    yaw_degrees: Optional[float]                 # None if pose unavailable

    def contains_point(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameFaces:
    pts_ms: int
    faces: List[FaceObs] = field(default_factory=list)


def video_frame_pts_ms(video_path: Path | str) -> List[int]:
    """All video packet PTS values in integer milliseconds, sorted
    ascending (presentation order)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(video_path)],
        capture_output=True, text=True, check=True,
    )
    pts: List[int] = []
    for line in out.stdout.splitlines():
        token = line.strip().split(",")[0]
        if token and token != "N/A":
            pts.append(decimal_seconds_to_ms(token))
    pts.sort()
    return pts


def scan_video(
    models: Any,
    video_path: Path | str,
    frame_pts: Sequence[int],
    speech_segments_local_ms: Sequence[Tuple[int, int]],
    params: EnrollmentParams,
) -> Tuple[List[FrameFaces], Dict[str, Any]]:
    """Decode the video once, sequentially, and return per-frame face
    observations for every ANALYZED frame (silence-stride skips are
    omitted entirely; downstream timing uses PTS, so gaps are fine).

    Returns (frames, stats) where stats carries decoded/listed counts
    and a pts_mismatch flag for the orchestrator's WARNING entry."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise Layer1Error(f"cannot open video: {video_path}")

    batch_size = environment_gate.VISUAL_BATCH_FRAMES
    stride = max(1, params.silence_stride)
    analyzed: List[FrameFaces] = []
    pending: List[Tuple[int, Any]] = []          # (pts_ms, BGR frame)
    decoded = 0
    pts_mismatch = False

    def _flush_pending() -> None:
        """YOLO person-gate the pending batch, then InsightFace the
        frames that contain a person. Sequential, fixed batch size."""
        if not pending:
            return
        results = models.yolo(
            [frame for _, frame in pending],
            conf=params.yolo_min_conf, classes=[0], verbose=False,
        )
        for (pts_ms, frame), result in zip(pending, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue                          # empty frame: skipped
            faces: List[FaceObs] = []
            for face in models.insightface.get(frame):
                det_score = float(face.det_score)
                if det_score < params.insightface_min_det_score:
                    continue                      # guardrail 5: not detected
                normed = getattr(face, "normed_embedding", None)
                embedding = (
                    normed.tolist() if normed is not None
                    else face.embedding.tolist()
                )
                landmarks = getattr(face, "landmark_2d_106", None)
                mar = (
                    compute_mar([tuple(point) for point in landmarks], params)
                    if landmarks is not None else None
                )
                pose = getattr(face, "pose", None)
                yaw = float(pose[1]) if pose is not None else None
                faces.append(FaceObs(
                    bbox=tuple(float(v) for v in face.bbox),
                    det_score=det_score,
                    embedding=embedding,
                    mar=mar,
                    yaw_degrees=yaw,
                ))
            analyzed.append(FrameFaces(pts_ms=pts_ms, faces=faces))
        pending.clear()

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        if index >= len(frame_pts):
            pts_mismatch = True
            break
        pts_ms = frame_pts[index]
        in_speech = vad_near(pts_ms, speech_segments_local_ms, params.vad_tol_ms)
        if in_speech or stride == 1 or index % stride == 0:
            pending.append((pts_ms, frame))
            if len(pending) >= batch_size:
                _flush_pending()
        index += 1
    _flush_pending()
    capture.release()

    if decoded < len(frame_pts):
        pts_mismatch = True
    stats = {
        "decoded_frames": decoded,
        "listed_pts": len(frame_pts),
        "analyzed_frames": len(analyzed),
        "pts_mismatch": pts_mismatch,
        "silence_stride": stride,
    }
    return analyzed, stats
