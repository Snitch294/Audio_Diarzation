"""
SPOVNOB — Operator tool: click_ui.py
=====================================

Purpose:    A local, single-file Flask UI for producing `clicks.json`.
            The operator scrubs the first video frame-by-frame, sees the
            face detections / MAR values / Silero VAD state the pipeline
            itself will see, and registers the `speaking_click` (and the
            optional `anti_click`) with LIVE guardrail validation. The
            exported file is exactly what `layer1_enrollment.load_clicks`
            expects — including the `NO_ANTI_PROFILE` path (anti_click
            key omitted entirely when the toggle is off).

Parity contract (the design rule of this module):
            The UI never re-implements pipeline logic. Every validation
            is performed by IMPORTING the production functions:
              - vision.video_frame_pts_ms / vision.scan_video
                (ffprobe packet PTS · YOLO person gate · InsightFace
                faces, det-score filter, MAR, yaw — guardrail 5 included)
              - layer0_preprocessor.extract_audio / silero_window_probs /
                segments_from_window_probs (exact Layer 0 audio + VAD)
              - enrollment._face_at_click / _build_obs / _run_machine /
                _match_face / _mean_embedding (click resolution, the
                observation stream, the real WindowMachine, F_target
                refinement — guardrails 1, 2, 3 and the anti-click
                checks fire from the same code the pipeline runs)
            Underscore imports are deliberate: if enrollment.py changes,
            this tool must break loudly rather than drift silently.

Guardrails enforced at click time (production reason strings):
            speaking: no_face_at_speaking_click ·
                      click_outside_speaking_window ·
                      seed_too_short · overlap_at_speaking_click
            anti:     no_face_at_anti_click · anti_click_matches_target ·
                      target_lips_open_at_anti_click ·
                      no_audio_energy_at_anti_click
            ordering: the anti click validates against F_target, which
                      only exists after the speaking click — enforced.

Chain of custody:
            This tool is OUTSIDE the audit chain: it writes no manifest
            entries and its output (clicks.json) is re-validated from
            scratch by Layer 1. It only writes clicks.json plus its own
            pre-scan cache under <work_dir>/ui_cache/.

Pre-scan cache:
            The startup scan (audio strip + Silero + YOLO/InsightFace
            over every frame + display JPEGs) is cached on disk, keyed
            by video SHA-256, the full EnrollmentParams payload, the
            vision batch constants, the device, and the model store's
            expected_hashes.json digest. A warm start needs no models,
            no torch, no GPU — startup is ~2 s.

Run:        python3 click_ui.py <video> --model-store /opt/spovnob/model_store
                [--work-dir session] [--port 5050] [--cpu] [--rescan]
                [--no-browser]
Self-test:  python3 click_ui.py --selftest   (stdlib + repo modules only:
            no flask, no torch, no cv2, no ffmpeg, no GPU — standing
            test policy; drives the REAL imported validation functions
            over synthetic frames)

CUDA determinism dependencies: import of environment_gate fixes the
process env; _load_ui_models applies the same torch determinism flags
as the gate (without the gate's audit checks — see chain-of-custody
note above). --cpu exists for development only; the bench runs CUDA.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from layer0_preprocessor import (
    SILERO_MIN_SILENCE_MS,
    SILERO_MIN_SPEECH_MS,
    SILERO_SPEECH_PAD_MS,
    SILERO_THRESHOLD,
    SILERO_WINDOW_SAMPLES,
    FileAudio,
    extract_audio,
    ms_from_samples,
    parse_probe,
    read_wav_pcm16,
    run_ffprobe,
    segments_from_window_probs,
    silero_window_probs,
)
from layer1_enrollment.encoding import cosine
from layer1_enrollment.enrollment import (
    CLICK_MATCH_MAX_GAP_MS,
    _build_obs,
    _face_at_click,
    _match_face,
    _mean_embedding,
    _run_machine,
)
from layer1_enrollment.errors import Layer1ReclickError
from layer1_enrollment.gates import vad_near
from layer1_enrollment.params import EnrollmentParams
from layer1_enrollment.vision import (
    FaceObs,
    FrameFaces,
    scan_video,
    video_frame_pts_ms,
)
from session_manifest import canonical_json, sha256_of_file, sha256_of_obj

# --- Tool constants -----------------------------------------------------------
CACHE_SCHEMA = "spovnob-clickui-prescan-v1"
DISPLAY_MAX_WIDTH = 800          # display JPEGs are capped at this width
DISPLAY_JPEG_QV = 4              # ffmpeg -q:v (2=best .. 31=worst)
DEFAULT_PORT = 5050
DEFAULT_WORK_DIR = Path("session")
CLICKS_FILENAME = "clicks.json"

# Every reason this tool can return. The production guardrail strings are
# byte-identical to enrollment.py's manifest entries so the operator can
# correlate UI feedback with pipeline logs.
SPEAKING_REASONS = (
    "no_face_at_speaking_click",
    "click_outside_speaking_window",
    "seed_too_short",
    "overlap_at_speaking_click",
)
ANTI_REASONS = (
    "no_face_at_anti_click",
    "anti_click_matches_target",
    "target_lips_open_at_anti_click",
    "no_audio_energy_at_anti_click",
)
UI_REASONS = (
    "speaking_click_required_first",
    "export_requires_speaking_click",
    "anti_click_not_registered",
    "invalid_request",
)
ALL_REASONS = SPEAKING_REASONS + ANTI_REASONS + UI_REASONS


class ClickUIError(RuntimeError):
    """Unrecoverable startup failure (bad arguments, missing models)."""


# =============================================================================
# Plain-English operator messages (parameter values injected, not hardcoded)
# =============================================================================

def reason_message(reason: str, detail: Dict[str, Any],
                   params: EnrollmentParams) -> str:
    d = detail

    def num(key: str, fmt: str = "{:.3f}", fallback: str = "?") -> str:
        value = d.get(key)
        return fmt.format(value) if isinstance(value, (int, float)) else fallback

    messages = {
        "no_face_at_speaking_click":
            f"No detected face within {CLICK_MATCH_MAX_GAP_MS} ms of this "
            "timestamp. Scrub to a frame with a face box and click inside it.",
        "click_outside_speaking_window":
            "The target is not inside a detected speaking window here (the "
            "lips-open + voice-activity state machine never latched at this "
            "timestamp). Scrub to where the target is clearly talking — green "
            "VAD strip and open lips.",
        "seed_too_short":
            f"The speaking window around this click is only {num('duration_ms', '{:.0f}')} ms; "
            f"the seed needs at least {params.seed_min_ms} ms of continuous "
            "speech. Click inside a longer uninterrupted speaking stretch.",
        "overlap_at_speaking_click":
            "Another face has open lips for "
            f"{num('non_target_lips_open_frac', '{:.0%}')} of this speaking window "
            f"(maximum {params.click_overlap_max_frac:.0%}) — possible overlapped "
            "speech. Pick a stretch where only the target is talking.",
        "no_face_at_anti_click":
            f"No detected face within {CLICK_MATCH_MAX_GAP_MS} ms of this "
            "timestamp. The interviewer must be on camera — use the orange "
            "2-face regions of the timeline.",
        "anti_click_matches_target":
            f"This face matches the locked target (cosine {num('sim_to_target')} ≥ "
            f"{params.face_reid_threshold:.2f}). Click the interviewer's face, "
            "not the target's.",
        "target_lips_open_at_anti_click":
            f"The target's lips are open here (MAR {num('target_mar')} ≥ "
            f"{params.mar_on:.2f}) — the target may be the one speaking. Choose "
            "a moment where the target is listening.",
        "no_audio_energy_at_anti_click":
            "Silero detects no speech at this timestamp — the anti-profile "
            "needs the interviewer's voice. Click inside a green VAD region.",
        "speaking_click_required_first":
            "Register the speaking click first: the anti click is validated "
            "against the target lock (F_target) derived from it.",
        "export_requires_speaking_click":
            "Cannot export: no valid speaking click is registered.",
        "anti_click_not_registered":
            "Anti-click is enabled but not registered. Register one, or "
            "uncheck “Include anti-click” to export the NO_ANTI_PROFILE path.",
        "invalid_request":
            "Malformed request (pts_ms must be an integer; x, y numbers; "
            "type 'speaking' or 'anti').",
    }
    return messages[reason]


def no_anti_warning(params: EnrollmentParams) -> str:
    return (
        "No anti-click: the pipeline runs the NO_ANTI_PROFILE branch — the "
        f"STRONG quality state requires {params.strong_ms_no_anti // 1000} s of "
        f"verified audio instead of {params.strong_ms // 1000} s. Use this only "
        "when the interviewer is never on camera."
    )


# =============================================================================
# Click session: registration state + production-parity validation
# =============================================================================

@dataclass
class RegisteredClick:
    pts_ms: int
    x: float
    y: float
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)


class ClickSession:
    """Server-side state for one video. The validate methods replicate
    enrollment.run_layer1's click handling by CALLING the same functions
    on the same inputs, in the same order."""

    def __init__(self, frames: List[FrameFaces], file_audio: FileAudio,
                 params: EnrollmentParams) -> None:
        self.frames = frames
        self.file_audio = file_audio
        self.params = params
        self.speaking: Optional[RegisteredClick] = None
        self.anti: Optional[RegisteredClick] = None
        self.f_target: Optional[List[float]] = None

    # -- result helpers ---------------------------------------------------------

    def _fail(self, reason: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": False,
            "reason": reason,
            "message": reason_message(reason, detail, self.params),
            "detail": detail,
        }

    def state_payload(self) -> Dict[str, Any]:
        def one(click: Optional[RegisteredClick]) -> Optional[Dict[str, Any]]:
            if click is None:
                return None
            return {"pts_ms": click.pts_ms, "x": click.x, "y": click.y,
                    "summary": click.summary}
        return {"speaking": one(self.speaking), "anti": one(self.anti)}

    # -- speaking click (guardrails 1 & 2 + window membership) -------------------

    def _validate_speaking(
        self, pts_ms: int, x: float, y: float
    ) -> Tuple[Dict[str, Any], Optional[List[float]]]:
        params = self.params
        try:
            # enrollment.py: anchor, _ = _face_at_click(...)
            anchor, matched_pts = _face_at_click(self.frames, pts_ms, x, y)
        except Layer1ReclickError as exc:
            return self._fail("no_face_at_speaking_click",
                              {"pts_ms": pts_ms, "detail": str(exc)}), None

        # enrollment.py: anchor obs stream + window machine, anchor as target
        anchor_obs = _build_obs(
            self.frames, anchor.embedding, None, self.file_audio, params,
        )
        seed_window = next(
            (w for w in _run_machine(anchor_obs.obs, params)
             if w.t_start_ms <= pts_ms <= w.t_stop_ms),
            None,
        )
        if seed_window is None:
            return self._fail("click_outside_speaking_window",
                              {"pts_ms": pts_ms}), None
        if seed_window.duration_ms < params.seed_min_ms:        # guardrail 2
            return self._fail("seed_too_short", {
                "duration_ms": seed_window.duration_ms,
                "seed_min_ms": params.seed_min_ms,
            }), None
        open_frac: Optional[float] = None
        if seed_window.interviewer_present_frames > 0:           # guardrail 1
            open_frac = 1.0 - (
                seed_window.interviewer_closed_frames
                / seed_window.interviewer_present_frames
            )
            if open_frac > params.click_overlap_max_frac:
                return self._fail("overlap_at_speaking_click",
                                  {"non_target_lips_open_frac": open_frac}), None

        # enrollment.py: refined biometric lock over the seed span
        matched = [
            face.embedding
            for frame in self.frames
            if seed_window.t_start_ms <= frame.pts_ms <= seed_window.t_stop_ms
            for face, sim in [_match_face(
                frame.faces, anchor.embedding, params.face_reid_threshold)]
            if face is not None
        ]
        f_target = _mean_embedding(matched) if matched else list(anchor.embedding)

        detail = {
            "matched_frame_pts_ms": matched_pts,
            "anchor_det_score": anchor.det_score,
            "window": {
                "t_start_ms": seed_window.t_start_ms,
                "t_stop_ms": seed_window.t_stop_ms,
                "duration_ms": seed_window.duration_ms,
                "end_reason": seed_window.end_reason,
                "interviewer_present_frames":
                    seed_window.interviewer_present_frames,
                "interviewer_closed_frames":
                    seed_window.interviewer_closed_frames,
                "non_target_lips_open_frac": open_frac,
            },
            "seed_min_ms": params.seed_min_ms,
            "f_target_frames": len(matched),
        }
        window = detail["window"]
        message = (
            "Speaking click valid — seed window "
            f"{window['t_start_ms']}–{window['t_stop_ms']} ms "
            f"({window['duration_ms']} ms ≥ {params.seed_min_ms} ms, "
            f"end: {window['end_reason']}). Target lock set from "
            f"{len(matched)} frames."
        )
        return {"ok": True, "message": message, "detail": detail}, f_target

    def register_speaking(self, pts_ms: int, x: float, y: float) -> Dict[str, Any]:
        result, f_target = self._validate_speaking(pts_ms, x, y)
        if not result["ok"]:
            return result
        anti_cleared = self.anti is not None
        self.speaking = RegisteredClick(
            pts_ms=pts_ms, x=x, y=y,
            summary=result["message"], detail=result["detail"],
        )
        self.f_target = f_target
        # The anti click is validated against F_target; a new speaking click
        # changes F_target, so any registered anti click is stale.
        self.anti = None
        result["anti_cleared"] = anti_cleared
        return result

    # -- anti click (guardrail 3 + lips/VAD checks) -------------------------------

    def _validate_anti(self, pts_ms: int, x: float, y: float) -> Dict[str, Any]:
        params = self.params
        if self.f_target is None:
            return self._fail("speaking_click_required_first", {})
        try:
            anti_face, anti_pts = _face_at_click(self.frames, pts_ms, x, y)
        except Layer1ReclickError as exc:
            return self._fail("no_face_at_anti_click",
                              {"pts_ms": pts_ms, "detail": str(exc)})
        sim_to_target = cosine(anti_face.embedding, self.f_target)
        if sim_to_target >= params.face_reid_threshold:          # guardrail 3
            return self._fail("anti_click_matches_target",
                              {"sim_to_target": sim_to_target})
        click_frame = min(self.frames, key=lambda f: abs(f.pts_ms - anti_pts))
        target_face, _ = _match_face(
            click_frame.faces, self.f_target, params.face_reid_threshold,
        )
        target_mar = target_face.mar if target_face is not None else None
        if (
            target_face is not None
            and target_face.mar is not None
            and target_face.mar >= params.mar_on
        ):
            return self._fail("target_lips_open_at_anti_click",
                              {"target_mar": target_face.mar})
        if not vad_near(anti_pts, self.file_audio.silero_segments_local_ms,
                        params.vad_tol_ms):
            return self._fail("no_audio_energy_at_anti_click",
                              {"pts_ms": anti_pts})
        detail = {
            "matched_frame_pts_ms": anti_pts,
            "sim_to_target_face": sim_to_target,
            "target_mar_at_click": target_mar,
            "anti_det_score": anti_face.det_score,
        }
        message = (
            "Anti click valid — face is distinct from the target "
            f"(cosine {sim_to_target:.3f} < {params.face_reid_threshold:.2f}), "
            "target's lips are not open, Silero shows speech at "
            f"{anti_pts} ms."
        )
        return {"ok": True, "message": message, "detail": detail}

    def register_anti(self, pts_ms: int, x: float, y: float) -> Dict[str, Any]:
        result = self._validate_anti(pts_ms, x, y)
        if result["ok"]:
            self.anti = RegisteredClick(
                pts_ms=pts_ms, x=x, y=y,
                summary=result["message"], detail=result["detail"],
            )
        return result

    # -- clearing ------------------------------------------------------------------

    def clear(self, click_type: str) -> None:
        if click_type == "speaking":
            # F_target derives from the speaking click; the anti click is
            # validated against F_target — both fall together.
            self.speaking = None
            self.f_target = None
            self.anti = None
        elif click_type == "anti":
            self.anti = None

    # -- export ----------------------------------------------------------------------

    def export_payload(self, include_anti: bool) -> Dict[str, Any]:
        assert self.speaking is not None
        payload: Dict[str, Any] = {
            "speaking_click": {
                "file_index": self.file_audio.file_index,
                "pts_ms": self.speaking.pts_ms,
                "x": round(self.speaking.x, 2),
                "y": round(self.speaking.y, 2),
            }
        }
        if include_anti and self.anti is not None:
            payload["anti_click"] = {
                "file_index": self.file_audio.file_index,
                "pts_ms": self.anti.pts_ms,
                "x": round(self.anti.x, 2),
                "y": round(self.anti.y, 2),
            }
        return payload

    def write_export(self, work_dir: Path, include_anti: bool) -> Dict[str, Any]:
        if self.speaking is None:
            return self._fail("export_requires_speaking_click", {})
        if include_anti and self.anti is None:
            return self._fail("anti_click_not_registered", {})
        payload = self.export_payload(include_anti)
        path = Path(work_dir) / CLICKS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(tmp_path, path)

        # Self-verifying export: the file must round-trip through the exact
        # parser Layer 1 will use.
        from layer1_enrollment.enrollment import load_clicks
        parsed = load_clicks(path)
        assert parsed.speaking.pts_ms == self.speaking.pts_ms
        assert (parsed.anti is not None) == ("anti_click" in payload)

        no_anti = "anti_click" not in payload
        message = f"clicks.json written: {path}"
        if no_anti:
            message += " — NO_ANTI_PROFILE path (anti_click key omitted)."
        return {
            "ok": True,
            "path": str(path),
            "clicks": payload,
            "no_anti": no_anti,
            "message": message,
        }


# =============================================================================
# Pre-scan + disk cache
# =============================================================================

@dataclass
class UISession:
    """Everything the HTTP layer needs, produced by prepare_session()."""
    video_path: Path
    work_dir: Path
    cache_dir: Path
    frames_dir: Path
    pts_list: List[int]
    frame_count: int                 # servable display frames
    native_width: int
    native_height: int
    scan_stats: Dict[str, Any]
    device: str
    from_cache: bool
    file_audio: FileAudio
    frames: List[FrameFaces]
    params: EnrollmentParams
    video_sha8: str = ""             # first 8 hex chars — used as browser cache-buster

    @property
    def servable_frames(self) -> int:
        return min(len(self.pts_list), self.frame_count)


def build_cache_key(
    video_sha256: str,
    params: EnrollmentParams,
    device: str,
    model_registry_sha256: Optional[str],
) -> Dict[str, Any]:
    """Everything that could change the pre-scan output. Exact-match
    compared against the cached key (model registry digest is skipped
    when --model-store is not provided on a warm start)."""
    return {
        "schema": CACHE_SCHEMA,
        "video_sha256": video_sha256,
        "params": params.manifest_payload(),
        "visual_batch_frames": environment_gate.VISUAL_BATCH_FRAMES,
        "insightface_det_size": list(environment_gate.INSIGHTFACE_DET_SIZE),
        "silero": {
            "threshold": SILERO_THRESHOLD,
            "window_samples": SILERO_WINDOW_SAMPLES,
            "min_speech_ms": SILERO_MIN_SPEECH_MS,
            "min_silence_ms": SILERO_MIN_SILENCE_MS,
            "speech_pad_ms": SILERO_SPEECH_PAD_MS,
        },
        "device": device,
        "display_max_width": DISPLAY_MAX_WIDTH,
        "display_jpeg_qv": DISPLAY_JPEG_QV,
        "model_registry_sha256": model_registry_sha256,
    }


def _keys_match(stored: Dict[str, Any], computed: Dict[str, Any],
                model_store_given: bool) -> bool:
    if model_store_given:
        return stored == computed
    skip = "model_registry_sha256"
    return ({k: v for k, v in stored.items() if k != skip}
            == {k: v for k, v in computed.items() if k != skip})


def _video_dimensions(probe_json: Dict[str, Any]) -> Tuple[int, int]:
    video = next(
        (s for s in probe_json.get("streams", [])
         if s.get("codec_type") == "video"),
        None,
    )
    if video is None or not video.get("width") or not video.get("height"):
        raise ClickUIError("no video stream with dimensions found")
    return int(video["width"]), int(video["height"])


def _extract_display_frames(video: Path, frames_dir: Path,
                            native_width: int) -> int:
    """One ffmpeg pass: every decoded frame (presentation order, -vsync 0
    passthrough so VFR sources are not resampled) to numbered JPEGs.
    Frame i (0-based) pairs with pts_list[i] — the same index pairing
    vision.scan_video uses for its sequential OpenCV decode."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", str(video),
        "-vsync", "0",
    ]
    if native_width > DISPLAY_MAX_WIDTH:
        command += ["-vf", f"scale={DISPLAY_MAX_WIDTH}:-2"]
    command += ["-q:v", str(DISPLAY_JPEG_QV), str(frames_dir / "%06d.jpg")]
    subprocess.run(command, capture_output=True, check=True)
    return sum(1 for _ in frames_dir.glob("*.jpg"))


def _load_ui_models(store: Path, device: str) -> SimpleNamespace:
    """The pre-scan model subset (Silero CPU + YOLO + InsightFace), loaded
    exactly like environment_gate.load_resident_models loads them, with
    the same torch determinism flags applied first. ECAPA and pyannote
    are not needed: the UI validates clicks, it never encodes audio."""
    if device == "cpu":
        # Must precede any torch/ultralytics import in this process.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(environment_gate.TORCH_NUM_THREADS)
    torch.manual_seed(environment_gate.GLOBAL_SEED)

    if device == "cuda" and not torch.cuda.is_available():
        raise ClickUIError(
            "CUDA is not available. The production bench runs CUDA; for "
            "development on this machine pass --cpu (results may differ "
            "from the bench at float precision level)."
        )

    from insightface.app import FaceAnalysis
    from ultralytics import YOLO

    silero_path = store / "silero-vad" / "files" / "silero_vad.jit"
    yolo_path = store / "yolov8" / "yolov8m.pt"
    insight_root = store / "insightface"
    for required in (silero_path, yolo_path, insight_root):
        if not required.exists():
            raise ClickUIError(
                f"model store incomplete: {required} not found "
                "(see UBUNTU_SETUP_GUIDE.md §4)"
            )

    silero = torch.jit.load(str(silero_path), map_location="cpu").eval()
    yolo = YOLO(str(yolo_path))
    providers = (
        ["CUDAExecutionProvider"] if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    insightface = FaceAnalysis(
        name="buffalo_l", root=str(insight_root), providers=providers,
    )
    insightface.prepare(
        ctx_id=0 if device == "cuda" else -1,
        det_size=environment_gate.INSIGHTFACE_DET_SIZE,
    )
    return SimpleNamespace(silero=silero, yolo=yolo, insightface=insightface)


def prepare_session(
    video: Path,
    work_dir: Path,
    model_store: Optional[Path],
    device: str,
    rescan: bool,
    log=print,
) -> UISession:
    """Warm path: load the disk cache (no models, no torch). Cold path:
    mini Layer 0 (exact production audio strip + Silero) + the production
    vision scan + display-frame extraction, then cache everything."""
    started = time.monotonic()

    def _log(message: str) -> None:
        log(f"[{time.monotonic() - started:6.1f}s] {message}")

    video = video.resolve()
    if not video.is_file():
        raise ClickUIError(f"video not found: {video}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    params = EnrollmentParams()

    _log(f"hashing video: {video.name}")
    video_sha = sha256_of_file(video)
    registry_sha = None
    if model_store is not None:
        registry_file = Path(model_store) / "expected_hashes.json"
        if registry_file.is_file():
            registry_sha = sha256_of_file(registry_file)
    key = build_cache_key(video_sha, params, device, registry_sha)

    cache_dir = work_dir / "ui_cache" / f"{video.stem}.{video_sha[:8]}.{device}"
    frames_dir = cache_dir / "frames"
    pickle_path = cache_dir / "prescan.pkl"

    if rescan and cache_dir.exists():
        _log(f"--rescan: removing {cache_dir}")
        shutil.rmtree(cache_dir)

    # --- Warm path -------------------------------------------------------------
    if pickle_path.is_file():
        try:
            with open(pickle_path, "rb") as handle:
                blob = pickle.load(handle)
        except Exception as exc:                      # corrupt cache: rebuild
            _log(f"cache unreadable ({exc!r}) — rebuilding")
            blob = None
        if blob is not None and blob.get("schema") == CACHE_SCHEMA:
            jpg_count = sum(1 for _ in frames_dir.glob("*.jpg"))
            if (
                _keys_match(blob["key"], key, model_store is not None)
                and jpg_count == blob["frame_count"]
            ):
                if model_store is None:
                    _log("cache key verified (model registry digest skipped "
                         "— no --model-store given); cached registry: "
                         f"{blob['key'].get('model_registry_sha256')}")
                _log(f"pre-scan cache HIT: {cache_dir}")
                return UISession(
                    video_path=video, work_dir=work_dir, cache_dir=cache_dir,
                    frames_dir=frames_dir, pts_list=blob["pts_list"],
                    frame_count=blob["frame_count"],
                    native_width=blob["native_width"],
                    native_height=blob["native_height"],
                    scan_stats=blob["scan_stats"], device=device,
                    from_cache=True, file_audio=blob["file_audio"],
                    frames=blob["frames"], params=params,
                    video_sha8=video_sha[:8],
                )
            _log("pre-scan cache STALE (key/frame mismatch) — rebuilding")

    # --- Cold path ---------------------------------------------------------------
    if model_store is None:
        raise ClickUIError(
            "no valid pre-scan cache for this video and --model-store was "
            "not given. Pass --model-store (canonical: "
            "/opt/spovnob/model_store) so the UI can run the scan."
        )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading models (silero + yolov8 + insightface) on {device} …")
    models = _load_ui_models(Path(model_store), device)

    # Mini Layer 0 — byte-identical to layer0_preprocessor._extract_one for
    # a single file (file_index 0, file_offset_ms 0).
    _log("ffprobe + PTS-true audio strip (Layer 0 extraction path) …")
    probe_json = run_ffprobe(video)
    probe = parse_probe(probe_json)
    native_width, native_height = _video_dimensions(probe_json)
    wav_path = cache_dir / "audio.16k.wav"
    extract_audio(video, wav_path)
    num_samples, pcm = read_wav_pcm16(wav_path)

    _log("Silero VAD over the full audio (Layer 0 segment derivation) …")
    data_segments = segments_from_window_probs(
        silero_window_probs(models.silero, pcm),
        total_samples=num_samples,
    )
    segments = [
        (probe.audio_start_pts_ms + start, probe.audio_start_pts_ms + end)
        for start, end in data_segments
    ]
    file_audio = FileAudio(
        file_index=0,
        source_path=str(video),
        wav_path=str(wav_path),
        source_sha256=video_sha,
        wav_sha256=sha256_of_file(wav_path),
        num_samples=num_samples,
        duration_ms=ms_from_samples(num_samples),
        audio_start_pts_ms=probe.audio_start_pts_ms,
        audio_start_missing=probe.audio_start_missing,
        vfr_suspected=probe.vfr_suspected,
        file_offset_ms=0,
        pcm=b"",                      # not needed: the UI never encodes audio
        silero_segments_local_ms=segments,
    )

    _log("frame PTS list (ffprobe packet timestamps) …")
    pts_list = video_frame_pts_ms(video)
    if not pts_list:
        raise ClickUIError("no video packet PTS found — is this a video file?")

    _log(f"production vision scan over {len(pts_list)} frames "
         "(YOLO person gate + InsightFace) — the slow step …")
    frames, scan_stats = scan_video(models, video, pts_list, segments, params)
    _log(f"scan done: {scan_stats['analyzed_frames']} frames analyzed, "
         f"{scan_stats['decoded_frames']} decoded")

    _log("extracting display JPEGs (single ffmpeg pass) …")
    frame_count = _extract_display_frames(video, frames_dir, native_width)
    if frame_count != scan_stats["decoded_frames"]:
        _log(f"NOTE: display frame count {frame_count} != scan decode count "
             f"{scan_stats['decoded_frames']} (pairing stops at the shorter)")

    blob = {
        "schema": CACHE_SCHEMA,
        "key": key,
        "frames": frames,
        "file_audio": file_audio,
        "pts_list": pts_list,
        "scan_stats": scan_stats,
        "native_width": native_width,
        "native_height": native_height,
        "frame_count": frame_count,
        "built_unix": int(time.time()),
    }
    with open(pickle_path, "wb") as handle:
        pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"pre-scan cached: {pickle_path}")

    return UISession(
        video_path=video, work_dir=work_dir, cache_dir=cache_dir,
        frames_dir=frames_dir, pts_list=pts_list, frame_count=frame_count,
        native_width=native_width, native_height=native_height,
        scan_stats=scan_stats, device=device, from_cache=False,
        file_audio=file_audio, frames=frames, params=params,
        video_sha8=video_sha[:8],
    )


# =============================================================================
# Timeline payload (pure — self-tested)
# =============================================================================

def timeline_entries(
    frames: Sequence[FrameFaces],
    pts_list: Sequence[int],
    count: int,
    file_audio: FileAudio,
    params: EnrollmentParams,
) -> List[Dict[str, Any]]:
    """Per display-frame metadata for the strips and the overlay drawing.
    `vad` uses the production vad_near predicate (the exact value the
    window machine sees as FrameObs.vad_speech), not raw segments."""
    faces_by_pts: Dict[int, List[FaceObs]] = {
        frame.pts_ms: frame.faces for frame in frames
    }
    entries: List[Dict[str, Any]] = []
    for index in range(count):
        pts = pts_list[index]
        faces = faces_by_pts.get(pts, [])
        entries.append({
            "pts_ms": pts,
            "vad": vad_near(pts, file_audio.silero_segments_local_ms,
                            params.vad_tol_ms),
            "faces": [
                {
                    "bbox": [round(v, 1) for v in face.bbox],
                    "mar": None if face.mar is None else round(face.mar, 4),
                    "det": round(face.det_score, 3),
                    "yaw": (None if face.yaw_degrees is None
                            else round(face.yaw_degrees, 1)),
                }
                for face in faces
            ],
        })
    return entries


# =============================================================================
# HTTP layer (flask imported lazily — the self-test must run without it)
# =============================================================================

def create_app(ui: UISession, session: ClickSession):
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask("spovnob_click_ui")
    lock = threading.Lock()
    count = ui.servable_frames
    timeline = timeline_entries(
        ui.frames, ui.pts_list, count, ui.file_audio, ui.params,
    )

    warnings: List[str] = []
    if ui.file_audio.vfr_suspected:
        warnings.append(
            "Variable frame rate suspected (avg ≠ r frame rate) — PTS "
            "pairing is still authoritative, but inspect boundaries."
        )
    if ui.scan_stats.get("pts_mismatch"):
        warnings.append(
            f"Frame/PTS count mismatch during scan (decoded "
            f"{ui.scan_stats['decoded_frames']}, listed "
            f"{ui.scan_stats['listed_pts']})."
        )
    if ui.frame_count != ui.scan_stats.get("decoded_frames"):
        warnings.append(
            f"Display frames ({ui.frame_count}) differ from scan decode "
            f"count ({ui.scan_stats.get('decoded_frames')}); pairing stops "
            "at the shorter."
        )

    def _no_store(payload) -> Any:
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> Any:
        return Response(HTML_PAGE, mimetype="text/html")

    @app.get("/meta")
    def meta() -> Any:
        p = ui.params
        return _no_store({
            "video_name": ui.video_path.name,
            "video_path": str(ui.video_path),
            "video_sha8": ui.video_sha8,
            "native_width": ui.native_width,
            "native_height": ui.native_height,
            "frame_count": count,
            "duration_ms": ui.file_audio.duration_ms,
            "audio_start_pts_ms": ui.file_audio.audio_start_pts_ms,
            "device": ui.device,
            "from_cache": ui.from_cache,
            "scan_stats": ui.scan_stats,
            "warnings": warnings,
            "clicks_path": str(ui.work_dir / CLICKS_FILENAME),
            "no_anti_warning": no_anti_warning(p),
            "params": {
                "mar_on": p.mar_on,
                "mar_off": p.mar_off,
                "yaw_max_degrees": p.yaw_max_degrees,
                "face_reid_threshold": p.face_reid_threshold,
                "seed_min_ms": p.seed_min_ms,
                "vad_tol_ms": p.vad_tol_ms,
                "click_overlap_max_frac": p.click_overlap_max_frac,
            },
            "state": session.state_payload(),
        })

    @app.get("/timeline")
    def get_timeline() -> Any:
        return _no_store({"frames": timeline})

    @app.get("/frames/<int:index>")
    def get_frame(index: int) -> Any:
        if not (0 <= index < count):
            return _no_store({"ok": False, "reason": "frame_out_of_range"}), 404
        return send_from_directory(
            ui.frames_dir, f"{index + 1:06d}.jpg",
            mimetype="image/jpeg", max_age=86400 * 30,
        )

    @app.get("/state")
    def get_state() -> Any:
        return _no_store({"state": session.state_payload()})

    @app.post("/click")
    def post_click() -> Any:
        data = request.get_json(silent=True) or {}
        pts_ms = data.get("pts_ms")
        x, y = data.get("x"), data.get("y")
        click_type = data.get("type")
        if (
            isinstance(pts_ms, bool) or not isinstance(pts_ms, int)
            or isinstance(x, bool) or not isinstance(x, (int, float))
            or isinstance(y, bool) or not isinstance(y, (int, float))
            or click_type not in ("speaking", "anti")
        ):
            return _no_store({
                "ok": False, "reason": "invalid_request",
                "message": reason_message("invalid_request", {}, ui.params),
                "state": session.state_payload(),
            }), 400
        with lock:
            if click_type == "speaking":
                result = session.register_speaking(pts_ms, float(x), float(y))
            else:
                result = session.register_anti(pts_ms, float(x), float(y))
            result["state"] = session.state_payload()
        return _no_store(result)

    @app.post("/clear")
    def post_clear() -> Any:
        data = request.get_json(silent=True) or {}
        click_type = data.get("type")
        if click_type not in ("speaking", "anti"):
            return _no_store({"ok": False, "reason": "invalid_request",
                              "state": session.state_payload()}), 400
        with lock:
            session.clear(click_type)
            return _no_store({"ok": True, "state": session.state_payload()})

    @app.post("/export")
    def post_export() -> Any:
        data = request.get_json(silent=True) or {}
        include_anti = bool(data.get("include_anti", True))
        with lock:
            result = session.write_export(ui.work_dir, include_anti)
            result["state"] = session.state_payload()
        return _no_store(result if result["ok"] else (result, 409))

    return app


# =============================================================================
# Frontend (single page, inline JS/CSS — air-gap safe, no external resources)
# =============================================================================

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Click UI</title>
<style>
  :root{--bg:#14161a;--panel:#1c1f24;--line:#2a2e35;--text:#d8dce2;--dim:#8a919c;
        --blue:#4a90d9;--orange:#e8842c;--green:#37d067;--red:#e06060;--purple:#b07fe0;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.45 -apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;}
  header{padding:10px 16px;border-bottom:1px solid var(--line);
         display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
  header .sub{color:var(--dim);font-size:12px}
  main{max-width:840px;margin:0 auto;padding:14px 16px 48px}
  #warnings .warn{margin-bottom:8px}
  #stage{background:#000;border:1px solid var(--line);border-radius:4px;
         display:flex;justify-content:center;min-height:120px}
  canvas#frame{display:block;cursor:crosshair;max-width:100%}
  .row{display:flex;align-items:center;gap:10px;margin-top:8px}
  input[type=range]{flex:1;accent-color:var(--blue)}
  #clock{font-variant-numeric:tabular-nums;color:var(--dim);font-size:12px;white-space:nowrap}
  canvas.strip{width:100%;display:block;border:1px solid var(--line);
               border-radius:2px;margin-top:6px;cursor:pointer}
  #tip{position:fixed;pointer-events:none;background:#000e;color:#fff;
       padding:3px 7px;border-radius:3px;font-size:11px;display:none;
       z-index:9;white-space:nowrap;font-variant-numeric:tabular-nums}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;
         padding:12px;margin-top:12px}
  .modes{display:flex;gap:8px}
  .modes button{flex:1;padding:7px;background:#23272e;color:var(--text);
                border:1px solid var(--line);border-radius:4px;cursor:pointer;font-size:13px}
  .modes button.active-s{border-color:var(--green);color:var(--green);font-weight:600}
  .modes button.active-a{border-color:var(--purple);color:var(--purple);font-weight:600}
  .modes button:disabled{opacity:.4;cursor:not-allowed}
  .clickrow{display:flex;align-items:center;gap:8px;margin-top:10px;
            font-variant-numeric:tabular-nums;font-size:13px}
  .clickrow .tag{width:108px;color:var(--dim)}
  .clickrow .val{flex:1}
  .clickrow button{background:#23272e;border:1px solid var(--line);color:var(--dim);
                   border-radius:3px;cursor:pointer;padding:2px 9px;font-size:12px}
  .clickrow.off{opacity:.4}
  #status div{margin-top:7px;font-size:13px}
  .ok{color:var(--green)} .err{color:var(--red)} .idle{color:var(--dim)}
  .warn{color:var(--orange);font-size:12px}
  .hint{color:var(--dim);font-size:12px;margin-top:8px}
  label.toggle{display:flex;gap:8px;align-items:center;margin-top:12px;font-size:13px;cursor:pointer}
  #noAntiWarn{margin-top:6px;display:none}
  #exportBtn{margin-top:12px;padding:8px 16px;background:var(--green);border:0;
             border-radius:4px;color:#08120b;font-weight:700;cursor:pointer;font-size:13px}
  #exportBtn:disabled{background:#2c3138;color:var(--dim);cursor:not-allowed}
  pre#exportOut{background:#101216;border:1px solid var(--line);border-radius:4px;
                padding:10px;font-size:12px;overflow:auto;display:none;margin-top:10px;
                white-space:pre-wrap}
  .legend b{color:var(--text)}
</style>
</head>
<body>
<header>
  <h1>Click UI</h1>
  <span class="sub" id="videoName">loading…</span>
  <span class="sub" id="devInfo"></span>
</header>
<main>
  <div id="warnings"></div>
  <div id="stage"><canvas id="frame"></canvas></div>
  <div class="row">
    <input type="range" id="scrub" min="0" max="0" value="0" step="1">
    <span id="clock"></span>
  </div>
  <canvas class="strip" id="stripFaces" height="16"></canvas>
  <canvas class="strip" id="stripVad" height="10"></canvas>
  <div class="hint legend">
    faces: <span style="color:var(--blue)">&#9632; 1</span>
    <span style="color:var(--orange)">&#9632; 2+</span> &nbsp;·&nbsp;
    audio: <span style="color:var(--green)">&#9632; Silero speech</span>
    &nbsp;·&nbsp; click a strip to seek &nbsp;·&nbsp;
    <b>&larr;/&rarr;</b> step (<b>&#8679;</b> ×10) &nbsp;·&nbsp;
    <b>s</b>/<b>a</b> mode &nbsp;·&nbsp; click a face on the frame to register
  </div>

  <div class="panel">
    <div class="modes">
      <button id="modeS">Speaking click&nbsp;(s)</button>
      <button id="modeA">Anti click&nbsp;(a)</button>
    </div>
    <div class="clickrow" id="rowS">
      <span class="tag">Speaking click</span>
      <span class="val" id="valS">— not set</span>
      <button id="clearS">Clear</button>
    </div>
    <div class="clickrow" id="rowA">
      <span class="tag">Anti click</span>
      <span class="val" id="valA">— not set</span>
      <button id="clearA">Clear</button>
    </div>
    <label class="toggle">
      <input type="checkbox" id="includeAnti" checked>
      Include anti-click (Track C) — uncheck for the NO_ANTI_PROFILE path
    </label>
    <div class="warn" id="noAntiWarn"></div>
    <div id="status">
      <div id="statS" class="idle">&#9675; speaking click not registered</div>
      <div id="statA" class="idle">&#9675; anti click not registered</div>
    </div>
    <button id="exportBtn" disabled>Export clicks.json</button>
    <pre id="exportOut"></pre>
  </div>
</main>
<div id="tip"></div>

<script>
"use strict";
const $ = id => document.getElementById(id);
let META = null, TL = [], N = 0, cur = 0, mode = "speaking";
let state = {speaking: null, anti: null};
let lastMsg = {speaking: null, anti: null};
let fx = 1, fy = 1;
const imgs = new Map();
let offF = null, offV = null;

function fmt(ms){
  const s = Math.floor(ms/1000), m = Math.floor(s/60);
  return String(m).padStart(2,"0") + ":" + String(s%60).padStart(2,"0") +
         "." + String(ms%1000).padStart(3,"0");
}
async function postJSON(url, body){
  const r = await fetch(url, {method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  return r.json();
}
function idxForPts(p){
  let lo = 0, hi = N - 1;
  while (lo < hi){ const m = (lo+hi)>>1; if (TL[m].pts_ms < p) lo = m+1; else hi = m; }
  if (lo > 0 && Math.abs(TL[lo-1].pts_ms-p) <= Math.abs(TL[lo].pts_ms-p)) lo--;
  return lo;
}

/* ---------- frame display ---------- */
function ensure(i){
  if (i < 0 || i >= N) return null;
  if (!imgs.has(i)){
    const im = new Image();
    im.onload = () => { if (i === cur) draw(); };
    im.src = "frames/" + i + "?v=" + META.video_sha8;
    imgs.set(i, im);
  }
  return imgs.get(i);
}
function show(i){
  cur = Math.max(0, Math.min(N - 1, i));
  $("scrub").value = cur;
  const im = ensure(cur);
  for (let d = 1; d <= 12; d++){ ensure(cur + d); ensure(cur - d); }
  if (im && im.complete && im.naturalWidth) draw();
  updateClock();
  drawStrips();
}
function updateClock(){
  if (!N) return;
  const pts = TL[cur].pts_ms;
  $("clock").textContent = "frame " + (cur+1) + "/" + N + " · PTS " + pts +
    " ms · " + fmt(pts) + " / " + fmt(META.duration_ms);
}
function crosshair(ctx, x, y, color, label){
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x-14, y); ctx.lineTo(x+14, y);
  ctx.moveTo(x, y-14); ctx.lineTo(x, y+14); ctx.stroke();
  ctx.font = "bold 12px sans-serif";
  ctx.fillText(label, x + 12, y - 10);
}
function draw(){
  const im = imgs.get(cur);
  if (!im || !im.naturalWidth) return;
  const cv = $("frame"), ctx = cv.getContext("2d");
  cv.width = im.naturalWidth; cv.height = im.naturalHeight;
  fx = im.naturalWidth / META.native_width;
  fy = im.naturalHeight / META.native_height;
  ctx.drawImage(im, 0, 0);
  const e = TL[cur];
  ctx.lineWidth = 4;
  ctx.strokeStyle = e.vad ? "#37d067" : "#3a3f46";
  ctx.strokeRect(2, 2, cv.width - 4, cv.height - 4);
  ctx.font = "12px sans-serif";
  for (const f of e.faces){
    const x1 = f.bbox[0]*fx, y1 = f.bbox[1]*fy,
          x2 = f.bbox[2]*fx, y2 = f.bbox[3]*fy;
    const suspended = f.yaw !== null && Math.abs(f.yaw) > META.params.yaw_max_degrees;
    ctx.lineWidth = 2;
    ctx.strokeStyle = suspended ? "#e8842c" : "#37d067";
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    const label = "MAR " + (f.mar === null ? "—" : f.mar.toFixed(3)) +
      "  det " + f.det.toFixed(2) +
      (suspended ? "  yaw " + f.yaw.toFixed(0) + "° (suspended)" : "");
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "#000c";
    ctx.fillRect(x1, Math.max(0, y1-16), tw+8, 15);
    ctx.fillStyle = suspended ? "#e8842c" : "#37d067";
    ctx.fillText(label, x1+4, Math.max(11, y1-4));
  }
  const pts = e.pts_ms;
  if (state.speaking && state.speaking.pts_ms === pts)
    crosshair(ctx, state.speaking.x*fx, state.speaking.y*fy, "#37d067", "S");
  if (state.anti && state.anti.pts_ms === pts)
    crosshair(ctx, state.anti.x*fx, state.anti.y*fy, "#b07fe0", "A");
}

/* ---------- timeline strips ---------- */
function buildStrips(){
  for (const id of ["stripFaces", "stripVad"]){
    const el = $(id); el.width = el.clientWidth || 800;
  }
  const sF = $("stripFaces"), sV = $("stripVad");
  offF = document.createElement("canvas"); offF.width = sF.width; offF.height = sF.height;
  offV = document.createElement("canvas"); offV.width = sV.width; offV.height = sV.height;
  const cF = offF.getContext("2d"), cV = offV.getContext("2d");
  cF.fillStyle = "#1d2025"; cF.fillRect(0, 0, offF.width, offF.height);
  cV.fillStyle = "#1d2025"; cV.fillRect(0, 0, offV.width, offV.height);
  const W = offF.width;
  for (let i = 0; i < N; i++){
    const x0 = i*W/N, w = Math.max(1, W/N);
    const n = TL[i].faces.length;
    if (n > 0){
      cF.fillStyle = n === 1 ? "#4a90d9" : "#e8842c";
      cF.fillRect(x0, 0, w, offF.height);
    }
    if (TL[i].vad){
      cV.fillStyle = "#3da35d";
      cV.fillRect(x0, 0, w, offV.height);
    }
  }
}
function markStrip(ctx, el, click, color){
  if (!click) return;
  const x = (idxForPts(click.pts_ms) + 0.5) * el.width / N;
  ctx.fillStyle = color;
  ctx.fillRect(x - 1.25, 0, 2.5, el.height);
}
function drawStrips(){
  if (!offF) return;
  for (const [id, off] of [["stripFaces", offF], ["stripVad", offV]]){
    const el = $(id), ctx = el.getContext("2d");
    ctx.clearRect(0, 0, el.width, el.height);
    ctx.drawImage(off, 0, 0);
    markStrip(ctx, el, state.speaking, "#37d067");
    markStrip(ctx, el, state.anti, "#b07fe0");
    const x = (cur + 0.5) * el.width / N;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x - 0.75, 0, 1.5, el.height);
  }
}
function stripIndex(el, ev){
  const r = el.getBoundingClientRect();
  const frac = (ev.clientX - r.left) / r.width;
  return Math.max(0, Math.min(N - 1, Math.floor(frac * N)));
}
function bindStrip(id){
  const el = $(id), tip = $("tip");
  el.addEventListener("click", ev => show(stripIndex(el, ev)));
  el.addEventListener("mousemove", ev => {
    const i = stripIndex(el, ev), e = TL[i];
    tip.textContent = e.pts_ms + " ms · " + e.faces.length + " face(s) · " +
                      (e.vad ? "speech" : "silence");
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 12) + "px";
    tip.style.top = (ev.clientY - 26) + "px";
  });
  el.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
}

/* ---------- registration state / status panel ---------- */
function setMode(m){
  if (m === "anti" && !$("includeAnti").checked) return;
  mode = m;
  $("modeS").className = m === "speaking" ? "active-s" : "";
  $("modeA").className = m === "anti" ? "active-a" : "";
}
function statusLine(el, type){
  const last = lastMsg[type], reg = state[type];
  if (last){
    el.className = last.ok ? "ok" : "err";
    el.innerHTML = (last.ok ? "&#10003; " : "&#10007; ") + last.message;
  } else if (reg){
    el.className = "ok";
    el.innerHTML = "&#10003; " + reg.summary;
  } else {
    el.className = "idle";
    el.innerHTML = "&#9675; " + (type === "speaking" ? "speaking" : "anti") +
                   " click not registered";
  }
}
function applyState(st){
  state = st;
  const vs = $("valS"), va = $("valA");
  vs.textContent = state.speaking
    ? state.speaking.pts_ms + " ms  (" + state.speaking.x.toFixed(0) + ", " +
      state.speaking.y.toFixed(0) + ")"
    : "— not set";
  va.textContent = state.anti
    ? state.anti.pts_ms + " ms  (" + state.anti.x.toFixed(0) + ", " +
      state.anti.y.toFixed(0) + ")"
    : "— not set";
  statusLine($("statS"), "speaking");
  statusLine($("statA"), "anti");
  updateExportButton();
  drawStrips();
  draw();
}
function updateExportButton(){
  const inc = $("includeAnti").checked;
  const ready = !!state.speaking && (!inc || !!state.anti);
  const btn = $("exportBtn");
  btn.disabled = !ready;
  btn.title = ready ? "" :
    (!state.speaking ? "Register the speaking click first"
                     : "Register the anti click or uncheck “Include anti-click”");
}
function applyAntiToggle(){
  const inc = $("includeAnti").checked;
  $("rowA").classList.toggle("off", !inc);
  $("modeA").disabled = !inc;
  $("statA").style.display = inc ? "" : "none";
  $("noAntiWarn").style.display = inc ? "none" : "block";
  if (!inc && mode === "anti") setMode("speaking");
  updateExportButton();
}

/* ---------- actions ---------- */
async function registerClick(ev){
  if (!N || !imgs.get(cur) || !imgs.get(cur).naturalWidth) return;
  const nx = Number((ev.offsetX / fx).toFixed(2));
  const ny = Number((ev.offsetY / fy).toFixed(2));
  const type = mode;
  const resp = await postJSON("click",
    {pts_ms: TL[cur].pts_ms, x: nx, y: ny, type: type});
  lastMsg[type] = {ok: resp.ok, message: resp.message};
  if (resp.anti_cleared)
    lastMsg.anti = {ok: false,
      message: "anti click cleared — the target lock changed; re-register it"};
  if (resp.ok && type === "speaking" && $("includeAnti").checked) setMode("anti");
  applyState(resp.state);
}
async function clearClick(type){
  const resp = await postJSON("clear", {type: type});
  lastMsg[type] = null;
  if (type === "speaking") lastMsg.anti = null;
  applyState(resp.state);
}
async function exportClicks(){
  const resp = await postJSON("export", {include_anti: $("includeAnti").checked});
  const out = $("exportOut");
  out.style.display = "block";
  if (resp.ok){
    out.style.color = "";
    out.textContent = "✔ " + resp.message +
      (resp.no_anti ? "\n⚠ " + META.no_anti_warning : "") +
      "\n\n" + JSON.stringify(resp.clicks, null, 2);
  } else {
    out.style.color = "#e06060";
    out.textContent = "✗ " + resp.message;
  }
  if (resp.state) applyState(resp.state);
}

/* ---------- init ---------- */
async function init(){
  META = await (await fetch("meta")).json();
  TL = (await (await fetch("timeline")).json()).frames;
  N = TL.length;
  $("videoName").textContent = META.video_name + "  →  " + META.clicks_path;
  $("devInfo").textContent = META.device +
    (META.from_cache ? " · cached pre-scan" : " · fresh pre-scan") +
    " · " + N + " frames · " + fmt(META.duration_ms);
  for (const w of META.warnings || []){
    const d = document.createElement("div");
    d.className = "warn"; d.textContent = "⚠ " + w;
    $("warnings").appendChild(d);
  }
  $("noAntiWarn").textContent = "⚠ " + META.no_anti_warning;
  $("scrub").max = N - 1;
  buildStrips();
  bindStrip("stripFaces"); bindStrip("stripVad");
  applyState(META.state);
  applyAntiToggle();
  setMode("speaking");
  show(0);

  $("scrub").addEventListener("input", () => show(+$("scrub").value));
  $("frame").addEventListener("click", registerClick);
  $("modeS").addEventListener("click", () => setMode("speaking"));
  $("modeA").addEventListener("click", () => setMode("anti"));
  $("clearS").addEventListener("click", () => clearClick("speaking"));
  $("clearA").addEventListener("click", () => clearClick("anti"));
  $("includeAnti").addEventListener("change", applyAntiToggle);
  $("exportBtn").addEventListener("click", exportClicks);
  document.addEventListener("keydown", ev => {
    if (ev.target.tagName === "INPUT" && ev.target.type === "checkbox") ev.target.blur();
    if (ev.target.tagName === "TEXTAREA") return;
    const step = ev.shiftKey ? 10 : 1;
    if (ev.key === "ArrowLeft"){ show(cur - step); ev.preventDefault(); }
    else if (ev.key === "ArrowRight"){ show(cur + step); ev.preventDefault(); }
    else if (ev.key === "Home"){ show(0); ev.preventDefault(); }
    else if (ev.key === "End"){ show(N - 1); ev.preventDefault(); }
    else if (ev.key === "s") setMode("speaking");
    else if (ev.key === "a") setMode("anti");
  });
}
init();
</script>
</body>
</html>
"""


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB clicking UI — visual, guardrail-validated "
                    "creation of clicks.json",
    )
    parser.add_argument("video", nargs="?", type=Path,
                        help="the first video of the session (file_index 0)")
    parser.add_argument("--selftest", action="store_true",
                        help="stdlib-only self-test (no flask/torch/ffmpeg)")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                        help="session work dir (clicks.json + ui_cache/) "
                             "[default: session]")
    parser.add_argument("--model-store", type=Path, default=None,
                        help="vendored model store (canonical: "
                             "/opt/spovnob/model_store); required on the "
                             "first scan of a video, optional on warm starts")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cpu", action="store_true",
                        help="development only: run the pre-scan on CPU "
                             "(the bench uses CUDA)")
    parser.add_argument("--rescan", action="store_true",
                        help="discard the cached pre-scan and rebuild")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.video is None:
        parser.error("video path is required (or use --selftest)")

    try:
        ui = prepare_session(
            video=args.video,
            work_dir=args.work_dir,
            model_store=args.model_store,
            device="cpu" if args.cpu else "cuda",
            rescan=args.rescan,
        )
    except ClickUIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    session = ClickSession(ui.frames, ui.file_audio, ui.params)
    try:
        app = create_app(ui, session)
    except ImportError:
        print("ERROR: flask is not installed — pip install flask==3.0.2 "
              "(see requirements.txt, operator tooling section)",
              file=sys.stderr)
        return 1

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    url = f"http://127.0.0.1:{args.port}/"
    print()
    print(f"  SPOVNOB Click UI — {ui.video_path.name}")
    print(f"  frames: {ui.servable_frames}   device: {ui.device}   "
          f"pre-scan: {'cache' if ui.from_cache else 'fresh'}")
    print(f"  export target: {ui.work_dir / CLICKS_FILENAME}")
    print(f"  open: {url}")
    print()
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False,
                use_reloader=False, threaded=True)
    except OSError as exc:
        print(f"ERROR: cannot bind 127.0.0.1:{args.port} ({exc}) — "
              "use --port", file=sys.stderr)
        return 1
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch, no
# flask, no ffmpeg, no GPU). Drives the REAL imported production functions
# over synthetic frames, so the guardrail wiring is exercised end-to-end.
# =============================================================================

def _selftest() -> int:
    import tempfile

    assert "torch" not in sys.modules, "torch imported at module level"
    assert "flask" not in sys.modules, "flask imported at module level"

    params = EnrollmentParams()
    emb_target = [1.0, 0.0, 0.0, 0.0]
    emb_interviewer = [0.0, 1.0, 0.0, 0.0]

    def face(embedding: List[float], mar: float, cx: float, cy: float) -> FaceObs:
        return FaceObs(
            bbox=(cx - 50.0, cy - 50.0, cx + 50.0, cy + 50.0),
            det_score=0.9, embedding=list(embedding), mar=mar, yaw_degrees=0.0,
        )

    def build_frames(
        interviewer_mar: float = 0.05,
        interviewer_spans: Tuple[Tuple[int, int], ...] = (
            (5000, 6000), (7300, 8000), (8800, 9100),
        ),
    ) -> List[FrameFaces]:
        # Target on screen 0–9600 ms at (400,300): speaking (MAR .5) during
        # 2000–7000 and 8300–8600, listening (MAR .05) otherwise. The
        # interviewer at (800,300), lips at `interviewer_mar`. 9700–10600 ms:
        # frames analyzed but empty (nobody detected).
        frames: List[FrameFaces] = []
        for pts in range(0, 10601, 100):
            faces: List[FaceObs] = []
            if pts <= 9600:
                speaking = 2000 <= pts <= 7000 or 8300 <= pts <= 8600
                faces.append(face(emb_target, 0.5 if speaking else 0.05,
                                  400.0, 300.0))
                if any(a <= pts <= b for a, b in interviewer_spans):
                    faces.append(face(emb_interviewer, interviewer_mar,
                                      800.0, 300.0))
            frames.append(FrameFaces(pts_ms=pts, faces=faces))
        return frames

    segments = [(1900, 7100), (7250, 8050), (8250, 8650)]
    file_audio = FileAudio(
        file_index=0, source_path="synthetic.mp4", wav_path="",
        source_sha256="", wav_sha256="", num_samples=10600 * 16,
        duration_ms=10600, audio_start_pts_ms=0, audio_start_missing=False,
        vfr_suspected=False, file_offset_ms=0, pcm=b"",
        silero_segments_local_ms=segments,
    )

    # 1. Ordering: anti before speaking is refused.
    session = ClickSession(build_frames(), file_audio, params)
    result = session.register_anti(3000, 800.0, 300.0)
    assert not result["ok"]
    assert result["reason"] == "speaking_click_required_first", result

    # 2. Valid speaking click — the real window machine finds the seed
    #    window (VAD-gated start at 2000, plosive expiry at 8100).
    result = session.register_speaking(4000, 400.0, 300.0)
    assert result["ok"], result
    window = result["detail"]["window"]
    assert window["t_start_ms"] == 2000 and window["t_stop_ms"] == 8100, window
    assert window["duration_ms"] >= params.seed_min_ms
    assert session.f_target is not None
    assert cosine(session.f_target, emb_target) > 0.99

    # 3. Click in silence: no speaking window contains it.
    result = session.register_speaking(9500, 400.0, 300.0)
    assert result["reason"] == "click_outside_speaking_window", result
    assert session.speaking is not None          # failures never mutate state

    # 4. Short burst (8300–9300 incl. plosive tail = 1000 ms): guardrail 2.
    result = session.register_speaking(8400, 400.0, 300.0)
    assert result["reason"] == "seed_too_short", result
    assert result["detail"]["duration_ms"] == 1000, result["detail"]

    # 5. Anti on the target's own face: guardrail 3.
    result = session.register_anti(3000, 400.0, 300.0)
    assert result["reason"] == "anti_click_matches_target", result
    assert result["detail"]["sim_to_target"] > 0.99

    # 6. Anti while the target's lips are open (target may be speaking).
    result = session.register_anti(5500, 800.0, 300.0)
    assert result["reason"] == "target_lips_open_at_anti_click", result

    # 7. Anti where Silero shows silence.
    result = session.register_anti(8900, 800.0, 300.0)
    assert result["reason"] == "no_audio_energy_at_anti_click", result

    # 8. Anti with no face within CLICK_MATCH_MAX_GAP_MS.
    result = session.register_anti(10300, 800.0, 300.0)
    assert result["reason"] == "no_face_at_anti_click", result

    # 9. Valid anti click: interviewer on camera, lips closed, target
    #    listening, Silero active.
    result = session.register_anti(7600, 800.0, 300.0)
    assert result["ok"], result
    assert result["detail"]["sim_to_target_face"] < params.face_reid_threshold

    # 10. Re-registering the speaking click drops the stale anti click.
    result = session.register_speaking(4100, 400.0, 300.0)
    assert result["ok"] and result["anti_cleared"] is True
    assert session.anti is None
    assert session.register_anti(7600, 800.0, 300.0)["ok"]

    # 11. Guardrail 1: interviewer visibly talking inside the seed window.
    noisy = ClickSession(
        build_frames(interviewer_mar=0.5, interviewer_spans=((5000, 6000),)),
        file_audio, params,
    )
    result = noisy.register_speaking(4000, 400.0, 300.0)
    assert result["reason"] == "overlap_at_speaking_click", result
    assert result["detail"]["non_target_lips_open_frac"] > params.click_overlap_max_frac

    # 12. Export round-trips through the production parser, both paths.
    from layer1_enrollment.enrollment import load_clicks
    with tempfile.TemporaryDirectory() as tmp:
        out = session.write_export(Path(tmp), include_anti=True)
        assert out["ok"], out
        parsed = load_clicks(out["path"])
        assert parsed.speaking.pts_ms == 4100 and parsed.speaking.file_index == 0
        assert parsed.anti is not None and parsed.anti.pts_ms == 7600

        out = session.write_export(Path(tmp), include_anti=False)
        assert out["ok"] and out["no_anti"], out
        raw = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
        assert "anti_click" not in raw
        assert load_clicks(out["path"]).anti is None

        empty = ClickSession(build_frames(), file_audio, params)
        out = empty.write_export(Path(tmp), include_anti=False)
        assert not out["ok"]
        assert out["reason"] == "export_requires_speaking_click"

        session.clear("anti")
        out = session.write_export(Path(tmp), include_anti=True)
        assert not out["ok"] and out["reason"] == "anti_click_not_registered"

    # 13. Clearing the speaking click clears the dependent state.
    session.clear("speaking")
    assert session.speaking is None and session.f_target is None
    assert session.anti is None

    # 14. Every reason renders a non-empty operator message.
    for reason in ALL_REASONS:
        message = reason_message(reason, {}, params)
        assert isinstance(message, str) and len(message) > 20, reason

    # 15. Timeline entries: production vad_near semantics + face metadata.
    frames = build_frames()
    pts_list = [frame.pts_ms for frame in frames]
    entries = timeline_entries(frames, pts_list, len(pts_list),
                               file_audio, params)
    assert len(entries) == len(pts_list)
    by_pts = {entry["pts_ms"]: entry for entry in entries}
    assert by_pts[4000]["vad"] is True and len(by_pts[4000]["faces"]) == 1
    assert by_pts[5500]["vad"] is True and len(by_pts[5500]["faces"]) == 2
    assert by_pts[9500]["vad"] is False
    assert by_pts[10300]["faces"] == []
    assert by_pts[4000]["faces"][0]["mar"] == 0.5
    # vad_near tolerance edge: frame 7200 sits outside segment (7250, 8050)
    # but within the ±vad_tol_ms=50 tolerance; 8800 is past 8650+50.
    assert by_pts[7200]["vad"] is True
    assert by_pts[8800]["vad"] is False

    # 16. Cache key: deterministic, sensitive to device and registry.
    key_a = build_cache_key("sha", params, "cuda", "reg")
    assert key_a == build_cache_key("sha", params, "cuda", "reg")
    assert key_a != build_cache_key("sha", params, "cpu", "reg")
    assert sha256_of_obj(key_a) == sha256_of_obj(
        build_cache_key("sha", params, "cuda", "reg"))
    assert _keys_match(build_cache_key("sha", params, "cuda", None), key_a,
                       model_store_given=False)
    assert not _keys_match(build_cache_key("sha", params, "cuda", None), key_a,
                           model_store_given=True)

    # 17. Frontend page sanity (catches accidental truncation/renames).
    for needle in ('id="frame"', 'id="scrub"', 'id="stripFaces"',
                   'id="stripVad"', 'id="includeAnti"', 'id="exportBtn"',
                   '"click"', '"export"', 'fetch("meta")', 'fetch("timeline")'):
        assert needle in HTML_PAGE, f"HTML_PAGE missing {needle}"

    assert "torch" not in sys.modules, "self-test imported torch"
    assert "flask" not in sys.modules, "self-test imported flask"
    assert "cv2" not in sys.modules, "self-test imported cv2"
    print("click_ui stdlib self-test OK — production guardrail functions "
          "exercised; no torch, no flask, no cv2, no ffmpeg, no GPU")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
