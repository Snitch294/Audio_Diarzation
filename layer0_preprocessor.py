"""
SPOVNOB — Module 1: layer0_preprocessor.py
===========================================

Layer:      Layer 0 — Foundational Extraction.
            Imports only Module 0a (session_manifest) and Module 0b
            (environment_gate, first import — fixes the process env).

Purpose:    Convert raw videos into PTS-true 16kHz mono audio plus a
            Silero speech segment map, preloaded into RAM for the whole
            batch. No identity decisions. No signal modification beyond
            the mandated 16kHz mono resample. Audio is NEVER zeroed,
            padded, trimmed, or interpolated in storage.

Inputs:     - batch of video files (.mp4/.mov/.mkv)
            - resident Silero VAD model (CPU TorchScript, from the
              environment gate's ResidentModels — never loaded here)
            - open SessionManifest
Outputs:    - per-file 16kHz mono PCM16 WAV on disk (work dir)
            - BatchAudio: PCM bytes in RAM + per-file PTS metadata
              (audio_start_pts_ms, duration_ms, file_offset_ms) +
              Silero segments as integer-ms (start_ms, end_ms) pairs
            - manifest entries: batch_init (source SHA-256s, ffmpeg
              version, canonical order), one layer0_file entry per file,
              video_gap entries for files after the first.

Implements (Audio_Diarization.md):
            - Layer 0: Purpose / Input Scenario / Key Constraints
            - "Global PTS Clock Initialization" (file_offset_ms,
              global_ms = file_offset_ms + local_pts_ms; logs carry both)
            - "FFmpeg Audio Strip" (PTS-true: the audio stream's own
              start PTS is probed and carried; video frame counting is
              never used for time anywhere in this module)
            - "Pre-Gate: Silero Neural VAD" (non-destructive masking:
              full unmodified audio in, segment map out)
            - System Environment: "Memory & Storage Policy" (RAM preload)
              and "Parallel Execution Model" (parallel CPU extraction;
              manifest written single-writer in canonical order after
              the parallel stage joins)
            - Cross-File Behavior: video gap logging

CUDA determinism dependencies:
            None directly — this module is CPU-only. It relies on
            environment_gate's import-time environment fixing and on the
            fixed torch thread count for Silero reproducibility. Silero
            windows are processed strictly sequentially per file, files
            in canonical order.

Determinism notes:
            - All time values are integer PTS milliseconds (Rule 6);
              sample->ms uses integer floor division; ffprobe decimal
              seconds -> ms uses Decimal with ROUND_HALF_EVEN.
            - Canonical file order = lexicographic full-path sort,
              recorded in the manifest; all per-file outputs and
              manifest entries are emitted in that order regardless of
              extraction scheduling.
            - Silero segment derivation is a pure function over the
              per-window probabilities (binary threshold per the
              parameter table; short-run drop / gap merge / edge pad
              constants documented below and manifest-logged).
            - The final partial Silero window is zero-padded for
              INFERENCE ONLY; the stored audio is untouched.
            - ffmpeg is invoked via subprocess with a fully explicit
              argument list (no ffmpeg-python call assembly) so the
              exact command line is auditable and manifest-recordable.

Self-test:  ``python3 layer0_preprocessor.py --selftest`` — stdlib only
            (zero pip installs, no torch, no GPU, no ffmpeg binary).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import json
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from session_manifest import (
    ManifestTimeError,
    Operation,
    SessionManifest,
    sha256_of_file,
    validate_time_fields,
)

# --- Layer 0 constants (manifest-logged on every batch) ----------------------
SAMPLE_RATE = 16000                  # architectural (whole pipeline is 16kHz)
SILERO_WINDOW_SAMPLES = 512          # 32 ms windows (Silero v4, 16kHz)
SILERO_THRESHOLD = 0.50              # parameter table: silero_threshold
SILERO_MIN_SPEECH_MS = 250           # drop speech runs shorter than this
SILERO_MIN_SILENCE_MS = 100          # merge speech runs across gaps under this
SILERO_SPEECH_PAD_MS = 30            # widen kept segments by this much per side
EXTRACT_WORKERS = 8                  # parallel ffmpeg extraction processes

OP_LAYER0_FILE = "layer0_file"


class Layer0Error(RuntimeError):
    """Raised on unrecoverable Layer 0 failures (always preceded by a
    blocking_halt manifest entry when a manifest is in scope)."""


# =============================================================================
# Pure time / parsing helpers (stdlib-only; covered by the self-test)
# =============================================================================

def ms_from_samples(num_samples: int) -> int:
    """Integer floor: sample count -> milliseconds at 16kHz."""
    return (num_samples * 1000) // SAMPLE_RATE


def decimal_seconds_to_ms(value: str) -> int:
    """ffprobe decimal-seconds string -> integer milliseconds
    (Decimal, ROUND_HALF_EVEN — fixed deterministic rounding rule)."""
    return int(
        (Decimal(value) * 1000).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    )


@dataclass
class ProbeInfo:
    audio_start_pts_ms: int
    audio_start_missing: bool
    source_sample_rate: Optional[int]
    vfr_suspected: bool
    avg_frame_rate: Optional[str]
    r_frame_rate: Optional[str]


def parse_probe(probe: Dict[str, Any]) -> ProbeInfo:
    """Extract PTS-relevant facts from an ffprobe JSON document.
    Pure function: testable without the ffprobe binary."""
    audio = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    if audio is None:
        raise Layer0Error("no audio stream found in container")
    start_raw = audio.get("start_time")
    if start_raw in (None, "N/A"):
        start_ms, start_missing = 0, True
    else:
        start_ms, start_missing = decimal_seconds_to_ms(start_raw), False

    video = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    avg_rate = video.get("avg_frame_rate") if video else None
    r_rate = video.get("r_frame_rate") if video else None

    def _fraction(text: Optional[str]) -> Optional[Fraction]:
        if not text or text in ("0/0", "N/A"):
            return None
        return Fraction(text)

    avg_f, r_f = _fraction(avg_rate), _fraction(r_rate)
    vfr = avg_f is not None and r_f is not None and avg_f != r_f

    rate_raw = audio.get("sample_rate")
    return ProbeInfo(
        audio_start_pts_ms=start_ms,
        audio_start_missing=start_missing,
        source_sample_rate=int(rate_raw) if rate_raw else None,
        vfr_suspected=vfr,
        avg_frame_rate=avg_rate,
        r_frame_rate=r_rate,
    )


def segments_from_window_probs(
    probs: Sequence[float],
    total_samples: int,
    threshold: float = SILERO_THRESHOLD,
    window_samples: int = SILERO_WINDOW_SAMPLES,
    min_speech_ms: int = SILERO_MIN_SPEECH_MS,
    min_silence_ms: int = SILERO_MIN_SILENCE_MS,
    pad_ms: int = SILERO_SPEECH_PAD_MS,
) -> List[Tuple[int, int]]:
    """Per-window speech probabilities -> integer-ms speech segments,
    relative to the start of the audio data (caller adds the audio
    stream's start PTS to obtain local PTS).

    Deterministic pipeline, in this fixed order:
      1. binary threshold per window (parameter table: binary decision)
      2. merge speech runs separated by silence < min_silence_ms
      3. drop runs shorter than min_speech_ms (measured before padding)
      4. pad each side by pad_ms, clamped to [0, audio duration],
         merging any overlaps the padding creates.
    Pure function: testable with injected probability lists."""
    end_of_audio_ms = ms_from_samples(total_samples)

    # 1. threshold -> window-index runs
    runs: List[List[int]] = []
    for index, prob in enumerate(probs):
        if prob >= threshold:
            if runs and runs[-1][1] == index:
                runs[-1][1] = index + 1
            else:
                runs.append([index, index + 1])

    # window run -> ms span (end clamped to true audio end)
    spans = [
        (
            ms_from_samples(first * window_samples),
            min(ms_from_samples(last * window_samples), end_of_audio_ms),
        )
        for first, last in runs
    ]

    # 2. merge across short silences
    merged: List[List[int]] = []
    for start, end in spans:
        if merged and (start - merged[-1][1]) < min_silence_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # 3. drop short speech, 4. pad + re-merge
    padded: List[List[int]] = []
    for start, end in merged:
        if (end - start) < min_speech_ms:
            continue
        start = max(0, start - pad_ms)
        end = min(end_of_audio_ms, end + pad_ms)
        if padded and start <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])
    return [(s, e) for s, e in padded]


def read_wav_pcm16(path: Path) -> Tuple[int, bytes]:
    """Read a 16kHz mono PCM16 WAV via the stdlib ``wave`` module.
    Returns (num_samples, raw little-endian int16 bytes)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise Layer0Error(f"{path}: expected mono, got {wav.getnchannels()}ch")
        if wav.getsampwidth() != 2:
            raise Layer0Error(f"{path}: expected 16-bit PCM")
        if wav.getframerate() != SAMPLE_RATE:
            raise Layer0Error(
                f"{path}: expected {SAMPLE_RATE}Hz, got {wav.getframerate()}"
            )
        num_samples = wav.getnframes()
        return num_samples, wav.readframes(num_samples)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FileAudio:
    """One file's preloaded audio + PTS metadata. ``pcm`` is the raw
    int16 little-endian sample buffer held in RAM for the whole batch."""

    file_index: int
    source_path: str
    wav_path: str
    source_sha256: str
    wav_sha256: str
    num_samples: int
    duration_ms: int
    audio_start_pts_ms: int
    audio_start_missing: bool
    vfr_suspected: bool
    file_offset_ms: int = 0
    pcm: bytes = b""
    silero_segments_local_ms: List[Tuple[int, int]] = field(default_factory=list)

    def to_global_ms(self, local_pts_ms: int) -> int:
        """global_ms = file_offset_ms + local_pts_ms (Layer 0 clock rule)."""
        return self.file_offset_ms + local_pts_ms

    def sample_to_local_pts_ms(self, sample_index: int) -> int:
        return self.audio_start_pts_ms + ms_from_samples(sample_index)


@dataclass
class BatchAudio:
    files: List[FileAudio]

    def total_speech_ms(self) -> int:
        return sum(
            end - start
            for file_audio in self.files
            for start, end in file_audio.silero_segments_local_ms
        )


def layer0_file_payload(file_audio: FileAudio) -> Dict[str, Any]:
    """OP_LAYER0_FILE manifest payload (pure — self-testable without ffmpeg).

    Segments are nested ``{"start_ms": ..., "end_ms": ...}`` dicts so Rule 6
    actively validates every boundary as an integer; a bare pair list under
    an ``_ms``-suffixed key would be rejected by ``validate_time_fields``.
    """
    return {
        "file_index": file_audio.file_index,
        "source": file_audio.source_path,
        "wav_sha256": file_audio.wav_sha256,
        "num_samples": file_audio.num_samples,
        "duration_ms": file_audio.duration_ms,
        "audio_start_pts_ms": file_audio.audio_start_pts_ms,
        "audio_start_missing": file_audio.audio_start_missing,
        "file_offset_ms": file_audio.file_offset_ms,
        "vfr_suspected": file_audio.vfr_suspected,
        "silero_segments": [
            {"start_ms": start, "end_ms": end}
            for start, end in file_audio.silero_segments_local_ms
        ],
        "silero_total_speech_ms": sum(
            end - start for start, end in file_audio.silero_segments_local_ms
        ),
    }


# =============================================================================
# Subprocess wrappers (ffmpeg / ffprobe — never invoked by the self-test)
# =============================================================================

def ffmpeg_version() -> str:
    out = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()[0].strip()


def run_ffprobe(path: Path) -> Dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def extract_audio(source: Path, destination: Path) -> List[str]:
    """PTS-true audio strip: first audio stream -> 16kHz mono PCM16 WAV.
    Fully explicit, auditable argument list. No video decoding, no frame
    counting; the audio stream's own PTS offset is captured by ffprobe
    and carried in metadata rather than baked into the samples."""
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", str(source),
        "-vn", "-map", "0:a:0",
        "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "wav", str(destination),
    ]
    subprocess.run(command, capture_output=True, check=True)
    return command


# =============================================================================
# Silero inference (torch imported lazily; model comes from the gate)
# =============================================================================

def silero_window_probs(silero_model: Any, pcm: bytes) -> List[float]:
    """Sequential 512-sample windows -> speech probability per window.
    The final partial window is zero-padded for inference only; the
    stored audio buffer is never modified."""
    import torch

    audio = (
        torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32)
        / 32768.0
    )
    if hasattr(silero_model, "reset_states"):
        silero_model.reset_states()
    probs: List[float] = []
    with torch.no_grad():
        for offset in range(0, audio.numel(), SILERO_WINDOW_SAMPLES):
            chunk = audio[offset:offset + SILERO_WINDOW_SAMPLES]
            if chunk.numel() < SILERO_WINDOW_SAMPLES:
                chunk = torch.cat(
                    [chunk, torch.zeros(SILERO_WINDOW_SAMPLES - chunk.numel())]
                )
            probs.append(float(silero_model(chunk, SAMPLE_RATE).item()))
    return probs


# =============================================================================
# Layer 0 entrypoint
# =============================================================================

def preprocess_batch(
    manifest: SessionManifest,
    video_paths: Sequence[Path | str],
    work_dir: Path | str,
    silero_model: Any,
    extract_workers: int = EXTRACT_WORKERS,
) -> BatchAudio:
    """Run Layer 0 over one batch. Extraction is parallel (CPU stage);
    Silero and ALL manifest writes happen sequentially in canonical file
    order, so the manifest is byte-equivalent in payload content
    regardless of extraction scheduling."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Canonical file order: lexicographic full-path sort (manifest-logged).
    sources = sorted(Path(p) for p in video_paths)
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        manifest.append(
            Operation.BLOCKING_HALT,
            {"reason": "layer0_missing_inputs", "files": missing},
        )
        raise Layer0Error(f"missing input files: {missing}")

    source_hashes = [sha256_of_file(p) for p in sources]
    manifest.append(
        Operation.BATCH_INIT,
        {
            "layer": 0,
            "canonical_order": "lexicographic_full_path",
            "ffmpeg_version": ffmpeg_version(),
            "sample_rate": SAMPLE_RATE,
            "extract_workers": extract_workers,
            "silero_params": {
                "threshold": SILERO_THRESHOLD,
                "window_samples": SILERO_WINDOW_SAMPLES,
                "min_speech_ms": SILERO_MIN_SPEECH_MS,
                "min_silence_ms": SILERO_MIN_SILENCE_MS,
                "speech_pad_ms": SILERO_SPEECH_PAD_MS,
            },
            "files": [
                {"file_index": i, "source": str(p), "source_sha256": h}
                for i, (p, h) in enumerate(zip(sources, source_hashes))
            ],
        },
    )

    # --- Parallel extraction stage (pure per-file work, no manifest IO) ----
    def _extract_one(index: int) -> FileAudio:
        source = sources[index]
        probe = parse_probe(run_ffprobe(source))
        wav_path = work / f"{index:03d}_{source.stem}.16k.wav"
        extract_audio(source, wav_path)
        num_samples, pcm = read_wav_pcm16(wav_path)
        return FileAudio(
            file_index=index,
            source_path=str(source),
            wav_path=str(wav_path),
            source_sha256=source_hashes[index],
            wav_sha256=sha256_of_file(wav_path),
            num_samples=num_samples,
            duration_ms=ms_from_samples(num_samples),
            audio_start_pts_ms=probe.audio_start_pts_ms,
            audio_start_missing=probe.audio_start_missing,
            vfr_suspected=probe.vfr_suspected,
            pcm=pcm,
        )

    with ThreadPoolExecutor(max_workers=max(1, extract_workers)) as pool:
        files = list(pool.map(_extract_one, range(len(sources))))

    # --- Sequential stage: offsets, Silero, manifest (canonical order) -----
    offset_ms = 0
    for file_audio in files:  # already index-ordered (pool.map preserves order)
        file_audio.file_offset_ms = offset_ms
        offset_ms += file_audio.duration_ms

        data_segments = segments_from_window_probs(
            silero_window_probs(silero_model, file_audio.pcm),
            total_samples=file_audio.num_samples,
        )
        file_audio.silero_segments_local_ms = [
            (file_audio.audio_start_pts_ms + s, file_audio.audio_start_pts_ms + e)
            for s, e in data_segments
        ]

        if file_audio.file_index > 0:
            manifest.append(
                Operation.VIDEO_GAP,
                {
                    "file_index": file_audio.file_index,
                    "file_offset_ms": file_audio.file_offset_ms,
                    "note": "new file boundary within continuous session",
                },
            )
        manifest.append(OP_LAYER0_FILE, layer0_file_payload(file_audio))
    return BatchAudio(files=files)


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB Layer 0 preprocessor (Module 1)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip, no torch, no ffmpeg)")
    mode.add_argument("--run", action="store_true",
                      help="run Layer 0 on a batch (Ubuntu deployment box)")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--operator", type=str, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_stdlib()

    for required in ("videos", "work_dir", "model_store", "manifest"):
        if getattr(args, required) is None:
            parser.error(f"--run requires --{required.replace('_', '-')}")
    with SessionManifest(args.manifest, operator_id=args.operator) as manifest:
        models = environment_gate.run_gate(args.model_store, manifest)
        batch = preprocess_batch(manifest, args.videos, args.work_dir,
                                 models.silero)
    print(f"layer0 complete — {len(batch.files)} files, "
          f"{batch.total_speech_ms()} ms total Silero speech")
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch/GPU)
# =============================================================================

def _selftest_stdlib() -> int:
    import tempfile

    assert "torch" not in sys.modules, "torch imported at module level"

    # 1. sample -> ms integer math
    assert ms_from_samples(16000) == 1000
    assert ms_from_samples(8000) == 500
    assert ms_from_samples(511) == 31
    assert ms_from_samples(0) == 0

    # 2. ffprobe decimal rounding (ROUND_HALF_EVEN)
    assert decimal_seconds_to_ms("0.023220") == 23
    assert decimal_seconds_to_ms("1.5") == 1500
    assert decimal_seconds_to_ms("2.0005") == 2000   # half -> even
    assert decimal_seconds_to_ms("2.0015") == 2002   # half -> even

    # 3. probe parsing + VFR suspicion
    probe = {
        "streams": [
            {"codec_type": "video", "avg_frame_rate": "30000/1001",
             "r_frame_rate": "30/1"},
            {"codec_type": "audio", "start_time": "0.023220",
             "sample_rate": "48000"},
        ]
    }
    info = parse_probe(probe)
    assert info.audio_start_pts_ms == 23 and not info.audio_start_missing
    assert info.vfr_suspected and info.source_sample_rate == 48000
    probe["streams"][0]["r_frame_rate"] = "30000/1001"
    assert not parse_probe(probe).vfr_suspected
    probe["streams"][1]["start_time"] = "N/A"
    info = parse_probe(probe)
    assert info.audio_start_pts_ms == 0 and info.audio_start_missing

    # 4. segment derivation (windows are 32 ms each)
    w, total = SILERO_WINDOW_SAMPLES, 20 * SILERO_WINDOW_SAMPLES
    speech, silence = 0.9, 0.0
    # A: 10 speech windows (320 ms) -> kept, padded 30 ms, clamped at 0
    segs = segments_from_window_probs([speech] * 10 + [silence] * 10, total)
    assert segs == [(0, 350)], segs
    # B: 5 speech windows (160 ms) -> dropped (< 250 ms, pre-padding)
    segs = segments_from_window_probs([speech] * 5 + [silence] * 15, total)
    assert segs == [], segs
    # C: 64 ms gap (< 100 ms) merges two runs; pad clamps inside audio
    probs = [speech] * 8 + [silence] * 2 + [speech] * 8 + [silence] * 2
    segs = segments_from_window_probs(probs, total)
    assert segs == [(0, 606)], segs
    # D: 128 ms gap (>= 100 ms) keeps runs apart; each >= 250 ms survives
    probs = [speech] * 8 + [silence] * 4 + [speech] * 8
    segs = segments_from_window_probs(probs, 20 * w)
    assert segs == [(0, 286), (354, 640)], segs

    # 5. WAV round-trip via stdlib wave
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tone.wav"
        frames = b"\x00\x01" * 16000  # 1 s of constant samples
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(frames)
        num_samples, pcm = read_wav_pcm16(wav_path)
        assert num_samples == 16000 and pcm == frames
        assert ms_from_samples(num_samples) == 1000

    # 6. PTS clock arithmetic (offsets + local/global mapping)
    def _fake(index: int, duration_ms: int, start_pts: int) -> FileAudio:
        return FileAudio(
            file_index=index, source_path=f"v{index}", wav_path="",
            source_sha256="", wav_sha256="",
            num_samples=duration_ms * 16, duration_ms=duration_ms,
            audio_start_pts_ms=start_pts, audio_start_missing=False,
            vfr_suspected=False,
        )

    files = [_fake(0, 300000, 23), _fake(1, 420000, 0)]
    offset = 0
    for file_audio in files:
        file_audio.file_offset_ms = offset
        offset += file_audio.duration_ms
    assert files[1].file_offset_ms == 300000
    assert files[0].to_global_ms(1000) == 1000
    assert files[1].to_global_ms(1000) == 301000
    assert files[0].sample_to_local_pts_ms(16000) == 1023

    # 7. OP_LAYER0_FILE payload passes Rule 6 (validate_time_fields), and
    #    the nested segment shape is actively covered by it — guards
    #    against payload-shape regressions preprocess_batch can't show
    #    under the zero-pip policy (it needs ffmpeg + Silero).
    file_audio = _fake(0, 300000, 23)
    file_audio.silero_segments_local_ms = [(23, 1373), (2500, 4000)]
    payload = layer0_file_payload(file_audio)
    validate_time_fields(payload)  # must not raise
    assert payload["silero_segments"] == [
        {"start_ms": 23, "end_ms": 1373},
        {"start_ms": 2500, "end_ms": 4000},
    ]
    assert payload["silero_total_speech_ms"] == 2850
    payload["silero_segments"][0]["start_ms"] = 23.0  # corrupt one boundary
    try:
        validate_time_fields(payload)
    except ManifestTimeError:
        pass
    else:
        raise AssertionError("Rule 6 missed a float inside silero_segments")

    assert "torch" not in sys.modules, "self-test imported torch"
    print("layer0_preprocessor stdlib self-test OK — "
          "no torch, no GPU, no pip, no ffmpeg")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
