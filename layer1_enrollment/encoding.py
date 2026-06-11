"""
SPOVNOB — Module 2 (Layer 1): encoding.py
==========================================

ECAPA-TDNN encoding plus the pure enrollment arithmetic. The pure half
(cosine, L2 normalize, duration-weighted mean, chunk planning, PCM
slicing math) is stdlib-only and self-tested; torch is imported lazily
inside the two functions that actually touch the model.

Implements: Audio_Diarization.md — "ECAPA-TDNN (Speaker Encoder) & The
Zero-Training Mandate" (duration-weighted mean pooling), "Single-Pass
ECAPA Encoding" (one forward pass, 60s sanity cap with 2s-overlap
chunking above it).

Determinism notes:
  - math.fsum is used for every reduction: it returns the correctly
    rounded exact sum, so results are independent of summation order.
  - Enrollment windows are encoded one per forward pass (batch of 1):
    they are variable-length, and the fixed 256-window batch constant
    applies to Layer 2's fixed-size sliding windows, not here.
  - d-vectors are L2-normalized before any pooling or comparison.

CUDA determinism dependencies: inherits the four constants from
environment_gate for the ECAPA forward passes; float32 throughout.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import math
from typing import Any, List, Sequence, Tuple

from .params import EnrollmentParams

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


# =============================================================================
# Pure arithmetic (stdlib-only; covered by the self-test)
# =============================================================================

def l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(y * y for y in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def duration_weighted_mean(
    vectors: Sequence[Sequence[float]], durations_ms: Sequence[int]
) -> List[float]:
    """Enrollment arithmetic: sum(d_i * duration_i) / sum(duration_i),
    computed per dimension with fsum. Inputs are expected to be
    L2-normalized d-vectors."""
    if not vectors or len(vectors) != len(durations_ms):
        raise ValueError("vectors/durations mismatch or empty pool")
    total = math.fsum(float(d) for d in durations_ms)
    if total <= 0:
        raise ValueError("total duration must be positive")
    dims = len(vectors[0])
    return [
        math.fsum(v[d] * w for v, w in zip(vectors, durations_ms)) / total
        for d in range(dims)
    ]


def plan_chunks(
    duration_ms: int, max_ms: int, overlap_ms: int
) -> List[Tuple[int, int]]:
    """Single-Pass rule: one chunk up to the 60s sanity cap; above it,
    split at max_ms boundaries with overlap_ms of overlap. Offsets are
    relative to the window start."""
    if duration_ms <= max_ms:
        return [(0, duration_ms)]
    chunks: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_ms, duration_ms)
        chunks.append((start, end))
        if end == duration_ms:
            return chunks
        start = end - overlap_ms


def pcm_slice(
    pcm: bytes,
    audio_start_pts_ms: int,
    num_samples: int,
    start_local_ms: int,
    stop_local_ms: int,
) -> bytes:
    """Slice raw int16 PCM by LOCAL PTS milliseconds. The audio stream's
    start PTS is subtracted to reach data-relative samples; bounds are
    clamped to the buffer. 1 ms == 16 samples at 16kHz exactly."""
    rel_start_ms = start_local_ms - audio_start_pts_ms
    rel_stop_ms = stop_local_ms - audio_start_pts_ms
    first = min(max(0, rel_start_ms * 16), num_samples)
    last = min(max(0, rel_stop_ms * 16), num_samples)
    return pcm[first * BYTES_PER_SAMPLE: last * BYTES_PER_SAMPLE]


# =============================================================================
# ECAPA inference (torch imported lazily; model comes from the gate)
# =============================================================================

def ecapa_encode_pcm(ecapa_model: Any, pcm: bytes) -> List[float]:
    """One ECAPA forward pass over one PCM16 slice -> L2-normalized
    192-dim d-vector as a plain float list (kept framework-free above
    this boundary)."""
    import torch

    audio = (
        torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32)
        / 32768.0
    ).unsqueeze(0)
    with torch.no_grad():
        embedding = ecapa_model.encode_batch(audio)
    flat = embedding.reshape(-1).to(torch.float32).cpu().tolist()
    return l2_normalize(flat)


def encode_window(
    ecapa_model: Any,
    pcm: bytes,
    audio_start_pts_ms: int,
    num_samples: int,
    start_local_ms: int,
    stop_local_ms: int,
    params: EnrollmentParams,
) -> Tuple[List[float], int]:
    """Encode one E_window: single pass below the 60s cap, otherwise
    60s/2s-overlap chunks pooled with duration weighting. Returns
    (L2-normalized d-vector, window duration_ms)."""
    duration_ms = stop_local_ms - start_local_ms
    chunks = plan_chunks(duration_ms, params.encode_max_ms, params.encode_overlap_ms)
    vectors: List[List[float]] = []
    durations: List[int] = []
    for chunk_start, chunk_end in chunks:
        chunk_pcm = pcm_slice(
            pcm, audio_start_pts_ms, num_samples,
            start_local_ms + chunk_start, start_local_ms + chunk_end,
        )
        vectors.append(ecapa_encode_pcm(ecapa_model, chunk_pcm))
        durations.append(chunk_end - chunk_start)
    if len(vectors) == 1:
        return vectors[0], duration_ms
    return l2_normalize(duration_weighted_mean(vectors, durations)), duration_ms
