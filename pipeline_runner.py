"""
SPOVNOB — Module 5: pipeline_runner.py
=======================================

Layer:      Batch orchestrator — the final module of this phase and the
            single production entrypoint. Imports every earlier module
            (0a/0b/1/2/3/4); nothing imports it.

Purpose:    Chain the complete pipeline over one batch, in order:
            environment gate (determinism + vendoring + resident models)
            -> Layer 0 (PTS-true extraction + Silero map, RAM preload)
            -> Layer 1 (visual-anchored enrollment, sequential)
            -> Layer 2 (calibrated sliding-window tracking, single
            authoritative pass) -> Layer 3 (overlap exclusion + clean
            segments), then write the final verified pipeline summary.
            No behavioral analysis. No WavLM. No HuBERT.

Inputs:     batch video files · operator clicks JSON · work dir ·
            model store · manifest path · operator id
Outputs:    - <work_dir>/pipeline_output.json — the batch summary
              (canonical JSON, SHA-256 recorded in the manifest):
              per-file facts, enrollment refs, Layer 2 thresholds +
              output hash, Layer 3 totals + output hash, and the full
              clean-segment listing (the phase's final deliverable).
            - a fully hash-chain-VERIFIED session manifest: after the
              manifest closes, the runner re-walks the entire chain
              (Module 0a verify_chain) and refuses success if any entry
              fails — the run does not count unless its audit trail
              re-verifies from disk.

Implements (Audio_Diarization.md):
            - the Full Pipeline Flow Diagram end to end (minus the
              deferred behavioral stage)
            - System Environment: Resident Model Policy (models loaded
              once by the gate, passed down, never reloaded) and the
              sequential-Layer-1 rule
            - the audit-trail closure: "Verification: auditor can
              reproduce the exact run and verify the SHA-256 output
              hash matches".

Determinism notes:
            - Wall-clock timings are deliberately EXCLUDED from all
              manifest payloads and the summary document (payloads must
              be bit-reproducible); per-layer durations are printed to
              the console only.
            - The summary document is canonical JSON; its SHA-256 is
              recorded in the manifest before the manifest closes.

CUDA determinism dependencies: everything via environment_gate.run_gate
(the runner itself performs no inference).

Self-test:  python3 pipeline_runner.py --selftest   (stdlib only: the
            summary builder and manifest finalization/verification run
            against fabricated layer results — zero pip installs, no
            torch, no GPU).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from layer0_preprocessor import BatchAudio, preprocess_batch
from layer1_enrollment import EnrollmentResult, load_clicks, run_layer1
from layer2_tracker import Layer2Result, run_layer2
from layer3_contamination import Layer3Result, run_layer3
from session_manifest import (
    SessionManifest,
    canonical_json,
    sha256_of_obj,
    validate_time_fields,
)

OP_COMPLETE = "pipeline_complete"
SUMMARY_SCHEMA = "spovnob-pipeline-output-v1"


# =============================================================================
# Pure summary assembly (stdlib-only; self-tested)
# =============================================================================

def build_pipeline_summary(
    batch: BatchAudio,
    enrollment: EnrollmentResult,
    layer2: Layer2Result,
    layer3: Layer3Result,
    manifest_path: Path | str,
) -> Dict[str, Any]:
    """The final batch summary document. Deterministic content only —
    no wall-clock data anywhere in this structure."""
    return {
        "schema": SUMMARY_SCHEMA,
        "manifest_path": str(manifest_path),
        "files": [
            {
                "file_index": f.file_index,
                "source_file": f.source_path,
                "source_sha256": f.source_sha256,
                "duration_ms": f.duration_ms,
                "file_offset_ms": f.file_offset_ms,
                "silero_speech_ms": sum(
                    e - s for s, e in f.silero_segments_local_ms
                ),
            }
            for f in batch.files
        ],
        "enrollment": {
            "e_composite_sha256": enrollment.e_composite_sha256,
            "e_anti_sha256": enrollment.e_anti_sha256,
            "no_anti_profile": enrollment.no_anti_profile,
            "total_verified_ms": enrollment.total_verified_ms,
            "pool_size": len(enrollment.pool),
            "anti_pool_size": len(enrollment.anti_pool),
            "final_quality_state": (
                enrollment.quality_history[-1]["state"]
                if enrollment.quality_history else None
            ),
        },
        "layer2": {
            "output_sha256": layer2.output_sha256,
            "calibration_ref": layer2.calibration.calibration_ref,
            "calibration": layer2.calibration.kind,
            "theta_high": layer2.calibration.theta_high,
            "theta_med": layer2.calibration.theta_med,
            "total_high_ms": sum(t.high_ms for t in layer2.files),
        },
        "layer3": {
            "output_sha256": layer3.output_sha256,
            "total_clean_ms": layer3.total_clean_ms,
            "total_contaminated_ms": layer3.total_contaminated_ms,
            "segment_count": sum(len(f.segments) for f in layer3.files),
            "nan_block_count": sum(len(f.nan_blocks) for f in layer3.files),
        },
        "clean_segments": [
            {
                "file_index": s.file_index,
                "start_local_ms": s.start_local_ms,
                "end_local_ms": s.end_local_ms,
                "start_global_ms": s.start_global_ms,
                "end_global_ms": s.end_global_ms,
                "duration_ms": s.duration_ms,
                "bridged_gap_count": len(s.bridged_gaps),
                "wav_path": s.wav_path,
                "wav_sha256": s.wav_sha256,
            }
            for f in layer3.files
            for s in f.segments
        ],
    }


def finalize_pipeline(
    manifest: SessionManifest,
    work_dir: Path | str,
    batch: BatchAudio,
    enrollment: EnrollmentResult,
    layer2: Layer2Result,
    layer3: Layer3Result,
    manifest_path: Path | str,
) -> Dict[str, Any]:
    """Write the summary document, record its hash in the manifest, and
    return {summary, output_path, output_sha256}. Factored out of
    run_pipeline so the self-test can exercise it with fabricated layer
    results and a real temp manifest."""
    summary = build_pipeline_summary(
        batch, enrollment, layer2, layer3, manifest_path,
    )
    validate_time_fields(summary)
    summary_sha = sha256_of_obj(summary)
    output_path = Path(work_dir) / "pipeline_output.json"
    output_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    manifest.append(OP_COMPLETE, {
        "output_path": str(output_path),
        "output_sha256": summary_sha,
        "enrollment_ref": enrollment.e_composite_sha256,
        "layer2_output_sha256": layer2.output_sha256,
        "layer3_output_sha256": layer3.output_sha256,
        "total_clean_ms": layer3.total_clean_ms,
        "total_contaminated_ms": layer3.total_contaminated_ms,
        "segment_count": sum(len(f.segments) for f in layer3.files),
    })
    return {
        "summary": summary,
        "output_path": str(output_path),
        "output_sha256": summary_sha,
    }


# =============================================================================
# The pipeline
# =============================================================================

def run_pipeline(
    videos: Sequence[Path | str],
    clicks_path: Path | str,
    work_dir: Path | str,
    model_store: Path | str,
    manifest_path: Path | str,
    operator: Optional[str] = None,
) -> Dict[str, Any]:
    """Full batch run. Returns the finalization dict plus the count of
    manifest entries that re-verified from disk after close."""
    clicks = load_clicks(clicks_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stage_clock: List[tuple] = []

    def _mark(stage: str, started: float) -> None:
        # Console-only diagnostics: wall time never enters any payload.
        stage_clock.append((stage, time.monotonic() - started))

    with SessionManifest(manifest_path, operator_id=operator) as manifest:
        started = time.monotonic()
        models = environment_gate.run_gate(model_store, manifest)
        _mark("environment_gate", started)

        started = time.monotonic()
        batch = preprocess_batch(manifest, videos, work, models.silero)
        _mark("layer0_preprocess", started)

        started = time.monotonic()
        enrollment = run_layer1(manifest, batch, models, clicks, work)
        _mark("layer1_enrollment", started)

        started = time.monotonic()
        layer2 = run_layer2(manifest, batch, models, enrollment, work)
        _mark("layer2_tracking", started)

        started = time.monotonic()
        layer3 = run_layer3(manifest, batch, models, layer2, work)
        _mark("layer3_contamination", started)

        result = finalize_pipeline(
            manifest, work, batch, enrollment, layer2, layer3, manifest_path,
        )

    # The run does not count unless its audit trail re-verifies from disk.
    entries = SessionManifest.verify_chain(manifest_path)
    result["manifest_entries_verified"] = len(entries)
    result["stage_seconds"] = {name: round(secs, 2) for name, secs in stage_clock}
    return result


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB pipeline runner (Module 5) — full batch run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip, no torch, no GPU)")
    mode.add_argument("--run", action="store_true",
                      help="run the complete pipeline (Ubuntu deployment box)")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--clicks", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--operator", type=str, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_stdlib()

    for required in ("videos", "clicks", "work_dir", "model_store", "manifest"):
        if getattr(args, required) is None:
            parser.error(f"--run requires --{required.replace('_', '-')}")

    result = run_pipeline(
        args.videos, args.clicks, args.work_dir,
        args.model_store, args.manifest, args.operator,
    )
    summary = result["summary"]
    print("pipeline complete —")
    print(f"  clean output:      {summary['layer3']['total_clean_ms']} ms in "
          f"{summary['layer3']['segment_count']} segments")
    print(f"  excluded (NaN):    {summary['layer3']['total_contaminated_ms']} ms")
    print(f"  summary document:  {result['output_path']} "
          f"(sha256 {result['output_sha256'][:12]}…)")
    print(f"  manifest verified: {result['manifest_entries_verified']} entries")
    for stage, seconds in result["stage_seconds"].items():
        print(f"  {stage:<22}{seconds:>8.2f} s")
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch/GPU)
# =============================================================================

def _selftest_stdlib() -> int:
    import tempfile

    from layer0_preprocessor import FileAudio
    from layer1_enrollment.enrollment import PoolEntry
    from layer2_tracker import Calibration, FileTrack
    from layer3_contamination import CleanSegment, FileContamination

    assert "torch" not in sys.modules, "torch imported at module level"

    duration = 20000
    batch = BatchAudio(files=[
        FileAudio(
            file_index=i, source_path=f"video_{i:02d}.mp4", wav_path="",
            source_sha256=f"{i}" * 64, wav_sha256="", num_samples=duration * 16,
            duration_ms=duration, audio_start_pts_ms=0,
            audio_start_missing=False, vfr_suspected=False,
            file_offset_ms=i * duration,
            silero_segments_local_ms=[(0, duration)],
        )
        for i in range(2)
    ])
    pool = [PoolEntry(vector=[1.0, 0.0], duration_ms=5000, kind="seed",
                      file_index=0, t_start_local_ms=0, t_stop_local_ms=5000)]
    enrollment = EnrollmentResult(
        f_target=[1.0], e_seed=[1.0, 0.0], e_composite=[1.0, 0.0],
        e_composite_sha256="e" * 64, e_anti=[0.0, 1.0],
        e_anti_sha256="a" * 64, no_anti_profile=False, pool=pool,
        anti_pool=[], total_verified_ms=50000,
        quality_history=[{"state": "STRONG"}],
    )
    calibration = Calibration(
        theta_high=0.58, theta_med=0.43, kind="DERIVED", record={},
        calibration_ref="c" * 64, overlap_warning=False,
    )
    layer2 = Layer2Result(
        calibration=calibration,
        files=[FileTrack(
            file_index=i, source_file=f"video_{i:02d}.mp4", high_runs=[],
            tier_counts={}, high_ms=5000, silero_ms=duration, ratio=0.25,
            ratio_level="LOW_ADVISORY", unattributed_speech_ms=15000,
            high_scores=[], blocks=[],
        ) for i in range(2)],
        no_anti_profile=False, authoritative=True,
        output_path="", output_sha256="2" * 64,
    )
    segment = CleanSegment(
        file_index=0, start_local_ms=7000, end_local_ms=11000,
        start_global_ms=7000, end_global_ms=11000, duration_ms=4000,
        block_count=4, bridged_gaps=[(9000, 9300)],
        wav_path="clean_000.wav", wav_sha256="d" * 64,
    )
    layer3 = Layer3Result(
        files=[FileContamination(
            file_index=0, source_file="video_00.mp4", segments=[segment],
            nan_blocks=[{"designation": "NaN", "start_local_ms": 11000,
                         "end_local_ms": 12000, "duration_ms": 1000}],
            overlap_regions=[(11200, 11400)], clean_ms=4000,
            contaminated_ms=1000, bridged_gap_ms=300,
        )],
        output_path="", output_sha256="3" * 64,
        total_clean_ms=4000, total_contaminated_ms=1000,
    )

    # 1. Summary builder: deterministic, integer-ms-valid, serializable.
    summary_one = build_pipeline_summary(batch, enrollment, layer2, layer3,
                                         "/tmp/m.jsonl")
    summary_two = build_pipeline_summary(batch, enrollment, layer2, layer3,
                                         "/tmp/m.jsonl")
    validate_time_fields(summary_one)
    assert sha256_of_obj(summary_one) == sha256_of_obj(summary_two)
    assert summary_one["schema"] == SUMMARY_SCHEMA
    assert summary_one["files"][1]["file_offset_ms"] == duration
    assert summary_one["enrollment"]["final_quality_state"] == "STRONG"
    assert summary_one["layer2"]["total_high_ms"] == 10000
    assert summary_one["layer3"]["segment_count"] == 1
    assert summary_one["clean_segments"][0]["bridged_gap_count"] == 1
    canonical_json(summary_one)

    # 2. Finalization into a real manifest: entry recorded, file written,
    # chain re-verifies, summary hash matches the manifest record.
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        manifest_path = tmp / "m.jsonl"
        with SessionManifest(manifest_path, operator_id="selftest") as manifest:
            result = finalize_pipeline(
                manifest, tmp, batch, enrollment, layer2, layer3,
                manifest_path,
            )
        entries = SessionManifest.verify_chain(manifest_path)
        complete = [e for e in entries if e["operation"] == OP_COMPLETE]
        assert len(complete) == 1
        assert complete[0]["payload"]["output_sha256"] == result["output_sha256"]
        assert complete[0]["payload"]["total_clean_ms"] == 4000
        written = (tmp / "pipeline_output.json").read_text(encoding="utf-8")
        assert sha256_of_obj(result["summary"]) == result["output_sha256"]
        assert written == canonical_json(result["summary"]) + "\n"

    for forbidden in ("torch", "cv2", "numpy"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"
    print("pipeline_runner stdlib self-test OK — summary + finalization "
          "exercised against fabricated layer results; no torch, no GPU, no pip")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
