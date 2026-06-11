# SPOVNOB Audio Diarization Module

SPOVNOB is a forensic speaker diarization pipeline. It processes batches of interview-style video recordings and isolates the clean, uncontaminated speech of a single visually-verified target speaker, producing PTS-timestamped, verified-clean audio segments as its final output (downstream behavioral analysis is a deferred future phase). The pipeline is fully deterministic and reproducible (bit-identical outputs for identical inputs), runs entirely offline/air-gapped with checksum-verified vendored models, performs zero audio synthesis or reconstruction (original signal only — contaminated segments are excluded, never repaired), and records every decision, parameter, and discard in an append-only, hash-chained session manifest.

The full architecture specification lives in [`Audio_Diarization.md`](Audio_Diarization.md). The module-by-module build plan is in [`implementation_order.md`](implementation_order.md).

## Implementation status

| Module | File | Status |
|---|---|---|
| 0a — Session manifest (hash-chained audit log) | `session_manifest.py` | ✅ Complete |
| 0b — Environment gate (determinism + model vendoring checks) | `environment_gate.py` | ✅ Complete |
| 1 — Layer 0 preprocessor (PTS-true extraction + VAD segment map) | `layer0_preprocessor.py` | ✅ Complete |
| 2 — Layer 1 enrollment (visual-anchored speaker profile) | `layer1_enrollment/` | ✅ Complete |
| 3 — Layer 2 tracker (calibrated sliding-window target tracking) | `layer2_tracker.py` | ✅ Complete |
| 4 — Layer 3 contamination flagging (overlap exclusion) | `layer3_contamination.py` | ✅ Complete |
| 5 — Pipeline runner (production entrypoint) | `pipeline_runner.py` | ✅ Complete |
| Behavioral analysis | — | ⏸ Deferred — requires separate design phase |

Every module ships a stdlib-only self-test that runs on a plain Python 3.10 with zero installed dependencies:

```bash
python3 session_manifest.py
python3 environment_gate.py --selftest
python3 layer0_preprocessor.py --selftest
python3 -m layer1_enrollment --selftest
python3 layer2_tracker.py --selftest
python3 layer3_contamination.py --selftest
python3 pipeline_runner.py --selftest
```

Full batch run (Ubuntu deployment box, after setup):

```bash
python3 pipeline_runner.py --run --videos <files...> --clicks clicks.json \
    --work-dir <dir> --model-store <store> --manifest <session.jsonl>
```

## Platform

Deployment target is **Ubuntu 22.04 LTS** with an **NVIDIA RTX 6000 Ada (48 GB)**, Python 3.10, CUDA 12.1, PyTorch only. Model weights are never committed — they are staged into a local, SHA-256-pinned model store. See **[`UBUNTU_SETUP_GUIDE.md`](UBUNTU_SETUP_GUIDE.md)** for the complete, sequential setup procedure (system prerequisites, pinned Python environment, model vendoring, and environment-gate verification), and [`requirements.txt`](requirements.txt) for the exact pinned dependency stack.
