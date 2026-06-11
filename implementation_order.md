# SPOVNOB — Implementation Order

**Revision 1 — 2026-06-11 · Target: Ubuntu 22.04 LTS · Python 3.10 · CUDA 12.1 · PyTorch only**

---

## Ordering Conflict — RESOLVED (FLAG 5, ruling 2026-06-11)

The directive placed `session_manifest.py` last (Module 5), but Rule 3 says *no module may import from a later module* and Rule 7 says *the manifest must be written before any destructive or irreversible operation*. Every module from `environment_gate` onward writes to the manifest (vendoring checksums, threshold calibration, operator clicks, discards). These three requirements were mutually unsatisfiable with the manifest last.

**Approved resolution:** `session_manifest.py` is implemented first, as Module 0a. It is pure stdlib (`json`, `hashlib`, `os`, `fcntl`) — no torch, no models — so it sits naturally at the bottom of the dependency graph and is also the easiest module to review and unit-test first.

## Module Order & Dependency Graph

```
session_manifest  ◄──  environment_gate  ◄──  layer0  ◄──  layer1  ◄──  layer2  ◄──  layer3
(0a, stdlib only)      (0b)                   (1)          (2)          (3)          (4)
```

Arrows point from importer to imported. Strictly acyclic; no module imports a later one.

| # | Module | Imports | Implements (Audio_Diarization.md sections) |
|---|---|---|---|
| 0a | `session_manifest.py` | stdlib only | Operator Threshold Manifest Format · Persistence, audit, and manifest rules for `E_anti` and `E_window` · Canonical Manifest Merge Rule (System Environment) · SHA-256 output-hash rules (Layer 2 Step 9) · append-only JSON-L, write-before-destructive-op |
| 0b | `environment_gate.py` | 0a | CUDA Determinism Constants · Model Vendoring Mandate (6-model SHA-256 startup gate, blocking halt) · Resident Model Policy (single loader, no unload) · 10-second determinism verification checksum · FLAG 1 guard (forbid pyannote's speechbrain wrapper) · FLAG 2 guard (assert CUDAExecutionProvider available) |
| 1 | `layer0_preprocessor.py` | 0a, 0b | Layer 0 — Global PTS Clock Initialization (`file_offset_ms`, `global_ms`) · FFmpeg Audio Strip (PTS-true, 16kHz mono) · Silero VAD segment map (non-destructive) · RAM preload of batch audio · parallel CPU stage (Parallel Execution Model) |
| 2 | `layer1_enrollment.py` | 0a, 0b, 1 | Layer 1 — Clicks 1/2 + 9 Guardrails · `E_seed` capture · `E_window` (MAR/hysteresis/plosive buffer/yaw filter) · Single-Pass ECAPA Encoding (60s cap) · Track B/C `E_anti` + M-Trap · Triple Validation Gate · quality states + cumulative pool · variance gates · **strictly sequential in canonical file order** |
| 3 | `layer2_tracker.py` | 0a, 0b, 1, 2 | Layer 2 — `E_anti` sanity check · Deterministic Threshold Calibration · 5s/1s sliding window (fixed batch 256) · Silero skip rule · median pooling · tiering + margin rule · Edge-Trim Boundary Refinement (2s/250ms, trim-only) · activity ratio · drift detection · single authoritative pass (preview optional) · parallel fan-out across files |
| 4 | `layer3_contamination.py` | 0a, 0b, 3 | Layer 3 — PyAnnote OVD on HIGH blocks and gaps · NaN-Only Exclusion Policy (no separation models, ever) · temporal smoothing (<400ms, clean gaps only) · CLEAN segment output for HuBERT |

## Cross-Module Rules (binding)

1. **Import discipline:** every module's first import is `environment_gate` (except 0a/0b themselves). `environment_gate` sets `CUBLAS_WORKSPACE_CONFIG` via `os.environ` **before** importing torch, then applies the three torch determinism flags — this ordering is mandatory because the env var must precede CUDA context creation.
2. **No `torch.cuda.empty_cache()` anywhere. No model load/unload between files.** All six models are loaded once by `environment_gate`'s loader and stay resident.
3. **Timestamps:** integers, milliseconds, PTS-derived. No floats for time, no frame indices, in any function signature or persisted record.
4. **Manifest-first:** any function performing a destructive/irreversible step receives the manifest handle and appends its entry before acting.
5. **Framework:** PyTorch exclusively; HuggingFace loads force the PyTorch backend; ONNXRuntime appears only inside the InsightFace wrapper in `layer1_enrollment.py`.
6. **No macOS assumptions:** POSIX paths via `pathlib`, `ffmpeg`/`ffprobe` resolved from PATH or a manifest-pinned absolute path, no Homebrew/Metal/`/opt/homebrew` references.
7. **Review gate:** each module is reviewed and approved before the next is started (Rule 1 of the directive). `layer1_enrollment.py` is the largest; if it needs splitting, it becomes a package (`layer1_enrollment/`) whose submodules still obey the graph above — decided at its review, not unilaterally.
8. **Self-test policy (standing directive, 2026-06-12):** every module ships a stdlib-only self-test runnable on plain Python 3.10 with zero pip installs — no torch, no GPU. Real execution happens only on the Ubuntu workbench via `environment_gate.py --run`.
9. **Setup guide maintenance (standing directive, 2026-06-12):** `UBUNTU_SETUP_GUIDE.md` is updated in the same change that introduces any new dependency, model, or system requirement.

## Per-Module Docstring Contract

Every module opens with a docstring stating: layer number and name · inputs and outputs · the exact `Audio_Diarization.md` sections implemented (as listed above) · CUDA determinism dependencies (which of the four constants it relies on, plus fixed batch constants if any).

---

**Status: APPROVED 2026-06-11 (all five flags ruled on; opencv switched to headless; Silero pinned by commit hash). Implementation in progress — Module 0a `session_manifest.py` first; each module is reviewed before the next begins.**
