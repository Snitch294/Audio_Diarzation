"""
SPOVNOB — Module 0b: environment_gate.py
=========================================

Layer:      Cross-layer startup gate (Module 0b). Imports only Module 0a
            (session_manifest). Every SPOVNOB entrypoint must import this
            module FIRST and call ``run_gate()`` before touching any model.

Purpose:    Prove the runtime environment is the one the architecture
            demands — deterministic, air-gapped, correctly pinned, with
            verified model weights — and load the five resident models
            exactly once. Refuses to start (blocking halt) otherwise.

Inputs:     - model store directory (vendored weights + expected_hashes.json)
            - session manifest path (Module 0a)
Outputs:    - manifest entries: determinism_check, model_checksum,
              blocking_halt (on any failure)
            - a ``ResidentModels`` registry holding all five models,
              loaded once and held for the entire batch (no unloading).

Implements (Audio_Diarization.md — System Environment & Execution Model):
            - "CUDA Determinism Constants (architectural, non-negotiable)"
            - "Model Vendoring Mandate (air-gap startup gate)"
            - "Resident Model Policy" (single loader, fixed batch constants)
            - 10-second startup determinism verification checksum
            - requirements.txt FLAG 1 guard (pyannote's SpeechBrain embedding
              wrapper is forbidden; enforced via exact version pins + policy
              constant) and FLAG 2 guard (ONNXRuntime CUDAExecutionProvider
              must be live — no silent CPU fallback).

CUDA determinism dependencies:
            This module is the SOURCE of all four constants:
              1. CUBLAS_WORKSPACE_CONFIG=":4096:8"  (set at import time,
                 before any CUDA context can exist)
              2. torch.use_deterministic_algorithms(True)
              3. torch.backends.cudnn.deterministic = True
              4. torch.backends.cudnn.benchmark = False
            plus float32-only policy, fixed torch thread count, and the
            fixed inference batch constants (ECAPA=256 windows, visual=32).

Import contract (Implementation Rule 4):
            Importing this module sets the process environment variables
            immediately and imports NOTHING heavy. ``import torch`` happens
            only inside ``run_gate()`` / loader functions, after the
            environment is fixed — this is what makes Rule 4 satisfiable
            and keeps the module's self-test runnable on a bare Python
            3.10 with zero pip installs (standing test policy).

Platform:   Ubuntu 22.04 target. ``run_gate()`` halts on any other OS.
            The stdlib self-test (``--selftest``) runs anywhere.
"""

from __future__ import annotations

# --- Step 0: fix the process environment at import time, before anything ----
# CUBLAS_WORKSPACE_CONFIG must exist before the CUDA context is created;
# the offline switches must exist before any HuggingFace import.
import os

DETERMINISM_ENV: dict = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",   # CUDA determinism constant #4
    "HF_HUB_OFFLINE": "1",                  # Model Vendoring Mandate
    "TRANSFORMERS_OFFLINE": "1",            # Model Vendoring Mandate
}
for _key, _value in DETERMINISM_ENV.items():
    os.environ[_key] = _value

# --- stdlib-only imports (torch is deliberately NOT imported here) -----------
import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from session_manifest import (
    Operation,
    SessionManifest,
    canonical_json,
    sha256_hex,
    sha256_of_file,
)

# --- Architectural constants (manifest-logged by run_gate) -------------------
ECAPA_BATCH_WINDOWS = 256       # fixed — FP reduction order (System Environment)
VISUAL_BATCH_FRAMES = 32        # fixed — FP reduction order (System Environment)
TORCH_NUM_THREADS = 8           # fixed intra-op threads — CPU reduction order
GLOBAL_SEED = 20260611          # defense-in-depth; pipeline is inference-only
INSIGHTFACE_DET_SIZE = (640, 640)
PYANNOTE_OVD_HYPERPARAMS = {"min_duration_on": 0.0, "min_duration_off": 0.0}
EXPECTED_CUDA_VERSION = "12.1"
EXPECTED_PYTHON_PREFIX = "3.10."
HASHES_FILENAME = "expected_hashes.json"
HASHES_SCHEMA = "spovnob-model-hashes-v1"

# FLAG 1 policy constant: this pyannote code path imports the removed
# `speechbrain.pretrained` module and is FORBIDDEN in SPOVNOB. Layer 3 uses
# OverlappedSpeechDetection only. (pyannote issues #1661/#1677.)
FORBIDDEN_IMPORTS = (
    "pyannote.audio.pipelines.speaker_verification",
)

# Pinned versions — must mirror requirements.txt exactly. The gate verifies
# the installed environment against these pins and halts on any mismatch.
PINNED_VERSIONS: Dict[str, str] = {
    "torch": "2.1.2+cu121",
    "torchaudio": "2.1.2+cu121",
    "torchvision": "0.16.2+cu121",
    "numpy": "1.26.4",
    "speechbrain": "1.0.0",
    "pyannote.audio": "3.1.1",
    "huggingface-hub": "0.21.3",
    "ultralytics": "8.1.47",
    "insightface": "0.7.3",
    "onnxruntime-gpu": "1.17.1",
    "onnx": "1.15.0",
    "opencv-python-headless": "4.9.0.80",
    "ffmpeg-python": "0.2.0",
    "soundfile": "0.12.1",
    "librosa": "0.10.1",
    "scipy": "1.12.0",
    "numba": "0.59.1",
    "llvmlite": "0.42.0",
    "lightning": "2.1.4",
    "torchmetrics": "1.3.1",
    "scikit-learn": "1.4.1.post1",
    "scikit-image": "0.22.0",
    "tqdm": "4.66.2",
}

# The five vendored models (Model Vendoring Mandate). Directory names are the
# canonical model-store layout documented in UBUNTU_SETUP_GUIDE.md.
# HuBERT was removed 2026-06-12: behavioral analysis is deferred to a
# separate design phase. WavLM was never in code (removed at architecture
# Rev 2).
REQUIRED_MODEL_DIRS: Dict[str, str] = {
    "silero_vad": "silero-vad",                      # commit 915dd3d639b8...
    "ecapa_tdnn": "speechbrain-spkrec-ecapa-voxceleb",
    "yolov8": "yolov8",
    "insightface": "insightface",
    "pyannote_ovd": "pyannote-segmentation-3.0",
}


class EnvironmentGateError(RuntimeError):
    """Raised on any blocking-halt condition. The halt is always recorded in
    the session manifest before this exception propagates."""


@dataclass
class ResidentModels:
    """All five models, loaded once, resident for the entire batch.
    No member of this registry is ever unloaded or replaced mid-batch."""

    device: str
    silero: Any = None              # CPU-resident TorchScript module
    ecapa: Any = None               # speechbrain EncoderClassifier (CUDA)
    yolo: Any = None                # ultralytics YOLO (CUDA)
    insightface: Any = None         # FaceAnalysis via CUDAExecutionProvider
    ovd_pipeline: Any = None        # pyannote OverlappedSpeechDetection (CUDA)
    loaded_names: List[str] = field(default_factory=list)


# =============================================================================
# Halt helper
# =============================================================================

def _halt(manifest: SessionManifest, reason: str, detail: Dict[str, Any]) -> None:
    manifest.append(Operation.BLOCKING_HALT, {"reason": reason, **detail})
    raise EnvironmentGateError(f"BLOCKING HALT — {reason}: {canonical_json(detail)}")


# =============================================================================
# Checks (each is pure / injectable so the stdlib self-test can exercise it)
# =============================================================================

def check_versions(
    manifest: SessionManifest,
    version_of: Optional[Callable[[str], str]] = None,
) -> None:
    """Verify every installed package against PINNED_VERSIONS. ``version_of``
    is injectable for the stdlib self-test; the default uses importlib."""
    if version_of is None:
        from importlib.metadata import PackageNotFoundError, version as _v

        def version_of(name: str) -> str:  # type: ignore[misc]
            try:
                return _v(name)
            except PackageNotFoundError:
                try:
                    return _v(name.replace("-", "_").replace(".", "-"))
                except PackageNotFoundError:
                    return "<not installed>"

    mismatches = {
        name: {"pinned": pinned, "installed": version_of(name)}
        for name, pinned in PINNED_VERSIONS.items()
        if version_of(name) != pinned
    }
    if mismatches:
        _halt(manifest, "version_pin_mismatch", {"mismatches": mismatches})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "version_pins", "result": "PASS",
         "packages_verified": len(PINNED_VERSIONS)},
    )


def _walk_store_files(store: Path) -> List[Path]:
    """Every regular file in the store except the hash registry itself,
    in deterministic sorted order."""
    return sorted(
        p for p in store.rglob("*")
        if p.is_file() and p.name != HASHES_FILENAME
    )


def freeze_model_hashes(store: Path) -> Path:
    """Vendoring step (run ONCE on the staging box): hash every file in the
    model store and write the immutable expected-hash registry."""
    import json

    missing = [n for n, d in REQUIRED_MODEL_DIRS.items()
               if not (store / d).is_dir() or not any((store / d).iterdir())]
    if missing:
        raise EnvironmentGateError(
            f"cannot freeze: model dirs missing or empty: {missing}"
        )
    files = {
        str(p.relative_to(store)): sha256_of_file(p)
        for p in _walk_store_files(store)
    }
    registry = {"schema": HASHES_SCHEMA, "files": files}
    out = store / HASHES_FILENAME
    out.write_text(canonical_json(registry) + "\n", encoding="utf-8")
    os.chmod(out, 0o444)  # read-only: the registry is immutable once frozen
    return out


def verify_model_store(manifest: SessionManifest, store: Path) -> None:
    """Model Vendoring Mandate: every file's SHA-256 must match the frozen
    registry exactly — no missing files, no extra files, no mismatches."""
    import json

    registry_path = store / HASHES_FILENAME
    if not registry_path.is_file():
        _halt(manifest, "hash_registry_missing", {"path": str(registry_path)})
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != HASHES_SCHEMA:
        _halt(manifest, "hash_registry_bad_schema",
              {"found": registry.get("schema")})

    missing_dirs = [n for n, d in REQUIRED_MODEL_DIRS.items()
                    if not (store / d).is_dir()]
    if missing_dirs:
        _halt(manifest, "model_dirs_missing", {"models": missing_dirs})

    expected: Dict[str, str] = registry["files"]
    actual = {
        str(p.relative_to(store)): sha256_of_file(p)
        for p in _walk_store_files(store)
    }
    problems = {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "mismatched": sorted(
            f for f in set(expected) & set(actual) if expected[f] != actual[f]
        ),
    }
    if any(problems.values()):
        _halt(manifest, "model_checksum_failure", problems)

    for rel_path in sorted(expected):
        manifest.append(
            Operation.MODEL_CHECKSUM,
            {"file": rel_path, "sha256": expected[rel_path], "result": "PASS"},
        )
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "model_store", "result": "PASS",
         "files_verified": len(expected), "store": str(store)},
    )


def check_runtime_platform(manifest: SessionManifest) -> None:
    if platform.system() != "Linux":
        _halt(manifest, "wrong_platform",
              {"expected": "Linux (Ubuntu 22.04)", "found": platform.system()})
    if not platform.python_version().startswith(EXPECTED_PYTHON_PREFIX):
        _halt(manifest, "wrong_python",
              {"expected": EXPECTED_PYTHON_PREFIX + "x",
               "found": platform.python_version()})
    for key, value in DETERMINISM_ENV.items():
        if os.environ.get(key) != value:
            _halt(manifest, "environment_variable_drift",
                  {"variable": key, "expected": value,
                   "found": os.environ.get(key)})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "platform", "result": "PASS",
         "os": platform.platform(), "python": platform.python_version()},
    )


def check_ffmpeg(
    manifest: SessionManifest,
    probe: Optional[Callable[[str], str]] = None,
) -> None:
    """Layers 0 and 1 shell out to BOTH ffmpeg and ffprobe (extraction,
    stream probing, frame PTS). A missing binary would otherwise surface
    mid-batch as an unrecorded FileNotFoundError, so presence is gated
    here. ``probe`` is injectable for the stdlib self-test; the default
    shells out and returns the first version line ("" = unusable)."""
    if probe is None:
        import subprocess

        def probe(binary: str) -> str:  # type: ignore[misc]
            try:
                out = subprocess.run(
                    [binary, "-version"], capture_output=True, text=True,
                )
            except OSError:
                return ""
            if out.returncode != 0 or not out.stdout:
                return ""
            return out.stdout.splitlines()[0].strip()

    versions = {binary: probe(binary) for binary in ("ffmpeg", "ffprobe")}
    missing = sorted(name for name, line in versions.items() if not line)
    if missing:
        _halt(manifest, "ffmpeg_missing",
              {"missing": missing,
               "hint": "sudo apt install ffmpeg "
                       "(UBUNTU_SETUP_GUIDE.md, system prerequisites)"})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "ffmpeg_binaries", "result": "PASS", "versions": versions},
    )


def enforce_torch_determinism(manifest: SessionManifest) -> None:
    """Apply CUDA determinism constants #1-#3 (the env var, #4, was set at
    import time) and verify the CUDA runtime matches the target."""
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.manual_seed(GLOBAL_SEED)

    if torch.version.cuda != EXPECTED_CUDA_VERSION:
        _halt(manifest, "wrong_cuda_version",
              {"expected": EXPECTED_CUDA_VERSION, "found": torch.version.cuda})
    if not torch.cuda.is_available():
        _halt(manifest, "cuda_unavailable", {})

    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "torch_determinism", "result": "PASS",
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "device": torch.cuda.get_device_name(0),
         "cudnn_deterministic": True, "cudnn_benchmark": False,
         "deterministic_algorithms": True,
         "torch_num_threads": TORCH_NUM_THREADS,
         "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
         "ecapa_batch_windows": ECAPA_BATCH_WINDOWS,
         "visual_batch_frames": VISUAL_BATCH_FRAMES},
    )


def check_onnxruntime_cuda(manifest: SessionManifest) -> None:
    """FLAG 2 guard: the CUDAExecutionProvider must actually be available.
    A CUDA-11.8 wheel on this CUDA-12.1 box would silently fall back to CPU;
    silent fallback is forbidden."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        _halt(manifest, "onnxruntime_cuda_provider_missing",
              {"available_providers": providers,
               "hint": "onnxruntime-gpu wheel must come from the CUDA-12 feed "
                       "(see requirements.txt FLAG 2)"})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "onnxruntime_cuda", "result": "PASS",
         "onnxruntime": ort.__version__, "providers": providers},
    )


def check_forbidden_imports(manifest: SessionManifest) -> None:
    """FLAG 1 guard: no SPOVNOB module may import the pyannote
    SpeechBrain-embedding wrapper directly (broken under speechbrain 1.x;
    unused by policy). Scope note: this check runs BEFORE the resident
    loader; pyannote's own package __init__ later pulls the module into
    sys.modules transitively (guarded inside pyannote by its optional-
    backend try/except, never instantiated by SPOVNOB), so post-load
    presence is expected and is not a violation. The check enforces the
    realistic violation vector: a direct import by SPOVNOB code."""
    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        _halt(manifest, "forbidden_module_imported", {"modules": loaded})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "forbidden_imports", "result": "PASS",
         "forbidden": list(FORBIDDEN_IMPORTS)},
    )


def gpu_determinism_selftest(manifest: SessionManifest) -> str:
    """The ~10-second startup verification checksum: a fixed, seeded float32
    workload exercising cuBLAS (matmul) and cuDNN (conv2d) runs twice from
    scratch; both passes must produce bit-identical SHA-256. The hash is
    recorded so auditors can compare across machines and dates."""
    import torch

    def _one_pass() -> str:
        generator = torch.Generator(device="cuda").manual_seed(GLOBAL_SEED)
        x = torch.randn(2048, 2048, generator=generator,
                        device="cuda", dtype=torch.float32)
        w = torch.randn(2048, 2048, generator=generator,
                        device="cuda", dtype=torch.float32)
        kernel = torch.randn(8, 1, 7, 7, generator=generator,
                             device="cuda", dtype=torch.float32)
        for _ in range(200):
            x = torch.tanh(x @ w) * 0.5
        feature_map = torch.nn.functional.conv2d(
            x.view(1, 1, 2048, 2048), kernel, padding=3
        )
        torch.cuda.synchronize()
        payload = (x.flatten()[:65536].cpu().numpy().tobytes()
                   + feature_map.flatten()[:65536].cpu().numpy().tobytes())
        return sha256_hex(payload)

    started = time.monotonic()
    first, second = _one_pass(), _one_pass()
    elapsed_ms = int((time.monotonic() - started) * 1000)
    if first != second:
        _halt(manifest, "gpu_determinism_failure",
              {"pass1_sha256": first, "pass2_sha256": second,
               "elapsed_ms": elapsed_ms})
    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "gpu_workload_checksum", "result": "PASS",
         "workload_sha256": first, "elapsed_ms": elapsed_ms,
         "seed": GLOBAL_SEED},
    )
    return first


# =============================================================================
# Resident model loader (Resident Model Policy — load once, never unload)
# =============================================================================

def load_resident_models(manifest: SessionManifest, store: Path) -> ResidentModels:
    import torch
    from insightface.app import FaceAnalysis
    from pyannote.audio import Model as PyannoteModel
    from pyannote.audio.pipelines import OverlappedSpeechDetection
    from speechbrain.inference.speaker import EncoderClassifier
    from ultralytics import YOLO

    device = "cuda"
    models = ResidentModels(device=device)

    # 1. Silero VAD — CPU-resident TorchScript (Layer 0). Loaded directly from
    #    the vendored snapshot (commit-pinned); torch.hub is bypassed.
    models.silero = torch.jit.load(
        str(store / "silero-vad" / "files" / "silero_vad.jit"),
        map_location="cpu",
    ).eval()
    models.loaded_names.append("silero_vad")

    # 2. ECAPA-TDNN C=1024 (Layers 1 & 2) — speechbrain, local source only.
    ecapa_dir = str(store / "speechbrain-spkrec-ecapa-voxceleb")
    models.ecapa = EncoderClassifier.from_hparams(
        source=ecapa_dir, savedir=ecapa_dir, run_opts={"device": device}
    )
    models.loaded_names.append("ecapa_tdnn")

    # 3. YOLOv8 person detection (Layer 1).
    models.yolo = YOLO(str(store / "yolov8" / "yolov8m.pt"))
    models.loaded_names.append("yolov8")

    # 4. InsightFace (Layer 1) — the sanctioned ONNXRuntime exception (FLAG 4).
    #    CUDAExecutionProvider only: CPU fallback is forbidden (FLAG 2 guard
    #    has already proven the provider is live).
    models.insightface = FaceAnalysis(
        name="buffalo_l",
        root=str(store / "insightface"),
        providers=["CUDAExecutionProvider"],
    )
    models.insightface.prepare(ctx_id=0, det_size=INSIGHTFACE_DET_SIZE)
    models.loaded_names.append("insightface")

    # 5. PyAnnote Overlapped Speech Detection (Layer 3) — segmentation-3.0,
    #    pinned hyperparameters, never the speaker-verification wrapper.
    segmentation = PyannoteModel.from_pretrained(
        str(store / "pyannote-segmentation-3.0" / "pytorch_model.bin")
    )
    models.ovd_pipeline = OverlappedSpeechDetection(segmentation=segmentation)
    models.ovd_pipeline.instantiate(dict(PYANNOTE_OVD_HYPERPARAMS))
    models.ovd_pipeline.to(torch.device(device))
    models.loaded_names.append("pyannote_ovd")

    manifest.append(
        Operation.DETERMINISM_CHECK,
        {"check": "resident_models_loaded", "result": "PASS",
         "models": models.loaded_names, "device": device,
         "insightface_det_size": list(INSIGHTFACE_DET_SIZE),
         "pyannote_ovd_hyperparams": PYANNOTE_OVD_HYPERPARAMS,
         "policy": "resident_for_entire_batch_no_unload"},
    )
    return models


# =============================================================================
# Gate entrypoint
# =============================================================================

def run_gate(
    model_store: Path | str,
    manifest: SessionManifest,
    load_models: bool = True,
) -> Optional[ResidentModels]:
    """Full startup gate, in order. Any failure records a blocking halt in
    the manifest and raises EnvironmentGateError. On success, returns the
    resident model registry (or None when ``load_models=False``)."""
    store = Path(model_store)
    check_runtime_platform(manifest)
    check_ffmpeg(manifest)
    check_versions(manifest)
    check_forbidden_imports(manifest)
    verify_model_store(manifest, store)
    enforce_torch_determinism(manifest)
    check_onnxruntime_cuda(manifest)
    gpu_determinism_selftest(manifest)
    if load_models:
        return load_resident_models(manifest, store)
    return None


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB environment gate (Module 0b)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true",
                      help="full startup gate (Ubuntu deployment box)")
    mode.add_argument("--freeze-hashes", action="store_true",
                      help="hash the model store and write the immutable "
                           "expected-hash registry (staging box, run once)")
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip installs, no torch)")
    parser.add_argument("--model-store", type=Path,
                        help="model store root directory")
    parser.add_argument("--manifest", type=Path,
                        help="session manifest path (.jsonl)")
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--no-load-models", action="store_true",
                        help="run all checks but skip resident model loading")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_stdlib()

    if args.freeze_hashes:
        if not args.model_store:
            parser.error("--freeze-hashes requires --model-store")
        registry = freeze_model_hashes(args.model_store)
        print(f"frozen hash registry written: {registry}")
        return 0

    if not args.model_store or not args.manifest:
        parser.error("--run requires --model-store and --manifest")
    with SessionManifest(args.manifest, operator_id=args.operator) as manifest:
        run_gate(args.model_store, manifest,
                 load_models=not args.no_load_models)
    print("environment gate PASSED — all checks recorded in manifest")
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch, no GPU)
# =============================================================================

def _selftest_stdlib() -> int:
    import tempfile

    assert "torch" not in sys.modules, "torch imported at module level"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        store = tmp_dir / "model_store"
        for directory in REQUIRED_MODEL_DIRS.values():
            (store / directory).mkdir(parents=True)
            (store / directory / "weights.bin").write_bytes(
                directory.encode("utf-8") * 64
            )

        # Freeze, then verify: must pass.
        freeze_model_hashes(store)
        with SessionManifest(tmp_dir / "m1.jsonl") as manifest:
            verify_model_store(manifest, store)

        # Tampered weight file: must halt.
        target = store / "yolov8" / "weights.bin"
        os.chmod(target, 0o644)
        target.write_bytes(b"tampered")
        try:
            with SessionManifest(tmp_dir / "m2.jsonl") as manifest:
                verify_model_store(manifest, store)
            raise AssertionError("tampered store passed verification")
        except EnvironmentGateError:
            pass
        halt_entries = [
            e for e in SessionManifest.verify_chain(tmp_dir / "m2.jsonl")
            if e["operation"] == Operation.BLOCKING_HALT
        ]
        assert halt_entries and halt_entries[-1]["payload"]["mismatched"] == [
            "yolov8/weights.bin"
        ]

        # Unexpected extra file: must halt.
        target.write_bytes(b"yolov8" * 64)  # restore content
        (store / "yolov8" / "extra.bin").write_bytes(b"smuggled")
        try:
            with SessionManifest(tmp_dir / "m3.jsonl") as manifest:
                verify_model_store(manifest, store)
            raise AssertionError("extra file passed verification")
        except EnvironmentGateError:
            pass

        # Version pin check with injected metadata: pass then halt.
        good = dict(PINNED_VERSIONS)
        with SessionManifest(tmp_dir / "m4.jsonl") as manifest:
            check_versions(manifest, version_of=lambda n: good[n])
        bad = dict(good, numpy="2.0.0")
        try:
            with SessionManifest(tmp_dir / "m5.jsonl") as manifest:
                check_versions(manifest, version_of=lambda n: bad[n])
            raise AssertionError("version mismatch passed verification")
        except EnvironmentGateError:
            pass

        # ffmpeg preflight with injected probes: pass, then halt.
        with SessionManifest(tmp_dir / "m6.jsonl") as manifest:
            check_ffmpeg(manifest,
                         probe=lambda b: f"{b} version 6.1.1-3ubuntu5")
        try:
            with SessionManifest(tmp_dir / "m7.jsonl") as manifest:
                check_ffmpeg(
                    manifest,
                    probe=lambda b: "" if b == "ffprobe"
                    else "ffmpeg version 6.1.1",
                )
            raise AssertionError("missing ffprobe passed the gate")
        except EnvironmentGateError:
            pass
        ffmpeg_halts = [
            e for e in SessionManifest.verify_chain(tmp_dir / "m7.jsonl")
            if e["operation"] == Operation.BLOCKING_HALT
        ]
        assert ffmpeg_halts
        assert ffmpeg_halts[-1]["payload"]["missing"] == ["ffprobe"]

        # Environment variables fixed at import time.
        for key, value in DETERMINISM_ENV.items():
            assert os.environ.get(key) == value, key

    assert "torch" not in sys.modules, "self-test imported torch"
    print("environment_gate stdlib self-test OK — no torch, no GPU, no pip")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
