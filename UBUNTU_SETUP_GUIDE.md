# SPOVNOB — Ubuntu Workbench Setup Guide

**Maintained automatically: this file is updated whenever a new dependency, model, or system requirement is introduced. It always reflects the current complete state of what Ubuntu needs.**

**Last updated:** 2026-06-15 (CUDA/driver reconciliation — host runs 13.x)
**Target:** Ubuntu 22.04 LTS · RTX 6000 Ada (48 GB) · Python 3.10 · driver 580 / **CUDA 13.x** host runtime · PyTorch `+cu121` wheels (CUDA 12.1 runtime, bundled — they run fine on the 13.x driver; see §5 for the extra cuFFT path this requires)

Recommended install root used throughout: `/opt/spovnob/`

```
/opt/spovnob/
├── code/                 # session_manifest.py, environment_gate.py, ...
├── .venv/                # Python 3.10 virtualenv
├── wheelhouse/           # vendored pip wheels (air-gap install source)
├── model_store/          # vendored model weights (layout in §4)
└── session/              # manifests + outputs per batch
```

---

## 1. System prerequisites (apt + NVIDIA driver)

```bash
sudo apt update
sudo apt install -y build-essential git curl unzip \
    python3.10 python3.10-venv python3.10-dev \
    ffmpeg libsndfile1 libglib2.0-0
```

Notes:
- **Ubuntu 22.04 ships Python 3.10 natively** — no PPA needed. Verify: `python3.10 --version` → `3.10.x`.
- **ffmpeg**: jammy repo version (4.4.x) is fine; the apt package also installs `ffprobe`, which Layer 0 requires for PTS probing. Record the exact version once installed: `ffmpeg -version | head -1` → goes into the session manifest at first batch init.
- **NVIDIA driver**: this box runs the **580-series** driver (CUDA 13.x host runtime). The `+cu121` wheels need a driver ≥ 535 *at minimum*, but 580 is what is installed here — and CUDA 13.x is precisely why §5 adds the `cufft/lib` entry to `LD_LIBRARY_PATH` (ORT 1.17.1's CUDA-12 build links `libcufft.so.11`, which CUDA 13 no longer ships system-wide):
  ```bash
  sudo apt install -y nvidia-driver-580-server
  sudo reboot
  nvidia-smi    # must show the RTX 6000 Ada and driver >= 580 (CUDA 13.x)
  ```
- **CUDA toolkit: NOT required.** The PyTorch `+cu121` wheels bundle the CUDA 12.1 runtime, cuBLAS, and cuDNN 8.9 as pip packages (`nvidia-*-cu12`). Do not apt-install `nvidia-cuda-toolkit` — it would add a second, unpinned CUDA to the box.
- **cuDNN for ONNXRuntime**: also satisfied by the pip-bundled `nvidia-cudnn-cu12`; the env exports in §5 make ORT find it. No system cuDNN install.

## 2. Python environment

```bash
sudo mkdir -p /opt/spovnob && sudo chown "$USER" /opt/spovnob
cd /opt/spovnob
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade "pip==24.0"
```

## 3. Python packages (pinned)

The single source of truth is `requirements.txt` (pinned, with two extra
wheel indexes baked in: PyTorch cu121 and the ONNXRuntime CUDA-12 feed).

**Online staging box** (same OS/arch as the deployment box):
```bash
pip download -r requirements.txt -d /opt/spovnob/wheelhouse
```

**Air-gapped deployment box** (after transferring `wheelhouse/`):
```bash
pip install --no-index --find-links /opt/spovnob/wheelhouse -r requirements.txt
```

If the workbench has internet (non-air-gapped bring-up), a direct
`pip install -r requirements.txt` is equivalent.

**After install — freeze and hash the environment:**
```bash
pip freeze > requirements.lock
sha256sum requirements.lock    # recorded in the session manifest at batch init
```

Critical wheel checks (these are the two known failure points, FLAG 2):
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
#   expected: 2.1.2+cu121 12.1   (if it prints 2.1.2 without +cu121, the
#   PyTorch extra index was missed — reinstall torch/torchaudio/torchvision)
python -c "import onnxruntime as o; print(o.get_available_providers())"
#   expected to include: CUDAExecutionProvider   (after §5 env exports;
#   if missing, the onnxruntime-gpu wheel came from default PyPI = CUDA 11.8
#   build — re-download via the CUDA-12 feed index in requirements.txt)
```

## 4. Model vendoring (five models → `model_store/`)

Run on the online staging box, then transfer `model_store/` whole.
Gated HF models need a logged-in token: `huggingface-cli login`.

```bash
export MS=/opt/spovnob/model_store && mkdir -p "$MS"

# 1. Silero VAD — git snapshot pinned to COMMIT (tag v4.0 was moved upstream;
#    commits are immutable, tags are not)
git clone https://github.com/snakers4/silero-vad.git "$MS/silero-vad"
git -C "$MS/silero-vad" checkout 915dd3d639b8333a52e001af095f87c5b7f1e0ac
rm -rf "$MS/silero-vad/.git"        # store is weights-only; pin lives in this guide + manifest

# 2. ECAPA-TDNN (SpeechBrain, C=1024, 192-dim) 
huggingface-cli download speechbrain/spkrec-ecapa-voxceleb \
    --local-dir "$MS/speechbrain-spkrec-ecapa-voxceleb"

# 3. YOLOv8 medium (person detection; variant is manifest-logged)
mkdir -p "$MS/yolov8"
curl -L -o "$MS/yolov8/yolov8m.pt" \
    https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt

# 4. InsightFace buffalo_l (SCRFD + ArcFace + 2d106det landmarks)
#    FaceAnalysis expects <root>/models/<pack_name>/
mkdir -p "$MS/insightface/models"
curl -L -o /tmp/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip /tmp/buffalo_l.zip -d "$MS/insightface/models/buffalo_l"

# 5. PyAnnote segmentation-3.0 (GATED — accept conditions at
#    huggingface.co/pyannote/segmentation-3.0 first, then:)
huggingface-cli download pyannote/segmentation-3.0 \
    --local-dir "$MS/pyannote-segmentation-3.0"
```

(HuBERT was removed from the pipeline 2026-06-12 — behavioral analysis is
deferred to a separate design phase. Five models, not six.)

**Freeze the hash registry (run ONCE, on the staging box, after all five
downloads):**
```bash
cd /opt/spovnob/code
python3 environment_gate.py --freeze-hashes --model-store /opt/spovnob/model_store
sha256sum /opt/spovnob/model_store/expected_hashes.json   # write this down
```

Expected SHA-256 values: this guide deliberately does **not** pre-list
per-file hashes. The authoritative hashes are frozen from the verified
staging download into `model_store/expected_hashes.json` (written read-only,
chmod 444). That file — whose own SHA-256 you recorded above — is the pinned
truth the gate verifies against on every startup; any added, removed, or
modified file in the store is a blocking halt.

**Transfer to the air-gapped box:** copy `code/`, `wheelhouse/`,
`model_store/` (including `expected_hashes.json`). Verify the archive:
`sha256sum` the tarball on both sides before extraction.

## 5. Runtime environment exports

Add to the shell profile (or the systemd unit) of the pipeline user —
ONNXRuntime needs to see the pip-bundled NVIDIA libraries:

```bash
source /opt/spovnob/.venv/bin/activate
NVLIB=$(python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")/nvidia
TORCHLIB=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$NVLIB/cufft/lib:$NVLIB/cuda_runtime/lib:$NVLIB/cublas/lib:$TORCHLIB:${LD_LIBRARY_PATH:-}"
```

**Why four entries instead of the original cudnn/cublas/cuda_runtime:**
This system runs CUDA 13.x (driver ≥ 580). The ORT 1.17.1 CUDA-12 build links
`libcufft.so.11` (CUDA 12's cuFFT soname — CUDA 13 uses `.so.12`) and
`libcudart.so.12`. The `nvidia-cufft-cu12` and `nvidia-cuda-runtime-cu12` pip
packages supply these exact libraries; `nvidia-cublas-cu12` supplies the
matching cuBLAS. The `torch/lib` path covers `libcublasLt.so.12` (from the
PyTorch +cu121 bundle). On CUDA ≤ 12.x systems the original three-path export
is sufficient; on CUDA 13.x the `cufft/lib` entry is mandatory.

Required pip packages: these are **transitive dependencies of `torch==2.1.2+cu121`**
(its bundled CUDA libs) at exactly these versions, so `pip download -r requirements.txt`
already vendors them into `wheelhouse/` and they install with the main pip step — no
separate line item in requirements.txt. The explicit install below is only a
belt-and-suspenders check if you suspect they were skipped:
```bash
pip install "nvidia-cufft-cu12==11.0.2.54" \
            "nvidia-cuda-runtime-cu12==12.1.105" \
            "nvidia-cublas-cu12==12.1.3.1"
```

(The determinism variables `CUBLAS_WORKSPACE_CONFIG`, `HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE` do NOT need exporting — `environment_gate.py` sets
them at import time, before any CUDA context exists.)

## 6. Verify the setup — environment gate

This is the go/no-go check. Nothing else runs until it passes.

```bash
cd /opt/spovnob/code
source /opt/spovnob/.venv/bin/activate

# Stage 1 — checks only (fast: pins, checksums, CUDA, ORT, GPU determinism):
python3 environment_gate.py --run --no-load-models \
    --model-store /opt/spovnob/model_store \
    --manifest /opt/spovnob/session/gate_check.manifest.jsonl

# Stage 2 — full gate including resident loading of all five models:
python3 environment_gate.py --run \
    --model-store /opt/spovnob/model_store \
    --manifest /opt/spovnob/session/gate_check.manifest.jsonl \
    --operator "<your operator id>"
```

Success looks like: `environment gate PASSED — all checks recorded in manifest`.
Every check (version pins, per-file model checksums, torch/CUDA state, ORT
provider, the ~10s GPU determinism workload hash, resident model load) is an
entry in the manifest; on any failure the gate writes a `blocking_halt` entry
and exits non-zero.

### Troubleshooting

| Gate halt reason | Cause | Fix |
|---|---|---|
| `version_pin_mismatch` | wrong wheel versions installed | reinstall from `requirements.txt` exactly; check `pip freeze` |
| `wrong_cuda_version` / torch without `+cu121` | PyTorch extra index missed | reinstall torch trio with the cu121 index |
| `onnxruntime_cuda_provider_missing` | default-PyPI ORT wheel (CUDA 11.8) or missing `LD_LIBRARY_PATH` | re-download ORT via the CUDA-12 feed (FLAG 2); apply §5 exports |
| `model_checksum_failure` | store modified / partial transfer | re-transfer `model_store`; never edit it post-freeze |
| `hash_registry_missing` | freeze step skipped | run `--freeze-hashes` on the staging box |
| `gpu_determinism_failure` | driver/library drift | verify driver ≥ 535, exact wheel pins; do not proceed |
| `wrong_platform` / `wrong_python` | not Ubuntu / not 3.10 venv | use the §2 venv on the deployment box |
| `ffmpeg_missing` | ffmpeg and/or ffprobe not on PATH | `sudo apt install ffmpeg` (§1), then re-run the gate |

---

*Dependency change log:*
- *2026-06-12 — initial guide: full pinned stack (requirements.txt Rev 1 + flag rulings), six-model vendoring, gate verification (Modules 0a/0b).*
- *2026-06-12 — Module 1 (layer0_preprocessor.py): no new dependencies. Uses the already-listed ffmpeg/ffprobe binaries and the resident Silero model. Standalone run: `python3 layer0_preprocessor.py --run --videos <files...> --work-dir <dir> --model-store <store> --manifest <jsonl>`.*
- *2026-06-12 — Module 2 (layer1_enrollment/ package): no new dependencies. Run: `python3 -m layer1_enrollment --run --videos <files...> --clicks clicks.json --work-dir <dir> --model-store <store> --manifest <jsonl>`. Operator clicks JSON format: `{"speaking_click": {"file_index": 0, "pts_ms": 41250, "x": 812, "y": 440}, "anti_click": {...optional...}}` (pts_ms must be integers). **Bench validation required before the first real batch:** (1) the MAR landmark indices from the architecture doc (`upper_inner_lip` 52/53/54, `lower_inner_lip` 61/62/63, `mouth_width_pair` 52/61) must be verified against real InsightFace 2d106det output — the width pair as documented duplicates the vertical pair and is geometrically suspect; correct values are EnrollmentParams fields (manifest-logged change, no code edit). (2) Confirm `face.pose` (head yaw) is populated by the buffalo_l pack; if absent, yaw suspension is inactive and a per-video warning is logged.*
- *2026-06-12 — Module 3 (layer2_tracker.py): no new dependencies. Run (chains gate → Layer 0 → Layer 1 → Layer 2 authoritative pass): `python3 layer2_tracker.py --run --videos <files...> --clicks clicks.json --work-dir <dir> --model-store <store> --manifest <jsonl>` (add `--preview` for a non-authoritative preview pass). Output: `<work-dir>/layer2/layer2_output.json` (canonical JSON, SHA-256 in manifest) plus per-file worker logs merged into the manifest in canonical order.*
- *2026-06-12 — Module 4 (layer3_contamination.py): no new dependencies. Uses the resident PyAnnote OVD pipeline (already vendored: pyannote-segmentation-3.0). Run (full chain gate → Layers 0-3): `python3 layer3_contamination.py --run --videos <files...> --clicks clicks.json --work-dir <dir> --model-store <store> --manifest <jsonl>`. Outputs: final verified clean segment WAVs + sidecars under `<work-dir>/layer3/clean/`, NaN exclusion log in the manifest, and `<work-dir>/layer3/layer3_output.json` (canonical JSON, SHA-256 in manifest).*
- *2026-06-12 — WavLM/HuBERT removal: behavioral analysis deferred to a separate design phase. `transformers` dropped from requirements.txt; `facebook/hubert-large-ll60k` dropped from the model store (§4) — **five** resident models. If you already vendored HuBERT, delete `model_store/hubert-large-ll60k/` and re-run `--freeze-hashes` (the gate rejects unexpected files).*
- *2026-06-12 — Module 5 (pipeline_runner.py): no new dependencies. **This is the production entrypoint** for full batch runs: `python3 pipeline_runner.py --run --videos <files...> --clicks clicks.json --work-dir <dir> --model-store <store> --manifest <jsonl> [--operator <id>]`. Chains gate → Layers 0-3, writes `<work-dir>/pipeline_output.json` (canonical, SHA-256 in manifest), re-verifies the full manifest hash chain from disk after close, and prints per-stage wall timings (console only — never in payloads).*
- *2026-06-12 — Pre-deployment hardening audit: no new dependencies. (1) The gate now verifies **ffmpeg AND ffprobe** on PATH as step 2 (new halt reason `ffmpeg_missing` — previously a missing binary failed mid-batch as an unrecorded FileNotFoundError). (2) Layer 2 calibration now handles a seed-only (single-window) enrollment pool by routing to `FALLBACK_DEFAULTS` instead of crashing in the leave-one-out scorer. (3) Operator-click face-match failures (`no_face_at_speaking_click` / `no_face_at_anti_click`) now write the re-click WARNING to the manifest before raising, like every other re-click path.*
- *2026-06-15 — CUDA/driver reconciliation (no dependency or pin changes): the deployment box runs the **580-series driver / CUDA 13.x host runtime**, not the 535/CUDA-12.1 stated in the original header. Header, §1, and the requirements.txt header now reflect this; §5's `cufft/lib` `LD_LIBRARY_PATH` entry (already present) is **mandatory** on this host because ORT 1.17.1's CUDA-12 build links `libcufft.so.11`, which CUDA 13 no longer ships system-wide. The PyTorch wheels are unchanged (`+cu121`); they run on the 13.x driver. The three `nvidia-*-cu12` libs in §5 are torch's transitive deps (auto-vendored), not separate requirements.txt entries — wording clarified.*
