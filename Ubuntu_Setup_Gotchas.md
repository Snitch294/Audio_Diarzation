
# SPOVNOB Setup Troubleshooting & Gotchas
*Keep this document handy when setting up the actual Ubuntu system tomorrow.*

During our WSL2 setup, we ran into a few edge cases caused by how Python packages and HuggingFace caching behave. Here is exactly how to solve them when you do this on the real server:

### 1. PyTorch CUDA Libraries Path (cuDNN / cuBLAS)
**The Problem:** The pipeline expects system-level CUDA/cuDNN, but the PyTorch `+cu121` wheels bundle their own libraries hidden inside the `torch` pip package.
**The Fix:** You must dynamically find where PyTorch installed them and export that to `LD_LIBRARY_PATH` before running the gate.
```bash
# Run this inside the activated venv:
CUDNN_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
CURAND_PATH=$(python -c "import site, os; print(os.path.join(site.getsitepackages()[0], 'nvidia', 'curand', 'lib'))")
export LD_LIBRARY_PATH="${CUDNN_PATH}:${CURAND_PATH}:${LD_LIBRARY_PATH}"
```

### 2. ONNX Runtime silently installing the CUDA 11.8 version
**The Problem:** The `requirements.txt` specifies `--extra-index-url` for the CUDA 12 version of `onnxruntime-gpu==1.17.1`. However, because the main PyPI index has the exact same version number (built for CUDA 11.8), `pip` gets confused and installs the wrong one, causing InsightFace to fail over to the CPU (`libcublasLt.so.11 not found`).
**The Fix:** Force pip to use *only* the Microsoft index for that specific package using `--index-url`:
```bash
pip uninstall -y onnxruntime-gpu
pip install onnxruntime-gpu==1.17.1 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

### 3. SpeechBrain crashing due to `HF_HUB_OFFLINE=1`
**The Problem:** `environment_gate.py` hardcodes offline mode (`HF_HUB_OFFLINE=1`). If you download the ECAPA-TDNN model using the `--local-dir` flag, HuggingFace bypasses its hidden cache. SpeechBrain then tries to verify `label_encoder.txt` online, sees offline mode is on, and immediately crashes.
**The Fix:** Download the HuggingFace models directly into the global cache *before* running the gate, ensuring the internet is briefly turned on:
```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
huggingface-cli download speechbrain/spkrec-ecapa-voxceleb
huggingface-cli download pyannote/segmentation-3.0
```
*Air-gap caution: on the production box, prefer fixing this by making the vendored model-store directory complete (every repo file, including `label_encoder.txt` and `hyperparams.yaml`), rather than relying on the global HF cache — the cache lives outside `expected_hashes.json`, so a cache-dependent load is not covered by the startup checksum gate. See `SPOVNOB_TECHNICAL_DEEP_DIVE.md`, Part IX.*

### 4. `expected_hashes.json` Permission Denied Error
**The Problem:** Running `python environment_gate.py --freeze-hashes` creates the hash registry file. If you ever need to re-run it (for example, if a model updates), the script throws a `PermissionError` because it intentionally locks the file to prevent tampering.
**The Fix:** You must manually delete the old file before re-freezing:
```bash
rm -f ~/model_store/expected_hashes.json
python environment_gate.py --freeze-hashes --model-store ~/model_store
```

---

### 5. Claude Code / Node.js Installation Failures
**The Problem:** The default `nodejs` package in Ubuntu 22.04 is v12.x, but Claude Code requires v18+. Trying to install Claude Code out of the box will throw an `EBADENGINE` and `SyntaxError: Unexpected token '.'` error.
**The Fix:** You must install Node.js v20 via the NodeSource repository before installing Claude Code:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
```

### 6. The `ManifestTimeError` Pipeline Crash — FIXED IN REPO (no manual edit needed)
**The Problem:** `session_manifest.py` enforces Rule 6: any dictionary key ending in `_ms` must hold an integer. `layer0_preprocessor.py` used to log a *list* of segment pairs under the key `silero_segments_local_ms`, which crashed the pipeline at the first Layer 0 manifest write.
**The Fix (already shipped):** Fixed properly in commit `7dc3daa` — the payload key is now `silero_segments`, holding nested `{"start_ms": ..., "end_ms": ...}` dicts, so Rule 6 actively validates every segment boundary as an integer instead of being bypassed. The shape is covered by `python3 layer0_preprocessor.py --selftest`. **On the server, just `git pull` — do not apply the old manual rename to `silero_segments_list`; that workaround is superseded.**

*Note: WSL2-specific issues (like installing WSL, or using the Windows NVIDIA driver instead of the Linux one) were omitted here since they will not apply to your bare-metal Ubuntu server tomorrow.*

---

### 7. InsightFace 2d106det Landmark Indices — Document Mapping is Wrong
**The Problem:** The architecture doc specifies `upper_inner_lip=(52,53,54)`, `lower_inner_lip=(61,62,63)`, `mouth_width_pair=(52,61)`. These are wrong: `eu(52,61)` is the ~80px horizontal left-to-right corner distance, which appears in the *numerator* alongside the other pairs, making MAR ~constant at 0.44–0.57 regardless of whether the mouth is open or closed. The window machine could never close windows reliably and produced a single 66-second mega-window.
Additionally, indices 72–86 are **nose landmarks**, not inner lip — the 2d106det model provides outer lip contour only (52–71); there are no dedicated inner lip points.

**The Fix (already in params.py as of 2026-06-12):** Use the inner-edge points of the outer lip contour — the bottom rim of the upper arc and the top rim of the lower arc:
- `upper_inner_lip = (71, 63, 68)` — bottom of upper lip outer arc (center, left-ctr, right-ctr)
- `lower_inner_lip = (62, 54, 57)` — top of lower lip outer arc (center, left-ctr, right-ctr)
- `mouth_width_pair = (52, 61)` — left/right outer corners (unchanged, ~80px horizontal)
- `mar_on = 0.15`, `mar_off = 0.10` — re-calibrated for new formula range (0.10–0.25)

Pairs (71,62), (63,54), (68,57) are nearly vertically aligned (Δx < 8px each) and correctly measure the mouth opening gap. The Silero VAD gate in `window_machine.py` remains as a required secondary trigger since even the corrected outer-lip MAR overlaps slightly between speaking and listening states at the single-frame level.

**Validated result:** Corrected formula found 3 extra enrollment windows and recovered 7,250ms more clean audio vs. the document's indices on the NT-clip test batch.
