# SPOVNOB — Technical Deep Dive

**Document:** `SPOVNOB_TECHNICAL_DEEP_DIVE.md` · v1.0 (first complete draft)
**Date:** 2026-06-12
**Ground truth:** the Python modules at repository head (`766462b`; Modules 0a–5, code-complete) and `Audio_Diarization.md` Revision 3.1.
**Authority order:** where any narrative document (including this one) disagrees with the code, **the code is authoritative**. Every divergence between the master specification and the implementation found during the audit for this document is recorded explicitly, most of it in Part IX.
**Audience:** forensic auditors, reviewing engineers, and future maintainers. The document assumes fluency in Python and basic signal processing, and zero prior exposure to this codebase.

**How to read this document.** Parts I–III establish the doctrine and the two substrate modules everything else stands on (the determinism gate and the session manifest). Parts IV–VII walk the four pipeline layers in execution order, each at the level of exact predicates, constants, and arithmetic. Part VIII closes the end-to-end audit loop. Part IX is the honesty section: everything that is flagged, assumed, or awaiting bench validation. The appendices are complete reference tables. All thresholds quoted use the comparison operator the code actually uses — `>` and `≥` are never interchangeable in a forensic gate.

---

## Table of Contents

- **Part I — Mission, Constraints, and Forensic Doctrine**
- **Part II — The Determinism Substrate** (`environment_gate.py`)
- **Part III — The Session Manifest: Chain of Custody as a Data Structure** (`session_manifest.py`)
- **Part IV — Layer 0: PTS-True Extraction and the Speech Map** (`layer0_preprocessor.py`)
- **Part V — Layer 1: Visual-Anchored Enrollment** (`layer1_enrollment/`)
- **Part VI — Layer 2: Calibrated Sliding-Window Tracking** (`layer2_tracker.py`)
- **Part VII — Layer 3: Overlap Exclusion and Final Output** (`layer3_contamination.py`)
- **Part VIII — The Pipeline Runner and the End-to-End Audit Story** (`pipeline_runner.py`)
- **Part IX — Bench-Validation Register and Known Limitations**
- **Appendix A — Complete Parameter Reference**
- **Appendix B — Manifest Operation Vocabulary**
- **Appendix C — Artifact Map, Dependency Graph, and Pinned Stack**

---

# Part I — Mission, Constraints, and Forensic Doctrine

## 1.1 What SPOVNOB does

SPOVNOB ingests one **batch** of interview video files — 5 to 15 files of 5–10 minutes each, all cut from **one continuous recording session** (same subject, same room, same microphone, same day) — plus one mandatory operator click (and one optional one) on the first video. It emits **PTS-timestamped WAV segments containing only the visually verified target speaker's speech, verified free of overlapped speech**, each segment SHA-256-hashed, plus a hash-chained audit log recording every parameter, decision, acceptance, discard, warning, and halt that produced them.

Two properties define the system more than any model inside it:

1. **Every output sample existed in the source.** The pipeline never synthesizes, interpolates, reconstructs, or separates audio. Output WAVs are byte slices of the original 16 kHz extraction. Contaminated audio is *excluded*, never repaired.
2. **The whole run is a deterministic function of its inputs.** Re-running the same batch with the same model store produces bit-identical decision payloads and output hashes. "Same inputs, same date or two years later, same SHA-256s" is the design standard, not an aspiration.

## 1.2 The non-negotiable constraints, and why each one exists

| # | Constraint | Forensic rationale |
|---|---|---|
| 1 | **Bit-identical determinism** (CUDA deterministic mode, float32-only, fixed batch shapes, order-independent reductions) | An independent auditor must be able to reproduce the run and match hashes. "Statistically similar" output invites the question *which run is the evidence?* — a question with no good answer. Bit-identity makes verification a string comparison. |
| 2 | **Zero synthesis / zero separation** (HTDemucs, SepFormer, SpeakerBeam and all source-separation models categorically forbidden) | Separation models *hallucinate* plausible acoustic structure by construction — they output what the network believes the isolated source sounded like. No witness can testify which emitted sample energies existed in the room. Exclusion is the only mathematically safe operation: it can lose information but cannot invent it. |
| 3 | **Time is absolute integer milliseconds derived from container PTS** — never frame indices, never floats | Frame-index timing assumes constant frame rate; variable-frame-rate (VFR) footage makes `index × nominal_rate` drift unboundedly. PTS is the container's own presentation clock. Integers, because float time accumulates representation error and breaks both equality comparison and hashing. Enforced mechanically by manifest Rule 6 (§3.4). |
| 4 | **Fully offline / air-gapped, with checksum-pinned models** | Model weights are part of the evidence chain: a silently updated hub checkpoint changes outputs with no code change. Every weight file's SHA-256 is verified against a frozen registry before anything runs (§2.6); all hub-download code paths are disabled at import time. |
| 5 | **Append-only, hash-chained audit manifest, written before destructive operations** | Chain of custody. Any retroactive edit, deletion, or reorder of the record is detectable by re-hashing (§3.3). The record of an action exists durably *before* the action does (§3.5). |
| 6 | **Minimal, validated, recorded human input** | Two clicks maximum; every operator threshold override requires an identity and a stated reason, recorded append-only. The human is in the loop but cannot be invisibly in the loop. |

## 1.3 Why not conventional deep-learning diarization

SPOVNOB's architecture is best understood as a sequence of refusals, each recorded in the master specification's decision records:

- **End-to-end neural diarization (EEND) was eliminated** for two structural hazards. *Label permutation:* EEND processes audio in chunks and assigns speaker labels that permute freely between chunks; stitching a coherent multi-file identity timeline out of permutation-invariant outputs is mathematically unstable and indefensible under cross-examination. *VFR tensor desync:* audio-visual EEND variants require perfectly paired audio/video tensors, and pairing VFR footage requires interpolation — which violates constraint 2.
- **Clustering diarizers** (embedding + spectral/AHC clustering) answer "how many voices, and which segments group together" — they still leave *which cluster is the target* to post-hoc inference. SPOVNOB replaces that inference with ground truth: an operator-witnessed visual identity anchor (Part V).
- **WavLM-based speaker verification reuse was rejected** on four audited grounds (master spec, Layer 2 Decision Record): the cached final-layer features are the wrong features (published SV heads consume a learnable weighted sum over *all* transformer layers, because speaker identity concentrates in lower layers); the turnkey checkpoint's backbone has diverged from the vanilla backbone; the accuracy delta is marginal at this operating point (two known speakers, same room, same microphone, ≥45 s verified enrollment — UniSpeech VoxCeleb1-O EERs: standalone ECAPA 1.08 %, frozen-WavLM+head 0.75 %, jointly fine-tuned 0.43 %, all measured on a far harder open-set task); and forensic defensibility favors the most independently replicated open speaker encoder in existence over a research-repo integration. The corollary: WavLM was deleted from Layer 0 entirely, because under the pure-ECAPA architecture its embeddings had **zero consumers**.
- **Probability-style scoring was rejected** after a concrete near-miss: early drafts treated ECAPA cosine similarities as `P(Target)` with 0.85/0.65/0.30 tiers. Cosine similarities are not calibrated probabilities (§6.1); the fix was structural, not cosmetic.

What remains is the doctrine: **frozen, widely replicated encoders, surrounded by deterministic arithmetic** — cosines, medians, quantiles, duration-weighted means — **and explicit, logged decision rules.** Nothing learns at runtime. Every number an auditor encounters is reproducible from the inputs with a pocket calculator and patience.

## 1.4 Execution environment and session topology

**Production hardware (fixed):** NVIDIA RTX 6000 Ada (48 GB VRAM) · 44 cores / 88 threads · 512 GB DDR5 · 2 TB NVMe · Ubuntu 22.04 LTS · Python 3.10.x · CUDA 12.1 · PyTorch-only stack (sole sanctioned exception: InsightFace's internal ONNXRuntime, §2.6).

**Resident Model Policy.** All five models — Silero VAD (CPU), ECAPA-TDNN, YOLOv8m, InsightFace `buffalo_l`, PyAnnote segmentation-3.0 OVD — are loaded **once** at batch start by the environment gate and held resident for the entire batch. There is no `torch.cuda.empty_cache()` anywhere in the codebase, and no load/unload state machine between files; combined footprint is far under the 48 GB available, and unload/reload cycles are both wasted time and a reproducibility risk surface.

**Memory policy.** The entire batch's 16 kHz PCM (~1–2 GB for 15 files) is preloaded into RAM as raw `bytes` buffers at Layer 0 (§4.5). Every later layer slices these buffers; no audio is re-read from disk mid-pipeline.

**Parallelism with determinism guardrails.** FFmpeg extraction fans out across CPU threads (§4.5). Layer 1 is **strictly sequential in canonical file order** — the cumulative enrollment pool is order-dependent *by design* (§5.10). Layers 2 and 3 are written through per-file `WorkerLog`s merged under the Canonical Manifest Merge Rule (§3.6), so their records are byte-identical regardless of scheduling, even though current code runs the GPU stages sequentially (a shared resident model is not provably race-free across threads, and at 2–3 inference batches per file fan-out buys nothing).

**Session topology guarantee.** A batch *is* a session. Per-batch threshold calibration (§6.2) is therefore per-session by construction; no session-grouping machinery exists because none is needed. Cross-video drift detection (§6.7) is informational only — same microphone, same room, drift is expected to be minimal.

## 1.5 Module map

```
session_manifest  ◄──  environment_gate  ◄──  layer0_preprocessor  ◄──  layer1_enrollment  ◄──  layer2_tracker  ◄──  layer3_contamination  ◄──  pipeline_runner
(0a, stdlib only)      (0b)                   (1)                       (2, package)            (3)                  (4)                       (5)
```

Arrows point from importer to imported. The graph is strictly acyclic; no module imports a later one (Implementation Rule 3). Every module from 0b onward begins with `import environment_gate` as its **first** import, because that import fixes the process environment (§2.2) before anything else can observe it. Every module ships a stdlib-only self-test runnable on a bare Python 3.10 with zero installed packages — the seams that make that possible are part of the architecture, not test scaffolding (§8.4).

---

# Part II — The Determinism Substrate (`environment_gate.py`)

## 2.1 Why the standard is bit-identity

Every decision payload in the manifest is hashed (`payload_sha256`), every layer output document is hashed, every emitted WAV is hashed. A single float that differs in its last mantissa bit anywhere in the decision path changes a hash and breaks the audit equality. So the engineering target is not "numerically stable" but **bit-identical**: the run must be a pure function from (input files, model store, parameters) to bytes. Everything in this part exists to make GPU inference — normally the least reproducible component in an ML system — satisfy that.

## 2.2 The four CUDA determinism constants, mechanism by mechanism

Applied by the gate; constants #2–#4 inside `enforce_torch_determinism()`, #1 at **module import time**, before any CUDA context can exist:

1. **`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`** — cuBLAS chooses GEMM reduction strategies (e.g., split-K) partly based on available workspace; different strategies sum partial products in different orders, and floating-point addition is not associative. Fixing the workspace configuration pins the strategy and therefore the summation order. It must be set before the CUDA context is created, which is why `environment_gate` sets it in module-level code and why every entrypoint imports the gate first.
2. **`torch.use_deterministic_algorithms(True)`** — forces PyTorch to select deterministic kernel implementations and to *raise* on operations that have no deterministic implementation, rather than silently varying (e.g., atomics-based scatter kernels).
3. **`torch.backends.cudnn.deterministic = True`** — restricts cuDNN to deterministic convolution algorithms (excludes the atomic-add-based backward paths, irrelevant here since the pipeline is inference-only, and certain non-deterministic forward algorithms).
4. **`torch.backends.cudnn.benchmark = False`** — disables cuDNN autotuning. The autotuner times candidate algorithms *at runtime* and picks the fastest; the winner depends on transient machine state, so two identical runs could execute different algorithms with different rounding behavior. Off means the algorithm choice is a pure function of the operation shape.

Supporting policies set in the same function: `torch.set_num_threads(8)` (`TORCH_NUM_THREADS` — CPU-side op partitioning affects reduction order too), `torch.manual_seed(20260611)` (`GLOBAL_SEED` — defense-in-depth; nothing in the pipeline samples at runtime, but a forgotten dropout or augmentation path would at least be *reproducibly* wrong), float32 precision throughout (no AMP, no TF32 paths in use), and `torch.no_grad()` around every forward pass in every module.

The same import-time block sets **`HF_HUB_OFFLINE=1`** and **`TRANSFORMERS_OFFLINE=1`** — the air-gap switches must exist before any HuggingFace-aware library is imported, for the same ordering reason as the cuBLAS variable.

## 2.3 Fixed batch shapes and floating-point reduction order

Because FP addition is non-associative, **the batch shape is part of the numeric function**: pooling statistics, batch-norm-free or not, the kernel tiling over a `[256, T]` tensor differs from a `[200, T]` tensor. SPOVNOB therefore treats inference batch sizes as *architectural constants*, recorded in the manifest at gate time:

- **`ECAPA_BATCH_WINDOWS = 256`** — Layer 2's sliding-window scorer (§6.3) groups window spans by length (uniform tensor shapes per forward pass), iterates groups in ascending length order, and slices each group into batches of exactly 256. When a length group spans more than one batch, the final partial batch is padded **by repeating the last window's tensor** until the batch is full; padded rows' outputs are discarded by index bookkeeping. Repeat-padding (rather than zero-padding) keeps the pad rows in-distribution and, more importantly, keeps every forward pass in the dominant scan at the identical `[256, T]` shape. One honest nuance: a length group with ≤256 spans total runs as a single batch at its natural size (`if pad and len(indices) > batch_size`). The shape is still a pure function of the input — same input, same shapes, same bits — so reproducibility holds; the constant's job is to make the *long* coarse scans shape-invariant, and it does.
- **`VISUAL_BATCH_FRAMES = 32`** — Layer 1's YOLO person gate consumes frames in fixed batches of 32 (§5.3).
- Layer 1 enrollment windows are deliberately encoded **batch-of-1** (`ecapa_encode_pcm`): they are variable-length by design ("no arbitrary truncation"), so a fixed multi-window batch shape is impossible; batch-of-1 *is* the fixed shape.

## 2.4 Order-independent CPU arithmetic

The CPU side gets the same treatment as the GPU side:

- **`math.fsum`** is used for every reduction that matters — pool means, duration-weighted means, variances, MAR vertical averages, dot products and norms in `cosine()`. `fsum` computes the correctly rounded exact sum of its inputs (Shewchuk's algorithm), so the result is independent of summation order by construction. A plain `sum()` over floats would make pool arithmetic dependent on list order.
- **`statistics.median`** aggregates overlapping window scores per block (§6.4). Its tie rule is fixed and documented: even-count inputs return the arithmetic mean of the two middle order statistics.
- **`decimal.Decimal` with `ROUND_HALF_EVEN`** converts every ffprobe decimal-seconds string to integer milliseconds (§4.2).
- **Integer milliseconds everywhere** (manifest Rule 6, §3.4). At 16 kHz, 1 ms is exactly 16 samples, so `ms × 16` is exact integer arithmetic in both directions; `ms_from_samples` floors (`samples * 1000 // 16000`).

## 2.5 The ~10-second GPU workload checksum

`gpu_determinism_selftest()` is the startup proof that the determinism constants are actually in force on *this* machine, *this* driver, *now* — not a configuration assertion but a measurement:

```python
generator = torch.Generator(device="cuda").manual_seed(20260611)
x = randn(2048, 2048); w = randn(2048, 2048)        # float32, seeded
kernel = randn(8, 1, 7, 7)
for _ in range(200):
    x = torch.tanh(x @ w) * 0.5                     # cuBLAS GEMM chain
feature_map = conv2d(x.view(1, 1, 2048, 2048), kernel, padding=3)   # cuDNN
payload = first 65,536 elements of x  +  first 65,536 of feature_map
return sha256(payload)
```

The workload runs **twice from scratch**; the two SHA-256s must match exactly or the gate halts (`gpu_determinism_failure`). Design notes: the 200-iteration GEMM chain amplifies any reduction-order divergence exponentially (a single differing bit in iteration 1 avalanches), `tanh(·)·0.5` keeps values bounded so the chain cannot saturate to ±∞ and mask divergence, and the conv2d exercises the cuDNN algorithm-selection path that constants #3/#4 govern. The passing hash is recorded in the manifest (`gpu_workload_checksum`), which converts an *intra-machine* proof into an *inter-machine* check: identical hardware/driver/stack claims between an original run and an audit replay become verifiable by comparing recorded workload hashes. What the checksum cannot prove: that a *different* GPU architecture would produce the same bits — deterministic-per-machine is the guarantee; cross-architecture identity is checked, not assumed, via the recorded hash.

## 2.6 The gate sequence and the Model Vendoring Mandate

`run_gate()` executes, in order, halting (with a `blocking_halt` manifest entry written *first*) on any failure:

| Step | Check | Halts on |
|---|---|---|
| 1 | `check_runtime_platform` | non-Linux OS; Python not 3.10.x; any of the three import-time env vars drifted |
| 2 | `check_ffmpeg` | ffmpeg or ffprobe missing/unusable on PATH (Layers 0–1 shell out to both; gated so the failure cannot surface mid-batch unrecorded) |
| 3 | `check_versions` | any of the 23 pinned packages (Appendix C) not installed at its exact pin |
| 4 | `check_forbidden_imports` | `pyannote.audio.pipelines.speaker_verification` present in `sys.modules` |
| 5 | `verify_model_store` | any weight file missing, unexpected, or hash-mismatched |
| 6 | `enforce_torch_determinism` | CUDA version ≠ 12.1; CUDA unavailable |
| 7 | `check_onnxruntime_cuda` | `CUDAExecutionProvider` not in available providers |
| 8 | `gpu_determinism_selftest` | the two workload hashes differ |
| 9 | `load_resident_models` | any loader failure |

Three of these encode lessons already validated in practice:

- **Step 4 (FLAG 1).** PyAnnote's SpeechBrain speaker-verification wrapper imports the `speechbrain.pretrained` module that SpeechBrain 1.x removed (pyannote issues #1661/#1677). SPOVNOB never uses that code path — Layer 3 uses `OverlappedSpeechDetection` only — so the broken import is guarded by policy rather than papered over by upgrading into an unpinned dependency set. Scope, stated precisely: the check runs *before* the resident loader, and pyannote's own package `__init__` later pulls the module into `sys.modules` transitively (wrapped in pyannote's optional-backend `try/except`, never instantiated by SPOVNOB) — so post-load presence is expected, and what the gate actually enforces is the realistic violation vector: a *direct* import by SPOVNOB code.
- **Step 7 (FLAG 2).** `onnxruntime-gpu==1.17.1` exists under the *same version number* on the default PyPI index built for CUDA 11.8 and on the Microsoft CUDA-12 feed built for CUDA 12.x. On this CUDA 12.1 box, the wrong wheel doesn't crash — InsightFace **silently falls back to CPU**, changing both performance and, potentially, numerics. The gate refuses to start unless the CUDA provider is actually live. This exact failure was reproduced during the WSL2 dry run (`Ubuntu_Setup_Gotchas.md` §2: `libcublasLt.so.11 not found` → CPU fallback), which is the strongest possible argument for the gate's existence. The related runtime prerequisite — PyTorch `+cu121` wheels bundle their own cuDNN/cuBLAS inside the `torch` package, and ONNXRuntime needs them on `LD_LIBRARY_PATH` — is a deployment step (Gotchas §1), and the gate is what catches it when forgotten.
- **Steps 5 & 9 (Model Vendoring Mandate).** On the staging box, `freeze_model_hashes()` runs **once**: it refuses to freeze if any of the five model directories is missing or empty, hashes every regular file under the store (sorted walk; the registry file itself excluded), writes `expected_hashes.json` as canonical JSON, and `chmod 444`s it — re-freezing requires a deliberate manual delete (Gotchas §4; intentional friction). On the air-gapped box, `verify_model_store()` re-hashes everything and computes a three-way set difference: `missing`, `unexpected`, `mismatched`. **`unexpected` is load-bearing**: a smuggled extra file inside a model directory halts the gate even though no expected file changed — the store's contents are closed-world. Every verified file gets its own `model_checksum` manifest entry. Loading is local-only by construction: Silero via `torch.jit.load` of the vendored snapshot (commit-pinned to `915dd3d639b8333a52e001af095f87c5b7f1e0ac` — pinned to a commit, not a tag, because the upstream `v4.0` tag was observed to move); ECAPA via `EncoderClassifier.from_hparams(source=<store dir>, savedir=<same dir>)`; YOLO from the local `yolov8m.pt`; InsightFace with `root=<store>` and `providers=["CUDAExecutionProvider"]` only; PyAnnote `Model.from_pretrained(<local pytorch_model.bin>)` wrapped in `OverlappedSpeechDetection` and instantiated with the pinned hyperparameters `{"min_duration_on": 0.0, "min_duration_off": 0.0}` (§7.2). One open hardening item from the dry run — SpeechBrain's loader can fall back to the global HuggingFace cache for auxiliary files (e.g., `label_encoder.txt`) if the vendored directory is incomplete, and that cache is *outside* the hash registry's coverage — is recorded in Part IX.

## 2.7 The time doctrine: three coordinate frames

All timing in SPOVNOB lives in one of exactly three integer-millisecond frames, and every conversion site is fixed:

| Frame | Zero point | Conversion |
|---|---|---|
| **Data-relative ms** | first audio sample of one file's extraction | — |
| **Local PTS ms** | the file's container clock | `local = data_relative + audio_start_pts_ms` |
| **Global session ms** | start of the batch timeline | `global = local + file_offset_ms` |

`audio_start_pts_ms` is the audio stream's `start_time` as reported by ffprobe (§4.1) — containers routinely start audio at a small nonzero PTS (e.g., 23 ms), and ignoring it would shift every downstream timestamp. `file_offset_ms` is the cumulative duration of all preceding files in canonical order (§4.3). Layer 1 operates in local PTS (its PCM slicer subtracts `audio_start_pts_ms` to reach samples). Layer 2 plans and scores windows in data-relative ms (its scorer slices with start offset 0), then logs and emits local and global. Layer 3's overlap regions are produced in local PTS (the provider adds `audio_start_pts_ms`, §7.2). Rule 6 (§3.4) enforces integerness at every manifest boundary, so a float can never leak into the timeline through any payload.

---

# Part III — The Session Manifest: Chain of Custody as a Data Structure (`session_manifest.py`)

## 3.1 Entry anatomy and the payload/audit split

The manifest is a single append-only JSON Lines file. Each line:

```json
{
  "schema":         "spovnob-manifest-v1",
  "seq":            17,
  "operation":      "calibration",
  "payload":        { "...deterministic content only..." },
  "payload_sha256": "sha256 of canonical-JSON payload",
  "prev_sha256":    "entry_sha256 of the previous line (or 64 zeros, GENESIS)",
  "audit": {
    "timestamp_utc": "2026-06-12T03:41:07.221Z",
    "operator_id":   "operator-7",
    "stated_reason": null
  },
  "entry_sha256":   "sha256 of this entry minus this field"
}
```

The decisive design choice is the **payload/audit split**. Payloads contain *only* deterministic content — no wall-clock time, no hostnames, no operator identity — so `payload_sha256` is **bit-reproducible across re-runs**: an auditor replaying the batch months later produces the *same payload hash stream*. Wall-clock time and operator identity live exclusively in the `audit` block, which exists for custody (when, who, why), not for reproduction. `entry_sha256` seals the *whole* entry including the audit block and the chain link, so the chain hashes are run-specific while the payload hashes are the cross-run comparable quantity. This is exactly the right factoring: *what happened* is reproducible; *when and by whom* is tamper-evident.

## 3.2 Canonical JSON

One serializer (`canonical_json`) is used for every hash input and every line written, with four locked flags, each load-bearing:

```python
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
```

`sort_keys` removes dict-insertion-order variance; fixed `separators` remove whitespace variance; `ensure_ascii` removes encoding variance (the byte stream is pure ASCII regardless of platform defaults); `allow_nan=False` makes float NaN/Inf unserializable — which means the *concept* "NaN block" in Layer 3 is, by necessity and by design, the **string designation** `"NaN"`, never an IEEE NaN (an IEEE NaN also breaks equality and hashing, so the serializer refusing it is defense-in-depth).

## 3.3 The hash chain, verification, and what tampering looks like

Each entry carries `prev_sha256` — the previous entry's seal — forming a chain from `GENESIS` (64 zeros). `SessionManifest.verify_chain(path)` is the auditor's entry point; for every line it re-derives and checks four things: (1) `entry_sha256` equals the hash of the entry minus that field; (2) `payload_sha256` equals the hash of the payload; (3) `prev_sha256` equals the previous entry's seal; (4) `seq` equals the line's position (0-based, dense). First inconsistency raises `ManifestChainError` with the line number.

| Tampering | Detected by |
|---|---|
| Edit any byte of any payload | checks 1 and 2 on that line |
| Edit the audit block (backdate a timestamp) | check 1 on that line |
| Delete a line | check 3 on the next line (and check 4 thereafter) |
| Reorder lines | checks 3 and 4 |
| Insert a forged line | check 3 on the following original line |
| **Truncate the tail** | *not* detectable by the chain alone — a valid prefix is a valid chain. Mitigations: the runner's terminal `pipeline_complete` entry is the designated last operation (§8.1), the runner re-verifies and records the entry count after close, and the recommended practice (Part IX) is to retain the final `entry_sha256` off-box, which converts truncation and full-rewrite attacks into detectable ones. |

Append-only is enforced three independent ways: the file handle is opened in append mode (never truncated); an exclusive **`fcntl` advisory lock** (`LOCK_EX | LOCK_NB`) makes a second concurrent writer fail fast with `ManifestLockError` (single-writer rule); and the chain makes any retroactive modification detectable. Every `append()` ends with `flush()` + `os.fsync()` before returning. On reopen, the constructor (default `verify_on_open=True`) re-verifies the entire existing chain before resuming the sequence — a crashed run cannot be silently continued atop a corrupted log.

## 3.4 Rule 6: `validate_time_fields`

Every key ending in `_ms`, at any nesting depth in any payload (the validator recurses through dicts, lists, and tuples), must hold a Python `int`. Two subtleties:

- **`bool` is rejected explicitly** (`isinstance(value, bool) or not isinstance(value, int)`) because `bool` subclasses `int` in Python — without the explicit check, `"start_ms": True` would validate.
- The rule is **suffix-driven, so payload shape matters**. Case study, fixed in commit `7dc3daa`: Layer 0's payload originally placed a list of `[start, end]` pairs under `"silero_segments_local_ms"` — a list is not an `int`, so the first real batch would have died with `ManifestTimeError` at the first Layer 0 write (the zero-pip self-tests couldn't reach that code path because it needs ffmpeg and Silero). The shipped fix does not rename the key to dodge the suffix; it restructures the data so the rule *actively covers it*: `"silero_segments": [{"start_ms": s, "end_ms": e}, ...]` — the validator recurses into each dict and enforces integerness on every boundary. The payload construction was extracted into the pure `layer0_file_payload()` so the self-test validates the exact dict production writes, plus a negative case asserting that corrupting one boundary to a float raises. The general lesson is itself doctrine: when a validation rule and a data shape collide, reshape the data *into* the rule's coverage, never around it.

## 3.5 Rule 7: the write-before-destruction guarantee

`guard_destructive(action, payload, ...)` appends a `destructive_op` entry and — like every append — returns only after the bytes are fsync'd. **Returning from the call is the guarantee**: the caller proceeds with the irreversible action only once its record is durable. If the process dies mid-action, the manifest already says what was about to happen and why. The inverse ordering (act, then log) is how chain-of-custody gaps are born.

## 3.6 Worker logs and the Canonical Manifest Merge Rule

Parallel (or parallelizable) stages never write the main manifest directly. Each per-file worker writes a `WorkerLog` — plain JSONL, *not* hash-chained (it is an intermediate artifact), each record carrying its canonical sort fields:

```json
{"file_index": 3, "start_ms": 127000, "operation": "layer2_block", "payload": {...}}
```

`start_ms = -1` marks file-level records (e.g., activity-ratio summaries), which deliberately sort before all timed records of their file. `merge_worker_logs()` then reads every record from every log and sorts by the four-tuple

```
(file_index, start_ms, operation, sha256(canonical_json(payload)))
```

— a **deterministic total order independent of worker count, scheduling, and arrival time** (the payload hash as final tiebreaker makes the order well-defined even for records identical in the first three fields). The single manifest writer appends the sorted records, then a `worker_log_merged` summary entry carrying each source log's file SHA-256 and record count — so the intermediate artifacts are integrity-pinned even though they aren't chained. Consequence, verified in the module self-test: two workers finishing in any order produce a byte-identical payload stream in the merged manifest. This is the mechanism that lets the architecture *permit* Layer 2/3 fan-out without ever letting scheduling reach the evidence.

---

# Part IV — Layer 0: PTS-True Extraction and the Speech Map (`layer0_preprocessor.py`)

## 4.1 Extraction and probing

Per file, two subprocess calls with fully explicit, auditable argument lists:

```
ffprobe -v error -print_format json -show_streams -show_format <source>
ffmpeg  -hide_banner -nostdin -y -i <source> -vn -map 0:a:0
        -acodec pcm_s16le -ar 16000 -ac 1 -f wav <dest.wav>
```

`-map 0:a:0` selects the **first audio stream deterministically** (no ffmpeg default-stream heuristics); `-vn` means no video decode; `pcm_s16le/16000/1` is the pipeline-wide sample contract, validated again on read-back by the stdlib `wave` reader (mono, 16-bit, 16 kHz, or `Layer0Error`). `parse_probe()` extracts the PTS-relevant facts as a pure function (testable without the binary):

- **`audio_start_pts_ms`**: the audio stream's `start_time`, converted by the half-even rule (§4.2). Missing or `"N/A"` → `0` **plus `audio_start_missing=True`** — recorded, never guessed; the flag travels into the manifest payload.
- **`vfr_suspected`**: ffprobe's `avg_frame_rate` and `r_frame_rate` are parsed as exact `Fraction`s and compared (`"30000/1001"` parses exactly; string comparison would false-positive on equivalent spellings, float comparison on precision). Inequality of the two — average achieved rate vs. base rate — is the standard VFR fingerprint. The flag is *diagnostic* (it matters to the visual layer's frame pairing, §5.3); audio processing is index-free by construction, so VFR cannot corrupt the audio timeline regardless.

## 4.2 The rounding rule

Every ffprobe decimal-seconds string becomes integer milliseconds via `Decimal(value) * 1000`, quantized with **`ROUND_HALF_EVEN`** (banker's rounding). Two reasons: the value is parsed *as a string* into `Decimal`, so `"0.023220"` is represented exactly rather than passing through binary float re-rounding; and half-even is bias-free under accumulation — over thousands of packet timestamps, half-up rounding adds a systematic +0.5 ms drift expectation to ties, half-even cancels it. From the self-test, the rule in action: `"2.0005" → 2000`, `"2.0015" → 2002` (both ties round to the even millisecond).

## 4.3 The global session clock

Canonical file order is the **lexicographic sort of full source paths** — recorded in the `batch_init` entry, along with the ffmpeg version string, every Silero parameter, and each source file's SHA-256. `file_offset_ms` accumulates: file *k*'s offset is the sum of durations of files 0..k−1. A `video_gap` manifest entry is written at each file boundary after the first ("new file boundary within continuous session") — the audit trail's explicit acknowledgment that the session timeline crosses a file seam there. The conversion helpers live on `FileAudio`: `to_global_ms(local) = file_offset_ms + local`, and `sample_to_local_pts_ms(i) = audio_start_pts_ms + ms_from_samples(i)`.

## 4.4 Silero VAD: the non-destructive speech map

Silero (CPU-resident TorchScript, ~1 MB, zero VRAM) is run sequentially per file: `reset_states()` (stateful model — per-file reset is itself a determinism requirement), then 512-sample windows (= exactly 32 ms at 16 kHz) in order, producing one speech probability per window. The final partial window is zero-padded **for inference only** — the stored buffer is never modified. That sentence is the whole doctrine in miniature: **Silero never touches the audio.** Hard-zeroing non-speech (an earlier design's mistake, fixed in Rev 2 of the master spec) creates artificial amplitude cliffs that corrupt downstream acoustic processing; SPOVNOB's VAD output is *a map*, not a mask.

`segments_from_window_probs()` converts probabilities to integer-ms segments in a **fixed four-step order**:

```
1. threshold:   window is speech iff prob >= 0.50          (SILERO_THRESHOLD)
2. merge:       join speech runs separated by silence < 100 ms
3. drop:        discard runs shorter than 250 ms  — measured BEFORE padding
4. pad:         widen each survivor by 30 ms per side, clamped to
                [0, audio_end]; re-merge any overlaps padding creates
```

The order is load-bearing: pad-before-drop would promote sub-250 ms blips over the survival threshold by inflation. Worked example (from the self-test, 32 ms windows): 8 speech windows, 4 silence, 8 speech → runs `0–256 ms` and `384–640 ms`; the 128 ms gap survives step 2 (≥100 ms); both runs survive step 3; padding yields `(0, 286)` and `(354, 640)` (start clamped at 0, end clamped at the 640 ms audio end). The caller adds `audio_start_pts_ms` to every pair, so `FileAudio.silero_segments_local_ms` is in local PTS. These segments gate Layer 1's window starts and Gate A coverage, Layer 2's skip rule, and the activity-ratio denominator — one map, four consumers, zero modifications.

## 4.5 Batch assembly

`preprocess_batch()`: canonical sort → missing-file check (blocking halt listing the absent paths) → source hashing → `batch_init` entry → **parallel** ffmpeg extraction across `EXTRACT_WORKERS = 8` threads → **sequential** (canonical-order) per-file Silero inference and manifest writes. The split is the parallelism doctrine applied: the expensive, side-effect-free stage fans out; everything that writes the record runs in canonical order, so the manifest's payload content is identical no matter how the extraction pool was scheduled. Each file ends as a `FileAudio`: index, paths, source/WAV SHA-256s, sample count, `duration_ms`, `audio_start_pts_ms` (+ missing flag), `vfr_suspected`, `file_offset_ms`, the in-RAM `pcm` bytes, and the Silero map — followed by its `layer0_file` payload (the Rule-6-validated nested-segment shape of §3.4, including `silero_total_speech_ms`).

---

# Part V — Layer 1: Visual-Anchored Enrollment (`layer1_enrollment/`)

## 5.1 The philosophy: replace identity inference with identity witness

The hard problem in diarization is not "where is speech" but "*whose* speech." Statistical systems answer it by clustering and post-hoc assignment — inference an auditor can only take on faith. SPOVNOB answers it with a witnessed anchor: the operator clicks the target *while watching them speak*; the clicked face becomes a biometric lock (`F_target`, ArcFace embedding); and from then on, **audio enters the enrollment pool only when the locked face is visibly producing speech** at the same PTS, with the audio side corroborated by VAD. Everything downstream — `E_seed`, `E_composite`, `E_anti` — is **arithmetic over frozen encoder outputs**: duration-weighted means and cosines, not training. The Zero-Training Mandate is absolute: no model weight changes anywhere, ever; "enrollment" is a weighted average that an auditor can recompute by hand from the persisted per-window WAVs and vectors.

## 5.2 Operator surface

The clicks file is JSON:

```json
{"speaking_click": {"file_index": 0, "pts_ms": 41250, "x": 812, "y": 440},
 "anti_click":     {"file_index": 0, "pts_ms": 95000, "x": 300, "y": 400}}
```

`anti_click` (Track C) is optional by design. `file_index`/`pts_ms` are integer-validated at parse; both clicks must target the **first video** or the run halts (`speaking_click_not_on_first_video`) — anchors propagate forward, never backward. Validation failures raise `Layer1ReclickError` *after* a `reclick_required` warning entry with the machine-readable reason — the operator feedback loop is itself part of the record. Because the pipeline is offline, the doctrine encourages the operator to pre-review the video and click inside the **longest clean speaking stretch** available: a 20–30 s verified seed is materially better than a minimal one, because `E_seed` is the reference for Gate B and the M-Trap (§5.7).

## 5.3 The vision stack

Frame timing comes from the container, never from frame counting: `video_frame_pts_ms()` reads every video packet's `pts_time` via ffprobe, converts with the half-even rule, and **sorts ascending** (packets arrive in decode order; presentation order is sorted PTS). OpenCV decodes sequentially in presentation order, so decoded frame *i* pairs with `pts[i]`. A count mismatch between decoded frames and listed PTS sets `pts_mismatch`, which the orchestrator escalates to a `frame_pts_mismatch` warning; pairing stops at the shorter length (review-flagged honesty: on mismatch the tail is unanalyzed rather than mis-timed).

Per batch of `VISUAL_BATCH_FRAMES = 32` frames: **YOLOv8m** runs as a person gate (`classes=[0]`, `conf=0.30`); frames with no person detection skip InsightFace entirely (the master spec's "empty frames skipped"). For person-containing frames, **InsightFace `buffalo_l`** (det size 640×640) yields faces; any face with `det_score < 0.50` is treated as **not detected** (guardrail 5 — a low-confidence face must not be allowed to *fail* an identity check; it must simply not exist). Each surviving face carries its ArcFace embedding (`normed_embedding` preferred), its 106-landmark set → MAR (§5.4), and `yaw = face.pose[1]` when the pose attribute exists (assumption registered in Part IX). An optional silence-stride rule (analyze every Nth frame outside Silero speech ± tolerance; default stride 1 = off) is documented and input-deterministic, but unnecessary at this batch size.

## 5.4 Mouth Aspect Ratio: geometry, smoothing, hysteresis, suspension

**Formula** (`compute_mar`): with 2d106det landmarks, vertical gaps are the Euclidean distances between paired upper/lower inner-lip indices `(52,61), (53,62), (54,63)`; MAR = `fsum(gaps)/3` divided by the mouth width `d(52, 61)`; `None` if the landmark array is short or the width is degenerate (≤1e-6). Division by width is the normalization that makes MAR face-scale-invariant — a face twice as close to the camera doubles both numerator and denominator.

> **Registered caveat (Part IX, item 1):** the width pair `(52, 61)` is the master document's verbatim value, and it *duplicates the first vertical pair* — as written, the ratio's denominator is one of its own numerator terms, which would make MAR hover near a constant. The indices are deliberately implemented as **parameters** (`EnrollmentParams`, manifest-logged), carrying a `VALIDATE-ON-BENCH` banner in `params.py`: confirming the true 2d106det mouth-corner indices against real model output is the first Ubuntu bench task, and correcting them is a *recorded parameter change*, not a code change.

**Smoothing**: a 5-frame causal EMA, `α = 2/(span+1) = 1/3`, update `v ← αx + (1−α)v`. The EMA is **pre-seeded with the first observed value** — a zero-initialized EMA needs ~4 frames to converge, and that warm-up artifact could suppress a start trigger at the head of a video (the master spec's "pre-seed the EMA buffer with frame 0's MAR" rule). The EMA **persists across windows** (it smooths a physical signal, not window state) and **freezes** during suspension or target absence — a stale-but-honest value beats a fabricated one.

**Hysteresis**: `MAR_on = 0.55` opens, `MAR_off = 0.40` arms the close timer. Two thresholds, not one: a single threshold turns lip-area noise at the boundary into open/close chatter; the 0.15 dead band means a state flip requires a decisive move.

**Yaw suspension**: when `|yaw| > 35°`, lips project into foreshortened geometry and MAR crashes artificially; MAR checking is **suspended entirely** (no transitions fire, the plosive timer pauses — §5.5) rather than fed a corrupt value. Unknown yaw does **not** suspend; instead the orchestrator logs `pose_unavailable` once per video, so an InsightFace build without pose degrades visibly, not silently.

## 5.5 The `E_window` capture state machine (`window_machine.py`)

A pure, deterministic FSM over per-frame observations (`FrameObs`: PTS, target presence/MAR/suspension, interviewer presence/MAR, VAD-near flag). States: `IDLE → ACTIVE → PLOSIVE_BUFFER → (ACTIVE | emit)`.

| State | Condition (exact) | Action |
|---|---|---|
| IDLE | target present ∧ ¬suspended ∧ smoothed MAR > `MAR_on` ∧ Silero speech within ±50 ms of frame PTS (`vad_near`) | `T_start = pts`; → ACTIVE |
| ACTIVE | suspended | accumulate; EMA frozen; wait for head return |
| ACTIVE | target absent **or** smoothed MAR < `MAR_off` | → PLOSIVE; `deadline = pts + 500 ms` |
| PLOSIVE | suspended | **pause the timer**: `deadline += pts − prev_pts` |
| PLOSIVE | `pts ≥ deadline` | emit window with **`T_stop = deadline`** (`plosive_expiry`) |
| PLOSIVE | interviewer present ∧ interviewer EMA > `MAR_on` | emit **immediately**, `T_stop = pts` (`interviewer_interjection`) |
| PLOSIVE | target present ∧ EMA > `MAR_on` | cancel timer; → ACTIVE |
| any open | end of video | emit at last frame PTS (`end_of_video`) |

Five rules deserve their "why":

- **The plosive buffer (500 ms)** exists because plosives and brief closures (/p/, /b/, swallows) close the lips mid-utterance; without the buffer, every "p" would split the window. 500 ms is the doc's default (tunable 300–700).
- **The Early Stop Rule** is the contamination firewall at capture time: if the *interviewer's* lips open while the target's closure timer runs, the window ends **now** and the buffer is discarded — otherwise a rapid interviewer interjection landing inside the 500 ms grace period would be captured into enrollment audio. This is the single most important transition in the machine.
- **Clean expiry stops at the deadline PTS, not the noticing frame.** Frames arrive at ~33 ms granularity; the timer may be observed expired up to a frame late. Using the observation frame's PTS would leak up to one frame of post-closure audio into every window. The deadline itself is the correct, frame-rate-independent stop.
- **Suspension pauses the timer** (deadline extended by the suspended wall time): a turned head removes the *evidence* of closure, and the conservative reading is that the closure clock should not run while the evidence is absent (review-flagged decision).
- **Target absence is treated as a closure** (same plosive semantics, EMA frozen): the master document does not specify the absence case; treating disappearance as the start of a possible stop is the conservative deterministic choice (review-flagged).

The machine also accumulates per-window statistics for Gate A and the audit trace: frame count, interviewer-present and interviewer-closed frame counts (closed = interviewer EMA < `MAR_off`), and the full `(pts, smoothed MAR | None-while-suspended)` trace, persisted in the window's JSON sidecar.

## 5.6 Seed construction and the biometric lock

On video 1: build an observation stream anchored on the *clicked face's* embedding (nearest analyzed frame with faces within `CLICK_MATCH_MAX_GAP_MS = 200` of the click PTS; prefer a bbox containing the click point, else nearest face center), run the window machine over the whole video, and select the emitted window **containing the click PTS**. Then the seed guardrails, in order: no containing window → re-click (`click_outside_speaking_window`); duration < `seed_min_ms = 3000` → re-click (guardrail 2; the code supersedes the legacy "3–8 s" wording — 3 s minimum, **no maximum**); fraction of seed frames where a non-target face had open lips > `click_overlap_max_frac = 0.20` → re-click (guardrail 1's visual overlap proxy). The biometric lock is then *refined*: `F_target` becomes the L2-normalized mean of **every** matched face embedding across the seed span — a multi-frame lock is robust to the single-frame pose/blur/expression lottery that a click-frame-only anchor would inherit. The seed window is encoded (§5.9's single-pass rule), pooled with `operator_verified=true`, and recorded (`layer1_seed` entry with the vector hash).

## 5.7 Track B: automatic anti-profile collection, and the M-Trap

`E_anti` — the interviewer's profile — powers Gate C and Layer 2's dual-target margin. Track B collects it automatically: a frame is a **candidate center** when a non-target face is present with **raw** MAR < `MAR_off` (lips closed; raw rather than smoothed is a review-flagged simplification) while Silero indicates speech within ±50 ms — *someone is audible and it is visibly not that face speaking... or is it?* Candidates are deduplicated to one per `trackb_min_spacing_ms = 2000`, windowed ±1000 ms (clamped; discarded if clipping leaves <1000 ms), and encoded.

Then the **M-Trap** (guardrail 4): if `cos(candidate, E_seed) > mtrap_sim_max = 0.60`, the candidate is discarded (logged as `mtrap_high_sim_to_seed`). The trap exists because the trigger condition has a systematic false positive: **bilabials and nasals (/m/, /b/, /n/) are produced with closed lips by the target** while the target is the one speaking. Without the trap, those windows — acoustically *target* speech — enter `E_anti`, and the poisoning is self-reinforcing: `sim(E_composite, E_anti)` rises, Gate C margins collapse, and the anti-profile begins rejecting the target's own enrollment windows. The trap's cost is acceptable by construction: a genuine interviewer vector scoring >0.60 against the seed is precisely the acoustically-confusable-voices case in which automatic anti-collection *should* abstain and let guardrail 8 (§5.11) escalate to a human.

Track B vectors aggregate by duration-weighted mean into `E_anti`, recomputed after each video; an increase in the anti-pool's pairwise-cosine variance > `pool_var_warning = 0.05` across videos logs a degradation warning.

## 5.8 Track C: the operator anti-click

When the interviewer is on camera, the operator may click them (lips closed) for a high-confidence anti vector, which also fixes `F_interviewer` — the identity used to *match* the interviewer in subsequent frames (otherwise the highest-confidence non-target face stands in). Three validations, all re-click on failure: clicked face must **not** match `F_target` (`cos ≥ face_reid_threshold = 0.40` → `anti_click_matches_target`, guardrail 3); the *target's* lips must be closed at the click frame (target MAR ≥ `MAR_off` → re-click — the audio at that PTS must not be the target's); Silero must show energy at the click PTS (an anti vector of room tone is worse than none). The ±1000 ms window is encoded and pooled with priority semantics (`anti_track_c`, `operator_verified=true`).

## 5.9 Encoding: the single-pass rule

Every accepted window is encoded by ECAPA-TDNN (SpeechBrain `spkrec-ecapa-voxceleb`, C=1024, 192-dim output) in **one forward pass regardless of length** — on 48 GB even a 60-second window is trivial, so the earlier 10 s sub-chunking rule (a VRAM workaround) was deleted as dead weight. Sanity cap only: above `encode_max_ms = 60000`, split at 60 s boundaries with 2 s overlap and duration-weight-pool the chunk vectors. All d-vectors are **L2-normalized at the encoding boundary** and handled as plain Python float lists above it (framework-free, `fsum`-reduced). PCM slicing is exact: local-PTS ms minus `audio_start_pts_ms`, times 16 samples/ms, clamped to the buffer.

## 5.10 The Triple Validation Gate and pool arithmetic

Candidate Track A windows (from the production observation stream, with identities assigned per frame by best-cosine-above-0.40 matching) first face two pre-gates: **seed-span overlap** (a window intersecting the seed's span on the seed's file is discarded — the seed must not double-count into the pool) and **minimum length** (< `min_enroll_len_ms = 2000` discarded). Survivors are encoded once and evaluated **A → B → C, first failure wins**, every discard logged with its full detail dict:

| Gate | Pass condition (exact code semantics) | Fail reasons logged |
|---|---|---|
| **A** | VAD coverage ≥ 0.50 (Silero-overlap ms ÷ window ms; the spec's "Silero confirms speech," quantified — review-flagged) **and**, if the interviewer was ever visible during the window, interviewer-closed fraction ≥ 0.80 (`int_lips_closed_frac`); never-visible ⇒ vacuous pass | `vad_coverage_low`, `interviewer_lips_open` |
| **B** | `cos(window, E_seed) ≥ 0.70` (`threshold_target`; fails on `<`) | `low_sim_to_seed` |
| **C** | only when the anti pool is non-empty: `cos(window, E_anti) ≤ 0.50` (`threshold_anti`; fails on `>`) **and** `cos_seed − cos_anti ≥ 0.15` (`margin_minimum`; fails on `<`) | `high_sim_to_anti`, `margin_too_small` |

**Gate C fails open**: with no anti profile, it is *skipped, never failed* (`anti_applied=false` recorded) — the deliberate robustness rule for interviewer-off-camera sessions, compensated by the NO_ANTI quality escalation (§5.11) and Layer 2's flagging.

**Pool arithmetic.** `E_composite = L2( Σᵢ vᵢ·dᵢ / Σᵢ dᵢ )` over the cumulative pool (seed + all accepted windows, all videos), recomputed after each video — so video 2's gates already use a better composite than video 1's (the designed order dependence behind the strict-sequential rule). Duration weighting is the defense against short-window noise: a marginal 2 s window cannot outvote a clean 10 s window, because votes are milliseconds. Pool **variance** is the population variance of all C(n,2) pairwise cosines (fixed i<j order, `fsum`) — a scalar self-consistency measure: a clean single-speaker pool has high mutual similarity and low variance; contamination or drift widens it.

## 5.11 Quality states, the second pass, and the freeze

After each video (`assess_quality`):

| State | Condition |
|---|---|
| **STRONG** | verified ≥ 45 000 ms (**60 000 ms when the anti pool is empty** — the NO_ANTI escalation: with Gate C inert, more evidence is demanded) **and** pool variance ≤ 0.05 |
| **MARGINAL** | verified ≥ 20 000 ms (this absorbs the spec's "high variance" branch: meeting the seconds bar with variance > 0.05 lands here, not STRONG) |
| **INSUFFICIENT** | below 20 000 ms |

**MARGINAL second pass** ("second pass with improved anchor"): this video's Gate-C-*failed* candidates are re-evaluated against the **end-of-video, grown** anti profile — vectors and `sim_seed` are reused (each window is encoded exactly once, ever), only `sim_anti` is recomputed. Acceptances re-enter the pool, the composite is recomputed, and the state is reassessed. This recovers windows that failed only because the anti profile was young when they were first judged.

**Guardrail 8**, after every video with an anti profile: `sim(E_composite, E_anti) > 0.45` → warning; `> 0.60` → **blocking halt** (`enrollment_contamination_critical`) — if the target profile resembles the anti profile, either enrollment is contaminated or the voices are genuinely confusable, and both demand a human before any tracking happens. **Guardrail 9**, at batch end: still under 20 s verified (or final state INSUFFICIENT) → terminal halt (`critical_enrollment_failure`). High final variance (> 0.05) is a *warning*, never an auto-discard — "human decides" is the spec's explicit rule.

The batch ends with the **freeze**: a `layer1_freeze` entry carrying `e_composite_sha256`, `e_anti_sha256`, the anti-pool hash, pool sizes, verified ms, variance, and the note "E_composite is FROZEN — never modified after this entry." That hash is the `enrollment_ref` every Layer 2 artifact points back to. Every accepted window has already been persisted as a raw WAV slice plus canonical JSON sidecar (PTS ranges local and global, duration, end reason, MAR trace) under `<work>/enroll/` — the auditor can re-encode any pool member from its preserved audio and reproduce the pool arithmetic end to end.

## 5.12 The nine guardrails, mapped

| # | Guard | Code site |
|---|---|---|
| 1 | Speaking-click overlap | seed window non-target lips-open fraction ≤ 0.20 (§5.6) |
| 2 | Speaking-click duration | seed ≥ 3000 ms (§5.6) |
| 3 | Anti-click identity | clicked face must not match `F_target` (§5.8) |
| 4 | M-Trap | `mtrap_discard` on Track B (§5.7) |
| 5 | InsightFace confidence | `det_score < 0.50` ⇒ face does not exist (§5.3) |
| 6 | Low detection quality | running mean ReID similarity < 0.50 → `low_detection_quality` warning |
| 7 | Separation margin | Gate C margin ≥ 0.15 (§5.10) |
| 8 | Acoustic similarity | `contamination_level` warning 0.45 / halt 0.60 (§5.11) |
| 9 | Critical failure | batch-end INSUFFICIENT → blocking halt (§5.11) |

---

# Part VI — Layer 2: Calibrated Sliding-Window Tracking (`layer2_tracker.py`)

## 6.1 Score semantics: a cosine is not a probability

Layer 2's per-window quantities are **raw cosine similarities in [−1, 1]**, named `S_target` and `S_interviewer` — never `P(·)`. Cosine is the *native* metric of this embedding space: ECAPA is trained with AAM-Softmax (ArcFace-family), which optimizes angular margins; inference that compares anything but angles is operating in a foreign geometry. The historical bug this section guards against: earlier drafts used probability-style fixed tiers (0.85/0.65/0.30). Empirically, genuine same-speaker short-window cosines land around 0.4–0.8 and different-speaker same-channel cosines around 0.1–0.4 — a fixed 0.85 HIGH gate would have starved the HIGH tier on perfectly good sessions and fired near-zero-activity alerts as the *normal* outcome. The structural fix is per-session calibration (§6.2) plus naming discipline enforced through every payload.

## 6.2 Deterministic threshold calibration

Calibration is **pure, ordered arithmetic on frozen vectors** — no sampling, no optimization — run once per batch after the sanity check, bit-reproducible by construction.

**Inputs.** Genuine scores: for each pool vector `vᵢ`, the **leave-one-out** score `sᵢ = cos(vᵢ, L2(duration_weighted_mean(pool ∖ i)))`. LOO-against-the-pooled-rest mirrors the *test condition* — at scan time, a short window is scored against a pooled profile — and the duration weighting matches the composite's own arithmetic, so the genuine distribution is measured with the same estimator the tracker uses. Impostor scores: `tⱼ = cos(aⱼ, E_composite)` for every anti-pool vector — each known interviewer sample against the actual conditioning profile.

**Derivation** (`derive_thresholds`, with every intermediate written into the calibration record):

```
theta_high_raw = max( q10(genuine),  max(impostor) + 0.05 )      # anti available
theta_high_raw = max( q10(genuine),  0.55 )                      # no anti pool
theta_high     = clamp(theta_high_raw, 0.45, 0.75)
theta_med      = max(theta_high − 0.15, 0.30)
evidence floor = 0.20                                            # architectural
```

Term by term: **`q10(genuine)`** places the gate to admit ≈90 % of genuine-like windows (windows the verified pool itself would score). **`max(impostor) + 0.05`** places it strictly above *every observed* impostor with a safety margin — the maximum, not a quantile, because the forensic posture is "no observed impostor may pass," not "few." The **clamp low (0.45)** stops a degenerate genuine distribution from opening the gate into the cross-speaker score zone; the **clamp high (0.75)** is the overlap detector: if the data pushes θ beyond 0.75, the genuine and impostor distributions overlap, the gate cannot separate them alone, a **`CALIBRATION_OVERLAP`** warning is logged, and the margin rule becomes the primary discriminator (a still-higher threshold would reject genuine speech wholesale — flagging beats failing silently). **`quantile_sorted`** is pinned to one definition — linear interpolation between order statistics at position `q·(n−1)` — because "the 10th percentile" has half a dozen conventions and an auditor must reproduce the exact number.

**Degenerations**, all flagged in every downstream block: anti pool empty → kind `DERIVED_NO_ANTI` with the 0.55 floor; fewer than `min_calibration_windows = 10` enrollment windows → kind `FALLBACK_DEFAULTS` with `θ_high = 0.60, θ_med = 0.40` (including the fully degenerate seed-only pool: leave-one-out needs at least two vectors, so a single-window pool yields zero genuine scores and routes here rather than crashing).

**What is deliberately *not* calibrated:** the dual-target margin (`margin_minimum = 0.15`). The margin is a *relative separation requirement between two scores of the same window*, not a property of either marginal distribution; deriving it from pooled statistics would couple it to enrollment quality and quietly weaken the one discriminator that still works when the distributions overlap. It is identical to Layer 1's Gate C margin by design.

**Audit.** The full record — both sorted score lists, n's, every intermediate, clamping events, the enrollment_ref — is written as a `calibration` manifest entry, and `calibration_ref = sha256(record)` stamps every output block. Operator overrides go only through `record_parameter_change` (identity + stated reason, append-only).

Before any scoring, the **sanity gate** re-checks `sim(E_composite, E_anti)` with the same 0.45/0.60 warning/halt constants as Layer 1 — re-checked at the consumer because Layer 2 may run in a different process lifetime than the layer that froze the profile.

## 6.3 Windowing, the skip rule, and the scorer

`plan_windows(duration, 5000, 1000)`: data-relative spans `[0,5000), [1000,6000), …` while they fit; a file shorter than one window yields a single full-file window. **Why 5 s / 1 s:** ECAPA d-vectors need seconds of context before cosine scores stabilize (sub-second windows are noisy in embedding space), while the 1 s hop samples the timeline densely enough that each interior 1 s block receives **five** overlapping estimates — the raw material median pooling needs.

**Silero skip rule:** a window whose overlap with the speech map is `< 20 %` of its length is not scored; it is logged (`layer2_window_skipped`, `SKIPPED_NONSPEECH`) with its PTS span. This is a *compute* skip with an audit trail — the audio is untouched, and the skipped spans are enumerable afterward.

The production scorer (`_ecapa_scorer`) implements §2.3's batching: spans grouped by length, ascending; fixed 256-window batches with repeat-padding on multi-batch groups; embeddings L2-normalized; `S_target = cos(v, E_composite)`, `S_interviewer = cos(v, E_anti)` or `None`. The scorer is a **callable seam** (`Scorer: spans → [(S_t, S_i|None)]`) — production binds the resident model; the self-test injects synthetic scorers and drives the *entire* downstream flow without torch (§8.4).

## 6.4 Median pooling onto the 1-second block grid

The grid has `duration_ms // 1000` blocks; a trailing partial second **never enters the grid** — it can only be excluded, never admitted (conservative by construction). A window votes only on blocks it **fully covers**: `first_block = ceil(w_start/1000)`, `last_block = floor(w_end/1000)` (exclusive). Interior blocks therefore collect 5 votes; blocks near file edges taper to 4…1; blocks with zero covering scored windows become `SKIPPED_NONSPEECH`. Per block, the score is the **median** of its votes (both tracks pooled independently).

**Why median, not mean:** the failure mode being defended against is the transient excursion — a window straddling a turn boundary, a cough, a sliding-window edge artifact. The median of *k* votes tolerates up to ⌊(k−1)/2⌋ corrupted votes before moving materially (for k=5: two bad windows out of five leave the median on clean evidence); a mean is dragged by every excursion proportionally. **Turn-boundary dilution** is the worked case: a 5 s window straddling a speaker change embeds a *mixture*, and its d-vector is a blend — the score is diluted, not wrong. Blocks deep inside a target turn are covered mostly by clean windows and stay HIGH; blocks at the turn edge are covered by mixed windows and sink to MEDIUM — which is the *desired* conservatism, because turn edges are exactly where interviewer bleed lives. Edge-trim (§6.6) then sharpens the surviving boundaries, and Layer 3 independently screens whatever survives.

## 6.5 Tiering and the margin demotion

Exact predicate per evaluated block (`tier_block`):

```
HIGH    iff S_target > theta_high  AND  ( no anti profile
                                          OR (S_interviewer present AND
                                              S_target − S_interviewer > 0.15) )
MEDIUM  iff S_target > theta_med            (also: margin-failed HIGH, see below)
SUB     iff S_target ≥ 0.20                 (sub-threshold evidence log)
REJECT  otherwise
```

A block **above θ_high that fails the margin** is demoted to MEDIUM with **`margin_failed = true`** — not rejected, because the target evidence is real; not HIGH, because the interviewer evidence is too close. That flag is load-bearing downstream: Layer 3's gap dominance guard (§7.5) treats `MEDIUM ∧ margin_failed` as interviewer evidence when deciding whether a gap may be bridged. Every block becomes an `OP_BLOCK` worker record (tier, both medians, evaluation count, margin flag, `no_anti_profile`), so the *full* tier map — not just the survivors — is in the manifest. Tier policy is the spec's Option A: MEDIUM is never auto-promoted to clean output; it is human-review evidence. SUB (0.20–θ_med) is investigative evidence only; below 0.20, the block is rejected with its score logged.

## 6.6 Edge-Trim Boundary Refinement

The coarse scan localizes target speech reliably, but its block edges inherit up to ±1 s of uncertainty — and that uncertainty lives exactly where interviewer bleed does. For each **maximal run of contiguous HIGH blocks**:

- **Positions:** 17 candidate edge positions `p ∈ [edge − 2000, edge + 2000]`, step 250 ms, per boundary.
- **Fine windows:** leading edge scores `[p, p+2000)`; trailing edge scores `[p−2000, p)`. Windows clamped at file bounds to under 1000 ms are skipped (marked `None` in the audit trace). Each boundary's windows go to the scorer as one batch.
- **Pass criterion:** *the same frozen standard as the coarse scan* — `S_target > θ_high` and, with an anti profile, margin `> 0.15`. The fine pass must not invent a new standard; it re-applies the calibrated one at 250 ms granularity.
- **Trim:** new start = the **smallest** passing position `p ≥ run_start`; new end = the **largest** passing position `p ≤ run_end`. Positions outside the run are scanned (their scores appear in the trace for the auditor) but are **ineligible** — which is the trim-only invariant *by construction*: the run can shrink, never grow, so refinement is monotonically conservative and audio outside the coarse HIGH region can never be promoted by this step.
- **Demotion rule:** no passing position, or a required trim beyond `edge_max_trim_ms = 750` → that boundary's **single edge 1 s block** is demoted to MEDIUM (`demoted_by_edge_trim`, tier counts adjusted), and the scan does **not** recurse onto the next block (review-flagged: recursion could cascade-demote an entire run off one bad fine scan; a bounded correction plus an audit trail beats unbounded automation). A one-block run demotes at most once (the trailing demotion is guarded against crossing the new start).
- **Audit:** `leading_trim_ms`, `trailing_trim_ms`, demotion flags, and the **complete fine-scale score traces** per boundary go into an `OP_EDGE` record.

Surviving runs emit per-block records carrying local *and* global PTS, both medians, the edge-trim attribution on the first/last block, and the `no_anti_profile` flag — boundaries now at 250 ms resolution for Layer 3.

## 6.7 Diagnostics: activity ratio and drift

Per file: `ratio = HIGH ms ÷ Silero speech ms` (None if the file has no speech) → `NORMAL > 0.25`, `LOW_ADVISORY 0.10–0.25`, `NEAR_ZERO_ALERT < 0.10` (the alert escalates to a main-manifest warning recommending manual review). Logged beside it: **`unattributed_speech_ms`** = Silero speech not claimed as HIGH — the analyst's context number for distinguishing "interviewer dominated the session" from "model failure." The system cannot auto-distinguish genuine target silence from tracking failure; it can only refuse to hide the question. Cross-video **drift**: the current file's mean `S_target` over its first 30 000 ms of HIGH blocks vs. the previous file's mean over *all* its HIGH blocks; a drop > 0.10 logs a `drift_notice` (informational; the previous mean carries forward across HIGH-less files; quantification review-flagged).

## 6.8 The single authoritative pass and the output document

Enrollment improves monotonically through the batch, so any file scored against an in-progress composite would need rescoring anyway; and Layer 2 has no feedback loop, so exactly one result matters — the one produced with `E_composite_final` and final-pool calibration. Layer 2 therefore runs **once per batch, after Layer 1 completes**, eliminating both the superseded-results clutter and the cold-start hazard (no file is ever scored against a young profile). An optional preview pass exists (operator flag; init entry marked `superseded_by: authoritative_pass`); **Layer 3 refuses non-authoritative input with a blocking halt** (§7.6), so a preview can never leak into output.

The run ends with `layer2_output.json` (schema `spovnob-layer2-output-v1`): `enrollment_ref`, `calibration_ref`, `thresholds_used` (θ values, margin, floor, calibration kind, `operator_modified: false`), and per file the tier counts, activity diagnostics, and HIGH runs with their per-block records. Canonical JSON, SHA-256 recorded as an `output_hash` manifest entry — the hash every Layer 3 artifact points back to.

---

# Part VII — Layer 3: Overlap Exclusion and Final Output (`layer3_contamination.py`)

## 7.1 Why a separate layer

Layer 2 answers *"where is the target speaking?"* with a tracker that, by its nature, cannot certify single-speaker-ness — a high target cosine is compatible with the interviewer talking *over* the target. Layer 3 answers *"is that speech contaminated by simultaneous speech?"* with a model trained for exactly that question, applied as the final filter before anything is called clean. The separation keeps each decision independently auditable, and lets Layer 2 stay (slightly) permissive while Layer 3 stays absolute.

## 7.2 The overlap detector

PyAnnote **segmentation-3.0** wrapped in `OverlappedSpeechDetection`, instantiated by the gate with `{"min_duration_on": 0.0, "min_duration_off": 0.0}` — the model's *own* temporal smoothing and minimum-duration filters are zeroed because **SPOVNOB owns its temporal policy explicitly**: a hidden 100 ms min-duration inside a third-party pipeline would be an unaudited rule shaping the evidence. OVD runs **once over each full file** (decision-noted): a full-file pass gives the segmentation model maximal context, is simpler and deterministic, and is strictly more information than per-block windowing; the resulting regions are then intersected with blocks and gaps in pure integer arithmetic. Region timestamps: the pipeline's float seconds × 1000, through Python `round()` (banker's rounding — same tie rule as Layer 0's Decimal path), plus `audio_start_pts_ms`; finally `merge_regions` (sort; merge overlapping **and touching**; drop degenerates) yields disjoint local-PTS spans, logged per file (`OP_OVD`) with the region count and total ms.

## 7.3 The NaN-Only Exclusion Policy

All interval logic is **half-open** `[start, end)` — `intervals_intersect(a0,a1,b0,b1) = a0 < b1 ∧ b0 < a1` — so blocks that merely *touch* a region (share a boundary millisecond) do not intersect it; adjacency is not contamination.

`classify_run_blocks`: any overlap region intersecting a block voids the **entire block** — a NaN record preserving the designation (`"NaN"` — a string, §3.2), decision `CONTAMINATED`, the span, both Layer 2 medians, and the exact regions hit (`OP_NAN`, permanently in the manifest). **Why whole-block voiding** rather than excising just the overlapped milliseconds: trimming to the overlap's edge means trusting the OVD boundary to the millisecond, and boundary-trust is precisely the guesswork the doctrine forbids — detection models are reliable about *presence*, fuzzy about *edges*. Losing up to one second of genuine speech is the price of never emitting a sample that overlapped; the trade is deliberate and one-directional. The surviving blocks regroup into clean sub-segments wherever adjacent (`prev.end == next.start`).

**The forbidden alternative**, stated in the code's own docstring because it must survive every summarization: separation/reconstruction models (HTDemucs, SepFormer, SpeakerBeam, …) are categorically banned. They would "rescue" the NaN blocks by synthesizing what the isolated target *probably* sounded like — hallucinated acoustic structure, forensically indefensible. Pure exclusion is the only mathematically safe path.

## 7.4 Temporal smoothing — here, and only here

Natural speech is full of sub-400 ms closures (plosives, breath intakes) that fragment HIGH runs. Layer 3 bridges a gap between two clean sub-segments **iff all three hold**:

1. `gap < merge_gap_ms = 400`;
2. the gap interval itself is **overlap-free** (no OVD region intersects it);
3. the **gap dominance guard** (§7.5) does not object.

Why smoothing was *relocated out of Layer 2* (a Rev-2 architectural fix): if Layer 2 bridged a 300 ms gap before overlap detection, it could swallow a rapid 300 ms interviewer interjection — fusing it *into* a block and masking it from the detector. Bridging is only legitimate after the gap has been proven empty, so the merge lives strictly downstream of OVD. Note the asymmetry with §7.3: bridged gap audio **is included in the output segment** — that is the point of smoothing — and it is *unmodified original signal* that has itself just been verified overlap-free; including it reconstructs nothing. A gap created by a voided block can never bridge: it intersects an overlap region by construction (and at ≥1000 ms it also fails the length rule). Every gap evaluation — bridged or refused — is logged with its reason (`bridged` / `gap_too_long` / `overlap_in_gap` / `interviewer_evidence_in_gap`) in the file summary.

## 7.5 The gap dominance guard: closing the OVD blind spot

OVD detects **simultaneous** speech. A solo interviewer interjection over target *silence* — "mm-hm" landing entirely inside a 350 ms target pause — is not overlap, and an overlap-only check would happily bridge that gap, **gluing interviewer audio into the middle of a "clean" target segment**. The guard (implementation-added, default ON, manifest-logged): refuse the bridge if any Layer 2 block overlapping the gap shows interviewer evidence, defined as

```
(tier == MEDIUM and margin_failed)        # target-ish score but interviewer too close
or (S_interviewer_median ≥ S_target_median, both present)
```

It consumes only **already-computed Layer 2 scores** (the full block map that `FileTrack.blocks` carries — the additive Module-4 amendment made for exactly this consumer); no new inference, no new model. And it is **refuse-only**: like trim-only edge refinement, the guard can only withhold a merge, never create or extend audio — the same one-directional safety class as everything else in the output path. With the block map absent it is inert (the spec's base rule is OVD-only).

## 7.6 Outputs and the authoritative-only contract

`run_layer3` **halts** if handed a non-authoritative Layer 2 result (`layer3_requires_authoritative_layer2`). Per surviving segment: the **original PCM** is sliced (`pcm_slice`, exact 16 samples/ms) and written as WAV plus a canonical JSON sidecar stamped `"policy": "original_signal_only_no_reconstruction"`, both SHA-256-hashed, both in the worker record (`OP_SEGMENT`) with local and global PTS, block count, and bridged gaps. Per file: clean/contaminated/bridged totals and the full gap-decision audit (`OP_FILE`). Batch-wide: `layer3_output.json` (schema `spovnob-layer3-output-v1`, carrying `layer2_output_sha256` — the explicit hash link upstream), canonicalized, hashed, recorded (`output_hash` with clean/contaminated totals). These segments are the pipeline's **final verified output** and the input contract for the deferred behavioral-analysis phase.

---

# Part VIII — The Pipeline Runner and the End-to-End Audit Story (`pipeline_runner.py`)

## 8.1 The chain, and the closing verification

`run_pipeline()` is the single production entrypoint:

```
run_gate (determinism + vendoring + resident models)
  → preprocess_batch (Layer 0; RAM preload)
  → run_layer1 (sequential enrollment; freeze)
  → run_layer2 (authoritative pass)
  → run_layer3 (overlap exclusion; clean segments)
  → finalize_pipeline (summary document + pipeline_complete entry)
```

Models load once at the top and are passed down — the Resident Model Policy is enforced by the call graph itself. Per-stage wall-clock durations are collected on a console-only stage clock; **wall time never enters any payload** (it would break bit-reproducibility; the only timestamps in the system live in audit blocks, §3.1). After the manifest context closes, the runner re-opens the file and **re-walks the entire hash chain from disk** (`SessionManifest.verify_chain`), refusing success on any inconsistency: *the run does not count unless its audit trail re-verifies.* The verified entry count is reported.

## 8.2 The summary document

`pipeline_output.json` (schema `spovnob-pipeline-output-v1`), built by the pure `build_pipeline_summary()` and Rule-6-validated before writing: per-file facts (source SHA-256, duration, offset, Silero totals) · enrollment refs (`e_composite_sha256`, `e_anti_sha256`, pool sizes, verified ms, final quality state) · Layer 2 (output hash, `calibration_ref`, kind, both θ values, total HIGH ms) · Layer 3 (output hash, clean/contaminated totals, segment and NaN counts) · the **complete clean-segment listing** (PTS spans local+global, durations, bridged-gap counts, WAV paths and hashes). Its SHA-256 is recorded in the terminal `pipeline_complete` manifest entry alongside the Layer 2 and Layer 3 output hashes — one entry from which every artifact of the run is reachable by hash.

## 8.3 The auditor's replay procedure

1. **Verify the record**: `SessionManifest.verify_chain(manifest)` — any tampering localizes to a line number (§3.3). Confirm the final entry is `pipeline_complete` (truncation check), and compare the final `entry_sha256` against the off-box retained copy if one exists (Part IX, item 10).
2. **Verify the materials**: re-hash the model store against `expected_hashes.json`; compare the recorded `gpu_workload_checksum` hash and gate entries (version pins, platform) against the replay machine's.
3. **Replay**: run `pipeline_runner.py --run` on the same inputs and store, fresh work directory.
4. **Compare**: the `payload_sha256` stream of decision entries; the three output-document hashes; every per-WAV SHA-256. Expected to differ: audit blocks (timestamps, operator), hence `entry_sha256` and the chain values — by design (§3.1).
5. **Path caveat** (Part IX, item 11): the hashed output documents embed absolute `wav_path`/`output_path` strings, so *document-level* hash equality requires replaying with identical paths; under a different work dir, compare the path-independent fields and the per-WAV content hashes, which are path-free.
6. Any divergence localizes to the **first differing payload** in manifest order — the chain gives the auditor a bisection, not just a verdict.

## 8.4 The testing architecture

Every heavy dependency sits behind an **injectable seam**: Layer 2's `Scorer` callable and `scorer_factory`, Layer 3's `OverlapProvider`, the gate's `version_of`. Production binds resident models; the self-tests inject synthetic implementations and drive the *complete decision flows* — calibration, pooling, tiering, edge-trim, NaN voiding, bridging, finalization — on a bare Python 3.10 with **zero pip installs, no torch, no GPU** (the standing test policy; every module asserts `torch not in sys.modules` on exit). What the self-tests deliberately cannot cover — GPU numerics, real model behavior, ffmpeg — is exactly what the environment gate measures *at every startup* (§2.5–2.6). The division is principled: pure logic is proven once per change by tests; environmental truth is proven once per run by the gate.

---

# Part IX — Bench-Validation Register and Known Limitations

Honesty section. Each item is either awaiting first-bench validation on the Ubuntu box, a review-flagged implementation decision the master spec left unspecified, or a structural limitation with a stated mitigation. Numbers are stable identifiers for bench notes.

| # | Item | Status / required action |
|---|---|---|
| 1 | **MAR landmark indices** — width pair `(52, 61)` duplicates the first vertical pair; as written MAR ≈ constant. | **Highest-priority bench task.** Confirm true 2d106det mouth-corner indices against real InsightFace output; correct via manifest-logged parameter change (`params.py` carries the VALIDATE-ON-BENCH banner). |
| 2 | **`face.pose` availability and ordering** — yaw is read as `pose[1]` assuming (pitch, yaw, roll); some InsightFace builds omit pose. | Bench-verify. Absence is non-silent (`pose_unavailable` warning; yaw suspension inert for that video). |
| 3 | **Gate B `threshold_target = 0.70`** vs. real seed-vs-window cosine distributions. | Benchmark on representative session audio; the spec itself anticipates lowering to 0.55–0.65. |
| 4 | **Calibration clamp range `[0.45, 0.75]`** | Benchmark against session score distributions. |
| 5 | **Gate A VAD coverage = 0.50** — implementation-defined quantification of "Silero confirms speech." | Review-flagged default; tune on bench. |
| 6 | **Target absence → plosive semantics** in the window machine (spec silent on the absence case). | Review-flagged conservative choice; revisit if benches show premature window closure. |
| 7 | **Edge demotion: exactly one block, no recursion.** | Review-flagged bounded-correction decision; deliberate. |
| 8 | **Drift quantification** (first-30 s mean vs. previous file's all-HIGH mean). | Review-flagged; informational-only signal. |
| 9 | **Frame/PTS count mismatch** stops pairing at the shorter length (tail unanalyzed, warned). | Acceptable for enrollment (audio layers unaffected); revisit if VFR benches show frequent mismatches. |
| 10 | **The hash chain is self-consistent, not externally anchored** — whole-file rewrite or tail truncation by an attacker with disk access defeats in-file verification. | Procedure: retain each run's final `entry_sha256` (and `pipeline_output.json` hash) off-box. One line in the run log converts these attacks into detectable ones. |
| 11 | **Absolute paths inside hashed output documents** (`wav_path`, `output_path`) couple document hashes to the filesystem layout. | Audit procedure §8.3 step 5; consider path-relativization in a future schema rev (would be `spovnob-*-output-v2`). |
| 12 | **Vendored-store completeness vs. the HF cache** — SpeechBrain's loader can fall back to the global HuggingFace cache for auxiliary files (`label_encoder.txt`, observed in the WSL2 dry run, Gotchas §3); the cache is outside `expected_hashes.json` coverage. | Hardening: ensure the vendored directories contain *every* repo file before freezing hashes, so the loader never needs the cache; treat a cache-dependent load as a vendoring defect. |
| 13 | **Single-batch length groups** in the ECAPA scorer run at natural (input-determined) batch size. | Reproducible by construction (§2.3); recorded for completeness. |
| 14 | **Behavioral/paralinguistic analysis** | Deferred — requires a separate design phase. Layer 3's clean segments are its declared input contract. WavLM and HuBERT are fully removed (models: five). |

---

# Appendix A — Complete Parameter Reference

**Environment gate (`environment_gate.py`) — architectural constants**

| Constant | Value | Role |
|---|---|---|
| `ECAPA_BATCH_WINDOWS` | 256 | fixed Layer 2 inference batch (FP reduction order) |
| `VISUAL_BATCH_FRAMES` | 32 | fixed YOLO frame batch |
| `TORCH_NUM_THREADS` | 8 | fixed CPU intra-op threads |
| `GLOBAL_SEED` | 20260611 | defense-in-depth seed |
| `INSIGHTFACE_DET_SIZE` | (640, 640) | detector input size |
| `PYANNOTE_OVD_HYPERPARAMS` | `min_duration_on=0.0, min_duration_off=0.0` | OVD's internal smoothing disabled |
| `EXPECTED_CUDA_VERSION` / `EXPECTED_PYTHON_PREFIX` | "12.1" / "3.10." | platform pins |
| `DETERMINISM_ENV` | `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | set at import time |

**Layer 0 (`layer0_preprocessor.py`)**

| Constant | Value | Role |
|---|---|---|
| `SAMPLE_RATE` | 16000 | pipeline-wide; 1 ms = 16 samples exactly |
| `SILERO_WINDOW_SAMPLES` | 512 | 32 ms VAD windows |
| `SILERO_THRESHOLD` | 0.50 | speech decision |
| `SILERO_MIN_SILENCE_MS` / `SILERO_MIN_SPEECH_MS` / `SILERO_SPEECH_PAD_MS` | 100 / 250 / 30 | merge / drop (pre-pad) / pad |
| `EXTRACT_WORKERS` | 8 | parallel ffmpeg threads |

**Layer 1 (`layer1_enrollment/params.py`, frozen dataclass `EnrollmentParams`)**

| Parameter | Default | Role |
|---|---|---|
| `face_reid_threshold` | 0.40 | ArcFace cosine for target lock |
| `reid_warning_floor` | 0.50 | guardrail 6 running-mean warning |
| `mar_on` / `mar_off` | 0.55 / 0.40 | MAR hysteresis |
| `plosive_ms` | 500 | closure grace timer |
| `vad_tol_ms` | 50 | Silero/PTS alignment tolerance |
| `min_enroll_len_ms` | 2000 | candidate floor |
| `seed_min_ms` | 3000 | seed floor (no maximum) |
| `int_lips_closed_frac` | 0.80 | Gate A interviewer-silent fraction |
| `threshold_target` | 0.70 | Gate B (≥) — bench item 3 |
| `threshold_anti` | 0.50 | Gate C ceiling (≤) |
| `margin_minimum` | 0.15 | Gate C margin (≥) |
| `mtrap_sim_max` | 0.60 | M-Trap discard (>) |
| `anti_contam_warning` / `anti_contam_halt` | 0.45 / 0.60 | guardrail 8 (also re-used by Layer 2 sanity) |
| `pool_var_warning` | 0.05 | anti-pool variance increase warning |
| `yaw_max_degrees` | 35.0 | MAR suspension |
| `ema_span` | 5 | causal EMA (α = 1/3), pre-seeded |
| `upper_inner_lip` / `lower_inner_lip` / `mouth_width_pair` | (52,53,54) / (61,62,63) / (52,61) | bench item 1 |
| `insightface_min_det_score` | 0.50 | guardrail 5 |
| `yolo_min_conf` | 0.30 | person gate |
| `silence_stride` | 1 | optional efficiency rule (off) |
| `encode_max_ms` / `encode_overlap_ms` | 60000 / 2000 | single-pass cap / chunk overlap |
| `gate_a_vad_min_coverage` | 0.50 | bench item 5 |
| `trackb_window_ms` / `trackb_min_spacing_ms` | 2000 / 2000 | Track B window / dedupe |
| `click_overlap_max_frac` | 0.20 | guardrail 1 proxy |
| `strong_ms` / `strong_ms_no_anti` / `marginal_ms` | 45000 / 60000 / 20000 | quality states |
| `variance_high` | 0.05 | STRONG variance ceiling |
| `CLICK_MATCH_MAX_GAP_MS` (enrollment.py) | 200 | click-to-frame match tolerance |

**Layer 2 (`Layer2Params`)** — architectural: `window_ms` 5000 · `hop_ms` 1000 · `block_ms` 1000 · `silero_skip_floor` 0.20 · `evidence_floor` 0.20 · `edge_fine_window_ms` 2000 · `edge_fine_hop_ms` 250 · `edge_scan_span_ms` 2000 · `edge_min_fine_window_ms` 1000 · `edge_max_trim_ms` 750. Calibration: `genuine_quantile` 0.10 · `impostor_safety_margin` 0.05 · clamps 0.45/0.75 · `theta_med_step` 0.15 · `theta_med_floor` 0.30 · `min_calibration_windows` 10 · fallbacks 0.60/0.40 · `no_anti_theta_floor` 0.55 · `margin_minimum` 0.15 (not calibrated). Diagnostics: contamination 0.45/0.60 · ratio 0.25/0.10 · drift 30000 ms / 0.10.

**Layer 3 (`Layer3Params`)** — `merge_gap_ms` 400 (operator-tunable via manifest, strongly discouraged) · `gap_dominance_guard` True.

# Appendix B — Manifest Operation Vocabulary

**Core (`session_manifest.Operation`):** `batch_init` · `model_checksum` · `determinism_check` · `parameter_modified` · `enrollment_vector` · `enrollment_discard` · `calibration` · `video_gap` · `drift_notice` · `warning` · `blocking_halt` · `destructive_op` · `output_hash` · `worker_log_merged`.

**Module-defined:** Layer 0: `layer0_file`. Layer 1: `layer1_init`, `layer1_seed`, `layer1_video_scan`, `layer1_quality`, `layer1_freeze`. Layer 2: `layer2_init`, `layer2_block`, `layer2_edge_trim`, `layer2_file_summary`, `layer2_window_skipped`. Layer 3: `layer3_init`, `layer3_ovd_regions`, `layer3_nan_block`, `layer3_segment`, `layer3_file_summary`. Runner: `pipeline_complete` (terminal).

Vector kinds (Layer 1 payloads): `seed`, `track_a_window`, `anti_track_b`, `anti_track_c`. Calibration kinds: `DERIVED`, `DERIVED_NO_ANTI`, `FALLBACK_DEFAULTS`. Tiers: `HIGH`, `MEDIUM`, `SUB_THRESHOLD`, `REJECT`, `SKIPPED_NONSPEECH`. End reasons: `plosive_expiry`, `interviewer_interjection`, `end_of_video`. Gap-decision reasons: `bridged`, `gap_too_long`, `overlap_in_gap`, `interviewer_evidence_in_gap`.

# Appendix C — Artifact Map, Dependency Graph, and Pinned Stack

**Work directory after a run:**

```
<work_dir>/
├── *.wav                      # Layer 0 full-file 16kHz extractions
├── enroll/                    # Layer 1: per-window WAV + JSON sidecar (seed, track_a, anti_b/c)
├── layer2/
│   ├── worker_NNN.jsonl       # per-file worker logs (hashed at merge)
│   └── layer2_output.json     # authoritative, hashed
├── layer3/
│   ├── worker_NNN.jsonl
│   ├── clean/                 # FINAL OUTPUT: clean_NNN_start_end.wav + .json sidecars
│   └── layer3_output.json     # hashed
└── pipeline_output.json       # batch summary, hashed in pipeline_complete
<manifest>.jsonl               # the hash-chained session manifest (single writer)
<model_store>/                 # silero-vad/ · speechbrain-spkrec-ecapa-voxceleb/ ·
                               # yolov8/ · insightface/ · pyannote-segmentation-3.0/ ·
                               # expected_hashes.json (chmod 444)
```

**Pinned stack** (mirrored by the gate's `PINNED_VERSIONS`; the gate halts on any mismatch): torch 2.1.2+cu121 · torchaudio 2.1.2+cu121 · torchvision 0.16.2+cu121 · numpy 1.26.4 · speechbrain 1.0.0 · pyannote.audio 3.1.1 · huggingface-hub 0.21.3 · ultralytics 8.1.47 · insightface 0.7.3 · onnxruntime-gpu 1.17.1 (CUDA-12 feed **only** — Gotchas §2) · onnx 1.15.0 · opencv-python-headless 4.9.0.80 · ffmpeg-python 0.2.0 · soundfile 0.12.1 · librosa 0.10.1 · scipy 1.12.0 · numba 0.59.1 · llvmlite 0.42.0 · lightning 2.1.4 · torchmetrics 1.3.1 · scikit-learn 1.4.1.post1 · scikit-image 0.22.0 · tqdm 4.66.2. Silero VAD is vendored, not pip-installed, pinned to commit `915dd3d639b8333a52e001af095f87c5b7f1e0ac` (commit, not tag — the upstream tag was observed to move).

---

*End of SPOVNOB_TECHNICAL_DEEP_DIVE.md v1.0 — 2026-06-12. Maintenance rule: this document is downstream of the code; any change to a constant, predicate, or schema named here must update this file in the same commit, and Part IX is the only section permitted to describe behavior the code does not yet have.*
