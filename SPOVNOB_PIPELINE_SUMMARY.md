# SPOVNOB — Forensic Audio Diarization Pipeline · Team Summary

**Last updated:** 2026-06-12 · **Status:** All modules complete (behavioral analysis deferred)  
**Deployment target:** Ubuntu 22.04 LTS · NVIDIA RTX 6000 Ada (48 GB VRAM) · Python 3.10 · CUDA 12.1

---

## What SPOVNOB Does (One Paragraph)

SPOVNOB is a forensic speaker diarization pipeline. It takes a batch of 5–15 interview-style video clips (5–10 min each, one continuous session), and outputs **clean, PTS-timestamped audio segments containing only the target speaker's uncontaminated speech** — nothing synthesized, nothing repaired, nothing guessed. It is fully air-gapped, fully deterministic (bit-identical outputs for identical inputs on any re-run), and records every decision, parameter, and discard in an append-only, hash-chained session manifest. The output is the input contract for a future behavioral/paralinguistic analysis phase.

---

## Pipeline Architecture at a Glance

```
RAW VIDEO BATCH
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PRE-FLIGHT                                                     │
│  • Session Manifest initialized (append-only, hash-chained)     │
│  • Environment Gate: SHA-256 verify all 5 model weight sets     │
│    ↳ Checksum mismatch → BLOCKING HALT before any processing    │
│  • CUDA determinism enforced (4 mandatory constants + float32)  │
│  • All 5 models loaded once, stay resident for entire batch     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0 — Foundational Extraction                              │
│  • FFmpeg strips audio → 16 kHz mono WAV, PTS-true timestamps   │
│    (no frame-counting; protects against VFR clock desync)       │
│  • Silero VAD (CPU, ~1 MB) produces speech segment map          │
│    ↳ Non-destructive — audio is NEVER modified, only labelled   │
│  • Full batch audio preloaded into RAM for zero-latency reads   │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Visual-Anchored Enrollment            (sequential)   │
│                                                                 │
│  Operator provides 2 clicks on Video 1:                         │
│    Click 1 (mandatory) — target face while clearly speaking     │
│    Click 2 (optional)  — interviewer face while lips closed     │
│                                                                 │
│  Phase 1: YOLOv8 + InsightFace lock onto target face (F_target) │
│           ECAPA-TDNN extracts seed d-vector E_seed (3–30 s)     │
│                                                                 │
│  Phase 2: Anti-profile built (E_anti)                           │
│    • Track C — operator click on interviewer (high confidence)  │
│    • Track B — auto-collected: non-target face, lips closed,    │
│      audio present, M-Trap guard prevents target self-enrollment│
│                                                                 │
│  Phase 3: Full-video Dual-Track Visual Confirmation Loop        │
│    • Track A — target face present + lips moving → E_window     │
│    • Every E_window passes Triple Validation Gate:              │
│        Gate A: Silero confirms speech + interviewer lips closed  │
│        Gate B: cosine_sim(window, E_seed) ≥ threshold_target    │
│        Gate C: cosine_sim(window, E_anti) ≤ threshold_anti      │
│                AND sim(target)−sim(anti) ≥ margin_minimum       │
│    • Accepted windows → cumulative pool → E_composite (running  │
│      duration-weighted mean-pool, recalculated after each video)│
│                                                                 │
│  Enrollment quality gates after each video:                     │
│    STRONG (≥ 45 s, low variance) → E_composite promoted, batch  │
│      proceeds to authoritative Layer 2 pass                     │
│    MARGINAL (20–45 s or high variance) → second pass, carry fwd │
│    INSUFFICIENT (< 20 s) → file flagged PENDING, carry forward  │
│    CRITICAL FAILURE (all videos exhausted, still insufficient)  │
│      → terminal halt, operator required                         │
│                                                                 │
│  E_composite FROZEN after Layer 1 completes. Never modified.    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Pure ECAPA Sliding-Window Scanning    (parallel)     │
│                                                                 │
│  Input: frozen E_composite + E_anti + raw 16 kHz audio          │
│                                                                 │
│  1. Sanity check: sim(E_composite, E_anti) → warning or halt    │
│  2. Deterministic threshold calibration (per session):          │
│       theta_high = max(q10 of genuine LOO scores,               │
│                        max impostor score + 0.05), clamped      │
│       theta_med  = theta_high − 0.15                            │
│  3. Slide 5 s window (1 s hop) across raw audio in batches      │
│     of 256 → ECAPA d-vector per window → cosine similarity      │
│     S_target vs E_composite, S_interviewer vs E_anti            │
│     (Windows < 20% Silero speech overlap are skipped, logged)   │
│  4. Median pooling per 1 s block (strips boundary artifacts)    │
│  5. Confidence tiering:                                         │
│       HIGH   → S_target > theta_high AND margin > 0.15         │
│       MEDIUM → theta_med – theta_high (human review only)       │
│       < 0.20 → rejected, timestamp logged                       │
│  6. Edge-trim refinement: 2 s windows at 250 ms hop trim HIGH   │
│     block edges inward only — never extend (forensically safe)  │
│  7. Activity ratio check vs Silero total speech (advisory)      │
│                                                                 │
│  Single authoritative pass after Layer 1 completes across ALL   │
│  videos — every file scored against final E_composite_final.    │
│  SHA-256 output hash recorded in manifest.                      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Contamination Flagging & Temporal Smoothing          │
│                                                                 │
│  Input: raw un-smoothed HIGH-confidence blocks from Layer 2     │
│                                                                 │
│  • PyAnnote OVD (Overlapping Voice Detection) checks every      │
│    block and every inter-block gap for simultaneous speech      │
│  • NaN-Only Exclusion Policy: overlap detected → entire block   │
│    logged as NaN (contaminated) and permanently excluded        │
│    ↳ NO separation models ever. Hallucinated audio is           │
│      forensically indefensible.                                 │
│  • Clean blocks: gaps < 400 ms bridged (natural plosives only,  │
│    and only across confirmed-clean gaps)                        │
│                                                                 │
│  Output: PTS-stamped, target-isolated CLEAN speech segments     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  FINAL OUTPUT
  Clean, verified, original-signal audio segments of the target
  speaker. Ready for downstream behavioral analysis (deferred).
```

---

## Key Design Principles

| Principle | What it means in practice |
|---|---|
| **No synthesis, ever** | Contaminated windows are excluded, not repaired. Separation models are explicitly forbidden. |
| **Visual anchoring** | Identity is locked visually (InsightFace face embedding) before any acoustic enrollment begins. Acoustic models never decide who someone is from scratch. |
| **Static enrollment** | E_composite is frozen after Layer 1. No feedback loop in Layer 2. Same inputs → bit-identical outputs on any date, any machine. |
| **Air-gapped & offline** | All 5 model weights pre-staged locally with SHA-256 pins. `HF_HUB_OFFLINE=1`. Any checksum mismatch is a blocking halt. |
| **Append-only audit trail** | Session manifest is never modified after a write. Every decision, threshold override, discard, and operator action has an immutable record with UTC timestamp. |
| **PTS-only timestamps** | All times are integer milliseconds derived from Presentation Timestamps. No frame counting. No floats for time anywhere in the codebase. |
| **Conservative by default** | MEDIUM-confidence windows are never auto-promoted. Ambiguous margins are discarded. Missed speech is preferred over contaminated speech. |

---

## Models Used

| Model | Role | Runs on |
|---|---|---|
| **Silero VAD** (~1 MB) | Speech / silence segment map | CPU |
| **YOLOv8** | Person detection per frame | GPU |
| **InsightFace buffalo_l** | Face biometric lock (F_target) | GPU |
| **ECAPA-TDNN C=1024** (SpeechBrain) | Speaker d-vector extraction and scoring | GPU |
| **PyAnnote OVD** | Overlapping speech detection | GPU |

All five are loaded once at batch start and held resident. No model is unloaded between files.

---

## Operator Input Required

Only **2 clicks** on Video 1, before processing starts. A purpose-built **Click UI** (`click_ui.py`, Flask web app) assists the operator with live face overlays, MAR readouts, face-count timeline strips, and real-time guardrail feedback — so invalid clicks are caught immediately with plain-English explanations.

| Click | Target | When | Required? |
|---|---|---|---|
| **Speaking Click** | Target speaker | While target is visibly speaking | **Mandatory** |
| **Anti-Profile Click** | Interviewer | Interviewer on camera, lips closed, audio present | Optional (NO_ANTI path if omitted) |

After Video 1, all anchors (F_target, E_seed, E_anti, cumulative pool) propagate automatically across all remaining videos. No further clicks are needed unless a guardrail fails.

---

## Module Map

| File | Role | Status |
|---|---|---|
| `session_manifest.py` | Append-only, hash-chained audit log | ✅ Complete |
| `environment_gate.py` | SHA-256 model vendoring gate + CUDA determinism setup | ✅ Complete |
| `layer0_preprocessor.py` | FFmpeg PTS extraction + Silero VAD segment map | ✅ Complete |
| `layer1_enrollment/` | Visual enrollment, E_seed, E_anti, E_composite | ✅ Complete |
| `layer2_tracker.py` | ECAPA sliding-window scanning + threshold calibration | ✅ Complete |
| `layer3_contamination.py` | PyAnnote OVD overlap exclusion + temporal smoothing | ✅ Complete |
| `pipeline_runner.py` | Production batch orchestrator (entrypoint) | ✅ Complete |
| `click_ui.py` | Operator clicking web UI (Flask) | 🔲 Planned |
| Behavioral analysis | Downstream paralinguistic analysis | ⏸ Deferred |

---

*SPOVNOB — Revision 3.1 · Summary prepared 2026-06-12*
