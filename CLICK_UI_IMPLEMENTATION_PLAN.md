# SPOVNOB Clicking UI — Implementation Plan

**Date drafted:** 2026-06-12
**Status:** Implemented 2026-06-13 (`click_ui.py`) — see "As-built deviations" below
**Motivation:** The current workflow requires the operator to manually determine correct `speaking_click` and `anti_click` timestamps by inspecting video files externally, then editing `clicks.json` by hand. This is error-prone and requires domain knowledge of the guardrail constraints. This plan describes a purpose-built web UI that makes the clicking process visual, validated, and self-documenting.

---

## Background: what the clicking step does

Before Layer 1 runs, the operator must supply two timestamps in `session/clicks.json`:

- **`speaking_click`**: a PTS (milliseconds) + pixel coordinate on the target speaker's face, during a frame where they are clearly speaking. This seeds the biometric lock (`F_target`) and the first enrollment window.
- **`anti_click`** *(optional)*: a PTS + pixel coordinate on the interviewer's face, during a frame where the interviewer is **on camera but not speaking**, while Silero detects audio energy. This seeds the anti-profile (`E_anti`).

Three guardrails fire at the speaking_click; three more fire at the anti_click. Getting the timestamps wrong produces re-click errors that are hard to diagnose without knowing the video content. The UI solves this by running the same guardrail checks in real time as the operator scrubs.

**Known footage-specific challenge (NT-clip27):** The interviewer is only visible in a two-shot from ~12 000 ms to ~15 000 ms. Before 12 000 ms, only the target is present — clicking there for `anti_click` fires `anti_click_matches_target`. The UI's face-count timeline strip makes this visible immediately.

---

## Suggestions being implemented

| # | Suggestion | Status |
|---|---|---|
| 1 | Clicking UI (video scrubber + face overlay + click registration) | This plan |
| 2 | Update SPOVNOB_TECHNICAL_DEEP_DIVE.md with bench findings | **Done 2026-06-12** |
| 3 | Expose `anti_click` as optional from the UI (`NO_ANTI_PROFILE` path) | Bundled into this plan (toggle) |
| 4 | UI shows face detection overlays so operator can find footage-specific valid timestamps | Bundled into this plan (timeline strip + live overlay) |

---

## Architecture

### Single file: `click_ui.py`

A standalone Flask application. Run with:

```bash
python click_ui.py <video_file_path> [--port 5050] [--work-dir session/]
```

Opens `http://localhost:5050` automatically. Requires only **InsightFace buffalo_l** and **Silero VAD** — not the full 5-model stack. Startup time ~4 s vs ~30 s for the full gate.

### Server-side components

**Startup (once)**

1. Run ffprobe to extract all PTS values → `pts_list` (same method as `layer0_preprocessor.py`)
2. Load InsightFace `buffalo_l` + Silero VAD from the vendored model store
3. Run a **face pre-scan**: analyze every frame (batched at `VISUAL_BATCH_FRAMES=32`), cache per-frame result: `{pts_ms, face_count, faces: [{bbox, embedding_hash, mar, cx, cy}]}`
4. Identify **face-count transitions** (1-face → 2-face → 1-face) and expose as timeline metadata to the frontend

**Endpoints**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page UI HTML |
| `GET` | `/frame?pts_ms=X` | Returns a JPEG of the nearest analyzed frame with: face bounding boxes drawn (green for detected), MAR value labelled per face, Silero VAD state as border (green=speech, grey=silence), click targets highlighted if registered |
| `GET` | `/timeline` | Returns JSON array of `{pts_ms, face_count, vad_speech}` for every frame — used to draw the timeline strips |
| `POST` | `/click` | Body: `{pts_ms, x, y, type: "speaking" | "anti"}`. Runs the same guardrail logic as `enrollment.py`. Returns `{ok: true, details: {...}}` or `{ok: false, reason: "guardrail_name", details: {...}}` |
| `POST` | `/export` | Writes `<work_dir>/clicks.json` and returns the JSON. Validates that `speaking_click` is registered; if anti_click toggle is off, omits `anti_click` key entirely (the pipeline interprets absence as `NO_ANTI_PROFILE`). |

### Client-side (single HTML page, inline JS ~150 lines)

**Layout**

```
┌─────────────────────────────────────────────────────┐
│  Video frame canvas (native resolution, max 800px)  │
│  [Face bboxes drawn over frame]                     │
├─────────────────────────────────────────────────────┤
│  PTS scrubber ──────────────────── [00:00 / 02:45]  │
├─────────────────────────────────────────────────────┤
│  [Face count timeline: 1-face=blue, 2-face=orange]  │
│  [VAD timeline:        speech=green, silence=grey]  │
├─────────────────────────────────────────────────────┤
│  Speaking click: [42000ms  (405, 213)]  [Clear]     │
│  Anti click:     [12000ms  (238, 150)]  [Clear]     │
│  [ ] Require anti-click  (unchecked = NO_ANTI path) │
├─────────────────────────────────────────────────────┤
│  Status: ✓ Speaking click valid                     │
│          ✓ Anti click valid                         │
│  [Export clicks.json]                               │
└─────────────────────────────────────────────────────┘
```

**Interaction**

1. Scrubber drag → debounced fetch of `/frame?pts_ms=X` → update canvas image
2. Canvas click → compute click coordinates relative to image → POST to `/click` with current PTS and `type` (determined by which "mode" button is active: "speaking" or "anti")
3. Guardrail result shown immediately in status box in plain English:
   - `anti_click_matches_target` → "Anti-click face matches the target speaker — choose a different face or timestamp"
   - `target_lips_open_at_anti_click` → "Target's lips appear open at this timestamp — target may be speaking; choose a timestamp where target is listening"
   - `no_speech_at_anti_click` → "No audio energy detected at this timestamp — Silero shows silence"
   - `click_outside_speaking_window` → "Target was not in an active speaking window at this timestamp"
   - etc.
4. Anti-click toggle checkbox → when unchecked, anti_click registration is hidden and export skips the `anti_click` key
5. "Export clicks.json" button → POST `/export` → shows success + file path

---

## Guardrail checks in `/click`

The `/click` endpoint replicates the logic from `enrollment.py` directly, using the pre-scanned frame cache and the loaded models. Checks run in the same order as the production pipeline:

**For `type=speaking`:**
1. Face present at PTS within `CLICK_MATCH_MAX_GAP_MS=200` ms
2. Face detection score ≥ `insightface_min_det_score=0.50`
3. Silero VAD shows speech at PTS ± `vad_tol_ms=50` ms
4. *(Does not run full window machine — shows a preview: "This frame has MAR=X, which is {'above' if X >= mar_on else 'below'} the mar_on=0.15 threshold")*

**For `type=anti`:**
1. Face present at PTS
2. Clicked face does NOT match `F_target` (cosine < `face_reid_threshold=0.40`) → `anti_click_matches_target`
3. Target's lips are NOT clearly open at PTS (target MAR < `mar_on=0.15`) → `target_lips_open_at_anti_click`
4. Silero VAD shows speech at PTS → `no_speech_at_anti_click`

Note: `F_target` is not known until the speaking_click is registered. The speaking_click must be set before the anti_click can be validated. The UI enforces this ordering.

---

## NO_ANTI_PROFILE path (Suggestion 3)

When the "Require anti-click" checkbox is **unchecked**:

- The anti_click panel is hidden (greyed out)
- `/export` writes `clicks.json` with only `speaking_click`
- The pipeline detects the absent `anti_click` key and routes to the `NO_ANTI_PROFILE` branch
- In `NO_ANTI_PROFILE` mode: `strong_ms_no_anti=60000` (60 s threshold vs the normal 45 s), quality state logic uses `DERIVED_NO_ANTI` calibration

The UI shows a warning when the checkbox is unchecked: "No anti-click: STRONG quality threshold raised to 60 s of verified audio. Use this only when the interviewer is never on camera."

---

## Face-count timeline strip (Suggestion 4)

The pre-scan caches `face_count` per frame. The frontend draws two thin timeline strips below the scrubber:

- **Face count strip**: each frame is a 2 px wide column, coloured blue (1 face) or orange (2+ faces) or white (0 faces). The operator can immediately see when the interviewer enters the shot.
- **VAD strip**: green (Silero speech) or grey (silence).

The scrubber thumb overlays both strips, so the operator sees exactly which region they are in. Hovering over the strip shows a tooltip with the exact PTS and face count.

This directly solves the NT-clip27 problem: the orange two-face region from 12 000–15 000 ms is immediately visible, and the operator knows to place the anti_click there.

---

## File layout

```
Audio_Diarization/
├── click_ui.py                  # The complete Flask app (new)
├── click_ui_frontend.html       # Embedded in click_ui.py as a string, or separate
├── session/
│   └── clicks.json              # Written by /export
```

The UI has no additional pip dependencies beyond what the pipeline already uses: Flask (add to requirements.txt), InsightFace, Silero (torch.hub or vendored), OpenCV, ffprobe (system).

Flask is the only new dependency. Add to `requirements.txt`:
```
flask>=3.0.0
```

---

## Implementation order

1. **`click_ui.py` server skeleton**: Flask routes, ffprobe PTS extraction, InsightFace + Silero loader (subset of `environment_gate.py`'s model-load logic)
2. **Face pre-scan + cache**: replicate the YOLO+InsightFace scan loop from `enrollment.py` scan phase, storing per-frame results in a dict
3. **`/frame` endpoint**: render frame + draw bboxes + MAR labels using OpenCV `cv2.rectangle` / `cv2.putText`, return as JPEG
4. **`/timeline` endpoint**: serialize the face-count and VAD arrays
5. **Frontend HTML**: scrubber, canvas, timeline strips, click mode buttons, status box
6. **`/click` endpoint**: guardrail logic using pre-cached frame data
7. **`/export` endpoint**: write clicks.json
8. **Anti_click toggle**: conditional rendering + export logic
9. **UX polish**: error messages in plain English, keyboard shortcuts (left/right arrow = ±1 frame, `s` = set speaking_click, `a` = set anti_click)

---

## Known constraints and design decisions

- **No video streaming**: the UI renders individual frames on demand, not a continuous video stream. This gives exact PTS control and avoids browser video codec issues with interview footage.
- **Pre-scan is blocking at startup**: for a 4×~90s batch at 30fps, ~10 800 frames. At InsightFace's ~3 ms/frame throughput (GPU), the scan takes ~33 s. Show a progress bar during startup. The scan only uses the first video (`file_index=0`) since that is the file that contains both the speaking_click and anti_click for the current session design.
- **F_target is computed from speaking_click**: once the operator clicks the target, the server embeds that face and stores it as `F_target` in memory. Subsequent anti_click validations use this embedding. Clearing the speaking_click also clears `F_target`.
- **Session output location**: all output stays inside `~/Documents/Audio_Diarization/session/` per project policy.
- **Air-gap safe**: the UI uses no CDN resources — all JS/CSS is inline.

---

## Future extensions (out of scope for v1)

- Multi-video support: allow clicking on a different `file_index` for the anti_click
- MAR trace playback: show the MAR value over time as the operator scrubs, to visualize window boundaries before running Layer 1
- Click history: show previous session's clicks.json as pre-populated defaults for the same target

---

## As-built deviations (2026-06-13)

Approved direction: the UI must *perfectly mirror* the production pipeline's
constraints. The implementation therefore upgrades several plan items:

1. **Full speaking-click validation, not a MAR preview.** The UI imports
   `enrollment._face_at_click/_build_obs/_run_machine/_match_face/_mean_embedding`
   and runs the real `WindowMachine` over the pre-scanned frames, so
   `click_outside_speaking_window`, `seed_too_short` (guardrail 2) and
   `overlap_at_speaking_click` (guardrail 1) are validated identically to
   `run_layer1`. `F_target` is the same refined seed-span mean embedding.
2. **YOLO stays in the vision stack.** The pre-scan IS `vision.scan_video`
   (YOLO person gate → InsightFace, det-score guardrail 5 included), and the
   audio/VAD side IS Layer 0's `extract_audio → silero_window_probs →
   segments_from_window_probs`. Nothing is re-implemented.
3. **Full embeddings cached, not hashes** — the anti-click cosine checks need
   them. Pre-scan results (frames + embeddings + Silero segments + PTS list +
   display JPEGs) are cached under `<work_dir>/ui_cache/`, keyed by video
   SHA-256 + EnrollmentParams payload + batch constants + device + the model
   store's `expected_hashes.json` digest. Warm starts load no models (~2 s).
4. **Guardrail reason strings are byte-identical to enrollment.py's manifest
   entries** (`no_audio_energy_at_anti_click`, not the plan's
   `no_speech_at_anti_click`), so UI feedback correlates with pipeline logs.
5. **Frames are served raw; overlays draw client-side** from `/timeline`
   metadata (bboxes, MAR, det score, yaw, VAD). Display JPEGs are extracted
   once by ffmpeg (`-vsync 0`, ≤800 px) — scrubbing is instant and the plan's
   `/frame?pts_ms=` server-side compositing endpoint is unnecessary.
6. **`/export` writes `file_index: 0`** on both clicks (required by
   `load_clicks`) and self-verifies by round-tripping the written file
   through the production parser.
7. **Startup progress** is staged terminal logging rather than a browser
   progress bar (`scan_video` exposes no per-frame hook; forking it for a
   progress callback would violate the parity rule).
8. **`--cpu` flag (development only)** — production validation runs on the
   Ubuntu/CUDA bench; CPU runs use a separate cache key.
9. **Self-test:** `python3 click_ui.py --selftest` is stdlib-only (no flask /
   torch / cv2 / ffmpeg) and drives the imported production functions over
   synthetic frames: every guardrail reason, the ordering rule, export
   round-trips, and cache-key behavior are asserted.
10. Flask is pinned (`flask==3.0.2`) in a new "operator tooling" section of
    requirements.txt — outside `environment_gate.PINNED_VERSIONS` because the
    UI is outside the chain of custody (clicks.json is re-validated by
    Layer 1; the UI writes no manifest entries).
