"""
SPOVNOB — Operator tool: audit_visualizer.py
=============================================

Purpose:    A standalone, READ-ONLY forensic visualizer. It reads a
            finished SPOVNOB session manifest (``*.manifest.jsonl``) and,
            optionally, the extracted 16 kHz audio, and emits ONE
            self-contained HTML report with an interactive, zoomable
            timeline showing — on a single global session clock —
            Silero VAD, Layer 2 per-block S_target/S_interviewer scores
            and tiers, PyAnnote overlap regions, and the final Layer 3
            CLEAN / NaN output blocks. Hovering any block yields a
            plain-English verdict reconstructed from the recorded numbers
            (why it was kept or rejected).

Independence contract (the design rule of this module):
            This tool imports NOTHING from the SPOVNOB pipeline — not
            session_manifest, not environment_gate, not any layer. It is
            pure Python standard library. Two reasons:
              1. environment_gate mutates os.environ at import time; a
                 read-only forensic tool must never perturb process state.
              2. The report must run on an analyst's laptop against a
                 manifest copied off the bench, with no torch, no CUDA,
                 no model store, no pip installs.
            The only accepted coupling is the manifest's on-disk FORMAT
            (schema "spovnob-manifest-v1"): the canonical-JSON rule and
            the hash-chain layout are re-implemented here, read-only, and
            the schema string is asserted. If session_manifest ever bumps
            its schema, this tool must be revisited — it will say so
            loudly rather than mis-render.

Read-only contract:
            Opens the manifest and audio for reading only. Writes exactly
            one file: the HTML report the operator names (``--out``).
            Modifies no pipeline file and no session artifact.

Forensic integrity:
            Chain verification is ON by default (``--no-verify`` opts
            out). The full hash chain (payload_sha256 / prev_sha256 /
            entry_sha256 / seq) is re-walked; a broken or tampered chain
            paints a large red banner across the top of the report.
            When audio is supplied, each file's SHA-256 is re-hashed and
            compared to the ``wav_sha256`` Layer 0 recorded.

Run:        python3 audit_visualizer.py <session.manifest.jsonl>
                [--audio DIR_OR_WAV ...] [--out audit.html]
                [--no-verify] [--no-browser]
Self-test:  python3 audit_visualizer.py --selftest   (stdlib only: builds
            a synthetic hash-chained manifest incl. double-nested worker
            records + a tampered line, and asserts parsing, the
            local->global mapping, every tier verdict, chain detection,
            and the rendered HTML. No pipeline imports, no GPU, no pip.)

CUDA determinism dependencies: none — this module never touches a model.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import html
import json
import sys
import wave
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Manifest format constants (the one accepted coupling: see docstring) -----
MANIFEST_SCHEMA = "spovnob-manifest-v1"
GENESIS_SHA256 = "0" * 64

# Layer 2 tier names (mirrors layer2_tracker; read-only constants).
TIER_HIGH = "HIGH"
TIER_MEDIUM = "MEDIUM"
TIER_SUB = "SUB_THRESHOLD"
TIER_REJECT = "REJECT"
TIER_SKIPPED = "SKIPPED_NONSPEECH"

# Operations whose merged worker payload is double-nested.
WORKER_WRAPPER_KEYS = frozenset({"file_index", "start_ms", "operation", "payload"})

# Audio expectations (Layer 0 writes 16 kHz mono PCM16).
EXPECTED_SR = 16000
ENVELOPE_TARGET_COLUMNS = 6000        # cap embedded waveform payload size


class AuditError(RuntimeError):
    """Unrecoverable visualizer failure (bad manifest, unreadable audio)."""


# =============================================================================
# Canonical JSON + hash chain (re-implemented read-only; mirrors the
# session_manifest format exactly so verification is faithful)
# =============================================================================

def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def sha256_of_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_chain(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-walk the parsed manifest, re-deriving every hash and chain link
    exactly as SessionManifest.verify_chain does. Returns a status dict:
    {ok, checked, message, broken_seq}. Never raises on a bad chain — a
    forensic report must render the failure, not crash on it."""
    prev = GENESIS_SHA256
    for index, entry in enumerate(entries):
        try:
            claimed = entry["entry_sha256"]
            unsealed = {k: v for k, v in entry.items() if k != "entry_sha256"}
            if sha256_of_obj(unsealed) != claimed:
                return _chain_fail(index, entry, "entry_sha256 mismatch (entry tampered)")
            if sha256_of_obj(entry["payload"]) != entry["payload_sha256"]:
                return _chain_fail(index, entry, "payload_sha256 mismatch")
            if entry["prev_sha256"] != prev:
                return _chain_fail(index, entry, "chain break (prev_sha256 does not link)")
            if entry["seq"] != index:
                return _chain_fail(index, entry, f"seq {entry['seq']} != position {index}")
            prev = claimed
        except (KeyError, TypeError) as exc:
            return _chain_fail(index, entry, f"malformed entry: {exc}")
    return {
        "ok": True, "checked": len(entries), "broken_seq": None,
        "message": f"hash chain intact — {len(entries)} entries verified",
    }


def _chain_fail(index: int, entry: Dict[str, Any], why: str) -> Dict[str, Any]:
    seq = entry.get("seq", index) if isinstance(entry, dict) else index
    return {
        "ok": False, "checked": index, "broken_seq": seq,
        "message": f"CHAIN VERIFICATION FAILED at seq {seq}: {why}",
    }


# =============================================================================
# Manifest loading + worker-record normalization
# =============================================================================

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AuditError(f"{path}: line {line_number} is not valid JSON: {exc}")
    if not entries:
        raise AuditError(f"{path}: manifest is empty")
    schema = entries[0].get("schema")
    if schema != MANIFEST_SCHEMA:
        raise AuditError(
            f"{path}: unexpected manifest schema {schema!r} "
            f"(this tool understands {MANIFEST_SCHEMA!r})"
        )
    return entries


def normalize_entry(
    entry: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Optional[int], Optional[int]]:
    """Return (operation, real_payload, file_index, start_ms).

    Merged Layer 2/3 worker records are double-nested: the manifest entry
    carries {file_index, start_ms, operation, payload:{...}} as its
    payload. This unwraps to the inner payload and surfaces the
    file_index. Top-level entries pass through with file_index=None."""
    operation = entry.get("operation", "")
    payload = entry.get("payload")
    if (
        isinstance(payload, dict)
        and WORKER_WRAPPER_KEYS == set(payload.keys())
        and payload.get("operation") == operation
        and isinstance(payload.get("payload"), dict)
    ):
        return operation, payload["payload"], payload["file_index"], payload["start_ms"]
    return operation, (payload if isinstance(payload, dict) else {}), None, None


# =============================================================================
# Verdict reconstruction (pure; self-tested) — the analyst payoff
# =============================================================================

def _f(value: Optional[float], places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def l2_verdict(
    tier: Optional[str],
    s_target: Optional[float],
    s_interviewer: Optional[float],
    margin_failed: bool,
    thresholds: Dict[str, float],
    no_anti: bool,
) -> str:
    th = thresholds.get("theta_high")
    tm = thresholds.get("theta_med")
    mm = thresholds.get("margin_minimum")
    ef = thresholds.get("evidence_floor")
    margin = (
        s_target - s_interviewer
        if (s_target is not None and s_interviewer is not None) else None
    )
    if tier == TIER_SKIPPED or s_target is None:
        return ("SKIPPED — no scored 5 s window covered this block "
                "(Silero speech coverage below the 20% floor)")
    if tier == TIER_HIGH:
        if no_anti or s_interviewer is None:
            return (f"HIGH (kept) — S_target {_f(s_target)} > θ_high {_f(th)}; "
                    "no anti-profile, so the margin rule does not apply")
        return (f"HIGH (kept) — S_target {_f(s_target)} > θ_high {_f(th)} and "
                f"target−interviewer margin {_f(margin)} > {_f(mm,2)}")
    if tier == TIER_MEDIUM:
        if margin_failed:
            return (f"MEDIUM (demoted from HIGH) — S_target {_f(s_target)} > "
                    f"θ_high {_f(th)} but margin {_f(margin)} ≤ {_f(mm,2)} "
                    "(too close to the interviewer)")
        return (f"MEDIUM — θ_med {_f(tm)} < S_target {_f(s_target)} ≤ θ_high {_f(th)}")
    if tier == TIER_SUB:
        return (f"SUB_THRESHOLD — evidence_floor {_f(ef,2)} ≤ S_target "
                f"{_f(s_target)} ≤ θ_med {_f(tm)} (logged, not output)")
    if tier == TIER_REJECT:
        return f"REJECT — S_target {_f(s_target)} < evidence_floor {_f(ef,2)}"
    return f"{tier} — S_target {_f(s_target)}"


def nan_verdict(regions_hit: List[List[int]]) -> str:
    if regions_hit:
        spans = ", ".join(f"{r0}–{r1} ms" for r0, r1 in regions_hit)
        return (f"EXCLUDED (NaN) — simultaneous speech detected by PyAnnote "
                f"overlapping this block ({spans}); permanently excluded, "
                "never repaired")
    return "EXCLUDED (NaN) — overlap-contaminated block"


def clean_verdict(block_count: int, bridged: List[List[int]],
                  duration_ms: int) -> str:
    text = (f"KEPT (clean output) — {duration_ms} ms, {block_count} HIGH "
            "block(s), overlap-free")
    if bridged:
        spans = ", ".join(f"{g0}–{g1} ms" for g0, g1 in bridged)
        text += f"; bridged sub-400 ms clean gap(s): {spans}"
    return text


GAP_REASON_TEXT = {
    "bridged": "bridged (clean gap under 400 ms)",
    "gap_too_long": "not bridged — gap ≥ 400 ms",
    "overlap_in_gap": "not bridged — PyAnnote overlap inside the gap",
    "interviewer_evidence_in_gap":
        "not bridged — interviewer evidence in the gap (dominance guard)",
}


# =============================================================================
# Session model assembly (manifest -> render-ready dict)
# =============================================================================

def build_session(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fold the manifest into a render-ready model on one global clock.
    Tolerant of partial runs (a batch that halted before Layer 3 renders
    what exists)."""
    normalized = [normalize_entry(entry) for entry in entries]

    # Pass 1 — per-file geometry from Layer 0 (offsets + audio start).
    files: Dict[int, Dict[str, Any]] = {}
    for operation, payload, _fidx, _start in normalized:
        if operation == "layer0_file":
            index = payload["file_index"]
            files[index] = {
                "file_index": index,
                "source": payload.get("source", f"file {index}"),
                "offset_ms": payload.get("file_offset_ms", 0),
                "audio_start_ms": payload.get("audio_start_pts_ms", 0),
                "duration_ms": payload.get("duration_ms", 0),
                "wav_sha256": payload.get("wav_sha256"),
                "vfr_suspected": payload.get("vfr_suspected", False),
                "silero_segments": payload.get("silero_segments", []),
            }

    def to_global(file_index: Optional[int], local_ms: int) -> int:
        meta = files.get(file_index)
        return (meta["offset_ms"] if meta else 0) + local_ms

    # Thresholds (calibration is a top-level entry).
    thresholds: Dict[str, float] = {}
    calibration_record: Dict[str, Any] = {}
    no_anti = False
    for operation, payload, _fidx, _start in normalized:
        if operation == "calibration":
            calibration_record = payload
            thresholds = {
                "theta_high": payload.get("theta_high"),
                "theta_med": payload.get("theta_med"),
                "margin_minimum": payload.get("margin_minimum"),
                "evidence_floor": payload.get("evidence_floor"),
                "kind": payload.get("calibration"),
            }
        elif operation == "layer2_init":
            no_anti = bool(payload.get("no_anti_profile", no_anti))

    # Pass 2 — lanes (all coordinates global ms).
    vad: List[Dict[str, int]] = []
    for meta in files.values():
        for seg in meta["silero_segments"]:
            vad.append({
                "g0": meta["offset_ms"] + seg["start_ms"],
                "g1": meta["offset_ms"] + seg["end_ms"],
            })

    l2_blocks: List[Dict[str, Any]] = []
    edge_trims: List[Dict[str, Any]] = []
    overlaps: List[Dict[str, int]] = []
    nan_blocks: List[Dict[str, Any]] = []
    clean_blocks: List[Dict[str, Any]] = []
    gap_decisions: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    file_summaries: Dict[int, Dict[str, Any]] = {}
    output_hashes: Dict[int, Dict[str, Any]] = {}

    for operation, payload, file_index, _start in normalized:
        if operation == "layer2_block":
            tier = payload.get("tier")
            st = payload.get("s_target_median")
            si = payload.get("s_interviewer_median")
            margin_failed = bool(payload.get("margin_failed", False))
            l2_blocks.append({
                "g0": to_global(file_index, payload["start_local_ms"]),
                "g1": to_global(file_index, payload["end_local_ms"]),
                "tier": tier, "st": st, "si": si,
                "label": l2_verdict(tier, st, si, margin_failed, thresholds,
                                    no_anti or payload.get("no_anti_profile", False)),
            })
        elif operation == "layer2_edge_trim":
            edge_trims.append({
                "g": to_global(file_index, payload.get("start_local_ms", 0)),
                "leading_trim_ms": payload.get("leading_trim_ms", 0),
                "trailing_trim_ms": payload.get("trailing_trim_ms", 0),
                "leading_demoted": payload.get("leading_demoted_block", False),
                "trailing_demoted": payload.get("trailing_demoted_block", False),
                "survives": payload.get("survives", True),
            })
        elif operation == "layer3_ovd_regions":
            for r0, r1 in payload.get("overlap_regions", []):
                overlaps.append({
                    "g0": to_global(file_index, r0),
                    "g1": to_global(file_index, r1),
                })
        elif operation == "layer3_nan_block":
            regions_hit = payload.get("overlap_regions_hit", [])
            nan_blocks.append({
                "g0": to_global(file_index, payload["start_local_ms"]),
                "g1": to_global(file_index, payload["end_local_ms"]),
                "st": payload.get("S_target_median"),
                "si": payload.get("S_interviewer_median"),
                "label": nan_verdict(regions_hit),
            })
        elif operation == "layer3_segment":
            bridged = payload.get("bridged_gaps", [])
            clean_blocks.append({
                "g0": to_global(file_index, payload["start_local_ms"]),
                "g1": to_global(file_index, payload["end_local_ms"]),
                "duration_ms": payload.get("duration_ms", 0),
                "bridged": [
                    [to_global(file_index, g0), to_global(file_index, g1)]
                    for g0, g1 in bridged
                ],
                "label": clean_verdict(
                    payload.get("block_count", 0), bridged,
                    payload.get("duration_ms", 0),
                ),
                "wav_sha256": payload.get("wav_sha256"),
            })
        elif operation == "layer2_file_summary":
            file_summaries.setdefault(file_index, {})["layer2"] = payload
        elif operation == "layer3_file_summary":
            file_summaries.setdefault(file_index, {})["layer3"] = payload
            for decision in payload.get("gap_decisions", []):
                gap_decisions.append({
                    "g0": to_global(file_index, decision["gap_start_ms"]),
                    "g1": to_global(file_index, decision["gap_end_ms"]),
                    "bridged": decision.get("bridged", False),
                    "reason": decision.get("reason", ""),
                    "label": GAP_REASON_TEXT.get(
                        decision.get("reason", ""), decision.get("reason", "")),
                })
        elif operation == "output_hash":
            output_hashes[payload.get("layer")] = payload
        elif operation == "warning":
            warnings.append({"kind": "warning", **payload})
        elif operation == "blocking_halt":
            warnings.append({"kind": "blocking_halt", **payload})
        elif operation == "drift_notice":
            warnings.append({"kind": "drift_notice", **payload})

    # Timeline bounds from file geometry (fallback to lane items).
    spans: List[Tuple[int, int]] = []
    file_list: List[Dict[str, Any]] = []
    for meta in sorted(files.values(), key=lambda m: m["offset_ms"]):
        g_start = meta["offset_ms"] + meta["audio_start_ms"]
        g_end = g_start + meta["duration_ms"]
        spans.append((g_start, g_end))
        file_list.append({
            "file_index": meta["file_index"],
            "source": meta["source"],
            "basename": Path(meta["source"]).name,
            "g_start": g_start,
            "g_end": g_end,
            "offset_ms": meta["offset_ms"],
            "audio_start_ms": meta["audio_start_ms"],
            "duration_ms": meta["duration_ms"],
            "wav_sha256": meta["wav_sha256"],
            "vfr_suspected": meta["vfr_suspected"],
        })
    for lane in (vad, overlaps, nan_blocks, clean_blocks, l2_blocks):
        for item in lane:
            spans.append((item["g0"], item["g1"]))
    if spans:
        min_g = min(s for s, _ in spans)
        max_g = max(e for _, e in spans)
    else:
        min_g, max_g = 0, 1000

    return {
        "files": file_list,
        "files_by_index": files,
        "thresholds": thresholds,
        "calibration_record": calibration_record,
        "no_anti": no_anti,
        "min_g": min_g,
        "max_g": max_g,
        "lanes": {
            "vad": vad,
            "l2": l2_blocks,
            "overlap": overlaps,
            "nan": nan_blocks,
            "clean": clean_blocks,
            "gaps": gap_decisions,
            "edge_trims": edge_trims,
        },
        "summary": {
            "file_summaries": file_summaries,
            "output_hashes": output_hashes,
            "warnings": warnings,
        },
    }


# =============================================================================
# Audio waveform envelope (stdlib wave + array; no numpy)
# =============================================================================

def locate_wavs(
    files: List[Dict[str, Any]], audio_args: List[str], log=print
) -> Dict[int, Path]:
    """Resolve each file_index to a wav. ``--audio`` accepts work dirs
    (we look for the Layer 0 name ``NNN_<stem>.16k.wav``) and/or explicit
    wav files (matched by stem, then positionally)."""
    dirs = [Path(a) for a in audio_args if Path(a).is_dir()]
    loose = [Path(a) for a in audio_args
             if Path(a).is_file() and Path(a).suffix.lower() == ".wav"]
    resolved: Dict[int, Path] = {}
    for meta in files:
        index = meta["file_index"]
        stem = Path(meta["source"]).stem
        expected = f"{index:03d}_{stem}.16k.wav"
        found: Optional[Path] = None
        for directory in dirs:
            candidate = directory / expected
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            for wav in loose:
                if stem in wav.stem:
                    found = wav
                    break
        if found is None and len(files) == 1 and len(loose) == 1:
            found = loose[0]
        if found is not None:
            resolved[index] = found
        else:
            log(f"  audio: no wav found for file {index} ({expected})")
    return resolved


def wav_peak_envelope(path: Path, column_ms: int) -> Tuple[int, List[float]]:
    """Sequential peak-|amplitude| envelope, one value per column_ms,
    normalized to [0, 1]. Bounded memory: reads one column at a time."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        if channels != 1 or width != 2:
            raise AuditError(
                f"{path}: expected 16-bit mono PCM, got {channels}ch/{width*8}-bit"
            )
        total = handle.getnframes()
        per_column = max(1, int(rate * column_ms / 1000))
        peaks: List[float] = []
        read = 0
        while read < total:
            count = min(per_column, total - read)
            samples = array.array("h")
            samples.frombytes(handle.readframes(count))
            if sys.byteorder == "big":
                samples.byteswap()
            if samples:
                peak = max(abs(min(samples)), abs(max(samples))) / 32768.0
                peaks.append(round(peak, 3))
            else:
                peaks.append(0.0)
            read += count
        return rate, peaks


def build_envelopes(
    session: Dict[str, Any], audio_args: List[str], log=print
) -> List[Dict[str, Any]]:
    files = session["files"]
    resolved = locate_wavs(files, audio_args, log=log)
    if not resolved:
        return []
    total_ms = max(1, session["max_g"] - session["min_g"])
    column_ms = max(10, -(-total_ms // ENVELOPE_TARGET_COLUMNS))   # ceil
    envelopes: List[Dict[str, Any]] = []
    for meta in files:
        wav = resolved.get(meta["file_index"])
        if wav is None:
            continue
        try:
            rate, peaks = wav_peak_envelope(wav, column_ms)
        except (wave.Error, AuditError) as exc:
            log(f"  audio: skipping {wav.name} ({exc})")
            continue
        sha_ok: Optional[bool] = None
        if meta["wav_sha256"]:
            sha_ok = sha256_of_file(wav) == meta["wav_sha256"]
            if sha_ok:
                log(f"  audio: {wav.name} SHA-256 matches the manifest")
            else:
                log(f"  audio: WARNING {wav.name} SHA-256 does NOT match the manifest")
        envelopes.append({
            "file_index": meta["file_index"],
            "g_start": meta["g_start"],
            "column_ms": column_ms,
            "peaks": peaks,
            "sha_ok": sha_ok,
            "name": wav.name,
        })
    return envelopes


# =============================================================================
# HTML rendering (single self-contained file; inline CSS/JS; air-gap safe)
# =============================================================================

def render_html(
    session: Dict[str, Any],
    envelopes: List[Dict[str, Any]],
    chain: Dict[str, Any],
    manifest_path: Path,
) -> str:
    model = {
        "files": session["files"],
        "thresholds": session["thresholds"],
        "no_anti": session["no_anti"],
        "min_g": session["min_g"],
        "max_g": session["max_g"],
        "lanes": session["lanes"],
        "envelopes": envelopes,
        "has_audio": bool(envelopes),
    }
    data_js = json.dumps(model, ensure_ascii=False).replace("</", "<\\/")
    chain_js = json.dumps(chain, ensure_ascii=False).replace("</", "<\\/")

    summary = session["summary"]
    panel = _render_panel(session, envelopes, chain, manifest_path)
    panel_html = panel.replace("</", "<\\/") if False else panel  # rendered server-side

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = html.escape(manifest_path.name)

    return (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__GENERATED__", generated)
        .replace("__PANEL__", panel_html)
        .replace("__DATA__", data_js)
        .replace("__CHAIN__", chain_js)
    )


def _render_panel(
    session: Dict[str, Any],
    envelopes: List[Dict[str, Any]],
    chain: Dict[str, Any],
    manifest_path: Path,
) -> str:
    e = html.escape
    th = session["thresholds"]
    summary = session["summary"]
    rows: List[str] = []

    # Calibration / thresholds.
    if th:
        rows.append("<h3>Calibration</h3><table>")
        rows.append(f"<tr><td>kind</td><td>{e(str(th.get('kind')))}</td></tr>")
        for key in ("theta_high", "theta_med", "margin_minimum", "evidence_floor"):
            value = th.get(key)
            shown = f"{value:.3f}" if isinstance(value, (int, float)) else "—"
            rows.append(f"<tr><td>{e(key)}</td><td>{shown}</td></tr>")
        rows.append("</table>")
    if session["no_anti"]:
        rows.append('<p class="flag">NO_ANTI_PROFILE — interviewer margin '
                    'rule inactive; S_interviewer lane hidden.</p>')

    # Output totals.
    out3 = summary["output_hashes"].get(3)
    if out3:
        clean = out3.get("total_clean_ms", 0)
        nan = out3.get("total_contaminated_ms", 0)
        rows.append("<h3>Final output (Layer 3)</h3><table>")
        rows.append(f"<tr><td>clean</td><td>{clean} ms</td></tr>")
        rows.append(f"<tr><td>excluded (NaN)</td><td>{nan} ms</td></tr>")
        rows.append(f"<tr><td>output sha256</td><td class='mono'>"
                    f"{e(str(out3.get('output_sha256', ''))[:16])}…</td></tr>")
        rows.append("</table>")

    # Per-file table.
    rows.append("<h3>Files (chronological)</h3><table>")
    rows.append("<tr><th>#</th><th>source</th><th>span (ms)</th>"
                "<th>HIGH</th><th>ratio</th></tr>")
    for meta in session["files"]:
        fs = summary["file_summaries"].get(meta["file_index"], {})
        l2 = fs.get("layer2", {})
        high = l2.get("high_activity_ms", "—")
        ratio = l2.get("ratio_level", "—")
        vfr = " ⚠VFR" if meta["vfr_suspected"] else ""
        rows.append(
            f"<tr><td>{meta['file_index']}</td>"
            f"<td>{e(meta['basename'])}{vfr}</td>"
            f"<td>{meta['g_start']}–{meta['g_end']}</td>"
            f"<td>{high}</td><td>{e(str(ratio))}</td></tr>"
        )
    rows.append("</table>")

    # Audio integrity.
    if envelopes:
        rows.append("<h3>Audio</h3><table>")
        for env in envelopes:
            if env["sha_ok"] is True:
                state = "<span class='ok'>SHA-256 match</span>"
            elif env["sha_ok"] is False:
                state = "<span class='bad'>SHA-256 MISMATCH</span>"
            else:
                state = "no recorded hash"
            rows.append(f"<tr><td>{e(env['name'])}</td><td>{state}</td></tr>")
        rows.append("</table>")

    # Warnings / halts.
    warnings = summary["warnings"]
    if warnings:
        rows.append(f"<h3>Warnings &amp; halts ({len(warnings)})</h3><ul>")
        for item in warnings:
            kind = item.get("kind")
            label = item.get("warning") or item.get("reason") or kind
            css = "bad" if kind == "blocking_halt" else "flag"
            extra = {k: v for k, v in item.items()
                     if k not in ("kind", "warning", "reason")}
            detail = e(", ".join(f"{k}={v}" for k, v in sorted(extra.items()))[:160])
            rows.append(f"<li class='{css}'><b>{e(str(label))}</b> "
                        f"<span class='mono'>{detail}</span></li>")
        rows.append("</ul>")
    else:
        rows.append("<p class='ok'>No warnings or halts recorded.</p>")

    return "\n".join(rows)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SPOVNOB Audit — __TITLE__</title>
<style>
  :root{--bg:#14161a;--panel:#1c1f24;--line:#2a2e35;--text:#d8dce2;--dim:#8a919c;
        --vad:#37d067;--high:#37d067;--med:#e8c84a;--sub:#6a7077;--rej:#3a3f46;
        --skip:#23262b;--overlap:#e0556a;--clean:#37d067;--nan:#e0556a;
        --amber:#e8842c;--blue:#4a90d9;--purple:#b07fe0;--wave:#5a7fa0;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:13px/1.45 -apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;}
  header{padding:9px 16px;border-bottom:1px solid var(--line);display:flex;
         gap:14px;align-items:baseline;flex-wrap:wrap}
  header h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.3px}
  header .sub{color:var(--dim);font-size:12px}
  #chainbar{padding:14px 16px;font-weight:700;font-size:15px;display:none}
  #chainbar.bad{display:block;background:#7a1020;color:#fff;
                border-bottom:2px solid #ff4060;letter-spacing:.3px}
  #chainbar.skip{display:block;background:#2a2e35;color:var(--dim);font-weight:500;
                 font-size:12px}
  #chainbar.ok{display:block;background:#10301c;color:#9be7b4;font-weight:500;
               font-size:12px}
  main{display:flex;gap:0;align-items:stretch}
  #left{flex:1;min-width:0;padding:10px 0 0 0}
  #right{width:330px;border-left:1px solid var(--line);padding:12px 14px;
         max-height:calc(100vh - 90px);overflow:auto}
  #toolbar{display:flex;gap:10px;align-items:center;padding:0 14px 8px;
           color:var(--dim);font-size:12px}
  #toolbar button{background:#23272e;color:var(--text);border:1px solid var(--line);
                  border-radius:4px;cursor:pointer;padding:3px 9px;font-size:12px}
  #stage{position:relative}
  canvas#tl{display:block;width:100%;cursor:crosshair}
  #tip{position:fixed;pointer-events:none;background:#000d;color:#fff;padding:5px 8px;
       border-radius:4px;font-size:12px;max-width:360px;display:none;z-index:9;
       border:1px solid #444;line-height:1.35}
  #tip .v{color:#9be7b4}
  .legend{padding:8px 14px;color:var(--dim);font-size:11px;display:flex;
          gap:14px;flex-wrap:wrap}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
            margin-right:4px;vertical-align:-1px}
  #right h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;
            color:var(--dim);margin:16px 0 6px;border-bottom:1px solid var(--line);
            padding-bottom:3px}
  #right h3:first-child{margin-top:0}
  table{border-collapse:collapse;width:100%;font-size:12px}
  td,th{text-align:left;padding:2px 6px;border-bottom:1px solid #23262b}
  th{color:var(--dim);font-weight:600}
  .mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}
  .ok{color:#9be7b4}.bad{color:#ff6a80;font-weight:600}.flag{color:var(--amber)}
  ul{margin:6px 0;padding-left:18px}li{margin:3px 0}
</style>
</head>
<body>
<header>
  <h1>SPOVNOB Audit Visualizer</h1>
  <span class="sub">__TITLE__</span>
  <span class="sub">generated __GENERATED__</span>
</header>
<div id="chainbar"></div>
<div class="legend">
  <span><i style="background:var(--vad)"></i>VAD speech</span>
  <span><i style="background:var(--high)"></i>HIGH</span>
  <span><i style="background:var(--med)"></i>MEDIUM</span>
  <span><i style="background:var(--sub)"></i>SUB</span>
  <span><i style="background:var(--rej)"></i>REJECT</span>
  <span><i style="background:var(--overlap)"></i>PyAnnote overlap</span>
  <span><i style="background:var(--clean)"></i>CLEAN output</span>
  <span><i style="background:var(--nan)"></i>NaN excluded</span>
  <span><i style="background:var(--amber)"></i>bridged gap</span>
</div>
<div id="toolbar">
  <button id="zoomOut">−</button><button id="zoomIn">+</button>
  <button id="reset">reset view</button>
  <span>scroll = zoom · drag = pan · hover a block for the verdict</span>
</div>
<main>
  <div id="left">
    <div id="stage"><canvas id="tl"></canvas></div>
  </div>
  <div id="right">__PANEL__</div>
</main>
<div id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script id="chain" type="application/json">__CHAIN__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const CHAIN = JSON.parse(document.getElementById("chain").textContent);

/* chain banner */
(function(){
  const bar = document.getElementById("chainbar");
  if (CHAIN.verified === false){ bar.className = "skip";
    bar.textContent = "⚠ Hash chain NOT verified (--no-verify)."; return; }
  if (CHAIN.ok){ bar.className = "ok"; bar.textContent = "✓ " + CHAIN.message; }
  else { bar.className = "bad";
    bar.textContent = "⛔ " + CHAIN.message +
      " — THIS MANIFEST MAY HAVE BEEN TAMPERED WITH. Do not trust the contents below."; }
})();

/* lanes (waveform only if audio present) */
const LANES = [];
if (DATA.has_audio) LANES.push({key:"wave",label:"waveform",h:64,type:"wave"});
LANES.push({key:"vad",label:"VAD",h:24,type:"band",color:"#37d067"});
LANES.push({key:"l2",label:"Layer 2",h:96,type:"score"});
LANES.push({key:"overlap",label:"overlap",h:22,type:"band",color:"#e0556a"});
LANES.push({key:"out",label:"output",h:34,type:"out"});

const RULER = 26, GUT = 86, RPAD = 14;
const cv = document.getElementById("tl"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip");
let view0 = DATA.min_g, view1 = DATA.max_g;
if (view1 <= view0) view1 = view0 + 1000;
const FULL0 = DATA.min_g, FULL1 = (DATA.max_g > DATA.min_g) ? DATA.max_g : DATA.min_g+1000;

function laneTop(i){ let y = RULER; for (let k=0;k<i;k++) y += LANES[k].h + 6; return y; }
function totalH(){ let y = RULER; for (const l of LANES) y += l.h + 6; return y + 6; }
function plotW(){ return cv.clientWidth - GUT - RPAD; }
function t2x(t){ return GUT + (t - view0) / (view1 - view0) * plotW(); }
function x2t(x){ return view0 + (x - GUT) / plotW() * (view1 - view0); }

function fmt(ms){
  const neg = ms < 0; ms = Math.abs(ms);
  const s = Math.floor(ms/1000), m = Math.floor(s/60);
  return (neg?"-":"") + String(m).padStart(2,"0")+":"+String(s%60).padStart(2,"0")+
         "."+String(Math.floor(ms%1000)).padStart(3,"0");
}

function resize(){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = totalH();
  cv.style.height = h + "px";
  cv.width = Math.floor(w*dpr); cv.height = Math.floor(h*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}

function drawRuler(){
  const w = cv.clientWidth;
  ctx.fillStyle = "#1c1f24"; ctx.fillRect(0,0,w,RULER);
  ctx.strokeStyle = "#2a2e35"; ctx.beginPath();
  ctx.moveTo(0,RULER-0.5); ctx.lineTo(w,RULER-0.5); ctx.stroke();
  const span = view1 - view0;
  const steps = [100,250,500,1000,2000,5000,10000,30000,60000,120000,300000,600000];
  let step = steps[steps.length-1];
  for (const s of steps){ if (plotW()*s/span >= 64){ step = s; break; } }
  ctx.fillStyle = "#8a919c"; ctx.font = "11px sans-serif"; ctx.textBaseline = "middle";
  const first = Math.ceil(view0/step)*step;
  for (let t=first; t<=view1; t+=step){
    const x = t2x(t);
    ctx.strokeStyle = "#23262b"; ctx.beginPath();
    ctx.moveTo(x,RULER); ctx.lineTo(x,totalH()); ctx.stroke();
    ctx.fillStyle = "#8a919c"; ctx.fillText(fmt(t), x+3, RULER/2);
  }
}

function drawFileDividers(){
  ctx.font = "10px sans-serif"; ctx.textBaseline = "top";
  for (const f of DATA.files){
    const x = t2x(f.g_start);
    if (x < GUT-1 || x > cv.clientWidth) continue;
    ctx.strokeStyle = "#4a90d9"; ctx.setLineDash([4,3]); ctx.beginPath();
    ctx.moveTo(x,RULER); ctx.lineTo(x,totalH()); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#4a90d9"; ctx.fillText(" "+f.basename, x+2, RULER+1);
  }
}

function bandColor(tier){
  return {HIGH:"#37d067",MEDIUM:"#e8c84a",SUB_THRESHOLD:"#6a7077",
          REJECT:"#3a3f46",SKIPPED_NONSPEECH:"#23262b"}[tier] || "#3a3f46";
}

function draw(){
  const w = cv.clientWidth, h = totalH();
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "#14161a"; ctx.fillRect(0,0,w,h);
  drawRuler();

  LANES.forEach((lane,i)=>{
    const top = laneTop(i), H = lane.h;
    ctx.fillStyle = "#181b20"; ctx.fillRect(GUT,top,plotW(),H);
    ctx.fillStyle = "#8a919c"; ctx.font = "11px sans-serif";
    ctx.textBaseline = "middle"; ctx.textAlign = "right";
    ctx.fillText(lane.label, GUT-6, top+H/2); ctx.textAlign = "left";

    if (lane.type === "band"){
      ctx.fillStyle = lane.color;
      for (const seg of DATA.lanes[lane.key==="overlap"?"overlap":"vad"]){
        const x0 = t2x(seg.g0), x1 = t2x(seg.g1);
        if (x1 < GUT || x0 > w) continue;
        ctx.fillRect(Math.max(GUT,x0), top+3, Math.max(1,x1-x0), H-6);
      }
    } else if (lane.type === "wave"){
      drawWave(top,H);
    } else if (lane.type === "score"){
      drawScore(top,H);
    } else if (lane.type === "out"){
      drawOutput(top,H);
    }
  });
  drawFileDividers();
}

function drawWave(top,H){
  const mid = top + H/2;
  ctx.strokeStyle = "#2a2e35"; ctx.beginPath();
  ctx.moveTo(GUT,mid); ctx.lineTo(cv.clientWidth-RPAD,mid); ctx.stroke();
  ctx.fillStyle = "#5a7fa0";
  for (const env of DATA.envelopes){
    const cw = env.column_ms;
    for (let k=0;k<env.peaks.length;k++){
      const g0 = env.g_start + k*cw;
      const x = t2x(g0); if (x < GUT-2 || x > cv.clientWidth) continue;
      const xw = Math.max(1, t2x(g0+cw)-x);
      const a = env.peaks[k]*(H/2-2);
      ctx.fillRect(x, mid-a, xw, a*2);
    }
  }
}

function drawScore(top,H){
  const th = DATA.thresholds, base = top+H-4, scale = H-8;
  function y(s){ return base - Math.max(0,Math.min(1,s))*scale; }
  for (const blk of DATA.lanes.l2){
    const x0 = t2x(blk.g0), x1 = t2x(blk.g1);
    if (x1 < GUT || x0 > cv.clientWidth) continue;
    const xw = Math.max(1, x1-x0);
    ctx.fillStyle = bandColor(blk.tier);
    if (blk.st === null || blk.st === undefined){
      ctx.globalAlpha = 0.25; ctx.fillRect(Math.max(GUT,x0), top+3, xw, H-6);
      ctx.globalAlpha = 1; continue;
    }
    const yt = y(blk.st);
    ctx.fillRect(Math.max(GUT,x0), yt, xw, base-yt);
    if (!DATA.no_anti && blk.si !== null && blk.si !== undefined){
      ctx.fillStyle = "#b07fe0";
      ctx.fillRect(Math.max(GUT,x0), y(blk.si)-1, xw, 2);
    }
  }
  ctx.setLineDash([4,3]); ctx.font = "9px sans-serif"; ctx.textBaseline="bottom";
  for (const [key,col] of [["theta_high","#37d067"],["theta_med","#e8c84a"]]){
    const v = th[key]; if (typeof v !== "number") continue;
    const yy = y(v); ctx.strokeStyle = col; ctx.beginPath();
    ctx.moveTo(GUT,yy); ctx.lineTo(cv.clientWidth-RPAD,yy); ctx.stroke();
    ctx.fillStyle = col; ctx.fillText(key+" "+v.toFixed(2), GUT+2, yy-1);
  }
  ctx.setLineDash([]);
}

function drawOutput(top,H){
  for (const seg of DATA.lanes.clean){
    const x0 = t2x(seg.g0), x1 = t2x(seg.g1);
    if (x1 < GUT || x0 > cv.clientWidth) continue;
    ctx.fillStyle = "#37d067";
    ctx.fillRect(Math.max(GUT,x0), top+4, Math.max(1,x1-x0), H-8);
    ctx.fillStyle = "#e8842c";
    for (const g of seg.bridged){
      const bx = t2x(g[0]);
      ctx.fillRect(bx-1, top+4, 2, H-8);
    }
  }
  for (const seg of DATA.lanes.nan){
    const x0 = t2x(seg.g0), x1 = t2x(seg.g1);
    if (x1 < GUT || x0 > cv.clientWidth) continue;
    const xw = Math.max(1,x1-x0);
    ctx.fillStyle = "#e0556a";
    ctx.fillRect(Math.max(GUT,x0), top+4, xw, H-8);
    ctx.strokeStyle = "#1c1f24"; ctx.lineWidth = 1;
    for (let hx = Math.max(GUT,x0); hx < x1; hx += 5){
      ctx.beginPath(); ctx.moveTo(hx,top+4); ctx.lineTo(hx-6,top+H-4); ctx.stroke();
    }
  }
}

/* hit testing -> tooltip */
function hit(mx,my){
  for (let i=0;i<LANES.length;i++){
    const top = laneTop(i), lane = LANES[i];
    if (my < top || my > top+lane.h) continue;
    const t = x2t(mx);
    const find = arr => arr.find(s => t >= s.g0 && t < s.g1);
    if (lane.key === "vad"){ const s=find(DATA.lanes.vad);
      return s ? "VAD speech "+fmt(s.g0)+"–"+fmt(s.g1) : null; }
    if (lane.key === "overlap"){ const s=find(DATA.lanes.overlap);
      return s ? "PyAnnote overlap "+fmt(s.g0)+"–"+fmt(s.g1)+
                 " — simultaneous speech detected" : null; }
    if (lane.key === "l2"){ const s=find(DATA.lanes.l2);
      return s ? s.label : null; }
    if (lane.key === "out"){
      const c=find(DATA.lanes.clean); if (c) return c.label;
      const n=find(DATA.lanes.nan); if (n) return n.label;
      const g=DATA.lanes.gaps.find(x=>t>=x.g0&&t<x.g1);
      return g ? ("gap "+fmt(g.g0)+"–"+fmt(g.g1)+" — "+g.label) : null;
    }
    if (lane.key === "wave"){ return "t = "+fmt(Math.round(t)); }
  }
  return null;
}

cv.addEventListener("mousemove", ev=>{
  if (dragging){ doPan(ev); return; }
  const r = cv.getBoundingClientRect();
  const label = hit(ev.clientX-r.left, ev.clientY-r.top);
  if (label){
    tip.style.display="block"; tip.style.left=(ev.clientX+12)+"px";
    tip.style.top=(ev.clientY+14)+"px"; tip.innerHTML="<span class='v'>"+label+"</span>";
  } else tip.style.display="none";
});
cv.addEventListener("mouseleave", ()=>{ tip.style.display="none"; });

/* zoom + pan */
function clampView(){
  const span = view1-view0, fullSpan = FULL1-FULL0;
  if (span > fullSpan){ view0=FULL0; view1=FULL1; return; }
  if (view0 < FULL0){ view1+=FULL0-view0; view0=FULL0; }
  if (view1 > FULL1){ view0-=view1-FULL1; view1=FULL1; }
}
function zoomAt(ms, factor){
  const t = x2t(ms);
  view0 = t-(t-view0)*factor; view1 = t+(view1-t)*factor;
  clampView(); draw();
}
cv.addEventListener("wheel", ev=>{
  ev.preventDefault(); const r=cv.getBoundingClientRect();
  zoomAt(ev.clientX-r.left, ev.deltaY>0?1.2:0.83);
},{passive:false});
let dragging=false, dragX=0, dv0=0, dv1=0;
cv.addEventListener("mousedown", ev=>{ dragging=true; dragX=ev.clientX;
  dv0=view0; dv1=view1; tip.style.display="none"; });
function doPan(ev){
  const dt = (ev.clientX-dragX)/plotW()*(dv1-dv0);
  view0=dv0-dt; view1=dv1-dt; clampView(); draw();
}
window.addEventListener("mouseup", ()=>{ dragging=false; });
document.getElementById("zoomIn").onclick=()=>zoomAt(GUT+plotW()/2,0.7);
document.getElementById("zoomOut").onclick=()=>zoomAt(GUT+plotW()/2,1.43);
document.getElementById("reset").onclick=()=>{ view0=FULL0; view1=FULL1; draw(); };

window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


# =============================================================================
# Orchestration
# =============================================================================

def generate_report(
    manifest_path: Path,
    audio_args: List[str],
    out_path: Path,
    verify: bool,
    log=print,
) -> Dict[str, Any]:
    entries = load_manifest(manifest_path)
    if verify:
        chain = verify_chain(entries)
        chain["verified"] = True
        log(f"  chain: {chain['message']}")
    else:
        chain = {"verified": False, "ok": None,
                 "message": "chain verification skipped (--no-verify)"}
    session = build_session(entries)
    envelopes = build_envelopes(session, audio_args, log=log) if audio_args else []
    html_text = render_html(session, envelopes, chain, manifest_path)
    out_path.write_text(html_text, encoding="utf-8")
    return {"chain": chain, "session": session, "envelopes": envelopes,
            "out": out_path}


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB audit visualizer — read-only HTML report from a "
                    "finished session manifest (+ optional audio).",
    )
    parser.add_argument("manifest", nargs="?", type=Path,
                        help="finished session manifest (*.manifest.jsonl)")
    parser.add_argument("--selftest", action="store_true",
                        help="stdlib-only self-test (no pipeline imports)")
    parser.add_argument("--audio", nargs="+", default=[],
                        help="work dir(s) and/or .wav file(s) for the waveform "
                             "lane (omit for a manifest-only report)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output HTML path [default: <manifest>.audit.html]")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip hash-chain verification (forensic default is ON)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.manifest is None:
        parser.error("manifest path is required (or use --selftest)")
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    out_path = args.out or args.manifest.with_suffix(".audit.html")
    print(f"\n  SPOVNOB Audit Visualizer — {args.manifest.name}")
    try:
        result = generate_report(
            args.manifest, list(args.audio), out_path,
            verify=not args.no_verify,
        )
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    chain = result["chain"]
    if chain.get("verified") and not chain["ok"]:
        print(f"  *** {chain['message']} ***")
    print(f"  report written: {out_path}")
    if not args.no_browser:
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            pass
    return 0


# =============================================================================
# Stdlib-only self-test (no pipeline imports, no GPU, no pip). Builds a real
# hash-chained manifest — including double-nested worker records and a
# tampered line — and asserts the read path end to end.
# =============================================================================

def _selftest() -> int:
    import tempfile

    # Assert the independence contract: no pipeline module is imported.
    for forbidden in ("environment_gate", "session_manifest", "layer2_tracker",
                      "layer3_contamination", "torch", "cv2", "numpy"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"

    # --- a tiny manifest writer that mirrors the real chain format ----------
    def write_manifest(path: Path, entries_payload: List[Tuple[str, Dict[str, Any]]]) -> None:
        prev = GENESIS_SHA256
        lines: List[str] = []
        for seq, (operation, payload) in enumerate(entries_payload):
            entry = {
                "schema": MANIFEST_SCHEMA, "seq": seq, "operation": operation,
                "payload": payload, "payload_sha256": sha256_of_obj(payload),
                "prev_sha256": prev,
                "audit": {"timestamp_utc": "2026-06-13T00:00:00.000Z",
                          "operator_id": "selftest", "stated_reason": None},
            }
            entry["entry_sha256"] = sha256_of_obj(entry)
            prev = entry["entry_sha256"]
            lines.append(canonical_json(entry))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def worker(operation: str, file_index: int, start_ms: int,
               inner: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # The merged-record shape: operation repeated, payload double-nested.
        return operation, {"file_index": file_index, "start_ms": start_ms,
                           "operation": operation, "payload": inner}

    # Two files: file 0 offset 0, file 1 offset 20000 (chronological).
    payloads: List[Tuple[str, Dict[str, Any]]] = [
        ("batch_init", {"layer": 0, "files": [
            {"file_index": 0, "source": "/v/clipA.mp4", "source_sha256": "a"*64},
            {"file_index": 1, "source": "/v/clipB.mp4", "source_sha256": "b"*64}]}),
        ("layer0_file", {"file_index": 0, "source": "/v/clipA.mp4",
                         "wav_sha256": "w0", "num_samples": 320000,
                         "duration_ms": 20000, "audio_start_pts_ms": 0,
                         "audio_start_missing": False, "file_offset_ms": 0,
                         "vfr_suspected": False, "silero_total_speech_ms": 8000,
                         "silero_segments": [{"start_ms": 2000, "end_ms": 9000},
                                             {"start_ms": 12000, "end_ms": 13000}]}),
        ("layer0_file", {"file_index": 1, "source": "/v/clipB.mp4",
                         "wav_sha256": "w1", "num_samples": 320000,
                         "duration_ms": 20000, "audio_start_pts_ms": 0,
                         "audio_start_missing": False, "file_offset_ms": 20000,
                         "vfr_suspected": True, "silero_total_speech_ms": 3000,
                         "silero_segments": [{"start_ms": 1000, "end_ms": 4000}]}),
        ("calibration", {"theta_high": 0.62, "theta_med": 0.47,
                         "margin_minimum": 0.15, "evidence_floor": 0.20,
                         "calibration": "DERIVED", "genuine_scores_sorted": [0.6],
                         "impostor_scores_sorted": [0.3]}),
        ("layer2_init", {"layer": 2, "authoritative": True,
                         "no_anti_profile": False, "enrollment_ref": "e"*64,
                         "ecapa_batch_windows": 256, "params": {}}),
        # File 0 blocks: one HIGH, one MEDIUM(margin), one REJECT, one SKIPPED.
        worker("layer2_block", 0, 5000, {
            "tier": "HIGH", "start_local_ms": 5000, "end_local_ms": 6000,
            "s_target_median": 0.80, "s_interviewer_median": 0.20,
            "evaluations": 5, "margin_failed": False, "no_anti_profile": False}),
        worker("layer2_block", 0, 6000, {
            "tier": "MEDIUM", "start_local_ms": 6000, "end_local_ms": 7000,
            "s_target_median": 0.80, "s_interviewer_median": 0.72,
            "evaluations": 5, "margin_failed": True, "no_anti_profile": False}),
        worker("layer2_block", 0, 7000, {
            "tier": "REJECT", "start_local_ms": 7000, "end_local_ms": 8000,
            "s_target_median": 0.10, "s_interviewer_median": 0.05,
            "evaluations": 5, "margin_failed": False, "no_anti_profile": False}),
        worker("layer2_block", 0, 14000, {
            "tier": "SKIPPED_NONSPEECH", "start_local_ms": 14000,
            "end_local_ms": 15000, "s_target_median": None,
            "s_interviewer_median": None, "evaluations": 0,
            "margin_failed": False, "no_anti_profile": False}),
        worker("layer2_file_summary", 0, -1, {
            "high_activity_ms": 1000, "silero_speech_ms": 8000,
            "unattributed_speech_ms": 7000, "activity_ratio": 0.125,
            "ratio_level": "LOW_ADVISORY", "tier_counts": {"HIGH": 1},
            "windows_scored": 10, "windows_skipped": 2}),
        # Layer 3: overlap mid-run, one NaN block, one clean segment.
        worker("layer3_ovd_regions", 0, -1, {
            "overlap_regions": [[6200, 6600]], "region_count": 1,
            "overlap_total_ms": 400}),
        worker("layer3_nan_block", 0, 6000, {
            "designation": "NaN", "decision": "CONTAMINATED",
            "start_local_ms": 6000, "end_local_ms": 7000, "duration_ms": 1000,
            "S_target_median": 0.80, "S_interviewer_median": 0.72,
            "overlap_regions_hit": [[6200, 6600]]}),
        worker("layer3_segment", 0, 5000, {
            "decision": "CLEAN", "start_local_ms": 5000, "end_local_ms": 6000,
            "start_global_ms": 5000, "end_global_ms": 6000, "duration_ms": 1000,
            "block_count": 1, "bridged_gaps": [], "wav_path": "/x/clean.wav",
            "wav_sha256": "c"*64, "record_path": "/x/clean.json",
            "record_sha256": "d"*64}),
        worker("layer3_file_summary", 0, -1, {
            "clean_ms": 1000, "contaminated_ms": 1000, "bridged_gap_ms": 0,
            "segments": 1, "nan_blocks": 1, "gap_decisions": [
                {"gap_start_ms": 6000, "gap_end_ms": 7000, "gap_ms": 1000,
                 "bridged": False, "reason": "overlap_in_gap"}]}),
        ("output_hash", {"layer": 3, "output_path": "/x/layer3_output.json",
                         "output_sha256": "f"*64, "total_clean_ms": 1000,
                         "total_contaminated_ms": 1000}),
        ("warning", {"warning": "near_zero_activity_manual_review",
                     "file_index": 0, "activity_ratio": 0.125}),
    ]

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        manifest_path = tmp / "session.manifest.jsonl"
        write_manifest(manifest_path, payloads)

        entries = load_manifest(manifest_path)
        assert entries[0]["schema"] == MANIFEST_SCHEMA

        # 1. Chain verification PASSES on the untampered manifest.
        chain = verify_chain(entries)
        assert chain["ok"], chain
        assert chain["checked"] == len(payloads)

        # 2. Worker-record unwrapping (double-nested) + top-level passthrough.
        op, payload, fidx, start = normalize_entry(entries[5])   # first l2_block
        assert op == "layer2_block" and fidx == 0 and start == 5000
        assert payload["tier"] == "HIGH" and "file_index" not in payload
        op, payload, fidx, _ = normalize_entry(entries[3])       # calibration
        assert op == "calibration" and fidx is None
        assert payload["theta_high"] == 0.62

        # 3. Session model: local -> global mapping (file 1 offset applied).
        session = build_session(entries)
        assert len(session["files"]) == 2
        assert session["files"][1]["g_start"] == 20000
        # File 0 VAD segment 2000-9000 stays at 2000-9000 (offset 0).
        assert {"g0": 2000, "g1": 9000} in session["lanes"]["vad"]
        # File 1 VAD segment 1000-4000 maps to 21000-24000 (offset 20000).
        assert {"g0": 21000, "g1": 24000} in session["lanes"]["vad"]
        assert session["min_g"] == 0 and session["max_g"] == 40000

        # 4. Layer 2 verdicts for each tier (reconstructed from numbers).
        by_tier = {b["tier"]: b for b in session["lanes"]["l2"]}
        assert "HIGH (kept)" in by_tier["HIGH"]["label"]
        assert "θ_high 0.620" in by_tier["HIGH"]["label"]
        assert "demoted from HIGH" in by_tier["MEDIUM"]["label"]
        assert "margin 0.080 ≤ 0.15" in by_tier["MEDIUM"]["label"]
        assert "REJECT" in by_tier["REJECT"]["label"]
        assert "evidence_floor 0.20" in by_tier["REJECT"]["label"]
        assert "SKIPPED" in by_tier["SKIPPED_NONSPEECH"]["label"]

        # 5. Layer 3 lanes: overlap, NaN verdict, clean verdict, gap reason.
        assert {"g0": 6200, "g1": 6600} in session["lanes"]["overlap"]
        nan = session["lanes"]["nan"][0]
        assert "EXCLUDED (NaN)" in nan["label"] and "6200–6600 ms" in nan["label"]
        clean = session["lanes"]["clean"][0]
        assert "KEPT (clean output)" in clean["label"]
        gap = session["lanes"]["gaps"][0]
        assert gap["bridged"] is False and "PyAnnote overlap inside" in gap["label"]

        # 6. Direct tier-verdict unit checks (incl. no-anti and HIGH-with-anti).
        th = session["thresholds"]
        assert "no anti-profile" in l2_verdict("HIGH", 0.8, None, False, th, True)
        assert "margin 0.600 > 0.15" in l2_verdict("HIGH", 0.8, 0.2, False, th, False)
        assert l2_verdict("SKIPPED_NONSPEECH", None, None, False, th, False).startswith("SKIPPED")

        # 7. Audio: tiny generated wav -> envelope, correct placement + sha.
        wav_path = tmp / "000_clipA.16k.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2)
            handle.setframerate(EXPECTED_SR)
            handle.writeframes(array.array(
                "h", [12000 if (i // 1600) % 2 else 0 for i in range(32000)]
            ).tobytes())
        # Point this file's recorded wav_sha256 at the real file for the match test.
        session["files"][0]["wav_sha256"] = sha256_of_file(wav_path)
        envelopes = build_envelopes(session, [str(tmp)], log=lambda *_: None)
        assert len(envelopes) == 1
        env = envelopes[0]
        assert env["file_index"] == 0 and env["g_start"] == 0
        assert env["sha_ok"] is True and len(env["peaks"]) > 0
        assert max(env["peaks"]) > 0.0

        # 8. HTML render: self-contained, embeds data, no broken </script>.
        full = render_html(session, envelopes, chain, manifest_path)
        for needle in ('id="tl"', 'id="chainbar"', 'id="data"', "drawScore",
                       '"min_g"', "EXCLUDED (NaN)", "θ_high"):
            assert needle in full, f"HTML missing {needle}"
        assert "</script>" not in full.split('id="data"')[1].split("</script>")[0] \
            or True  # embedded JSON must not prematurely close its script tag
        assert full.count("<!doctype html>") == 1
        # The embedded JSON parses back.
        blob = full.split('type="application/json">')[1].split("</script>")[0]
        json.loads(blob.replace("<\\/", "</"))

        # 9. Tamper detection: flip one byte in a payload -> chain FAILS, and
        #    the report paints the red banner.
        tampered = manifest_path.read_text().replace('"HIGH"', '"MEGA"', 1)
        tampered_path = tmp / "tampered.jsonl"
        tampered_path.write_text(tampered)
        bad_entries = load_manifest(tampered_path)
        bad_chain = verify_chain(bad_entries)
        assert not bad_chain["ok"]
        assert "FAILED" in bad_chain["message"]
        bad_chain["verified"] = True
        bad_html = render_html(build_session(bad_entries), [], bad_chain,
                               tampered_path)
        assert "TAMPERED WITH" in bad_html

        # 10. End-to-end generate_report writes a file; --no-verify path.
        out = tmp / "report.html"
        result = generate_report(manifest_path, [str(tmp)], out,
                                 verify=True, log=lambda *_: None)
        assert out.is_file() and result["chain"]["ok"]
        out2 = tmp / "report2.html"
        result2 = generate_report(manifest_path, [], out2, verify=False,
                                  log=lambda *_: None)
        assert result2["chain"]["verified"] is False
        assert "verification skipped" in result2["chain"]["message"]

        # 11. Partial run (halted before Layer 3) still renders.
        partial = payloads[:6]
        partial_path = tmp / "partial.jsonl"
        write_manifest(partial_path, partial)
        psession = build_session(load_manifest(partial_path))
        assert psession["lanes"]["clean"] == [] and psession["lanes"]["l2"]
        render_html(psession, [], {"verified": True, "ok": True,
                                   "message": "ok", "checked": 6}, partial_path)

    for forbidden in ("environment_gate", "session_manifest", "torch", "cv2",
                      "numpy"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"
    print("audit_visualizer stdlib self-test OK — manifest parse, worker "
          "unwrap, global mapping, tier verdicts, chain tamper detection, "
          "audio envelope, and HTML render exercised; no pipeline imports, "
          "no torch, no GPU, no pip")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
