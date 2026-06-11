"""
SPOVNOB — Module 0a: session_manifest.py
=========================================

Layer:      Cross-layer service (Module 0a — implemented first per the
            approved FLAG 5 ruling, 2026-06-11). Serves environment_gate
            and Layers 0-3. Imports nothing from any other SPOVNOB module.

Purpose:    Append-only, tamper-evident session manifest (JSON Lines).
            The forensic chain of custody for every parameter change,
            enrollment artifact, model checksum, discard, warning, and
            output hash produced during a SPOVNOB batch run.

Inputs:     Structured payload dicts from all other modules; per-file
            worker logs produced by the parallel Layer 2/3 fan-out.
Outputs:    - one hash-chained, append-only ``*.manifest.jsonl`` file
            - SHA-256 utilities reused by environment_gate (Model
              Vendoring Mandate) and Layer 2 Step 9 (output hash).

Implements (Audio_Diarization.md):
            - "Operator Threshold Manifest Format"                (Layer 2)
            - "Persistence, audit, and manifest rules for
              `E_anti` and `E_window`"                            (Layer 1)
            - "Canonical Manifest Merge Rule"          (System Environment)
            - checksum recording for the "Model Vendoring Mandate"
                                                       (System Environment)
            - SHA-256 output hash rules                  (Layer 2, Step 9)
            - Implementation Rule 7: the manifest entry is written and
              fsync'd BEFORE any destructive or irreversible operation.

Determinism contract:
            - Entry *payloads* are canonical JSON (sorted keys, fixed
              separators, ``allow_nan=False``) and contain no wall-clock
              data; ``payload_sha256`` is therefore bit-reproducible
              across re-runs. Wall-clock time and operator identity live
              only in the ``audit`` block, which exists for custody, not
              reproduction, and is excluded from reproducibility hashes.
            - Every key ending in ``_ms`` must be an ``int`` (PTS
              milliseconds). Floats, bools, and frame indices are
              rejected at write time (Implementation Rule 6).
            - Worker logs merge in canonical order
              ``(file_index, start_ms, operation, payload_sha256)`` —
              never arrival order — so the merged manifest is byte-
              identical in payload content regardless of scheduling.
            - Entries are hash-chained (``prev_sha256`` -> ``entry_sha256``)
              so any post-hoc edit, deletion, or reorder is detectable
              via ``SessionManifest.verify_chain()``.

CUDA determinism dependencies: none. Pure stdlib. This module must never
import torch, numpy, or any model framework.

Platform: Ubuntu 22.04 target (POSIX-only ``fcntl`` advisory locking
enforces the single-writer rule).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TextIO

SCHEMA = "spovnob-manifest-v1"
GENESIS_SHA256 = "0" * 64
_HASH_CHUNK_BYTES = 1024 * 1024


class Operation:
    """Canonical operation names. Modules may extend, but these cover the
    operations named explicitly in Audio_Diarization.md."""

    BATCH_INIT = "batch_init"
    MODEL_CHECKSUM = "model_checksum"            # Model Vendoring Mandate
    DETERMINISM_CHECK = "determinism_check"      # 10s startup verification
    PARAMETER_MODIFIED = "parameter_modified"    # Operator Threshold Manifest Format
    ENROLLMENT_VECTOR = "enrollment_vector"      # E_seed / E_window / E_anti created
    ENROLLMENT_DISCARD = "enrollment_discard"    # gate failures, M-Trap discards
    CALIBRATION = "calibration"                  # Layer 2 threshold derivation
    VIDEO_GAP = "video_gap"                      # cross-file gap entries
    DRIFT_NOTICE = "drift_notice"                # Layer 2 Step 8
    WARNING = "warning"                          # non-blocking warnings
    BLOCKING_HALT = "blocking_halt"              # halts (recorded before halting)
    DESTRUCTIVE_OP = "destructive_op"            # Rule 7 pre-action guard
    OUTPUT_HASH = "output_hash"                  # Layer 2 Step 9
    WORKER_LOG_MERGED = "worker_log_merged"      # Canonical Manifest Merge Rule


class ManifestError(Exception):
    """Base class for all manifest failures."""


class ManifestTimeError(ManifestError):
    """A ``*_ms`` field was not an integer (Rule 6 violation)."""


class ManifestChainError(ManifestError):
    """Hash chain, sequence, or content hash verification failed."""


class ManifestLockError(ManifestError):
    """A second writer attempted to open the manifest (single-writer rule)."""


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization: sorted keys, fixed separators,
    ASCII-only, NaN/Inf forbidden. The only serializer this module uses."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def sha256_of_file(path: Path) -> str:
    """Streaming file hash; used for model weights (Vendoring Mandate),
    worker logs, and output corpora."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_time_fields(obj: Any, _path: str = "payload") -> None:
    """Recursively enforce Rule 6: every key ending in ``_ms`` must hold an
    ``int`` (bool is rejected explicitly: it subclasses int)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{_path}.{key}"
            if isinstance(key, str) and key.endswith("_ms"):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ManifestTimeError(
                        f"{here} must be an integer millisecond value, "
                        f"got {type(value).__name__}: {value!r}"
                    )
            validate_time_fields(value, here)
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            validate_time_fields(item, f"{_path}[{index}]")


def _utc_now_iso_ms() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class SessionManifest:
    """Single-writer, append-only, hash-chained JSON-L manifest.

    Each line is one entry::

        {
          "schema":         "spovnob-manifest-v1",
          "seq":            <int, 0-based, strictly increasing>,
          "operation":      <str>,
          "payload":        <dict — deterministic content only>,
          "payload_sha256": <sha256 of canonical payload>,
          "prev_sha256":    <entry_sha256 of previous entry, or GENESIS>,
          "audit":          {"timestamp_utc": ..., "operator_id": ...,
                             "stated_reason": ...},
          "entry_sha256":   <sha256 of canonical entry minus this field>
        }

    Append-only is enforced three ways: the file is opened in append mode
    (never truncated), an exclusive ``fcntl`` lock rejects concurrent
    writers, and the hash chain makes any retroactive modification
    detectable by ``verify_chain``.
    """

    def __init__(
        self,
        path: Path | str,
        operator_id: Optional[str] = None,
        verify_on_open: bool = True,
    ) -> None:
        self.path = Path(path)
        self.operator_id = operator_id
        self._fh: TextIO = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            raise ManifestLockError(
                f"{self.path} is already open by another writer "
                f"(single-writer rule): {exc}"
            ) from exc
        if verify_on_open:
            self._seq, self._prev_sha256 = self._resume_from_verified_chain()
        else:
            self._seq, self._prev_sha256 = self._resume_from_tail()

    # -- chain state ---------------------------------------------------------

    def _resume_from_verified_chain(self) -> tuple[int, str]:
        entries = self.verify_chain(self.path)
        if not entries:
            return 0, GENESIS_SHA256
        return entries[-1]["seq"] + 1, entries[-1]["entry_sha256"]

    def _resume_from_tail(self) -> tuple[int, str]:
        self._fh.seek(0)
        last_line = ""
        for line in self._fh:
            if line.strip():
                last_line = line
        if not last_line:
            return 0, GENESIS_SHA256
        last = json.loads(last_line)
        return last["seq"] + 1, last["entry_sha256"]

    # -- writing -------------------------------------------------------------

    def append(
        self,
        operation: str,
        payload: Dict[str, Any],
        operator_id: Optional[str] = None,
        stated_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append one entry, flush, and fsync before returning.

        Returning from this method is the guarantee Rule 7 relies on: once
        the caller has the entry back, it is durable on disk and the caller
        may proceed with the operation it recorded.
        """
        if not isinstance(payload, dict):
            raise ManifestError(
                f"payload must be a dict, got {type(payload).__name__}"
            )
        validate_time_fields(payload)
        entry: Dict[str, Any] = {
            "schema": SCHEMA,
            "seq": self._seq,
            "operation": operation,
            "payload": payload,
            "payload_sha256": sha256_of_obj(payload),
            "prev_sha256": self._prev_sha256,
            "audit": {
                "timestamp_utc": _utc_now_iso_ms(),
                "operator_id": operator_id or self.operator_id,
                "stated_reason": stated_reason,
            },
        }
        entry["entry_sha256"] = sha256_of_obj(entry)
        self._fh.write(canonical_json(entry) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._seq += 1
        self._prev_sha256 = entry["entry_sha256"]
        return entry

    def guard_destructive(
        self,
        action: str,
        payload: Dict[str, Any],
        operator_id: Optional[str] = None,
        stated_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rule 7 entry point: record an imminent destructive/irreversible
        action. Callers must invoke this and receive the entry back BEFORE
        performing the action itself."""
        return self.append(
            Operation.DESTRUCTIVE_OP,
            {"action": action, **payload},
            operator_id=operator_id,
            stated_reason=stated_reason,
        )

    def record_parameter_change(
        self,
        parameter: str,
        default_value: Any,
        operator_value: Any,
        modified_by: str,
        stated_reason: str,
    ) -> Dict[str, Any]:
        """'Operator Threshold Manifest Format' (Layer 2)."""
        return self.append(
            Operation.PARAMETER_MODIFIED,
            {
                "parameter": parameter,
                "default_value": default_value,
                "operator_value": operator_value,
            },
            operator_id=modified_by,
            stated_reason=stated_reason,
        )

    def close(self) -> None:
        if not self._fh.closed:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()

    def __enter__(self) -> "SessionManifest":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- verification --------------------------------------------------------

    @staticmethod
    def verify_chain(path: Path | str) -> List[Dict[str, Any]]:
        """Re-walk the full manifest, re-deriving every hash and chain link.
        Returns the parsed entries on success; raises ManifestChainError on
        the first inconsistency. This is the auditor's entry point."""
        entries: List[Dict[str, Any]] = []
        prev_sha256 = GENESIS_SHA256
        with open(path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                claimed_entry_sha = entry.get("entry_sha256")
                unsealed = {k: v for k, v in entry.items() if k != "entry_sha256"}
                if sha256_of_obj(unsealed) != claimed_entry_sha:
                    raise ManifestChainError(
                        f"line {line_number}: entry_sha256 mismatch (tampered?)"
                    )
                if sha256_of_obj(entry["payload"]) != entry["payload_sha256"]:
                    raise ManifestChainError(
                        f"line {line_number}: payload_sha256 mismatch"
                    )
                if entry["prev_sha256"] != prev_sha256:
                    raise ManifestChainError(
                        f"line {line_number}: chain break "
                        f"(expected prev {prev_sha256[:12]}…, "
                        f"got {entry['prev_sha256'][:12]}…)"
                    )
                if entry["seq"] != len(entries):
                    raise ManifestChainError(
                        f"line {line_number}: seq {entry['seq']} != {len(entries)}"
                    )
                prev_sha256 = claimed_entry_sha
                entries.append(entry)
        return entries


class WorkerLog:
    """Per-file record log written by one parallel Layer 2/3 worker.

    Worker logs are NOT hash-chained (they are intermediate artifacts);
    their integrity is captured by recording each log's file SHA-256 in
    the main manifest at merge time. Records carry the canonical sort
    fields required by the merge rule."""

    def __init__(self, path: Path | str, file_index: int) -> None:
        if isinstance(file_index, bool) or not isinstance(file_index, int):
            raise ManifestError("file_index must be an int")
        self.path = Path(path)
        self.file_index = file_index
        self._fh: TextIO = open(self.path, "a", encoding="utf-8")

    def append(
        self,
        operation: str,
        payload: Dict[str, Any],
        start_ms: int = -1,
    ) -> None:
        """``start_ms = -1`` marks file-level records (e.g. activity ratio),
        which sort before all timed records of the same file."""
        if isinstance(start_ms, bool) or not isinstance(start_ms, int):
            raise ManifestTimeError("start_ms must be an int (PTS milliseconds)")
        validate_time_fields(payload)
        record = {
            "file_index": self.file_index,
            "start_ms": start_ms,
            "operation": operation,
            "payload": payload,
        }
        self._fh.write(canonical_json(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "WorkerLog":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def merge_worker_logs(
    manifest: SessionManifest, log_paths: Iterable[Path | str]
) -> int:
    """Canonical Manifest Merge Rule (System Environment section).

    Reads every record from every worker log, sorts by
    ``(file_index, start_ms, operation, payload_sha256)`` — a deterministic
    total order independent of worker scheduling and arrival time — and
    appends the records to the main manifest through its single writer.
    Finishes with a WORKER_LOG_MERGED summary entry carrying each source
    log's file SHA-256 and record count."""
    sources = sorted(Path(p) for p in log_paths)
    records: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for source in sources:
        count = 0
        with open(source, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
                    count += 1
        summary.append(
            {
                "log": source.name,
                "log_sha256": sha256_of_file(source),
                "records": count,
            }
        )
    records.sort(
        key=lambda r: (
            r["file_index"],
            r["start_ms"],
            r["operation"],
            sha256_of_obj(r["payload"]),
        )
    )
    for record in records:
        manifest.append(record["operation"], record)
    manifest.append(
        Operation.WORKER_LOG_MERGED,
        {"merged_records": len(records), "sources": summary},
    )
    return len(records)


if __name__ == "__main__":
    # Self-test: chain integrity, Rule 6 enforcement, scheduling-invariant merge.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest_path = tmp_dir / "session.manifest.jsonl"

        with SessionManifest(manifest_path, operator_id="selftest") as manifest:
            manifest.append(Operation.BATCH_INIT, {"batch_id": "selftest-001"})

            try:
                manifest.append(Operation.WARNING, {"start_ms": 1000.5})
                raise AssertionError("float _ms accepted — Rule 6 broken")
            except ManifestTimeError:
                pass

            # Two workers finishing out of order must merge identically.
            with WorkerLog(tmp_dir / "w1.jsonl", file_index=1) as worker_one:
                worker_one.append("block", {"start_ms": 5000, "tier": "HIGH"}, 5000)
                worker_one.append("block", {"start_ms": 2000, "tier": "HIGH"}, 2000)
            with WorkerLog(tmp_dir / "w0.jsonl", file_index=0) as worker_zero:
                worker_zero.append("block", {"start_ms": 9000, "tier": "HIGH"}, 9000)

            merged = merge_worker_logs(
                manifest, [tmp_dir / "w1.jsonl", tmp_dir / "w0.jsonl"]
            )
            assert merged == 3
            manifest.guard_destructive(
                "delete_preview_outputs", {"target": "preview/"},
                stated_reason="selftest",
            )

        entries = SessionManifest.verify_chain(manifest_path)
        merged_blocks = [e for e in entries if e["operation"] == "block"]
        order = [(e["payload"]["file_index"], e["payload"]["start_ms"])
                 for e in merged_blocks]
        assert order == [(0, 9000), (1, 2000), (1, 5000)], order

        # Tamper detection: flip one byte, expect chain failure.
        tampered = manifest_path.read_text().replace('"HIGH"', '"MEGA"', 1)
        tampered_path = tmp_dir / "tampered.jsonl"
        tampered_path.write_text(tampered)
        try:
            SessionManifest.verify_chain(tampered_path)
            raise AssertionError("tampered manifest passed verification")
        except ManifestChainError:
            pass

        print(f"session_manifest self-test OK — {len(entries)} entries verified")
