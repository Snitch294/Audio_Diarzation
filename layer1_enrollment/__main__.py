"""
SPOVNOB — Module 2 (Layer 1): __main__.py
==========================================

CLI entrypoint:
  python3 -m layer1_enrollment --selftest
  python3 -m layer1_enrollment --run --videos <files...> --clicks clicks.json
      --work-dir <dir> --model-store <store> --manifest <jsonl>
      [--operator <id>]

--run chains: environment gate (all checks + resident models) ->
Layer 0 preprocessing -> Layer 1 enrollment. Ubuntu deployment box only.

CUDA determinism dependencies: via environment_gate.run_gate.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from session_manifest import SessionManifest


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB Layer 1 enrollment (Module 2)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip, no torch, no GPU)")
    mode.add_argument("--run", action="store_true",
                      help="run Layers 0+1 on a batch (Ubuntu deployment box)")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--clicks", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--operator", type=str, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        from .selftest import run as run_selftest
        return run_selftest()

    for required in ("videos", "clicks", "work_dir", "model_store", "manifest"):
        if getattr(args, required) is None:
            parser.error(f"--run requires --{required.replace('_', '-')}")

    from layer0_preprocessor import preprocess_batch

    from .enrollment import load_clicks, run_layer1

    clicks = load_clicks(args.clicks)
    with SessionManifest(args.manifest, operator_id=args.operator) as manifest:
        models = environment_gate.run_gate(args.model_store, manifest)
        batch = preprocess_batch(manifest, args.videos, args.work_dir,
                                 models.silero)
        result = run_layer1(manifest, batch, models, clicks, args.work_dir)
    print(
        f"layer1 complete — E_composite frozen "
        f"({result.e_composite_sha256[:12]}…), "
        f"{result.total_verified_ms} ms verified across "
        f"{len(result.pool)} windows, "
        f"anti_pool={len(result.anti_pool)}, "
        f"final state={result.quality_history[-1]['state']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
