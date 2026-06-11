"""
SPOVNOB — Module 2 (Layer 1): errors.py
========================================

Layer 1 exception types. Kept in a leaf module so every other submodule
(including vision and enrollment, which would otherwise be circular)
can import them.

CUDA determinism dependencies: none.
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)


class Layer1Error(RuntimeError):
    """Unrecoverable Layer 1 failure. Always preceded by a blocking_halt
    or warning manifest entry when a manifest is in scope."""


class Layer1ReclickError(Layer1Error):
    """An operator click failed validation (guardrails 1, 2, 3): the run
    stops and asks the operator for a corrected click. Recorded as a
    WARNING entry before raising."""
