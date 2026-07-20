"""Deterministic learning layer — outcome attribution and performance memory.

Everything in this package is arithmetic. No model, no LLM, no network. It is
the half of the learning loop that must run on EVERY closed position, at zero
token cost, before any reasoning layer is consulted.

Lazy imports only: submodules here are imported by runtime paths that must not
pay for optional scientific dependencies at package-import time (see the lazy-
import gate in ``.claude/rules/improvement.md``).
"""

from __future__ import annotations

__all__ = ["outcome_attribution", "oanda_outcome_adapter"]
