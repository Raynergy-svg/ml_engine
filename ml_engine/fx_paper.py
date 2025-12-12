"""Compatibility wrapper for `fx_paper`.

Allows imports like `from ml_engine.fx_paper import ...` while the canonical
implementation lives at the repo root as `fx_paper.py`.
"""

from __future__ import annotations

from fx_paper import *  # noqa: F403
