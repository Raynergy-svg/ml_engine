#!/usr/bin/env python3
"""Console entry point for the Buddy CLI."""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: macOS OpenMP/MKL/LLVM Conflict Resolution
# ═══════════════════════════════════════════════════════════════════════════════
# MUST be set BEFORE any imports to prevent Segmentation Fault 11 on macOS.
# See main.py for detailed documentation.
# ═══════════════════════════════════════════════════════════════════════════════
import os
import platform

if platform.system() == "Darwin":  # macOS
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import runpy
import sys
from pathlib import Path

from cli.monitoring_cli import buddy as click_buddy


def main() -> None:
    """Dispatch to the Click monitor command or the legacy CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        click_buddy(prog_name="buddy")
        return

    repo_root = Path(__file__).resolve().parent
    script = repo_root / "main.py"
    runpy.run_path(str(script), run_name="__main__")


__all__ = ["main"]
