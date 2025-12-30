#!/usr/bin/env python3
"""Tiny shim to run the repository's `main.py` script as a console_scripts entry.

This keeps `main.py` unchanged (it executes as a script) while providing a
callable entrypoint for packaging tooling.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "main.py"
    # Execute the script as __main__ so its top-level CLI logic runs unchanged.
    runpy.run_path(str(script), run_name="__main__")
