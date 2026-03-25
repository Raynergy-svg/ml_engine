#!/usr/bin/env python
"""Smoke test: verify CLI boots without tensorflow/xgboost/lightgbm/catboost.

US-034: All heavy ML imports must be lazy. The scanner, CLI, and training
facades must import cleanly with none of these packages installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class _HeavyDependencyBlocker:
    """Meta-path finder that blocks heavy ML packages at import time."""

    BLOCKED = {"tensorflow", "xgboost", "lightgbm", "catboost"}

    def find_module(self, name: str, path=None):
        root = name.split(".")[0]
        if root in self.BLOCKED:
            return self
        return None

    def load_module(self, name: str):
        raise ImportError(
            f"BLOCKED by test_no_tf_boot.py: '{name}' must not be eagerly imported"
        )


def main() -> int:
    # Install the blocker BEFORE any project imports
    blocker = _HeavyDependencyBlocker()
    sys.meta_path.insert(0, blocker)

    failures: list[str] = []

    # --- Test 1: Scanner chain ---
    try:
        from src.scanner.engine import Scanner  # noqa: F401
        from src.scanner.config import ScannerConfig  # noqa: F401
        from src.scanner.agents import ScannerAgentTeam  # noqa: F401
        from src.scanner.execution import ExecutionManager  # noqa: F401

        print("✓ Scanner chain imports OK")
    except ImportError as e:
        failures.append(f"Scanner chain: {e}")
        print(f"✗ Scanner chain FAILED: {e}")

    # --- Test 2: Scanner instantiation ---
    try:
        from src.scanner.config import ScannerConfig
        from src.scanner.engine import Scanner

        config = ScannerConfig()
        scanner = Scanner(config=config)
        print(f"✓ Scanner instantiation OK (ensemble_health: {scanner._ensemble_health})")
    except ImportError as e:
        failures.append(f"Scanner instantiation: {e}")
        print(f"✗ Scanner instantiation FAILED: {e}")
    except Exception as e:
        # Non-import errors are OK (e.g., missing OANDA keys)
        print(f"⚠ Scanner instantiation non-import error: {type(e).__name__}: {e}")

    # --- Test 3: Training facades ---
    try:
        from src.training.modular_trainers import BaseTrainer, TrainerConfig  # noqa: F401

        print("✓ modular_trainers facade imports OK")
    except ImportError as e:
        failures.append(f"modular_trainers: {e}")
        print(f"✗ modular_trainers FAILED: {e}")

    try:
        from src.training.trainers import BaseTrainer as BT2  # noqa: F401

        print("✓ trainers/__init__ imports OK")
    except ImportError as e:
        failures.append(f"trainers/__init__: {e}")
        print(f"✗ trainers/__init__ FAILED: {e}")

    # --- Test 4: CLI commands ---
    try:
        from src.cli import commands  # noqa: F401

        print("✓ CLI commands import OK")
    except ImportError as e:
        failures.append(f"CLI commands: {e}")
        print(f"✗ CLI commands FAILED: {e}")
    except Exception as e:
        print(f"⚠ CLI commands non-import error: {type(e).__name__}: {e}")

    # --- Summary ---
    print()
    if failures:
        print(f"FAIL: {len(failures)} eager import(s) detected:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("PASS: All import chains are lazy — CLI boots without heavy ML deps.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
