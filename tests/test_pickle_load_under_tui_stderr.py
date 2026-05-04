"""Regression guard: model loads must work when sys.stderr is replaced.

Mythos audit 2026-05-04 — the root cause of the [Errno 9] Bad file
descriptor cascade. Textual replaces sys.stderr with its own pipe at
TUI startup. The previous _suppress_native_stderr implementation did
`os.dup2(devnull, sys.stderr.fileno())` which corrupted file
descriptors in that context — every subsequent gate-model load
raised EBADF. Symptom: scanner reported `momentum: none,
risk_passed=False, agent_passed=False` with raw_confidence
~0.11–0.16 across every scan, even though lgbm_momentum.pkl +
ridge_confidence.pkl + lgbm_risk.pkl were all on disk.

NO MOCKS. Tests exercise the actual context manager against real
artifacts on disk while sys.stderr is replaced (the real production
failure mode under Textual).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

from src.scanner.gates import _load_pickle_quietly, _suppress_native_stderr


REAL_ARTIFACT = Path("trained_data/models/joint/lgbm_momentum.pkl")


@pytest.fixture
def tui_stderr(monkeypatch: pytest.MonkeyPatch):
    """Replace sys.stderr with a Textual-style sink that claims fd 2.

    Mirrors what Textual's app does at startup. The class returns 2
    from fileno() but the *connection* between fd 2 and Python's
    sys.stderr has been broken — that's the corruption surface.
    """
    class TextualStderrSink(io.TextIOBase):
        def write(self, s: str) -> int:
            return len(s)

        def fileno(self) -> int:
            return 2

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stderr", TextualStderrSink())


def test_artifact_load_succeeds_under_tui_stderr(tui_stderr) -> None:
    """The actual production failure mode this fix addresses."""
    if not REAL_ARTIFACT.exists():
        pytest.skip(f"{REAL_ARTIFACT} not present — run training first")

    with open(REAL_ARTIFACT, "rb") as f:
        result = _load_pickle_quietly(f)

    assert isinstance(result, dict), (
        f"Expected dict from joint LightGBM artifact, got "
        f"{type(result).__name__}"
    )
    assert "momentum_model" in result or "model" in result, (
        f"Joint artifact missing model key — got keys: "
        f"{list(result.keys())[:8]}"
    )


def test_suppress_native_stderr_is_a_noop_in_tui(tui_stderr) -> None:
    """Inside the TUI branch, the context manager must not touch fd 2."""
    fd2_before = os.fstat(2).st_ino
    with _suppress_native_stderr():
        assert os.fstat(2).st_ino == fd2_before
    fd2_after = os.fstat(2).st_ino
    assert fd2_after == fd2_before, (
        "_suppress_native_stderr modified fd 2 under TUI conditions — "
        "model loads will start failing with [Errno 9] Bad file descriptor"
    )


def test_suppress_native_stderr_still_works_in_cli(monkeypatch) -> None:
    """Negative branch: under real stderr, OS-level redirection still runs."""
    monkeypatch.setattr(sys, "stderr", sys.__stderr__)

    with _suppress_native_stderr():
        pass

    if REAL_ARTIFACT.exists():
        with open(REAL_ARTIFACT, "rb") as f:
            result = _load_pickle_quietly(f)
        assert isinstance(result, dict)
