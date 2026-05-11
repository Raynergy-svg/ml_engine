"""Default --granularity must be M15, not H1.

Background: 2026-05-11 incident — autonomous_trainer spawned
``scripts/scheduled_retrain.py`` WITHOUT a ``--granularity`` argument
(see ``src/scanner/automation/autonomous_trainer.py:283``). The argparse
default of ``H1`` produced an H1-trained transformer that scored 35.7%
on the M15 holdout (Team-3 incident log
``logs/autonomous_retrain/retrain_20260511T120726Z.log``). The new
M15-mandatory deploy gate (commit ``d93e37f``) correctly REFUSED the
deploy, but the system stayed halted on stale models indefinitely
because every subsequent autonomous retrain repeated the same H1
mistake.

Fix shape (Option A, minimum blast radius): change the argparse
default from ``H1`` → ``M15`` in ``_build_parser()`` of
``scripts/scheduled_retrain.py``. M15 is the production trading TF per
CLAUDE.md "Modernization stance" lines 64-66. Five other callers
(``autonomous_trainer.py``, ``retrain_worker.py``, ``execution.py``,
``continuous.py``, ``drift_remediator.py``) all omit ``--granularity``
and were silently inheriting the H1 default — the same one-line fix
corrects every one of them. Explicit callers (``expand_pairs_m15.py``)
remain unaffected because they pass ``--granularity M15`` already, and
any caller that genuinely needs H1 can still pass it explicitly.

Tests follow the No-Mock Rule (``.claude/rules/improvement.md``,
promoted 2026-05-01). They drive the REAL ``argparse.ArgumentParser``
returned by ``_build_parser()`` and assert on real ``args`` objects —
no ``unittest.mock`` anywhere.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# scripts/ is not on sys.path by default — mirror the import pattern used
# by tests/test_deploy_gate_m15_mandatory_2026_05_11.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scheduled_retrain import _build_parser  # noqa: E402


# --- L2 verification: argparse default --------------------------------------


def test_default_granularity_is_m15():
    """No --granularity flag passed -> args.granularity == 'M15'.

    This is the load-bearing assertion. Pre-fix this returned 'H1'.
    """
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.granularity == "M15", (
        f"Default granularity must be M15 (production TF per CLAUDE.md "
        f"modernization stance + d93e37f deploy gate), got {args.granularity!r}. "
        f"The 2026-05-11 incident recurs every time an autonomous spawner "
        f"omits --granularity and inherits this default."
    )


def test_explicit_h1_still_supported():
    """Backwards-compat: H1 must still parse cleanly when passed explicitly.

    Some diagnostic / regime-fragility runs (e.g. multi-TF holdout
    evaluation) still legitimately need H1. The fix changes the default,
    not the accepted values.
    """
    parser = _build_parser()
    args = parser.parse_args(["--granularity", "H1"])
    assert args.granularity == "H1"


def test_explicit_m15_works():
    """Symmetric to the H1 case — passing M15 explicitly still works.

    Callers like ``scripts/expand_pairs_m15.py`` that already pass
    ``--granularity M15`` must keep working.
    """
    parser = _build_parser()
    args = parser.parse_args(["--granularity", "M15"])
    assert args.granularity == "M15"


def test_explicit_h4_still_supported():
    """H4 is another legitimate diagnostic TF (multi-TF holdout).

    Confirms the change is to the DEFAULT, not the accepted vocabulary.
    """
    parser = _build_parser()
    args = parser.parse_args(["--granularity", "H4"])
    assert args.granularity == "H4"


# --- help text documentation ------------------------------------------------


def test_help_text_documents_m15_default():
    """`--help` output must surface that M15 is the default, with rationale.

    Operators discovering this script via ``--help`` should not have to
    grep source to learn what timeframe will be trained. The help text
    also cites the d93e37f deploy gate so the next operator who runs
    into a halt-on-stale-model situation finds the breadcrumb.
    """
    parser = _build_parser()
    help_text = parser.format_help()
    assert "M15" in help_text, (
        "Help text must mention M15 as the default granularity. "
        f"Got:\n{help_text}"
    )
    assert "default: M15" in help_text, (
        "Help text must explicitly state 'default: M15' so operators "
        "running --help discover the production default without grepping "
        f"source. Got:\n{help_text}"
    )


# --- L1 verification: source-level default ----------------------------------


def test_source_default_is_m15():
    """The string literal default in source must be 'M15'.

    Belt-and-suspenders check on the L1 layer. If a future merge resets
    the default to 'H1' (e.g. revert conflict), this test catches it
    independently of the L2 argparse assertion above.
    """
    src = (PROJECT_ROOT / "scripts" / "scheduled_retrain.py").read_text()
    # Locate the --granularity add_argument block and inspect its default
    needle = '"--granularity"'
    idx = src.find(needle)
    assert idx >= 0, "--granularity argument missing from scheduled_retrain.py"
    # Inspect the next ~400 chars after the needle for the default= line
    window = src[idx : idx + 400]
    assert 'default="M15"' in window, (
        f"--granularity must default to 'M15' in source. "
        f"Window after '--granularity':\n{window}"
    )
    assert 'default="H1"' not in window, (
        f"--granularity must NOT default to 'H1' (the 2026-05-11 bug). "
        f"Window after '--granularity':\n{window}"
    )


# --- L3 verification: end-to-end CLI smoke ----------------------------------


def test_cli_help_smoke_shows_m15_default():
    """Run the actual script with --help in a subprocess.

    Real CLI invocation against the real script as installed on disk.
    Catches any path/import regression that would prevent the script
    being launched as a subprocess (which is exactly how autonomous_trainer
    invokes it).
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "scheduled_retrain.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"scheduled_retrain.py --help exited non-zero ({result.returncode}). "
        f"stderr:\n{result.stderr}"
    )
    assert "default: M15" in result.stdout, (
        f"--help output must show 'default: M15'. Got stdout:\n{result.stdout}"
    )


# --- L3 verification: autonomous_trainer call site --------------------------


def test_autonomous_trainer_does_not_override_with_h1():
    """The autonomous_trainer subprocess command must not pin --granularity H1.

    The bug was that autonomous_trainer omits --granularity entirely and
    inherits whatever default scheduled_retrain.py advertises. The fix
    above changes the default to M15. This test guards the OTHER failure
    mode: a future operator "helpfully" pinning --granularity H1 in the
    spawner. That would re-introduce the 2026-05-11 incident even with
    the M15 default in place.

    Reads the real file off disk. No mocks.
    """
    src = (
        PROJECT_ROOT
        / "src"
        / "scanner"
        / "automation"
        / "autonomous_trainer.py"
    ).read_text()
    # Find the scheduled_retrain.py subprocess command construction
    needle = "scheduled_retrain.py"
    idx = src.find(needle)
    assert idx >= 0, (
        "autonomous_trainer.py no longer references scheduled_retrain.py — "
        "the wiring has changed; reassess this test's invariant."
    )
    # Inspect surrounding context for any explicit H1 override
    window = src[max(0, idx - 200) : idx + 600]
    # Allow comments mentioning "H1" but not an actual CLI override
    forbidden = '"--granularity",\n        "H1"'
    assert forbidden not in window, (
        f"autonomous_trainer.py must not pin --granularity to H1. "
        f"Window:\n{window}"
    )
    # Also check a single-line variant
    assert '"H1"' not in window or "granularity" not in window.split('"H1"')[0][-30:], (
        f"autonomous_trainer.py may be pinning H1 inline near granularity. "
        f"Window:\n{window}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
