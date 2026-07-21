"""Clean-tree import gate — the repository must contain a runnable AXIOM.

Born 2026-07-21: a clean checkout could not import the live FX trend lane
because ``src/axiom_training`` (4,485 lines including the lane's risk gate and
position sizer) existed only on one developer machine — never committed, not
gitignored, imported by seven tracked files. Every test passed locally while
the repository silently did not contain the trading system.

This gate imports every tracked production consumer FOR REAL — no import
monkeypatching, no copied local directories, no reliance on the developer
checkout. The rule for failures:

* A missing FIRST-PARTY module (``src.*``, ``dashboard.*``, ``scripts.*``)
  is ALWAYS a hard failure — that is precisely the defect this gate exists
  to catch, and no environment gap excuses it.
* A missing THIRD-PARTY distribution (fastapi, textual, ...) skips, because
  minimal environments legitimately omit optional runtime extras. The skip
  is only taken after proving the failure is not first-party.

A source grep is insufficient (a reference proves nothing about existence),
so the symbol checks import and assert callability.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every tracked production consumer of src.axiom_training, plus the package
#: modules whose absence broke the clean tree. Importing the consumer proves
#: the full transitive closure resolves from the tracked tree alone.
CONSUMERS = [
    "src.axiom_training",
    "src.axiom_training.risk_runtime",
    "src.equity.oanda_trend",
    "src.agent_runtime.loop",
    "src.scanner.feedback.post_trade_loop",
    "dashboard.server.data_sources",
    "dashboard.server.app",
    "dashboard.server.axiom_evidence_control",
]


def _import_or_classify(module_name: str):
    """Import for real; skip ONLY for a proven third-party gap."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        first_party = missing == "src" or missing.startswith(
            ("src.", "dashboard.", "scripts.")
        )
        if first_party:
            pytest.fail(
                f"clean tree cannot import {module_name}: first-party module "
                f"{missing!r} is missing from the tracked tree. This is the "
                "unversioned-production-code failure mode — commit the module, "
                "do not skip this test."
            )
        pytest.skip(f"third-party dependency {missing!r} not installed")


@pytest.mark.parametrize("module_name", CONSUMERS)
def test_tracked_consumer_imports_from_the_tracked_tree(module_name):
    assert _import_or_classify(module_name) is not None


def test_run_oanda_trend_script_resolves_from_the_tracked_tree():
    """The runner script is not a package module; load it via its spec."""
    path = REPO_ROOT / "scripts" / "run_oanda_trend.py"
    assert path.exists(), "scripts/run_oanda_trend.py missing from tree"
    spec = importlib.util.spec_from_file_location("_gate_run_oanda_trend", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "src" or missing.startswith(("src.", "dashboard.")):
            pytest.fail(
                f"run_oanda_trend.py cannot execute from a clean tree: "
                f"first-party module {missing!r} missing."
            )
        pytest.skip(f"third-party dependency {missing!r} not installed")


def test_the_exact_consumed_risk_symbols_exist_and_are_callable():
    """The live lane's risk gate and sizer — by import, not by grep."""
    from src.axiom_training.risk_runtime import (
        evaluate_runtime_risk,
        scale_target_units,
    )

    assert callable(evaluate_runtime_risk)
    assert callable(scale_target_units)


def test_scale_target_units_is_derisk_only_from_the_tracked_tree():
    """Behavioral smoke on the committed copy: scaling can only reduce."""
    from src.axiom_training.risk_runtime import scale_target_units

    want = {"EUR_USD": 1000, "USD_JPY": -2000}
    assert scale_target_units(want, 0.5) == {"EUR_USD": 500, "USD_JPY": -1000}
    # Invalid and out-of-range multipliers fail to ZERO, never amplify.
    assert scale_target_units(want, 1.7) == {"EUR_USD": 0, "USD_JPY": 0}
    assert scale_target_units(want, float("nan")) == {"EUR_USD": 0, "USD_JPY": 0}
    assert scale_target_units(want, None) == {"EUR_USD": 0, "USD_JPY": 0}


def test_axiom_training_package_is_tracked_not_machine_local():
    """The package must be visible to git, not a working-directory accident."""
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "src/axiom_training/"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert "src/axiom_training/__init__.py" in out, (
        "src/axiom_training is not tracked by git — the running system "
        "depends on unversioned code again"
    )
    assert "src/axiom_training/risk_runtime.py" in out
