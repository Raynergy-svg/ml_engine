"""Axiom 2.0 namespace boundary tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBPROCESS_TIMEOUT_SECONDS = 10
_OANDA_RUNTIME_MODULE_PREFIXES = (
    "src.brokers.oanda",
    "src.brokers.oanda_v20",
)

_NAMESPACE_IMPORT_SCRIPT = r"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


package_name = sys.argv[1]
checkout_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(checkout_root))

module = importlib.import_module(package_name)
module_file = getattr(module, "__file__", None)
if module_file is None:
    raise AssertionError(f"{package_name} has no importable file")

module_path = Path(module_file).resolve()
try:
    module_path.relative_to(checkout_root / "src")
except ValueError as exc:
    raise AssertionError(
        f"{package_name} imported from {module_path}, outside checkout {checkout_root}"
    ) from exc

oanda_runtime_modules = sorted(
    module_name
    for module_name in sys.modules
    if any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in ("src.brokers.oanda", "src.brokers.oanda_v20")
    )
)
if oanda_runtime_modules:
    raise AssertionError(
        f"{package_name} imported OANDA runtime modules: {oanda_runtime_modules!r}"
    )
"""


def _run_namespace_import(
    package_name: str,
    *,
    checkout_root: Path = _REPO_ROOT,
) -> None:
    """Import one Axiom package in a clean interpreter and report diagnostics."""
    command = [
        sys.executable,
        "-S",
        "-c",
        _NAMESPACE_IMPORT_SCRIPT,
        package_name,
        str(checkout_root),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ""
    environment["PYTHONNOUSERSITE"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=checkout_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise AssertionError(
            f"namespace import timed out after {_SUBPROCESS_TIMEOUT_SECONDS}s: {command!r}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc

    if completed.returncode != 0:
        raise AssertionError(
            f"namespace import failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


@pytest.mark.parametrize("package_name", ("src.axiom2", "src.axiom2.contracts"))
def test_axiom2_namespace_import_does_not_load_oanda_runtime(
    package_name: str,
) -> None:
    """Each Axiom namespace package must import cleanly without OANDA runtime code."""
    _run_namespace_import(package_name)


def test_namespace_check_isolated_from_parent_oanda_module() -> None:
    """A parent-process OANDA import must not affect fresh namespace checks."""
    module_name = "src.brokers.oanda"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = types.ModuleType(module_name)
    try:
        _run_namespace_import("src.axiom2")
        _run_namespace_import("src.axiom2.contracts")
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_namespace_guard_detects_prohibited_oanda_import(tmp_path: Path) -> None:
    """The isolated guard must fail when a temporary Axiom copy imports OANDA."""
    fixture_root = tmp_path / "checkout"
    fixture_axiom = fixture_root / "src" / "axiom2"
    shutil.copytree(_REPO_ROOT / "src" / "axiom2", fixture_axiom)

    fixture_brokers = fixture_root / "src" / "brokers"
    fixture_brokers.mkdir(parents=True)
    (fixture_brokers / "oanda.py").write_text(
        "RUNTIME_FIXTURE = True\n",
        encoding="utf-8",
    )

    initializer = fixture_axiom / "__init__.py"
    initializer.write_text(
        initializer.read_text(encoding="utf-8") + "\nimport src.brokers.oanda\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=r"src\.brokers\.oanda"):
        _run_namespace_import("src.axiom2", checkout_root=fixture_root)
