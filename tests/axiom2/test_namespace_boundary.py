"""Axiom 2.0 namespace boundary tests."""

from __future__ import annotations

import importlib
import sys


def test_axiom2_namespace_import_does_not_load_oanda_runtime() -> None:
    """The Axiom namespace must be importable without loading OANDA runtime code."""
    importlib.import_module("src.axiom2")

    assert "src.brokers.oanda" not in sys.modules
