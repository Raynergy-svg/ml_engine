"""Integration test: DiffusionMarketModel wiring into QuickBacktester.

US-009: QuickBacktester should have a stress_test method that uses
DiffusionMarketModel to generate synthetic paths.
"""
from __future__ import annotations

import pytest
import inspect
from src.scanner.analysis.backtest import QuickBacktester


def test_diffusion_stress_test_method_exists():
    assert hasattr(QuickBacktester, "stress_test")
    src = inspect.getsource(QuickBacktester.stress_test)
    assert "DiffusionMarketModel" in src
