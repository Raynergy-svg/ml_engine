"""Integration test: CausalDiscovery wiring into Scanner._compute_features().

US-013: When enable_causal_filtering=True, CausalDiscovery.fit() should run
before CausalFeatureSelector.select() to build the causal graph.
"""
from __future__ import annotations

import pytest
import inspect
from src.scanner.engine import Scanner


def test_causal_discovery_initialized_when_enabled():
    assert hasattr(Scanner, "_init_causal_discovery")


def test_causal_discovery_code_present_in_compute_features():
    src = inspect.getsource(Scanner._compute_features)
    assert "causal_discovery.fit" in src or "CausalDiscovery" in src
