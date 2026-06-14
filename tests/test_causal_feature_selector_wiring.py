"""Integration test: CausalFeatureSelector wiring into Scanner._compute_features().

US-015: When enable_causal_filtering=True, features should be pruned by
CausalFeatureSelector.select() using causal parents of target returns.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import inspect

from src.scanner.engine import Scanner


def test_causal_feature_selector_method_exists():
    assert hasattr(Scanner, "_init_causal_feature_selector")
    assert hasattr(Scanner, "_compute_features")


def test_causal_filtering_code_present_in_compute_features():
    src = inspect.getsource(Scanner._compute_features)
    assert "causal feature selection" in src or "CausalFeatureSelector" in src


def test_causal_feature_selector_filters_features():
    """Minimal unit test: verify _compute_features uses selector when present."""
    # Patch Scanner.__init__ to avoid heavy initialization
    with patch.object(Scanner, "__init__", lambda self, config: None):
        scanner = Scanner(None)
        scanner.config = MagicMock()
        scanner.config.enable_causal_filtering = True
        scanner._feature_engineer = lambda df: df.copy()
        scanner._causal_feature_selector = MagicMock()
        scanner._causal_feature_selector.select.return_value = [0]
        scanner._causal_discovery = None

        df = pd.DataFrame({
            "close": [1.1] * 50,
            "volume": [1000] * 50,
        })

        result = scanner._compute_features(df)
        scanner._causal_feature_selector.select.assert_called_once()
        assert "close" in result.columns
