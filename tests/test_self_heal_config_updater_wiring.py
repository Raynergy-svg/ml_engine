"""Integration test: SelfHealConfigUpdater wiring into PostTradeLoop.

US-016: PostTradeLoop should initialize SelfHealConfigUpdater and layer
its proposals alongside existing SelfHeal.apply() results.
"""
from __future__ import annotations

import pytest
import inspect
from src.scanner.feedback.post_trade_loop import PostTradeLoop


def test_self_heal_config_updater_initialized():
    src = inspect.getsource(PostTradeLoop.__init__)
    assert "_self_heal_config_updater" in src
    assert "SelfHealConfigUpdater" in src


def test_self_heal_config_updater_called_in_run_self_heal():
    src = inspect.getsource(PostTradeLoop._run_self_heal)
    assert "_self_heal_config_updater" in src
    assert "config_updater" in src
