"""US-513: Model staleness banner + uncertainty hard-block enforcement.

Validates the .claude/rules/trading.md (2026-04-15) rule:
    max_component_age_days > 7 ⇒ uncertainty hard-block tightens 0.45 → 0.35
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest

from src.scanner.engine import (
    STALENESS_AGE_THRESHOLD_DAYS,
    STALENESS_UNCERTAINTY_THRESHOLD,
    _evaluate_staleness_uncertainty_block,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper — covers AC #5 + #6 (reason code + log line)
# ─────────────────────────────────────────────────────────────────────────────


def test_8day_old_model_uncertainty_040_rejects(caplog):
    """8d staleness + uncertainty=0.40 must hard-block with the canonical reason."""
    caplog.set_level(logging.WARNING, logger="src.scanner.engine")
    blocked, reason = _evaluate_staleness_uncertainty_block(
        uncertainty_score=0.40,
        staleness_days=8.0,
    )
    assert blocked is True
    assert reason == "stale_models_uncertainty_block"
    # Canonical log line: "HARD-BLOCK: models stale {n}d, uncertainty {u:.2f} > 0.35"
    log_text = caplog.text
    assert "HARD-BLOCK" in log_text
    assert "stale 8d" in log_text
    assert "0.40" in log_text
    assert "0.35" in log_text


def test_3day_old_model_uncertainty_040_passes_old_threshold():
    """3d staleness + uncertainty=0.40 passes the staleness gate (old 0.45 path)."""
    blocked, reason = _evaluate_staleness_uncertainty_block(
        uncertainty_score=0.40,
        staleness_days=3.0,
    )
    assert blocked is False
    assert reason is None


def test_8day_old_model_uncertainty_030_below_tightened_threshold_passes():
    """8d staleness + uncertainty=0.30 still below tightened 0.35 — no block."""
    blocked, reason = _evaluate_staleness_uncertainty_block(
        uncertainty_score=0.30,
        staleness_days=8.0,
    )
    assert blocked is False
    assert reason is None


def test_threshold_constants():
    assert STALENESS_AGE_THRESHOLD_DAYS == 7.0
    assert STALENESS_UNCERTAINTY_THRESHOLD == 0.35


def test_exactly_7day_does_not_trigger_strict_inequality():
    """Rule is > 7d, not >= 7d. 7.0 exact must NOT trigger."""
    blocked, _ = _evaluate_staleness_uncertainty_block(
        uncertainty_score=0.99,
        staleness_days=7.0,
    )
    assert blocked is False


# ─────────────────────────────────────────────────────────────────────────────
# Integration — Scanner instance method routes to the helper (AC #1)
# ─────────────────────────────────────────────────────────────────────────────


def test_scanner_get_model_staleness_days_uses_freshness_module():
    """Scanner.get_model_staleness_days() returns oldest_age_days from freshness."""
    from src.scanner.engine import Scanner

    fake = SimpleNamespace(get_model_staleness_days=Scanner.get_model_staleness_days)
    with mock.patch(
        "src.scanner.automation.model_freshness.get_model_freshness",
        return_value={"oldest_age_days": 11.4, "status": "STALE"},
    ):
        # Bound method-like call on a bare namespace; method is pure (no self use).
        result = Scanner.get_model_staleness_days(fake)  # type: ignore[arg-type]
    assert result == pytest.approx(11.4)


def test_scanner_evaluate_staleness_block_hard_blocks(caplog):
    """Scanner.evaluate_staleness_uncertainty_block() returns the canonical reason."""
    from src.scanner.engine import Scanner

    caplog.set_level(logging.WARNING, logger="src.scanner.engine")
    fake = SimpleNamespace(
        get_model_staleness_days=lambda: 9.0,
    )
    blocked, reason = Scanner.evaluate_staleness_uncertainty_block(  # type: ignore[arg-type]
        fake, uncertainty_score=0.42
    )
    assert blocked is True
    assert reason == "stale_models_uncertainty_block"
    assert "HARD-BLOCK" in caplog.text


def test_scanner_evaluate_staleness_block_passes_when_fresh():
    from src.scanner.engine import Scanner

    fake = SimpleNamespace(get_model_staleness_days=lambda: 2.0)
    blocked, reason = Scanner.evaluate_staleness_uncertainty_block(  # type: ignore[arg-type]
        fake, uncertainty_score=0.80
    )
    assert blocked is False
    assert reason is None


# ─────────────────────────────────────────────────────────────────────────────
# Banner widget renders + DashboardSnapshot field present
# ─────────────────────────────────────────────────────────────────────────────


def test_staleness_banner_widget_importable():
    from src.tui.widgets.staleness_banner import StalenessBanner, STALENESS_THRESHOLD_DAYS

    assert STALENESS_THRESHOLD_DAYS == 7.0
    assert StalenessBanner is not None


def test_dashboard_snapshot_has_max_component_age_days():
    from src.tui.data_provider import DashboardSnapshot

    snap = DashboardSnapshot()
    assert hasattr(snap, "max_component_age_days")
    assert snap.max_component_age_days == 0.0


def test_banner_visible_when_stale_hidden_when_fresh():
    """Integration-style: simulate snapshot → banner display flips correctly."""
    from src.tui.data_provider import DashboardSnapshot
    from src.tui.widgets.staleness_banner import StalenessBanner

    banner = StalenessBanner()
    # Simulate on_mount
    banner.display = False

    snap_fresh = DashboardSnapshot()
    snap_fresh.max_component_age_days = 3.0
    banner.update_from_snapshot(snap_fresh)
    assert banner.display is False

    snap_stale = DashboardSnapshot()
    snap_stale.max_component_age_days = 12.5
    banner.update_from_snapshot(snap_stale)
    assert banner.display is True
    assert banner.staleness_days == pytest.approx(12.5)
