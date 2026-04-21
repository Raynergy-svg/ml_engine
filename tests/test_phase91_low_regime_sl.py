"""Tests for US-501: LOW regime sl_mult default and load-time validation."""

import pytest

from src.scanner.config import ScannerConfig


def test_default_low_sl_mult_is_1_2():
    """AC-1: Default config LOW regime sl_mult must equal 1.2."""
    config = ScannerConfig()
    low_sl_mult = config.regime_atr_multipliers["LOW"]["sl_mult"]
    assert low_sl_mult == 1.2, f"Expected 1.2, got {low_sl_mult}"


def test_low_sl_mult_below_1_2_raises():
    """AC-2: Setting LOW sl_mult to 0.8 raises ValueError at construction."""
    with pytest.raises(ValueError, match="LOW regime sl_mult must be >= 1.2"):
        ScannerConfig(
            regime_atr_multipliers={
                "LOW": {"sl_mult": 0.8, "tp_mult": 1.2},
                "NORMAL": {"sl_mult": 1.0, "tp_mult": 1.5},
                "HIGH": {"sl_mult": 1.1, "tp_mult": 2.0},
                "EXTREME": {"sl_mult": 1.2, "tp_mult": 2.8},
            }
        )


def test_low_sl_mult_at_boundary_passes():
    """Exactly 1.2 is the minimum valid value — must not raise."""
    config = ScannerConfig(
        regime_atr_multipliers={
            "LOW": {"sl_mult": 1.2, "tp_mult": 1.2},
            "NORMAL": {"sl_mult": 1.0, "tp_mult": 1.5},
            "HIGH": {"sl_mult": 1.1, "tp_mult": 2.0},
            "EXTREME": {"sl_mult": 1.2, "tp_mult": 2.8},
        }
    )
    assert config.regime_atr_multipliers["LOW"]["sl_mult"] == 1.2


def test_low_regime_sl_pips_wider_than_0_8_multiplier():
    """AC-3: LOW sl_mult=1.2 produces wider sl_pips than the old 0.8 value.

    Simulates the sl_pips formula from engine.py _scan_pair:
        sl_pips = max(min_sl_pips, min(atr_pips * sl_mult, max_sl_pips))
    """
    config = ScannerConfig()
    atr_pips = 20.0
    low_sl_mult_new = config.regime_atr_multipliers["LOW"]["sl_mult"]  # 1.2
    low_sl_mult_old = 0.8

    sl_pips_new = max(config.min_sl_pips, min(atr_pips * low_sl_mult_new, config.max_sl_pips))
    sl_pips_old = max(config.min_sl_pips, min(atr_pips * low_sl_mult_old, config.max_sl_pips))

    assert sl_pips_new > sl_pips_old, (
        f"LOW regime sl_pips with mult=1.2 ({sl_pips_new}) should be wider "
        f"than with mult=0.8 ({sl_pips_old})"
    )


def test_low_sl_mult_missing_key_raises():
    """Missing LOW key with no sl_mult defaults to 0 which is < 1.2, must raise."""
    with pytest.raises(ValueError, match="LOW regime sl_mult must be >= 1.2"):
        ScannerConfig(
            regime_atr_multipliers={
                "LOW": {"tp_mult": 1.2},  # sl_mult absent → defaults to 0 in validation
                "NORMAL": {"sl_mult": 1.0, "tp_mult": 1.5},
                "HIGH": {"sl_mult": 1.1, "tp_mult": 2.0},
                "EXTREME": {"sl_mult": 1.2, "tp_mult": 2.8},
            }
        )
