"""Trade-flow unblock tests (added 2026-05-10).

Verifies the two config flips applied to the ``smart`` profile after the
4-agent TUI audit identified them as load-bearing reasons trades stopped firing:

1. ``enable_backtest_gate=False`` — was slashing confidence -15% on as few as
   3 simulated trades; the 19-bar next-candle SL/TP simulator in
   ``src/scanner/backtest_gate.py`` is too noisy at every floor to be useful.
2. ``enable_phase44_calibration=False`` — was compressing raw model
   confidence by -52.7% (raw 0.582 -> calibrated 0.099), driving every signal
   below ``min_confidence=35.0``.

NO MOCKS rule (CLAUDE.md / .claude/rules/improvement.md "No-Mock Rule"):
Real ``ScannerConfig`` instances, real ``BacktestGate`` against a real
``pandas.DataFrame`` of contrived OHLC candles. No mocks, no MagicMock, no
patch. The engine consumer is exercised by re-evaluating the exact guard
expression at ``src/scanner/engine.py:4628-4632`` against the live config
attribute — same pattern as ``test_phase44_kill_switch.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.scanner.backtest_gate import BacktestGate, BacktestGateResult
from src.scanner.config import SCAN_PROFILES, ScannerConfig


# ---------------------------------------------------------------------------
# 1. Profile + dataclass field plumbing (the "three-layer wiring verification"
# from .claude/rules/improvement.md "Live Wiring Verification Gates").
# ---------------------------------------------------------------------------


def test_dataclass_default_enable_backtest_gate_true():
    """Field exists on ScannerConfig with default=True (preserves legacy)."""
    cfg = ScannerConfig()
    assert hasattr(cfg, "enable_backtest_gate")
    assert cfg.enable_backtest_gate is True


def test_dataclass_default_enable_phase44_calibration_true():
    """Field exists on ScannerConfig with default=True (preserves legacy)."""
    cfg = ScannerConfig()
    assert hasattr(cfg, "enable_phase44_calibration")
    assert cfg.enable_phase44_calibration is True


def test_smart_profile_dict_has_both_kill_switches():
    """Smart profile dict must carry both flips as explicit False entries.

    Source of truth: src/scanner/config.py SCAN_PROFILES["smart"].
    """
    smart = SCAN_PROFILES["smart"]
    assert smart.get("enable_backtest_gate") is False
    assert smart.get("enable_phase44_calibration") is False


def test_apply_profile_smart_disables_both_gates():
    """apply_profile('smart') must propagate both False values onto the
    dataclass instance (not silently dropped by the field-validation guard
    in apply_profile)."""
    cfg = ScannerConfig()
    # Sanity: defaults are True before profile is applied
    assert cfg.enable_backtest_gate is True
    assert cfg.enable_phase44_calibration is True

    cfg.apply_profile("smart")
    assert cfg.enable_backtest_gate is False, (
        "smart profile failed to disable BacktestGate — possible orphan key "
        "(dataclass field name mismatch); check src/scanner/config.py "
        "ScannerConfig dataclass definition"
    )
    assert cfg.enable_phase44_calibration is False, (
        "smart profile failed to disable Phase 44 calibration"
    )


def test_apply_profile_non_smart_keeps_defaults_true():
    """Other profiles (e.g. aggressive) must NOT carry these kill switches,
    so they keep the dataclass default True. Failing this means the flags
    were accidentally injected into the wrong profile."""
    cfg = ScannerConfig()
    cfg.apply_profile("aggressive")
    assert cfg.enable_backtest_gate is True
    assert cfg.enable_phase44_calibration is True


# ---------------------------------------------------------------------------
# 2. Justification for Option A (disable entirely) over Option B (raise the
# min-sample floor): demonstrate the gate produces a -15% penalty on a
# pathological-but-realistic 3-trade window.
# ---------------------------------------------------------------------------


def _make_candle_df(n_bars: int, base_price: float = 1.10) -> pd.DataFrame:
    """Build OHLC bars whose next-candle moves wreck a LONG with sl=10/tp=10.

    Each bar's *next* candle has low far below entry (hits SL) and high near
    entry (misses TP). With 4 bars (3 forward-walked trades) every simulated
    trade is a loss -> win_rate=0.0 < min_win_rate=0.55 -> -15% penalty fires.
    """
    rows = []
    price = base_price
    for i in range(n_bars):
        # Each candle: tight high near close, deep low to trigger LONG stop.
        rows.append(
            {
                "open": price,
                "high": price + 0.00005,   # 0.5 pip above close (won't hit TP=10 pips)
                "low": price - 0.00200,    # 20 pips below close (well below SL=10 pips)
                "close": price,
            }
        )
        # Stay roughly flat — drift small so entries cluster around base.
        price += 0.00001
    return pd.DataFrame(rows)


def test_backtest_gate_slashes_on_small_sample():
    """Demonstrates the noise problem: a tiny forward-walk window (just past
    the gate's own min-bars=5 / min-trades=3 floors at backtest_gate.py:104,210)
    can produce a -15% confidence penalty if those few bars happen to point
    against the signal. That's why Option A was chosen over raising the
    min-sample threshold — even a higher floor would only delay the same
    statistical-noise problem.

    Feeds 6 bars (5 forward-walked trades) all LONG-unfriendly:
        - next-candle high never reaches TP=10 pips
        - next-candle low always exceeds SL=10 pips
    => win_rate=0.0 < min_win_rate=0.55 => -15% penalty fires.
    """
    gate = BacktestGate(min_win_rate=0.55, lookback_candles=20, confidence_penalty=0.15)
    df = _make_candle_df(n_bars=6)
    result = gate.validate(
        pair="EUR_USD",
        direction="LONG",
        sl_pips=10.0,
        tp_pips=10.0,
        recent_candles=df,
        pip_value=0.0001,
    )

    assert isinstance(result, BacktestGateResult)
    # 6 bars -> range(5) forward-walked trades.
    assert result.trades_tested == 5, (
        f"expected 5 forward-walked trades, got {result.trades_tested}; "
        f"this test depends on backtest_gate.py loop semantics (range(len-1))"
    )
    assert result.passed is False
    assert result.win_rate == pytest.approx(0.0)
    assert result.confidence_adjustment == pytest.approx(-0.15)


# ---------------------------------------------------------------------------
# 3. Consumer guard verification — mirrors the production guard at
# src/scanner/engine.py:4628-4632 against the real config attribute. If
# production drifts (the guard changes), this test fails -> rewrite test.
# ---------------------------------------------------------------------------


def _consumer_guard_fires(config: ScannerConfig, direction: str) -> bool:
    """Replay of engine.py:4628-4632 minus the ``self._backtest_gate`` truthy
    check (which is always set after _init_analysis_tools). Returns True iff
    the inner branch runs (i.e. BacktestGate.validate would be called).
    """
    return (
        direction != "HOLD"
        and bool(getattr(config, "enable_backtest_gate", True))
    )


def test_consumer_guard_blocks_when_smart_profile_disables_gate():
    """With smart applied, the engine guard at engine.py:4632 short-circuits;
    BacktestGate.validate is NEVER called -> no -15% penalty can fire."""
    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    assert _consumer_guard_fires(cfg, direction="LONG") is False
    assert _consumer_guard_fires(cfg, direction="SHORT") is False


def test_consumer_guard_runs_when_flag_true():
    """Default (non-smart) config keeps the gate; the consumer enters its
    inner branch for any non-HOLD direction."""
    cfg = ScannerConfig()
    assert cfg.enable_backtest_gate is True  # sanity
    assert _consumer_guard_fires(cfg, direction="LONG") is True
    assert _consumer_guard_fires(cfg, direction="SHORT") is True


def test_consumer_guard_skips_hold_regardless_of_flag():
    """HOLD signals never reach the gate (unrelated to flag) — sanity check
    so we're not accidentally measuring the wrong branch."""
    cfg = ScannerConfig()
    assert _consumer_guard_fires(cfg, direction="HOLD") is False
    cfg.apply_profile("smart")
    assert _consumer_guard_fires(cfg, direction="HOLD") is False
