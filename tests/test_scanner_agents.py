"""
Tests for the three new scanner agents: momentum, session_timing, support_resistance.

Uses pytest style with a shared helper for building AgentDecisionContext.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.scanner.agents import AgentDecisionContext, ScannerAgentTeam
from src.scanner.results import PairAnalysis


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Return a minimal config namespace that ScannerAgentTeam accepts."""
    defaults = {
        "confidence_threshold": 0.55,
        "momentum_threshold": 0.40,
        "atr_sl_multiplier": 1.5,
        "atr_tp_multiplier": 2.5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_context(
    pair: str = "EUR_USD",
    direction: str = "LONG",
    atr_pips: float = 10.0,
    df_raw: pd.DataFrame | None = None,
    df_feat: pd.DataFrame | None = None,
    gate_details: dict | None = None,
    config=None,
    **analysis_overrides,
) -> AgentDecisionContext:
    """Build a minimal AgentDecisionContext with sensible defaults."""
    analysis = PairAnalysis(
        pair=pair,
        direction=direction,
        confidence=0.65,
        atr_pips=atr_pips,
        current_price=1.1000,
        **analysis_overrides,
    )
    if df_raw is None:
        df_raw = pd.DataFrame()
    if df_feat is None:
        df_feat = pd.DataFrame()
    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details=gate_details or {},
        config=config or _make_config(),
    )


def _make_team(config=None) -> ScannerAgentTeam:
    """Create a ScannerAgentTeam with no learned weights (uses base weights)."""
    with patch.object(ScannerAgentTeam, "_load_learned_weights", return_value={}):
        return ScannerAgentTeam(config or _make_config())


# ---------------------------------------------------------------------------
# Momentum agent tests
# ---------------------------------------------------------------------------


class TestMomentumAgent:
    """Tests for _evaluate_momentum."""

    def test_aligned_macd_and_roc_long(self):
        """When MACD histogram and ROC both favour LONG, score should be high."""
        df_feat = pd.DataFrame(
            {
                "macd_norm": [0.002],
                "macd_signal_norm": [0.001],
                "macd_hist_norm": [0.001],
                "roc_5": [0.5],
            }
        )
        ctx = _make_context(direction="LONG", df_feat=df_feat)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        assert verdict.name == "momentum"
        assert verdict.passed is True
        assert verdict.reason_code == "momentum_aligned"
        assert verdict.confidence_delta > 0
        assert verdict.metadata["macd_aligned"] is True
        assert verdict.metadata["roc_aligned"] is True

    def test_aligned_macd_and_roc_short(self):
        """When MACD histogram and ROC both favour SHORT, score should be high."""
        df_feat = pd.DataFrame(
            {
                "macd_norm": [-0.002],
                "macd_signal_norm": [-0.001],
                "macd_hist_norm": [-0.001],
                "roc_5": [-0.5],
            }
        )
        ctx = _make_context(direction="SHORT", df_feat=df_feat)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        assert verdict.passed is True
        assert verdict.reason_code == "momentum_aligned"
        assert verdict.metadata["macd_aligned"] is True
        assert verdict.metadata["roc_aligned"] is True

    def test_opposing_signals(self):
        """When MACD and ROC oppose the direction, score should be low."""
        df_feat = pd.DataFrame(
            {
                "macd_norm": [-0.003],
                "macd_signal_norm": [-0.001],
                "macd_hist_norm": [-0.002],
                "roc_5": [-0.8],
            }
        )
        ctx = _make_context(direction="LONG", df_feat=df_feat)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        assert verdict.passed is False
        assert verdict.reason_code == "momentum_opposed"
        assert verdict.confidence_delta < 0
        assert verdict.metadata["macd_aligned"] is False
        assert verdict.metadata["roc_aligned"] is False

    def test_mixed_signals(self):
        """When MACD aligns but ROC opposes, result is mixed."""
        df_feat = pd.DataFrame(
            {
                "macd_norm": [0.002],
                "macd_signal_norm": [0.001],
                "macd_hist_norm": [0.001],
                "roc_5": [-0.3],
            }
        )
        ctx = _make_context(direction="LONG", df_feat=df_feat)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        assert verdict.reason_code == "momentum_mixed"
        assert verdict.confidence_delta == 0.0
        assert verdict.metadata["macd_aligned"] is True
        assert verdict.metadata["roc_aligned"] is False

    def test_missing_columns_fallback(self):
        """When feature columns are missing, agent falls back to defaults without error."""
        df_feat = pd.DataFrame({"irrelevant_col": [1.0]})
        ctx = _make_context(direction="LONG", df_feat=df_feat)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        # Should not raise; returns a valid verdict with baseline score
        assert verdict.name == "momentum"
        assert 0.0 <= verdict.score <= 1.0

    def test_roc_fallback_to_close_prices(self):
        """When roc_5 and roc are missing, momentum calculates ROC from close prices."""
        df_feat = pd.DataFrame({"macd_norm": [0.001], "macd_signal_norm": [0.0005], "macd_hist_norm": [0.0005]})
        closes = [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1060]
        df_raw = pd.DataFrame({"close": closes})
        ctx = _make_context(direction="LONG", df_feat=df_feat, df_raw=df_raw)
        team = _make_team()

        verdict = team._evaluate_momentum(ctx)

        # ROC computed from close prices should be positive (rising)
        assert verdict.metadata["roc"] > 0
        assert verdict.metadata["roc_aligned"] is True


# ---------------------------------------------------------------------------
# Session timing agent tests
# ---------------------------------------------------------------------------


class TestSessionTimingAgent:
    """Tests for _evaluate_session_timing."""

    def _make_ctx_with_timestamp(self, hour: int, weekday: int, pair: str = "EUR_USD"):
        """Build a context whose df_raw index has a specific UTC hour/weekday."""
        # weekday: 0=Monday ... 6=Sunday
        # Pick a date that matches the desired weekday
        # 2026-03-16 is Monday; offset to reach desired weekday
        base = datetime(2026, 3, 16, hour, 30, tzinfo=timezone.utc)
        # Shift to desired weekday (base is Monday = 0)
        from datetime import timedelta

        delta_days = weekday  # Monday=0
        ts = base + timedelta(days=delta_days)
        idx = pd.DatetimeIndex([ts])
        df_raw = pd.DataFrame({"close": [1.1000]}, index=idx)
        return _make_context(pair=pair, direction="LONG", df_raw=df_raw)

    def test_london_ny_overlap(self):
        """During London/NY overlap (13-16 UTC) on a weekday, score is high."""
        ctx = self._make_ctx_with_timestamp(hour=14, weekday=1)  # Tuesday 14:00 UTC
        team = _make_team()

        verdict = team._evaluate_session_timing(ctx)

        assert verdict.passed is True
        assert verdict.score >= 0.65
        assert verdict.reason_code == "session_optimal"
        assert verdict.confidence_delta > 0
        assert "London/NY overlap" in verdict.metadata["active_sessions"]

    def test_london_session_eur_pair(self):
        """EUR pair during London hours (not overlap) scores well."""
        ctx = self._make_ctx_with_timestamp(hour=9, weekday=2, pair="EUR_GBP")  # Wed 09 UTC
        team = _make_team()

        verdict = team._evaluate_session_timing(ctx)

        assert verdict.passed is True
        assert "London" in verdict.metadata["active_sessions"]
        # EUR/GBP gets the higher London bonus
        assert verdict.score >= 0.60

    def test_off_hours(self):
        """Outside all major sessions the score should be low."""
        ctx = self._make_ctx_with_timestamp(hour=23, weekday=3)  # Thu 23:00 UTC
        team = _make_team()

        verdict = team._evaluate_session_timing(ctx)

        assert verdict.passed is False
        assert verdict.reason_code == "session_weak"
        assert verdict.confidence_delta < 0
        assert "off-hours" in verdict.metadata["active_sessions"]

    def test_weekend_penalty(self):
        """Weekend trading should incur a penalty."""
        ctx = self._make_ctx_with_timestamp(hour=14, weekday=5)  # Saturday 14:00
        team = _make_team()

        verdict = team._evaluate_session_timing(ctx)

        assert "weekend" in verdict.metadata["active_sessions"]
        # Even during overlap hours, weekend penalty should drag score down
        assert verdict.score < 0.65

    def test_tokyo_session_jpy_pair(self):
        """JPY pair during Tokyo session should get the higher bonus."""
        ctx = self._make_ctx_with_timestamp(hour=3, weekday=0, pair="USD_JPY")  # Mon 03 UTC
        team = _make_team()

        verdict = team._evaluate_session_timing(ctx)

        assert verdict.passed is True
        assert "Tokyo" in verdict.metadata["active_sessions"]
        assert verdict.score >= 0.60

    def test_fallback_to_system_clock(self):
        """When df_raw has no datetime index, agent falls back to datetime.now(UTC)."""
        ctx = _make_context(direction="LONG", df_raw=pd.DataFrame({"close": [1.1]}))
        team = _make_team()

        mock_dt = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)  # Wednesday overlap
        with patch("src.scanner.agents.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
            verdict = team._evaluate_session_timing(ctx)

        assert verdict.name == "session_timing"
        assert 0.0 <= verdict.score <= 1.0


# ---------------------------------------------------------------------------
# Support/resistance agent tests
# ---------------------------------------------------------------------------


class TestSupportResistanceAgent:
    """Tests for _evaluate_support_resistance."""

    def _make_sr_data(self, n: int = 60, support_at: float = 1.0950, resistance_at: float = 1.1100):
        """Build a df_raw with a clear support and resistance level via 5-bar pivots.

        Creates a price series with dips at support_at and spikes at resistance_at
        so the pivot detection picks them up.
        """
        close = np.linspace(1.0980, 1.1050, n)
        high = close + 0.0010
        low = close - 0.0010

        # Plant a pivot high (resistance): bar at index 30 should be highest
        # among its 2-bar neighbours on each side
        high[30] = resistance_at
        high[28] = resistance_at - 0.0020
        high[29] = resistance_at - 0.0015
        high[31] = resistance_at - 0.0015
        high[32] = resistance_at - 0.0020

        # Plant a pivot low (support): bar at index 20 should be lowest
        low[20] = support_at
        low[18] = support_at + 0.0020
        low[19] = support_at + 0.0015
        low[21] = support_at + 0.0015
        low[22] = support_at + 0.0020

        return pd.DataFrame({"close": close, "high": high, "low": low})

    def test_long_near_support(self):
        """LONG entry near support with room to resistance should score well."""
        # Current price near support, resistance far away
        df_raw = self._make_sr_data(support_at=1.1045, resistance_at=1.1200)
        # Set last close near the support level
        df_raw.loc[df_raw.index[-1], "close"] = 1.1048
        ctx = _make_context(
            direction="LONG",
            df_raw=df_raw,
            atr_pips=10.0,
            current_price=1.1048,
        )
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is not None
        assert verdict.name == "support_resistance"
        assert verdict.passed is True
        assert verdict.score >= 0.60
        assert verdict.metadata["dist_to_support_atr"] < 2.0

    def test_resistance_blocking_long(self):
        """LONG with resistance immediately above should score poorly."""
        df_raw = self._make_sr_data(support_at=1.0800, resistance_at=1.1055)
        df_raw.loc[df_raw.index[-1], "close"] = 1.1052
        ctx = _make_context(
            direction="LONG",
            df_raw=df_raw,
            atr_pips=10.0,
            current_price=1.1052,
        )
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is not None
        # Resistance right above on a LONG should drag score down
        assert verdict.score < 0.50 or verdict.metadata["dist_to_resistance_atr"] < 1.0

    def test_insufficient_data_returns_none(self):
        """With fewer than 30 bars the agent should return None."""
        df_raw = pd.DataFrame(
            {
                "close": np.ones(20),
                "high": np.ones(20) + 0.001,
                "low": np.ones(20) - 0.001,
            }
        )
        ctx = _make_context(direction="LONG", df_raw=df_raw)
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is None

    def test_missing_columns_returns_none(self):
        """If required columns are absent, returns None instead of raising."""
        df_raw = pd.DataFrame({"close": np.ones(40)})  # missing high/low
        ctx = _make_context(direction="LONG", df_raw=df_raw)
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is None

    def test_short_near_resistance(self):
        """SHORT entry near resistance with room to support should score well."""
        df_raw = self._make_sr_data(support_at=1.0850, resistance_at=1.1055)
        df_raw.loc[df_raw.index[-1], "close"] = 1.1052
        ctx = _make_context(
            direction="SHORT",
            df_raw=df_raw,
            atr_pips=10.0,
            current_price=1.1052,
        )
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is not None
        assert verdict.passed is True
        assert verdict.score >= 0.60

    def test_no_pivots_returns_none(self):
        """If price is flat with no swing highs/lows, returns None."""
        n = 40
        flat = np.ones(n) * 1.1000
        df_raw = pd.DataFrame({"close": flat, "high": flat, "low": flat})
        ctx = _make_context(direction="LONG", df_raw=df_raw)
        team = _make_team()

        verdict = team._evaluate_support_resistance(ctx)

        assert verdict is None
