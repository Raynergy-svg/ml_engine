"""Unit tests for FX guardrails (Tier-1)."""

from __future__ import annotations

import unittest
from datetime import datetime
from tempfile import TemporaryDirectory

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from fx_guardrails import (
    FxDailyState,
    can_open_new_trade,
    check_daily_stops,
    confidence_band,
    load_fx_policy,
    load_state,
    profit_stop_pct_for_band,
    session_date_str,
    save_state,
)


class TestFxGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "fx": {
                "enabled": True,
                "tier": 1,
                "instruments": ["EUR_USD", "GBP_USD", "USD_JPY"],
                "granularity": "M5",
                "horizon_bars": 1,
                "timezone": "America/New_York",
                "session": {"start": "08:00", "end": "11:30", "force_flat_at": "11:55", "reset_at": "08:00"},
                "limits": {"max_open_positions": 1, "max_entries_per_day": 1},
                "risk": {"risk_per_trade_pct": 0.005, "daily_loss_stop_pct": 0.10, "atr_stop_mult": 1.5, "rr_take_profit": 1.5},
                "costs": {
                    "max_spread_pips": {"EUR_USD": 1.2},
                    "slippage_pips_min": 0.2,
                    "slippage_pips_spread_mult": 0.25,
                    "spread_fallback_pips": {"EUR_USD": 1.0},
                },
                "confidence": {"low_lt": 0.60, "high_gte": 0.75, "profit_stop_pct_by_band": {"medium": 0.30, "high": 0.30}},
                "execution": {"require_confirmation": True, "practice_only": True},
            }
        }

    def test_confidence_band_thresholds(self) -> None:
        policy = load_fx_policy(self.cfg)
        self.assertEqual(confidence_band(policy, 0.0), "low")
        self.assertEqual(confidence_band(policy, 0.59), "low")
        self.assertEqual(confidence_band(policy, 0.60), "medium")
        self.assertEqual(confidence_band(policy, 0.74), "medium")
        self.assertEqual(confidence_band(policy, 0.75), "high")
        self.assertEqual(confidence_band(policy, 1.0), "high")

    def test_profit_stop_pct_for_band(self) -> None:
        policy = load_fx_policy(self.cfg)
        self.assertEqual(profit_stop_pct_for_band(policy, "high"), 0.30)
        self.assertEqual(profit_stop_pct_for_band(policy, "medium"), 0.30)
        self.assertIsNone(profit_stop_pct_for_band(policy, "low"))

    def test_daily_stops_loss_and_profit(self) -> None:
        policy = load_fx_policy(self.cfg)

        hit, reason, kind = check_daily_stops(policy, drawdown_pct=-0.10, realized_pct=0.0, confidence_band="medium")
        self.assertTrue(hit)
        self.assertIn("loss stop", (reason or "").lower())
        self.assertEqual(kind, "loss")

        hit, reason, kind = check_daily_stops(policy, drawdown_pct=-0.09, realized_pct=0.30, confidence_band="medium")
        self.assertTrue(hit)
        self.assertIn("profit stop", (reason or "").lower())
        self.assertEqual(kind, "profit")

        hit, reason, kind = check_daily_stops(policy, drawdown_pct=-0.09, realized_pct=0.20, confidence_band="high")
        self.assertFalse(hit)
        self.assertIsNone(reason)
        self.assertIsNone(kind)

        hit, reason, kind = check_daily_stops(policy, drawdown_pct=-0.09, realized_pct=0.30, confidence_band="high")
        self.assertTrue(hit)
        self.assertIn("profit stop", (reason or "").lower())
        self.assertEqual(kind, "profit")

    def test_state_persistence_round_trip(self) -> None:
        policy = load_fx_policy(self.cfg)
        with TemporaryDirectory() as d:
            cfg = {"paths": {"log_dir": d}}
            st = FxDailyState(
                date="2025-01-02",
                start_nav=10_000.0,
                start_balance=10_000.0,
                entries_today=1,
                disabled_reason="Daily loss stop hit (-10.0%)",
                disabled_kind="loss",
            )
            p = save_state(cfg, policy, st)
            self.assertTrue(p.exists())
            st2 = load_state(cfg, policy, now=datetime(2025, 1, 2, 15, 0, tzinfo=(ZoneInfo("UTC") if ZoneInfo else None)))
            self.assertEqual(st2.date, "2025-01-02")
            self.assertEqual(st2.entries_today, 1)
            self.assertEqual(st2.start_nav, 10_000.0)
            self.assertEqual(st2.disabled_kind, "loss")

    def test_can_open_new_trade_outside_session(self) -> None:
        if ZoneInfo is None:
            self.skipTest("zoneinfo not available")

        policy = load_fx_policy(self.cfg)
        state = FxDailyState(date="2025-01-02")

        # 12:00 UTC == 07:00 EST (outside 08:00-11:30)
        now = datetime(2025, 1, 2, 12, 0, tzinfo=ZoneInfo("UTC"))
        ok, reason = can_open_new_trade(policy, state, now=now)
        self.assertFalse(ok)
        self.assertIn("outside", reason.lower())

    def test_session_date_respects_reset_at(self) -> None:
        if ZoneInfo is None:
            self.skipTest("zoneinfo not available")

        policy = load_fx_policy(self.cfg)

        # 12:30 UTC == 07:30 EST on Jan 2, but reset_at is 08:00 => session date is Jan 1.
        now = datetime(2025, 1, 2, 12, 30, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(session_date_str(policy, now=now), "2025-01-01")

        # 13:05 UTC == 08:05 EST => session date is Jan 2.
        now2 = datetime(2025, 1, 2, 13, 5, tzinfo=ZoneInfo("UTC"))
        self.assertEqual(session_date_str(policy, now=now2), "2025-01-02")


if __name__ == "__main__":
    unittest.main()
# — Raynergy-svg —
