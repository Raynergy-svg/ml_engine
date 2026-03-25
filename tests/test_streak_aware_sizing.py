"""
Test suite for streak-aware anti-martingale position sizing.

Tests cover:
- StreakConfig validation
- StreakTracker core functionality
- Integration with AdaptivePositionSizer
- State persistence (save/load)
- Edge cases and boundary conditions
"""

import json
import os
import tempfile
import unittest
from typing import Optional

from src.scanner.adaptive_position_sizing import (
    AdaptivePositionSizer,
    AdaptivePositionSizingConfig,
    StreakConfig,
    StreakTracker,
)


class TestStreakConfigValidation(unittest.TestCase):
    """Test StreakConfig validation."""

    def test_valid_config_defaults(self):
        """Test that default StreakConfig is valid."""
        config = StreakConfig()
        self.assertEqual(config.win_multiplier, 1.3)
        self.assertEqual(config.loss_divisor, 1.8)
        self.assertEqual(config.max_boost, 2.0)
        self.assertEqual(config.min_floor, 0.3)
        self.assertTrue(config.reset_on_regime_change)

    def test_valid_custom_config(self):
        """Test that custom valid config is accepted."""
        config = StreakConfig(
            win_multiplier=1.5,
            loss_divisor=2.0,
            max_boost=2.5,
            min_floor=0.25,
            reset_on_regime_change=False,
        )
        self.assertEqual(config.win_multiplier, 1.5)
        self.assertEqual(config.loss_divisor, 2.0)

    def test_invalid_win_multiplier_too_low(self):
        """Test that win_multiplier <= 1.0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StreakConfig(win_multiplier=1.0)
        self.assertIn("win_multiplier must be > 1.0", str(ctx.exception))

    def test_invalid_win_multiplier_negative(self):
        """Test that negative win_multiplier raises ValueError."""
        with self.assertRaises(ValueError):
            StreakConfig(win_multiplier=-1.5)

    def test_invalid_loss_divisor_too_low(self):
        """Test that loss_divisor <= 1.0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StreakConfig(loss_divisor=1.0)
        self.assertIn("loss_divisor must be > 1.0", str(ctx.exception))

    def test_invalid_loss_divisor_negative(self):
        """Test that negative loss_divisor raises ValueError."""
        with self.assertRaises(ValueError):
            StreakConfig(loss_divisor=-1.8)

    def test_invalid_max_boost_too_low(self):
        """Test that max_boost < 1.0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StreakConfig(max_boost=0.9)
        self.assertIn("max_boost must be >= 1.0", str(ctx.exception))

    def test_invalid_min_floor_zero(self):
        """Test that min_floor = 0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StreakConfig(min_floor=0.0)
        self.assertIn("min_floor must be in (0, 1]", str(ctx.exception))

    def test_invalid_min_floor_too_high(self):
        """Test that min_floor > 1.0 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            StreakConfig(min_floor=1.5)
        self.assertIn("min_floor must be in (0, 1]", str(ctx.exception))


class TestStreakTrackerInitialization(unittest.TestCase):
    """Test StreakTracker initialization."""

    def test_default_initialization(self):
        """Test StreakTracker initializes with defaults."""
        tracker = StreakTracker()
        self.assertIsNotNone(tracker.config)
        self.assertEqual(tracker._consecutive_wins, 0)
        self.assertEqual(tracker._consecutive_losses, 0)
        self.assertIsNone(tracker._last_regime)
        self.assertEqual(tracker._total_trades, 0)

    def test_custom_config_initialization(self):
        """Test StreakTracker with custom config."""
        config = StreakConfig(win_multiplier=1.5, loss_divisor=2.0)
        tracker = StreakTracker(config=config)
        self.assertEqual(tracker.config.win_multiplier, 1.5)
        self.assertEqual(tracker.config.loss_divisor, 2.0)


class TestStreakTrackerRecordOutcome(unittest.TestCase):
    """Test StreakTracker.record_outcome() method."""

    def test_record_single_win(self):
        """Test recording a single win."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        self.assertEqual(tracker._consecutive_wins, 1)
        self.assertEqual(tracker._consecutive_losses, 0)
        self.assertEqual(tracker._total_trades, 1)

    def test_record_single_loss(self):
        """Test recording a single loss."""
        tracker = StreakTracker()
        tracker.record_outcome(win=False)
        self.assertEqual(tracker._consecutive_wins, 0)
        self.assertEqual(tracker._consecutive_losses, 1)
        self.assertEqual(tracker._total_trades, 1)

    def test_record_consecutive_wins(self):
        """Test recording multiple consecutive wins."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        self.assertEqual(tracker._consecutive_wins, 3)
        self.assertEqual(tracker._consecutive_losses, 0)
        self.assertEqual(tracker._total_trades, 3)

    def test_record_consecutive_losses(self):
        """Test recording multiple consecutive losses."""
        tracker = StreakTracker()
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        self.assertEqual(tracker._consecutive_wins, 0)
        self.assertEqual(tracker._consecutive_losses, 3)
        self.assertEqual(tracker._total_trades, 3)

    def test_win_breaks_loss_streak(self):
        """Test that a win breaks a loss streak."""
        tracker = StreakTracker()
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=True)
        self.assertEqual(tracker._consecutive_wins, 1)
        self.assertEqual(tracker._consecutive_losses, 0)

    def test_loss_breaks_win_streak(self):
        """Test that a loss breaks a win streak."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=False)
        self.assertEqual(tracker._consecutive_wins, 0)
        self.assertEqual(tracker._consecutive_losses, 1)


class TestStreakTrackerMultiplier(unittest.TestCase):
    """Test StreakTracker.get_multiplier() method."""

    def test_no_streak_multiplier_is_one(self):
        """Test that no streak results in multiplier = 1.0."""
        tracker = StreakTracker()
        self.assertEqual(tracker.get_multiplier(), 1.0)

    def test_single_win_multiplier(self):
        """Test multiplier after 1 win."""
        tracker = StreakTracker(config=StreakConfig(win_multiplier=1.3))
        tracker.record_outcome(win=True)
        multiplier = tracker.get_multiplier()
        self.assertAlmostEqual(multiplier, 1.3, places=3)

    def test_two_win_multiplier(self):
        """Test multiplier after 2 consecutive wins."""
        tracker = StreakTracker(config=StreakConfig(win_multiplier=1.3))
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        multiplier = tracker.get_multiplier()
        expected = 1.3 ** 2  # 1.69
        self.assertAlmostEqual(multiplier, expected, places=3)

    def test_three_win_multiplier(self):
        """Test multiplier after 3 consecutive wins."""
        tracker = StreakTracker(config=StreakConfig(win_multiplier=1.3, max_boost=3.0))
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        multiplier = tracker.get_multiplier()
        expected = 1.3 ** 3  # ~2.197
        self.assertAlmostEqual(multiplier, expected, places=3)

    def test_win_streak_capped_at_max_boost(self):
        """Test that win streak is capped at max_boost."""
        tracker = StreakTracker(config=StreakConfig(win_multiplier=1.3, max_boost=2.0))
        # Record enough wins to exceed max_boost
        for _ in range(10):
            tracker.record_outcome(win=True)
        multiplier = tracker.get_multiplier()
        self.assertEqual(multiplier, 2.0)

    def test_single_loss_multiplier(self):
        """Test multiplier after 1 loss."""
        tracker = StreakTracker(config=StreakConfig(loss_divisor=1.8))
        tracker.record_outcome(win=False)
        multiplier = tracker.get_multiplier()
        expected = 1.0 / 1.8
        self.assertAlmostEqual(multiplier, expected, places=3)

    def test_two_loss_multiplier(self):
        """Test multiplier after 2 consecutive losses."""
        tracker = StreakTracker(config=StreakConfig(loss_divisor=1.8))
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        multiplier = tracker.get_multiplier()
        expected = 1.0 / (1.8 ** 2)
        self.assertAlmostEqual(multiplier, expected, places=3)

    def test_three_loss_multiplier(self):
        """Test multiplier after 3 consecutive losses."""
        tracker = StreakTracker(config=StreakConfig(loss_divisor=1.8, min_floor=0.1))
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        tracker.record_outcome(win=False)
        multiplier = tracker.get_multiplier()
        expected = 1.0 / (1.8 ** 3)
        self.assertAlmostEqual(multiplier, expected, places=3)

    def test_loss_streak_floored_at_min_floor(self):
        """Test that loss streak is floored at min_floor."""
        tracker = StreakTracker(config=StreakConfig(loss_divisor=1.8, min_floor=0.3))
        # Record enough losses to go below min_floor
        for _ in range(10):
            tracker.record_outcome(win=False)
        multiplier = tracker.get_multiplier()
        self.assertEqual(multiplier, 0.3)


class TestStreakTrackerRegimeChange(unittest.TestCase):
    """Test StreakTracker regime change handling."""

    def test_regime_change_resets_streak_when_enabled(self):
        """Test that regime change resets streak when enabled."""
        config = StreakConfig(reset_on_regime_change=True)
        tracker = StreakTracker(config=config)
        tracker.record_outcome(win=True, regime_name="NORMAL")
        tracker.record_outcome(win=True, regime_name="NORMAL")
        self.assertEqual(tracker._consecutive_wins, 2)
        # Change regime
        tracker.record_outcome(win=True, regime_name="HIGH")
        self.assertEqual(tracker._consecutive_wins, 1)  # Streak reset, new one started

    def test_regime_change_does_not_reset_when_disabled(self):
        """Test that regime change doesn't reset streak when disabled."""
        config = StreakConfig(reset_on_regime_change=False)
        tracker = StreakTracker(config=config)
        tracker.record_outcome(win=True, regime_name="NORMAL")
        tracker.record_outcome(win=True, regime_name="NORMAL")
        self.assertEqual(tracker._consecutive_wins, 2)
        # Change regime
        tracker.record_outcome(win=True, regime_name="HIGH")
        self.assertEqual(tracker._consecutive_wins, 3)  # Streak continues

    def test_regime_tracked(self):
        """Test that last regime is tracked."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True, regime_name="NORMAL")
        self.assertEqual(tracker._last_regime, "NORMAL")
        tracker.record_outcome(win=True, regime_name="HIGH")
        self.assertEqual(tracker._last_regime, "HIGH")


class TestStreakTrackerSerialization(unittest.TestCase):
    """Test StreakTracker serialization (to_dict/from_dict)."""

    def test_to_dict_basic(self):
        """Test to_dict() returns valid dict."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        data = tracker.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["consecutive_wins"], 2)
        self.assertEqual(data["consecutive_losses"], 0)
        self.assertEqual(data["total_trades"], 2)

    def test_to_dict_includes_config(self):
        """Test that to_dict includes config."""
        config = StreakConfig(win_multiplier=1.5, loss_divisor=2.0)
        tracker = StreakTracker(config=config)
        data = tracker.to_dict()
        self.assertIn("config", data)
        self.assertEqual(data["config"]["win_multiplier"], 1.5)
        self.assertEqual(data["config"]["loss_divisor"], 2.0)

    def test_from_dict_basic(self):
        """Test from_dict() reconstructs tracker."""
        data = {
            "consecutive_wins": 3,
            "consecutive_losses": 1,
            "last_regime": "NORMAL",
            "total_trades": 4,
            "config": {
                "win_multiplier": 1.3,
                "loss_divisor": 1.8,
                "max_boost": 2.0,
                "min_floor": 0.3,
                "reset_on_regime_change": True,
            },
        }
        tracker = StreakTracker.from_dict(data)
        self.assertEqual(tracker._consecutive_wins, 3)
        self.assertEqual(tracker._consecutive_losses, 1)
        self.assertEqual(tracker._last_regime, "NORMAL")
        self.assertEqual(tracker._total_trades, 4)

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict round-trip works."""
        tracker1 = StreakTracker(config=StreakConfig(win_multiplier=1.5))
        tracker1.record_outcome(win=True)
        tracker1.record_outcome(win=True)
        tracker1.record_outcome(win=False)

        data = tracker1.to_dict()
        tracker2 = StreakTracker.from_dict(data)

        self.assertEqual(tracker1._consecutive_wins, tracker2._consecutive_wins)
        self.assertEqual(tracker1._consecutive_losses, tracker2._consecutive_losses)
        self.assertEqual(tracker1._total_trades, tracker2._total_trades)
        self.assertAlmostEqual(
            tracker1.get_multiplier(), tracker2.get_multiplier(), places=4
        )


class TestStreakTrackerInfo(unittest.TestCase):
    """Test StreakTracker.streak_info property."""

    def test_streak_info_basic(self):
        """Test streak_info property returns expected fields."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        tracker.record_outcome(win=True)
        info = tracker.streak_info
        self.assertEqual(info["consecutive_wins"], 2)
        self.assertEqual(info["consecutive_losses"], 0)
        self.assertEqual(info["multiplier"], 1.69)  # 1.3^2
        self.assertEqual(info["total_trades"], 2)


class TestAdaptivePositionSizerIntegration(unittest.TestCase):
    """Test StreakTracker integration with AdaptivePositionSizer."""

    def test_streak_disabled_by_default_when_disabled(self):
        """Test that streak tracker is None when disabled."""
        config = AdaptivePositionSizingConfig(streak_enabled=False)
        sizer = AdaptivePositionSizer(config=config)
        self.assertIsNone(sizer._streak_tracker)

    def test_streak_enabled_by_default(self):
        """Test that streak tracker is initialized when enabled."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)
        self.assertIsNotNone(sizer._streak_tracker)
        self.assertIsInstance(sizer._streak_tracker, StreakTracker)

    def test_record_trade_updates_streak(self):
        """Test that record_trade updates streak tracker."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)
        sizer._last_regime_name = "NORMAL"

        sizer.record_trade(pnl=100, win=True)
        sizer.record_trade(pnl=150, win=True)

        self.assertEqual(sizer._streak_tracker._consecutive_wins, 2)

    def test_calculate_size_tracks_regime(self):
        """Test that calculate_size sets _last_regime_name."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)

        sizer.calculate_size(
            account_equity=10000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="HIGH",
        )

        self.assertEqual(sizer._last_regime_name, "HIGH")

    def test_calculate_size_applies_streak_multiplier(self):
        """Test that calculate_size applies streak multiplier."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)

        # Simulate consecutive wins
        sizer._last_regime_name = "NORMAL"
        sizer.record_trade(pnl=100, win=True)
        sizer.record_trade(pnl=150, win=True)

        result1 = sizer.calculate_size(
            account_equity=10000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="NORMAL",
        )

        self.assertEqual(result1.streak_multiplier, 1.3 ** 2)
        self.assertGreater(result1.final_size, 0)

    def test_streak_multiplier_in_result(self):
        """Test that PositionSizeResult includes streak_multiplier."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)

        result = sizer.calculate_size(
            account_equity=10000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="NORMAL",
        )

        self.assertIn("streak_multiplier", result.__dict__)
        self.assertEqual(result.streak_multiplier, 1.0)  # No streak yet

    def test_streak_details_in_result(self):
        """Test that result.details includes streak info."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)
        sizer._last_regime_name = "NORMAL"
        sizer.record_trade(pnl=100, win=True)

        result = sizer.calculate_size(
            account_equity=10000,
            confidence=0.7,
            current_drawdown_pct=0.0,
            regime_name="NORMAL",
        )

        self.assertIn("streak_multiplier", result.details)
        self.assertIn("streak_adjusted_size", result.details)


class TestAdaptivePositionSizerStatePersistence(unittest.TestCase):
    """Test that state persistence includes streak tracker."""

    def test_save_state_includes_streak(self):
        """Test that save_state includes streak tracker state."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer = AdaptivePositionSizer(config=config)
        sizer._last_regime_name = "NORMAL"
        sizer.record_trade(pnl=100, win=True)
        sizer.record_trade(pnl=150, win=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "sizer_state.json")
            sizer.save_state(state_file)

            with open(state_file, "r") as f:
                state = json.load(f)

            self.assertIn("streak_tracker", state)
            self.assertEqual(state["streak_tracker"]["consecutive_wins"], 2)

    def test_load_state_restores_streak(self):
        """Test that load_state restores streak tracker state."""
        config = AdaptivePositionSizingConfig(streak_enabled=True)
        sizer1 = AdaptivePositionSizer(config=config)
        sizer1._last_regime_name = "NORMAL"
        sizer1.record_trade(pnl=100, win=True)
        sizer1.record_trade(pnl=150, win=True)
        sizer1.record_trade(pnl=200, win=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "sizer_state.json")
            sizer1.save_state(state_file)

            # Load into new sizer
            sizer2 = AdaptivePositionSizer(config=config)
            sizer2.load_state(state_file)

            self.assertEqual(
                sizer2._streak_tracker._consecutive_wins,
                sizer1._streak_tracker._consecutive_wins,
            )
            self.assertEqual(
                sizer2._streak_tracker._consecutive_losses,
                sizer1._streak_tracker._consecutive_losses,
            )
            self.assertEqual(
                sizer2._streak_tracker._total_trades,
                sizer1._streak_tracker._total_trades,
            )


class TestAlternatingWinLoss(unittest.TestCase):
    """Test edge case: alternating win/loss keeps multiplier at 1.0."""

    def test_alternating_win_loss_multiplier(self):
        """Test that alternating wins and losses keeps multiplier at 1.0."""
        tracker = StreakTracker()
        tracker.record_outcome(win=True)
        self.assertEqual(tracker.get_multiplier(), 1.3)

        tracker.record_outcome(win=False)
        self.assertEqual(tracker.get_multiplier(), 1.0 / 1.8)

        tracker.record_outcome(win=True)
        self.assertEqual(tracker.get_multiplier(), 1.3)

        tracker.record_outcome(win=False)
        self.assertEqual(tracker.get_multiplier(), 1.0 / 1.8)


class TestLargeWinStreak(unittest.TestCase):
    """Test edge case: large win streak caps at max_boost."""

    def test_large_win_streak_capped(self):
        """Test that very large win streaks cap at max_boost."""
        config = StreakConfig(win_multiplier=1.3, max_boost=2.0)
        tracker = StreakTracker(config=config)

        # Record 20 wins
        for _ in range(20):
            tracker.record_outcome(win=True)

        multiplier = tracker.get_multiplier()
        self.assertEqual(multiplier, 2.0)  # Should be capped


class TestLargeLossStreak(unittest.TestCase):
    """Test edge case: large loss streak floors at min_floor."""

    def test_large_loss_streak_floored(self):
        """Test that very large loss streaks floor at min_floor."""
        config = StreakConfig(loss_divisor=1.8, min_floor=0.3)
        tracker = StreakTracker(config=config)

        # Record 20 losses
        for _ in range(20):
            tracker.record_outcome(win=False)

        multiplier = tracker.get_multiplier()
        self.assertEqual(multiplier, 0.3)  # Should be floored


if __name__ == "__main__":
    unittest.main()
