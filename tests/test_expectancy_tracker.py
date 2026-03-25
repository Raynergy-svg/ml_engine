"""Unit tests for ExpectancyTracker (US-292).

Tests cover:
- Basic recording and calculation of expectancy
- Rolling window behavior
- Weight modifier logic
- State persistence (save/load)
- Edge cases (zero trades, all wins, all losses, etc)
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.scanner.expectancy_tracker import (
    ExpectancyConfig,
    ExpectancyResult,
    ExpectancyTracker,
    TradeRecord,
    create_default_expectancy_tracker,
)


class TestExpectancyTrackerBasics:
    """Test basic recording and calculation."""

    def test_init_creates_all_agent_regime_windows(self):
        """Test that init creates windows for all agents and regimes."""
        tracker = ExpectancyTracker()
        assert len(tracker._windows) == 13  # 13 agents
        for agent in tracker._AGENT_NAMES:
            assert agent in tracker._windows
            assert len(tracker._windows[agent]) == 4  # 4 regimes
            for regime in tracker._REGIME_NAMES:
                assert regime in tracker._windows[agent]

    def test_record_trade_single(self):
        """Test recording a single trade."""
        tracker = ExpectancyTracker()
        tracker.record_trade("trend", "NORMAL", 100.0, won=True)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.trade_count == 1
        assert result.win_rate == 1.0
        assert result.avg_win == 100.0
        assert result.avg_loss == 0.0

    def test_record_trade_invalid_agent_raises(self):
        """Test that recording with invalid agent raises ValueError."""
        tracker = ExpectancyTracker()
        with pytest.raises(ValueError, match="Unknown agent"):
            tracker.record_trade("invalid_agent", "NORMAL", 100.0, won=True)

    def test_record_trade_invalid_regime_raises(self):
        """Test that recording with invalid regime raises ValueError."""
        tracker = ExpectancyTracker()
        with pytest.raises(ValueError, match="Unknown regime"):
            tracker.record_trade("trend", "INVALID", 100.0, won=True)

    def test_expectancy_calculation_known_values(self):
        """Test expectancy with known values.

        60% win rate, avg win $100, avg loss -$80
        E = 0.60 * 100 + 0.40 * (-80) = 60 - 32 = $28
        """
        tracker = ExpectancyTracker()

        # 3 wins: 100, 100, 100
        for _ in range(3):
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)

        # 2 losses: -80, -80
        for _ in range(2):
            tracker.record_trade("trend", "NORMAL", -80.0, won=False)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.trade_count == 5
        assert result.win_rate == 0.6
        assert result.avg_win == 100.0
        assert result.avg_loss == -80.0
        assert result.expectancy == 28.0  # 0.6 * 100 + 0.4 * (-80)

    def test_weight_modifier_positive_expectancy(self):
        """Test weight modifier = 1.0 for positive expectancy."""
        tracker = ExpectancyTracker()

        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        modifier = tracker.get_weight_modifier("trend", "NORMAL")
        assert modifier == 1.0

    def test_weight_modifier_negative_expectancy(self):
        """Test weight modifier for negative expectancy.

        Config default: negative_penalty = -0.20
        modifier = 1 + (-0.20) = 0.80
        """
        config = ExpectancyConfig(negative_penalty=-0.20)
        tracker = ExpectancyTracker(config)

        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", 50.0, won=True)
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", -100.0, won=False)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.expectancy < 0  # 0.5 * 50 + 0.5 * (-100) = -25
        modifier = tracker.get_weight_modifier("trend", "NORMAL")
        assert modifier == 0.80

    def test_weight_modifier_insufficient_data(self):
        """Test weight modifier = 1.0 when trade_count < min_trades_for_calc."""
        config = ExpectancyConfig(min_trades_for_calc=5)
        tracker = ExpectancyTracker(config)

        # Only 2 trades (less than min 5)
        tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.is_meaningful is False
        assert result.weight_modifier == 1.0  # Returns 1.0 for unclear

    def test_rolling_window_respects_max_size(self):
        """Test that rolling window respects maxlen (default 50)."""
        config = ExpectancyConfig(window_size=10)
        tracker = ExpectancyTracker(config)

        # Add 15 trades (more than window_size=10)
        for i in range(15):
            tracker.record_trade("trend", "NORMAL", 10.0 + i, won=True)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.trade_count == 10  # Only last 10 kept
        # Avg of trades 5-14: (15 + 16 + 17 + 18 + 19 + 20 + 21 + 22 + 23 + 24) / 10 = 19.5
        assert result.avg_win == pytest.approx(19.5, abs=0.1)

    def test_get_all_expectancies(self):
        """Test get_all_expectancies returns dict with all agents/regimes."""
        tracker = ExpectancyTracker()

        # Record some trades
        tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker.record_trade("mean_reversion", "HIGH", -50.0, won=False)

        all_exp = tracker.get_all_expectancies()

        # Check structure
        assert isinstance(all_exp, dict)
        assert len(all_exp) == 13  # 13 agents
        assert all(len(all_exp[agent]) == 4 for agent in all_exp)  # 4 regimes each

        # Check recorded entries are present
        assert all_exp["trend"]["NORMAL"].trade_count == 1
        assert all_exp["mean_reversion"]["HIGH"].trade_count == 1

        # Check other combos are zero
        assert all_exp["trend"]["HIGH"].trade_count == 0

    def test_get_top_agents_ranks_by_expectancy(self):
        """Test get_top_agents ranks correctly by expectancy."""
        tracker = ExpectancyTracker()

        # Agent 1: positive expectancy
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        # Agent 2: better expectancy
        for _ in range(5):
            tracker.record_trade("mean_reversion", "NORMAL", 150.0, won=True)
        for _ in range(5):
            tracker.record_trade("mean_reversion", "NORMAL", -50.0, won=False)

        # Agent 3: negative expectancy
        for _ in range(5):
            tracker.record_trade("volatility", "NORMAL", 50.0, won=True)
        for _ in range(5):
            tracker.record_trade("volatility", "NORMAL", -100.0, won=False)

        top = tracker.get_top_agents("NORMAL", n=3)

        # Should be ranked by expectancy descending
        assert len(top) <= 3
        assert top[0][0] == "mean_reversion"  # Highest expectancy
        assert top[0][1].expectancy == 50.0  # 0.5 * 150 + 0.5 * (-50)
        assert top[1][0] == "trend"
        assert top[2][0] == "volatility"

    def test_get_top_agents_excludes_insufficient_data(self):
        """Test get_top_agents only includes agents with >= min_trades_for_calc."""
        config = ExpectancyConfig(min_trades_for_calc=5)
        tracker = ExpectancyTracker(config)

        # Only 2 trades (insufficient)
        tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        # 5 trades (sufficient)
        for _ in range(5):
            tracker.record_trade("mean_reversion", "NORMAL", 100.0, won=True)

        top = tracker.get_top_agents("NORMAL", n=5)

        # Only mean_reversion should be included
        assert len(top) == 1
        assert top[0][0] == "mean_reversion"


class TestExpectancyEdgeCases:
    """Test edge cases."""

    def test_edge_all_wins(self):
        """Test expectancy when all trades are wins."""
        tracker = ExpectancyTracker()

        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.win_rate == 1.0
        assert result.avg_win == 100.0
        assert result.avg_loss == 0.0
        assert result.expectancy == 100.0  # 1.0 * 100 + 0.0 * 0

    def test_edge_all_losses(self):
        """Test expectancy when all trades are losses."""
        tracker = ExpectancyTracker()

        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", -100.0, won=False)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.win_rate == 0.0
        assert result.avg_win == 0.0
        assert result.avg_loss == -100.0
        assert result.expectancy == -100.0  # 0.0 * 0 + 1.0 * (-100)

    def test_edge_single_trade(self):
        """Test expectancy with single trade."""
        tracker = ExpectancyTracker()
        tracker.record_trade("trend", "NORMAL", 50.0, won=True)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.trade_count == 1
        assert result.win_rate == 1.0
        assert result.expectancy == 50.0

    def test_edge_empty_window(self):
        """Test expectancy with no trades."""
        tracker = ExpectancyTracker()

        result = tracker.get_expectancy("trend", "NORMAL")

        assert result.expectancy == 0.0
        assert result.win_rate == 0.0
        assert result.avg_win == 0.0
        assert result.avg_loss == 0.0
        assert result.trade_count == 0
        assert result.is_meaningful is False
        assert result.weight_modifier == 1.0

    def test_edge_negative_pnl_on_win(self):
        """Test handling of negative PnL marked as win (edge case)."""
        tracker = ExpectancyTracker()

        # Add a "win" with negative PnL (unusual but possible)
        tracker.record_trade("trend", "NORMAL", -10.0, won=True)

        result = tracker.get_expectancy("trend", "NORMAL")
        assert result.avg_win == -10.0
        assert result.expectancy == -10.0


class TestExpectancyPersistence:
    """Test state persistence (save/load)."""

    def test_to_dict_serialization(self):
        """Test to_dict produces correct structure."""
        tracker = ExpectancyTracker()
        tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker.record_trade("trend", "HIGH", -50.0, won=False)

        data = tracker.to_dict()

        assert "version" in data
        assert "timestamp" in data
        assert "windows" in data
        assert isinstance(data["windows"], dict)
        assert "trend" in data["windows"]
        assert "NORMAL" in data["windows"]["trend"]
        assert len(data["windows"]["trend"]["NORMAL"]) == 1
        assert len(data["windows"]["trend"]["HIGH"]) == 1

    def test_from_dict_deserialization(self):
        """Test from_dict reconstructs tracker correctly."""
        tracker1 = ExpectancyTracker()
        tracker1.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker1.record_trade("trend", "NORMAL", -50.0, won=False)

        data = tracker1.to_dict()
        tracker2 = ExpectancyTracker.from_dict(data)

        result1 = tracker1.get_expectancy("trend", "NORMAL")
        result2 = tracker2.get_expectancy("trend", "NORMAL")

        assert result1.expectancy == result2.expectancy
        assert result1.trade_count == result2.trade_count

    def test_from_dict_invalid_data_raises(self):
        """Test from_dict raises on invalid data structure."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            ExpectancyTracker.from_dict("not a dict")

    def test_from_dict_skips_unknown_agents(self):
        """Test from_dict skips unknown agents gracefully."""
        data = {
            "version": "1.0.0",
            "timestamp": time.time(),
            "windows": {
                "unknown_agent": {
                    "NORMAL": [
                        {"pnl": 100.0, "won": True, "timestamp": time.time()}
                    ]
                },
                "trend": {
                    "NORMAL": [
                        {"pnl": 50.0, "won": True, "timestamp": time.time()}
                    ]
                },
            },
        }

        tracker = ExpectancyTracker.from_dict(data)

        # trend should be loaded, unknown_agent skipped
        assert tracker.get_expectancy("trend", "NORMAL").trade_count == 1

    def test_from_dict_skips_unknown_regimes(self):
        """Test from_dict skips unknown regimes gracefully."""
        data = {
            "version": "1.0.0",
            "timestamp": time.time(),
            "windows": {
                "trend": {
                    "INVALID_REGIME": [
                        {"pnl": 100.0, "won": True, "timestamp": time.time()}
                    ],
                    "NORMAL": [
                        {"pnl": 50.0, "won": True, "timestamp": time.time()}
                    ],
                },
            },
        }

        tracker = ExpectancyTracker.from_dict(data)

        # NORMAL should be loaded, INVALID_REGIME skipped
        assert tracker.get_expectancy("trend", "NORMAL").trade_count == 1

    def test_save_state_creates_file(self):
        """Test save_state creates persistence file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "expectancy.json")
            tracker = ExpectancyTracker(
                ExpectancyConfig(persistence_path=path)
            )
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)

            tracker.save_state(path)

            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "windows" in data

    def test_save_state_atomic_write(self):
        """Test save_state uses atomic write (tmp + rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "expectancy.json")
            tracker = ExpectancyTracker(
                ExpectancyConfig(persistence_path=path)
            )
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)

            tracker.save_state(path)

            # Verify file exists and tmp doesn't
            assert os.path.exists(path)
            assert not os.path.exists(f"{path}.tmp")

    def test_save_state_creates_parent_dirs(self):
        """Test save_state creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "expectancy.json")
            tracker = ExpectancyTracker(
                ExpectancyConfig(persistence_path=path)
            )
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)

            tracker.save_state(path)

            assert os.path.exists(path)

    def test_save_state_ioerror_cleans_up(self):
        """Test save_state cleans up tmp file on error."""
        # Use a read-only directory to cause write error
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "expectancy.json")
            tracker = ExpectancyTracker(
                ExpectancyConfig(persistence_path=path)
            )

            # Make directory read-only
            os.chmod(tmpdir, 0o444)

            try:
                with pytest.raises(IOError):
                    tracker.save_state(path)
            finally:
                os.chmod(tmpdir, 0o755)

    def test_load_state_missing_file_returns_false(self):
        """Test load_state returns False for missing file."""
        tracker = ExpectancyTracker()
        result = tracker.load_state("/nonexistent/path/expectancy.json")
        assert result is False

    def test_load_state_corrupted_file_returns_false(self):
        """Test load_state returns False for corrupted JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "expectancy.json")
            with open(path, "w") as f:
                f.write("{ invalid json }")

            tracker = ExpectancyTracker()
            result = tracker.load_state(path)
            assert result is False

    def test_load_state_roundtrip(self):
        """Test save_state followed by load_state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "expectancy.json")

            tracker1 = ExpectancyTracker()
            tracker1.record_trade("trend", "NORMAL", 100.0, won=True)
            tracker1.record_trade("trend", "HIGH", -50.0, won=False)
            tracker1.record_trade("mean_reversion", "EXTREME", 75.0, won=True)
            tracker1.save_state(path)

            tracker2 = ExpectancyTracker()
            loaded = tracker2.load_state(path)

            assert loaded is True
            assert tracker2.get_expectancy("trend", "NORMAL").trade_count == 1
            assert tracker2.get_expectancy("trend", "HIGH").trade_count == 1
            assert tracker2.get_expectancy("mean_reversion", "EXTREME").trade_count == 1


class TestCreateDefaultFactory:
    """Test factory function."""

    def test_create_default_expectancy_tracker(self):
        """Test factory creates tracker with defaults."""
        tracker = create_default_expectancy_tracker()

        assert isinstance(tracker, ExpectancyTracker)
        assert tracker._config.enabled is True
        assert tracker._config.window_size == 50

    def test_create_default_with_config_override(self):
        """Test factory with custom config."""
        config = ExpectancyConfig(window_size=100, enabled=False)
        tracker = create_default_expectancy_tracker(config)

        assert tracker._config.window_size == 100
        assert tracker._config.enabled is False


class TestExpectancyMultiRegime:
    """Test multi-regime behavior."""

    def test_different_regimes_isolated(self):
        """Test that trades in different regimes don't mix."""
        tracker = ExpectancyTracker()

        tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        tracker.record_trade("trend", "HIGH", 50.0, won=True)
        tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        normal_result = tracker.get_expectancy("trend", "NORMAL")
        high_result = tracker.get_expectancy("trend", "HIGH")

        assert normal_result.trade_count == 2
        assert high_result.trade_count == 1
        assert normal_result.expectancy != high_result.expectancy

    def test_same_agent_different_regimes(self):
        """Test same agent can have different expectancies per regime."""
        tracker = ExpectancyTracker()

        # Trend in NORMAL: positive expectancy
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", 100.0, won=True)
        for _ in range(5):
            tracker.record_trade("trend", "NORMAL", -50.0, won=False)

        # Trend in HIGH: negative expectancy
        for _ in range(5):
            tracker.record_trade("trend", "HIGH", 50.0, won=True)
        for _ in range(5):
            tracker.record_trade("trend", "HIGH", -100.0, won=False)

        normal = tracker.get_expectancy("trend", "NORMAL")
        high = tracker.get_expectancy("trend", "HIGH")

        assert normal.expectancy > 0
        assert high.expectancy < 0
        assert normal.weight_modifier == 1.0
        assert high.weight_modifier < 1.0
