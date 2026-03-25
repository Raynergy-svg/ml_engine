"""
Test suite for AlertManager.

Validates all alert types, cooldown behavior, and state persistence.
Run with: pytest src/scanner/automation/test_alert_manager.py -v
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.scanner.automation.alert_manager import Alert, AlertManager


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            alert_type="drawdown",
            severity="CRITICAL",
            message="Portfolio down 5%",
            timestamp=datetime.utcnow().isoformat(),
            value=0.05,
            threshold=0.05,
            pair="EUR_USD",
        )

        assert alert.alert_type == "drawdown"
        assert alert.severity == "CRITICAL"
        assert alert.acknowledged is False

    def test_alert_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            alert_type="test",
            severity="WARNING",
            message="Test message",
            timestamp="2026-03-19T10:00:00",
            value=1.5,
            threshold=1.0,
        )

        d = alert.to_dict()
        assert d["alert_type"] == "test"
        assert d["severity"] == "WARNING"
        assert d["value"] == 1.5


class TestAlertManager:
    """Test AlertManager core functionality."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing."""
        log_path = tmp_path / "alerts.jsonl"
        state_path = tmp_path / ".claude" / "alert_state.json"
        return log_path, state_path

    @pytest.fixture
    def manager(self, temp_paths):
        """Create a test AlertManager."""
        log_path, state_path = temp_paths
        return AlertManager(
            alert_log_path=str(log_path),
            state_path=str(state_path),
        )

    def test_init(self, manager):
        """Test manager initialization."""
        assert len(manager._configs) == 4  # 4 alert types
        assert "drawdown" in manager._configs
        assert "consecutive_losses" in manager._configs
        assert "win_rate_drop" in manager._configs
        assert "weight_instability" in manager._configs

    def test_drawdown_alert_below_threshold(self, manager):
        """Test drawdown alert when below threshold."""
        alert = manager.check_drawdown(current_nav=9500, peak_nav=10000)
        assert alert is not None  # 5% drawdown at threshold
        assert alert.value == 0.05
        assert "5.0%" in alert.message

    def test_drawdown_alert_above_threshold(self, manager):
        """Test drawdown alert when above threshold."""
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None  # 6% drawdown exceeds 5% threshold
        assert alert.value == 0.06

    def test_drawdown_no_alert_safe(self, manager):
        """Test no drawdown alert when below threshold."""
        alert = manager.check_drawdown(current_nav=9600, peak_nav=10000)
        assert alert is None  # 4% drawdown below 5% threshold

    def test_drawdown_zero_peak(self, manager):
        """Test drawdown with zero peak nav."""
        alert = manager.check_drawdown(current_nav=100, peak_nav=0)
        assert alert is None

    def test_consecutive_losses_no_trades(self, manager):
        """Test consecutive losses check with no trades."""
        alert = manager.check_consecutive_losses([])
        assert alert is None

    def test_consecutive_losses_threshold_exceeded(self, manager):
        """Test consecutive losses alert."""
        trades = [
            {"outcome": {"trade_won": True}, "pair": "EUR_USD"},
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
        ]
        alert = manager.check_consecutive_losses(trades)
        assert alert is not None
        assert alert.value == 3

    def test_consecutive_losses_with_open_trades(self, manager):
        """Test consecutive losses skips open trades."""
        trades = [
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
            {"outcome": {"trade_won": False}, "pair": "EUR_USD"},
            {"outcome": None, "pair": "EUR_USD"},  # Open trade
        ]
        alert = manager.check_consecutive_losses(trades)
        assert alert is not None  # Still 3 consecutive losses
        assert alert.value == 3

    def test_win_rate_drop_insufficient_history(self, manager):
        """Test win rate drop with insufficient history."""
        trades = [{"outcome": {"trade_won": True}} for _ in range(10)]
        alert = manager.check_win_rate_drop(trades, window=20)
        assert alert is None  # Not enough history

    def test_win_rate_drop_detected(self, manager):
        """Test win rate drop alert."""
        # Create history: 80% win rate early, 30% recently
        old_trades = [
            {"outcome": {"trade_won": i % 5 != 0}} for i in range(20)
        ]  # 80% wins
        new_trades = [
            {"outcome": {"trade_won": i % 3 == 0}} for i in range(20)
        ]  # 30% wins
        trades = old_trades + new_trades

        alert = manager.check_win_rate_drop(trades, window=20)
        assert alert is not None
        assert alert.alert_type == "win_rate_drop"

    def test_weight_instability_no_weights(self, manager):
        """Test weight instability with no weights."""
        alert = manager.check_weight_instability(None, None)
        assert alert is None

        alert = manager.check_weight_instability({}, {})
        assert alert is None

    def test_weight_instability_detected(self, manager):
        """Test weight instability alert."""
        current = {"trend": 0.8, "mean_reversion": 0.5, "volatility": 0.6}
        previous = {"trend": 0.5, "mean_reversion": 0.5, "volatility": 0.6}
        # Max change is 0.3 (trend went from 0.5 to 0.8)

        alert = manager.check_weight_instability(current, previous)
        assert alert is not None
        assert alert.value == 0.3

    def test_cooldown_blocks_duplicate_alerts(self, manager):
        """Test that cooldown prevents duplicate alerts."""
        # First alert should fire
        alert1 = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert1 is not None
        manager._fire_alert(alert1)

        # Second alert should be blocked by cooldown
        alert2 = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert2 is None

    def test_cooldown_expiry(self, manager):
        """Test that cooldown expires."""
        # Fire first alert
        alert1 = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert1 is not None
        manager._fire_alert(alert1)

        # Manually expire cooldown
        manager._last_fired["drawdown"] = datetime.utcnow() - timedelta(
            minutes=61
        )

        # Second alert should fire now
        alert2 = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert2 is not None

    def test_check_all(self, manager):
        """Test check_all runs all checks."""
        trades = [
            {"outcome": {"trade_won": False}} for _ in range(5)
        ]  # Consecutive losses
        current_weights = {"trend": 0.5}
        previous_weights = {"trend": 1.0}

        alerts = manager.check_all(
            nav=9400,
            peak_nav=10000,
            recent_trades=trades,
            current_weights=current_weights,
            previous_weights=previous_weights,
        )

        # Should have multiple alerts
        assert len(alerts) >= 1
        assert any(a.alert_type == "drawdown" for a in alerts)

    def test_state_persistence(self, manager, temp_paths):
        """Test saving and loading alert state."""
        # Fire an alert
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None
        manager._fire_alert(alert)

        # Create new manager and load state
        new_manager = AlertManager(
            alert_log_path=str(temp_paths[0]),
            state_path=str(temp_paths[1]),
        )

        # Cooldown should be preserved
        assert "drawdown" in new_manager._last_fired
        new_alert = new_manager.check_drawdown(
            current_nav=9400, peak_nav=10000
        )
        assert new_alert is None  # Still in cooldown

    def test_corrupted_state_file(self, manager, temp_paths):
        """Test graceful handling of corrupted state file."""
        _, state_path = temp_paths

        # Write invalid JSON
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{invalid json")

        # Should load gracefully
        new_manager = AlertManager(
            alert_log_path=str(temp_paths[0]),
            state_path=str(state_path),
        )
        assert len(new_manager._last_fired) == 0  # Fresh start

    def test_get_active_alerts(self, manager):
        """Test retrieving active alerts."""
        alert1 = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert1 is not None
        manager._fire_alert(alert1)

        active = manager.get_active_alerts()
        assert len(active) == 1
        assert active[0].alert_type == "drawdown"

    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None
        manager._fire_alert(alert)

        manager.acknowledge("drawdown")

        active = manager.get_active_alerts()
        assert len(active) == 0

    def test_alert_log_file(self, manager):
        """Test that alerts are logged to JSONL file."""
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None
        manager._fire_alert(alert)

        # Check file was written
        log_path = manager.alert_log_path
        assert log_path.exists()

        # Read and verify content
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1

        logged_alert = json.loads(lines[0])
        assert logged_alert["alert_type"] == "drawdown"

    def test_get_summary(self, manager):
        """Test alert summary generation."""
        # No alerts
        summary = manager.get_summary()
        assert "No active alerts" in summary

        # With alerts
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None
        manager._fire_alert(alert)

        summary = manager.get_summary()
        assert "active alert" in summary
        assert "drawdown" in summary

    def test_clear_cooldowns(self, manager):
        """Test clearing cooldown timers."""
        # Fire an alert
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is not None
        manager._fire_alert(alert)

        # Cooldown is set
        assert "drawdown" in manager._last_fired

        # Clear cooldowns
        manager.clear_cooldowns()
        assert len(manager._last_fired) == 0

        # New alert should fire
        new_alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert new_alert is not None


class TestAlertConfigCustomization:
    """Test custom alert configurations."""

    def test_custom_config(self, tmp_path):
        """Test customizing alert thresholds."""
        custom = {
            "drawdown": {
                "threshold": 0.10,  # Looser threshold
                "severity": "WARNING",
                "cooldown_minutes": 30,
                "message_template": "Custom: {value:.1%} > {threshold:.1%}",
            }
        }

        manager = AlertManager(
            alert_log_path=str(tmp_path / "alerts.jsonl"),
            state_path=str(tmp_path / "state.json"),
            custom_configs=custom,
        )

        # Alert with 6% drawdown should not fire (threshold is 10%)
        alert = manager.check_drawdown(current_nav=9400, peak_nav=10000)
        assert alert is None

        # Alert with 11% drawdown should fire
        alert = manager.check_drawdown(current_nav=8900, peak_nav=10000)
        assert alert is not None
        assert "Custom:" in alert.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
