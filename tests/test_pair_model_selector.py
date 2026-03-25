"""Test suite for PairModelSelector.

Tests per-pair model selection with rolling accuracy tracking, automatic switching,
and file persistence with concurrent access handling.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.scanner.automation.pair_model_selector import (
    PairModelRecord,
    PairModelSelector,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def selector(temp_dir):
    """Create a PairModelSelector with temporary paths."""
    return PairModelSelector(
        registry_path=str(temp_dir / "registry.json"),
        selections_log_path=str(temp_dir / "selections.jsonl"),
        models_dir=str(temp_dir / "models"),
        rolling_window=10,
        min_trades=5,
        switch_threshold=0.05,
    )


class TestPairModelRecord:
    """Test PairModelRecord dataclass."""

    def test_creation(self):
        """Test creating a PairModelRecord."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.6,
            per_pair_accuracy=0.55,
            joint_trades=10,
            per_pair_trades=8,
            last_switch="",
            reason="default",
        )
        assert rec.pair == "EUR_USD"
        assert rec.active_model == "joint"
        assert rec.joint_accuracy == 0.6
        assert rec.per_pair_trades == 8


class TestPairModelSelector:
    """Test PairModelSelector functionality."""

    def test_initialization(self, selector):
        """Test selector initialization."""
        assert selector.rolling_window == 10
        assert selector.min_trades == 5
        assert selector.switch_threshold == 0.05
        assert len(selector._registry) == 0

    def test_get_active_model_default(self, selector):
        """Test default active model for new pair."""
        model = selector.get_active_model("EUR_USD")
        assert model == "joint"

    def test_get_active_model_existing(self, selector):
        """Test getting active model for existing pair."""
        selector._registry["GBP_USD"] = PairModelRecord(
            pair="GBP_USD",
            active_model="per_pair",
            joint_accuracy=0.5,
            per_pair_accuracy=0.65,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="2026-03-19T10:00:00Z",
            reason="per_pair superior",
        )
        model = selector.get_active_model("GBP_USD")
        assert model == "per_pair"

    def test_record_prediction_correct(self, selector):
        """Test recording a correct prediction."""
        selector.record_prediction(
            pair="EUR_USD",
            model_type="joint",
            predicted_direction="LONG",
            actual_direction="LONG",
        )
        assert "EUR_USD" in selector._registry
        rec = selector._registry["EUR_USD"]
        assert rec.joint_trades == 1
        assert rec.joint_accuracy == 1.0  # Correct

    def test_record_prediction_incorrect(self, selector):
        """Test recording an incorrect prediction."""
        selector.record_prediction(
            pair="EUR_USD",
            model_type="joint",
            predicted_direction="LONG",
            actual_direction="SHORT",
        )
        rec = selector._registry["EUR_USD"]
        assert rec.joint_trades == 1
        assert rec.joint_accuracy == 0.0  # Incorrect

    def test_record_prediction_per_pair(self, selector):
        """Test recording per_pair model predictions."""
        selector.record_prediction(
            pair="EUR_USD",
            model_type="per_pair",
            predicted_direction="SHORT",
            actual_direction="SHORT",
        )
        rec = selector._registry["EUR_USD"]
        assert rec.per_pair_trades == 1
        assert rec.per_pair_accuracy == 1.0

    def test_record_prediction_ema_update(self, selector):
        """Test exponential moving average update."""
        pair = "EUR_USD"
        # First prediction: correct
        selector.record_prediction(pair, "joint", "LONG", "LONG")
        assert selector._registry[pair].joint_accuracy == 1.0

        # Second prediction: incorrect
        selector.record_prediction(pair, "joint", "LONG", "SHORT")
        # EMA: alpha * 0.0 + (1-alpha) * 1.0
        # alpha = 2/(10+1) ≈ 0.1818
        expected_acc = (1 - 2 / (10 + 1)) * 1.0
        assert abs(selector._registry[pair].joint_accuracy - expected_acc) < 0.01

    def test_record_prediction_invalid_pair(self, selector):
        """Test recording with invalid pair."""
        selector.record_prediction(
            pair=None,
            model_type="joint",
            predicted_direction="LONG",
            actual_direction="LONG",
        )
        assert len(selector._registry) == 0

    def test_record_prediction_invalid_model_type(self, selector):
        """Test recording with invalid model type."""
        selector.record_prediction(
            pair="EUR_USD",
            model_type="invalid",
            predicted_direction="LONG",
            actual_direction="LONG",
        )
        assert len(selector._registry) == 0

    def test_check_switch_insufficient_trades(self, selector):
        """Test no switch with insufficient trades."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.5,
            per_pair_accuracy=0.8,  # Much better
            joint_trades=3,  # Below min_trades=5
            per_pair_trades=3,
            last_switch="",
            reason="default",
        )
        selector._registry["EUR_USD"] = rec

        switch = selector.check_switch("EUR_USD")
        assert switch is None

    def test_check_switch_joint_to_per_pair(self, selector):
        """Test switching from joint to per_pair when superior."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.55,
            per_pair_accuracy=0.62,  # 7% better
            joint_trades=10,
            per_pair_trades=10,
            last_switch="",
            reason="default",
        )
        selector._registry["EUR_USD"] = rec

        switch = selector.check_switch("EUR_USD")
        assert switch == "per_pair"

    def test_check_switch_no_switch_below_threshold(self, selector):
        """Test no switch when improvement is below threshold."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.55,
            per_pair_accuracy=0.58,  # Only 3% better (threshold is 5%)
            joint_trades=10,
            per_pair_trades=10,
            last_switch="",
            reason="default",
        )
        selector._registry["EUR_USD"] = rec

        switch = selector.check_switch("EUR_USD")
        assert switch is None

    def test_check_switch_per_pair_to_joint(self, selector):
        """Test switching back to joint when joint becomes superior."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="per_pair",
            joint_accuracy=0.65,  # Joint now better
            per_pair_accuracy=0.55,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="2026-03-18T10:00:00Z",
            reason="per_pair was better",
        )
        selector._registry["EUR_USD"] = rec

        switch = selector.check_switch("EUR_USD")
        assert switch == "joint"

    def test_execute_switch(self, selector):
        """Test executing a model switch."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.55,
            per_pair_accuracy=0.62,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="",
            reason="default",
        )
        selector._registry["EUR_USD"] = rec

        selector.execute_switch("EUR_USD", "per_pair")

        # Check updated record
        updated = selector._registry["EUR_USD"]
        assert updated.active_model == "per_pair"
        assert updated.last_switch != ""
        assert "switched from joint to per_pair" in updated.reason.lower()

    def test_execute_switch_logs_to_jsonl(self, selector, temp_dir):
        """Test that switch is logged to JSONL file."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.55,
            per_pair_accuracy=0.62,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="",
            reason="default",
        )
        selector._registry["EUR_USD"] = rec

        selector.execute_switch("EUR_USD", "per_pair")

        # Read JSONL log
        log_path = temp_dir / "selections.jsonl"
        assert log_path.exists()
        with open(log_path) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["pair"] == "EUR_USD"
            assert event["old_model"] == "joint"
            assert event["new_model"] == "per_pair"

    def test_has_per_pair_model(self, selector, temp_dir):
        """Test checking for per-pair model files."""
        # Create a per-pair model directory with .keras file
        pair_dir = temp_dir / "models" / "EUR_USD"
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "model.keras").touch()

        assert selector.has_per_pair_model("EUR_USD") is True
        assert selector.has_per_pair_model("GBP_USD") is False

    def test_get_status(self, selector):
        """Test getting status of all pairs."""
        selector._registry["EUR_USD"] = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.6,
            per_pair_accuracy=0.55,
            joint_trades=10,
            per_pair_trades=8,
            last_switch="",
            reason="default",
        )
        selector._registry["GBP_USD"] = PairModelRecord(
            pair="GBP_USD",
            active_model="per_pair",
            joint_accuracy=0.5,
            per_pair_accuracy=0.65,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="2026-03-19T10:00:00Z",
            reason="per_pair superior",
        )

        status = selector.get_status()
        assert len(status) == 2
        assert status["EUR_USD"]["active_model"] == "joint"
        assert status["GBP_USD"]["active_model"] == "per_pair"

    def test_get_summary(self, selector):
        """Test getting human-readable summary."""
        selector._registry["EUR_USD"] = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.6,
            per_pair_accuracy=0.55,
            joint_trades=10,
            per_pair_trades=8,
            last_switch="",
            reason="default",
        )

        summary = selector.get_summary()
        assert "EUR_USD" in summary
        assert "joint" in summary.lower()
        assert "0.600" in summary or "0.60" in summary

    def test_reset_pair(self, selector):
        """Test resetting a pair's history."""
        selector._registry["EUR_USD"] = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.6,
            per_pair_accuracy=0.55,
            joint_trades=10,
            per_pair_trades=8,
            last_switch="",
            reason="default",
        )

        selector.reset_pair("EUR_USD")
        assert "EUR_USD" not in selector._registry

    def test_get_pair_stats(self, selector):
        """Test getting detailed stats for a pair."""
        rec = PairModelRecord(
            pair="EUR_USD",
            active_model="joint",
            joint_accuracy=0.55,
            per_pair_accuracy=0.65,
            joint_trades=10,
            per_pair_trades=10,
            last_switch="2026-03-18T10:00:00Z",
            reason="joint was better",
        )
        selector._registry["EUR_USD"] = rec

        stats = selector.get_pair_stats("EUR_USD")
        assert stats is not None
        assert stats["pair"] == "EUR_USD"
        assert stats["active_model"] == "joint"
        assert stats["joint_accuracy"] == 0.55
        assert stats["per_pair_accuracy"] == 0.65
        assert stats["recommended_model"] == "per_pair"  # Should switch

    def test_persistence(self, temp_dir):
        """Test saving and loading registry from JSON."""
        # Create and save
        selector1 = PairModelSelector(
            registry_path=str(temp_dir / "registry.json"),
            selections_log_path=str(temp_dir / "selections.jsonl"),
            models_dir=str(temp_dir / "models"),
        )
        selector1._registry["EUR_USD"] = PairModelRecord(
            pair="EUR_USD",
            active_model="per_pair",
            joint_accuracy=0.55,
            per_pair_accuracy=0.65,
            joint_trades=20,
            per_pair_trades=20,
            last_switch="2026-03-19T10:00:00Z",
            reason="switched to per_pair",
        )
        selector1._save_registry()

        # Load in new instance
        selector2 = PairModelSelector(
            registry_path=str(temp_dir / "registry.json"),
            selections_log_path=str(temp_dir / "selections.jsonl"),
            models_dir=str(temp_dir / "models"),
        )

        assert "EUR_USD" in selector2._registry
        rec = selector2._registry["EUR_USD"]
        assert rec.active_model == "per_pair"
        assert rec.joint_accuracy == 0.55

    def test_concurrent_writes(self, selector, temp_dir):
        """Test file locking handles concurrent writes gracefully."""
        import threading

        pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
        results = []

        def write_records(pair):
            for i in range(5):
                selector.record_prediction(
                    pair=pair,
                    model_type="joint",
                    predicted_direction="LONG",
                    actual_direction="LONG" if i % 2 == 0 else "SHORT",
                )
            results.append(pair)

        threads = [threading.Thread(target=write_records, args=(p,)) for p in pairs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all writes succeeded
        assert len(results) == 3
        assert all(p in selector._registry for p in pairs)


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_workflow_new_pair_with_per_pair_model(self, selector, temp_dir):
        """Test complete workflow: new pair, record predictions, switch models."""
        # Setup: create per-pair model file
        pair_dir = temp_dir / "models" / "EUR_USD"
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "model.keras").touch()

        # Record predictions favoring per_pair
        for i in range(15):
            joint_correct = i % 3 == 0  # 33% win rate
            per_pair_correct = i % 2 == 0  # 50% win rate
            selector.record_prediction(
                "EUR_USD", "joint", "LONG", "LONG" if joint_correct else "SHORT"
            )
            selector.record_prediction(
                "EUR_USD", "per_pair", "LONG", "LONG" if per_pair_correct else "SHORT"
            )

        # Check switch recommendation
        switch = selector.check_switch("EUR_USD")
        assert switch == "per_pair"

        # Execute switch
        selector.execute_switch("EUR_USD", "per_pair")
        assert selector.get_active_model("EUR_USD") == "per_pair"

    def test_workflow_pair_regression(self, selector):
        """Test switching back when a model regresses."""
        # Start with per_pair
        rec = PairModelRecord(
            pair="GBP_USD",
            active_model="per_pair",
            joint_accuracy=0.45,
            per_pair_accuracy=0.70,
            joint_trades=20,
            per_pair_trades=20,
            last_switch="2026-03-18T10:00:00Z",
            reason="per_pair superior",
        )
        selector._registry["GBP_USD"] = rec

        # Simulate per_pair degrading
        for _ in range(10):
            selector.record_prediction(
                "GBP_USD", "per_pair", "LONG", "SHORT"  # Wrong predictions
            )
        for _ in range(10):
            selector.record_prediction(
                "GBP_USD", "joint", "LONG", "LONG"  # Correct predictions
            )

        # Joint should now be recommended
        switch = selector.check_switch("GBP_USD")
        assert switch == "joint"
