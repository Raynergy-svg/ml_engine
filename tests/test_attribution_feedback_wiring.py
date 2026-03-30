"""Unit tests for Phase 50 (US-308-ext): Attribution feedback wiring.

Tests that attribution weight recommendations are properly piped to the
gate threshold optimizer feedback log.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.scanner.gate_threshold_optimizer import GateThresholdOptimizer


class TestGateThresholdOptimizerAgentFeedback:
    """Test agent feedback recording in GateThresholdOptimizer."""

    def test_record_agent_feedback(self) -> None:
        """Can record agent feedback."""
        optimizer = GateThresholdOptimizer()
        optimizer.record_agent_feedback(
            agent_name="trend_detector",
            weight_delta=1.15,
            source="trade_attribution",
        )

        feedback = optimizer.get_agent_feedback()
        assert len(feedback) == 1
        assert feedback[0]["agent_name"] == "trend_detector"
        assert feedback[0]["weight_delta"] == 1.15
        assert feedback[0]["source"] == "trade_attribution"

    def test_record_multiple_feedbacks(self) -> None:
        """Can record multiple agent feedbacks."""
        optimizer = GateThresholdOptimizer()
        optimizer.record_agent_feedback("agent1", 1.10, "attribution")
        optimizer.record_agent_feedback("agent2", 0.95, "attribution")
        optimizer.record_agent_feedback("agent3", 1.05, "attribution")

        feedback = optimizer.get_agent_feedback()
        assert len(feedback) == 3
        assert feedback[0]["agent_name"] == "agent1"
        assert feedback[1]["agent_name"] == "agent2"
        assert feedback[2]["agent_name"] == "agent3"

    def test_feedback_includes_timestamp(self) -> None:
        """Recorded feedback includes timestamp."""
        optimizer = GateThresholdOptimizer()
        optimizer.record_agent_feedback("test_agent", 1.05)

        feedback = optimizer.get_agent_feedback()
        assert "timestamp" in feedback[0]
        assert len(feedback[0]["timestamp"]) > 0

    def test_feedback_trim_on_overflow(self) -> None:
        """Feedback history trimmed when exceeds max."""
        optimizer = GateThresholdOptimizer()

        # Record 250 feedbacks (exceeds default 200 max)
        for i in range(250):
            optimizer.record_agent_feedback(f"agent_{i % 5}", 1.0 + (i % 10) / 100)

        feedback = optimizer.get_agent_feedback()
        # Should trim to approximately half (100 entries kept from last 200)
        assert len(feedback) <= 150
        assert len(feedback) > 0

    def test_feedback_with_default_source(self) -> None:
        """Feedback defaults source to 'unknown'."""
        optimizer = GateThresholdOptimizer()
        optimizer.record_agent_feedback("test", 1.05)

        feedback = optimizer.get_agent_feedback()
        assert feedback[0]["source"] == "unknown"

    def test_feedback_persists_in_save_state(self) -> None:
        """Agent feedback is included in save_state()."""
        import tempfile
        import json
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "optimizer_state.json")
            config = MagicMock()
            config.persistence_path = path
            config.max_history_per_gate = 10

            optimizer = GateThresholdOptimizer(config=config)
            optimizer.record_agent_feedback("momentum", 1.12, "attribution")
            optimizer.record_agent_feedback("volatility", 0.98, "attribution")
            optimizer.save_state()

            # Verify file contains feedback
            with open(path, "r") as f:
                saved = json.load(f)

            assert "agent_feedback" in saved
            assert len(saved["agent_feedback"]) == 2
            assert saved["agent_feedback"][0]["agent_name"] == "momentum"

    def test_feedback_loads_from_saved_state(self) -> None:
        """Agent feedback loads correctly from saved state."""
        import tempfile
        import json
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "optimizer_state.json")
            config = MagicMock()
            config.persistence_path = path
            config.max_history_per_gate = 10

            # Create and save
            optimizer1 = GateThresholdOptimizer(config=config)
            optimizer1.record_agent_feedback("agent_x", 1.20)
            optimizer1.record_agent_feedback("agent_y", 0.90)
            optimizer1.save_state()

            # Load into new instance
            optimizer2 = GateThresholdOptimizer(config=config)
            feedback = optimizer2.get_agent_feedback()

            assert len(feedback) == 2
            assert feedback[0]["agent_name"] == "agent_x"
            assert feedback[0]["weight_delta"] == 1.20
            assert feedback[1]["agent_name"] == "agent_y"

    def test_record_feedback_exception_handling(self) -> None:
        """Exception in record_agent_feedback is handled gracefully."""
        optimizer = GateThresholdOptimizer()

        # Should not raise even with invalid types
        optimizer.record_agent_feedback(None, None, None)  # type: ignore
        optimizer.record_agent_feedback("test", "invalid", 123)  # type: ignore

        # Should still be functional
        optimizer.record_agent_feedback("valid", 1.05)
        feedback = optimizer.get_agent_feedback()
        assert any(f["agent_name"] == "valid" for f in feedback)


class TestAttributionFeedbackIntegration:
    """Integration tests for attribution feedback piping."""

    def test_multiple_agents_same_feedback_round(self) -> None:
        """Multiple agents can contribute feedback in one round."""
        optimizer = GateThresholdOptimizer()

        # Simulate attribution engine returning multiple weight recommendations
        weight_recs = {
            "trend_detector": 1.15,
            "momentum_agent": 1.08,
            "volatility_agent": 0.97,
            "mean_reversion_agent": 1.02,
        }

        for agent_name, weight_delta in weight_recs.items():
            optimizer.record_agent_feedback(agent_name, weight_delta, "trade_attribution")

        feedback = optimizer.get_agent_feedback()
        assert len(feedback) == 4
        agents = {f["agent_name"] for f in feedback}
        assert agents == {"trend_detector", "momentum_agent", "volatility_agent", "mean_reversion_agent"}

    def test_feedback_across_multiple_rounds(self) -> None:
        """Feedback accumulates across multiple trade attribution rounds."""
        optimizer = GateThresholdOptimizer()

        # Round 1
        for agent, delta in {"a": 1.10, "b": 0.95}.items():
            optimizer.record_agent_feedback(agent, delta, "round_1")

        # Round 2
        for agent, delta in {"a": 1.05, "c": 1.12}.items():
            optimizer.record_agent_feedback(agent, delta, "round_2")

        feedback = optimizer.get_agent_feedback()
        assert len(feedback) == 4
        # Both agents "a" and "b" can appear multiple times (per-trade basis)
        agent_counts = {}
        for f in feedback:
            agent_counts[f["agent_name"]] = agent_counts.get(f["agent_name"], 0) + 1
        assert agent_counts["a"] == 2
        assert agent_counts["b"] == 1
        assert agent_counts["c"] == 1

    def test_weight_delta_ranges(self) -> None:
        """Feedback correctly records various weight delta ranges."""
        optimizer = GateThresholdOptimizer()

        deltas = [0.5, 0.90, 1.0, 1.05, 1.50, 2.0]
        for i, delta in enumerate(deltas):
            optimizer.record_agent_feedback(f"agent_{i}", delta)

        feedback = optimizer.get_agent_feedback()
        recorded_deltas = [f["weight_delta"] for f in feedback]
        assert recorded_deltas == deltas
