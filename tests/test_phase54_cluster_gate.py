"""
Tests for Phase 54 US-337: Cluster-Aware Confidence Gate.

Tests cover:
  - Confidence boost for high-win cluster match
  - Confidence penalty for low-win cluster match
  - Pass-through with no clusters
  - Adjustment capping
  - Stats tracking
  - Lazy imports and wiring
"""

import pytest
from unittest.mock import MagicMock

from src.scanner.cluster_gate import ClusterConfidenceGate, ClusterGateResult
from src.scanner.trade_clusters import ClusterInfo


@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    analyzer.get_cluster_registry.return_value = {0: {}, 1: {}}
    return analyzer


@pytest.fixture
def high_win_cluster():
    return ClusterInfo(
        cluster_id=0, win_rate=0.65, count=30,
        confidence_adjustment=0.05, label="high_win",
        centroid=[1, 1, 0.7, 0.6, 10, 0],
    )


@pytest.fixture
def low_win_cluster():
    return ClusterInfo(
        cluster_id=1, win_rate=0.20, count=25,
        confidence_adjustment=-0.10, label="low_win",
        centroid=[0, 2, 0.3, 0.3, 30, 3],
    )


class TestConfidenceAdjustment:

    def test_boost_high_win_cluster(self, mock_analyzer, high_win_cluster):
        """High-win cluster boosts confidence."""
        mock_analyzer.classify_trade.return_value = high_win_cluster
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.60)
        assert result.adjusted_confidence > 0.60
        assert result.cluster_label == "high_win"

    def test_penalize_low_win_cluster(self, mock_analyzer, low_win_cluster):
        """Low-win cluster penalizes confidence."""
        mock_analyzer.classify_trade.return_value = low_win_cluster
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.60)
        assert result.adjusted_confidence < 0.60
        assert result.cluster_label == "low_win"

    def test_no_match_passthrough(self, mock_analyzer):
        """No cluster match passes through unchanged."""
        mock_analyzer.classify_trade.return_value = None
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.60)
        assert result.adjusted_confidence == 0.60
        assert result.cluster_matched is False

    def test_no_analyzer_passthrough(self):
        """No analyzer available passes through."""
        gate = ClusterConfidenceGate(cluster_analyzer=None)
        gate._analyzer = None  # Force no analyzer
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.70)
        assert result.adjusted_confidence == 0.70

    def test_adjustment_capped_max_boost(self, mock_analyzer):
        """Adjustment capped at max_boost."""
        big_boost = ClusterInfo(
            cluster_id=0, win_rate=0.9, count=50,
            confidence_adjustment=0.50, label="high_win",
        )
        mock_analyzer.classify_trade.return_value = big_boost
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer, max_boost=0.10)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.50)
        assert result.confidence_adjustment <= 0.10

    def test_adjustment_capped_max_penalty(self, mock_analyzer):
        """Adjustment capped at max_penalty."""
        big_penalty = ClusterInfo(
            cluster_id=1, win_rate=0.1, count=50,
            confidence_adjustment=-0.50, label="low_win",
        )
        mock_analyzer.classify_trade.return_value = big_penalty
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer, max_penalty=-0.10)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.50)
        assert result.confidence_adjustment >= -0.10

    def test_confidence_bounded_0_1(self, mock_analyzer, low_win_cluster):
        """Adjusted confidence stays in [0, 1]."""
        mock_analyzer.classify_trade.return_value = low_win_cluster
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer)
        result = gate.evaluate({"pair": "EUR_USD"}, confidence=0.05)
        assert result.adjusted_confidence >= 0.0
        assert result.adjusted_confidence <= 1.0


class TestGateResult:

    def test_to_dict(self):
        r = ClusterGateResult(
            original_confidence=0.6, adjusted_confidence=0.65,
            cluster_matched=True, cluster_id=0, cluster_label="high_win",
        )
        d = r.to_dict()
        assert d["cluster_matched"] is True
        assert abs(d["original_confidence"] - 0.6) < 0.01


class TestStats:

    def test_stats_tracking(self, mock_analyzer, high_win_cluster):
        mock_analyzer.classify_trade.return_value = high_win_cluster
        gate = ClusterConfidenceGate(cluster_analyzer=mock_analyzer)
        gate.evaluate({}, confidence=0.5)
        gate.evaluate({}, confidence=0.6)
        stats = gate.get_stats()
        assert stats["total_evaluations"] == 2
        assert stats["match_count"] == 2
        assert stats["match_rate"] == 1.0


class TestLazyImports:

    def test_cluster_gate_import(self):
        from src.scanner import ClusterConfidenceGate
        assert ClusterConfidenceGate is not None


class TestWiring:

    def test_execution_manager_has_cluster_gate(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_cluster_gate" in source
            assert "ClusterConfidenceGate" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
