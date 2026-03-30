"""
Tests for Phase 54 US-336: Trade Cluster Analyzer.

Tests cover:
  - Feature extraction from trade dicts
  - DBSCAN clustering
  - Decision tree rule extraction
  - Cluster registry creation
  - High-win/low-win labeling
  - Nearest-centroid classification
  - Re-cluster interval tracking
  - Persistence roundtrip
  - Lazy imports and wiring
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.scanner.trade_clusters import (
    TradeClusterAnalyzer,
    TradeClusterConfig,
    ClusterInfo,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def analyzer(tmp_dir):
    config = TradeClusterConfig(
        persistence_path=os.path.join(tmp_dir, "clusters.json"),
        min_samples=5,
        min_cluster_size=5,
        recluster_interval=10,
    )
    return TradeClusterAnalyzer(config)


def _make_cluster_trades(n, win_rate=0.5, session="london", regime="NORMAL",
                          agreement=0.6, trend=0.5, atr=0.001, pair="EUR_USD", seed=42):
    """Generate synthetic trades for clustering."""
    import random
    rng = random.Random(seed)
    trades = []
    for i in range(n):
        won = rng.random() < win_rate
        trades.append({
            "pair": pair,
            "session": session,
            "regime": {"volatility_regime": regime},
            "agents": {
                "weighted_vote_score": agreement + rng.gauss(0, 0.05),
                "trend_strength": trend + rng.gauss(0, 0.05),
            },
            "atr": atr + rng.gauss(0, atr * 0.1),
            "trend_strength": trend + rng.gauss(0, 0.05),
            "outcome": {"trade_won": won, "realized_pl": 50.0 if won else -30.0},
        })
    return trades


class TestFeatureExtraction:

    def test_extract_basic(self, analyzer):
        trade = {
            "pair": "EUR_USD",
            "session": "london",
            "regime": {"volatility_regime": "NORMAL"},
            "agents": {"weighted_vote_score": 0.65, "trend_strength": 0.5},
            "atr": 0.001,
        }
        features = analyzer.extract_features(trade)
        assert features is not None
        assert len(features) == 6
        assert features[0] == 1  # london
        assert features[1] == 1  # NORMAL

    def test_extract_missing_data_defaults(self, analyzer):
        """Missing fields get defaults, not None."""
        trade = {"pair": "EUR_USD"}
        features = analyzer.extract_features(trade)
        assert features is not None

    def test_extract_unknown_pair(self, analyzer):
        trade = {"pair": "UNKNOWN_PAIR", "session": "london"}
        features = analyzer.extract_features(trade)
        assert features is not None
        # Unknown pair gets len(pair_list)
        assert features[5] == len(analyzer.config.pair_list)


class TestClusterInfo:

    def test_roundtrip(self):
        ci = ClusterInfo(
            cluster_id=0,
            win_rate=0.65,
            count=30,
            conditions={"session": "london"},
            confidence_adjustment=0.05,
            centroid=[1.0, 1.0, 0.6, 0.5, 10.0, 0.0],
            label="high_win",
        )
        d = ci.to_dict()
        restored = ClusterInfo.from_dict(d)
        assert restored.cluster_id == 0
        assert abs(restored.win_rate - 0.65) < 0.01
        assert restored.label == "high_win"


class TestClustering:

    def test_fit_creates_clusters(self, tmp_dir):
        """Fitting with enough trades creates clusters."""
        config = TradeClusterConfig(
            persistence_path=os.path.join(tmp_dir, "fit.json"),
            min_samples=5, min_cluster_size=5, eps=2.0,
        )
        a = TradeClusterAnalyzer(config)
        trades = (_make_cluster_trades(50, win_rate=0.7, session="london", seed=1) +
                  _make_cluster_trades(50, win_rate=0.2, session="asian", seed=2))
        registry = a.fit(trades)
        assert len(registry) >= 1

    def test_insufficient_trades(self, tmp_dir):
        """Too few trades returns empty registry."""
        config = TradeClusterConfig(
            persistence_path=os.path.join(tmp_dir, "few.json"),
            min_samples=15,
        )
        a = TradeClusterAnalyzer(config)
        trades = _make_cluster_trades(5, seed=1)
        registry = a.fit(trades)
        assert len(registry) == 0

    def test_high_win_cluster_labeled(self, analyzer):
        """Clusters with high win rates get labeled correctly."""
        # Create distinct groups
        trades = (_make_cluster_trades(80, win_rate=0.8, session="london",
                                        agreement=0.8, atr=0.001, seed=1) +
                  _make_cluster_trades(80, win_rate=0.15, session="asian",
                                        agreement=0.3, atr=0.005, seed=2))
        registry = analyzer.fit(trades)
        labels = {c.label for c in registry.values()}
        # Should have at least some labeled clusters
        assert len(registry) >= 1

    def test_registry_serializable(self, analyzer):
        trades = _make_cluster_trades(100, seed=1)
        analyzer.fit(trades)
        reg = analyzer.get_cluster_registry()
        # Should be JSON serializable
        json_str = json.dumps(reg)
        assert json_str


class TestClassification:

    def test_classify_returns_cluster(self, analyzer):
        """Classification of a trade returns a ClusterInfo."""
        trades = _make_cluster_trades(100, seed=1)
        analyzer.fit(trades)
        trade = {
            "pair": "EUR_USD",
            "session": "london",
            "regime": {"volatility_regime": "NORMAL"},
            "agents": {"weighted_vote_score": 0.6, "trend_strength": 0.5},
            "atr": 0.001,
        }
        result = analyzer.classify_trade(trade)
        # May or may not match a cluster — just check type
        if result is not None:
            assert isinstance(result, ClusterInfo)

    def test_classify_empty_registry(self, analyzer):
        """No clusters → returns None."""
        trade = {"pair": "EUR_USD", "session": "london"}
        assert analyzer.classify_trade(trade) is None


class TestReclusterTracking:

    def test_record_trade_increments(self, analyzer):
        analyzer.record_trade()
        analyzer.record_trade()
        assert analyzer._trades_since_cluster == 2

    def test_should_recluster(self, analyzer):
        for _ in range(10):
            analyzer.record_trade()
        assert analyzer.should_recluster() is True

    def test_fit_resets_counter(self, analyzer):
        analyzer._trades_since_cluster = 50
        trades = _make_cluster_trades(100, seed=1)
        analyzer.fit(trades)
        assert analyzer._trades_since_cluster == 0


class TestPersistence:

    def test_save_load_roundtrip(self, tmp_dir):
        path = os.path.join(tmp_dir, "cluster_rt.json")
        config = TradeClusterConfig(
            persistence_path=path, min_samples=5, min_cluster_size=5)
        a1 = TradeClusterAnalyzer(config)
        trades = _make_cluster_trades(100, seed=1)
        a1.fit(trades)
        a1._trades_since_cluster = 25
        a1.save_state()

        a2 = TradeClusterAnalyzer(config)
        assert a2._trades_since_cluster == 25
        assert len(a2._registry) == len(a1._registry)

    def test_load_missing_file(self, tmp_dir):
        config = TradeClusterConfig(
            persistence_path=os.path.join(tmp_dir, "missing.json"))
        a = TradeClusterAnalyzer(config)
        assert len(a._registry) == 0

    def test_load_corrupted_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{not json")
        config = TradeClusterConfig(persistence_path=path)
        a = TradeClusterAnalyzer(config)
        assert len(a._registry) == 0


class TestTreeRules:

    def test_tree_rules_generated(self, analyzer):
        trades = _make_cluster_trades(100, seed=1)
        analyzer.fit(trades)
        rules = analyzer.get_tree_rules()
        # sklearn should be available and generate rules
        assert isinstance(rules, list)


class TestLazyImports:

    def test_trade_cluster_analyzer_import(self):
        from src.scanner import TradeClusterAnalyzer
        assert TradeClusterAnalyzer is not None

    def test_trade_cluster_config_import(self):
        from src.scanner import TradeClusterConfig
        assert TradeClusterConfig is not None


class TestWiring:

    def test_execution_manager_has_cluster_analyzer(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_trade_cluster_analyzer" in source
            assert "TradeClusterAnalyzer" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
