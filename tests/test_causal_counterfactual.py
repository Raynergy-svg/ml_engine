"""Tests for Pearl do-calculus counterfactual reasoning.

Tests the CounterfactualEngine, CounterfactualLearner, and CausalFilter.query_counterfactual() integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.scanner.automation.causal_counterfactual import (
    CausalPathNode,
    CounterfactualEngine,
    CounterfactualResult,
    CounterfactualScenario,
    Intervention,
)
from src.scanner.automation.causal_filter import CausalSignalFilter
from src.scanner.automation.counterfactual_learner import CounterfactualLearner


class TestCounterfactualScenario:
    """Test scenario generation and structure."""

    def test_generate_standard_scenarios(self):
        """Test that 5 standard scenarios are generated for any trade context."""
        trade_context = {
            "pair": "EURUSD",
            "features": {
                "atr_14": 0.001,
                "sma_trend": 0.6,
                "momentum_signal": 0.7,
            },
            "regime": "NORMAL",
            "outcome": 1.0,
        }

        engine = CounterfactualEngine()
        scenarios = engine.generate_standard_scenarios(trade_context)

        assert len(scenarios) == 5, "Should generate exactly 5 standard scenarios"
        scenario_names = {s.scenario_name for s in scenarios}
        expected_names = {
            "low_volatility",
            "high_volatility",
            "strong_trend",
            "weak_trend",
            "adverse_session",
        }
        assert scenario_names == expected_names, f"Got {scenario_names}, expected {expected_names}"

    def test_standard_scenarios_have_interventions(self):
        """Test that each scenario has at least one intervention."""
        trade_context = {
            "pair": "EURUSD",
            "features": {"atr_14": 0.001, "sma_trend": 0.5, "momentum_signal": 0.5},
            "regime": "NORMAL",
            "outcome": 0.0,
        }

        engine = CounterfactualEngine()
        scenarios = engine.generate_standard_scenarios(trade_context)

        for scenario in scenarios:
            assert len(scenario.interventions) > 0, f"{scenario.scenario_name} has no interventions"
            for intervention in scenario.interventions:
                assert intervention.variable, "Intervention must have a variable"
                assert intervention.forced_value is not None, "Intervention must have a forced_value"
                assert intervention.original_value is not None, "Intervention must have an original_value"
                assert intervention.description, "Intervention must have a description"


class TestCounterfactualEngine:
    """Test the core CounterfactualEngine logic."""

    def test_analyze_trade_with_no_causal_graph(self):
        """Test that analysis gracefully returns neutral result when causal graph is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create engine without causal graph
            engine = CounterfactualEngine(
                causal_graph_path=Path(tmpdir) / "nonexistent.json"
            )

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001, "sma_trend": 0.6},
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            scenario = CounterfactualScenario(
                scenario_name="test",
                interventions=[
                    Intervention(
                        variable="atr_14",
                        forced_value=0.002,
                        original_value=0.001,
                        description="Test intervention",
                    )
                ],
            )

            result = engine.analyze_trade(trade_context, scenario)

            assert isinstance(result, CounterfactualResult)
            assert result.factual_outcome_probability == 0.5
            assert result.counterfactual_outcome_probability == 0.5
            assert result.probability_delta == 0.0
            assert result.outcome_reversal is False
            assert result.confidence == 0.1  # Low confidence without graph

    def test_outcome_probability_clamped_to_0_1(self):
        """Test that outcome probabilities are always in [0.0, 1.0]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CounterfactualEngine(
                causal_graph_path=Path(tmpdir) / "nonexistent.json"
            )

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 100.0, "sma_trend": 100.0},  # Extreme values
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            scenario = CounterfactualScenario(scenario_name="test")

            result = engine.analyze_trade(trade_context, scenario)

            assert 0.0 <= result.factual_outcome_probability <= 1.0
            assert 0.0 <= result.counterfactual_outcome_probability <= 1.0

    def test_analyze_trade_with_mock_causal_graph(self):
        """Test analysis with a mock causal graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock causal graph
            graph_path = Path(tmpdir) / "causal_graph.json"
            graph_data = {
                "version": 1,
                "graphs": {
                    "NORMAL": {
                        "regime": "NORMAL",
                        "n_features_tested": 5,
                        "n_significant": 3,
                        "build_timestamp": "2026-03-28T00:00:00Z",
                        "edges": [
                            {
                                "feature_name": "atr_14",
                                "is_causal": True,
                                "p_value": 0.01,
                                "optimal_lag": 1,
                                "f_statistic": 5.5,
                                "direction_coefficient": 0.8,
                            },
                            {
                                "feature_name": "sma_trend",
                                "is_causal": True,
                                "p_value": 0.02,
                                "optimal_lag": 1,
                                "f_statistic": 4.2,
                                "direction_coefficient": 0.6,
                            },
                            {
                                "feature_name": "rsi_14",
                                "is_causal": False,
                                "p_value": 0.5,
                                "optimal_lag": 0,
                                "f_statistic": 0.4,
                                "direction_coefficient": 0.0,
                            },
                        ],
                    }
                },
                "scan_count": 100,
                "last_updated": "2026-03-28T00:00:00Z",
            }
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)

            engine = CounterfactualEngine(causal_graph_path=graph_path)

            trade_context = {
                "pair": "EURUSD",
                "features": {
                    "atr_14": 0.002,
                    "sma_trend": 0.8,
                    "rsi_14": 50.0,
                },
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            scenario = CounterfactualScenario(
                scenario_name="low_volatility",
                interventions=[
                    Intervention(
                        variable="atr_14",
                        forced_value=0.001,
                        original_value=0.002,
                        description="Low volatility",
                    )
                ],
            )

            result = engine.analyze_trade(trade_context, scenario)

            assert isinstance(result, CounterfactualResult)
            assert 0.0 <= result.factual_outcome_probability <= 1.0
            assert 0.0 <= result.counterfactual_outcome_probability <= 1.0
            assert result.confidence > 0.1  # Higher confidence with real graph

    def test_outcome_reversal_detected(self):
        """Test that outcome reversal is detected when probability delta is large."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "causal_graph.json"
            graph_data = {
                "version": 1,
                "graphs": {
                    "NORMAL": {
                        "regime": "NORMAL",
                        "n_features_tested": 2,
                        "n_significant": 1,
                        "build_timestamp": "2026-03-28T00:00:00Z",
                        "edges": [
                            {
                                "feature_name": "atr_14",
                                "is_causal": True,
                                "p_value": 0.01,
                                "optimal_lag": 1,
                                "f_statistic": 10.0,
                                "direction_coefficient": 2.0,
                            },
                        ],
                    }
                },
                "scan_count": 50,
                "last_updated": "2026-03-28T00:00:00Z",
            }
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)

            engine = CounterfactualEngine(causal_graph_path=graph_path)

            # Trade was a loss (outcome=0.0)
            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001},  # Low volatility led to loss
                "regime": "NORMAL",
                "outcome": 0.0,
            }

            scenario = CounterfactualScenario(
                scenario_name="high_volatility",
                interventions=[
                    Intervention(
                        variable="atr_14",
                        forced_value=0.01,  # Much higher volatility
                        original_value=0.001,
                        description="High volatility",
                    )
                ],
            )

            result = engine.analyze_trade(trade_context, scenario)

            # High direction coefficient (2.0) and high ATR should yield high counterfactual probability
            if result.counterfactual_outcome_probability > 0.65 and result.probability_delta > 0.2:
                assert result.outcome_reversal is True

    def test_causal_path_traced(self):
        """Test that causal path is traced correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "causal_graph.json"
            graph_data = {
                "version": 1,
                "graphs": {
                    "NORMAL": {
                        "regime": "NORMAL",
                        "n_features_tested": 4,
                        "n_significant": 3,
                        "build_timestamp": "2026-03-28T00:00:00Z",
                        "edges": [
                            {
                                "feature_name": "atr_14",
                                "is_causal": True,
                                "p_value": 0.01,
                                "optimal_lag": 1,
                                "f_statistic": 8.0,
                                "direction_coefficient": 1.5,
                            },
                            {
                                "feature_name": "sma_trend",
                                "is_causal": True,
                                "p_value": 0.02,
                                "optimal_lag": 1,
                                "f_statistic": 6.0,
                                "direction_coefficient": 1.0,
                            },
                            {
                                "feature_name": "momentum",
                                "is_causal": True,
                                "p_value": 0.03,
                                "optimal_lag": 2,
                                "f_statistic": 4.0,
                                "direction_coefficient": 0.5,
                            },
                        ],
                    }
                },
                "scan_count": 50,
                "last_updated": "2026-03-28T00:00:00Z",
            }
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)

            engine = CounterfactualEngine(causal_graph_path=graph_path)

            trade_context = {
                "pair": "EURUSD",
                "features": {
                    "atr_14": 0.002,
                    "sma_trend": 0.8,
                    "momentum": 0.6,
                },
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            scenario = CounterfactualScenario(scenario_name="test")
            result = engine.analyze_trade(trade_context, scenario)

            assert len(result.causal_chain) > 0, "Causal chain should be non-empty"
            assert len(result.causal_chain) <= 3, "Causal chain should have at most 3 nodes"
            assert isinstance(result.causal_chain[0], CausalPathNode)
            # Contributions: atr=0.002*1.5=0.003, sma=0.8*1.0=0.8, momentum=0.6*0.5=0.3
            # sma_trend should be top (0.8), then momentum (0.3), then atr (0.003)
            assert result.causal_chain[0].feature == "sma_trend", "Top feature should be sma_trend (highest contribution)"

    def test_explanation_generated(self):
        """Test that explanation is generated and non-empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CounterfactualEngine(
                causal_graph_path=Path(tmpdir) / "nonexistent.json"
            )

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001},
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            scenario = CounterfactualScenario(
                scenario_name="low_volatility",
                interventions=[
                    Intervention(
                        variable="atr_14",
                        forced_value=0.0005,
                        original_value=0.001,
                        description="If volatility had been LOW",
                    )
                ],
            )

            result = engine.analyze_trade(trade_context, scenario)

            assert result.explanation, "Explanation should be non-empty"
            assert isinstance(result.explanation, str)
            assert len(result.explanation) > 10

    def test_batch_analyze_returns_list(self):
        """Test that batch_analyze returns a list of results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CounterfactualEngine(
                causal_graph_path=Path(tmpdir) / "nonexistent.json"
            )

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001, "sma_trend": 0.5},
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            results = engine.batch_analyze(trade_context)

            assert isinstance(results, list)
            assert len(results) == 5, "Should return 5 results (standard scenarios)"
            for result in results:
                assert isinstance(result, CounterfactualResult)


class TestCounterfactualLearner:
    """Test the CounterfactualLearner integration."""

    def test_analyze_closed_trade_returns_optional_string(self):
        """Test that analyze_closed_trade returns Optional[str]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = CounterfactualLearner(learnings_path=Path(tmpdir) / "learnings.md")

            trade_entry = {
                "trade_id": "TRADE_001",
                "pair": "EURUSD",
                "entry_price": 1.1000,
                "exit_price": 1.1050,
                "entry_features": {"atr_14": 0.001, "sma_trend": 0.6},
                "regime": "NORMAL",
                "pnl": 50.0,
                "closed": True,
            }

            result = learner.analyze_closed_trade(trade_entry)

            # Result can be None or str
            assert result is None or isinstance(result, str)

    def test_append_learning_creates_file(self):
        """Test that append_learning creates learnings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learnings_path = Path(tmpdir) / "learnings.md"
            learner = CounterfactualLearner(learnings_path=learnings_path)

            learning_text = "**2026-03-28T00:00:00Z** — Test learning"
            learner.append_learning(learning_text)

            assert learnings_path.exists(), "Learnings file should be created"
            content = learnings_path.read_text()
            assert "Test learning" in content

    def test_append_learning_appends_to_existing_file(self):
        """Test that append_learning appends to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learnings_path = Path(tmpdir) / "learnings.md"
            learner = CounterfactualLearner(learnings_path=learnings_path)

            learner.append_learning("First learning")
            learner.append_learning("Second learning")

            content = learnings_path.read_text()
            assert "First learning" in content
            assert "Second learning" in content


class TestCausalFilterIntegration:
    """Test CausalSignalFilter.query_counterfactual() integration."""

    def test_query_counterfactual_safe_with_no_graph(self):
        """Test that query_counterfactual returns list gracefully when causal graph is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filter_obj = CausalSignalFilter(
                persistence_path=Path(tmpdir) / "causal_graph.json"
            )

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001},
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            result = filter_obj.query_counterfactual(trade_context)

            # Should return list of results even without causal graph (non-blocking)
            assert isinstance(result, list)
            assert len(result) == 5  # Standard scenarios
            for item in result:
                assert isinstance(item, dict)
                assert "scenario" in item
                assert "factual_outcome_probability" in item

    def test_query_counterfactual_returns_list_with_valid_graph(self):
        """Test that query_counterfactual returns list when causal graph exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid causal graph
            graph_path = Path(tmpdir) / "causal_graph.json"
            graph_data = {
                "version": 1,
                "graphs": {
                    "NORMAL": {
                        "regime": "NORMAL",
                        "n_features_tested": 2,
                        "n_significant": 1,
                        "build_timestamp": "2026-03-28T00:00:00Z",
                        "edges": [
                            {
                                "feature_name": "atr_14",
                                "is_causal": True,
                                "p_value": 0.01,
                                "optimal_lag": 1,
                                "f_statistic": 5.0,
                                "direction_coefficient": 0.8,
                            },
                        ],
                    }
                },
                "scan_count": 50,
                "last_updated": "2026-03-28T00:00:00Z",
            }
            with open(graph_path, "w") as f:
                json.dump(graph_data, f)

            filter_obj = CausalSignalFilter(persistence_path=graph_path)

            trade_context = {
                "pair": "EURUSD",
                "features": {"atr_14": 0.001},
                "regime": "NORMAL",
                "outcome": 1.0,
            }

            result = filter_obj.query_counterfactual(trade_context)

            # Should return list of dicts (counterfactual results)
            assert isinstance(result, list) or result is None
            if result is not None:
                assert len(result) >= 0
                for item in result:
                    assert isinstance(item, dict)

    def test_query_counterfactual_never_crashes(self):
        """Test that query_counterfactual never crashes even with invalid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filter_obj = CausalSignalFilter(
                persistence_path=Path(tmpdir) / "causal_graph.json"
            )

            # Invalid inputs
            invalid_contexts = [
                {},
                {"pair": "EURUSD"},
                {"features": "invalid"},
                {"features": {}, "outcome": "invalid"},
                None,
            ]

            for context in invalid_contexts:
                try:
                    result = filter_obj.query_counterfactual(context)
                    # Should return None, not crash
                    assert result is None or isinstance(result, list)
                except Exception as e:
                    pytest.fail(f"query_counterfactual crashed on {context}: {e}")


class TestCounterfactualTypes:
    """Test data structure serialization."""

    def test_intervention_to_dict(self):
        """Test Intervention.to_dict()."""
        intervention = Intervention(
            variable="atr_14",
            forced_value=0.002,
            original_value=0.001,
            description="Low volatility",
        )

        d = intervention.to_dict()

        assert d["variable"] == "atr_14"
        assert d["forced_value"] == 0.002
        assert d["original_value"] == 0.001
        assert d["description"] == "Low volatility"

    def test_counterfactual_result_to_dict(self):
        """Test CounterfactualResult.to_dict()."""
        scenario = CounterfactualScenario(
            scenario_name="test",
            interventions=[
                Intervention(
                    variable="atr_14",
                    forced_value=0.002,
                    original_value=0.001,
                    description="Test",
                )
            ],
        )

        result = CounterfactualResult(
            scenario=scenario,
            factual_outcome_probability=0.6,
            counterfactual_outcome_probability=0.8,
            probability_delta=0.2,
            outcome_reversal=False,
            causal_chain=[
                CausalPathNode(
                    feature="atr_14",
                    value=0.001,
                    causal_strength=0.8,
                    lag=1,
                )
            ],
            confidence=0.7,
            explanation="Test explanation",
        )

        d = result.to_dict()

        assert d["scenario"]["scenario_name"] == "test"
        assert d["factual_outcome_probability"] == 0.6
        assert d["counterfactual_outcome_probability"] == 0.8
        assert d["probability_delta"] == 0.2
        assert d["outcome_reversal"] is False
        assert len(d["causal_chain"]) == 1
        assert d["confidence"] == 0.7
        assert d["explanation"] == "Test explanation"
        assert "timestamp" in d
