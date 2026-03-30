"""Pearl do-calculus counterfactual reasoning for FX trade analysis.

Implements:
1. Post-trade counterfactuals: "Would this trade have won if [condition] was different?"
2. Pre-trade interventions: "What if I set volatility to LOW before entering?"
3. Causal path tracing: "Which feature chain led to this outcome?"
4. Regime sensitivity: "How does outcome probability change across regimes?"

Based on: Pearl, J. (2009) Causality. Cambridge University Press.
Simplified do-calculus: P(Y | do(X=x)) via backdoor adjustment using
the causal graph discovered by Granger analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class Intervention:
    """A do() intervention: force a variable to a specific value."""

    variable: str           # feature name (e.g., "atr_14", "rsi_14", "volatility_regime")
    forced_value: float     # the counterfactual value
    original_value: float   # what it actually was
    description: str        # human-readable: "If volatility had been LOW (0.3 vs actual 0.8)"

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "forced_value": round(self.forced_value, 6),
            "original_value": round(self.original_value, 6),
            "description": self.description,
        }


@dataclass
class CounterfactualScenario:
    """A set of interventions defining a counterfactual world."""

    scenario_name: str           # e.g., "low_volatility", "strong_trend", "calm_session"
    interventions: List[Intervention] = field(default_factory=list)
    regime: str = "NORMAL"       # the regime context for this scenario

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "interventions": [i.to_dict() for i in self.interventions],
            "regime": self.regime,
        }


@dataclass
class CausalPathNode:
    """One node in a causal chain from feature to outcome."""

    feature: str
    value: float
    causal_strength: float   # direction_coefficient from GrangerResult
    lag: int

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(self.value, 6),
            "causal_strength": round(self.causal_strength, 6),
            "lag": self.lag,
        }


@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""

    scenario: CounterfactualScenario
    factual_outcome_probability: float     # P(win | actual conditions)
    counterfactual_outcome_probability: float  # P(win | do(interventions))
    probability_delta: float               # counterfactual - factual
    outcome_reversal: bool                 # would the outcome have flipped?
    causal_chain: List[CausalPathNode]     # dominant causal path
    confidence: float                      # 0.0-1.0 confidence in counterfactual
    explanation: str                       # human-readable narrative
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.to_dict(),
            "factual_outcome_probability": round(self.factual_outcome_probability, 4),
            "counterfactual_outcome_probability": round(self.counterfactual_outcome_probability, 4),
            "probability_delta": round(self.probability_delta, 4),
            "outcome_reversal": self.outcome_reversal,
            "causal_chain": [node.to_dict() for node in self.causal_chain],
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }


class CounterfactualEngine:
    """Applies Pearl do-calculus interventions using Granger causal graphs.

    Approach:
    The do-calculus formula P(Y | do(X=x)) requires adjusting for
    backdoor paths. In our causal graph:
    - Nodes: features (ATR, RSI, etc.) + price_returns
    - Edges: Granger-significant causality with direction_coefficient

    For each intervention do(X=x):
    1. "Cut" all incoming edges to X (severing natural causes of X)
    2. Set X = x
    3. Propagate through the causal graph using the Granger coefficients
    4. Estimate P(positive_return) from the propagated values

    Simplification: We use a linear propagation model (true do-calculus
    requires full structural causal model — this is an approximation
    appropriate for the current 30% → 60% improvement target).
    """

    def __init__(
        self,
        causal_graph_path: Optional[Path] = None,
        persistence_path: Optional[Path] = None,
    ):
        self.causal_graph_path = causal_graph_path or (
            _PROJECT_ROOT / "trained_data" / "causal_graph.json"
        )
        self.persistence_path = persistence_path or (
            _PROJECT_ROOT / "trained_data" / "counterfactual_log.jsonl"
        )
        self._causal_graph = None
        self._load_causal_graph()

    def analyze_trade(
        self,
        trade_context: Dict[str, Any],
        scenario: CounterfactualScenario,
    ) -> CounterfactualResult:
        """Analyze a single trade under a counterfactual scenario.

        Args:
            trade_context: Dict with keys: pair, features, regime, outcome
                - features: Dict[str, float] of current feature values
                - outcome: float (1.0 for win, 0.0 for loss)
            scenario: The counterfactual scenario to test

        Returns:
            CounterfactualResult with probabilities and explanation.
        """
        try:
            features = trade_context.get("features", {})
            if not isinstance(features, dict):
                features = {}

            outcome_actual = trade_context.get("outcome", 0.0)

            # Estimate P(win | actual conditions)
            factual_p = self._estimate_outcome_probability(features)

            # Apply interventions and estimate P(win | do(...))
            counterfactual_features = self._apply_interventions(features, scenario)
            counterfactual_p = self._estimate_outcome_probability(counterfactual_features)

            # Clamp probabilities to [0, 1]
            factual_p = max(0.0, min(1.0, factual_p))
            counterfactual_p = max(0.0, min(1.0, counterfactual_p))

            probability_delta = counterfactual_p - factual_p

            # Detect outcome reversal: if factual was loss but counterfactual was win
            outcome_reversal = (outcome_actual < 0.5 and counterfactual_p > 0.65) or \
                               (outcome_actual >= 0.5 and counterfactual_p < 0.35)

            # Trace causal path
            causal_chain = self._trace_dominant_causal_path(features)

            # Estimate confidence based on causal graph quality
            confidence = self._estimate_confidence(scenario)

            # Generate explanation
            explanation = self._generate_explanation(
                factual_p,
                counterfactual_p,
                scenario,
                causal_chain,
            )

            return CounterfactualResult(
                scenario=scenario,
                factual_outcome_probability=factual_p,
                counterfactual_outcome_probability=counterfactual_p,
                probability_delta=probability_delta,
                outcome_reversal=outcome_reversal,
                causal_chain=causal_chain,
                confidence=confidence,
                explanation=explanation,
            )

        except Exception as e:
            logger.warning(f"CounterfactualEngine.analyze_trade failed: {e}")
            # Return a neutral result on error
            return CounterfactualResult(
                scenario=scenario,
                factual_outcome_probability=0.5,
                counterfactual_outcome_probability=0.5,
                probability_delta=0.0,
                outcome_reversal=False,
                causal_chain=[],
                confidence=0.0,
                explanation=f"Counterfactual analysis unavailable: {e}",
            )

    def batch_analyze(
        self,
        trade_context: Dict[str, Any],
        scenarios: Optional[List[CounterfactualScenario]] = None,
    ) -> List[CounterfactualResult]:
        """Analyze a trade under multiple scenarios.

        Args:
            trade_context: Dict with keys: pair, features, regime, outcome
            scenarios: List of scenarios to test. If None, generates standard 5.

        Returns:
            List of CounterfactualResult, one per scenario.
        """
        if scenarios is None:
            scenarios = self.generate_standard_scenarios(trade_context)

        results = []
        for scenario in scenarios:
            result = self.analyze_trade(trade_context, scenario)
            results.append(result)

        return results

    def generate_standard_scenarios(
        self, trade_context: Dict[str, Any]
    ) -> List[CounterfactualScenario]:
        """Generate 5 standard counterfactual scenarios for any trade:
        1. "low_volatility": What if volatility had been at 30th percentile?
        2. "high_volatility": What if volatility had been at 70th percentile?
        3. "strong_trend": What if trend signal had been strongly aligned?
        4. "weak_trend": What if trend had been neutral/conflicting?
        5. "adverse_session": What if this had occurred in a less favorable session?
        """
        features = trade_context.get("features", {})
        regime = trade_context.get("regime", "NORMAL")

        scenarios = []

        # Scenario 1: Low volatility
        atr_val = features.get("atr_14", 0.001)
        low_volatility_scenario = CounterfactualScenario(
            scenario_name="low_volatility",
            regime=regime,
            interventions=[
                Intervention(
                    variable="atr_14",
                    forced_value=atr_val * 0.5,
                    original_value=atr_val,
                    description=f"If volatility had been LOW (ATR={atr_val * 0.5:.4f} vs actual {atr_val:.4f})",
                )
            ],
        )
        scenarios.append(low_volatility_scenario)

        # Scenario 2: High volatility
        high_volatility_scenario = CounterfactualScenario(
            scenario_name="high_volatility",
            regime=regime,
            interventions=[
                Intervention(
                    variable="atr_14",
                    forced_value=atr_val * 1.5,
                    original_value=atr_val,
                    description=f"If volatility had been HIGH (ATR={atr_val * 1.5:.4f} vs actual {atr_val:.4f})",
                )
            ],
        )
        scenarios.append(high_volatility_scenario)

        # Scenario 3: Strong trend
        trend_val = features.get("sma_trend", 0.0)
        strong_trend_scenario = CounterfactualScenario(
            scenario_name="strong_trend",
            regime=regime,
            interventions=[
                Intervention(
                    variable="sma_trend",
                    forced_value=max(0.8, trend_val),
                    original_value=trend_val,
                    description=f"If trend had been STRONG (0.80 vs actual {trend_val:.2f})",
                )
            ],
        )
        scenarios.append(strong_trend_scenario)

        # Scenario 4: Weak trend
        weak_trend_scenario = CounterfactualScenario(
            scenario_name="weak_trend",
            regime=regime,
            interventions=[
                Intervention(
                    variable="sma_trend",
                    forced_value=0.1,
                    original_value=trend_val,
                    description=f"If trend had been WEAK (0.10 vs actual {trend_val:.2f})",
                )
            ],
        )
        scenarios.append(weak_trend_scenario)

        # Scenario 5: Adverse session (reduce all positive signals)
        momentum_val = features.get("momentum_signal", 0.5)
        adverse_session_scenario = CounterfactualScenario(
            scenario_name="adverse_session",
            regime=regime,
            interventions=[
                Intervention(
                    variable="momentum_signal",
                    forced_value=max(0.0, momentum_val * 0.5),
                    original_value=momentum_val,
                    description=f"If session had been ADVERSE (momentum={momentum_val * 0.5:.2f} vs actual {momentum_val:.2f})",
                )
            ],
        )
        scenarios.append(adverse_session_scenario)

        return scenarios

    def _load_causal_graph(self) -> None:
        """Load the Granger causal graph from disk."""
        if not self.causal_graph_path.exists():
            logger.debug(f"Causal graph not found at {self.causal_graph_path}")
            self._causal_graph = None
            return

        try:
            with open(self.causal_graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("Causal graph JSON is not a dict")
                self._causal_graph = None
                return

            graphs = data.get("graphs", {})
            if graphs:
                # Load the first available graph (usually NORMAL regime)
                regime_key = list(graphs.keys())[0]
                graph_data = graphs[regime_key]
                # Store the causal edges
                self._causal_graph = graph_data.get("edges", [])
                logger.debug(
                    f"Loaded {len(self._causal_graph)} causal edges from {regime_key} regime"
                )
            else:
                logger.debug("No graphs in causal_graph.json")
                self._causal_graph = None

        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Failed to load causal graph: {e}")
            self._causal_graph = None

    def _apply_interventions(
        self,
        features: Dict[str, float],
        scenario: CounterfactualScenario,
    ) -> Dict[str, float]:
        """Apply do() interventions: set variables to their counterfactual values.

        Args:
            features: Original feature dict
            scenario: Scenario with interventions

        Returns:
            Modified feature dict with interventions applied.
        """
        counterfactual_features = features.copy()
        for intervention in scenario.interventions:
            counterfactual_features[intervention.variable] = intervention.forced_value

        return counterfactual_features

    def _estimate_outcome_probability(
        self,
        features: Dict[str, float],
    ) -> float:
        """Estimate P(positive_return) from feature values via causal graph weights.

        Uses a linear propagation model:
        score = sum(feature_value * direction_coefficient) for causal edges
        P(win) = sigmoid(score / normalization)
        """
        if not self._causal_graph or not features:
            # No causal info available — return neutral probability
            return 0.5

        score = 0.0
        causal_count = 0

        for edge in self._causal_graph:
            if not isinstance(edge, dict):
                continue

            feature_name = edge.get("feature_name", "")
            direction_coeff = edge.get("direction_coefficient", 0.0)
            is_causal = edge.get("is_causal", False)

            if not is_causal or feature_name not in features:
                continue

            feature_value = features[feature_name]

            # Linear contribution: feature * causal_direction
            contribution = feature_value * direction_coeff
            score += contribution
            causal_count += 1

        if causal_count == 0:
            return 0.5

        # Normalize score
        normalized_score = score / max(1.0, causal_count)

        # Apply sigmoid to convert to probability [0, 1]
        # sigmoid(x) = 1 / (1 + exp(-x))
        try:
            probability = 1.0 / (1.0 + np.exp(-normalized_score))
        except (OverflowError, ValueError):
            # Clamp extreme values
            probability = 1.0 if normalized_score > 0 else 0.0

        return float(probability)

    def _trace_dominant_causal_path(
        self,
        features: Dict[str, float],
    ) -> List[CausalPathNode]:
        """Trace the top-3 causal features by contribution strength.

        Returns:
            List of CausalPathNode sorted by strength (descending).
        """
        if not self._causal_graph or not features:
            return []

        contributions = []

        for edge in self._causal_graph:
            if not isinstance(edge, dict):
                continue

            feature_name = edge.get("feature_name", "")
            direction_coeff = edge.get("direction_coefficient", 0.0)
            is_causal = edge.get("is_causal", False)
            lag = edge.get("optimal_lag", 0)

            if not is_causal or feature_name not in features:
                continue

            feature_value = features[feature_name]

            # Contribution = |feature_value * direction_coefficient|
            contribution_strength = abs(feature_value * direction_coeff)

            contributions.append((feature_name, feature_value, direction_coeff, lag, contribution_strength))

        # Sort by contribution strength (descending)
        contributions.sort(key=lambda x: x[4], reverse=True)

        # Take top 3
        nodes = []
        for feature_name, feature_value, direction_coeff, lag, _ in contributions[:3]:
            node = CausalPathNode(
                feature=feature_name,
                value=feature_value,
                causal_strength=direction_coeff,
                lag=lag,
            )
            nodes.append(node)

        return nodes

    def _estimate_confidence(self, scenario: CounterfactualScenario) -> float:
        """Estimate confidence in the counterfactual result.

        Factors:
        - Number of causal edges available
        - Number of interventions in the scenario
        """
        if not self._causal_graph:
            return 0.1  # Very low confidence without causal graph

        causal_edge_count = sum(
            1 for edge in self._causal_graph
            if isinstance(edge, dict) and edge.get("is_causal", False)
        )

        # Confidence scales with causal edge count
        # 0-5 edges = 0.3, 5-10 = 0.5, 10+ = 0.7
        if causal_edge_count >= 10:
            base_confidence = 0.7
        elif causal_edge_count >= 5:
            base_confidence = 0.5
        elif causal_edge_count > 0:
            base_confidence = 0.3
        else:
            return 0.1

        # Adjustment based on intervention count
        intervention_count = len(scenario.interventions)
        if intervention_count == 0:
            return 0.1
        elif intervention_count == 1:
            confidence = base_confidence * 0.9
        else:
            confidence = base_confidence * 0.7

        return max(0.1, min(0.95, confidence))

    def _generate_explanation(
        self,
        factual_p: float,
        counterfactual_p: float,
        scenario: CounterfactualScenario,
        causal_chain: List[CausalPathNode],
    ) -> str:
        """Generate a plain-English explanation of the counterfactual result."""
        try:
            # Start with scenario description
            scenario_desc = scenario.scenario_name.replace("_", " ").title()
            interventions_desc = ", ".join(
                i.description for i in scenario.interventions
            )

            # Impact description
            probability_delta = counterfactual_p - factual_p
            if probability_delta > 0.15:
                impact = f"would have INCREASED your win probability by {probability_delta:.1%}"
            elif probability_delta < -0.15:
                impact = f"would have DECREASED your win probability by {abs(probability_delta):.1%}"
            else:
                impact = f"would have had minimal impact on your win probability (delta: {probability_delta:+.1%})"

            # Causal chain description
            if causal_chain:
                top_feature = causal_chain[0].feature
                chain_desc = f"The primary driver was {top_feature}."
            else:
                chain_desc = "Causal drivers were unclear."

            explanation = (
                f"[{scenario_desc}] {interventions_desc} — this {impact}. "
                f"(Factual: {factual_p:.1%} win probability, "
                f"Counterfactual: {counterfactual_p:.1%}.) {chain_desc}"
            )

            return explanation

        except Exception as e:
            logger.debug(f"Failed to generate explanation: {e}")
            return f"Counterfactual: {factual_p:.1%} → {counterfactual_p:.1%}"

    def save_analysis(
        self,
        trade_id: str,
        results: List[CounterfactualResult],
    ) -> None:
        """Persist counterfactual analysis to trained_data/counterfactual_log.jsonl."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            for result in results:
                entry = {
                    "trade_id": trade_id,
                    "analysis": result.to_dict(),
                }
                line = json.dumps(entry, indent=None)
                with open(self.persistence_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

            logger.debug(f"Saved {len(results)} counterfactual analyses for trade {trade_id}")

        except Exception as e:
            logger.warning(f"Failed to save counterfactual analysis: {e}")
