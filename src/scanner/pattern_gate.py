"""
Pattern Gate — Phase 51 (US-319).

Wires journal-mined patterns into active trading gates.
Loads significant patterns from journal_patterns.json and applies:
- Pair preference: boost/penalize confidence based on historical pair win rate
- Direction bias: LONG entries get confidence nudge (49.5% vs 27.5% SHORT baseline)
- Duration gate: flag trades with expected duration outside sweet spot

Usage:
    gate = PatternGate()
    result = gate.evaluate("GBP_AUD", direction="LONG", confidence=0.65)
    # result.adjusted_confidence, result.warnings, result.pair_boost
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternGateConfig:
    """Configuration for pattern gate."""
    patterns_path: str = "trained_data/models/journal_patterns.json"
    # Pair boosts/penalties based on statistical significance
    pair_boost_max: float = 0.10       # Max +10% confidence for strong pairs
    pair_penalty_max: float = -0.15    # Max -15% confidence for weak pairs
    # Direction bias (LONG observed 49.5% win rate vs baseline 38%)
    long_bias_boost: float = 0.05      # +5% for LONG direction
    short_bias_penalty: float = -0.03  # -3% for SHORT direction
    # Duration sweet spot (8-24h had 52% win rate)
    duration_sweet_min_hours: float = 8.0
    duration_sweet_max_hours: float = 24.0
    # Baseline win rate for comparison
    baseline_win_rate: float = 0.384   # From 522-trade journal
    # Significance threshold (only apply patterns with p < 0.05)
    p_threshold: float = 0.05
    # Skip pair fitness threshold
    skip_fitness_threshold: float = 0.30


@dataclass
class PatternGateResult:
    """Result of pattern gate evaluation."""
    adjusted_confidence: float = 0.0
    confidence_delta: float = 0.0
    pair_boost: float = 0.0
    direction_boost: float = 0.0
    duration_warning: str = ""
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True


class PatternGate:
    """Applies mined journal patterns as active trading gates."""

    def __init__(self, config: Optional[PatternGateConfig] = None):
        self.config = config or PatternGateConfig()
        self._patterns: Dict[str, Any] = {}
        self._pair_win_rates: Dict[str, float] = {}
        self._direction_win_rates: Dict[str, float] = {}
        self._load_patterns()

    def evaluate(
        self,
        pair: str,
        direction: str,
        confidence: float,
        expected_duration_hours: float = 0.0,
    ) -> PatternGateResult:
        """Evaluate a trade setup against mined patterns.

        Args:
            pair: Currency pair (e.g., "GBP_AUD")
            direction: "LONG" or "SHORT"
            confidence: Raw confidence score 0.0-1.0
            expected_duration_hours: Expected holding time (0 = unknown)

        Returns:
            PatternGateResult with adjusted confidence and warnings.
        """
        result = PatternGateResult(adjusted_confidence=confidence)
        total_delta = 0.0
        warnings = []

        # 1. Pair preference
        pair_boost = self._compute_pair_boost(pair)
        result.pair_boost = round(pair_boost, 4)
        total_delta += pair_boost
        if pair_boost < -0.05:
            warnings.append(f"Weak pair: {pair} (historical penalty {pair_boost:+.2%})")

        # 2. Direction bias
        direction_boost = self._compute_direction_boost(direction.upper())
        result.direction_boost = round(direction_boost, 4)
        total_delta += direction_boost

        # 3. Duration warning
        if expected_duration_hours > 0:
            dur_warning = self._check_duration(expected_duration_hours)
            if dur_warning:
                result.duration_warning = dur_warning
                warnings.append(dur_warning)

        # Apply adjustments
        result.confidence_delta = round(total_delta, 4)
        result.adjusted_confidence = max(0.0, min(1.0, confidence + total_delta))
        result.warnings = warnings
        result.passed = True  # Pattern gate doesn't block — only adjusts confidence

        result.details = {
            "pair": pair,
            "direction": direction,
            "raw_confidence": confidence,
            "pair_win_rate": self._pair_win_rates.get(pair, 0.0),
            "direction_win_rate": self._direction_win_rates.get(direction.upper(), 0.0),
            "baseline_win_rate": self.config.baseline_win_rate,
        }

        if total_delta != 0:
            logger.info(
                "Phase 51 US-319: %s %s pattern gate: confidence %.3f → %.3f (delta %+.3f, pair=%+.3f, dir=%+.3f)",
                pair, direction, confidence, result.adjusted_confidence,
                total_delta, pair_boost, direction_boost,
            )

        return result

    def _compute_pair_boost(self, pair: str) -> float:
        """Compute confidence adjustment based on pair historical performance."""
        pair_wr = self._pair_win_rates.get(pair, 0.0)
        if pair_wr <= 0:
            return 0.0  # No data — no adjustment

        baseline = self.config.baseline_win_rate
        if baseline <= 0:
            return 0.0

        # Relative performance vs baseline
        relative = (pair_wr - baseline) / baseline

        if relative > 0:
            # Positive performer — boost proportionally, cap at max
            boost = min(relative * 0.15, self.config.pair_boost_max)
        else:
            # Underperformer — penalize, cap at max penalty
            boost = max(relative * 0.20, self.config.pair_penalty_max)

        return boost

    def _compute_direction_boost(self, direction: str) -> float:
        """Compute confidence adjustment based on direction bias."""
        dir_wr = self._direction_win_rates.get(direction, 0.0)
        if dir_wr <= 0:
            return 0.0

        if direction == "LONG":
            return self.config.long_bias_boost
        elif direction == "SHORT":
            return self.config.short_bias_penalty
        return 0.0

    def _check_duration(self, hours: float) -> str:
        """Check if expected duration is in the sweet spot."""
        min_h = self.config.duration_sweet_min_hours
        max_h = self.config.duration_sweet_max_hours

        if hours < min_h:
            return f"Duration {hours:.1f}h below sweet spot ({min_h}-{max_h}h) — scalp risk"
        elif hours > max_h:
            return f"Duration {hours:.1f}h above sweet spot ({min_h}-{max_h}h) — overhold risk"
        return ""

    def _load_patterns(self) -> None:
        """Load patterns from journal_patterns.json."""
        try:
            with open(self.config.patterns_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug("Phase 51: No patterns file at %s: %s", self.config.patterns_path, e)
            return

        if not isinstance(data, dict):
            return

        self._patterns = data

        # Extract pair win rates from breakdowns
        pair_bd = data.get("breakdowns", {}).get("by_pair", {})
        for pair_name, pdata in pair_bd.items():
            if isinstance(pdata, dict):
                wr = pdata.get("win_rate", 0.0)
                p_val = pdata.get("p_value", 1.0)
                if isinstance(wr, (int, float)) and isinstance(p_val, (int, float)):
                    self._pair_win_rates[pair_name] = float(wr)

        # Extract direction win rates
        dir_bd = data.get("breakdowns", {}).get("by_direction", {})
        for dir_name, ddata in dir_bd.items():
            if isinstance(ddata, dict):
                wr = ddata.get("win_rate", 0.0)
                if isinstance(wr, (int, float)):
                    self._direction_win_rates[dir_name.upper()] = float(wr)

        # Also check significant_patterns list for stronger signals
        sig_patterns = data.get("significant_patterns", [])
        for sp in sig_patterns:
            if not isinstance(sp, dict):
                continue
            category = sp.get("category", "")
            if category == "pair":
                pair_name = sp.get("value", "")
                wr = sp.get("win_rate", 0.0)
                p_val = sp.get("p_value", 1.0)
                if p_val < self.config.p_threshold:
                    self._pair_win_rates[pair_name] = float(wr)

        logger.info(
            "Phase 51: Loaded patterns — %d pairs, %d directions",
            len(self._pair_win_rates), len(self._direction_win_rates),
        )

    def get_pair_ranking(self) -> List[Dict[str, Any]]:
        """Return pairs ranked by historical win rate."""
        ranking = []
        for pair, wr in sorted(self._pair_win_rates.items(), key=lambda x: -x[1]):
            ranking.append({
                "pair": pair,
                "win_rate": round(wr, 4),
                "vs_baseline": round(wr - self.config.baseline_win_rate, 4),
                "boost": round(self._compute_pair_boost(pair), 4),
            })
        return ranking
