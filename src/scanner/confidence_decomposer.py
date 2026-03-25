"""
Confidence Decomposer — Phase 52 (US-324).

Decomposes raw confidence into 3 components:
1. directional_strength — agent agreement on direction × trend alignment
2. timing_quality — session alignment × MTF confluence × duration sweet spot
3. magnitude_estimate — ATR percentile × regime-conditional expected move

Key insight: confidence 0.71-0.80 has only 16.1% win rate — the composite score
masks directional disagreement. Decomposing catches these failure modes.

High-confidence penalty: if recomposed > 0.70 and directional_strength < 0.55, cap to 0.65.

Usage:
    decomposer = ConfidenceDecomposer()
    result = decomposer.decompose(
        raw_confidence=0.65,
        agent_verdicts={"trend": True, "mean_reversion": False, ...},
        uncertainty_score=0.25,
        model_disagreement=0.15,
        session_name="LONDON",
        mtf_signal="PROCEED",
        regime="NORMAL",
        atr_percentile=0.60,
    )
    # result.recomposed_confidence respects directional safety cap
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecomposerConfig:
    """Configuration for confidence decomposition."""
    # Component weights (must sum to 1.0)
    weight_directional: float = 0.45
    weight_timing: float = 0.30
    weight_magnitude: float = 0.25

    # Directional strength parameters
    min_agreement_ratio: float = 0.55  # Below this → directional weakness detected

    # High-confidence penalty
    high_conf_threshold: float = 0.70   # Confidence above this triggers safety check
    high_conf_cap: float = 0.65         # Cap applied when directional_strength < min_agreement_ratio

    # Session quality scores
    session_scores: Dict[str, float] = field(default_factory=lambda: {
        "LONDON": 1.0,
        "OVERLAP": 0.90,
        "NEW_YORK": 0.85,
        "TOKYO": 0.70,
        "OFF_HOURS": 0.50,
    })

    # MTF confluence signal scores
    mtf_scores: Dict[str, float] = field(default_factory=lambda: {
        "PROCEED": 1.0,
        "CAUTION": 0.60,
        "REJECT": 0.20,
    })

    # Regime magnitude multipliers
    regime_magnitude: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 0.70,
        "NORMAL": 1.00,
        "HIGH": 0.85,
        "EXTREME": 0.60,
    })


@dataclass
class DecompositionResult:
    """Result of confidence decomposition."""
    raw_confidence: float = 0.0
    directional_strength: float = 0.0
    timing_quality: float = 0.0
    magnitude_estimate: float = 0.0
    recomposed_confidence: float = 0.0
    high_conf_penalty_applied: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class ConfidenceDecomposer:
    """Decomposes raw confidence into directional/timing/magnitude components.

    Addresses the inverse confidence-win rate problem at high levels by
    detecting cases where directional agreement is weak despite high raw score.
    """

    def __init__(self, config: Optional[DecomposerConfig] = None):
        self.config = config or DecomposerConfig()

    def decompose(
        self,
        raw_confidence: float,
        agent_verdicts: Optional[Dict[str, bool]] = None,
        uncertainty_score: float = 0.0,
        model_disagreement: float = 0.0,
        session_name: str = "LONDON",
        mtf_signal: str = "PROCEED",
        regime: str = "NORMAL",
        atr_percentile: float = 0.50,
        duration_hours: float = 12.0,
    ) -> DecompositionResult:
        """Decompose raw confidence into 3 components.

        Args:
            raw_confidence: Raw confidence 0.0-1.0
            agent_verdicts: Dict of {agent_name: passed_bool} for direction agreement
            uncertainty_score: Uncertainty 0.0-1.0 (higher = worse)
            model_disagreement: Model disagreement 0.0-1.0 (higher = worse)
            session_name: Trading session (LONDON, NEW_YORK, TOKYO, OVERLAP, OFF_HOURS)
            mtf_signal: Multi-timeframe signal (PROCEED, CAUTION, REJECT)
            regime: Volatility regime (LOW, NORMAL, HIGH, EXTREME)
            atr_percentile: ATR percentile 0.0-1.0 (higher = more volatile)
            duration_hours: Expected trade duration in hours

        Returns:
            DecompositionResult with recomposed confidence.
        """
        c = self.config

        # 1. Directional Strength: agent agreement × (1 - disagreement)
        directional = self._compute_directional_strength(
            agent_verdicts or {},
            uncertainty_score,
            model_disagreement,
        )

        # 2. Timing Quality: session × MTF × duration sweet spot
        timing = self._compute_timing_quality(
            session_name, mtf_signal, duration_hours,
        )

        # 3. Magnitude Estimate: ATR percentile × regime multiplier
        magnitude = self._compute_magnitude_estimate(
            atr_percentile, regime,
        )

        # Recompose: weighted average
        recomposed = (
            c.weight_directional * directional
            + c.weight_timing * timing
            + c.weight_magnitude * magnitude
        )
        recomposed = max(0.0, min(1.0, recomposed))

        # High-confidence penalty: cap if directional strength is weak
        penalty_applied = False
        if recomposed > c.high_conf_threshold and directional < c.min_agreement_ratio:
            recomposed = min(recomposed, c.high_conf_cap)
            penalty_applied = True
            logger.info(
                "Phase 52 (US-324): High-confidence penalty — recomposed %.3f capped to %.3f "
                "(directional_strength=%.3f < %.3f)",
                recomposed, c.high_conf_cap, directional, c.min_agreement_ratio,
            )

        result = DecompositionResult(
            raw_confidence=raw_confidence,
            directional_strength=round(directional, 4),
            timing_quality=round(timing, 4),
            magnitude_estimate=round(magnitude, 4),
            recomposed_confidence=round(recomposed, 4),
            high_conf_penalty_applied=penalty_applied,
            details={
                "weights": {
                    "directional": c.weight_directional,
                    "timing": c.weight_timing,
                    "magnitude": c.weight_magnitude,
                },
                "session": session_name,
                "mtf_signal": mtf_signal,
                "regime": regime,
                "atr_percentile": atr_percentile,
            },
        )

        logger.debug(
            "Phase 52 (US-324): Decomposition — dir=%.3f tim=%.3f mag=%.3f → recomposed=%.3f%s",
            directional, timing, magnitude, recomposed,
            " (PENALTY)" if penalty_applied else "",
        )

        return result

    def _compute_directional_strength(
        self,
        agent_verdicts: Dict[str, bool],
        uncertainty: float,
        disagreement: float,
    ) -> float:
        """Compute directional strength from agent agreement.

        High agreement + low uncertainty + low disagreement = high strength.
        """
        if not agent_verdicts:
            return 0.5  # Neutral — no data

        n_total = len(agent_verdicts)
        n_passed = sum(1 for v in agent_verdicts.values() if v)
        agreement_ratio = n_passed / n_total if n_total > 0 else 0.5

        # Weight by uncertainty and disagreement
        certainty_factor = 1.0 - min(1.0, uncertainty)
        agreement_factor = 1.0 - min(1.0, disagreement)

        # Composite: agreement * certainty * agreement_factor
        directional = agreement_ratio * (0.5 + 0.25 * certainty_factor + 0.25 * agreement_factor)

        return max(0.0, min(1.0, directional))

    def _compute_timing_quality(
        self,
        session: str,
        mtf_signal: str,
        duration_hours: float,
    ) -> float:
        """Compute timing quality from session, MTF, and duration."""
        session_score = self.config.session_scores.get(session.upper(), 0.5)
        mtf_score = self.config.mtf_scores.get(mtf_signal.upper(), 0.5)

        # Duration sweet spot scoring (8-24h is optimal per Phase 50 research)
        if 8.0 <= duration_hours <= 24.0:
            duration_score = 1.0
        elif 4.0 <= duration_hours < 8.0 or 24.0 < duration_hours <= 48.0:
            duration_score = 0.70
        else:
            duration_score = 0.40

        # Weighted timing: session(0.40) + MTF(0.40) + duration(0.20)
        timing = 0.40 * session_score + 0.40 * mtf_score + 0.20 * duration_score

        return max(0.0, min(1.0, timing))

    def _compute_magnitude_estimate(
        self,
        atr_percentile: float,
        regime: str,
    ) -> float:
        """Compute magnitude estimate from ATR and regime.

        Sweet spot: moderate ATR (30th-70th percentile) in NORMAL regime.
        Extremes (very low or very high ATR) reduce magnitude score.
        """
        regime_mult = self.config.regime_magnitude.get(regime.upper(), 1.0)

        # ATR scoring: peak at 0.50, drops off at extremes
        if 0.30 <= atr_percentile <= 0.70:
            atr_score = 1.0
        elif 0.15 <= atr_percentile < 0.30 or 0.70 < atr_percentile <= 0.85:
            atr_score = 0.75
        else:
            atr_score = 0.50

        magnitude = atr_score * regime_mult

        return max(0.0, min(1.0, magnitude))
