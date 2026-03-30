"""Devil's Advocate Agent — Adversarial bear-case evaluator.

The Devil's Advocate agent runs LAST in the agent team evaluation chain.
Its role is to construct the strongest possible bear case against every trade signal
and block trades when systematic risks are too high.

It evaluates 8 specific risk factors:
1. support_resistance_conflict: Trading into known S/R zones
2. news_risk_amplified: High news risk + event timing < 2 hours
3. spread_at_limit: Spread > 1.5x median
4. regime_mismatch: Direction conflicts with volatility regime
5. ensemble_conflict: Model disagreement > 0.25
6. uncertainty_elevated: Uncertainty score > 0.40
7. episodic_pattern_warning: Episodic suppression flag active
8. rr_barely_passes: R:R ratio < 1.35 (barely above 1.2 minimum)

Each factor is weighted and scored 0.0–1.0 to produce a bear_score.
If bear_score >= 0.60: blocks trade (block_trade=True).
If bear_score >= 0.40: soft veto (passed=False, confidence penalty).
If bear_score < 0.40: passes (passed=True).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.scanner.agents._team import (
    AgentDecisionContext,
    AgentVerdict,
    _clip01,
    _safe_float,
    _last_value,
)

logger = logging.getLogger(__name__)


class DevilsAdvocateAgent:
    """Adversarial bear-case agent that runs last before execution."""

    # Factor weights (sum ~3.0 max possible bear_score)
    _WEIGHTS: Dict[str, float] = {
        "support_resistance_conflict": 0.65,
        "news_risk_amplified": 0.55,
        "spread_at_limit": 0.40,
        "regime_mismatch": 0.50,
        "ensemble_conflict": 0.45,
        "uncertainty_elevated": 0.35,
        "episodic_pattern_warning": 0.60,
        "rr_barely_passes": 0.30,
    }

    def __init__(self) -> None:
        """Initialize the Devil's Advocate agent."""
        self.total_weight = sum(self._WEIGHTS.values())

    def evaluate(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Construct and score the bear case against a trade signal.

        Args:
            ctx: Agent decision context (analysis, dataframes, gates, config).

        Returns:
            AgentVerdict with name="devil_advocate", score inverted from bear_score,
            block_trade and passed set based on bear_score thresholds.
        """
        analysis = ctx.analysis
        gate_details = ctx.gate_details or {}

        # Evaluate all 8 bear factors
        sr_conflict = self._check_sr_conflict(analysis, gate_details)
        news_amplified = self._check_news_amplified(analysis, gate_details)
        spread_limit = self._check_spread_limit(analysis)
        regime_mismatch = self._check_regime_mismatch(analysis)
        ensemble_conflict = self._check_ensemble_conflict(analysis, gate_details)
        uncertainty_high = self._check_uncertainty_elevated(analysis)
        episodic_warn = self._check_episodic_warning(gate_details)
        rr_marginal = self._check_rr_barely_passes(analysis)

        # Log each factor evaluation at DEBUG level
        logger.debug(
            "DA factors for %s: sr_conflict=%.2f, news_amp=%.2f, spread_lim=%.2f, "
            "regime_mismatch=%.2f, ens_conflict=%.2f, uncertainty=%.2f, episodic=%.2f, rr_marginal=%.2f",
            getattr(analysis, "pair", "?"),
            sr_conflict,
            news_amplified,
            spread_limit,
            regime_mismatch,
            ensemble_conflict,
            uncertainty_high,
            episodic_warn,
            rr_marginal,
        )

        # Compute normalized bear_score
        # bear_score = sum of triggered factors / total_weight
        # Only triggered factors (>0) count toward the score
        bear_components = {
            "support_resistance_conflict": sr_conflict * self._WEIGHTS["support_resistance_conflict"],
            "news_risk_amplified": news_amplified * self._WEIGHTS["news_risk_amplified"],
            "spread_at_limit": spread_limit * self._WEIGHTS["spread_at_limit"],
            "regime_mismatch": regime_mismatch * self._WEIGHTS["regime_mismatch"],
            "ensemble_conflict": ensemble_conflict * self._WEIGHTS["ensemble_conflict"],
            "uncertainty_elevated": uncertainty_high * self._WEIGHTS["uncertainty_elevated"],
            "episodic_pattern_warning": episodic_warn * self._WEIGHTS["episodic_pattern_warning"],
            "rr_barely_passes": rr_marginal * self._WEIGHTS["rr_barely_passes"],
        }

        total_bear = sum(bear_components.values())
        bear_score = _clip01(total_bear / self.total_weight) if self.total_weight > 0 else 0.0

        # Determine pass/block logic
        block_trade = bear_score >= 0.60
        passed = bear_score < 0.40
        soft_veto = 0.40 <= bear_score < 0.60

        # confidence_delta: penalize based on bear_score
        # Scale from -0.40 (full block) to 0.0 (passing)
        confidence_delta = -0.40 * bear_score

        # Agent score inverted from bear_score (agent is how bullish, bear is how bearish)
        agent_score = 1.0 - bear_score

        # Reason and reason_code
        triggered_factors = [k for k, v in bear_components.items() if v > 0.01]
        if block_trade:
            reason_code = "DA_BLOCK"
            reason = f"Bear case blocks trade: {', '.join(triggered_factors[:3])}"
        elif soft_veto:
            reason_code = "DA_WARN"
            reason = f"Bear case caution: {', '.join(triggered_factors[:2])}"
        else:
            reason_code = "DA_PASS"
            reason = "Bear case weak; trade acceptable"

        logger.debug(
            "DA verdict for %s: bear_score=%.3f, agent_score=%.3f, passed=%s, block=%s, reason_code=%s",
            getattr(analysis, "pair", "?"),
            bear_score,
            agent_score,
            passed,
            block_trade,
            reason_code,
        )

        if block_trade:
            logger.warning(
                "DA_BLOCK for %s: bear_score=%.3f (triggered: %s)",
                getattr(analysis, "pair", "?"),
                bear_score,
                ", ".join(triggered_factors),
            )

        return AgentVerdict(
            name="devil_advocate",
            score=_clip01(agent_score),
            passed=passed,
            weight=1.30,  # Highest weight — runs last, is final check
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "bear_score": _clip01(bear_score),
                "soft_veto": soft_veto,
                "triggered_factors": triggered_factors,
                **bear_components,
            },
        )

    # --- Factor evaluators ---

    def _check_sr_conflict(
        self, analysis: Any, gate_details: Dict[str, Any]
    ) -> float:
        """Check if signal trades into a known support/resistance zone.

        Returns:
            0.0 if no conflict, 1.0 if trades into S/R zone, 0.5 if uncertain.
        """
        try:
            # Look for S/R data in gate_details or analysis attributes
            sr_data = gate_details.get("support_resistance_data")
            if sr_data is None:
                sr_data = getattr(analysis, "support_resistance_data", None)

            if sr_data is None:
                return 0.0  # No S/R data available, no factor triggered

            # Check if entry price is within S/R zone (within 10 pips)
            entry_price = _safe_float(getattr(analysis, "current_price", 0.0))
            direction = getattr(analysis, "direction", "HOLD")

            # sr_data expected format: {"resistance": [R1, R2, ...], "support": [S1, S2, ...]}
            resistance_levels = sr_data.get("resistance", [])
            support_levels = sr_data.get("support", [])

            sr_margin_pips = 10.0

            if direction == "LONG" and resistance_levels:
                # LONG signal near resistance is risky
                nearest_resistance = min(resistance_levels, key=lambda x: abs(float(x) - entry_price))
                if abs(float(nearest_resistance) - entry_price) < sr_margin_pips:
                    logger.debug(
                        "DA: LONG signal within %.1f pips of resistance at %.5f (entry %.5f)",
                        abs(float(nearest_resistance) - entry_price),
                        nearest_resistance,
                        entry_price,
                    )
                    return 1.0

            elif direction == "SHORT" and support_levels:
                # SHORT signal near support is risky
                nearest_support = min(support_levels, key=lambda x: abs(float(x) - entry_price))
                if abs(float(nearest_support) - entry_price) < sr_margin_pips:
                    logger.debug(
                        "DA: SHORT signal within %.1f pips of support at %.5f (entry %.5f)",
                        abs(float(nearest_support) - entry_price),
                        nearest_support,
                        entry_price,
                    )
                    return 1.0

            return 0.0
        except Exception as e:
            logger.debug(f"DA: S/R conflict check failed: {e}")
            return 0.0

    def _check_news_amplified(
        self, analysis: Any, gate_details: Dict[str, Any]
    ) -> float:
        """Check if news_risk_score is high AND event is within 2 hours.

        Returns:
            1.0 if both conditions met, 0.0 otherwise.
        """
        try:
            news_risk = _safe_float(getattr(analysis, "news_risk_score", 0.0))
            news_event_countdown = gate_details.get("news_event_countdown_minutes")

            if news_event_countdown is None:
                news_event_countdown = getattr(analysis, "news_event_countdown_minutes", None)

            # Both conditions: news_risk > 0.60 AND event within 120 minutes
            if news_risk > 0.60 and news_event_countdown is not None:
                countdown_minutes = _safe_float(news_event_countdown, 0.0)
                if 0.0 < countdown_minutes < 120.0:
                    logger.debug(
                        "DA: News amplified (risk=%.2f, countdown=%.0f min)",
                        news_risk,
                        countdown_minutes,
                    )
                    return 1.0

            return 0.0
        except Exception as e:
            logger.debug(f"DA: News amplified check failed: {e}")
            return 0.0

    def _check_spread_limit(self, analysis: Any) -> float:
        """Check if current spread is > 1.5x the pair's 20-period median spread.

        Returns:
            1.0 if spread exceeds 1.5x median, 0.0 otherwise.
        """
        try:
            current_spread = _safe_float(getattr(analysis, "spread_pips", 0.0))
            median_spread = getattr(analysis, "spread_median_pips", None)

            if median_spread is None or median_spread <= 0:
                return 0.0

            median_spread = _safe_float(median_spread, 0.0)
            if current_spread > 1.5 * median_spread:
                logger.debug(
                    "DA: Spread at limit (current=%.2f, 1.5x median=%.2f)",
                    current_spread,
                    1.5 * median_spread,
                )
                return 1.0

            return 0.0
        except Exception as e:
            logger.debug(f"DA: Spread limit check failed: {e}")
            return 0.0

    def _check_regime_mismatch(self, analysis: Any) -> float:
        """Check if signal direction conflicts with current volatility regime.

        HIGH volatility + LONG bias = mismatch (prefer LONG in calm conditions)
        EXTREME volatility + any direction = proceed with caution (reduced factor)

        Returns:
            1.0 if major mismatch, 0.5 if moderate, 0.0 if aligned or safe.
        """
        try:
            regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
            direction = getattr(analysis, "direction", "HOLD")

            # Volatility regime preferences (heuristic):
            # LOW: any direction OK but risky due to low vol
            # NORMAL: any direction OK
            # HIGH: prefer SHORT (downside volatility) or cautious LONG
            # EXTREME: all signals risky, but proceed (high vol = high potential)

            if regime == "HIGH":
                # HIGH vol favors SHORT direction
                if direction == "LONG":
                    logger.debug("DA: LONG signal in HIGH volatility regime (mismatch)")
                    return 0.5  # Moderate mismatch

            elif regime == "EXTREME":
                # EXTREME vol is risky for any direction, but we don't block outright
                # Return 0.25 as a warning (minor factor)
                logger.debug("DA: Any direction in EXTREME volatility (caution)")
                return 0.25

            elif regime == "LOW":
                # LOW vol trades are risky overall (low profit potential)
                # But not a direct direction mismatch
                return 0.0

            return 0.0
        except Exception as e:
            logger.debug(f"DA: Regime mismatch check failed: {e}")
            return 0.0

    def _check_ensemble_conflict(
        self, analysis: Any, gate_details: Dict[str, Any]
    ) -> float:
        """Check if model disagreement metric > 0.25.

        Returns:
            Scaled value 0.0–1.0 based on disagreement level.
        """
        try:
            disagreement = _safe_float(getattr(analysis, "model_disagreement", 0.0))

            # Disagreement threshold: 0.25 starts the warning, scales to 1.0 at 0.40
            if disagreement > 0.25:
                # Scale: 0.25 → 0.0, 0.40 → 1.0
                factor = _clip01((disagreement - 0.25) / (0.40 - 0.25))
                logger.debug("DA: Model disagreement high (%.3f, factor=%.2f)", disagreement, factor)
                return factor

            return 0.0
        except Exception as e:
            logger.debug(f"DA: Ensemble conflict check failed: {e}")
            return 0.0

    def _check_uncertainty_elevated(self, analysis: Any) -> float:
        """Check if uncertainty_score > 0.40.

        Returns:
            Scaled value 0.0–1.0 based on uncertainty level.
        """
        try:
            uncertainty = _safe_float(getattr(analysis, "uncertainty_score", 0.0))

            # Uncertainty threshold: 0.40 starts warning, scales to 1.0 at 0.60
            if uncertainty > 0.40:
                # Scale: 0.40 → 0.0, 0.60 → 1.0
                factor = _clip01((uncertainty - 0.40) / (0.60 - 0.40))
                logger.debug("DA: Uncertainty elevated (%.3f, factor=%.2f)", uncertainty, factor)
                return factor

            return 0.0
        except Exception as e:
            logger.debug(f"DA: Uncertainty check failed: {e}")
            return 0.0

    def _check_episodic_warning(self, gate_details: Dict[str, Any]) -> float:
        """Check if episodic_suppression=True in gate_details.

        Returns:
            1.0 if episodic suppression active, 0.0 otherwise.
        """
        try:
            episodic_suppression = gate_details.get("episodic_suppression", False)
            if episodic_suppression is True:
                logger.debug("DA: Episodic suppression pattern detected")
                return 1.0
            return 0.0
        except Exception as e:
            logger.debug(f"DA: Episodic warning check failed: {e}")
            return 0.0

    def _check_rr_barely_passes(self, analysis: Any) -> float:
        """Check if R:R ratio is marginally above the 1.2 minimum (i.e., < 1.35).

        Returns:
            Scaled value 0.0–1.0 based on how close to 1.2 minimum.
        """
        try:
            sl_pips = _safe_float(getattr(analysis, "sl_pips", 1.0))
            tp_pips = _safe_float(getattr(analysis, "tp_pips", 1.0))

            if sl_pips <= 0.0:
                sl_pips = 1.0

            rr_ratio = tp_pips / sl_pips

            # RR threshold: 1.2 (passes hard gate), 1.35 (soft pass, no warning)
            # Below 1.35, we warn
            if rr_ratio < 1.35:
                # Scale: 1.2 → 1.0 (max warning), 1.35 → 0.0 (no warning)
                factor = _clip01((1.35 - rr_ratio) / (1.35 - 1.2))
                logger.debug("DA: R:R barely passes (%.2f, factor=%.2f)", rr_ratio, factor)
                return factor

            return 0.0
        except Exception as e:
            logger.debug(f"DA: R:R check failed: {e}")
            return 0.0
