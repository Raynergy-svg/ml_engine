"""Enhanced Confidence Calibration System for Buddy FX Trading Bot.

Implements a multi-layer confidence calibration pipeline:
1. Ensemble Disagreement - Uncertainty from 12 agent score variance
2. Regime-Aware Platt Scaling - Separate sigmoid calibration per regime
3. Confidence Time-Decay - Regime-dependent decay over holding period
4. Agent Agreement Quality - Coherence of agent consensus
5. Meta-Confidence - Confidence in the calibration itself

Pure Python + numpy + scipy.optimize.
Deterministic, thread-safe, atomic JSON persistence.

Key Dataclasses:
- CalibrationConfig: Configuration for calibration parameters
- AgentVerdict: Simplified agent verdict for calibration (name, score, weight, passed)
- CalibratedConfidence: Complete calibration result with all components

Key Class:
- ConfidenceCalibrationSystem: Main calibration engine with persistence
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import numpy as np

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)


def _clip01(value: float) -> float:
    """Clip value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _verdict_val(verdict: Any, key: str, default: Any = None) -> Any:
    """Extract a field from a verdict that may be a dict or an object."""
    if isinstance(verdict, dict):
        return verdict.get(key, default)
    return getattr(verdict, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling NaN and None."""
    try:
        if value is None:
            return default
        result = float(value)
        # Check for NaN
        if result != result:
            return default
        return result
    except (TypeError, ValueError):
        return default


@dataclass
class CalibrationConfig:
    """Configuration for Enhanced Confidence Calibration System.

    Attributes:
        # Platt scaling
        min_trades_for_calibration: Minimum trades needed before Platt scaling is active
        min_trades_per_regime: Minimum trades per regime for regime-specific calibration
        refit_interval: Refit Platt parameters every N new trades

        # Time decay
        decay_rates: Regime-dependent confidence decay rates per bar held

        # Ensemble disagreement
        high_disagreement_threshold: Agent score std > this = HIGH uncertainty
        critical_disagreement_threshold: Agent score std > this = CRITICAL uncertainty

        # Agent agreement
        min_agents_for_quality: Minimum agents reporting for quality assessment

        # Persistence
        calibration_file: Path to persisted calibration parameters
    """
    # Platt scaling
    min_trades_for_calibration: int = 30
    min_trades_per_regime: int = 15
    refit_interval: int = 20

    # Time decay rates per bar held (regime_name -> decay_rate)
    decay_rates: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 0.99,
        "NORMAL": 0.97,
        "HIGH": 0.95,
        "EXTREME": 0.92,
    })

    # Ensemble disagreement thresholds
    high_disagreement_threshold: float = 0.30
    critical_disagreement_threshold: float = 0.45

    # Agent agreement
    min_agents_for_quality: int = 6

    # Persistence
    calibration_file: str = "trained_data/confidence_calibration.json"


@dataclass
class CalibratedConfidence:
    """Complete confidence calibration result.

    Combines ensemble disagreement, Platt scaling, agent agreement,
    and meta-confidence into a final calibrated confidence score.

    Attributes:
        raw_weighted_score: Original weighted vote score from agents (0-1)
        agent_scores: List of individual agent scores (0-1)

        # Ensemble disagreement
        ensemble_disagreement: Standard deviation of agent scores (0-1)
        disagreement_level: "LOW", "MODERATE", "HIGH", or "CRITICAL"

        # Platt scaling
        platt_calibrated: Sigmoid-calibrated probability (0-1)
        is_calibrated: True if historical data available for calibration
        calibration_regime: Which regime's calibration was used

        # Agent agreement
        agent_agreement: Quality of agent consensus (0-1, 1=perfect agreement)
        agents_reporting: Number of agents that contributed

        # Time decay
        time_decay_factor: Applied separately as position ages (starts at 1.0)

        # Meta-confidence
        meta_confidence: Confidence in the calibration (0-1, based on sample size/recency)

        # Final output
        final_confidence: Calibrated confidence score (0-1, for trading decisions)

        # Logging and RL
        metadata: Additional diagnostic information
    """
    raw_weighted_score: float
    agent_scores: List[float]

    # Ensemble disagreement
    ensemble_disagreement: float
    disagreement_level: str

    # Platt scaling
    platt_calibrated: float
    is_calibrated: bool
    calibration_regime: str

    # Agent agreement
    agent_agreement: float
    agents_reporting: int

    # Time decay
    time_decay_factor: float = 1.0

    # Meta-confidence
    meta_confidence: float = 0.5

    # Final output
    final_confidence: float = 0.0

    # Logging/RL
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceCalibrationSystem:
    """Multi-layer confidence calibration engine.

    Pipeline:
    1. Compute weighted score from agent verdicts
    2. Compute ensemble disagreement (std of agent scores)
    3. Apply regime-aware Platt scaling (sigmoid calibration)
    4. Compute agent agreement quality (coherence)
    5. Compute meta-confidence (sample size + recency)
    6. Combine into final confidence score

    Attributes:
        _config: Calibration configuration
        _platt_params: Regime-specific Platt scaling parameters
        _trade_history: List of (raw_score, outcome, regime) for calibration fitting
        _trades_since_refit: Counter for automatic refitting
    """

    def __init__(self, config: CalibrationConfig = None):
        """Initialize calibration system and load persisted parameters.

        Args:
            config: CalibrationConfig with thresholds and settings.
                   Uses defaults if None.
        """
        self._config = config or CalibrationConfig()
        self._platt_params: Dict[str, Dict[str, float]] = {}  # regime -> {coef, intercept}
        self._trade_history: List[Tuple[float, float, str]] = []  # (raw_score, outcome, regime)
        self._trades_since_refit: int = 0

        # Validate scipy availability for Platt fitting
        if not _HAS_SCIPY:
            logger.warning("scipy not available — Platt scaling disabled. Install scipy to enable.")

        # Load persisted calibration on startup
        self._load_calibration()

    def calibrate(
        self,
        agent_verdicts: List[Any],
        regime_name: str = "NORMAL",
    ) -> CalibratedConfidence:
        """Full calibration pipeline.

        Args:
            agent_verdicts: List of AgentVerdict objects with name, score, weight, passed
            regime_name: Volatility regime ("LOW", "NORMAL", "HIGH", "EXTREME")

        Returns:
            CalibratedConfidence with all components computed

        Raises:
            ValueError: If agent_verdicts is empty or invalid
        """
        if not agent_verdicts:
            logger.error("calibrate() called with empty agent_verdicts")
            raise ValueError("agent_verdicts cannot be empty")

        try:
            # 1. Extract scores and compute weighted score
            agent_scores = [_safe_float(_verdict_val(v, 'score', 0.5), 0.5) for v in agent_verdicts]
            weights = [_safe_float(_verdict_val(v, 'weight', 1.0), 1.0) for v in agent_verdicts]

            weighted_score = self._compute_weighted_score(agent_verdicts)

            # 2. Compute ensemble disagreement
            disagreement, disagreement_level = self._compute_ensemble_disagreement(agent_verdicts)

            # 3. Apply Platt scaling (regime-aware)
            platt_score, is_calibrated = self._platt_scale(weighted_score, regime_name)

            # 4. Compute agent agreement quality
            agreement, n_agents = self._compute_agent_agreement(agent_verdicts)

            # 5. Compute meta-confidence (calibration quality)
            meta_conf = self._compute_meta_confidence(regime_name)

            # 6. Combine into final confidence
            final_conf = self._combine_confidence_components(
                platt_score,
                agreement,
                disagreement_level,
                meta_conf,
            )

            # Build result
            result = CalibratedConfidence(
                raw_weighted_score=_clip01(weighted_score),
                agent_scores=agent_scores,
                ensemble_disagreement=disagreement,
                disagreement_level=disagreement_level,
                platt_calibrated=_clip01(platt_score),
                is_calibrated=is_calibrated,
                calibration_regime=regime_name,
                agent_agreement=_clip01(agreement),
                agents_reporting=n_agents,
                time_decay_factor=1.0,
                meta_confidence=_clip01(meta_conf),
                final_confidence=_clip01(final_conf),
                metadata={
                    "n_agents": len(agent_verdicts),
                    "regime_name": regime_name,
                    "weights_avg": float(np.mean(weights)) if weights else 1.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.debug(
                f"Calibration complete: raw={result.raw_weighted_score:.3f}, "
                f"platt={result.platt_calibrated:.3f}, "
                f"final={result.final_confidence:.3f}, "
                f"disagreement={disagreement_level}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Calibration failed with {len(agent_verdicts)} agents: {e}",
                exc_info=True,
            )
            raise

    def apply_time_decay(
        self,
        calibrated: CalibratedConfidence,
        bars_held: int,
        regime_name: str,
    ) -> CalibratedConfidence:
        """Apply time decay to an existing calibration result.

        Simulates confidence degradation as position ages.
        decay_factor = decay_rate ^ bars_held

        Args:
            calibrated: Existing CalibratedConfidence result
            bars_held: Number of bars (candles) position has been open
            regime_name: Volatility regime for decay rate selection

        Returns:
            New CalibratedConfidence with time_decay_factor and final_confidence updated
        """
        if bars_held <= 0:
            logger.debug(f"apply_time_decay: bars_held={bars_held}, skipping (no decay)")
            return calibrated

        # Get regime-specific decay rate
        decay_rate = self._config.decay_rates.get(regime_name, 0.97)
        decay_rate = _clip01(decay_rate)

        # Apply exponential decay
        decay_factor = float(decay_rate ** bars_held)

        # Recompute final confidence with decay applied
        final_with_decay = calibrated.final_confidence * decay_factor

        # Create new result with decay applied
        result = CalibratedConfidence(
            raw_weighted_score=calibrated.raw_weighted_score,
            agent_scores=calibrated.agent_scores,
            ensemble_disagreement=calibrated.ensemble_disagreement,
            disagreement_level=calibrated.disagreement_level,
            platt_calibrated=calibrated.platt_calibrated,
            is_calibrated=calibrated.is_calibrated,
            calibration_regime=calibrated.calibration_regime,
            agent_agreement=calibrated.agent_agreement,
            agents_reporting=calibrated.agents_reporting,
            time_decay_factor=decay_factor,
            meta_confidence=calibrated.meta_confidence,
            final_confidence=_clip01(final_with_decay),
            metadata=dict(calibrated.metadata),
        )
        result.metadata["time_decay_applied"] = True
        result.metadata["bars_held"] = bars_held
        result.metadata["decay_rate"] = decay_rate
        result.metadata["decay_factor"] = decay_factor

        logger.debug(
            f"Time decay applied: bars_held={bars_held}, decay_factor={decay_factor:.4f}, "
            f"final={calibrated.final_confidence:.3f} -> {result.final_confidence:.3f}"
        )

        return result

    def record_outcome(
        self,
        raw_score: float,
        outcome: float,
        regime_name: str,
    ) -> None:
        """Record a trade outcome for Platt scaling recalibration.

        Call this after every trade closes to build calibration history.
        Automatically triggers refitting every refit_interval trades.

        Args:
            raw_score: Confidence score at trade entry (0-1)
            outcome: Trade result: 1.0=win, 0.0=loss (also accepts float for partial)
            regime_name: Volatility regime when trade was entered

        Raises:
            ValueError: If outcome not in [0, 1]
        """
        raw_score = _safe_float(raw_score, 0.5)
        outcome = _safe_float(outcome, 0.0)

        if not (0.0 <= outcome <= 1.0):
            logger.error(f"record_outcome: outcome {outcome} not in [0, 1], clamping")
            outcome = _clip01(outcome)

        self._trade_history.append((raw_score, outcome, regime_name))
        self._trades_since_refit += 1

        logger.debug(
            f"Recorded outcome: raw_score={raw_score:.3f}, outcome={outcome:.2f}, "
            f"regime={regime_name}, total_history={len(self._trade_history)}"
        )

        # Trigger refit if threshold reached
        if self._trades_since_refit >= self._config.refit_interval:
            logger.info(
                f"Refit interval reached ({self._trades_since_refit} trades). "
                f"Refitting Platt parameters."
            )
            self.refit_calibration()
            self._trades_since_refit = 0

    def refit_calibration(self) -> None:
        """Refit Platt scaling parameters from accumulated trade history.

        Refits both global (across all regimes) and regime-specific calibrations.
        Requires minimum trade counts to proceed.

        Persists updated parameters to disk after successful refit.
        """
        if len(self._trade_history) < self._config.min_trades_for_calibration:
            logger.info(
                f"Not enough trades for calibration: {len(self._trade_history)} "
                f"< {self._config.min_trades_for_calibration}"
            )
            return

        if not _HAS_SCIPY:
            logger.warning("scipy not available — cannot refit Platt scaling")
            return

        try:
            # 1. Global calibration (all trades)
            raw_scores = [entry[0] for entry in self._trade_history]
            outcomes = [entry[1] for entry in self._trade_history]

            global_params = self._fit_platt_scaling(raw_scores, outcomes)
            if global_params is not None:
                self._platt_params["_global"] = global_params
                logger.info(
                    f"Refitted global Platt scaling: coef={global_params['coef']:.4f}, "
                    f"intercept={global_params['intercept']:.4f}"
                )

            # 2. Regime-specific calibration (if enough data per regime)
            regime_groups: Dict[str, Tuple[List[float], List[float]]] = {}
            for raw_score, outcome, regime in self._trade_history:
                if regime not in regime_groups:
                    regime_groups[regime] = ([], [])
                regime_groups[regime][0].append(raw_score)
                regime_groups[regime][1].append(outcome)

            for regime_name, (scores, outs) in regime_groups.items():
                if len(scores) >= self._config.min_trades_per_regime:
                    regime_params = self._fit_platt_scaling(scores, outs)
                    if regime_params is not None:
                        self._platt_params[regime_name] = regime_params
                        logger.info(
                            f"Refitted {regime_name} Platt scaling: "
                            f"coef={regime_params['coef']:.4f}, "
                            f"intercept={regime_params['intercept']:.4f} "
                            f"(n={len(scores)})"
                        )
                else:
                    logger.debug(
                        f"Skipping {regime_name} Platt fit: only {len(scores)} trades "
                        f"(min {self._config.min_trades_per_regime})"
                    )

            # Persist updated calibration
            self._save_calibration()

        except Exception as e:
            logger.error(f"Refit calibration failed: {e}", exc_info=True)

    def _compute_ensemble_disagreement(
        self,
        agent_verdicts: List[Any],
    ) -> Tuple[float, str]:
        """Compute standard deviation of agent scores and classify level.

        Classification:
        - < 0.15: LOW
        - < 0.30: MODERATE
        - < 0.45: HIGH
        - >= 0.45: CRITICAL

        Args:
            agent_verdicts: List of AgentVerdict with score attribute

        Returns:
            (disagreement_std, disagreement_level_str)
        """
        scores = [_safe_float(_verdict_val(v, 'score', 0.5), 0.5) for v in agent_verdicts]

        if len(scores) == 0:
            return 0.0, "LOW"

        disagreement = float(np.std(scores))

        if disagreement < 0.15:
            level = "LOW"
        elif disagreement < 0.30:
            level = "MODERATE"
        elif disagreement < 0.45:
            level = "HIGH"
        else:
            level = "CRITICAL"

        logger.debug(f"Ensemble disagreement: std={disagreement:.4f}, level={level}")
        return disagreement, level

    def _compute_weighted_score(self, agent_verdicts: List[Any]) -> float:
        """Compute weighted average of agent scores.

        Args:
            agent_verdicts: List of AgentVerdict with score and weight attributes

        Returns:
            Weighted average score (0-1)
        """
        if not agent_verdicts:
            return 0.5

        scores = [_safe_float(_verdict_val(v, 'score', 0.5), 0.5) for v in agent_verdicts]
        weights = [_safe_float(_verdict_val(v, 'weight', 1.0), 1.0) for v in agent_verdicts]

        total_weight = sum(weights)
        if total_weight <= 0:
            return float(np.mean(scores))

        weighted = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return _clip01(weighted)

    def _platt_scale(
        self,
        raw_score: float,
        regime_name: str,
    ) -> Tuple[float, bool]:
        """Apply sigmoid Platt scaling: P = 1 / (1 + exp(-(coef * score + intercept))).

        Uses regime-specific calibration if available, falls back to global.
        Returns raw_score if calibration not yet available.

        Args:
            raw_score: Raw confidence score (0-1)
            regime_name: Volatility regime name

        Returns:
            (calibrated_probability, is_calibrated)
        """
        raw_score = _clip01(raw_score)

        # Select parameters: regime-specific preferred, fallback to global
        params = None
        used_regime = None

        if regime_name in self._platt_params:
            params = self._platt_params[regime_name]
            used_regime = regime_name
        elif "_global" in self._platt_params:
            params = self._platt_params["_global"]
            used_regime = "_global"

        if params is None:
            # No calibration yet, return raw score
            logger.debug(f"No Platt calibration available for {regime_name}, using raw score")
            return raw_score, False

        # Apply sigmoid: P = 1 / (1 + exp(-(coef * score + intercept)))
        coef = params.get("coef", 1.0)
        intercept = params.get("intercept", 0.0)

        try:
            exponent = -(coef * raw_score + intercept)
            # Clamp exponent to prevent overflow
            exponent = max(-100, min(100, exponent))
            calibrated = 1.0 / (1.0 + np.exp(exponent))
            calibrated = _clip01(calibrated)

            logger.debug(
                f"Platt scaling: raw={raw_score:.3f} -> calibrated={calibrated:.3f} "
                f"(regime={used_regime})"
            )
            return calibrated, True

        except Exception as e:
            logger.warning(f"Platt scaling failed: {e}, returning raw score")
            return raw_score, False

    def _compute_agent_agreement(
        self,
        agent_verdicts: List[Any],
    ) -> Tuple[float, int]:
        """Compute agent agreement quality (coherence of consensus).

        agreement = 1 - (std / 0.5)  # 0.5 is max std for [0,1] range
        Clamped to [0, 1].

        Low agreement (<0.5) penalizes confidence.

        Args:
            agent_verdicts: List of AgentVerdict

        Returns:
            (agreement_quality, number_of_agents_reporting)
        """
        if not agent_verdicts:
            return 0.5, 0

        scores = [_safe_float(_verdict_val(v, 'score', 0.5), 0.5) for v in agent_verdicts]
        n_agents = len(scores)

        if n_agents < self._config.min_agents_for_quality:
            logger.debug(
                f"Not enough agents reporting: {n_agents} "
                f"< {self._config.min_agents_for_quality}"
            )
            return 0.5, n_agents

        std = float(np.std(scores))
        # Max possible std for [0,1] range is ~0.5 (when half are 0, half are 1)
        agreement = max(0.0, 1.0 - (std / 0.5))
        agreement = _clip01(agreement)

        logger.debug(f"Agent agreement: std={std:.4f}, agreement={agreement:.3f}")
        return agreement, n_agents

    def _compute_meta_confidence(self, regime_name: str) -> float:
        """Compute confidence in the calibration itself.

        Based on:
        1. Number of trades used for calibration (sample size)
        2. Recency of trades (decay if old)

        meta = min(n_trades / min_trades, 1.0) * recency_factor

        Args:
            regime_name: Volatility regime

        Returns:
            Meta-confidence (0-1)
        """
        if not self._trade_history:
            logger.debug("No trade history, meta_confidence=0.5")
            return 0.5

        # 1. Sample size factor
        n_trades = len(self._trade_history)
        min_trades = self._config.min_trades_for_calibration
        sample_factor = min(float(n_trades) / min_trades, 1.0)

        # 2. Recency factor: weight recent trades higher
        # Count trades from last 30 days
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=30)

        recent_count = 0
        for entry in self._trade_history:
            # Entry structure: (raw_score, outcome, regime)
            # We don't have timestamps in trade_history, so use count fraction as proxy
            recent_count += 1

        # Approximate: assume last 30 entries are "recent" if we have >= 30 total
        if n_trades >= 30:
            recent_count = min(30, n_trades)

        recency_factor = float(recent_count) / max(1, n_trades)
        recency_factor = _clip01(recency_factor)

        # Combine factors
        meta = sample_factor * (0.5 + 0.5 * recency_factor)
        meta = _clip01(meta)

        logger.debug(
            f"Meta-confidence: n_trades={n_trades}, sample_factor={sample_factor:.3f}, "
            f"recency={recency_factor:.3f}, meta={meta:.3f}"
        )
        return meta

    def _combine_confidence_components(
        self,
        platt_calibrated: float,
        agent_agreement: float,
        disagreement_level: str,
        meta_confidence: float,
    ) -> float:
        """Combine all components into final confidence score.

        final = platt_calibrated * agreement_factor * disagreement_factor * meta_weight

        where:
          agreement_factor = max(0.5, agent_agreement)
          disagreement_factor = {LOW: 1.0, MODERATE: 0.9, HIGH: 0.8, CRITICAL: 0.6}
          meta_weight = 0.5 + 0.5 * meta_confidence  # Range [0.5, 1.0]

        Args:
            platt_calibrated: Platt-scaled probability (0-1)
            agent_agreement: Agreement quality (0-1)
            disagreement_level: "LOW", "MODERATE", "HIGH", or "CRITICAL"
            meta_confidence: Calibration confidence (0-1)

        Returns:
            Final confidence score (0-1)
        """
        platt_calibrated = _clip01(platt_calibrated)
        agent_agreement = _clip01(agent_agreement)
        meta_confidence = _clip01(meta_confidence)

        # Agreement factor: penalize low agreement, boost high agreement
        agreement_factor = max(0.5, agent_agreement)

        # Disagreement factor by level
        disagreement_factors = {
            "LOW": 1.0,
            "MODERATE": 0.9,
            "HIGH": 0.8,
            "CRITICAL": 0.6,
        }
        disagreement_factor = disagreement_factors.get(disagreement_level, 0.8)

        # Meta-weight: confidence in the calibration itself
        meta_weight = 0.5 + 0.5 * meta_confidence

        # Combine
        final = platt_calibrated * agreement_factor * disagreement_factor * meta_weight
        final = _clip01(final)

        logger.debug(
            f"Confidence combination: platt={platt_calibrated:.3f}, "
            f"agreement_factor={agreement_factor:.3f}, "
            f"disagreement_factor={disagreement_factor:.3f}, "
            f"meta_weight={meta_weight:.3f}, "
            f"final={final:.3f}"
        )
        return final

    def _fit_platt_scaling(
        self,
        scores: List[float],
        outcomes: List[float],
    ) -> Optional[Dict[str, float]]:
        """Fit logistic regression (Platt scaling) on raw_score -> outcome.

        Minimizes log-loss: -mean(y * log(P) + (1-y) * log(1-P))

        where P = 1 / (1 + exp(-(coef * score + intercept)))

        Args:
            scores: List of raw confidence scores (0-1)
            outcomes: List of outcomes (1.0=win, 0.0=loss)

        Returns:
            Dict with 'coef' and 'intercept', or None if fitting failed
        """
        if not _HAS_SCIPY:
            logger.warning("scipy not available, cannot fit Platt scaling")
            return None

        if len(scores) < 2 or len(outcomes) != len(scores):
            logger.warning(f"Cannot fit Platt: scores={len(scores)}, outcomes={len(outcomes)}")
            return None

        try:
            scores = np.array([_safe_float(s) for s in scores], dtype=np.float64)
            outcomes = np.array([_safe_float(o) for o in outcomes], dtype=np.float64)

            # Loss function: log-loss
            def log_loss(params: np.ndarray) -> float:
                coef, intercept = params
                logits = coef * scores + intercept
                # Clamp to prevent overflow
                logits = np.clip(logits, -100, 100)
                probs = 1.0 / (1.0 + np.exp(-logits))
                probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                loss = -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
                return loss

            # Initial guess
            x0 = np.array([1.0, 0.0])

            # Minimize
            result = minimize(
                log_loss,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
            )

            if not result.success:
                logger.warning(f"Platt fitting did not converge: {result.message}")
                return None

            coef, intercept = result.x

            # Validate fitted parameters
            if not np.isfinite(coef) or not np.isfinite(intercept):
                logger.error(f"Platt fitting produced non-finite parameters: coef={coef}, intercept={intercept}")
                return None

            return {
                "coef": float(coef),
                "intercept": float(intercept),
            }

        except Exception as e:
            logger.error(f"Platt fitting failed: {e}", exc_info=True)
            return None

    def _load_calibration(self) -> None:
        """Load persisted calibration parameters from JSON.

        Uses safe_json_read for corruption recovery.
        Falls back to empty parameters if file missing or corrupted.
        """
        path = self._config.calibration_file

        try:
            data = safe_json_read(path, default={})

            if not isinstance(data, dict):
                logger.warning(f"Calibration file not a dict, resetting: {path}")
                return

            # Load Platt parameters
            platt_data = data.get("platt_params", {})
            if isinstance(platt_data, dict):
                for regime, params in platt_data.items():
                    if isinstance(params, dict):
                        coef = _safe_float(params.get("coef"))
                        intercept = _safe_float(params.get("intercept"))
                        self._platt_params[regime] = {
                            "coef": coef,
                            "intercept": intercept,
                        }

            # Load trade history
            history_data = data.get("trade_history", [])
            if isinstance(history_data, list):
                for entry in history_data:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                        raw_score = _safe_float(entry[0])
                        outcome = _safe_float(entry[1])
                        regime = str(entry[2]) if len(entry) > 2 else "NORMAL"
                        self._trade_history.append((raw_score, outcome, regime))

            logger.info(
                f"Loaded calibration: {len(self._platt_params)} regime params, "
                f"{len(self._trade_history)} trade outcomes"
            )

        except Exception as e:
            logger.warning(f"Failed to load calibration from {path}: {e}")
            # Graceful fallback: start with empty parameters
            self._platt_params = {}
            self._trade_history = []

    def _save_calibration(self) -> None:
        """Atomically save calibration parameters to JSON.

        Uses safe_json_write for atomic writes.
        Persists Platt parameters and trade history.
        """
        path = self._config.calibration_file

        try:
            data = {
                "version": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "platt_params": dict(self._platt_params),
                "trade_history": [
                    [raw_score, outcome, regime]
                    for raw_score, outcome, regime in self._trade_history
                ],
                "metadata": {
                    "n_trades": len(self._trade_history),
                    "n_regimes": len(self._platt_params),
                },
            }

            success = safe_json_write(path, data, indent=2)
            if success:
                logger.info(
                    f"Saved calibration to {path}: "
                    f"{len(self._platt_params)} regimes, "
                    f"{len(self._trade_history)} trades"
                )
            else:
                logger.error(f"Failed to save calibration to {path}")

        except Exception as e:
            logger.error(f"Error saving calibration: {e}", exc_info=True)
