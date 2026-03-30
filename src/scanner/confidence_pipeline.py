"""
Unified Confidence Pipeline — Phase 53 (US-331).

Consolidates isotonic calibration + confidence decomposition into a single
configurable path. Prevents double-transforms (isotonic → decompose → recompose)
that reduce signal coherence.

Config flag `use_decomposed_confidence`:
  - True:  decomposition-only (directional + timing + magnitude → recomposed)
  - False: isotonic-only (PAVA-based calibration)

Also measures calibration error: P(win|pred_conf) vs actual across 10 deciles.
Monthly calibration monitoring with degradation flag (cal_error > 0.10).

Usage:
    pipeline = UnifiedConfidencePipeline()
    result = pipeline.calibrate(
        raw_confidence=0.62,
        agent_verdicts={"trend": True, "momentum": False, ...},
        regime="NORMAL",
        isotonic_calibrator=calibrator_instance,
        decomposer=decomposer_instance,
    )
    # result.final_confidence, result.path_used, result.calibration_error
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/calibration_monitor.json"


@dataclass
class ConfidencePipelineConfig:
    """Configuration for unified confidence pipeline."""
    use_decomposed_confidence: bool = True  # True = decomposition, False = isotonic-only
    calibration_error_threshold: float = 0.10  # Flag if cal_error exceeds this
    monitoring_window: int = 100  # Recent predictions to track
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    num_deciles: int = 10


@dataclass
class PipelineResult:
    """Result of unified confidence calibration."""
    final_confidence: float = 0.0
    raw_confidence: float = 0.0
    path_used: str = ""  # "isotonic" or "decomposition" or "raw"
    isotonic_value: float = 0.0
    decomposed_value: float = 0.0
    calibration_error: float = 0.0
    degradation_flag: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class CalibrationMonitor:
    """Tracks calibration quality over time.

    For each prediction, records (predicted_confidence, actual_outcome).
    Computes calibration error across deciles:
        cal_error = mean(|predicted_bin_conf - actual_bin_win_rate|)
    """

    def __init__(
        self,
        num_deciles: int = 10,
        threshold: float = 0.10,
        persistence_path: str = _DEFAULT_PERSISTENCE_PATH,
    ):
        self.num_deciles = num_deciles
        self.threshold = threshold
        self.persistence_path = persistence_path
        self._predictions: List[Tuple[float, bool]] = []  # (confidence, won)
        self._monthly_errors: List[Dict[str, Any]] = []
        self._load_state()

    def record_prediction(self, confidence: float, won: bool) -> None:
        """Record a prediction-outcome pair."""
        self._predictions.append((confidence, won))
        # Keep bounded
        if len(self._predictions) > 2000:
            self._predictions = self._predictions[-2000:]

    def compute_calibration_error(self, recent_n: int = 200) -> float:
        """Compute calibration error across deciles.

        Groups recent predictions into confidence deciles, computes
        actual win rate per decile, and returns mean absolute error
        between predicted and actual.

        Args:
            recent_n: Number of recent predictions to use

        Returns:
            Mean absolute calibration error (0.0-1.0)
        """
        predictions = self._predictions[-recent_n:] if len(self._predictions) > recent_n else self._predictions
        if len(predictions) < 20:
            return 0.0  # Not enough data

        # Group by decile
        decile_data: Dict[int, List[bool]] = {}
        for conf, won in predictions:
            decile = min(int(conf * self.num_deciles), self.num_deciles - 1)
            if decile not in decile_data:
                decile_data[decile] = []
            decile_data[decile].append(won)

        if not decile_data:
            return 0.0

        # Compute error per decile
        errors = []
        for decile, outcomes in decile_data.items():
            predicted_conf = (decile + 0.5) / self.num_deciles
            actual_win_rate = sum(outcomes) / len(outcomes)
            errors.append(abs(predicted_conf - actual_win_rate))

        return sum(errors) / len(errors)

    def check_degradation(self) -> bool:
        """Check if calibration is degrading (error > threshold)."""
        error = self.compute_calibration_error()
        return error > self.threshold

    def get_decile_breakdown(self, recent_n: int = 200) -> Dict[str, Any]:
        """Get detailed decile-level calibration breakdown.

        Returns:
            {"deciles": [{decile, predicted, actual, count, error}], "overall_error": float}
        """
        predictions = self._predictions[-recent_n:] if len(self._predictions) > recent_n else self._predictions
        if len(predictions) < 10:
            return {"deciles": [], "overall_error": 0.0}

        decile_data: Dict[int, List[bool]] = {}
        for conf, won in predictions:
            d = min(int(conf * self.num_deciles), self.num_deciles - 1)
            if d not in decile_data:
                decile_data[d] = []
            decile_data[d].append(won)

        deciles = []
        errors = []
        for d in range(self.num_deciles):
            outcomes = decile_data.get(d, [])
            if not outcomes:
                continue
            predicted = (d + 0.5) / self.num_deciles
            actual = sum(outcomes) / len(outcomes)
            error = abs(predicted - actual)
            errors.append(error)
            deciles.append({
                "decile": d,
                "predicted": round(predicted, 3),
                "actual": round(actual, 3),
                "count": len(outcomes),
                "error": round(error, 3),
            })

        overall = sum(errors) / len(errors) if errors else 0.0
        return {"deciles": deciles, "overall_error": round(overall, 4)}

    def save_state(self) -> None:
        """Persist monitoring state with retry (US-342)."""
        path = self.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "predictions": [
                    {"conf": round(c, 4), "won": w}
                    for c, w in self._predictions[-500:]  # Save last 500
                ],
                "monthly_errors": self._monthly_errors[-12:],
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, data, context="calibration_monitor")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 53 (US-331): Failed to save calibration monitor: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                raw_preds = data.get("predictions", [])
                self._predictions = [
                    (float(p.get("conf", 0.5)), bool(p.get("won", False)))
                    for p in raw_preds if isinstance(p, dict)
                ]
                self._monthly_errors = data.get("monthly_errors", [])
                logger.info(
                    "Phase 53 (US-331): Loaded calibration monitor — %d predictions",
                    len(self._predictions),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 53 (US-331): Could not load calibration monitor: %s", e)


class UnifiedConfidencePipeline:
    """Consolidates isotonic calibration + decomposition into single path.

    Eliminates double-transforms by choosing ONE path via config flag.
    Monitors calibration quality and flags degradation.
    """

    def __init__(self, config: Optional[ConfidencePipelineConfig] = None):
        self.config = config or ConfidencePipelineConfig()
        self.monitor = CalibrationMonitor(
            num_deciles=self.config.num_deciles,
            threshold=self.config.calibration_error_threshold,
            persistence_path=self.config.persistence_path,
        )

    def calibrate(
        self,
        raw_confidence: float,
        agent_verdicts: Optional[Dict[str, bool]] = None,
        regime: str = "NORMAL",
        isotonic_calibrator: Optional[Any] = None,
        decomposer: Optional[Any] = None,
        session_name: str = "LONDON",
        mtf_signal: str = "PROCEED",
        atr_percentile: float = 0.50,
        uncertainty_score: float = 0.0,
        model_disagreement: float = 0.0,
    ) -> PipelineResult:
        """Run unified confidence calibration.

        Chooses between isotonic-only or decomposition-only based on config.
        Never applies both in sequence (prevents double-transform).

        Args:
            raw_confidence: Raw weighted vote score
            agent_verdicts: {agent_name: passed_bool}
            regime: Market regime
            isotonic_calibrator: IsotonicCalibrator instance (or None)
            decomposer: ConfidenceDecomposer instance (or None)
            session_name: Trading session
            mtf_signal: Multi-timeframe signal
            atr_percentile: ATR percentile
            uncertainty_score: Agent uncertainty
            model_disagreement: Agent disagreement

        Returns:
            PipelineResult with final confidence and diagnostics
        """
        result = PipelineResult(
            raw_confidence=raw_confidence,
            final_confidence=raw_confidence,
        )

        c = self.config

        if c.use_decomposed_confidence and decomposer is not None:
            # ── DECOMPOSITION PATH ──
            try:
                _decomp = decomposer.decompose(
                    raw_confidence=raw_confidence,
                    agent_verdicts=agent_verdicts or {},
                    uncertainty_score=uncertainty_score,
                    model_disagreement=model_disagreement,
                    session_name=session_name,
                    mtf_signal=mtf_signal,
                    regime=regime,
                    atr_percentile=atr_percentile,
                )
                result.final_confidence = max(0.0, min(1.0, _decomp.recomposed_confidence))
                result.decomposed_value = result.final_confidence
                result.path_used = "decomposition"
                result.details = {
                    "directional": getattr(_decomp, "directional_strength", 0.0),
                    "timing": getattr(_decomp, "timing_quality", 0.0),
                    "magnitude": getattr(_decomp, "magnitude_estimate", 0.0),
                    "penalty_applied": getattr(_decomp, "high_conf_penalty_applied", False),
                }
            except Exception as e:
                logger.debug(f"Phase 53 (US-331): Decomposition failed, using raw: {e}")
                result.path_used = "raw"

        elif not c.use_decomposed_confidence and isotonic_calibrator is not None:
            # ── ISOTONIC PATH ──
            try:
                _iso_result = isotonic_calibrator.calibrate(raw_confidence)
                if _iso_result is not None:
                    result.final_confidence = max(0.0, min(1.0, _iso_result))
                    result.isotonic_value = result.final_confidence
                    result.path_used = "isotonic"
                else:
                    result.path_used = "raw"
            except Exception as e:
                logger.debug(f"Phase 53 (US-331): Isotonic failed, using raw: {e}")
                result.path_used = "raw"

        else:
            # ── FALLBACK: raw confidence ──
            result.path_used = "raw"

        # Compute current calibration error
        result.calibration_error = self.monitor.compute_calibration_error()
        result.degradation_flag = result.calibration_error > c.calibration_error_threshold

        if result.degradation_flag:
            logger.warning(
                "Phase 53 (US-331): Calibration degradation detected — "
                "error=%.3f > threshold=%.3f (path=%s)",
                result.calibration_error, c.calibration_error_threshold,
                result.path_used,
            )

        return result

    def record_outcome(self, confidence: float, won: bool) -> None:
        """Record a trade outcome for calibration monitoring."""
        self.monitor.record_prediction(confidence, won)

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get full calibration report with decile breakdown."""
        return self.monitor.get_decile_breakdown()

    def save_state(self) -> None:
        """Persist monitoring state."""
        self.monitor.save_state()
