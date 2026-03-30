"""
Feature Drift Detector — Phase 54 (US-335).

Detects distribution shifts in model features and win-rate changes using:
  - Kolmogorov-Smirnov test on top features (recent vs baseline)
  - Page-Hinkley change detector on rolling win rate

When drift is detected:
  - Sets drift_detected flag consumed by WFEStabilityMonitor
  - Logs warning with drift details
  - Persists drift history for audit trail

Usage:
    detector = FeatureDriftDetector()
    result = detector.check_drift(recent_trades, baseline_trades)
    if result.drift_detected:
        # Flag for WFE monitor
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/feature_drift_history.json"

# Features extracted from trade journal for drift detection
_TOP_FEATURES = [
    "confidence",
    "weighted_vote_score",
    "uncertainty_score",
    "model_disagreement",
    "atr",
    "trend_strength",
    "momentum_score",
    "regime_score",
    "spread_atr_pct",
    "setup_quality_score",
]


@dataclass
class DriftCheckResult:
    """Result of a single drift check."""
    drift_detected: bool = False
    ks_drift_count: int = 0        # Number of features with p < threshold
    ks_results: Dict[str, float] = field(default_factory=dict)  # {feature: p_value}
    ph_signal: bool = False         # Page-Hinkley triggered
    ph_cumulative: float = 0.0      # Page-Hinkley cumulative statistic
    ph_minimum: float = 0.0         # Page-Hinkley running minimum
    win_rate_recent: float = 0.0
    win_rate_baseline: float = 0.0
    feature_rank_correlation: float = 0.0  # Spearman correlation
    reason: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "ks_drift_count": self.ks_drift_count,
            "ks_results": {k: round(v, 6) for k, v in self.ks_results.items()},
            "ph_signal": self.ph_signal,
            "ph_cumulative": round(self.ph_cumulative, 4),
            "ph_minimum": round(self.ph_minimum, 4),
            "win_rate_recent": round(self.win_rate_recent, 4),
            "win_rate_baseline": round(self.win_rate_baseline, 4),
            "feature_rank_correlation": round(self.feature_rank_correlation, 4),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftCheckResult":
        return cls(
            drift_detected=bool(d.get("drift_detected", False)),
            ks_drift_count=int(d.get("ks_drift_count", 0)),
            ks_results=d.get("ks_results", {}),
            ph_signal=bool(d.get("ph_signal", False)),
            ph_cumulative=float(d.get("ph_cumulative", 0.0)),
            ph_minimum=float(d.get("ph_minimum", 0.0)),
            win_rate_recent=float(d.get("win_rate_recent", 0.0)),
            win_rate_baseline=float(d.get("win_rate_baseline", 0.0)),
            feature_rank_correlation=float(d.get("feature_rank_correlation", 0.0)),
            reason=str(d.get("reason", "")),
            timestamp=str(d.get("timestamp", "")),
        )


@dataclass
class FeatureDriftConfig:
    """Configuration for feature drift detection."""
    ks_p_threshold: float = 0.05       # KS test significance threshold
    ks_drift_min_features: int = 3     # Min features drifting to flag
    ph_threshold: float = 3.0          # Page-Hinkley detection threshold
    ph_lambda: float = 500.0           # Page-Hinkley sensitivity parameter
    recent_window: int = 200           # Recent trades to compare
    baseline_window: int = 500         # Baseline trades to compare against
    check_interval: int = 50           # Check every N synced trades
    max_history: int = 50              # Max drift check results to keep
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    features: List[str] = field(default_factory=lambda: list(_TOP_FEATURES))


class PageHinkleyDetector:
    """Page-Hinkley change detection on a streaming signal.

    Detects decreases in win rate (one-sided test for decline).
    Signal triggers when cumulative sum - minimum exceeds threshold.
    """

    def __init__(self, threshold: float = 3.0, lambda_: float = 500.0):
        self.threshold = threshold
        self.lambda_ = lambda_
        self._sum: float = 0.0
        self._min_sum: float = 0.0
        self._max_sum: float = 0.0
        self._count: int = 0
        self._mean: float = 0.0

    def update(self, value: float) -> bool:
        """Update with new observation. Returns True if change detected.

        Uses Page-Hinkley test for detecting decrease in mean.
        The cumulative sum tracks deviations above the running mean minus
        a tolerance (lambda). A change is signaled when the difference
        between the maximum cumulative sum and the current sum exceeds threshold.
        """
        self._count += 1
        self._mean += (value - self._mean) / self._count
        # Track cumulative deviations: positive when value > mean
        self._sum += value - self._mean + self.lambda_
        # For detecting decrease: track maximum and signal when sum drops far below
        if self._count == 1:
            self._min_sum = self._sum
        self._min_sum = min(self._min_sum, self._sum)
        # Also track maximum for the upward test
        if not hasattr(self, "_max_sum"):
            self._max_sum = self._sum
        self._max_sum = max(self._max_sum, self._sum)
        # Signal decrease when max - current exceeds threshold
        return (self._max_sum - self._sum) > self.threshold

    @property
    def cumulative(self) -> float:
        return self._sum

    @property
    def minimum(self) -> float:
        return self._min_sum

    def reset(self):
        self._sum = 0.0
        self._min_sum = 0.0
        self._max_sum = 0.0
        self._count = 0
        self._mean = 0.0


class FeatureDriftDetector:
    """Detects feature distribution drift and win-rate changes.

    Uses KS tests on feature distributions and Page-Hinkley on win rate
    to flag when model assumptions may be violated.
    """

    def __init__(self, config: Optional[FeatureDriftConfig] = None):
        self.config = config or FeatureDriftConfig()
        self._ph_detector = PageHinkleyDetector(
            threshold=self.config.ph_threshold,
            lambda_=self.config.ph_lambda,
        )
        self._history: List[DriftCheckResult] = []
        self._drift_detected: bool = False
        self._trades_since_check: int = 0
        self._load_state()

    @property
    def drift_detected(self) -> bool:
        """Flag consumed by WFEStabilityMonitor."""
        return self._drift_detected

    def record_trade(self, won: bool) -> None:
        """Record a single trade outcome for Page-Hinkley monitoring."""
        self._trades_since_check += 1
        # Page-Hinkley tracks win rate decline: 1.0 for win, 0.0 for loss
        self._ph_detector.update(1.0 if won else 0.0)

    def should_check(self) -> bool:
        """Whether enough trades have accumulated for a drift check."""
        return self._trades_since_check >= self.config.check_interval

    def check_drift(
        self,
        recent_trades: List[Dict[str, Any]],
        baseline_trades: List[Dict[str, Any]],
    ) -> DriftCheckResult:
        """Run full drift detection: KS tests + Page-Hinkley + rank correlation.

        Args:
            recent_trades: Last N trades (typically 200)
            baseline_trades: Baseline trades (typically 500)

        Returns:
            DriftCheckResult with detection details
        """
        now = datetime.now(timezone.utc).isoformat()
        result = DriftCheckResult(timestamp=now)

        # ── KS tests on feature distributions ──
        ks_results = self._run_ks_tests(recent_trades, baseline_trades)
        result.ks_results = ks_results
        drifted_features = {
            f: p for f, p in ks_results.items()
            if p < self.config.ks_p_threshold
        }
        result.ks_drift_count = len(drifted_features)

        # ── Page-Hinkley signal ──
        result.ph_signal = (
            self._ph_detector._max_sum - self._ph_detector.cumulative
        ) > self.config.ph_threshold
        result.ph_cumulative = self._ph_detector.cumulative
        result.ph_minimum = self._ph_detector.minimum

        # ── Win rates ──
        result.win_rate_recent = self._compute_win_rate(recent_trades)
        result.win_rate_baseline = self._compute_win_rate(baseline_trades)

        # ── Feature importance rank correlation ──
        result.feature_rank_correlation = self._compute_rank_correlation(
            recent_trades, baseline_trades
        )

        # ── Drift decision ──
        reasons = []
        if result.ks_drift_count >= self.config.ks_drift_min_features:
            reasons.append(
                f"KS drift: {result.ks_drift_count} features (p<{self.config.ks_p_threshold})"
            )
        if result.ph_signal:
            reasons.append(
                f"Page-Hinkley signal: cumulative={result.ph_cumulative:.2f}"
            )

        result.drift_detected = len(reasons) > 0
        result.reason = "; ".join(reasons) if reasons else "No drift detected"

        # Update internal state
        self._drift_detected = result.drift_detected
        self._trades_since_check = 0
        self._history.append(result)
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        if result.drift_detected:
            logger.warning(
                "Phase 54 (US-335): Feature drift DETECTED — %s "
                "(KS=%d features, PH=%s, win_rate %.3f→%.3f)",
                result.reason, result.ks_drift_count, result.ph_signal,
                result.win_rate_baseline, result.win_rate_recent,
            )
        else:
            logger.info(
                "Phase 54 (US-335): Drift check clean — "
                "KS=%d features, PH=%s, rank_corr=%.3f",
                result.ks_drift_count, result.ph_signal,
                result.feature_rank_correlation,
            )

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get drift check history."""
        return [r.to_dict() for r in self._history]

    def get_status(self) -> Dict[str, Any]:
        """Get current drift detector status."""
        return {
            "drift_detected": self._drift_detected,
            "trades_since_check": self._trades_since_check,
            "history_length": len(self._history),
            "ph_cumulative": round(self._ph_detector.cumulative, 4),
            "ph_minimum": round(self._ph_detector.minimum, 4),
        }

    def reset_drift_flag(self) -> None:
        """Clear drift flag (e.g., after retraining)."""
        self._drift_detected = False
        self._ph_detector.reset()

    # ── Internal methods ──

    def _run_ks_tests(
        self,
        recent: List[Dict[str, Any]],
        baseline: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Run KS test on each feature. Returns {feature: p_value}."""
        results = {}
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            logger.debug("Phase 54 (US-335): scipy not available for KS tests")
            return results

        for feature in self.config.features:
            recent_vals = self._extract_feature(recent, feature)
            baseline_vals = self._extract_feature(baseline, feature)

            if len(recent_vals) < 10 or len(baseline_vals) < 10:
                continue

            try:
                stat, p_value = ks_2samp(recent_vals, baseline_vals)
                results[feature] = p_value
            except Exception as e:
                logger.debug(f"Phase 54: KS test failed for {feature}: {e}")

        return results

    def _extract_feature(
        self, trades: List[Dict[str, Any]], feature: str
    ) -> List[float]:
        """Extract numeric feature values from trade entries."""
        values = []
        for trade in trades:
            val = trade.get(feature)
            if val is None:
                # Check nested locations
                agents = trade.get("agents", {})
                val = agents.get(feature)
            if val is None:
                regime = trade.get("regime", {})
                if isinstance(regime, dict):
                    val = regime.get(feature)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        return values

    def _compute_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Compute win rate from trade entries."""
        if not trades:
            return 0.0
        wins = sum(
            1 for t in trades
            if t.get("outcome", {}).get("trade_won", False)
        )
        return wins / len(trades)

    def _compute_rank_correlation(
        self,
        recent: List[Dict[str, Any]],
        baseline: List[Dict[str, Any]],
    ) -> float:
        """Compute Spearman rank correlation of feature importance between wins/losses."""
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return 0.0

        features = self.config.features

        # Compute feature importance as mean difference (wins - losses)
        def _importance(trades: List[Dict[str, Any]]) -> List[float]:
            importance = []
            for feat in features:
                vals = self._extract_feature(trades, feat)
                wins = self._extract_feature(
                    [t for t in trades if t.get("outcome", {}).get("trade_won", False)],
                    feat,
                )
                losses = self._extract_feature(
                    [t for t in trades if not t.get("outcome", {}).get("trade_won", False)],
                    feat,
                )
                if wins and losses:
                    importance.append(
                        sum(wins) / len(wins) - sum(losses) / len(losses)
                    )
                else:
                    importance.append(0.0)
            return importance

        recent_imp = _importance(recent)
        baseline_imp = _importance(baseline)

        if len(recent_imp) < 3 or len(baseline_imp) < 3:
            return 0.0

        try:
            corr, _ = spearmanr(recent_imp, baseline_imp)
            return corr if corr == corr else 0.0  # NaN check
        except Exception:
            return 0.0

    def save_state(self) -> None:
        """Persist drift detection state."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "drift_detected": self._drift_detected,
                "trades_since_check": self._trades_since_check,
                "ph_state": {
                    "sum": self._ph_detector._sum,
                    "min_sum": self._ph_detector._min_sum,
                    "max_sum": self._ph_detector._max_sum,
                    "count": self._ph_detector._count,
                    "mean": self._ph_detector._mean,
                },
                "history": [r.to_dict() for r in self._history[-self.config.max_history:]],
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 54 (US-335): Failed to save drift state: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._drift_detected = bool(data.get("drift_detected", False))
                self._trades_since_check = int(data.get("trades_since_check", 0))

                ph_state = data.get("ph_state", {})
                if isinstance(ph_state, dict):
                    self._ph_detector._sum = float(ph_state.get("sum", 0.0))
                    self._ph_detector._min_sum = float(ph_state.get("min_sum", 0.0))
                    self._ph_detector._max_sum = float(ph_state.get("max_sum", 0.0))
                    self._ph_detector._count = int(ph_state.get("count", 0))
                    self._ph_detector._mean = float(ph_state.get("mean", 0.0))

                history = data.get("history", [])
                if isinstance(history, list):
                    self._history = [
                        DriftCheckResult.from_dict(r)
                        for r in history if isinstance(r, dict)
                    ]
                logger.info(
                    "Phase 54 (US-335): Loaded drift detector — "
                    "%d history entries, drift_detected=%s",
                    len(self._history), self._drift_detected,
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 54 (US-335): Could not load drift state: %s", e)
