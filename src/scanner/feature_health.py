"""
Phase 55 (US-344): Feature Health Monitor — PSI per feature with surgical importance downweighting.

Computes Population Stability Index (PSI) per feature, comparing a recent window
(last 200 trades) against a baseline window (previous 500 trades) to identify
which specific features are drifting and by how much.

PSI thresholds (industry standard from credit risk monitoring):
  < 0.10  → stable (no action)
  0.10-0.25 → moderate drift (monitor)
  > 0.25  → significant drift (recommend retraining)

Architecture:
    sync_closed_trades_rl() → record_trade() → should_check() → check_health()
    → feature_health_score (0-1) → WFEStabilityMonitor alert if < 0.60
    → persist to trained_data/feature_health.json

Relationship to Phase 54:
    FeatureDriftDetector (US-335): detects THAT drift occurred (KS test, Page-Hinkley)
    FeatureHealthMonitor (US-344): identifies WHERE and HOW MUCH (PSI per feature)
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PERSISTENCE_PATH = "trained_data/feature_health.json"
_DEFAULT_RECENT_WINDOW = 200      # Trades in "recent" window for PSI
_DEFAULT_BASELINE_WINDOW = 500    # Trades in "baseline" window for PSI
_CHECK_INTERVAL = 50              # Check every N closed trades
_PSI_STABLE = 0.10                # PSI < this → stable
_PSI_MODERATE = 0.25              # PSI < this → moderate drift; >= this → significant
_HEALTH_ALERT_THRESHOLD = 0.60    # feature_health_score < this → alert
_PSI_NORMALIZATION = 0.50         # PSI value that maps to health_score = 0.0
_NUM_BINS = 10                    # Bins for PSI histogram computation
_EPSILON = 1e-6                   # Avoid log(0)

# Top-10 features tracked — mirrors FeatureDriftDetector._TOP_FEATURES
TOP_FEATURES = [
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

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeaturePSI:
    """PSI result for a single feature."""
    feature: str
    psi_score: float
    status: str              # "stable" | "moderate" | "significant"
    recent_count: int = 0    # Non-null values in recent window
    baseline_count: int = 0  # Non-null values in baseline window

    @property
    def health_contribution(self) -> float:
        """Health contribution: 1.0 (fully healthy) → 0.0 (maximally drifted)."""
        return 1.0 - min(self.psi_score / _PSI_NORMALIZATION, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "psi_score": round(self.psi_score, 4),
            "status": self.status,
            "health_contribution": round(self.health_contribution, 4),
            "recent_count": self.recent_count,
            "baseline_count": self.baseline_count,
        }


@dataclass
class FeatureHealthResult:
    """Result of a single feature health check."""
    feature_health_score: float          # Aggregate 0-1 health score
    feature_psi: List[FeaturePSI]        # Per-feature PSI scores
    alert_triggered: bool = False        # True if health_score < threshold
    significant_drift_features: List[str] = field(default_factory=list)
    moderate_drift_features: List[str] = field(default_factory=list)
    recent_window_size: int = 0
    baseline_window_size: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_health_score": round(self.feature_health_score, 4),
            "alert_triggered": self.alert_triggered,
            "significant_drift_features": self.significant_drift_features,
            "moderate_drift_features": self.moderate_drift_features,
            "feature_psi": [f.to_dict() for f in self.feature_psi],
            "recent_window_size": self.recent_window_size,
            "baseline_window_size": self.baseline_window_size,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> Optional[float]:
    """Try to convert val to float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _extract_feature(trades: List[Dict[str, Any]], feature: str) -> List[float]:
    """Extract numeric feature values from trade journal entries.

    Checks top-level, agents sub-dict, and regime sub-dict (mirrors FeatureDriftDetector).
    """
    values: List[float] = []
    for trade in trades:
        val = trade.get(feature)
        if val is None:
            agents = trade.get("agents", {})
            if isinstance(agents, dict):
                val = agents.get(feature)
        if val is None:
            regime = trade.get("regime", {})
            if isinstance(regime, dict):
                val = regime.get(feature)
        converted = _safe_float(val)
        if converted is not None:
            values.append(converted)
    return values


def _compute_psi(recent_vals: List[float], baseline_vals: List[float], n_bins: int = _NUM_BINS) -> float:
    """Compute Population Stability Index between two value lists.

    PSI = Σ (A_i - E_i) × ln(A_i / E_i)
    where A_i = actual (recent) proportion, E_i = expected (baseline) proportion per bin.

    Returns 0.0 if either list is empty or has insufficient variance.
    """
    if len(recent_vals) < 5 or len(baseline_vals) < 5:
        return 0.0

    # Build bins from combined range
    all_vals = recent_vals + baseline_vals
    min_val = min(all_vals)
    max_val = max(all_vals)
    if max_val <= min_val:
        return 0.0  # No variance — treat as stable

    bin_edges = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
    bin_edges[-1] += _EPSILON  # Ensure max value falls in last bin

    def bin_counts(vals: List[float]) -> List[int]:
        counts = [0] * n_bins
        for v in vals:
            for i in range(n_bins):
                if bin_edges[i] <= v < bin_edges[i + 1]:
                    counts[i] += 1
                    break
        return counts

    baseline_counts = bin_counts(baseline_vals)
    recent_counts = bin_counts(recent_vals)
    n_baseline = len(baseline_vals)
    n_recent = len(recent_vals)

    psi = 0.0
    for bc, rc in zip(baseline_counts, recent_counts):
        expected = max(bc, 1) / n_baseline       # E_i (floor at 1 to avoid /0)
        actual = max(rc, 1) / n_recent           # A_i (floor at 1 to avoid log(0))
        psi += (actual - expected) * math.log(actual / expected)

    return max(0.0, psi)


def _classify_psi(psi: float) -> str:
    """Map PSI value to stability label."""
    if psi < _PSI_STABLE:
        return "stable"
    if psi < _PSI_MODERATE:
        return "moderate"
    return "significant"


def _write_json_atomic(path: str, data: Any) -> None:
    """Atomic JSON write: tmp → rename (JSON Safety Gates)."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
        except Exception:
            os.unlink(tmp_path)
            raise
        os.rename(tmp_path, path)
    except Exception as exc:
        logger.warning("FeatureHealthMonitor: atomic write failed for %s: %s", path, exc)
        raise


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureHealthMonitor:
    """Monitors per-feature PSI and aggregates a feature_health_score (0-1).

    Integrates with sync_closed_trades_rl():
        1. record_trade() increments the trade counter
        2. should_check() returns True every CHECK_INTERVAL trades
        3. check_health(recent, baseline) computes PSI per feature
        4. Triggers WFEStabilityMonitor alert when health_score < 0.60
        5. Persists results to trained_data/feature_health.json
    """

    def __init__(
        self,
        persistence_path: str = _DEFAULT_PERSISTENCE_PATH,
        recent_window: int = _DEFAULT_RECENT_WINDOW,
        baseline_window: int = _DEFAULT_BASELINE_WINDOW,
        check_interval: int = _CHECK_INTERVAL,
        health_alert_threshold: float = _HEALTH_ALERT_THRESHOLD,
        features: Optional[List[str]] = None,
    ) -> None:
        self.persistence_path = persistence_path
        self.recent_window = recent_window
        self.baseline_window = baseline_window
        self.check_interval = check_interval
        self.health_alert_threshold = health_alert_threshold
        self.features: List[str] = features or TOP_FEATURES

        self._trade_counter: int = 0
        self._last_check_at: int = 0
        self._last_result: Optional[FeatureHealthResult] = None
        self._history: List[Dict[str, Any]] = []   # Last N health check results

        self._load_state()
        logger.info(
            "Phase 55 (US-344): FeatureHealthMonitor initialized — "
            "%d features, interval=%d, alert_threshold=%.2f",
            len(self.features), self.check_interval, self.health_alert_threshold,
        )

    # ------------------------------------------------------------------
    # Public API (matches sync_closed_trades_rl() integration pattern)
    # ------------------------------------------------------------------

    def record_trade(self) -> None:
        """Increment the trade counter. Call once per synced closed trade."""
        self._trade_counter += 1

    def should_check(self) -> bool:
        """Return True when enough new trades have accumulated for a check."""
        return (self._trade_counter - self._last_check_at) >= self.check_interval

    def check_health(
        self,
        recent_trades: List[Dict[str, Any]],
        baseline_trades: List[Dict[str, Any]],
        wfe_monitor: Any = None,
    ) -> FeatureHealthResult:
        """Compute PSI per feature and aggregate feature_health_score.

        Args:
            recent_trades:   Last ~200 trade journal entries.
            baseline_trades: Previous ~500 trade journal entries.
            wfe_monitor:     Optional WFEStabilityMonitor to trigger alert on.

        Returns:
            FeatureHealthResult with per-feature PSI and aggregate health score.
        """
        self._last_check_at = self._trade_counter
        feature_psi_list: List[FeaturePSI] = []
        significant_features: List[str] = []
        moderate_features: List[str] = []

        for feat in self.features:
            recent_vals = _extract_feature(recent_trades, feat)
            baseline_vals = _extract_feature(baseline_trades, feat)
            psi_score = _compute_psi(recent_vals, baseline_vals)
            status = _classify_psi(psi_score)

            feature_psi = FeaturePSI(
                feature=feat,
                psi_score=psi_score,
                status=status,
                recent_count=len(recent_vals),
                baseline_count=len(baseline_vals),
            )
            feature_psi_list.append(feature_psi)

            if status == "significant":
                significant_features.append(feat)
                logger.warning(
                    "Phase 55 (US-344): Feature '%s' PSI=%.4f — SIGNIFICANT DRIFT (>0.25), recommend retraining",
                    feat, psi_score,
                )
            elif status == "moderate":
                moderate_features.append(feat)
                logger.info(
                    "Phase 55 (US-344): Feature '%s' PSI=%.4f — moderate drift (0.10-0.25)",
                    feat, psi_score,
                )

        # Aggregate health score: mean of per-feature health contributions
        if feature_psi_list:
            health_contributions = [f.health_contribution for f in feature_psi_list]
            feature_health_score = sum(health_contributions) / len(health_contributions)
        else:
            feature_health_score = 1.0

        alert_triggered = feature_health_score < self.health_alert_threshold

        result = FeatureHealthResult(
            feature_health_score=round(feature_health_score, 4),
            feature_psi=feature_psi_list,
            alert_triggered=alert_triggered,
            significant_drift_features=significant_features,
            moderate_drift_features=moderate_features,
            recent_window_size=len(recent_trades),
            baseline_window_size=len(baseline_trades),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._last_result = result
        self._history.append(result.to_dict())
        if len(self._history) > 50:
            self._history = self._history[-50:]

        logger.info(
            "Phase 55 (US-344): Feature health check — score=%.3f, significant=%d, moderate=%d, alert=%s",
            feature_health_score, len(significant_features), len(moderate_features), alert_triggered,
        )

        # Trigger WFEStabilityMonitor alert if available and health is poor
        if alert_triggered and wfe_monitor is not None:
            try:
                # Record a synthetic low WFE to signal instability
                if hasattr(wfe_monitor, "record_wfe"):
                    wfe_monitor.record_wfe(
                        wfe=feature_health_score,
                        notes=f"FeatureHealthMonitor alert: score={feature_health_score:.3f}, "
                              f"drifted_features={significant_features}",
                    )
                    logger.warning(
                        "Phase 55 (US-344): WFEStabilityMonitor alerted — "
                        "feature_health_score=%.3f < %.2f threshold",
                        feature_health_score, self.health_alert_threshold,
                    )
            except Exception as _wfe_err:
                logger.debug("Phase 55: WFE monitor alert failed: %s", _wfe_err)

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> Optional[FeatureHealthResult]:
        return self._last_result

    @property
    def last_health_score(self) -> float:
        if self._last_result is not None:
            return self._last_result.feature_health_score
        return 1.0  # Assume healthy until first check

    def get_drifted_features(self) -> List[str]:
        """Return list of features with significant PSI (> 0.25)."""
        if self._last_result is None:
            return []
        return list(self._last_result.significant_drift_features)

    def get_feature_weights(self) -> Dict[str, float]:
        """Return per-feature health weights (0-1) for importance downweighting.

        Features with high PSI get lower weights. Stable features stay at 1.0.
        Consumers can multiply feature importance by these weights.
        """
        if self._last_result is None:
            return {f: 1.0 for f in self.features}
        return {
            fp.feature: round(fp.health_contribution, 4)
            for fp in self._last_result.feature_psi
        }

    # ------------------------------------------------------------------
    # Persistence (JSON Safety Gates)
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Atomically persist feature health state to disk."""
        payload: Dict[str, Any] = {
            "_schema_version": SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "trade_counter": self._trade_counter,
            "last_check_at": self._last_check_at,
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "history": self._history[-20:],  # Keep last 20 checks in file
        }
        try:
            _write_json_atomic(self.persistence_path, payload)
        except Exception as exc:
            logger.error("Phase 55 (US-344): Failed to persist feature health: %s", exc)

    def _load_state(self) -> None:
        """Load persisted state with safe fallback on corruption."""
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return

            self._trade_counter = int(data.get("trade_counter", 0))
            self._last_check_at = int(data.get("last_check_at", 0))

            raw_history = data.get("history", [])
            if isinstance(raw_history, list):
                self._history = [r for r in raw_history if isinstance(r, dict)]

            # Restore last result if available
            raw_last = data.get("last_result")
            if isinstance(raw_last, dict):
                try:
                    self._last_result = FeatureHealthResult(
                        feature_health_score=float(raw_last.get("feature_health_score", 1.0)),
                        feature_psi=[
                            FeaturePSI(
                                feature=fp.get("feature", ""),
                                psi_score=float(fp.get("psi_score", 0.0)),
                                status=fp.get("status", "stable"),
                                recent_count=int(fp.get("recent_count", 0)),
                                baseline_count=int(fp.get("baseline_count", 0)),
                            )
                            for fp in raw_last.get("feature_psi", [])
                            if isinstance(fp, dict)
                        ],
                        alert_triggered=bool(raw_last.get("alert_triggered", False)),
                        significant_drift_features=raw_last.get("significant_drift_features", []),
                        moderate_drift_features=raw_last.get("moderate_drift_features", []),
                        recent_window_size=int(raw_last.get("recent_window_size", 0)),
                        baseline_window_size=int(raw_last.get("baseline_window_size", 0)),
                        timestamp=raw_last.get("timestamp", ""),
                    )
                except Exception as _restore_err:
                    logger.debug("Phase 55: Could not restore last health result: %s", _restore_err)

            logger.info(
                "Phase 55 (US-344): Loaded feature health state — "
                "total_trades=%d, history_entries=%d",
                self._trade_counter, len(self._history),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Phase 55 (US-344): Corrupt feature health state, starting fresh: %s", exc)
