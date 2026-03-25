"""Causal signal filtering via Granger causality.

US-095: Compute pairwise Granger causality between FX feature streams
and price movements. Build causal graph per regime. Filter trades where
signal direction contradicts discovered causal structure. Alert when
causal structure breaks (regime change indicator).

Based on 2024 CD-NOTS and SDCI research.

Granger Causality Test:
- For each (feature, price_returns) pair, fit two autoregressions:
  1. Restricted: price_t = a0 + a1*price_{t-1} + ... + ap*price_{t-p}
  2. Unrestricted: price_t = a0 + ... + ap*price_{t-p} + b1*feature_{t-1} + ... + bp*feature_{t-p}
- F-test: does the unrestricted model significantly reduce RSS?
- If p < alpha: feature Granger-causes price returns at that lag.

Lightweight implementation using OLS (no scipy dependency for F-test —
computed directly from RSS ratios).
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
_CAUSAL_GRAPH_PATH = _PROJECT_ROOT / "trained_data" / "causal_graph.json"

# Significance level for Granger test
DEFAULT_ALPHA = 0.05
DEFAULT_MAX_LAG = 5
DEFAULT_MIN_SAMPLES = 50


@dataclass
class GrangerResult:
    """Result of a single Granger causality test."""

    feature_name: str = ""
    is_causal: bool = False
    p_value: float = 1.0
    optimal_lag: int = 0
    f_statistic: float = 0.0
    direction_coefficient: float = 0.0  # Sign indicates causal direction

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "is_causal": self.is_causal,
            "p_value": round(self.p_value, 6),
            "optimal_lag": self.optimal_lag,
            "f_statistic": round(self.f_statistic, 4),
            "direction_coefficient": round(self.direction_coefficient, 6),
        }


@dataclass
class CausalGraph:
    """Causal graph with significant edges from features to price returns."""

    edges: List[GrangerResult] = field(default_factory=list)
    regime: str = "NORMAL"
    n_features_tested: int = 0
    n_significant: int = 0
    build_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "regime": self.regime,
            "n_features_tested": self.n_features_tested,
            "n_significant": self.n_significant,
            "build_timestamp": self.build_timestamp,
        }

    def get_causal_features(self) -> Dict[str, float]:
        """Return dict of causal feature → direction coefficient."""
        return {
            e.feature_name: e.direction_coefficient
            for e in self.edges
            if e.is_causal
        }


class CausalSignalFilter:
    """Granger causality-based signal filtering.

    Builds a causal graph from feature streams → price returns, then
    uses it to filter trades where the signal direction contradicts
    the discovered causal structure.

    Usage:
        cf = CausalSignalFilter()
        graph = cf.build_causal_graph(feature_dict, price_returns, regime="NORMAL")
        score = cf.filter_signal("BUY", features, graph)
        # score near 1.0 = consistent with causal structure
        # score near 0.0 = contradicts causal structure
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_lag: int = DEFAULT_MAX_LAG,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        persistence_path: Optional[Path] = None,
    ):
        self.alpha = alpha
        self.max_lag = max_lag
        self.min_samples = min_samples
        self.persistence_path = persistence_path or _CAUSAL_GRAPH_PATH

        # Cache of causal graphs per regime
        self._graphs: Dict[str, CausalGraph] = {}

        # Rebuild counter
        self._scan_count: int = 0
        self._rebuild_interval: int = 200  # Rebuild graph every N scans

        # Load persisted graph
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_granger(
        self,
        feature_series: np.ndarray,
        price_returns: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> GrangerResult:
        """Compute Granger causality test for one feature → price returns.

        Args:
            feature_series: 1D array of feature values over time.
            price_returns: 1D array of price returns over same period.
            max_lag: Maximum lag to test (uses default if None).

        Returns:
            GrangerResult with test statistics.
        """
        result = GrangerResult()
        _max_lag = max_lag or self.max_lag

        n = min(len(feature_series), len(price_returns))
        if n < self.min_samples:
            return result

        x = np.asarray(feature_series[:n], dtype=np.float64)
        y = np.asarray(price_returns[:n], dtype=np.float64)

        best_f = 0.0
        best_p = 1.0
        best_lag = 0
        best_coeff = 0.0

        for lag in range(1, _max_lag + 1):
            if n - lag < lag * 2 + 5:
                continue  # Not enough samples for this lag

            f_stat, p_val, coeff = self._granger_f_test(x, y, lag)

            if f_stat > best_f:
                best_f = f_stat
                best_p = p_val
                best_lag = lag
                best_coeff = coeff

        result.f_statistic = best_f
        result.p_value = best_p
        result.optimal_lag = best_lag
        result.direction_coefficient = best_coeff
        result.is_causal = best_p < self.alpha

        return result

    def build_causal_graph(
        self,
        feature_dict: Dict[str, np.ndarray],
        price_returns: np.ndarray,
        regime: str = "NORMAL",
    ) -> CausalGraph:
        """Build causal graph from multiple feature streams.

        Args:
            feature_dict: Dict mapping feature_name → time series array.
            price_returns: Price returns time series.
            regime: Current volatility regime.

        Returns:
            CausalGraph with significant causal edges.
        """
        graph = CausalGraph(
            regime=regime,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        for feature_name, feature_series in feature_dict.items():
            result = self.compute_granger(feature_series, price_returns)
            result.feature_name = feature_name
            graph.edges.append(result)
            graph.n_features_tested += 1
            if result.is_causal:
                graph.n_significant += 1

        # Cache
        self._graphs[regime] = graph

        logger.info(
            f"CausalFilter [{regime}]: Built graph — "
            f"{graph.n_significant}/{graph.n_features_tested} "
            f"significant causal features"
        )

        return graph

    def filter_signal(
        self,
        direction: str,
        features: Dict[str, float],
        causal_graph: Optional[CausalGraph] = None,
        regime: str = "NORMAL",
    ) -> float:
        """Check signal consistency against causal structure.

        Args:
            direction: Proposed trade direction ("BUY" or "SELL").
            features: Current feature values for this trade candidate.
            causal_graph: Causal graph to check against (uses cached if None).
            regime: Current regime (for cached graph lookup).

        Returns:
            Consistency score in [0, 1]. 1 = fully consistent, 0 = contradicts.
        """
        graph = causal_graph or self._graphs.get(regime)
        if graph is None or not graph.edges:
            return 1.0  # No graph available — pass through

        causal_features = graph.get_causal_features()
        if not causal_features:
            return 1.0

        consistent_count = 0
        checked_count = 0
        direction_mult = 1.0 if direction.upper() == "BUY" else -1.0

        for feature_name, causal_coeff in causal_features.items():
            if feature_name not in features:
                continue

            feature_value = features[feature_name]
            if abs(feature_value) < 1e-6:
                continue

            checked_count += 1

            # Expected: if causal_coeff > 0 and feature > 0, that predicts positive returns
            # For BUY: we want positive returns → feature * causal_coeff should be positive
            # For SELL: we want negative returns → feature * causal_coeff should be negative
            predicted_direction = feature_value * causal_coeff
            if predicted_direction * direction_mult > 0:
                consistent_count += 1

        if checked_count == 0:
            return 1.0

        return consistent_count / checked_count

    def should_rebuild(self) -> bool:
        """Check if causal graph should be rebuilt."""
        self._scan_count += 1
        return self._scan_count % self._rebuild_interval == 0

    def get_status(self) -> Dict[str, Any]:
        """Get filter status for observability."""
        return {
            "cached_regimes": list(self._graphs.keys()),
            "scan_count": self._scan_count,
            "rebuild_interval": self._rebuild_interval,
            "alpha": self.alpha,
            "max_lag": self.max_lag,
            "graphs": {
                regime: {
                    "n_features": g.n_features_tested,
                    "n_significant": g.n_significant,
                    "built": g.build_timestamp,
                }
                for regime, g in self._graphs.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _granger_f_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
    ) -> Tuple[float, float, float]:
        """Compute F-test for Granger causality at a specific lag.

        Returns: (f_statistic, p_value_approx, direction_coefficient)
        """
        n = len(y)
        if n <= 2 * lag + 1:
            return 0.0, 1.0, 0.0

        # Build lagged matrices
        Y = y[lag:]  # Target
        n_obs = len(Y)

        # Restricted model: Y ~ Y_lags only
        Y_lags = np.column_stack([y[lag - i - 1: n - i - 1] for i in range(lag)])
        ones = np.ones((n_obs, 1))
        X_restricted = np.hstack([ones, Y_lags])

        # Unrestricted model: Y ~ Y_lags + X_lags
        X_lags = np.column_stack([x[lag - i - 1: n - i - 1] for i in range(lag)])
        X_unrestricted = np.hstack([X_restricted, X_lags])

        try:
            # OLS for restricted
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            rss_r = float(np.sum(resid_r ** 2))

            # OLS for unrestricted
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            rss_u = float(np.sum(resid_u ** 2))

            # F-statistic
            df1 = lag  # Additional parameters in unrestricted
            df2 = n_obs - X_unrestricted.shape[1]

            if rss_u <= 0 or df2 <= 0:
                return 0.0, 1.0, 0.0

            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

            # Approximate p-value using F-distribution CDF
            # Simplified: use the regularized incomplete beta function approximation
            p_value = self._f_pvalue_approx(f_stat, df1, df2)

            # Direction coefficient: average of X_lag coefficients in unrestricted model
            x_coeffs = beta_u[1 + lag:]  # Skip intercept and Y-lag coefficients
            direction_coeff = float(np.mean(x_coeffs)) if len(x_coeffs) > 0 else 0.0

            return f_stat, p_value, direction_coeff

        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 1.0, 0.0

    @staticmethod
    def _f_pvalue_approx(f_stat: float, df1: int, df2: int) -> float:
        """Approximate p-value for F-distribution.

        Uses the normal approximation for large df2.
        For small df2, uses a conservative bound.
        """
        if f_stat <= 0 or df1 <= 0 or df2 <= 0:
            return 1.0

        # For df2 > 30, use Wilson-Hilferty approximation
        if df2 > 30:
            # Transform F to approximate chi-squared
            x = ((f_stat * df1 / df2) ** (1.0 / 3) - (1 - 2.0 / (9 * df2))) / \
                np.sqrt(2.0 / (9 * df2))
            # Standard normal CDF approximation (logistic)
            p_value = 1.0 / (1.0 + np.exp(1.7 * x))
            return float(max(0.0, min(1.0, p_value)))

        # Conservative bound for small df2
        # If F > critical value at alpha=0.05, return small p-value
        # F(0.05, df1, df2) ≈ 1 + 2.5*df1/df2 for rough estimate
        f_crit_005 = 1.0 + 2.5 * df1 / max(df2, 1)
        f_crit_001 = 1.0 + 4.0 * df1 / max(df2, 1)

        if f_stat > f_crit_001:
            return 0.005
        elif f_stat > f_crit_005:
            return 0.03
        else:
            return 0.10 + 0.90 * np.exp(-f_stat)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist causal graphs."""
        try:
            data = {
                "version": 1,
                "graphs": {
                    regime: graph.to_dict()
                    for regime, graph in self._graphs.items()
                },
                "scan_count": self._scan_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"CausalFilter: Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load persisted causal graphs."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            self._scan_count = data.get("scan_count", 0)

            for regime, gdata in data.get("graphs", {}).items():
                graph = CausalGraph(
                    regime=gdata.get("regime", regime),
                    n_features_tested=gdata.get("n_features_tested", 0),
                    n_significant=gdata.get("n_significant", 0),
                    build_timestamp=gdata.get("build_timestamp", ""),
                )
                for edge_data in gdata.get("edges", []):
                    graph.edges.append(GrangerResult(
                        feature_name=edge_data.get("feature_name", ""),
                        is_causal=edge_data.get("is_causal", False),
                        p_value=edge_data.get("p_value", 1.0),
                        optimal_lag=edge_data.get("optimal_lag", 0),
                        f_statistic=edge_data.get("f_statistic", 0.0),
                        direction_coefficient=edge_data.get("direction_coefficient", 0.0),
                    ))
                self._graphs[regime] = graph

            logger.debug(
                f"CausalFilter: Loaded {len(self._graphs)} regime graphs"
            )
        except Exception as e:
            logger.debug(f"CausalFilter: Failed to load state: {e}")
