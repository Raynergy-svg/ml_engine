"""EWMA Correlation Engine for Dynamic Portfolio Risk Management.

Implements exponentially-weighted moving average correlation tracking across FX pairs.
Formula: Σₜ = λ·Σₜ₋₁ + (1-λ)·rₜ·rₜᵀ

Provides:
1. Real-time correlation matrix (updated with each returns observation)
2. Diversification multiplier per pair vs portfolio
3. Correlation regime detection (RISK_OFF / RISK_ON / NEUTRAL)
4. Pair clustering by correlation strength

Pure Python + numpy only. Thread-safe. Atomic JSON persistence.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from src.scanner.automation.safe_json import safe_json_write
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Correlation regime constants
RISK_OFF = "RISK_OFF"
RISK_ON = "RISK_ON"
NEUTRAL = "NEUTRAL"


@dataclass
class EWMACorrelationConfig:
    """Configuration for EWMA correlation engine.

    Attributes:
        lambda_decay: EWMA decay factor (RiskMetrics FX standard is 0.94)
        min_observations: Minimum observations before correlation is reliable
        risk_off_threshold: Average correlation > this = RISK_OFF regime
        risk_on_threshold: Average correlation < this = RISK_ON regime
        high_correlation_threshold: Pairs above this are "correlated"
        max_pairs: Maximum pairs to track
    """

    lambda_decay: float = 0.94
    min_observations: int = 20
    risk_off_threshold: float = 0.6
    risk_on_threshold: float = 0.3
    high_correlation_threshold: float = 0.7
    max_pairs: int = 20

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 < self.lambda_decay < 1:
            raise ValueError(
                f"lambda_decay must be in (0, 1), got {self.lambda_decay}"
            )
        if self.min_observations < 1:
            raise ValueError(
                f"min_observations must be >= 1, got {self.min_observations}"
            )
        if not 0 <= self.risk_on_threshold <= 1:
            raise ValueError(
                f"risk_on_threshold must be in [0, 1], got {self.risk_on_threshold}"
            )
        if not 0 <= self.risk_off_threshold <= 1:
            raise ValueError(
                f"risk_off_threshold must be in [0, 1], got {self.risk_off_threshold}"
            )
        if not 0 <= self.high_correlation_threshold <= 1:
            raise ValueError(
                f"high_correlation_threshold must be in [0, 1], "
                f"got {self.high_correlation_threshold}"
            )
        if self.risk_on_threshold > self.risk_off_threshold:
            raise ValueError(
                f"risk_on_threshold ({self.risk_on_threshold}) must be <= "
                f"risk_off_threshold ({self.risk_off_threshold})"
            )


@dataclass
class CorrelationState:
    """Current state of correlation engine.

    Attributes:
        correlation_matrix: Dict of pair -> (pair -> correlation)
        avg_correlation: Average correlation across all pairs
        regime: Current regime (RISK_OFF, RISK_ON, NEUTRAL)
        observation_count: Number of observations processed
        pairs_tracked: List of pairs being tracked
        last_updated: ISO timestamp of last update
    """

    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    avg_correlation: float = 0.0
    regime: str = NEUTRAL
    observation_count: int = 0
    pairs_tracked: List[str] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "correlation_matrix": self.correlation_matrix,
            "avg_correlation": float(self.avg_correlation),
            "regime": self.regime,
            "observation_count": self.observation_count,
            "pairs_tracked": self.pairs_tracked,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> CorrelationState:
        """Create from dictionary."""
        return cls(
            correlation_matrix=data.get("correlation_matrix", {}),
            avg_correlation=float(data.get("avg_correlation", 0.0)),
            regime=data.get("regime", NEUTRAL),
            observation_count=int(data.get("observation_count", 0)),
            pairs_tracked=list(data.get("pairs_tracked", [])),
            last_updated=data.get("last_updated", ""),
        )


class EWMACorrelationEngine:
    """EWMA correlation engine for dynamic portfolio risk management.

    Tracks correlation between FX pairs using exponentially-weighted moving average.
    Updates covariance matrix with each price observation and converts to correlation.
    Thread-safe with atomic JSON persistence.
    """

    def __init__(
        self,
        config: Optional[EWMACorrelationConfig] = None,
        pairs: Optional[List[str]] = None,
    ):
        """Initialize EWMA correlation engine.

        Args:
            config: Configuration (uses defaults if None)
            pairs: List of pairs to track (grows dynamically if None)
        """
        self.config = config or EWMACorrelationConfig()
        self._lock = threading.RLock()

        # Initialize pair tracking
        self._pairs: List[str] = list(pairs) if pairs else []
        self._pair_idx: Dict[str, int] = {}
        self._update_pair_index()

        # Internal state
        self._observation_count = 0
        self._means: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._avg_correlation: float = 0.0
        self._regime: str = NEUTRAL

    def _update_pair_index(self) -> None:
        """Update the pair index mapping."""
        self._pair_idx = {pair: i for i, pair in enumerate(self._pairs)}

    def _ensure_pair_exists(self, pair: str) -> None:
        """Ensure a pair exists in our tracking list."""
        if pair in self._pair_idx:
            return

        if len(self._pairs) >= self.config.max_pairs:
            logger.warning(
                f"Max pairs ({self.config.max_pairs}) reached, skipping new pair {pair}"
            )
            return

        self._pairs.append(pair)
        self._update_pair_index()
        logger.debug(f"Added pair {pair}, now tracking {len(self._pairs)} pairs")

    def _resize_matrices(self, n: int) -> None:
        """Resize covariance and means matrices to handle n pairs."""
        if self._means is not None and len(self._means) == n:
            return

        self._means = np.zeros(n)
        self._covariance = np.zeros((n, n))

    def update(self, returns_dict: Dict[str, float]) -> CorrelationState:
        """Update EWMA covariance with new returns observation.

        Args:
            returns_dict: Dict of pair_name -> log return for this period

        Returns:
            Updated CorrelationState
        """
        with self._lock:
            # Ensure all pairs exist
            for pair in returns_dict.keys():
                self._ensure_pair_exists(pair)

            if not self._pairs:
                logger.warning("No pairs to update")
                return self._get_state()

            # Ensure matrices are sized correctly
            n = len(self._pairs)
            self._resize_matrices(n)

            # Build returns vector
            returns_vec = np.zeros(n)
            for pair, ret in returns_dict.items():
                if pair in self._pair_idx:
                    idx = self._pair_idx[pair]
                    returns_vec[idx] = float(ret) if not np.isnan(ret) else 0.0

            # Update running means: μₜ = λ·μₜ₋₁ + (1-λ)·rₜ
            self._means = (
                self.config.lambda_decay * self._means
                + (1 - self.config.lambda_decay) * returns_vec
            )

            # Compute demeaned returns: dₜ = rₜ - μₜ
            demeaned = returns_vec - self._means

            # Update covariance: Σₜ = λ·Σₜ₋₁ + (1-λ)·dₜ·dₜᵀ
            rank1_cov = np.outer(demeaned, demeaned)
            self._covariance = (
                self.config.lambda_decay * self._covariance
                + (1 - self.config.lambda_decay) * rank1_cov
            )

            # Increment observation count
            self._observation_count += 1

            # Update correlation matrix
            self._update_correlation_matrix()

            # Detect regime
            self._regime = self.detect_correlation_regime()

            logger.debug(
                f"Updated EWMA: {self._observation_count} observations, "
                f"avg_corr={self._avg_correlation:.3f}, regime={self._regime}"
            )

            return self._get_state()

    def _update_correlation_matrix(self) -> None:
        """Convert covariance matrix to correlation matrix."""
        self._correlation_matrix = {}

        if self._covariance is None or self._pairs is None:
            return

        n = len(self._pairs)

        # Convert covariance to correlation: ρᵢⱼ = Σᵢⱼ / √(Σᵢᵢ·Σⱼⱼ)
        for i, pair_i in enumerate(self._pairs):
            self._correlation_matrix[pair_i] = {}

            var_i = self._covariance[i, i]
            # H-8 fix: Clamp to small positive to prevent NaN from sqrt of negative
            var_i = max(float(var_i), 1e-8)
            if var_i <= 1e-8:
                # Effectively zero variance - set all correlations to 0
                for j, pair_j in enumerate(self._pairs):
                    self._correlation_matrix[pair_i][pair_j] = (
                        1.0 if i == j else 0.0
                    )
                continue

            sqrt_var_i = np.sqrt(var_i)

            for j, pair_j in enumerate(self._pairs):
                var_j = self._covariance[j, j]
                # H-8 fix: Clamp to prevent NaN from negative floating-point variance
                var_j = max(float(var_j), 1e-8)
                if var_j <= 1e-8:
                    self._correlation_matrix[pair_i][pair_j] = (
                        1.0 if i == j else 0.0
                    )
                    continue

                sqrt_var_j = np.sqrt(var_j)
                denom = sqrt_var_i * sqrt_var_j

                if denom <= 0:
                    self._correlation_matrix[pair_i][pair_j] = (
                        1.0 if i == j else 0.0
                    )
                else:
                    corr = self._covariance[i, j] / denom
                    # Clamp to [-1, 1] due to numerical errors
                    corr = np.clip(corr, -1.0, 1.0)
                    self._correlation_matrix[pair_i][pair_j] = float(corr)

        # Calculate average correlation (excluding diagonals)
        if n > 1:
            total_corr = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total_corr += abs(self._correlation_matrix[self._pairs[i]][
                        self._pairs[j]
                    ])
                    count += 1

            self._avg_correlation = total_corr / count if count > 0 else 0.0
        else:
            self._avg_correlation = 0.0

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get current correlation matrix as nested dict.

        Returns:
            Dict mapping pair -> (pair -> correlation)
        """
        with self._lock:
            return {
                pair: dict(corr_dict)
                for pair, corr_dict in self._correlation_matrix.items()
            }

    def get_correlation(self, pair_a: str, pair_b: str) -> Optional[float]:
        """Get correlation between two specific pairs.

        Args:
            pair_a: First pair name
            pair_b: Second pair name

        Returns:
            Correlation value or None if pair not found
        """
        with self._lock:
            if pair_a not in self._correlation_matrix:
                return None
            return self._correlation_matrix[pair_a].get(pair_b)

    def diversification_multiplier(
        self, pair: str, portfolio_pairs: List[str]
    ) -> float:
        """Calculate diversification multiplier for a pair.

        Formula: D = 1 / sqrt(1 + avg_correlation_to_portfolio)

        Args:
            pair: Pair to calculate multiplier for
            portfolio_pairs: List of pairs already in portfolio

        Returns:
            Multiplier in [0.5, 1.0] — lower when highly correlated to portfolio
        """
        with self._lock:
            if not portfolio_pairs:
                return 1.0

            if pair not in self._correlation_matrix:
                return 1.0

            # Calculate average correlation to portfolio
            total_corr = 0.0
            count = 0

            for port_pair in portfolio_pairs:
                if port_pair == pair:
                    continue
                corr = self.get_correlation(pair, port_pair)
                if corr is not None:
                    total_corr += abs(corr)
                    count += 1

            if count == 0:
                return 1.0

            avg_corr = total_corr / count

            # Apply formula: D = 1 / sqrt(1 + avg_corr)
            multiplier = 1.0 / np.sqrt(1.0 + avg_corr)

            # Clamp to [0.5, 1.0]
            return float(np.clip(multiplier, 0.5, 1.0))

    def detect_correlation_regime(self) -> str:
        """Detect current correlation regime.

        Returns:
            "RISK_OFF" if avg correlation > risk_off_threshold
            "RISK_ON" if avg correlation < risk_on_threshold
            "NEUTRAL" otherwise
        """
        with self._lock:
            if self._observation_count < self.config.min_observations:
                return NEUTRAL

            if self._avg_correlation > self.config.risk_off_threshold:
                return RISK_OFF
            elif self._avg_correlation < self.config.risk_on_threshold:
                return RISK_ON
            else:
                return NEUTRAL

    def get_correlated_pairs(
        self, pair: str, threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Get pairs correlated above threshold with given pair.

        Args:
            pair: Pair to find correlations for
            threshold: Correlation threshold (uses config default if None)

        Returns:
            List of (pair_name, correlation) tuples, sorted by correlation descending
        """
        with self._lock:
            threshold = (
                threshold if threshold is not None
                else self.config.high_correlation_threshold
            )

            if pair not in self._correlation_matrix:
                return []

            correlated = []
            for other_pair, corr in self._correlation_matrix[pair].items():
                if other_pair != pair and abs(corr) >= threshold:
                    correlated.append((other_pair, corr))

            # Sort by absolute correlation descending
            correlated.sort(key=lambda x: abs(x[1]), reverse=True)
            return correlated

    def get_state(self) -> CorrelationState:
        """Get current correlation state.

        Returns:
            CorrelationState with current values
        """
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CorrelationState:
        """Internal version without locking (assumes lock is held)."""
        return CorrelationState(
            correlation_matrix=self.get_correlation_matrix(),
            avg_correlation=self._avg_correlation,
            regime=self._regime,
            observation_count=self._observation_count,
            pairs_tracked=list(self._pairs),
            last_updated=datetime.utcnow().isoformat(),
        )

    def save_state(self, path: str) -> None:
        """Save state atomically via safe_json_write (fcntl + tmp + fsync + rename).

        Args:
            path: File path to save to
        """
        with self._lock:
            state = self._get_state()
            data = state.to_dict()

            # Add version field for forward compatibility
            data["_version"] = "1.0"

            if not safe_json_write(path, data, sort_keys=True):
                msg = f"Failed to save EWMA correlation state to {path}"
                logger.error(msg)
                raise IOError(msg)
            logger.debug(f"Saved EWMA correlation state to {path}")

    def load_state(self, path: str) -> None:
        """Load state from JSON file.

        Args:
            path: File path to load from
        """
        with self._lock:
            try:
                if not os.path.exists(path):
                    logger.debug(f"State file {path} not found, starting fresh")
                    return

                with open(path, "r") as f:
                    data = json.load(f)

                # Validate structure
                if "correlation_matrix" not in data:
                    logger.warning(f"Invalid state file {path}, missing correlation_matrix")
                    return

                state = CorrelationState.from_dict(data)

                # Restore state
                self._pairs = state.pairs_tracked
                self._update_pair_index()
                self._correlation_matrix = state.correlation_matrix
                self._avg_correlation = state.avg_correlation
                self._regime = state.regime
                self._observation_count = state.observation_count

                # Reconstruct covariance matrix if possible
                n = len(self._pairs)
                self._resize_matrices(n)

                logger.debug(
                    f"Loaded EWMA correlation state from {path}: "
                    f"{self._observation_count} observations, "
                    f"{len(self._pairs)} pairs"
                )

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse state file {path}: {e}")
                # Gracefully continue with empty state
            except Exception as e:
                logger.error(f"Failed to load EWMA correlation state: {e}")
                raise

    def reset(self) -> None:
        """Reset all state to initial."""
        with self._lock:
            self._pairs = []
            self._pair_idx = {}
            self._observation_count = 0
            self._means = None
            self._covariance = None
            self._correlation_matrix = {}
            self._avg_correlation = 0.0
            self._regime = NEUTRAL
            logger.debug("Reset EWMA correlation engine")


# Factory functions
def create_default_ewma_engine(
    pairs: Optional[List[str]] = None,
) -> EWMACorrelationEngine:
    """Create EWMA engine with default FX settings (λ=0.94).

    Args:
        pairs: Optional list of initial pairs to track

    Returns:
        Configured EWMACorrelationEngine
    """
    config = EWMACorrelationConfig(lambda_decay=0.94)
    return EWMACorrelationEngine(config=config, pairs=pairs)


def create_conservative_ewma_engine(
    pairs: Optional[List[str]] = None,
) -> EWMACorrelationEngine:
    """Create conservative engine (λ=0.97, higher thresholds).

    More responsive to recent changes, slower drift detection.

    Args:
        pairs: Optional list of initial pairs to track

    Returns:
        Configured EWMACorrelationEngine
    """
    config = EWMACorrelationConfig(
        lambda_decay=0.97,
        risk_off_threshold=0.65,
        risk_on_threshold=0.25,
    )
    return EWMACorrelationEngine(config=config, pairs=pairs)


def create_responsive_ewma_engine(
    pairs: Optional[List[str]] = None,
) -> EWMACorrelationEngine:
    """Create responsive engine (λ=0.90, faster adaptation).

    More sensitive to recent changes, faster regime transitions.

    Args:
        pairs: Optional list of initial pairs to track

    Returns:
        Configured EWMACorrelationEngine
    """
    config = EWMACorrelationConfig(
        lambda_decay=0.90,
        risk_off_threshold=0.55,
        risk_on_threshold=0.35,
    )
    return EWMACorrelationEngine(config=config, pairs=pairs)
