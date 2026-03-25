"""Cross-pair lead-lag detection and entry timing.

US-087: Identify which FX pairs lead others using lagged correlation
analysis. EUR_USD often leads GBP_USD by 1-5 bars. When a lead pair
generates a strong signal, prepare entries on lagging pairs with timing
offset. Recompute lead-lag matrix periodically. Based on DeltaLag 2024
research (neural lagged correlation).

Algorithm:
1. compute_lead_lag(pair1_returns, pair2_returns, max_lag) — cross-correlate
   at each lag and pick optimal offset + correlation strength.
2. build_lead_lag_matrix(price_data_dict) — NxN matrix of all pair
   relationships.
3. get_lagging_signals(leader_pair, leader_direction, leader_confidence)
   — returns predicted moves on lagging pairs with timing offsets.
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
_LEAD_LAG_PATH = _PROJECT_ROOT / "trained_data" / "lead_lag_matrix.json"

# Minimum correlation to consider a lead-lag relationship meaningful
MIN_CORRELATION = 0.25
# Maximum lag (bars) to test
MAX_LAG_DEFAULT = 10
# Minimum observations for reliable correlation
MIN_OBSERVATIONS = 50
# Recompute matrix after this many scans
RECOMPUTE_INTERVAL = 100


@dataclass
class LeadLagRelationship:
    """A single lead-lag relationship between two pairs."""

    leader: str = ""
    lagger: str = ""
    optimal_lag: int = 0       # Positive = leader leads by N bars
    correlation: float = 0.0   # Strength of lagged correlation
    confidence: float = 0.0    # How reliable this relationship is (0-1)

    def to_dict(self) -> dict:
        return {
            "leader": self.leader,
            "lagger": self.lagger,
            "optimal_lag": self.optimal_lag,
            "correlation": round(self.correlation, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class LaggingSignal:
    """Predicted signal on a lagging pair based on leader movement."""

    pair: str = ""
    predicted_direction: str = ""   # LONG or SHORT
    expected_lag_bars: int = 0      # How many bars until the move
    signal_strength: float = 0.0    # 0-1 confidence in the prediction
    leader_pair: str = ""
    leader_direction: str = ""
    leader_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "predicted_direction": self.predicted_direction,
            "expected_lag_bars": self.expected_lag_bars,
            "signal_strength": round(self.signal_strength, 4),
            "leader_pair": self.leader_pair,
            "leader_direction": self.leader_direction,
            "leader_confidence": round(self.leader_confidence, 4),
        }


@dataclass
class LeadLagMatrix:
    """Full NxN lead-lag matrix result."""

    pairs: List[str] = field(default_factory=list)
    relationships: List[LeadLagRelationship] = field(default_factory=list)
    computed_at: str = ""
    observation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "pairs": self.pairs,
            "relationships": [r.to_dict() for r in self.relationships],
            "computed_at": self.computed_at,
            "observation_count": self.observation_count,
        }

    def get_leaders_for(self, pair: str) -> List[LeadLagRelationship]:
        """Get all pairs that lead the given pair."""
        return [r for r in self.relationships if r.lagger == pair]

    def get_laggers_for(self, pair: str) -> List[LeadLagRelationship]:
        """Get all pairs that lag behind the given pair."""
        return [r for r in self.relationships if r.leader == pair]


class LeadLagDetector:
    """Detect cross-pair lead-lag relationships for entry timing.

    Uses lagged cross-correlation to identify which FX pairs tend to
    move before others. When a leader pair generates a strong signal,
    the detector predicts moves on lagging pairs with timing offsets.
    """

    def __init__(
        self,
        max_lag: int = MAX_LAG_DEFAULT,
        min_correlation: float = MIN_CORRELATION,
        min_observations: int = MIN_OBSERVATIONS,
        persistence_path: Optional[Path] = None,
    ):
        self.max_lag = max_lag
        self.min_correlation = min_correlation
        self.min_observations = min_observations
        self.persistence_path = persistence_path or _LEAD_LAG_PATH

        self._matrix: Optional[LeadLagMatrix] = None
        self._scan_counter: int = 0

        # Try to load persisted matrix
        self._load_matrix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_lead_lag(
        self,
        pair1_returns: np.ndarray,
        pair2_returns: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> Tuple[int, float]:
        """Compute optimal lead-lag between two return series.

        Uses normalized cross-correlation at each lag offset.

        Args:
            pair1_returns: Return series for pair 1
            pair2_returns: Return series for pair 2
            max_lag: Maximum lag to test (uses default if None)

        Returns:
            (optimal_lag, correlation) — positive lag means pair1 leads pair2
        """
        lag_limit = max_lag or self.max_lag
        n = min(len(pair1_returns), len(pair2_returns))

        if n < self.min_observations:
            return 0, 0.0

        r1 = np.array(pair1_returns[-n:], dtype=np.float64)
        r2 = np.array(pair2_returns[-n:], dtype=np.float64)

        # Normalize
        r1_std = np.std(r1)
        r2_std = np.std(r2)
        if r1_std < 1e-10 or r2_std < 1e-10:
            return 0, 0.0

        r1_norm = (r1 - np.mean(r1)) / r1_std
        r2_norm = (r2 - np.mean(r2)) / r2_std

        best_lag = 0
        best_corr = 0.0

        for lag in range(-lag_limit, lag_limit + 1):
            if lag == 0:
                corr = float(np.mean(r1_norm * r2_norm))
            elif lag > 0:
                # pair1 leads pair2 by `lag` bars
                if n - lag < 10:
                    continue
                corr = float(np.mean(r1_norm[:-lag] * r2_norm[lag:]))
            else:
                # pair2 leads pair1 by `|lag|` bars
                abs_lag = abs(lag)
                if n - abs_lag < 10:
                    continue
                corr = float(np.mean(r1_norm[abs_lag:] * r2_norm[:-abs_lag]))

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag, best_corr

    def build_lead_lag_matrix(
        self,
        price_data_dict: Dict[str, np.ndarray],
    ) -> LeadLagMatrix:
        """Build full NxN lead-lag matrix from price return data.

        Args:
            price_data_dict: {pair_name: close_price_array} for all pairs

        Returns:
            LeadLagMatrix with all significant relationships
        """
        pairs = sorted(price_data_dict.keys())
        relationships: List[LeadLagRelationship] = []

        # Compute returns for each pair
        returns_dict: Dict[str, np.ndarray] = {}
        for pair, prices in price_data_dict.items():
            if len(prices) < self.min_observations + 1:
                continue
            p = np.array(prices, dtype=np.float64)
            ret = np.diff(p) / np.where(p[:-1] == 0, 1.0, p[:-1])
            returns_dict[pair] = ret

        valid_pairs = sorted(returns_dict.keys())
        obs_count = min(len(r) for r in returns_dict.values()) if returns_dict else 0

        # Pairwise cross-correlation
        for i, p1 in enumerate(valid_pairs):
            for j, p2 in enumerate(valid_pairs):
                if i >= j:
                    continue  # Skip self and duplicates

                lag, corr = self.compute_lead_lag(
                    returns_dict[p1], returns_dict[p2]
                )

                if abs(corr) < self.min_correlation:
                    continue

                # Determine leader/lagger from lag sign
                if lag > 0:
                    leader, lagger = p1, p2
                    opt_lag = lag
                elif lag < 0:
                    leader, lagger = p2, p1
                    opt_lag = abs(lag)
                else:
                    # Simultaneous: skip (not useful for entry timing)
                    continue

                # Confidence based on correlation strength and observation count
                conf = min(1.0, abs(corr) * (obs_count / 200.0))

                relationships.append(LeadLagRelationship(
                    leader=leader,
                    lagger=lagger,
                    optimal_lag=opt_lag,
                    correlation=abs(corr),
                    confidence=conf,
                ))

        # Sort by correlation strength (strongest first)
        relationships.sort(key=lambda r: r.correlation, reverse=True)

        self._matrix = LeadLagMatrix(
            pairs=valid_pairs,
            relationships=relationships,
            computed_at=datetime.now(timezone.utc).isoformat(),
            observation_count=obs_count,
        )

        logger.info(
            f"LeadLag: Built matrix with {len(relationships)} relationships "
            f"across {len(valid_pairs)} pairs ({obs_count} observations)"
        )

        return self._matrix

    def get_lagging_signals(
        self,
        leader_pair: str,
        leader_direction: str,
        leader_confidence: float = 0.5,
    ) -> List[LaggingSignal]:
        """Get predicted signals on lagging pairs based on leader movement.

        Args:
            leader_pair: The pair that generated the signal
            leader_direction: LONG or SHORT
            leader_confidence: Confidence of the leader signal (0-1)

        Returns:
            List of LaggingSignals for pairs expected to follow
        """
        if self._matrix is None:
            return []

        laggers = self._matrix.get_laggers_for(leader_pair)
        if not laggers:
            return []

        signals: List[LaggingSignal] = []
        for rel in laggers:
            # Positive correlation: lagger moves same direction
            # Negative correlation would be inverse (rare for FX majors)
            if rel.correlation > 0:
                predicted_dir = leader_direction
            else:
                predicted_dir = "SHORT" if leader_direction == "LONG" else "LONG"

            signal_strength = min(1.0, leader_confidence * rel.confidence * rel.correlation)

            # Only report signals above a minimum threshold
            if signal_strength < 0.15:
                continue

            signals.append(LaggingSignal(
                pair=rel.lagger,
                predicted_direction=predicted_dir,
                expected_lag_bars=rel.optimal_lag,
                signal_strength=signal_strength,
                leader_pair=leader_pair,
                leader_direction=leader_direction,
                leader_confidence=leader_confidence,
            ))

        # Sort by signal strength
        signals.sort(key=lambda s: s.signal_strength, reverse=True)

        if signals:
            logger.info(
                f"LeadLag: {leader_pair} {leader_direction} → "
                f"{len(signals)} lagging signals "
                f"(top: {signals[0].pair} lag={signals[0].expected_lag_bars} "
                f"strength={signals[0].signal_strength:.2f})"
            )

        return signals

    def should_recompute(self) -> bool:
        """Check if the matrix needs recomputation."""
        self._scan_counter += 1
        if self._matrix is None:
            return True
        return self._scan_counter % RECOMPUTE_INTERVAL == 0

    def get_matrix(self) -> Optional[LeadLagMatrix]:
        """Get the current lead-lag matrix."""
        return self._matrix

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_matrix(self) -> None:
        """Persist the lead-lag matrix to disk."""
        if self._matrix is None:
            return
        try:
            data = {
                "version": 1,
                "matrix": self._matrix.to_dict(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
            logger.debug(
                f"LeadLag: Saved matrix with {len(self._matrix.relationships)} "
                f"relationships to {self.persistence_path}"
            )
        except Exception as e:
            logger.debug(f"LeadLag: Failed to save matrix: {e}")

    def _load_matrix(self) -> None:
        """Load persisted lead-lag matrix."""
        if not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text())
            m = data.get("matrix", {})
            relationships = []
            for r in m.get("relationships", []):
                relationships.append(LeadLagRelationship(
                    leader=r.get("leader", ""),
                    lagger=r.get("lagger", ""),
                    optimal_lag=r.get("optimal_lag", 0),
                    correlation=r.get("correlation", 0.0),
                    confidence=r.get("confidence", 0.0),
                ))
            self._matrix = LeadLagMatrix(
                pairs=m.get("pairs", []),
                relationships=relationships,
                computed_at=m.get("computed_at", ""),
                observation_count=m.get("observation_count", 0),
            )
            logger.debug(
                f"LeadLag: Loaded matrix with {len(relationships)} relationships"
            )
        except Exception as e:
            logger.debug(f"LeadLag: Failed to load matrix: {e}")
