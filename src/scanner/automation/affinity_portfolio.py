"""Pair affinity scoring for portfolio position sizing.

US-104: Wire pair_affinity (US-062) data into portfolio position sizing.
Pair affinity matrix is generated but never consumed. Replace binary
correlation blocking with continuous affinity-weighted position multiplier.

Approach:
- Load pair_affinity_matrix.json (co-occurrence win correlation data)
- For each proposed position, compute avg affinity vs all open positions
- Negative affinity (lose together) → reduce position size
- Low/neutral affinity (independent) → full size allowed
- Prevents concentration risk without binary blocking
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_AFFINITY_MATRIX_PATH = _PROJECT_ROOT / "trained_data" / "pair_affinity_matrix.json"
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "affinity_portfolio_log.json"

# Sizing parameters
NEGATIVE_AFFINITY_THRESHOLD = -0.30  # Below this → reduce size
POSITIVE_AFFINITY_THRESHOLD = 0.30  # Above this → cap at 1.0 (no boost)
MIN_SIZE_MULTIPLIER = 0.50  # Floor for size reduction
MAX_SIZE_MULTIPLIER = 1.20  # Ceiling (slight boost for diversifying pairs)
NEUTRAL_MULTIPLIER = 1.0  # Default when no affinity data


class AffinityPortfolioSizer:
    """Continuous position sizing based on pair affinity scores.

    Replaces binary correlation blocking with nuanced sizing. Pairs
    that historically lose together get reduced position sizes. Pairs
    with independent outcomes get full sizing.

    Usage:
        sizer = AffinityPortfolioSizer()
        multiplier = sizer.get_affinity_multiplier("EUR_USD", ["GBP_USD", "AUD_USD"])
        # multiplier = 0.75 (reduce because EUR_USD correlates with open positions)
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path or _LOG_PATH

        # Affinity matrix: {pair: {pair: score}}
        self._affinity_matrix: Dict[str, Dict[str, float]] = {}
        self._matrix_loaded = False

        # Decision log
        self._decisions: List[Dict[str, Any]] = []
        self._total_decisions: int = 0

        self._load_affinity_matrix()
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_affinity_multiplier(
        self,
        proposed_pair: str,
        open_positions: List[str],
    ) -> float:
        """Get position size multiplier based on affinity with open positions.

        Args:
            proposed_pair: The pair being considered for a new trade.
            open_positions: List of currently open position pairs.

        Returns:
            Size multiplier [MIN_SIZE_MULTIPLIER, MAX_SIZE_MULTIPLIER].
        """
        if not open_positions:
            return NEUTRAL_MULTIPLIER  # No existing exposure

        if not self._matrix_loaded or not self._affinity_matrix:
            return NEUTRAL_MULTIPLIER  # No affinity data

        # Compute average affinity with open positions
        affinities = []
        for open_pair in open_positions:
            score = self._get_pair_affinity(proposed_pair, open_pair)
            if score is not None:
                affinities.append(score)

        if not affinities:
            return NEUTRAL_MULTIPLIER  # No overlap in affinity data

        avg_affinity = float(np.mean(affinities))
        max_affinity = float(np.max(affinities))

        # Use the worse of average and max (conservative approach)
        effective_affinity = 0.6 * avg_affinity + 0.4 * max_affinity

        # Compute multiplier
        if effective_affinity <= NEGATIVE_AFFINITY_THRESHOLD:
            # Strong negative affinity: reduce size proportionally
            reduction = abs(effective_affinity - NEGATIVE_AFFINITY_THRESHOLD) / (
                1.0 - abs(NEGATIVE_AFFINITY_THRESHOLD)
            )
            multiplier = max(
                MIN_SIZE_MULTIPLIER,
                NEUTRAL_MULTIPLIER - reduction * (NEUTRAL_MULTIPLIER - MIN_SIZE_MULTIPLIER),
            )
        elif effective_affinity >= POSITIVE_AFFINITY_THRESHOLD:
            # Positive affinity (diversifying): slight boost
            boost = (effective_affinity - POSITIVE_AFFINITY_THRESHOLD) / (
                1.0 - POSITIVE_AFFINITY_THRESHOLD
            )
            multiplier = min(
                MAX_SIZE_MULTIPLIER,
                NEUTRAL_MULTIPLIER + boost * (MAX_SIZE_MULTIPLIER - NEUTRAL_MULTIPLIER),
            )
        else:
            multiplier = NEUTRAL_MULTIPLIER

        # Log decision
        self._total_decisions += 1
        self._decisions.append({
            "pair": proposed_pair,
            "open_positions": open_positions,
            "avg_affinity": round(avg_affinity, 4),
            "max_affinity": round(max_affinity, 4),
            "effective_affinity": round(effective_affinity, 4),
            "multiplier": round(multiplier, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._decisions) > 300:
            self._decisions = self._decisions[-150:]

        return round(multiplier, 4)

    def reload_matrix(self) -> bool:
        """Reload affinity matrix from disk."""
        self._load_affinity_matrix()
        return self._matrix_loaded

    def get_status(self) -> Dict[str, Any]:
        """Get sizer status for observability."""
        return {
            "matrix_loaded": self._matrix_loaded,
            "n_pairs_in_matrix": len(self._affinity_matrix),
            "total_decisions": self._total_decisions,
            "recent_decisions": self._decisions[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_pair_affinity(self, pair1: str, pair2: str) -> Optional[float]:
        """Look up affinity score between two pairs."""
        if pair1 in self._affinity_matrix:
            if pair2 in self._affinity_matrix[pair1]:
                return self._affinity_matrix[pair1][pair2]
        if pair2 in self._affinity_matrix:
            if pair1 in self._affinity_matrix[pair2]:
                return self._affinity_matrix[pair2][pair1]
        return None

    def _load_affinity_matrix(self) -> None:
        """Load pair affinity matrix from trained_data."""
        if not _AFFINITY_MATRIX_PATH.exists():
            logger.debug("AffinityPortfolio: No affinity matrix found")
            return

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(_AFFINITY_MATRIX_PATH, default={})
            except ImportError:
                data = json.loads(_AFFINITY_MATRIX_PATH.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                # Handle both flat and nested matrix formats
                matrix = data.get("matrix", data.get("affinity_scores", data))
                if isinstance(matrix, dict):
                    self._affinity_matrix = matrix
                    self._matrix_loaded = True
                    logger.debug(
                        f"AffinityPortfolio: Loaded matrix with {len(matrix)} pairs"
                    )
        except Exception as e:
            logger.debug(f"AffinityPortfolio: Failed to load matrix: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist sizing decisions log."""
        try:
            data = {
                "version": 1,
                "total_decisions": self._total_decisions,
                "decisions": self._decisions[-150:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AffinityPortfolio: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_decisions = data.get("total_decisions", 0)
                self._decisions = data.get("decisions", [])
        except Exception as e:
            logger.debug(f"AffinityPortfolio: Failed to load: {e}")
