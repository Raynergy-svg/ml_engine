"""Market impact model for position sizing.

US-092: Estimate and incorporate slippage/market impact into position
sizing using simplified Almgren-Chriss model. Computes permanent +
temporary impact from trade size, spread, and volume. Reduces position
size when estimated slippage exceeds threshold. Regime-aware impact
coefficients scale impact during low-liquidity periods.

Model:
- Permanent impact: a * (size / baseline_volume) ^ 1.5
- Temporary impact: 0.5 * spread * (size / daily_volume) ^ 0.5
- Total slippage = permanent + temporary (in pips)
- If slippage > max_slippage: reduce size by (max_slippage / slippage)

Based on 2024 Almgren-Chriss extensions with regime-dependent coefficients.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_IMPACT_LOG_PATH = _PROJECT_ROOT / "trained_data" / "market_impact_log.json"

# Default impact coefficients per regime
REGIME_IMPACT_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "LOW": {
        "permanent_coeff": 0.00005,
        "temporary_mult": 1.0,
    },
    "NORMAL": {
        "permanent_coeff": 0.0001,
        "temporary_mult": 1.0,
    },
    "HIGH": {
        "permanent_coeff": 0.0002,
        "temporary_mult": 1.5,
    },
    "EXTREME": {
        "permanent_coeff": 0.0005,
        "temporary_mult": 2.5,
    },
}

# Default baseline daily volume in lots (for FX majors)
DEFAULT_BASELINE_VOLUME = 5000.0
# Max acceptable slippage in pips
DEFAULT_MAX_SLIPPAGE = 1.5


@dataclass
class ImpactResult:
    """Result of market impact estimation."""

    permanent_impact_pips: float = 0.0
    temporary_impact_pips: float = 0.0
    total_impact_pips: float = 0.0
    original_size_lots: float = 0.0
    adjusted_size_lots: float = 0.0
    size_reduction_pct: float = 0.0
    regime: str = "NORMAL"
    exceeds_threshold: bool = False

    def to_dict(self) -> dict:
        return {
            "permanent_impact_pips": round(self.permanent_impact_pips, 4),
            "temporary_impact_pips": round(self.temporary_impact_pips, 4),
            "total_impact_pips": round(self.total_impact_pips, 4),
            "original_size_lots": round(self.original_size_lots, 4),
            "adjusted_size_lots": round(self.adjusted_size_lots, 4),
            "size_reduction_pct": round(self.size_reduction_pct, 2),
            "regime": self.regime,
            "exceeds_threshold": self.exceeds_threshold,
        }


class MarketImpactModel:
    """Estimate market impact and adjust position sizes.

    Uses simplified Almgren-Chriss framework with regime-dependent
    coefficients. Permanent impact scales with size^1.5 (square-root
    law), temporary impact scales with spread and size^0.5.
    """

    def __init__(
        self,
        max_slippage_pips: float = DEFAULT_MAX_SLIPPAGE,
        baseline_volume: float = DEFAULT_BASELINE_VOLUME,
        persistence_path: Optional[Path] = None,
    ):
        self.max_slippage = max_slippage_pips
        self.baseline_volume = baseline_volume
        self.persistence_path = persistence_path or _IMPACT_LOG_PATH

        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_impact(
        self,
        size_lots: float,
        spread_pips: float,
        daily_volume: Optional[float] = None,
        regime: str = "NORMAL",
    ) -> ImpactResult:
        """Estimate permanent and temporary market impact.

        Args:
            size_lots: Trade size in lots
            spread_pips: Current bid-ask spread in pips
            daily_volume: Estimated daily volume in lots (uses baseline if None)
            regime: Current volatility regime

        Returns:
            ImpactResult with impact breakdown
        """
        result = ImpactResult(
            original_size_lots=size_lots,
            adjusted_size_lots=size_lots,
            regime=regime,
        )

        vol = daily_volume or self.baseline_volume
        if vol <= 0 or size_lots <= 0:
            return result

        coeffs = REGIME_IMPACT_COEFFICIENTS.get(
            regime, REGIME_IMPACT_COEFFICIENTS["NORMAL"]
        )

        # Permanent impact: a * (size/volume)^1.5
        size_ratio = size_lots / vol
        result.permanent_impact_pips = (
            coeffs["permanent_coeff"] * (size_ratio ** 1.5) * 10000  # Convert to pips
        )

        # Temporary impact: 0.5 * spread * sqrt(size/volume) * regime_mult
        result.temporary_impact_pips = (
            0.5 * spread_pips * (size_ratio ** 0.5) * coeffs["temporary_mult"]
        )

        result.total_impact_pips = (
            result.permanent_impact_pips + result.temporary_impact_pips
        )

        return result

    def adjust_position_size(
        self,
        base_size_lots: float,
        spread_pips: float,
        daily_volume: Optional[float] = None,
        regime: str = "NORMAL",
    ) -> ImpactResult:
        """Estimate impact and reduce size if slippage exceeds threshold.

        Args:
            base_size_lots: Desired trade size in lots
            spread_pips: Current bid-ask spread
            daily_volume: Estimated daily volume
            regime: Current volatility regime

        Returns:
            ImpactResult with adjusted size
        """
        result = self.estimate_impact(
            base_size_lots, spread_pips, daily_volume, regime
        )

        if result.total_impact_pips > self.max_slippage:
            result.exceeds_threshold = True
            reduction_factor = self.max_slippage / result.total_impact_pips
            result.adjusted_size_lots = max(
                0.01, base_size_lots * reduction_factor
            )
            result.size_reduction_pct = (
                (1.0 - reduction_factor) * 100.0
            )

            logger.info(
                f"MarketImpact [{regime}]: Slippage {result.total_impact_pips:.2f} pips "
                f"> max {self.max_slippage:.1f}. Size {base_size_lots:.2f} "
                f"→ {result.adjusted_size_lots:.2f} lots "
                f"(-{result.size_reduction_pct:.0f}%)"
            )

        # Log
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
        })

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist impact history."""
        try:
            data = {
                "version": 1,
                "history": self._history[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"MarketImpact: Failed to save log: {e}")
