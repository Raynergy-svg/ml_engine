"""Portfolio-level Kelly criterion with VaR constraints.

US-084: Current position sizing treats each trade independently.
Implements portfolio-level Kelly criterion that considers all open
positions, their correlations, and overall portfolio risk.

Key features:
- Full Kelly and Half-Kelly computation
- Portfolio VaR constraint: max 15% of NAV at 95% confidence
- Correlation-aware sizing: reduces size when new trade correlates
  with existing positions
- Uses historical trade stats (win_rate, avg_win, avg_loss) from journal

Safety: Half-Kelly is the production default (full Kelly is too
aggressive for real trading due to estimation error in parameters).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
_KELLY_PATH = _PROJECT_ROOT / "trained_data" / "kelly_portfolio.json"

# Default constraints
MAX_PORTFOLIO_RISK_PCT = 0.15  # 15% of NAV
VAR_CONFIDENCE = 0.95  # 95% VaR
MAX_SINGLE_POSITION_PCT = 0.05  # 5% max per position
MIN_TRADES_FOR_KELLY = 20  # Need history for reliable stats

# Static FX pair correlations (fallback when not enough data)
STATIC_CORRELATIONS: Dict[str, Dict[str, float]] = {
    "EUR_USD": {"GBP_USD": 0.85, "AUD_USD": 0.70, "NZD_USD": 0.65, "USD_CHF": -0.90, "USD_JPY": -0.30, "USD_CAD": -0.50},
    "GBP_USD": {"EUR_USD": 0.85, "AUD_USD": 0.60, "NZD_USD": 0.55, "USD_CHF": -0.80, "USD_JPY": -0.25, "USD_CAD": -0.45},
    "AUD_USD": {"EUR_USD": 0.70, "GBP_USD": 0.60, "NZD_USD": 0.90, "USD_CHF": -0.60, "USD_JPY": 0.10, "USD_CAD": -0.50},
    "NZD_USD": {"EUR_USD": 0.65, "GBP_USD": 0.55, "AUD_USD": 0.90, "USD_CHF": -0.55, "USD_JPY": 0.05, "USD_CAD": -0.45},
    "USD_JPY": {"EUR_USD": -0.30, "GBP_USD": -0.25, "AUD_USD": 0.10, "NZD_USD": 0.05, "USD_CHF": 0.50, "USD_CAD": 0.40},
    "USD_CHF": {"EUR_USD": -0.90, "GBP_USD": -0.80, "AUD_USD": -0.60, "NZD_USD": -0.55, "USD_JPY": 0.50, "USD_CAD": 0.50},
    "USD_CAD": {"EUR_USD": -0.50, "GBP_USD": -0.45, "AUD_USD": -0.50, "NZD_USD": -0.45, "USD_JPY": 0.40, "USD_CHF": 0.50},
}


@dataclass
class KellyResult:
    """Result of Kelly criterion computation."""

    full_kelly_fraction: float = 0.0
    half_kelly_fraction: float = 0.0
    recommended_risk_pct: float = 0.0
    recommended_lots: float = 0.0
    portfolio_risk_used_pct: float = 0.0
    portfolio_risk_remaining_pct: float = 0.0
    correlation_reduction: float = 1.0  # Multiplier for correlation
    var_constrained: bool = False
    win_rate: float = 0.0
    avg_win_loss_ratio: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "full_kelly_fraction": round(self.full_kelly_fraction, 6),
            "half_kelly_fraction": round(self.half_kelly_fraction, 6),
            "recommended_risk_pct": round(self.recommended_risk_pct, 6),
            "recommended_lots": round(self.recommended_lots, 4),
            "portfolio_risk_used_pct": round(self.portfolio_risk_used_pct, 4),
            "portfolio_risk_remaining_pct": round(self.portfolio_risk_remaining_pct, 4),
            "correlation_reduction": round(self.correlation_reduction, 4),
            "var_constrained": self.var_constrained,
            "win_rate": round(self.win_rate, 4),
            "avg_win_loss_ratio": round(self.avg_win_loss_ratio, 4),
            "reason": self.reason,
        }


class KellyPortfolioOptimizer:
    """Portfolio-level Kelly criterion optimizer.

    Pipeline:
    1. Compute win_rate and avg_win/avg_loss from trade journal
    2. Calculate full Kelly: f* = (p*b - q) / b
    3. Apply Half-Kelly for production safety
    4. Check portfolio VaR constraint
    5. Apply correlation reduction for correlated positions
    6. Return recommended position size
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        persistence_path: Optional[Path] = None,
        max_portfolio_risk: float = MAX_PORTFOLIO_RISK_PCT,
        var_confidence: float = VAR_CONFIDENCE,
        max_single_position: float = MAX_SINGLE_POSITION_PCT,
        use_half_kelly: bool = True,
    ):
        self.journal_path = journal_path or _JOURNAL_PATH
        self.persistence_path = persistence_path or _KELLY_PATH
        self.max_portfolio_risk = max_portfolio_risk
        self.var_confidence = var_confidence
        self.max_single_position = max_single_position
        self.use_half_kelly = use_half_kelly

        # Cached stats
        self._win_rate: float = 0.0
        self._avg_win_pips: float = 0.0
        self._avg_loss_pips: float = 0.0
        self._n_trades: int = 0
        self._stats_initialized: bool = False

        # Load stats from journal
        self._compute_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_kelly(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """Compute full Kelly fraction.

        f* = (p * b - q) / b
        where:
            p = win probability
            q = 1 - p (loss probability)
            b = avg_win / avg_loss (win/loss ratio)

        Returns:
            Kelly fraction (can be negative = don't trade)
        """
        p = win_rate if win_rate is not None else self._win_rate
        w = avg_win if avg_win is not None else self._avg_win_pips
        l_val = avg_loss if avg_loss is not None else self._avg_loss_pips

        if l_val <= 0 or p <= 0 or p >= 1:
            return 0.0

        b = w / l_val  # Win/loss ratio
        q = 1.0 - p

        kelly = (p * b - q) / b
        return kelly

    def compute_half_kelly(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> float:
        """Compute half-Kelly (conservative production sizing)."""
        return self.compute_kelly(win_rate, avg_win, avg_loss) / 2.0

    def portfolio_optimal_size(
        self,
        pair: str,
        direction: str,
        confidence: float,
        sl_pips: float,
        open_positions: Optional[List[Dict[str, Any]]] = None,
        nav: float = 10000.0,
        pip_value_per_lot: float = 10.0,
    ) -> KellyResult:
        """Compute portfolio-optimal position size.

        Args:
            pair: FX pair for the new trade
            direction: LONG or SHORT
            confidence: Model confidence for this trade
            sl_pips: Stop-loss in pips
            open_positions: List of currently open positions
            nav: Net Asset Value (account balance)
            pip_value_per_lot: Dollar value per pip per standard lot

        Returns:
            KellyResult with recommended sizing
        """
        result = KellyResult(
            win_rate=self._win_rate,
            avg_win_loss_ratio=(
                self._avg_win_pips / max(self._avg_loss_pips, 0.01)
                if self._avg_loss_pips > 0 else 0.0
            ),
        )

        if not self._stats_initialized:
            result.reason = "Insufficient trade history for Kelly sizing"
            result.recommended_risk_pct = self.max_single_position
            result.recommended_lots = self._risk_to_lots(
                self.max_single_position, nav, sl_pips, pip_value_per_lot
            )
            return result

        # Step 1: Compute Kelly fraction
        full_k = self.compute_kelly()
        half_k = full_k / 2.0
        result.full_kelly_fraction = full_k
        result.half_kelly_fraction = half_k

        if full_k <= 0:
            result.reason = "Negative Kelly edge — don't trade"
            result.recommended_risk_pct = 0.0
            result.recommended_lots = 0.0
            return result

        # Step 2: Use half-Kelly for production
        base_risk = half_k if self.use_half_kelly else full_k

        # Scale by confidence (higher confidence → closer to Kelly optimal)
        base_risk *= min(confidence / 0.60, 1.2)  # Boost slightly for high confidence

        # Step 3: Portfolio VaR constraint
        open_risk = self._compute_open_risk(open_positions or [], nav)
        result.portfolio_risk_used_pct = open_risk
        remaining_budget = self.max_portfolio_risk - open_risk
        result.portfolio_risk_remaining_pct = max(0.0, remaining_budget)

        if remaining_budget <= 0:
            result.reason = "Portfolio risk budget exhausted"
            result.recommended_risk_pct = 0.0
            result.recommended_lots = 0.0
            result.var_constrained = True
            return result

        if base_risk > remaining_budget:
            base_risk = remaining_budget
            result.var_constrained = True

        # Step 4: Correlation reduction
        corr_mult = self._correlation_reduction(pair, direction, open_positions or [])
        result.correlation_reduction = corr_mult
        base_risk *= corr_mult

        # Step 5: Cap at max single position
        base_risk = min(base_risk, self.max_single_position)
        base_risk = max(base_risk, 0.005)  # Floor at 0.5%

        result.recommended_risk_pct = base_risk
        result.recommended_lots = self._risk_to_lots(
            base_risk, nav, sl_pips, pip_value_per_lot
        )
        result.reason = (
            f"Half-Kelly={half_k:.4f}, "
            f"corr_adj={corr_mult:.2f}, "
            f"portfolio_used={open_risk:.2%}"
        )

        logger.info(
            f"KellyPortfolio [{pair}]: risk={base_risk:.4f} "
            f"({result.recommended_lots:.4f} lots), "
            f"kelly={half_k:.4f}, corr={corr_mult:.2f}"
        )

        # Persist snapshot
        self._save_snapshot(result, pair)

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_stats(self) -> None:
        """Load and compute win rate / avg win / avg loss from journal."""
        trades = self._load_trades()
        if not trades:
            return

        wins: List[float] = []
        losses: List[float] = []

        for t in trades:
            outcome = t.get("outcome", "").upper()
            pnl = float(t.get("pnl_pips", 0.0))

            if outcome in ("WIN", "WON", "TP_HIT") and pnl > 0:
                wins.append(pnl)
            elif outcome in ("LOSS", "LOST", "SL_HIT", "STOPPED") and pnl != 0:
                losses.append(abs(pnl))

        total = len(wins) + len(losses)
        if total < MIN_TRADES_FOR_KELLY:
            logger.debug(
                f"KellyPortfolio: Only {total} trades "
                f"(need {MIN_TRADES_FOR_KELLY})"
            )
            return

        self._win_rate = len(wins) / total
        self._avg_win_pips = float(np.mean(wins)) if wins else 0.0
        self._avg_loss_pips = float(np.mean(losses)) if losses else 1.0
        self._n_trades = total
        self._stats_initialized = True

        logger.info(
            f"KellyPortfolio: Stats from {total} trades — "
            f"win_rate={self._win_rate:.2%}, "
            f"avg_win={self._avg_win_pips:.1f}, "
            f"avg_loss={self._avg_loss_pips:.1f}"
        )

    def _compute_open_risk(
        self,
        open_positions: List[Dict[str, Any]],
        nav: float,
    ) -> float:
        """Compute total risk from open positions as % of NAV."""
        if not open_positions or nav <= 0:
            return 0.0

        total_risk = 0.0
        for pos in open_positions:
            lots = float(pos.get("lots", 0.0))
            sl_pips = float(pos.get("sl_pips", 10.0))
            pip_val = float(pos.get("pip_value_per_lot", 10.0))
            risk_usd = lots * sl_pips * pip_val
            total_risk += risk_usd

        return total_risk / nav

    def _correlation_reduction(
        self,
        new_pair: str,
        new_direction: str,
        open_positions: List[Dict[str, Any]],
    ) -> float:
        """Compute correlation-based size reduction.

        If the new trade is highly correlated with existing positions,
        reduce its size to avoid concentration risk.

        Returns:
            Multiplier 0.0-1.0 (lower = more reduction)
        """
        if not open_positions:
            return 1.0

        max_corr_impact = 0.0
        pair_corrs = STATIC_CORRELATIONS.get(new_pair, {})

        for pos in open_positions:
            pos_pair = pos.get("pair", "")
            pos_dir = pos.get("direction", "")

            corr = pair_corrs.get(pos_pair, 0.0)

            # Same direction + positive correlation = concentration
            # Opposite direction + negative correlation = also concentration
            if new_direction == pos_dir:
                effective_corr = corr
            else:
                effective_corr = -corr

            # High positive effective correlation = risk concentration
            if effective_corr > 0:
                max_corr_impact = max(max_corr_impact, effective_corr)

        # Reduction: 1.0 for uncorrelated, down to 0.3 for highly correlated
        reduction = max(0.3, 1.0 - max_corr_impact * 0.7)
        return reduction

    def _risk_to_lots(
        self,
        risk_pct: float,
        nav: float,
        sl_pips: float,
        pip_value_per_lot: float,
    ) -> float:
        """Convert risk percentage to lot size."""
        if sl_pips <= 0 or pip_value_per_lot <= 0:
            return 0.01

        risk_usd = nav * risk_pct
        lots = risk_usd / (sl_pips * pip_value_per_lot)
        return max(0.01, round(lots, 4))

    def _load_trades(self) -> List[Dict[str, Any]]:
        """Load trades from journal."""
        if not self.journal_path.exists():
            return []
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.journal_path, default=[])
            except ImportError:
                data = json.loads(self.journal_path.read_text())
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.debug(f"KellyPortfolio: Failed to load journal: {e}")
            return []

    def _save_snapshot(self, result: KellyResult, pair: str) -> None:
        """Persist portfolio risk snapshot."""
        try:
            data = {
                "version": 1,
                "pair": pair,
                "result": result.to_dict(),
                "stats": {
                    "win_rate": self._win_rate,
                    "avg_win_pips": self._avg_win_pips,
                    "avg_loss_pips": self._avg_loss_pips,
                    "n_trades": self._n_trades,
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"KellyPortfolio: Failed to save snapshot: {e}")
