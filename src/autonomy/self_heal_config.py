"""Self-healing configuration updater.

US-016: Detects performance degradation and autonomously adjusts
thresholds and risk parameters within safe bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SelfHealConfig:
    """Configuration for self-healing updater."""

    min_win_rate_threshold: float = 0.45
    max_drawdown_threshold: float = 0.05
    min_sharpe_threshold: float = 0.5
    win_rate_lookback: int = 20
    adjustment_step: float = 0.02
    max_adjustment: float = 0.10
    require_approval_for_major: bool = True


class SelfHealConfigUpdater:
    """Autonomous configuration updater with safety bounds.

    Monitors recent performance metrics and adjusts gate thresholds,
    risk parameters, and confidence floors within reversible bounds.
    Major changes (exceeding max_adjustment) require operator approval.
    """

    def __init__(self, config: Optional[SelfHealConfig] = None):
        self.cfg = config or SelfHealConfig()
        self._history: List[Dict[str, Any]] = []

    def update(
        self,
        current_config: Dict[str, Any],
        recent_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Propose configuration adjustments based on recent performance.

        Args:
            current_config: Current scanner configuration dict
            recent_trades: List of recent trade outcomes

        Returns:
            Dict with 'proposed_config', 'changes', 'requires_approval', 'reason'
        """
        if len(recent_trades) < self.cfg.win_rate_lookback:
            return {
                "proposed_config": current_config.copy(),
                "changes": [],
                "requires_approval": False,
                "reason": "insufficient_trade_history",
            }

        # Compute metrics
        wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(recent_trades)
        pnl_series = [t.get("pnl", 0) for t in recent_trades]
        cumulative = [sum(pnl_series[: i + 1]) for i in range(len(pnl_series))]
        running_max = [max(cumulative[: i + 1]) for i in range(len(cumulative))]
        drawdowns = [running_max[i] - cumulative[i] for i in range(len(cumulative))]
        max_dd = max(drawdowns) if drawdowns else 0.0

        mean_pnl = sum(pnl_series) / len(pnl_series)
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnl_series) / len(pnl_series)) ** 0.5
        if std_pnl > 0:
            sharpe = mean_pnl / std_pnl
        elif mean_pnl > 0:
            sharpe = float('inf')  # all identical positive returns = perfect Sharpe
        elif mean_pnl < 0:
            sharpe = float('-inf')  # all identical negative returns = terrible Sharpe
        else:
            sharpe = 0.0

        proposed = current_config.copy()
        changes: List[Dict[str, Any]] = []
        requires_approval = False
        reason = []

        # Win rate too low -> lower confidence threshold
        if win_rate < self.cfg.min_win_rate_threshold:
            old = proposed.get("min_confidence", 42.0)
            new = max(old - self.cfg.adjustment_step * 100, 30.0)
            if abs(new - old) > self.cfg.max_adjustment * 100:
                requires_approval = True
            proposed["min_confidence"] = new
            changes.append({"param": "min_confidence", "old": old, "new": new})
            reason.append(f"win_rate={win_rate:.2f} below threshold")

        # Drawdown too high -> reduce risk per trade
        if max_dd > self.cfg.max_drawdown_threshold:
            old = proposed.get("risk_per_trade_pct", 0.05)
            new = max(old - self.cfg.adjustment_step, 0.01)
            if abs(new - old) > self.cfg.max_adjustment:
                requires_approval = True
            proposed["risk_per_trade_pct"] = new
            changes.append({"param": "risk_per_trade_pct", "old": old, "new": new})
            reason.append(f"max_dd={max_dd:.4f} above threshold")

        # Sharpe too low -> raise gate floor (but ignore infinite negative)
        if sharpe < self.cfg.min_sharpe_threshold and sharpe != float('-inf'):
            old = proposed.get("compound_gate_floor", 0.1)
            new = min(old + self.cfg.adjustment_step, 0.3)
            if abs(new - old) > self.cfg.max_adjustment:
                requires_approval = True
            proposed["compound_gate_floor"] = new
            changes.append({"param": "compound_gate_floor", "old": old, "new": new})
            reason.append(f"sharpe={sharpe:.2f} below threshold")

        if self.cfg.require_approval_for_major and requires_approval:
            logger.warning("SelfHeal: major changes proposed, awaiting approval: %s", changes)

        self._history.append({
            "win_rate": win_rate,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "changes": changes,
        })

        return {
            "proposed_config": proposed,
            "changes": changes,
            "requires_approval": requires_approval,
            "reason": "; ".join(reason) if reason else "metrics_healthy",
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history.copy()
