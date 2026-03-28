"""Online RL weight micro-updater.

Lightweight incremental weight adjustments from recent trade outcomes,
applied every N scan cycles without waiting for full RL sync.

US-060: Create online RL weight updater for tighter feedback loops.

Usage:
    updater = OnlineWeightUpdater()
    adjustments = updater.maybe_update(scan_cycle=current_cycle)
    # Returns list of adjustments made, or empty if not time yet
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRADE_JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
AGENT_WEIGHTS_PATH = _PROJECT_ROOT / "trained_data" / "models" / "agent_weights.json"
UPDATE_LOG_PATH = _PROJECT_ROOT / "trained_data" / "online_rl_updates.jsonl"

# Guard rails
MAX_ADJUSTMENT = 0.10  # Max ±10% per micro-update
MIN_WEIGHT = 0.05      # No agent weight below 5%
MAX_WEIGHT = 3.0        # No agent weight above 300%


@dataclass
class WeightAdjustment:
    """Record of a single weight micro-adjustment."""

    agent: str
    old_weight: float
    new_weight: float
    delta: float
    reason: str
    trade_count: int


class OnlineWeightUpdater:
    """Incremental RL weight updates from recent trade outcomes.

    Reads the last N entries from trade_journal_rl.json, computes
    per-agent win/loss signals, and applies small weight adjustments
    every `update_interval` scan cycles.

    Unlike the full RL sync (which runs on trade close), this runs
    proactively during the scan loop to tighten the feedback loop.
    """

    def __init__(
        self,
        update_interval: int = 5,
        lookback_trades: int = 10,
        learning_rate: float = 0.02,
        journal_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ):
        """Initialize online updater.

        Args:
            update_interval: Run update every N scan cycles
            lookback_trades: Number of recent trades to consider
            learning_rate: Base step size for weight adjustments
            journal_path: Path to trade journal (default: trained_data/trade_journal_rl.json)
            weights_path: Path to agent weights (default: trained_data/models/agent_weights.json)
            log_path: Path for update audit log
        """
        self.update_interval = update_interval
        self.lookback_trades = lookback_trades
        self.learning_rate = learning_rate
        self._base_learning_rate = learning_rate
        self.journal_path = journal_path or TRADE_JOURNAL_PATH
        self.weights_path = weights_path or AGENT_WEIGHTS_PATH
        self.log_path = log_path or UPDATE_LOG_PATH
        self._last_journal_size: int = 0  # Track journal size to detect new trades

        # US-091: Adaptive LR scheduler (optional)
        self._lr_scheduler = None
        try:
            from src.scanner.automation.adaptive_lr import AdaptiveLRScheduler
            self._lr_scheduler = AdaptiveLRScheduler(target_lr=learning_rate)
            logger.debug("Online RL: Adaptive LR scheduler initialized")
        except Exception:
            pass  # Will use fixed learning_rate

    def maybe_update(self, scan_cycle: int) -> List[WeightAdjustment]:
        """Run micro-update if it's time and new trades exist.

        Args:
            scan_cycle: Current scan cycle number

        Returns:
            List of adjustments made (empty if skipped or no changes)
        """
        if scan_cycle % self.update_interval != 0:
            return []

        # Load recent trades
        trades = self._load_recent_trades()
        if len(trades) < 3:
            logger.debug("Online RL: not enough recent trades (%d < 3)", len(trades))
            return []

        # Check if journal has grown since last update
        current_size = len(trades)
        if current_size == self._last_journal_size:
            logger.debug("Online RL: no new trades since last update")
            return []
        self._last_journal_size = current_size

        # Compute per-agent signals
        agent_signals = self._compute_agent_signals(trades)
        if not agent_signals:
            return []

        # Load current weights
        weights = self._load_weights()
        if not weights:
            return []

        # Apply adjustments
        adjustments = self._apply_adjustments(weights, agent_signals, len(trades))

        if adjustments:
            self._save_weights(weights)
            self._log_updates(adjustments)
            logger.info(
                "Online RL: %d weight adjustments from %d recent trades",
                len(adjustments), len(trades),
            )

        return adjustments

    def _load_recent_trades(self) -> List[Dict[str, Any]]:
        """Load the last N completed trades from journal."""
        if not self.journal_path.exists():
            return []

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.journal_path, default=[])
            except ImportError:
                data = json.loads(self.journal_path.read_text(encoding="utf-8"))

            if not isinstance(data, list):
                return []

            # Filter to completed trades only (must have outcome)
            completed = [
                t for t in data
                if t.get("outcome") in ("win", "loss", "breakeven")
            ]

            # Return last N
            return completed[-self.lookback_trades:]

        except Exception as e:
            logger.debug("Online RL: failed to load journal: %s", e)
            return []

    def _compute_agent_signals(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute per-agent reward signals from recent trades.

        For each agent, compute: (wins - losses) / total_trades as signal.
        Positive signal → agent helped, negative → agent hurt.

        Returns:
            Dict mapping agent name → signal in [-1.0, 1.0]
        """
        agent_wins: Dict[str, int] = {}
        agent_losses: Dict[str, int] = {}
        agent_total: Dict[str, int] = {}

        for trade in trades:
            outcome = trade.get("outcome", "")
            # Get agents that voted for this trade
            agents_for = trade.get("agents_for", [])
            agents_against = trade.get("agents_against", [])

            # Also check nested structure
            if not agents_for and not agents_against:
                agent_votes = trade.get("agent_votes", {})
                if isinstance(agent_votes, dict):
                    for agent_name, vote in agent_votes.items():
                        if vote in (True, "for", "buy", "sell", 1):
                            agents_for.append(agent_name)
                        else:
                            agents_against.append(agent_name)

            all_agents = set(agents_for) | set(agents_against)

            for agent in all_agents:
                agent_total[agent] = agent_total.get(agent, 0) + 1

                voted_for = agent in agents_for
                if outcome == "win" and voted_for:
                    agent_wins[agent] = agent_wins.get(agent, 0) + 1
                elif outcome == "loss" and voted_for:
                    agent_losses[agent] = agent_losses.get(agent, 0) + 1
                elif outcome == "win" and not voted_for:
                    # Agent voted against a winning trade — slight negative
                    agent_losses[agent] = agent_losses.get(agent, 0) + 1
                elif outcome == "loss" and not voted_for:
                    # Agent voted against a losing trade — slight positive
                    agent_wins[agent] = agent_wins.get(agent, 0) + 1

        # Compute signals
        signals: Dict[str, float] = {}
        for agent in agent_total:
            total = agent_total[agent]
            if total < 2:
                continue  # Need at least 2 trades to form a signal
            wins = agent_wins.get(agent, 0)
            losses = agent_losses.get(agent, 0)
            signals[agent] = (wins - losses) / total

        return signals

    def _load_weights(self) -> Dict[str, Any]:
        """Load current agent weights from disk."""
        if not self.weights_path.exists():
            return {}

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                return safe_json_read(self.weights_path, default={})
            except ImportError:
                return json.loads(self.weights_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Online RL: failed to load weights: %s", e)
            return {}

    def _save_weights(self, weights: Dict[str, Any]) -> None:
        """Save updated weights to disk."""
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.weights_path, weights)
            except ImportError:
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                self.weights_path.write_text(
                    json.dumps(weights, indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.warning("Online RL: failed to save weights: %s", e)

    def _apply_adjustments(
        self,
        weights: Dict[str, Any],
        signals: Dict[str, float],
        trade_count: int,
    ) -> List[WeightAdjustment]:
        """Apply micro-adjustments to the _global weight key.

        Modifies weights dict in place.
        """
        adjustments: List[WeightAdjustment] = []

        # Target the _global key (regime-specific weights are handled by full RL sync)
        global_weights = weights.get("_global", {})
        if not isinstance(global_weights, dict):
            return adjustments

        for agent, signal in signals.items():
            if agent not in global_weights:
                continue

            old_weight = float(global_weights[agent])

            # Compute clamped delta
            # US-091: Use adaptive LR if available, else fixed
            _lr = self.learning_rate
            if self._lr_scheduler is not None:
                _lr = self._lr_scheduler.get_lr()
                self.learning_rate = _lr  # Update for logging
            raw_delta = signal * _lr
            delta = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, raw_delta))

            if abs(delta) < 0.001:
                continue  # Skip negligible adjustments

            new_weight = old_weight + delta
            new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
            new_weight = round(new_weight, 4)

            if new_weight == old_weight:
                continue

            global_weights[agent] = new_weight

            reason = (
                f"signal={signal:+.2f} from {trade_count} trades → "
                f"delta={delta:+.4f}"
            )

            adjustments.append(WeightAdjustment(
                agent=agent,
                old_weight=old_weight,
                new_weight=new_weight,
                delta=round(new_weight - old_weight, 4),
                reason=reason,
                trade_count=trade_count,
            ))

            logger.debug(
                "Online RL: %s %.4f → %.4f (%s)",
                agent, old_weight, new_weight, reason,
            )

        weights["_global"] = global_weights
        return adjustments

    def _log_updates(self, adjustments: List[WeightAdjustment]) -> None:
        """Append updates to audit log."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()

            try:
                from src.scanner.automation.safe_json import safe_jsonl_append
                for adj in adjustments:
                    safe_jsonl_append(self.log_path, {
                        "timestamp": now,
                        "agent": adj.agent,
                        "old_weight": adj.old_weight,
                        "new_weight": adj.new_weight,
                        "delta": adj.delta,
                        "reason": adj.reason,
                        "trade_count": adj.trade_count,
                    })
            except ImportError:
                # Fallback when safe_json unavailable: acquire fcntl lock to prevent
                # concurrent JSONL corruption (L-1 fix — unlocked append was a race).
                lock_path = str(self.log_path) + ".lock"
                _lock_fd = None
                try:
                    import fcntl as _fcntl
                    import os as _os
                    _lock_fd = _os.open(lock_path, _os.O_CREAT | _os.O_RDWR)
                    _fcntl.flock(_lock_fd, _fcntl.LOCK_EX)
                except Exception:
                    pass  # best-effort; proceed without lock on fcntl-unavailable systems
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        for adj in adjustments:
                            f.write(json.dumps({
                                "timestamp": now,
                                "agent": adj.agent,
                                "old_weight": adj.old_weight,
                                "new_weight": adj.new_weight,
                                "delta": adj.delta,
                                "reason": adj.reason,
                                "trade_count": adj.trade_count,
                            }, default=str) + "\n")
                finally:
                    if _lock_fd is not None:
                        try:
                            import fcntl as _fcntl2, os as _os2
                            _fcntl2.flock(_lock_fd, _fcntl2.LOCK_UN)
                            _os2.close(_lock_fd)
                        except Exception:
                            pass

        except Exception as e:
            logger.debug("Online RL: failed to log updates: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get updater status for observability."""
        return {
            "update_interval": self.update_interval,
            "lookback_trades": self.lookback_trades,
            "learning_rate": self.learning_rate,
            "last_journal_size": self._last_journal_size,
            "max_adjustment": MAX_ADJUSTMENT,
        }
