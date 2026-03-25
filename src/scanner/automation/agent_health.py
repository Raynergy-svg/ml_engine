"""Agent health monitoring with selective stale-agent decay.

Tracks per-agent contribution metrics (win rate, vote alignment, trade count)
and provides selective decay rates: underperforming agents decay faster.
All decay events are logged for audit trail.

Usage:
    monitor = AgentHealthMonitor()
    monitor.record_outcome(agent_verdicts, trade_won=True)
    report = monitor.get_health_report()
    decay_rates = monitor.get_selective_decay_rates()
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HEALTH_PATH = _PROJECT_ROOT / "trained_data" / "agent_health.json"
_DECAY_LOG_PATH = _PROJECT_ROOT / "trained_data" / "agent_decay_log.jsonl"
_ATTRIBUTION_PATH = _PROJECT_ROOT / "trained_data" / "agent_attribution.json"


@dataclass
class AgentMetrics:
    """Per-agent health metrics."""

    trades_contributed: int = 0
    wins: int = 0
    losses: int = 0
    aligned_votes: int = 0  # voted with winning direction
    misaligned_votes: int = 0  # voted against winning direction
    last_trade_timestamp: str = ""

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def alignment_rate(self) -> float:
        total = self.aligned_votes + self.misaligned_votes
        return self.aligned_votes / total if total > 0 else 0.5

    @property
    def is_stale(self) -> bool:
        """Agent is stale if win rate < 35% over 20+ trades."""
        return self.trades_contributed >= 20 and self.win_rate < 0.35


class AgentHealthMonitor:
    """Monitors agent health and provides selective decay rates."""

    def __init__(
        self,
        health_path: Optional[Path] = None,
        decay_log_path: Optional[Path] = None,
        stale_win_rate: float = 0.35,
        stale_min_trades: int = 20,
        normal_decay_rate: float = 0.02,
        stale_decay_multiplier: float = 2.0,
    ):
        self.health_path = Path(health_path) if health_path else _HEALTH_PATH
        self.decay_log_path = Path(decay_log_path) if decay_log_path else _DECAY_LOG_PATH
        self.stale_win_rate = stale_win_rate
        self.stale_min_trades = stale_min_trades
        self.normal_decay_rate = normal_decay_rate
        self.stale_decay_multiplier = stale_decay_multiplier

        self._metrics: Dict[str, AgentMetrics] = {}
        # US-065: Per-pair per-agent tracking {pair: {agent: AgentMetrics}}
        self._pair_metrics: Dict[str, Dict[str, AgentMetrics]] = {}
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load agent metrics from disk."""
        if not self.health_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.health_path, default={})
            except ImportError:
                data = json.loads(self.health_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                for name, metrics_dict in data.items():
                    if isinstance(metrics_dict, dict):
                        self._metrics[name] = AgentMetrics(
                            trades_contributed=int(metrics_dict.get("trades_contributed", 0)),
                            wins=int(metrics_dict.get("wins", 0)),
                            losses=int(metrics_dict.get("losses", 0)),
                            aligned_votes=int(metrics_dict.get("aligned_votes", 0)),
                            misaligned_votes=int(metrics_dict.get("misaligned_votes", 0)),
                            last_trade_timestamp=str(metrics_dict.get("last_trade_timestamp", "")),
                        )
        except Exception as e:
            logger.debug(f"Failed to load agent health: {e}")

    def _save_metrics(self) -> None:
        """Persist agent metrics to disk."""
        data = {name: asdict(m) for name, m in self._metrics.items()}
        try:
            from src.scanner.automation.safe_json import safe_json_write
            safe_json_write(self.health_path, data)
        except ImportError:
            self.health_path.parent.mkdir(parents=True, exist_ok=True)
            self.health_path.write_text(json.dumps(data, indent=2, default=str))

    def _log_decay_event(self, agent_name: str, old_rate: float, new_rate: float, reason: str) -> None:
        """Append a decay event to the audit log."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "old_decay_rate": round(old_rate, 4),
            "new_decay_rate": round(new_rate, 4),
            "reason": reason,
        }
        try:
            from src.scanner.automation.safe_json import safe_jsonl_append
            safe_jsonl_append(self.decay_log_path, event)
        except ImportError:
            try:
                self.decay_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.decay_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            except Exception as e:
                logger.debug(f"Failed to log decay event: {e}")

    def record_outcome(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
        pair: Optional[str] = None,
    ) -> None:
        """Record a trade outcome for all participating agents.

        Args:
            agent_verdicts: List of agent verdict dicts with 'name' and 'passed' keys
            trade_won: Whether the trade was profitable
            pair: Currency pair (US-065: enables per-pair attribution tracking)
        """
        now = datetime.now(timezone.utc).isoformat()

        for verdict in agent_verdicts:
            name = str(verdict.get("name", ""))
            if not name:
                continue

            if name not in self._metrics:
                self._metrics[name] = AgentMetrics()

            m = self._metrics[name]
            m.trades_contributed += 1
            m.last_trade_timestamp = now

            if trade_won:
                m.wins += 1
            else:
                m.losses += 1

            # Vote alignment: agent passed + trade won = aligned
            # agent passed + trade lost = misaligned (and vice versa)
            agent_passed = bool(verdict.get("passed", False))
            if (agent_passed and trade_won) or (not agent_passed and not trade_won):
                m.aligned_votes += 1
            else:
                m.misaligned_votes += 1

            # US-065: Per-pair attribution tracking
            if pair:
                if pair not in self._pair_metrics:
                    self._pair_metrics[pair] = {}
                if name not in self._pair_metrics[pair]:
                    self._pair_metrics[pair][name] = AgentMetrics()

                pm = self._pair_metrics[pair][name]
                pm.trades_contributed += 1
                pm.last_trade_timestamp = now
                if trade_won:
                    pm.wins += 1
                else:
                    pm.losses += 1
                if (agent_passed and trade_won) or (not agent_passed and not trade_won):
                    pm.aligned_votes += 1
                else:
                    pm.misaligned_votes += 1

        self._save_metrics()

    def get_selective_decay_rates(self) -> Dict[str, float]:
        """Get per-agent decay rates based on health metrics.

        Stale agents (win rate < threshold over min_trades) get accelerated decay.

        Returns:
            Dict mapping agent name to decay rate (higher = faster decay toward baseline)
        """
        rates: Dict[str, float] = {}

        for name, m in self._metrics.items():
            if m.trades_contributed >= self.stale_min_trades and m.win_rate < self.stale_win_rate:
                # Stale agent — accelerated decay
                rate = self.normal_decay_rate * self.stale_decay_multiplier
                self._log_decay_event(
                    name,
                    self.normal_decay_rate,
                    rate,
                    f"Stale: {m.win_rate:.1%} win rate over {m.trades_contributed} trades "
                    f"(threshold: {self.stale_win_rate:.0%} over {self.stale_min_trades}+)",
                )
                rates[name] = rate
            else:
                rates[name] = self.normal_decay_rate

        return rates

    def get_health_report(self) -> Dict[str, Any]:
        """Get a health summary for all tracked agents.

        Returns:
            Dict with per-agent metrics and overall summary
        """
        agents = {}
        stale_count = 0
        total_trades = 0

        for name, m in sorted(self._metrics.items()):
            is_stale = (
                m.trades_contributed >= self.stale_min_trades
                and m.win_rate < self.stale_win_rate
            )
            if is_stale:
                stale_count += 1
            total_trades += m.trades_contributed

            agents[name] = {
                "trades": m.trades_contributed,
                "win_rate": round(m.win_rate, 3),
                "alignment_rate": round(m.alignment_rate, 3),
                "is_stale": is_stale,
                "last_trade": m.last_trade_timestamp,
            }

        return {
            "agents": agents,
            "total_agents_tracked": len(self._metrics),
            "stale_agents": stale_count,
            "total_outcomes_recorded": total_trades,
        }

    def get_pair_agent_matrix(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get per-pair per-agent performance matrix (US-065).

        Returns:
            {pair: {agent: {win_rate, trades, alignment_rate, is_stale}}}
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for pair, agents in self._pair_metrics.items():
            result[pair] = {}
            for agent_name, m in agents.items():
                result[pair][agent_name] = {
                    "win_rate": round(m.win_rate, 3),
                    "trades": m.trades_contributed,
                    "alignment_rate": round(m.alignment_rate, 3),
                    "is_stale": m.is_stale,
                }
        return result

    def get_pair_specific_decay_rates(self, pair: str) -> Dict[str, float]:
        """Get per-agent decay rates specific to a pair (US-065).

        Agents with <30% win rate on this specific pair get pair-specific
        accelerated decay (3x instead of the normal 2x for globally stale).
        """
        rates: Dict[str, float] = {}
        pair_agents = self._pair_metrics.get(pair, {})

        for name, m in pair_agents.items():
            if m.trades_contributed >= 10 and m.win_rate < 0.30:
                # Pair-specific stale: even more aggressive decay
                rate = self.normal_decay_rate * 3.0
                rates[name] = rate
            else:
                rates[name] = self.normal_decay_rate

        return rates

    def save_attribution(self) -> None:
        """Persist per-pair attribution data to trained_data/agent_attribution.json."""
        data = {}
        for pair, agents in self._pair_metrics.items():
            data[pair] = {name: asdict(m) for name, m in agents.items()}
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(_ATTRIBUTION_PATH, data)
            except ImportError:
                _ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
                _ATTRIBUTION_PATH.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Failed to save attribution: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return monitor configuration status."""
        return {
            "tracked_agents": len(self._metrics),
            "stale_threshold": f"{self.stale_win_rate:.0%} over {self.stale_min_trades}+ trades",
            "decay_multiplier": self.stale_decay_multiplier,
            "health_path": str(self.health_path),
        }
