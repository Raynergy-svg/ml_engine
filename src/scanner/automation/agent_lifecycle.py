"""Agent lifecycle manager with fitness tracking and quarantine.

US-117: Track real-time agent fitness and manage quarantine/promotion.
Currently agents marked stale continue voting with full weight. This
module introduces fitness scoring, quarantine (0.3x weight), candidate
pipeline for shadow-testing, and retirement for persistently poor agents.

Approach:
1. Track rolling accuracy per agent (last 20 trades)
2. Status transitions: ACTIVE → QUARANTINED → RETIRED (or back to ACTIVE)
3. Quarantine: agent still votes but at 0.3x weight for 20 trades
4. If accuracy recovers in quarantine → ACTIVE; if not → RETIRED (0.0x)
5. CANDIDATE status for new agents shadow-testing before promotion
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "agent_lifecycle_log.json"

# Fitness parameters
ROLLING_WINDOW = 20  # Trades for fitness computation
MIN_TRADES_FOR_ACTION = 15  # Minimum trades before quarantine/promotion

# Thresholds
QUARANTINE_THRESHOLD = 0.35  # Below this → quarantine
RECOVERY_THRESHOLD = 0.50    # Must exceed this to recover from quarantine
RETIREMENT_THRESHOLD = 0.25  # Below this in quarantine → retire
PROMOTION_THRESHOLD = 0.60   # Candidate must exceed this for promotion
PROMOTION_MIN_TRADES = 30    # Candidate needs this many trades

# Weight multipliers
ACTIVE_WEIGHT = 1.0
QUARANTINE_WEIGHT = 0.30
CANDIDATE_WEIGHT = 0.50     # Shadow-test at half weight
RETIRED_WEIGHT = 0.0

# Valid statuses
VALID_STATUSES = {"ACTIVE", "QUARANTINED", "CANDIDATE", "RETIRED"}


class AgentRecord:
    """Tracks an agent's lifecycle state and fitness."""

    def __init__(self, name: str, status: str = "ACTIVE"):
        self.name = name
        self.status = status
        self.outcomes: List[bool] = []
        self.quarantine_start_idx: int = 0
        self.total_trades: int = 0
        self.transitions: List[Dict[str, Any]] = []

    @property
    def fitness(self) -> Optional[float]:
        """Rolling accuracy over last ROLLING_WINDOW trades."""
        if len(self.outcomes) < MIN_TRADES_FOR_ACTION:
            return None
        recent = self.outcomes[-ROLLING_WINDOW:]
        return sum(recent) / len(recent)

    @property
    def weight_modifier(self) -> float:
        """Get weight modifier based on status."""
        return {
            "ACTIVE": ACTIVE_WEIGHT,
            "QUARANTINED": QUARANTINE_WEIGHT,
            "CANDIDATE": CANDIDATE_WEIGHT,
            "RETIRED": RETIRED_WEIGHT,
        }.get(self.status, ACTIVE_WEIGHT)

    def add_outcome(self, won: bool) -> None:
        self.outcomes.append(won)
        if len(self.outcomes) > ROLLING_WINDOW * 3:
            self.outcomes = self.outcomes[-ROLLING_WINDOW * 2:]
        self.total_trades += 1

    def transition(self, new_status: str, reason: str) -> None:
        old_status = self.status
        self.status = new_status
        self.transitions.append({
            "from": old_status,
            "to": new_status,
            "reason": reason,
            "fitness": self.fitness,
            "trade_count": self.total_trades,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.transitions) > 20:
            self.transitions = self.transitions[-10:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "fitness": round(self.fitness, 4) if self.fitness is not None else None,
            "weight_modifier": self.weight_modifier,
            "total_trades": self.total_trades,
            "recent_outcomes": self.outcomes[-10:],
            "transitions": self.transitions[-5:],
        }


class AgentLifecycleManager:
    """Manages agent fitness, quarantine, and promotion lifecycle.

    Extends agent_health.py with actionable state transitions.
    Stale agents get quarantined (0.3x weight), and if they don't
    recover, retired (0.0x). New candidate agents shadow-test at
    0.5x weight before full promotion.

    Usage:
        lifecycle = AgentLifecycleManager()
        lifecycle.update_fitness("momentum", won=True, regime="NORMAL")
        status = lifecycle.get_agent_status("momentum")
        weight = lifecycle.get_weight_modifier("momentum")
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        self._agents: Dict[str, AgentRecord] = {}
        self._lifecycle_log: List[Dict[str, Any]] = []
        self._total_updates: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_fitness(
        self, agent: str, won: bool, regime: str = "NORMAL"
    ) -> None:
        """Update agent fitness with trade outcome.

        Args:
            agent: Agent name.
            won: Whether the trade was a winner.
            regime: Current market regime (for context).
        """
        if agent not in self._agents:
            self._agents[agent] = AgentRecord(agent)

        record = self._agents[agent]
        record.add_outcome(won)
        self._total_updates += 1

        # Check for status transitions
        self._evaluate_transitions(agent)

    def get_agent_status(self, agent: str) -> str:
        """Get agent's current lifecycle status."""
        record = self._agents.get(agent)
        if not record:
            return "ACTIVE"
        return record.status

    def get_weight_modifier(self, agent: str) -> float:
        """Get weight modifier for agent based on lifecycle status.

        Returns:
            1.0 for ACTIVE, 0.30 for QUARANTINED, 0.50 for CANDIDATE,
            0.0 for RETIRED.
        """
        record = self._agents.get(agent)
        if not record:
            return ACTIVE_WEIGHT
        return record.weight_modifier

    def quarantine_agent(self, agent: str, reason: str = "manual") -> None:
        """Force-quarantine an agent."""
        if agent not in self._agents:
            self._agents[agent] = AgentRecord(agent)
        record = self._agents[agent]
        if record.status != "QUARANTINED":
            record.transition("QUARANTINED", reason)
            record.quarantine_start_idx = record.total_trades

    def promote_candidate(self, agent: str) -> bool:
        """Attempt to promote a candidate agent to active.

        Returns:
            True if promotion succeeded.
        """
        record = self._agents.get(agent)
        if not record or record.status != "CANDIDATE":
            return False

        fitness = record.fitness
        if fitness is not None and fitness >= PROMOTION_THRESHOLD and record.total_trades >= PROMOTION_MIN_TRADES:
            record.transition("ACTIVE", f"promoted_fitness_{fitness:.2f}")
            return True
        return False

    def check_degradation(self) -> List[Dict[str, Any]]:
        """Phase 26 (US-158): Check all active agents for degradation via z-score.

        Computes z-score of recent win rate vs 30-trade baseline.
        Flags agents with z-score < -1.5 as degrading BEFORE they hit
        the quarantine threshold (0.35).

        Returns:
            List of warning dicts for degrading agents.
        """
        _BASELINE_WINDOW = 30
        _RECENT_WINDOW = 10
        _Z_THRESHOLD = -1.5
        warnings: List[Dict[str, Any]] = []

        for name, record in self._agents.items():
            if record.status != "ACTIVE":
                continue
            if len(record.outcomes) < _BASELINE_WINDOW:
                continue

            baseline = record.outcomes[-_BASELINE_WINDOW:]
            recent = record.outcomes[-_RECENT_WINDOW:]
            baseline_rate = sum(baseline) / len(baseline)
            recent_rate = sum(recent) / len(recent)

            # Z-score: (observed - expected) / std_error
            # Under binomial, std_error = sqrt(p*(1-p)/n)
            p = max(baseline_rate, 0.01)
            se = float(np.sqrt(p * (1 - p) / len(recent)))
            if se < 0.001:
                continue
            z_score = (recent_rate - baseline_rate) / se

            if z_score < _Z_THRESHOLD:
                warnings.append({
                    "agent_name": name,
                    "baseline_rate": round(baseline_rate, 3),
                    "recent_rate": round(recent_rate, 3),
                    "z_score": round(z_score, 2),
                    "trades_in_baseline": len(baseline),
                    "trades_in_recent": len(recent),
                    "recommendation": "review_for_quarantine",
                })
                logger.warning(
                    "US-158: Agent '%s' degrading — z=%.2f "
                    "(baseline=%.1f%% recent=%.1f%%)",
                    name, z_score, baseline_rate * 100, recent_rate * 100,
                )

        return warnings

    def add_candidate(self, agent: str) -> None:
        """Register a new candidate agent for shadow-testing."""
        if agent not in self._agents:
            self._agents[agent] = AgentRecord(agent, status="CANDIDATE")
        else:
            self._agents[agent].transition("CANDIDATE", "added_as_candidate")

    def get_lifecycle_report(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle report for all agents."""
        by_status: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for name, record in self._agents.items():
            by_status[record.status].append(record.to_dict())

        return {
            "total_agents": len(self._agents),
            "by_status": dict(by_status),
            "status_counts": {
                status: len(agents) for status, agents in by_status.items()
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get manager status for observability."""
        return {
            "total_updates": self._total_updates,
            "agents": {
                name: {
                    "status": record.status,
                    "fitness": round(record.fitness, 4) if record.fitness else None,
                    "weight": record.weight_modifier,
                }
                for name, record in self._agents.items()
            },
            "recent_log": self._lifecycle_log[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_transitions(self, agent: str) -> None:
        """Evaluate if agent should transition status."""
        record = self._agents[agent]
        fitness = record.fitness

        if fitness is None:
            return  # Not enough data

        if record.status == "ACTIVE":
            if fitness < QUARANTINE_THRESHOLD:
                record.transition(
                    "QUARANTINED",
                    f"low_fitness_{fitness:.2f}"
                )
                record.quarantine_start_idx = record.total_trades
                self._log_transition(agent, "ACTIVE", "QUARANTINED", fitness)

        elif record.status == "QUARANTINED":
            trades_in_quarantine = record.total_trades - record.quarantine_start_idx

            if trades_in_quarantine >= MIN_TRADES_FOR_ACTION:
                # Compute fitness during quarantine period
                quarantine_outcomes = record.outcomes[-trades_in_quarantine:]
                quarantine_fitness = (
                    sum(quarantine_outcomes) / len(quarantine_outcomes)
                    if quarantine_outcomes else 0.0
                )

                if quarantine_fitness >= RECOVERY_THRESHOLD:
                    record.transition(
                        "ACTIVE",
                        f"recovered_fitness_{quarantine_fitness:.2f}"
                    )
                    self._log_transition(agent, "QUARANTINED", "ACTIVE", quarantine_fitness)

                elif quarantine_fitness < RETIREMENT_THRESHOLD:
                    record.transition(
                        "RETIRED",
                        f"failed_quarantine_{quarantine_fitness:.2f}"
                    )
                    self._log_transition(agent, "QUARANTINED", "RETIRED", quarantine_fitness)

        elif record.status == "CANDIDATE":
            if record.total_trades >= PROMOTION_MIN_TRADES:
                if fitness >= PROMOTION_THRESHOLD:
                    record.transition(
                        "ACTIVE",
                        f"auto_promoted_{fitness:.2f}"
                    )
                    self._log_transition(agent, "CANDIDATE", "ACTIVE", fitness)
                elif fitness < QUARANTINE_THRESHOLD:
                    record.transition(
                        "RETIRED",
                        f"candidate_failed_{fitness:.2f}"
                    )
                    self._log_transition(agent, "CANDIDATE", "RETIRED", fitness)

    def _log_transition(
        self, agent: str, from_status: str, to_status: str, fitness: float
    ) -> None:
        """Log a lifecycle transition."""
        self._lifecycle_log.append({
            "agent": agent,
            "from": from_status,
            "to": to_status,
            "fitness": round(fitness, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._lifecycle_log) > 200:
            self._lifecycle_log = self._lifecycle_log[-100:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist lifecycle state."""
        try:
            agents_data = {}
            for name, record in self._agents.items():
                agents_data[name] = {
                    "status": record.status,
                    "outcomes": record.outcomes[-ROLLING_WINDOW * 2:],
                    "total_trades": record.total_trades,
                    "quarantine_start_idx": record.quarantine_start_idx,
                    "transitions": record.transitions[-10:],
                }

            data = {
                "version": 1,
                "total_updates": self._total_updates,
                "agents": agents_data,
                "lifecycle_log": self._lifecycle_log[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AgentLifecycleManager: Failed to save: {e}")

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
                self._total_updates = data.get("total_updates", 0)
                self._lifecycle_log = data.get("lifecycle_log", [])
                raw_agents = data.get("agents", {})
                for name, agent_data in raw_agents.items():
                    record = AgentRecord(name, status=agent_data.get("status", "ACTIVE"))
                    record.outcomes = agent_data.get("outcomes", [])
                    record.total_trades = agent_data.get("total_trades", 0)
                    record.quarantine_start_idx = agent_data.get("quarantine_start_idx", 0)
                    record.transitions = agent_data.get("transitions", [])
                    self._agents[name] = record
        except Exception as e:
            logger.debug(f"AgentLifecycleManager: Failed to load: {e}")
