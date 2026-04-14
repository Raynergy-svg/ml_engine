"""Expectancy Tracking system for agents per regime (US-292).

Tracks expectancy (E = Win% × AvgWin + Loss% × AvgLoss) per agent per regime.
Uses rolling windows to provide adaptive, regime-sensitive agent evaluation
and weight modifiers for negative-expectancy agents.

Key features:
- Per-agent per-regime rolling window (default 50 trades)
- Expectancy calculation with weight modifiers
- Atomic JSON persistence with atomic writes
- State validation and freshness checks
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExpectancyConfig:
    """Configuration for expectancy tracking."""
    enabled: bool = True
    window_size: int = 50  # Rolling window per agent per regime
    negative_penalty: float = -0.20  # Weight penalty for negative expectancy agents
    min_trades_for_calc: int = 5  # Minimum trades before expectancy is meaningful
    persistence_path: str = "trained_data/expectancy_tracker.json"
    version: str = "1.0.0"
    stale_warning_hours: float = 24.0


@dataclass
class TradeRecord:
    """Single trade outcome record."""
    pnl: float
    won: bool
    timestamp: float  # time.time()


@dataclass
class ExpectancyResult:
    """Calculated expectancy for an agent in a regime."""
    expectancy: float  # E value (can be negative)
    win_rate: float  # Win% in window (0.0 to 1.0)
    avg_win: float  # Average winning trade PnL
    avg_loss: float  # Average losing trade PnL (negative)
    trade_count: int  # Trades in window
    is_meaningful: bool  # True if trade_count >= min_trades_for_calc
    weight_modifier: float  # 1.0 if positive, (1 + negative_penalty) if negative


def create_default_expectancy_tracker(
    config: Optional[ExpectancyConfig] = None,
) -> ExpectancyTracker:
    """Factory function to create a default ExpectancyTracker."""
    return ExpectancyTracker(config or ExpectancyConfig())


class ExpectancyTracker:
    """Tracks expectancy per agent per regime using rolling windows."""
    _warned_stale_paths: set[str] = set()

    # Standard agent names from ScannerAgentTeam._BASE_WEIGHTS
    _AGENT_NAMES = [
        "trend",
        "mean_reversion",
        "volatility",
        "risk_sentinel",
        "uncertainty",
        "execution_quality",
        "news_risk",
        "multi_timeframe",
        "pair_performance",
        "momentum",
        "session_timing",
        "support_resistance",
        "trader_readiness",
    ]

    # Standard volatility regimes
    _REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]

    def __init__(self, config: Optional[ExpectancyConfig] = None):
        """Initialize tracker with optional config override.

        Args:
            config: ExpectancyConfig or None for defaults.
        """
        self._config = config or ExpectancyConfig()
        # Structure: {agent_name: {regime_name: deque(maxlen=window_size)}}
        self._windows: Dict[str, Dict[str, deque]] = {}
        self._version = self._config.version

        # Initialize empty windows for all agents and regimes
        for agent in self._AGENT_NAMES:
            self._windows[agent] = {}
            for regime in self._REGIME_NAMES:
                self._windows[agent][regime] = deque(maxlen=self._config.window_size)

    def record_trade(self, agent_name: str, regime: str, pnl: float, won: bool) -> None:
        """Record a trade outcome for an agent in a regime.

        Args:
            agent_name: Name of the agent (from _AGENT_NAMES)
            regime: Volatility regime (LOW, NORMAL, HIGH, EXTREME)
            pnl: Trade P&L in account currency
            won: True if trade won (hit TP), False if lost (hit SL)

        Raises:
            ValueError: If agent_name or regime not recognized
        """
        if agent_name not in self._windows:
            raise ValueError(f"Unknown agent: {agent_name}")
        if regime not in self._windows.get(agent_name, {}):
            raise ValueError(f"Unknown regime: {regime}")

        record = TradeRecord(pnl=pnl, won=won, timestamp=time.time())
        self._windows[agent_name][regime].append(record)
        logger.debug(
            f"Recorded trade for {agent_name}/{regime}: "
            f"pnl={pnl}, won={won}"
        )

    def get_expectancy(self, agent_name: str, regime: str) -> ExpectancyResult:
        """Calculate expectancy for an agent in a regime.

        E = Win% × AvgWin + Loss% × AvgLoss (where AvgLoss is negative)

        Args:
            agent_name: Name of the agent
            regime: Volatility regime

        Returns:
            ExpectancyResult with calculated values and modifiers
        """
        if agent_name not in self._windows:
            return self._neutral_result()
        if regime not in self._windows.get(agent_name, {}):
            return self._neutral_result()

        window = self._windows[agent_name][regime]
        if len(window) == 0:
            return self._neutral_result()

        # Split into wins and losses
        wins = [r.pnl for r in window if r.won]
        losses = [r.pnl for r in window if not r.won]

        trade_count = len(window)
        win_count = len(wins)
        loss_count = len(losses)

        # Calculate rates and averages
        win_rate = win_count / trade_count if trade_count > 0 else 0.0
        loss_rate = loss_count / trade_count if trade_count > 0 else 0.0

        avg_win = (sum(wins) / win_count) if win_count > 0 else 0.0
        avg_loss = (sum(losses) / loss_count) if loss_count > 0 else 0.0

        # Expectancy formula: E = Win% * AvgWin + Loss% * AvgLoss
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Weight modifier: 1.0 if positive, (1 + penalty) if negative
        is_meaningful = trade_count >= self._config.min_trades_for_calc
        if not is_meaningful:
            weight_modifier = 1.0
        elif expectancy >= 0:
            weight_modifier = 1.0
        else:
            weight_modifier = 1.0 + self._config.negative_penalty

        return ExpectancyResult(
            expectancy=round(expectancy, 2),
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            trade_count=trade_count,
            is_meaningful=is_meaningful,
            weight_modifier=round(weight_modifier, 4),
        )

    def _neutral_result(self) -> ExpectancyResult:
        """Return neutral result when no data available."""
        return ExpectancyResult(
            expectancy=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            trade_count=0,
            is_meaningful=False,
            weight_modifier=1.0,
        )

    def get_weight_modifier(self, agent_name: str, regime: str) -> float:
        """Get weight modifier for an agent in a regime.

        Returns 1.0 if positive expectancy or insufficient data,
        (1 + negative_penalty) if negative expectancy.

        Args:
            agent_name: Name of the agent
            regime: Volatility regime

        Returns:
            Weight modifier value (1.0 if positive/unclear, < 1.0 if negative)
        """
        result = self.get_expectancy(agent_name, regime)
        return result.weight_modifier

    def get_all_expectancies(self) -> Dict[str, Dict[str, ExpectancyResult]]:
        """Get expectancy for all tracked agent/regime combinations.

        Returns:
            Dict[agent_name, Dict[regime_name, ExpectancyResult]]
        """
        result = {}
        for agent in self._AGENT_NAMES:
            result[agent] = {}
            for regime in self._REGIME_NAMES:
                result[agent][regime] = self.get_expectancy(agent, regime)
        return result

    def get_top_agents(
        self, regime: str, n: int = 5
    ) -> List[Tuple[str, ExpectancyResult]]:
        """Get top N agents by expectancy in a given regime.

        Ranks by expectancy (descending), only includes agents with
        meaningful data (>= min_trades_for_calc).

        Args:
            regime: Volatility regime
            n: Number of top agents to return

        Returns:
            List of (agent_name, ExpectancyResult) tuples, sorted by expectancy descending
        """
        agents_with_expectancy = []
        for agent in self._AGENT_NAMES:
            result = self.get_expectancy(agent, regime)
            if result.is_meaningful:
                agents_with_expectancy.append((agent, result))

        # Sort by expectancy descending
        agents_with_expectancy.sort(key=lambda x: x[1].expectancy, reverse=True)
        return agents_with_expectancy[:n]

    def save_state(self, path: Optional[str] = None) -> None:
        """Save state to JSON file atomically.

        Writes to .tmp file first, then renames (atomic operation).
        Follows JSON Safety Gates from improvement.md.

        Args:
            path: Optional override path (defaults to config.persistence_path)

        Raises:
            IOError: If write fails
        """
        target_path = path or self._config.persistence_path
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.to_dict()

        tmp_path = f"{target_path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(state_dict, f, indent=2, sort_keys=True)
            # Atomic rename
            os.replace(tmp_path, target_path)
            logger.debug(f"Saved expectancy state to {target_path}")
        except IOError as e:
            logger.error(f"Failed to save expectancy state to {target_path}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load state from JSON file with validation.

        Validates state file freshness (warns if > 1 hour old).
        Returns gracefully on missing/corrupted file (no crash).

        Args:
            path: Optional override path (defaults to config.persistence_path)

        Returns:
            True if loaded successfully, False if file missing or corrupted
        """
        target_path = path or self._config.persistence_path
        if not os.path.exists(target_path):
            logger.debug(f"Expectancy state file not found: {target_path}")
            return False

        try:
            with open(target_path, "r") as f:
                data = json.load(f)

            # Prefer the payload timestamp to avoid false positives from copied files.
            state_ts = data.get("timestamp")
            if isinstance(state_ts, (int, float)):
                age_seconds = time.time() - float(state_ts)
            else:
                age_seconds = time.time() - os.path.getmtime(target_path)

            stale_after_seconds = max(3600.0, float(self._config.stale_warning_hours) * 3600.0)
            if age_seconds > stale_after_seconds and target_path not in self._warned_stale_paths:
                self._warned_stale_paths.add(target_path)
                logger.warning(
                    f"Expectancy state file is stale ({age_seconds / 3600:.1f}h old): {target_path}"
                )

            loaded = ExpectancyTracker.from_dict(data, config=self._config)
            # Copy loaded state into self
            self._windows = loaded._windows
            logger.info(f"Loaded expectancy state from {target_path}")
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load expectancy state from {target_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading expectancy state: {e}")
            return False

    def to_dict(self) -> dict:
        """Serialize full state to dict.

        Returns:
            Dict with structure {agent: {regime: [trade_dicts]}, _version, _timestamp}
        """
        windows_dict = {}
        for agent, regimes in self._windows.items():
            windows_dict[agent] = {}
            for regime, window in regimes.items():
                windows_dict[agent][regime] = [
                    asdict(record) for record in window
                ]

        return {
            "version": self._version,
            "timestamp": time.time(),
            "windows": windows_dict,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        config: Optional[ExpectancyConfig] = None,
    ) -> ExpectancyTracker:
        """Deserialize from dict (loads state into new or existing instance).

        Args:
            data: Dict from to_dict()
            config: Optional config override

        Returns:
            ExpectancyTracker instance with loaded state

        Raises:
            ValueError: If data structure invalid
        """
        instance = cls(config)

        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        windows_data = data.get("windows", {})
        if not isinstance(windows_data, dict):
            raise ValueError("windows must be a dictionary")

        for agent, regimes_data in windows_data.items():
            if agent not in instance._windows:
                logger.warning(f"Skipping unknown agent in loaded state: {agent}")
                continue
            if not isinstance(regimes_data, dict):
                logger.warning(f"Skipping invalid regime data for {agent}")
                continue

            for regime, trades_data in regimes_data.items():
                if regime not in instance._windows.get(agent, {}):
                    logger.warning(
                        f"Skipping unknown regime in loaded state: {agent}/{regime}"
                    )
                    continue
                if not isinstance(trades_data, list):
                    logger.warning(
                        f"Skipping invalid trades data for {agent}/{regime}"
                    )
                    continue

                # Reconstruct trade records
                for trade_dict in trades_data:
                    try:
                        record = TradeRecord(
                            pnl=float(trade_dict.get("pnl", 0.0)),
                            won=bool(trade_dict.get("won", False)),
                            timestamp=float(trade_dict.get("timestamp", time.time())),
                        )
                        instance._windows[agent][regime].append(record)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Skipping invalid trade record for {agent}/{regime}: {e}"
                        )
                        continue

        logger.info(
            f"Loaded expectancy state: {sum(len(r) for a in instance._windows.values() for r in a.values())} total trades"
        )
        return instance
