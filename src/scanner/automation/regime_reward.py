"""Regime-aware RL reward shaping.

US-090: Replace simple PnL reward with regime-conditioned reward function.
In trending markets, reward momentum-following and position holding. In
choppy markets, reward quick exits. In crisis, reward capital preservation
(strong penalty for drawdown). Prevents reward hacking by monitoring
PnL-reward alignment.

Reward profiles:
- trending: Momentum bonus for holding through pullbacks, duration bonus
- mean_revert: Quick exit bonus, penalty for overstaying
- crisis: Preservation bonus, heavy drawdown penalty
- choppy: Neutral base, slight bonus for fast in/out

Anti-reward-hacking: track correlation between PnL and shaped reward.
If correlation drops below 0.5, emit alert (reward may be hacked).
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REWARD_LOG_PATH = _PROJECT_ROOT / "trained_data" / "regime_reward_log.json"

# Rolling window for PnL-reward correlation tracking
CORRELATION_WINDOW = 50


@dataclass
class RewardResult:
    """Result of regime-aware reward computation."""

    raw_pnl_reward: float = 0.0     # Simple PnL reward
    shaped_reward: float = 0.0       # Regime-conditioned shaped reward
    regime: str = "normal"
    regime_bonus: float = 0.0        # Extra reward from regime shaping
    drawdown_penalty: float = 0.0    # Penalty for drawdown
    duration_component: float = 0.0  # Duration-based reward component
    pnl_reward_correlation: float = 1.0  # Rolling PnL-reward correlation
    is_reward_hack_alert: bool = False   # True if correlation too low

    def to_dict(self) -> dict:
        return {
            "raw_pnl_reward": round(self.raw_pnl_reward, 6),
            "shaped_reward": round(self.shaped_reward, 6),
            "regime": self.regime,
            "regime_bonus": round(self.regime_bonus, 6),
            "drawdown_penalty": round(self.drawdown_penalty, 6),
            "duration_component": round(self.duration_component, 6),
            "pnl_reward_correlation": round(self.pnl_reward_correlation, 4),
            "is_reward_hack_alert": self.is_reward_hack_alert,
        }


# Regime reward profile configurations
REGIME_PROFILES: Dict[str, Dict[str, float]] = {
    "trending": {
        "momentum_bonus": 0.15,       # Bonus for holding with trend
        "duration_bonus_per_bar": 0.002,  # Small bonus for each bar held
        "max_duration_bonus": 0.10,   # Cap on duration bonus
        "drawdown_penalty_mult": 1.0,  # Standard drawdown penalty
        "quick_exit_penalty": 0.05,   # Penalty for exiting too fast
        "min_hold_bars": 5,           # Minimum bars before quick-exit penalty
    },
    "mean_revert": {
        "momentum_bonus": 0.0,        # No momentum bonus
        "duration_bonus_per_bar": -0.003,  # Penalty per bar held (want quick trades)
        "max_duration_bonus": 0.0,
        "drawdown_penalty_mult": 1.5,  # Higher drawdown penalty
        "quick_exit_penalty": 0.0,    # No penalty for quick exits
        "min_hold_bars": 0,
        "quick_exit_bonus": 0.08,     # Bonus for exiting within 10 bars
        "quick_exit_threshold": 10,   # Bars threshold for quick exit
    },
    "crisis": {
        "momentum_bonus": 0.0,
        "duration_bonus_per_bar": -0.005,  # Strong penalty per bar in crisis
        "max_duration_bonus": 0.0,
        "drawdown_penalty_mult": 3.0,  # Triple drawdown penalty
        "quick_exit_penalty": 0.0,
        "min_hold_bars": 0,
        "preservation_bonus": 0.20,   # Big bonus for preserving capital
        "preservation_threshold": 0.005,  # Max loss for preservation bonus
    },
    "choppy": {
        "momentum_bonus": 0.0,
        "duration_bonus_per_bar": -0.001,  # Slight penalty for holding
        "max_duration_bonus": 0.0,
        "drawdown_penalty_mult": 1.2,
        "quick_exit_penalty": 0.0,
        "min_hold_bars": 0,
    },
    "normal": {
        "momentum_bonus": 0.05,
        "duration_bonus_per_bar": 0.0,
        "max_duration_bonus": 0.0,
        "drawdown_penalty_mult": 1.0,
        "quick_exit_penalty": 0.0,
        "min_hold_bars": 0,
    },
}


class RegimeAwareReward:
    """Compute regime-conditioned rewards for RL weight updates.

    Replaces simple PnL reward with a shaped function that encourages
    regime-appropriate behavior:
    - Trending: hold winners, ride momentum
    - Mean-revert: exit fast, don't overstay
    - Crisis: preserve capital above all
    - Choppy: stay flat or trade fast

    Monitors PnL-reward correlation to detect reward hacking.
    """

    def __init__(
        self,
        base_reward_scale: float = 1.0,
        hack_alert_threshold: float = 0.50,
        persistence_path: Optional[Path] = None,
    ):
        self.base_reward_scale = base_reward_scale
        self.hack_alert_threshold = hack_alert_threshold
        self.persistence_path = persistence_path or _REWARD_LOG_PATH

        # Rolling history for anti-hacking correlation
        self._pnl_history: deque = deque(maxlen=CORRELATION_WINDOW)
        self._reward_history: deque = deque(maxlen=CORRELATION_WINDOW)

        # Full reward log
        self._log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        pnl: float,
        max_drawdown: float,
        regime: str,
        duration_bars: int,
        entry_confidence: float = 0.5,
    ) -> RewardResult:
        """Compute shaped reward for a closed trade.

        Args:
            pnl: Trade PnL (positive = profit, negative = loss)
            max_drawdown: Maximum drawdown during the trade (0-1 fraction)
            regime: Market regime during trade (trending/mean_revert/crisis/choppy/normal)
            duration_bars: Number of bars the position was held
            entry_confidence: Model confidence at entry time

        Returns:
            RewardResult with shaped reward and diagnostics
        """
        result = RewardResult(regime=regime.lower())

        # 1. Base PnL reward (normalized)
        result.raw_pnl_reward = pnl * self.base_reward_scale

        # 2. Get regime profile
        profile = REGIME_PROFILES.get(
            result.regime, REGIME_PROFILES["normal"]
        )

        # 3. Drawdown penalty (regime-scaled)
        dd_mult = profile["drawdown_penalty_mult"]
        result.drawdown_penalty = -abs(max_drawdown) * dd_mult * self.base_reward_scale

        # 4. Duration component
        dur_rate = profile["duration_bonus_per_bar"]
        dur_component = dur_rate * duration_bars
        max_dur = profile["max_duration_bonus"]
        if max_dur > 0:
            dur_component = min(dur_component, max_dur)
        result.duration_component = dur_component

        # 5. Regime-specific bonuses
        regime_bonus = 0.0

        # Momentum bonus (trending)
        if pnl > 0 and profile.get("momentum_bonus", 0) > 0:
            regime_bonus += profile["momentum_bonus"]

        # Quick exit bonus (mean-revert)
        if (
            "quick_exit_bonus" in profile
            and duration_bars <= profile.get("quick_exit_threshold", 10)
            and pnl > 0
        ):
            regime_bonus += profile["quick_exit_bonus"]

        # Quick exit penalty (trending — don't exit too early)
        if (
            profile.get("quick_exit_penalty", 0) > 0
            and duration_bars < profile.get("min_hold_bars", 5)
            and pnl > 0
        ):
            regime_bonus -= profile["quick_exit_penalty"]

        # Capital preservation bonus (crisis)
        if (
            "preservation_bonus" in profile
            and abs(pnl) < profile.get("preservation_threshold", 0.005)
        ):
            regime_bonus += profile["preservation_bonus"]

        result.regime_bonus = regime_bonus

        # 6. Combine into shaped reward
        result.shaped_reward = (
            result.raw_pnl_reward
            + result.drawdown_penalty
            + result.duration_component
            + result.regime_bonus
        )

        # 7. Anti-reward-hacking: track PnL vs shaped reward correlation
        self._pnl_history.append(pnl)
        self._reward_history.append(result.shaped_reward)
        result.pnl_reward_correlation = self._compute_correlation()
        result.is_reward_hack_alert = (
            result.pnl_reward_correlation < self.hack_alert_threshold
            and len(self._pnl_history) >= 20
        )

        if result.is_reward_hack_alert:
            logger.warning(
                f"RegimeReward: REWARD HACK ALERT — PnL-reward correlation "
                f"dropped to {result.pnl_reward_correlation:.3f} "
                f"(threshold={self.hack_alert_threshold})"
            )

        # Log
        self._log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
            "pnl": round(pnl, 6),
            "duration_bars": duration_bars,
            "entry_confidence": round(entry_confidence, 4),
        })

        logger.info(
            f"RegimeReward [{result.regime}]: "
            f"pnl={pnl:.4f} → shaped={result.shaped_reward:.4f} "
            f"(bonus={regime_bonus:+.4f}, dd_pen={result.drawdown_penalty:.4f}, "
            f"dur={result.duration_component:+.4f})"
        )

        return result

    def get_reward_summary(self) -> Dict[str, Any]:
        """Summary statistics of recent rewards."""
        if not self._pnl_history:
            return {"n_trades": 0}

        pnl_arr = np.array(list(self._pnl_history))
        reward_arr = np.array(list(self._reward_history))

        return {
            "n_trades": len(self._pnl_history),
            "mean_pnl": float(np.mean(pnl_arr)),
            "mean_shaped_reward": float(np.mean(reward_arr)),
            "pnl_reward_correlation": self._compute_correlation(),
            "reward_std": float(np.std(reward_arr)),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_correlation(self) -> float:
        """Compute rolling PnL-reward correlation."""
        if len(self._pnl_history) < 10:
            return 1.0  # Assume aligned until enough data

        pnl_arr = np.array(list(self._pnl_history))
        reward_arr = np.array(list(self._reward_history))

        pnl_std = np.std(pnl_arr)
        reward_std = np.std(reward_arr)
        if pnl_std < 1e-10 or reward_std < 1e-10:
            return 1.0

        corr = float(np.corrcoef(pnl_arr, reward_arr)[0, 1])
        return corr if not math.isnan(corr) else 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist reward history log."""
        try:
            data = {
                "version": 1,
                "log": self._log[-100:],  # Keep last 100 entries
                "summary": self.get_reward_summary(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"RegimeReward: Failed to save log: {e}")

    def load_log(self) -> None:
        """Load persisted reward log."""
        if not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text())
            self._log = data.get("log", [])
            logger.debug(f"RegimeReward: Loaded {len(self._log)} log entries")
        except Exception as e:
            logger.debug(f"RegimeReward: Failed to load log: {e}")
