"""
Adaptive Position Sizing Module for FX Trading Bot

Implements 5 complementary sizing strategies:
1. Fractional Kelly Criterion (win rate + risk/reward based)
2. Confidence-Weighted Sigmoid (agent consensus based)
3. Drawdown-Recovery Sizing (equity protection)
4. Regime-Dependent Multipliers (volatility-aware)
5. Composite Sizing (weighted combination)

All calculations are performed atomically and persisted to JSON with safety guarantees.
Thread-safe: no mutable state shared between calls.
"""

import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "AdaptivePositionSizingConfig",
    "PositionSizeResult",
    "AdaptivePositionSizer",
    "StreakConfig",
    "StreakTracker",
    "create_default_position_sizer",
    "create_conservative_position_sizer",
    "create_aggressive_position_sizer",
]

logger = logging.getLogger(__name__)


@dataclass
class AdaptivePositionSizingConfig:
    """Configuration for adaptive position sizing strategies."""

    # Fractional Kelly Criterion
    kelly_fraction: float = 0.33
    kelly_window: int = 50

    # Confidence-Weighted Sigmoid
    sigmoid_steepness: float = 4.0
    sigmoid_midpoint: float = 0.5

    # Drawdown-Recovery Sizing
    max_acceptable_drawdown: float = 0.15
    drawdown_floor: float = 0.3

    # Composite Sizing
    base_risk_pct: float = 0.05
    max_risk_per_trade: float = 0.10
    min_position_units: int = 1000

    # Regime multipliers
    regime_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 1.3,
            "NORMAL": 1.0,
            "HIGH": 0.65,
            "EXTREME": 0.40,
        }
    )

    # Component weighting
    component_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "kelly": 0.3,
            "confidence": 0.3,
            "drawdown": 0.2,
            "regime": 0.2,
        }
    )

    # Streak-aware anti-martingale sizing
    streak_enabled: bool = True

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not (0.0 < self.kelly_fraction < 1.0):
            raise ValueError(f"kelly_fraction must be in (0, 1), got {self.kelly_fraction}")

        if self.kelly_window < 1:
            raise ValueError(f"kelly_window must be >= 1, got {self.kelly_window}")

        if self.sigmoid_steepness <= 0:
            raise ValueError(
                f"sigmoid_steepness must be > 0, got {self.sigmoid_steepness}"
            )

        if not (0.0 < self.sigmoid_midpoint < 1.0):
            raise ValueError(
                f"sigmoid_midpoint must be in (0, 1), got {self.sigmoid_midpoint}"
            )

        if not (0.0 < self.max_acceptable_drawdown <= 1.0):
            raise ValueError(
                f"max_acceptable_drawdown must be in (0, 1], got {self.max_acceptable_drawdown}"
            )

        if not (0.0 <= self.drawdown_floor < 1.0):
            raise ValueError(f"drawdown_floor must be in [0, 1), got {self.drawdown_floor}")

        if not (0.0 < self.base_risk_pct < 1.0):
            raise ValueError(f"base_risk_pct must be in (0, 1), got {self.base_risk_pct}")

        if not (0.0 < self.max_risk_per_trade < 1.0):
            raise ValueError(
                f"max_risk_per_trade must be in (0, 1), got {self.max_risk_per_trade}"
            )

        if self.base_risk_pct > self.max_risk_per_trade:
            raise ValueError(
                f"base_risk_pct ({self.base_risk_pct}) must be <= max_risk_per_trade ({self.max_risk_per_trade})"
            )

        if self.min_position_units < 0:
            raise ValueError(
                f"min_position_units must be >= 0, got {self.min_position_units}"
            )

        # Validate regime multipliers
        if not isinstance(self.regime_multipliers, dict):
            raise ValueError("regime_multipliers must be a dict")
        for regime, mult in self.regime_multipliers.items():
            if not isinstance(regime, str):
                raise ValueError(f"regime name must be string, got {type(regime)}")
            if mult <= 0:
                raise ValueError(
                    f"regime multiplier must be > 0, got {mult} for regime {regime}"
                )

        # Validate component weights sum to 1.0
        if not isinstance(self.component_weights, dict):
            raise ValueError("component_weights must be a dict")
        weight_sum = sum(self.component_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(
                f"component_weights must sum to 1.0, got {weight_sum}"
            )


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    final_size: float  # final position size in units
    base_size: float  # base size before adjustments
    kelly_factor: float  # Kelly criterion adjustment factor
    confidence_factor: float  # confidence-weighted sigmoid factor
    drawdown_factor: float  # drawdown recovery adjustment factor
    regime_factor: float  # regime-dependent multiplier
    regime_name: str  # regime used
    streak_multiplier: float = 1.0  # streak-aware anti-martingale multiplier
    details: Dict = field(default_factory=dict)  # detailed breakdown

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return asdict(self)


class AdaptivePositionSizer:
    """
    Adaptive position sizing engine combining 5 strategies.

    Attributes:
        config: AdaptivePositionSizingConfig instance
        _trade_history: deque of (pnl, win) tuples for Kelly calculation
    """

    def __init__(self, config: Optional[AdaptivePositionSizingConfig] = None):
        """
        Initialize the adaptive position sizer.

        Args:
            config: AdaptivePositionSizingConfig instance. If None, uses defaults.
        """
        self.config = config or AdaptivePositionSizingConfig()
        self._trade_history = deque(maxlen=self.config.kelly_window)
        self._streak_tracker: Optional[StreakTracker] = (
            StreakTracker() if self.config.streak_enabled else None
        )
        self._last_regime_name: Optional[str] = None

    def record_trade(self, pnl: float, win: bool) -> None:
        """
        Record a trade outcome for rolling Kelly calculation and streak tracking.

        Args:
            pnl: Profit/loss in dollars (or account currency)
            win: Boolean indicating trade outcome

        Raises:
            ValueError: If pnl is NaN or infinite
        """
        if not np.isfinite(pnl):
            logger.warning(f"Skipping non-finite PnL: {pnl}")
            return

        self._trade_history.append((float(pnl), bool(win)))

        # Update streak tracker if enabled
        if self._streak_tracker is not None:
            regime = self._last_regime_name or "NORMAL"
            self._streak_tracker.record_outcome(win=win, regime_name=regime)

        logger.debug(f"Recorded trade: PnL={pnl}, Win={win}, History length={len(self._trade_history)}")

    def calculate_kelly(self) -> float:
        """
        Calculate fractional Kelly criterion factor.

        Formula: K% = (W*R - L) / R * fraction
        Where:
            W = win_rate
            L = loss_rate (1 - win_rate)
            R = avg_win / avg_loss
            fraction = configurable Kelly fraction

        Returns:
            Kelly factor in [0, 1]. If insufficient data, returns conservative default (0.01).
            Never exceeds fraction value.

        Raises:
            ValueError: If trades recorded but unable to compute (e.g., all losses, no variance)
        """
        if len(self._trade_history) == 0:
            logger.debug("No trade history; using conservative Kelly default 0.01")
            return 0.01

        wins = sum(1 for _, win in self._trade_history if win)
        win_rate = wins / len(self._trade_history)

        # Handle edge cases
        if win_rate == 0.0:
            logger.warning("Zero win rate; Kelly factor = 0.01")
            return 0.01

        if win_rate == 1.0:
            logger.warning("Perfect win rate; Kelly factor = kelly_fraction")
            return self.config.kelly_fraction

        loss_rate = 1.0 - win_rate

        # Separate wins and losses
        win_pnls = [pnl for pnl, win in self._trade_history if win]
        loss_pnls = [pnl for pnl, win in self._trade_history if not win]

        avg_win = np.mean(win_pnls) if win_pnls else 0.0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0

        # Handle case where avg_loss is zero or negative
        if avg_loss <= 0:
            logger.warning(f"Non-positive avg_loss ({avg_loss}); Kelly factor = kelly_fraction")
            return self.config.kelly_fraction

        risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else 1.0

        # Kelly formula: K = (W*R - L) / R * fraction
        kelly_raw = (win_rate * risk_reward_ratio - loss_rate) / risk_reward_ratio
        kelly_factored = kelly_raw * self.config.kelly_fraction

        # Clamp to valid range [0, kelly_fraction]
        kelly_factor = max(0.01, min(kelly_factored, self.config.kelly_fraction))

        logger.debug(
            f"Kelly: W={win_rate:.3f}, R={risk_reward_ratio:.3f}, "
            f"raw={kelly_raw:.3f}, factored={kelly_factor:.3f}"
        )

        return kelly_factor

    def calculate_confidence_multiplier(self, confidence: float) -> float:
        """
        Calculate confidence-weighted sigmoid multiplier.

        Formula: σ(x) = 1 / (1 + e^(-k * (x - midpoint)))
        Maps confidence [0, 1] to multiplier [~0.02, ~0.98]

        Args:
            confidence: Confidence level in [0, 1]

        Returns:
            Sigmoid multiplier in approximately [0.02, 0.98]

        Raises:
            ValueError: If confidence is outside [0, 1]
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        # Sigmoid function: 1 / (1 + e^(-k * (x - midpoint)))
        exponent = -self.config.sigmoid_steepness * (
            confidence - self.config.sigmoid_midpoint
        )

        # Protect against overflow
        exponent = np.clip(exponent, -500, 500)

        sigmoid = 1.0 / (1.0 + np.exp(exponent))

        logger.debug(f"Sigmoid: confidence={confidence:.3f}, multiplier={sigmoid:.3f}")

        return float(sigmoid)

    def calculate_drawdown_recovery(self, current_drawdown_pct: float) -> float:
        """
        Calculate drawdown recovery adjustment factor.

        Formula: recovery_factor = max(floor, 1 - (current_dd / max_dd))
        When no drawdown, returns 1.0.

        Args:
            current_drawdown_pct: Current drawdown as percentage (0.0 to 1.0)

        Returns:
            Recovery factor in [drawdown_floor, 1.0]

        Raises:
            ValueError: If drawdown is negative or exceeds 1.0
        """
        if not (0.0 <= current_drawdown_pct <= 1.0):
            raise ValueError(
                f"current_drawdown_pct must be in [0, 1], got {current_drawdown_pct}"
            )

        if current_drawdown_pct == 0.0:
            logger.debug("No drawdown; recovery_factor = 1.0")
            return 1.0

        # Drawdown recovery: as we approach max_acceptable_drawdown, size shrinks
        recovery_factor = 1.0 - (
            current_drawdown_pct / self.config.max_acceptable_drawdown
        )

        # Clamp to floor
        recovery_factor = max(self.config.drawdown_floor, recovery_factor)

        logger.debug(
            f"Drawdown: current={current_drawdown_pct:.3f}, "
            f"max_acceptable={self.config.max_acceptable_drawdown:.3f}, "
            f"recovery_factor={recovery_factor:.3f}"
        )

        return recovery_factor

    def get_regime_multiplier(self, regime_name: str) -> float:
        """
        Get regime-dependent position multiplier.

        Args:
            regime_name: Regime name (e.g., 'LOW', 'NORMAL', 'HIGH', 'EXTREME')

        Returns:
            Regime multiplier (default 1.0 if regime not found, with warning)

        Raises:
            TypeError: If regime_name is not a string
        """
        if not isinstance(regime_name, str):
            raise TypeError(f"regime_name must be string, got {type(regime_name)}")

        multiplier = self.config.regime_multipliers.get(regime_name, 1.0)

        if regime_name not in self.config.regime_multipliers:
            logger.warning(
                f"Unknown regime '{regime_name}'; defaulting to 1.0. "
                f"Available: {list(self.config.regime_multipliers.keys())}"
            )

        logger.debug(f"Regime: {regime_name}, multiplier={multiplier:.3f}")

        return multiplier

    def calculate_size(
        self,
        account_equity: float,
        confidence: float,
        current_drawdown_pct: float,
        regime_name: str = "NORMAL",
    ) -> PositionSizeResult:
        """
        Calculate final position size combining all 5 strategies plus streak overlay.

        Strategy combination:
        1. base_size = account_equity * base_risk_pct
        2. kelly_adj = kelly_factor
        3. confidence_adj = sigmoid(confidence)
        4. drawdown_adj = recovery_factor
        5. regime_adj = regime_multiplier
        6. composite = weighted combination of above
        7. intermediate = base_size * composite
        8. streak_adj = anti-martingale multiplier (if streak_enabled)
        9. final = intermediate * streak_adj, clamped to [min, max]

        Args:
            account_equity: Account equity in dollars
            confidence: Agent consensus confidence [0, 1]
            current_drawdown_pct: Current drawdown as percentage [0, 1]
            regime_name: Market regime name (default 'NORMAL')

        Returns:
            PositionSizeResult with final size, streak multiplier, and detailed breakdown

        Raises:
            ValueError: If account_equity <= 0 or confidence outside [0, 1]
        """
        # Track regime for streak reset detection
        self._last_regime_name = regime_name

        if account_equity <= 0:
            raise ValueError(f"account_equity must be > 0, got {account_equity}")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        if not (0.0 <= current_drawdown_pct <= 1.0):
            raise ValueError(
                f"current_drawdown_pct must be in [0, 1], got {current_drawdown_pct}"
            )

        # Step 1: Calculate base size
        base_size = account_equity * self.config.base_risk_pct

        # Step 2: Kelly adjustment
        kelly_factor = self.calculate_kelly()

        # Step 3: Confidence adjustment
        confidence_factor = self.calculate_confidence_multiplier(confidence)

        # Step 4: Drawdown recovery
        drawdown_factor = self.calculate_drawdown_recovery(current_drawdown_pct)

        # Step 5: Regime multiplier
        regime_factor = self.get_regime_multiplier(regime_name)

        # Step 6: Composite sizing with weighted combination
        composite = (
            kelly_factor * self.config.component_weights["kelly"]
            + confidence_factor * self.config.component_weights["confidence"]
            + drawdown_factor * self.config.component_weights["drawdown"]
            + regime_factor * self.config.component_weights["regime"]
        )

        intermediate_size = base_size * composite

        # Step 7: Apply streak multiplier (anti-martingale overlay)
        streak_multiplier = 1.0
        if self._streak_tracker is not None:
            streak_multiplier = self._streak_tracker.get_multiplier()

        streak_adjusted_size = intermediate_size * streak_multiplier

        # Step 8: Apply clamping
        max_position_size = account_equity * self.config.max_risk_per_trade
        final_size = max(
            self.config.min_position_units,
            min(streak_adjusted_size, max_position_size),
        )

        # Log streak adjustment if active
        if streak_multiplier != 1.0:
            logger.info(
                f"Phase 45: Streak multiplier={streak_multiplier:.3f} "
                f"(wins={self._streak_tracker._consecutive_wins}, "
                f"losses={self._streak_tracker._consecutive_losses})"
            )

        # Build result
        details = {
            "account_equity": account_equity,
            "base_size": base_size,
            "kelly_factor": kelly_factor,
            "confidence_factor": confidence_factor,
            "drawdown_factor": drawdown_factor,
            "regime_factor": regime_factor,
            "composite_weighted": composite,
            "intermediate_size": intermediate_size,
            "streak_multiplier": streak_multiplier,
            "streak_adjusted_size": streak_adjusted_size,
            "min_position_size": self.config.min_position_units,
            "max_position_size": max_position_size,
            "clamped": final_size != streak_adjusted_size,
        }

        result = PositionSizeResult(
            final_size=float(final_size),
            base_size=float(base_size),
            kelly_factor=float(kelly_factor),
            confidence_factor=float(confidence_factor),
            drawdown_factor=float(drawdown_factor),
            regime_factor=float(regime_factor),
            regime_name=regime_name,
            streak_multiplier=float(streak_multiplier),
            details=details,
        )

        logger.info(
            f"Position size: {final_size:.0f} units (base={base_size:.0f}, "
            f"kelly={kelly_factor:.3f}, conf={confidence_factor:.3f}, "
            f"dd={drawdown_factor:.3f}, regime={regime_factor:.3f}, "
            f"streak={streak_multiplier:.3f})"
        )

        return result

    def save_state(self, path: str) -> None:
        """
        Save sizer state (trade history + config + streak tracker) to JSON atomically.

        Writes to temporary file first, then renames to final path to ensure atomicity.

        Args:
            path: File path to save to

        Raises:
            IOError: If write fails after retries
        """
        state = {
            "config": asdict(self.config),
            "trade_history": list(self._trade_history),
            "streak_tracker": (
                self._streak_tracker.to_dict() if self._streak_tracker is not None else None
            ),
        }

        tmp_path = f"{path}.tmp"

        try:
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)

            os.rename(tmp_path, path)
            logger.info(f"Saved state to {path}")

        except IOError as e:
            logger.error(f"Failed to save state to {path}: {e}")
            # Clean up temp file if it exists
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def load_state(self, path: str) -> None:
        """
        Load sizer state (trade history + config + streak tracker) from JSON.

        Gracefully handles missing or corrupted files.

        Args:
            path: File path to load from

        Raises:
            ValueError: If state structure is invalid
        """
        if not os.path.exists(path):
            logger.warning(f"State file not found at {path}; using current state")
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read state from {path}: {e}")
            logger.warning("Using current state")
            return

        # Validate structure
        if not isinstance(state, dict):
            logger.error(f"State file does not contain a dict; got {type(state)}")
            return

        if "trade_history" not in state or "config" not in state:
            logger.error("State file missing required keys: 'trade_history', 'config'")
            return

        try:
            # Restore trade history
            trade_history = state["trade_history"]
            if not isinstance(trade_history, list):
                logger.error(f"trade_history must be list, got {type(trade_history)}")
                return

            self._trade_history.clear()
            for item in trade_history:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    logger.warning(f"Skipping malformed trade record: {item}")
                    continue
                self._trade_history.append((float(item[0]), bool(item[1])))

            logger.info(f"Loaded {len(self._trade_history)} trades from {path}")

            # Restore streak tracker state if present
            if self._streak_tracker is not None and "streak_tracker" in state:
                streak_data = state.get("streak_tracker")
                if streak_data is not None:
                    try:
                        self._streak_tracker = StreakTracker.from_dict(streak_data)
                        logger.info("Restored streak tracker state")
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error parsing streak_tracker: {e}")
                        # Continue with fresh streak tracker

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing state: {e}")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AdaptivePositionSizer(trades={len(self._trade_history)}, "
            f"kelly_window={self.config.kelly_window})"
        )


@dataclass
class StreakConfig:
    """Configuration for streak-aware anti-martingale sizing."""

    win_multiplier: float = 1.3  # Multiply size by this per consecutive win
    loss_divisor: float = 1.8  # Divide size by this per consecutive loss
    max_boost: float = 2.0  # Maximum streak multiplier (cap)
    min_floor: float = 0.3  # Minimum streak multiplier (floor)
    reset_on_regime_change: bool = True  # Reset streak when regime changes

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.win_multiplier <= 1.0:
            raise ValueError(
                f"win_multiplier must be > 1.0, got {self.win_multiplier}"
            )
        if self.loss_divisor <= 1.0:
            raise ValueError(
                f"loss_divisor must be > 1.0, got {self.loss_divisor}"
            )
        if self.max_boost < 1.0:
            raise ValueError(f"max_boost must be >= 1.0, got {self.max_boost}")
        if not (0.0 < self.min_floor <= 1.0):
            raise ValueError(f"min_floor must be in (0, 1], got {self.min_floor}")


class StreakTracker:
    """Tracks consecutive win/loss streaks and computes anti-martingale multiplier.

    Anti-Martingale: increase position after wins, decrease after losses.
    - After N consecutive wins: multiplier = min(win_mult^N, max_boost)
    - After N consecutive losses: multiplier = max(1/loss_div^N, min_floor)
    - Neutral (no streak): multiplier = 1.0
    """

    def __init__(self, config: Optional[StreakConfig] = None):
        """
        Initialize the streak tracker.

        Args:
            config: StreakConfig instance. If None, uses defaults.
        """
        self.config = config or StreakConfig()
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._last_regime = None
        self._total_trades = 0

    def record_outcome(self, win: bool, regime_name: str = "NORMAL") -> None:
        """Record trade outcome and update streak.

        Args:
            win: True if trade was profitable
            regime_name: Current market regime
        """
        # Reset streak on regime change if configured
        if (
            self.config.reset_on_regime_change
            and self._last_regime is not None
            and regime_name != self._last_regime
        ):
            logger.info(
                f"Phase 45: Streak reset on regime change "
                f"{self._last_regime} -> {regime_name}"
            )
            self._consecutive_wins = 0
            self._consecutive_losses = 0

        self._last_regime = regime_name
        self._total_trades += 1

        if win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def get_multiplier(self) -> float:
        """Get current anti-martingale multiplier.

        Returns:
            Multiplier in [min_floor, max_boost]. 1.0 when no active streak.
        """
        if self._consecutive_wins > 0:
            # Anti-martingale: increase after wins
            raw = self.config.win_multiplier ** self._consecutive_wins
            return min(raw, self.config.max_boost)
        elif self._consecutive_losses > 0:
            # Decrease after losses
            raw = 1.0 / (self.config.loss_divisor ** self._consecutive_losses)
            return max(raw, self.config.min_floor)
        else:
            return 1.0

    @property
    def streak_info(self) -> Dict[str, Any]:
        """Get current streak information."""
        return {
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "multiplier": round(self.get_multiplier(), 4),
            "last_regime": self._last_regime,
            "total_trades": self._total_trades,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "last_regime": self._last_regime,
            "total_trades": self._total_trades,
            "config": {
                "win_multiplier": self.config.win_multiplier,
                "loss_divisor": self.config.loss_divisor,
                "max_boost": self.config.max_boost,
                "min_floor": self.config.min_floor,
                "reset_on_regime_change": self.config.reset_on_regime_change,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreakTracker":
        """Deserialize from dict."""
        config_data = data.get("config", {})
        config = StreakConfig(**config_data) if config_data else StreakConfig()
        tracker = cls(config=config)
        tracker._consecutive_wins = int(data.get("consecutive_wins", 0))
        tracker._consecutive_losses = int(data.get("consecutive_losses", 0))
        tracker._last_regime = data.get("last_regime")
        tracker._total_trades = int(data.get("total_trades", 0))
        return tracker


# Factory Functions
def create_default_position_sizer() -> AdaptivePositionSizer:
    """
    Create a default adaptive position sizer.

    Uses balanced defaults:
    - Kelly fraction: 0.33
    - Drawdown floor: 0.3
    - Max risk per trade: 0.10 (10%)

    Returns:
        AdaptivePositionSizer with default configuration
    """
    config = AdaptivePositionSizingConfig()
    sizer = AdaptivePositionSizer(config)
    logger.info("Created default position sizer")
    return sizer


def create_conservative_position_sizer() -> AdaptivePositionSizer:
    """
    Create a conservative adaptive position sizer.

    More protective than default:
    - Kelly fraction: 0.25
    - Drawdown floor: 0.4
    - Max risk per trade: 0.05 (5%)

    Returns:
        AdaptivePositionSizer with conservative configuration
    """
    config = AdaptivePositionSizingConfig(
        kelly_fraction=0.25,
        drawdown_floor=0.4,
        max_risk_per_trade=0.05,
    )
    sizer = AdaptivePositionSizer(config)
    logger.info("Created conservative position sizer")
    return sizer


def create_aggressive_position_sizer() -> AdaptivePositionSizer:
    """
    Create an aggressive adaptive position sizer.

    More growth-oriented than default:
    - Kelly fraction: 0.40
    - Drawdown floor: 0.2
    - Max risk per trade: 0.15 (15%)

    Returns:
        AdaptivePositionSizer with aggressive configuration
    """
    config = AdaptivePositionSizingConfig(
        kelly_fraction=0.40,
        drawdown_floor=0.2,
        max_risk_per_trade=0.15,
    )
    sizer = AdaptivePositionSizer(config)
    logger.info("Created aggressive position sizer")
    return sizer
