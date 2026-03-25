"""Adaptive learning rate scheduling for online RL.

US-091: Replace fixed RL learning rate with adaptive cosine-annealing +
warmup schedule. Three phases:
1. Warmup (first N trades): linear ramp from min_lr to target_lr
2. Cosine decay: target_lr decays to min_lr over decay_period trades
3. Plateau restart: if loss stalls, reset to restart_lr and mini-cycle

Tracks LR vs convergence metrics. Based on 2025 meta-RL research showing
15-25% faster convergence with adaptive schedules.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LR_LOG_PATH = _PROJECT_ROOT / "trained_data" / "adaptive_lr_log.json"

# Default schedule parameters
DEFAULT_WARMUP_TRADES = 20
DEFAULT_TARGET_LR = 0.02
DEFAULT_MIN_LR = 0.001
DEFAULT_RESTART_LR = 0.005
DEFAULT_DECAY_PERIOD = 200
DEFAULT_PLATEAU_WINDOW = 15
DEFAULT_PLATEAU_THRESHOLD = 0.005  # Min improvement to not be plateau


@dataclass
class LRState:
    """Current state of the adaptive learning rate scheduler."""

    current_lr: float = 0.02
    phase: str = "warmup"     # warmup, cosine_decay, plateau_restart
    trade_count: int = 0
    restart_count: int = 0
    plateau_detected: bool = False
    convergence_score: float = 0.0  # Rolling measure of improvement rate

    def to_dict(self) -> dict:
        return {
            "current_lr": round(self.current_lr, 6),
            "phase": self.phase,
            "trade_count": self.trade_count,
            "restart_count": self.restart_count,
            "plateau_detected": self.plateau_detected,
            "convergence_score": round(self.convergence_score, 6),
        }


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler for online RL weight updates.

    Provides three-phase scheduling:
    - Warmup: Stabilizes initial random weights with low LR
    - Cosine decay: Main training with gradually decreasing LR
    - Plateau restart: Mini-optimization cycles when learning stalls

    Usage:
        scheduler = AdaptiveLRScheduler()
        for trade in trades:
            lr = scheduler.get_lr(trade_idx, loss_history)
            weight_update = gradient * lr
            scheduler.step(loss)
    """

    def __init__(
        self,
        warmup_trades: int = DEFAULT_WARMUP_TRADES,
        target_lr: float = DEFAULT_TARGET_LR,
        min_lr: float = DEFAULT_MIN_LR,
        restart_lr: float = DEFAULT_RESTART_LR,
        decay_period: int = DEFAULT_DECAY_PERIOD,
        plateau_window: int = DEFAULT_PLATEAU_WINDOW,
        plateau_threshold: float = DEFAULT_PLATEAU_THRESHOLD,
        persistence_path: Optional[Path] = None,
    ):
        self.warmup_trades = warmup_trades
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.restart_lr = restart_lr
        self.decay_period = decay_period
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.persistence_path = persistence_path or _LR_LOG_PATH

        # Internal state
        self._trade_count: int = 0
        self._phase: str = "warmup"
        self._restart_count: int = 0
        self._cosine_start_trade: int = warmup_trades
        self._current_lr: float = min_lr

        # Loss history for plateau detection
        self._loss_history: deque = deque(maxlen=100)

        # LR history for diagnostics
        self._lr_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr(
        self,
        trade_idx: Optional[int] = None,
        loss_history: Optional[List[float]] = None,
    ) -> float:
        """Get the current learning rate based on schedule phase.

        Args:
            trade_idx: Current trade index (uses internal counter if None)
            loss_history: Recent loss values (uses internal if None)

        Returns:
            Current learning rate
        """
        idx = trade_idx if trade_idx is not None else self._trade_count

        # Phase 1: Warmup (linear ramp)
        if idx < self.warmup_trades:
            self._phase = "warmup"
            progress = idx / max(1, self.warmup_trades)
            self._current_lr = self.min_lr + (self.target_lr - self.min_lr) * progress

        # Check for plateau before cosine decay
        elif self.detect_plateau(loss_history):
            self._phase = "plateau_restart"
            self._restart_count += 1
            self._cosine_start_trade = idx
            self._current_lr = self.restart_lr
            logger.info(
                f"AdaptiveLR: Plateau restart #{self._restart_count} "
                f"at trade {idx}, LR reset to {self.restart_lr:.4f}"
            )

        # Phase 2: Cosine decay
        else:
            self._phase = "cosine_decay"
            elapsed = idx - self._cosine_start_trade
            progress = min(1.0, elapsed / max(1, self.decay_period))
            cosine_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
            self._current_lr = self.min_lr + (self.target_lr - self.min_lr) * cosine_factor

        # Clamp
        self._current_lr = max(self.min_lr, min(self.target_lr, self._current_lr))

        return self._current_lr

    def step(self, loss: Optional[float] = None) -> LRState:
        """Advance the scheduler by one trade.

        Args:
            loss: Loss value for this trade (for plateau detection)

        Returns:
            Current LRState
        """
        self._trade_count += 1

        if loss is not None:
            self._loss_history.append(loss)

        state = LRState(
            current_lr=self._current_lr,
            phase=self._phase,
            trade_count=self._trade_count,
            restart_count=self._restart_count,
            plateau_detected=self._phase == "plateau_restart",
            convergence_score=self._compute_convergence_score(),
        )

        # Log periodically
        if self._trade_count % 10 == 0:
            self._lr_history.append({
                "trade": self._trade_count,
                "lr": round(self._current_lr, 6),
                "phase": self._phase,
                "convergence": round(state.convergence_score, 6),
            })

        return state

    def detect_plateau(
        self,
        loss_history: Optional[List[float]] = None,
    ) -> bool:
        """Detect if learning has plateaued.

        Compares mean of recent losses to mean of previous window.
        If improvement is below threshold, plateau is detected.

        Args:
            loss_history: External loss history (uses internal if None)

        Returns:
            True if plateau detected
        """
        history = loss_history or list(self._loss_history)

        if len(history) < self.plateau_window * 2:
            return False

        recent = history[-self.plateau_window:]
        previous = history[-(self.plateau_window * 2):-self.plateau_window]

        recent_mean = float(np.mean(recent))
        previous_mean = float(np.mean(previous))

        if previous_mean == 0:
            return False

        improvement = (previous_mean - recent_mean) / abs(previous_mean)
        return improvement < self.plateau_threshold

    def get_state(self) -> LRState:
        """Get current scheduler state."""
        return LRState(
            current_lr=self._current_lr,
            phase=self._phase,
            trade_count=self._trade_count,
            restart_count=self._restart_count,
            plateau_detected=self._phase == "plateau_restart",
            convergence_score=self._compute_convergence_score(),
        )

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self._trade_count = 0
        self._phase = "warmup"
        self._restart_count = 0
        self._cosine_start_trade = self.warmup_trades
        self._current_lr = self.min_lr
        self._loss_history.clear()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_convergence_score(self) -> float:
        """Compute rolling convergence speed metric.

        Returns a score 0-1 where 1 = rapidly improving, 0 = stalled.
        """
        if len(self._loss_history) < 10:
            return 0.5  # Neutral until enough data

        recent = list(self._loss_history)[-10:]
        # Linear regression slope of recent losses
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])

        # Normalize: negative slope (improving) = high score
        # Clamp to 0-1
        score = max(0.0, min(1.0, 0.5 - slope * 10))
        return score

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist LR schedule history."""
        try:
            data = {
                "version": 1,
                "state": self.get_state().to_dict(),
                "lr_history": self._lr_history[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AdaptiveLR: Failed to save log: {e}")

    def load_state(self) -> None:
        """Load persisted scheduler state."""
        if not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text())
            state = data.get("state", {})
            self._trade_count = state.get("trade_count", 0)
            self._phase = state.get("phase", "warmup")
            self._restart_count = state.get("restart_count", 0)
            self._current_lr = state.get("current_lr", self.min_lr)
            self._lr_history = data.get("lr_history", [])
            logger.debug(
                f"AdaptiveLR: Loaded state — trade={self._trade_count}, "
                f"phase={self._phase}, lr={self._current_lr:.4f}"
            )
        except Exception as e:
            logger.debug(f"AdaptiveLR: Failed to load state: {e}")
