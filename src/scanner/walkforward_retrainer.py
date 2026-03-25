"""
Walk-Forward Agent Weight Retrainer — Phase 51 (US-320).

Implements sliding-window walk-forward optimization for agent weights.
60-trade training window, 20-trade test window, slide by 20.

Key safeguards:
- WFE (Walk-Forward Efficiency) check: reject updates if OOS/IS < 0.50
- Weight stability check: reject if any agent weight changes > 30%
- Epoch history persistence for audit trail

Usage:
    retrainer = WalkForwardRetrainer()
    epochs = retrainer.run(journal_path="trained_data/trade_journal_rl.json")
    if epochs and epochs[-1].accepted:
        retrainer.apply_weights(epochs[-1].weights)
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default 12 agents with equal initial weights
DEFAULT_AGENTS = [
    "trend", "mean_reversion", "volatility", "risk_sentinel",
    "uncertainty", "execution_quality", "momentum", "news_risk",
    "multi_timeframe", "pair_performance", "session_timing",
    "support_resistance",
]


@dataclass
class WFEpoch:
    """Single walk-forward epoch result."""
    epoch_idx: int = 0
    train_start: int = 0
    train_end: int = 0
    test_start: int = 0
    test_end: int = 0
    n_train: int = 0
    n_test: int = 0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    wfe: float = 0.0                    # Walk-Forward Efficiency
    weights: Dict[str, float] = field(default_factory=dict)
    weight_change_max: float = 0.0      # Max % change from previous
    accepted: bool = False
    rejection_reason: str = ""


@dataclass
class WFConfig:
    """Walk-forward retraining configuration."""
    train_window: int = 60     # Number of trades for training
    test_window: int = 20      # Number of trades for testing
    slide_step: int = 20       # Slide by this many trades per epoch
    min_wfe: float = 0.50      # Minimum Walk-Forward Efficiency
    max_weight_change: float = 0.30   # Max 30% change per agent
    min_test_trades: int = 10         # Minimum test trades per epoch
    learning_rate: float = 0.15       # How much to adjust weights per outcome
    weight_floor: float = 0.02        # Minimum agent weight
    weight_ceiling: float = 0.25      # Maximum agent weight


class WalkForwardRetrainer:
    """Walk-forward agent weight retraining engine."""

    def __init__(self, config: Optional[WFConfig] = None):
        self.config = config or WFConfig()
        self._current_weights: Dict[str, float] = {}
        self._epoch_history: List[WFEpoch] = []

    def run(
        self,
        journal_path: str,
        current_weights_path: str = "trained_data/models/agent_weights.json",
        output_path: str = "trained_data/models/walkforward_epochs.json",
    ) -> List[WFEpoch]:
        """Run full walk-forward optimization across journal trades.

        Args:
            journal_path: Path to trade_journal_rl.json
            current_weights_path: Current agent weights file
            output_path: Where to save epoch history

        Returns:
            List of WFEpoch results.
        """
        trades = self._load_trades(journal_path)
        if len(trades) < self.config.train_window + self.config.test_window:
            logger.warning(
                "Phase 51 US-320: Only %d trades — need %d+ for walk-forward",
                len(trades), self.config.train_window + self.config.test_window,
            )
            return []

        # Load current weights as baseline
        self._current_weights = self._load_weights(current_weights_path)
        if not self._current_weights:
            self._current_weights = {a: 1.0 / len(DEFAULT_AGENTS) for a in DEFAULT_AGENTS}

        epochs = []
        previous_weights = dict(self._current_weights)
        epoch_idx = 0

        # Sliding window
        start = 0
        while start + self.config.train_window + self.config.test_window <= len(trades):
            train_end = start + self.config.train_window
            test_end = train_end + self.config.test_window

            train_trades = trades[start:train_end]
            test_trades = trades[train_end:test_end]

            if len(test_trades) < self.config.min_test_trades:
                break

            # Retrain on training window
            new_weights = self._retrain_weights(train_trades, dict(previous_weights))
            train_wr = self._compute_win_rate(train_trades)
            test_wr = self._compute_weighted_win_rate(test_trades, new_weights)

            # Walk-Forward Efficiency
            wfe = test_wr / train_wr if train_wr > 0 else 0.0

            # Weight stability check
            max_change = self._max_weight_change(previous_weights, new_weights)

            # Acceptance criteria
            accepted = True
            reason = ""
            if wfe < self.config.min_wfe:
                accepted = False
                reason = f"WFE {wfe:.3f} < {self.config.min_wfe} (overfitting)"
            elif max_change > self.config.max_weight_change:
                accepted = False
                reason = f"Weight drift {max_change:.3f} > {self.config.max_weight_change} (unstable)"

            epoch = WFEpoch(
                epoch_idx=epoch_idx,
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
                n_train=len(train_trades),
                n_test=len(test_trades),
                train_win_rate=round(train_wr, 4),
                test_win_rate=round(test_wr, 4),
                wfe=round(wfe, 4),
                weights={k: round(v, 6) for k, v in new_weights.items()},
                weight_change_max=round(max_change, 4),
                accepted=accepted,
                rejection_reason=reason,
            )
            epochs.append(epoch)

            if accepted:
                previous_weights = dict(new_weights)
                logger.info(
                    "Phase 51 US-320: Epoch %d ACCEPTED — WFE %.3f, test WR %.3f, max_change %.3f",
                    epoch_idx, wfe, test_wr, max_change,
                )
            else:
                logger.info(
                    "Phase 51 US-320: Epoch %d REJECTED — %s",
                    epoch_idx, reason,
                )

            start += self.config.slide_step
            epoch_idx += 1

        self._epoch_history = epochs

        # Save epoch history
        self._save_epochs(epochs, output_path)

        # If last accepted epoch has weights, update current
        last_accepted = None
        for e in reversed(epochs):
            if e.accepted:
                last_accepted = e
                break

        if last_accepted:
            self._current_weights = dict(last_accepted.weights)
            logger.info(
                "Phase 51 US-320: Final weights from epoch %d (WFE %.3f)",
                last_accepted.epoch_idx, last_accepted.wfe,
            )

        return epochs

    def get_current_weights(self) -> Dict[str, float]:
        """Return current (possibly retrained) weights."""
        return dict(self._current_weights)

    def save_weights(self, path: str) -> bool:
        """Save retrained weights atomically."""
        if not self._current_weights:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "source": "walkforward_retraining",
                "version": 2,
                "weights": self._current_weights,
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            return True
        except Exception as e:
            logger.error("Phase 51: Failed to save retrained weights: %s", e)
            return False

    # ── Private methods ──────────────────────────────────────────────

    def _load_trades(self, path: str) -> List[Dict[str, Any]]:
        """Load trades with agent verdict data."""
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Phase 51: Failed to load journal: %s", e)
            return []

        if not isinstance(raw, list):
            return []

        valid = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            outcome = entry.get("outcome")
            if isinstance(outcome, dict):
                if "trade_won" in outcome:
                    valid.append(entry)
            elif isinstance(outcome, str):
                valid.append(entry)
        return valid

    def _load_weights(self, path: str) -> Dict[str, float]:
        """Load current agent weights."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                weights = data.get("weights", data)
                if isinstance(weights, dict):
                    return {k: float(v) for k, v in weights.items()
                            if isinstance(v, (int, float))}
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {}

    def _retrain_weights(
        self, trades: List[Dict], base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Retrain agent weights using outcome-weighted Bayesian update.

        For each trade:
        - If won: increase weights of agents that voted correctly
        - If lost: decrease weights of agents that voted correctly (they were wrong)
        - Normalize to sum to 1.0
        """
        weights = dict(base_weights)
        lr = self.config.learning_rate

        for trade in trades:
            outcome = trade.get("outcome", {})
            if isinstance(outcome, dict):
                won = bool(outcome.get("trade_won", False))
            elif isinstance(outcome, str):
                won = outcome.lower() == "win"
            else:
                continue

            # Extract agent verdicts
            verdicts = trade.get("agent_verdicts", {})
            direction = trade.get("direction", "LONG")

            for agent_name in weights:
                agent_data = verdicts.get(agent_name, {})
                if isinstance(agent_data, dict):
                    agent_vote = agent_data.get("verdict", "HOLD")
                elif isinstance(agent_data, (tuple, list)) and len(agent_data) >= 1:
                    agent_vote = agent_data[0]
                else:
                    continue

                # Did agent agree with trade direction?
                agent_aligned = (
                    (agent_vote in ("BUY", "LONG") and direction in ("BUY", "LONG"))
                    or (agent_vote in ("SELL", "SHORT") and direction in ("SELL", "SHORT"))
                )

                if won and agent_aligned:
                    # Reward: agent voted correctly and trade won
                    weights[agent_name] *= (1.0 + lr)
                elif not won and agent_aligned:
                    # Penalize: agent voted for this trade but it lost
                    weights[agent_name] *= (1.0 - lr * 0.5)
                elif not won and not agent_aligned:
                    # Reward contrarian: agent disagreed and trade lost
                    weights[agent_name] *= (1.0 + lr * 0.3)

                # Clamp
                weights[agent_name] = max(
                    self.config.weight_floor,
                    min(self.config.weight_ceiling, weights[agent_name])
                )

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _compute_win_rate(self, trades: List[Dict]) -> float:
        """Compute simple win rate from trades."""
        if not trades:
            return 0.0
        wins = 0
        for t in trades:
            outcome = t.get("outcome", {})
            if isinstance(outcome, dict):
                if outcome.get("trade_won", False):
                    wins += 1
            elif isinstance(outcome, str) and outcome.lower() == "win":
                wins += 1
        return wins / len(trades)

    def _compute_weighted_win_rate(
        self, trades: List[Dict], weights: Dict[str, float]
    ) -> float:
        """Compute win rate weighted by agent consensus alignment.

        Trades where high-weight agents aligned are counted more.
        """
        if not trades:
            return 0.0

        weighted_wins = 0.0
        total_weight = 0.0

        for t in trades:
            outcome = t.get("outcome", {})
            if isinstance(outcome, dict):
                won = bool(outcome.get("trade_won", False))
            elif isinstance(outcome, str):
                won = outcome.lower() == "win"
            else:
                continue

            # Weight this trade by how aligned agents were
            verdicts = t.get("agent_verdicts", {})
            direction = t.get("direction", "LONG")
            alignment_score = 0.0

            for agent_name, w in weights.items():
                agent_data = verdicts.get(agent_name, {})
                if isinstance(agent_data, dict):
                    v = agent_data.get("verdict", "HOLD")
                elif isinstance(agent_data, (tuple, list)):
                    v = agent_data[0] if agent_data else "HOLD"
                else:
                    continue
                if (v in ("BUY", "LONG") and direction in ("BUY", "LONG")) or \
                   (v in ("SELL", "SHORT") and direction in ("SELL", "SHORT")):
                    alignment_score += w

            trade_weight = max(0.1, alignment_score)
            total_weight += trade_weight
            if won:
                weighted_wins += trade_weight

        return weighted_wins / total_weight if total_weight > 0 else 0.0

    def _max_weight_change(
        self, old: Dict[str, float], new: Dict[str, float]
    ) -> float:
        """Compute maximum absolute weight change across agents."""
        max_change = 0.0
        for agent in old:
            old_w = old.get(agent, 0.0)
            new_w = new.get(agent, 0.0)
            if old_w > 0:
                change = abs(new_w - old_w) / old_w
                max_change = max(max_change, change)
        return max_change

    def _save_epochs(self, epochs: List[WFEpoch], path: str) -> None:
        """Save epoch history atomically."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "version": 1,
                "n_epochs": len(epochs),
                "n_accepted": sum(1 for e in epochs if e.accepted),
                "epochs": [asdict(e) for e in epochs],
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            logger.info("Phase 51: Saved %d walk-forward epochs to %s", len(epochs), path)
        except Exception as e:
            logger.error("Phase 51: Failed to save epochs: %s", e)
