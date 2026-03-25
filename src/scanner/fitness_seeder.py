"""
Strategy Fitness Seeder — Phase 50 (US-313).

Seeds the StrategyFitnessDetector with per-pair fitness scores
calculated from historical trade journal data.

Usage:
    seeder = FitnessSeeder()
    result = seeder.seed_from_journal("trained_data/trade_journal_rl.json")
    seeder.save_state(result, "trained_data/models/strategy_fitness_state.json")
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PairFitness:
    """Fitness metrics for a single pair."""
    pair: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float      # sum_wins / sum_losses (in pips)
    avg_pnl_pips: float
    avg_duration_min: float
    fitness_score: float       # Composite 0.0-1.0
    action: str                # NORMAL, REDUCE_SIZE, SKIP
    consecutive_losses: int    # Max streak
    last_5_win_rate: float     # Recent form


@dataclass
class FitnessSeederConfig:
    """Configuration for fitness seeding."""
    min_trades_per_pair: int = 10
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "profit_factor": 0.40,
        "win_rate": 0.35,
        "recent_form": 0.25,
    })
    skip_threshold: float = 0.30
    reduce_threshold: float = 0.50


class FitnessSeeder:
    """Seeds strategy fitness from historical trades."""

    def __init__(self, config: Optional[FitnessSeederConfig] = None):
        self.config = config or FitnessSeederConfig()

    def seed_from_journal(self, journal_path: str) -> Dict[str, PairFitness]:
        """Load journal and compute per-pair fitness."""
        trades = self._load_trades(journal_path)
        if not trades:
            return {}

        # Group by pair
        by_pair: Dict[str, List[Dict]] = {}
        for t in trades:
            pair = t.get("pair", "unknown")
            by_pair.setdefault(pair, []).append(t)

        results = {}
        for pair, pair_trades in by_pair.items():
            if len(pair_trades) < self.config.min_trades_per_pair:
                logger.debug(f"Phase 50: Skipping {pair} — only {len(pair_trades)} trades")
                continue
            fitness = self._compute_pair_fitness(pair, pair_trades)
            results[pair] = fitness

        logger.info(f"Phase 50: Seeded fitness for {len(results)} pairs")
        return results

    def _load_trades(self, path: str) -> List[Dict[str, Any]]:
        """Load and validate journal entries."""
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Phase 50: Failed to load journal: {e}")
            return []

        if not isinstance(raw, list):
            return []

        valid = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            outcome = entry.get("outcome")
            if not isinstance(outcome, dict) and not isinstance(outcome, str):
                continue
            valid.append(entry)
        return valid

    def _compute_pair_fitness(self, pair: str, trades: List[Dict]) -> PairFitness:
        """Compute fitness score for a single pair."""
        total = len(trades)
        wins = 0
        losses = 0
        sum_win_pips = 0.0
        sum_loss_pips = 0.0
        durations = []
        outcomes_sequence = []  # True=win, False=loss

        for t in trades:
            outcome = t.get("outcome", {})
            if isinstance(outcome, dict):
                won = outcome.get("trade_won", False)
                pnl = outcome.get("pnl_pips", 0.0)
                dur = outcome.get("duration_minutes", 0.0)
            elif isinstance(outcome, str):
                won = outcome.lower() == "win"
                pnl = 0.0
                dur = 0.0
            else:
                continue

            if not isinstance(pnl, (int, float)):
                pnl = 0.0
            if not isinstance(dur, (int, float)):
                dur = 0.0

            outcomes_sequence.append(bool(won))
            if won:
                wins += 1
                sum_win_pips += abs(pnl)
            else:
                losses += 1
                sum_loss_pips += abs(pnl)
            durations.append(dur)

        win_rate = wins / total if total > 0 else 0.0

        # Profit factor
        if sum_loss_pips > 0:
            profit_factor = sum_win_pips / sum_loss_pips
        elif sum_win_pips > 0:
            profit_factor = 3.0  # Cap when no losses
        else:
            profit_factor = 0.0

        avg_pnl = (sum_win_pips - sum_loss_pips) / total if total > 0 else 0.0
        avg_dur = sum(durations) / len(durations) if durations else 0.0

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for won in outcomes_sequence:
            if not won:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        # Last 5 win rate
        last_5 = outcomes_sequence[-5:] if len(outcomes_sequence) >= 5 else outcomes_sequence
        last_5_wr = sum(1 for w in last_5 if w) / len(last_5) if last_5 else 0.0

        # Composite fitness score
        w = self.config.fitness_weights
        # Normalize profit factor to 0-1 range (PF of 2.0 → 1.0, PF of 0.5 → 0.25)
        pf_norm = min(profit_factor / 2.0, 1.0)
        fitness_score = (
            w.get("profit_factor", 0.4) * pf_norm
            + w.get("win_rate", 0.35) * win_rate
            + w.get("recent_form", 0.25) * last_5_wr
        )
        fitness_score = round(max(0.0, min(1.0, fitness_score)), 4)

        # Determine action
        if fitness_score < self.config.skip_threshold:
            action = "SKIP"
        elif fitness_score < self.config.reduce_threshold:
            action = "REDUCE_SIZE"
        else:
            action = "NORMAL"

        return PairFitness(
            pair=pair,
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            avg_pnl_pips=round(avg_pnl, 2),
            avg_duration_min=round(avg_dur, 1),
            fitness_score=fitness_score,
            action=action,
            consecutive_losses=max_consec_losses,
            last_5_win_rate=round(last_5_wr, 4),
        )

    def save_state(self, results: Dict[str, PairFitness], path: str) -> bool:
        """Save seeded fitness state atomically."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "seeded": True,
                "source": "historical_journal",
                "pairs": {k: asdict(v) for k, v in results.items()},
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp, path)
            logger.info(f"Phase 50: Saved fitness state for {len(results)} pairs to {path}")
            return True
        except Exception as e:
            logger.error(f"Phase 50: Failed to save fitness state: {e}")
            return False

    @staticmethod
    def load_state(path: str) -> Optional[Dict[str, PairFitness]]:
        """Load seeded fitness state from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not data.get("seeded"):
                return None
            pairs = {}
            for k, v in data.get("pairs", {}).items():
                pairs[k] = PairFitness(**v)
            return pairs
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Phase 50: No fitness state at {path}: {e}")
            return None


def create_default_fitness_seeder() -> FitnessSeeder:
    """Create a FitnessSeeder with default config."""
    return FitnessSeeder(FitnessSeederConfig())
