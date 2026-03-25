"""Synthetic data augmentation for retrain robustness.

US-107: Wire observational_learning (US-078) synthetic agents into the
retrain pipeline. Synthetic scenarios are generated but never injected
into training data. This module blends synthetic examples (15-20% ratio)
into retrain datasets to improve model robustness on rare regimes.

Approach:
1. Load real trades from trade_journal_rl.json
2. Generate synthetic scenarios using observational learning patterns
3. Blend at configurable ratio (default 20% synthetic)
4. Weight synthetic examples lower than real (synthetic_weight = 0.50)
5. Track augmentation statistics for observability
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "augmentation_log.json"

# Augmentation parameters
SYNTHETIC_RATIO = 0.20  # 20% of augmented dataset is synthetic
SYNTHETIC_WEIGHT = 0.50  # Synthetic examples weighted at 50% of real
MIN_REAL_TRADES = 20  # Minimum real trades before augmentation
REGIME_BALANCE_TARGET = 0.25  # Target proportion per regime

# Synthetic scenario templates (simplified behavioral patterns)
SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "momentum_chase": {
        "direction_bias": 1.0,  # Follow momentum
        "confidence_range": (0.55, 0.80),
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "win_rate": 0.45,
        "preferred_regimes": ["NORMAL", "HIGH"],
    },
    "mean_reversion": {
        "direction_bias": -1.0,  # Fade moves
        "confidence_range": (0.50, 0.70),
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.5,
        "win_rate": 0.55,
        "preferred_regimes": ["LOW", "NORMAL"],
    },
    "risk_averse": {
        "direction_bias": 0.0,  # Neutral
        "confidence_range": (0.60, 0.90),
        "sl_atr_mult": 0.8,
        "tp_atr_mult": 1.2,
        "win_rate": 0.60,
        "preferred_regimes": ["LOW"],
    },
    "crisis_adaptive": {
        "direction_bias": -0.5,  # Slightly bearish
        "confidence_range": (0.40, 0.65),
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "win_rate": 0.35,
        "preferred_regimes": ["HIGH", "EXTREME"],
    },
    "breakout_hunter": {
        "direction_bias": 1.0,
        "confidence_range": (0.65, 0.85),
        "sl_atr_mult": 1.2,
        "tp_atr_mult": 2.5,
        "win_rate": 0.40,
        "preferred_regimes": ["NORMAL", "HIGH"],
    },
}


@dataclass
class AugmentationStats:
    """Statistics for an augmentation run."""

    n_real: int = 0
    n_synthetic: int = 0
    n_total: int = 0
    synthetic_ratio: float = 0.0
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    scenario_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "n_real": self.n_real,
            "n_synthetic": self.n_synthetic,
            "n_total": self.n_total,
            "synthetic_ratio": round(self.synthetic_ratio, 4),
            "regime_distribution": self.regime_distribution,
            "scenario_distribution": self.scenario_distribution,
            "timestamp": self.timestamp,
        }


class TrainingAugmenter:
    """Augments training datasets with synthetic scenarios.

    Bridges observational_learning (which defines behavioral patterns)
    and the retrain pipeline (which needs diverse training data).
    Generates synthetic trade scenarios to improve model robustness
    on underrepresented regimes.

    Usage:
        augmenter = TrainingAugmenter()
        augmented = augmenter.generate_augmented_dataset(real_trades, "EUR_USD", "HIGH")
        # augmented contains real trades + 20% synthetic scenarios
    """

    def __init__(
        self,
        synthetic_ratio: float = SYNTHETIC_RATIO,
        persistence_path: Optional[Path] = None,
    ):
        self.synthetic_ratio = synthetic_ratio
        self.persistence_path = persistence_path or _LOG_PATH

        # Augmentation history
        self._stats_log: List[Dict[str, Any]] = []
        self._total_augmentations: int = 0
        self._total_synthetic_generated: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_augmented_dataset(
        self,
        real_trades: List[Dict[str, Any]],
        pair: str,
        regime: str = "NORMAL",
    ) -> List[Dict[str, Any]]:
        """Generate augmented dataset with synthetic scenarios.

        Args:
            real_trades: List of real trade dictionaries from journal.
            pair: Currency pair for context.
            regime: Current regime for scenario generation.

        Returns:
            Augmented list with real + synthetic trades.
        """
        if len(real_trades) < MIN_REAL_TRADES:
            logger.debug(
                f"TrainingAugmenter: Too few real trades "
                f"({len(real_trades)} < {MIN_REAL_TRADES})"
            )
            return real_trades

        # Compute number of synthetic examples needed
        n_synthetic = max(1, int(len(real_trades) * self.synthetic_ratio / (1 - self.synthetic_ratio)))

        # Analyze regime distribution of real trades
        regime_counts = self._count_regimes(real_trades)

        # Generate synthetic scenarios, prioritizing underrepresented regimes
        synthetic_trades = self._generate_synthetic(
            n_synthetic, pair, regime, regime_counts
        )

        # Mark synthetic trades
        for trade in synthetic_trades:
            trade["is_synthetic"] = True
            trade["sample_weight"] = SYNTHETIC_WEIGHT

        # Mark real trades
        for trade in real_trades:
            trade.setdefault("is_synthetic", False)
            trade.setdefault("sample_weight", 1.0)

        # Combine
        augmented = list(real_trades) + synthetic_trades

        # Log stats
        self._total_augmentations += 1
        self._total_synthetic_generated += len(synthetic_trades)

        stats = AugmentationStats(
            n_real=len(real_trades),
            n_synthetic=len(synthetic_trades),
            n_total=len(augmented),
            synthetic_ratio=len(synthetic_trades) / max(len(augmented), 1),
            regime_distribution=self._count_regimes(augmented),
            scenario_distribution=self._count_scenarios(synthetic_trades),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._stats_log.append(stats.to_dict())
        if len(self._stats_log) > 100:
            self._stats_log = self._stats_log[-50:]

        logger.info(
            f"TrainingAugmenter: {pair} — {stats.n_real} real + "
            f"{stats.n_synthetic} synthetic = {stats.n_total} total"
        )

        return augmented

    def get_regime_balance_report(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Analyze regime balance of a dataset.

        Returns:
            Dict mapping regime → proportion.
        """
        counts = self._count_regimes(trades)
        total = sum(counts.values()) or 1
        return {r: count / total for r, count in counts.items()}

    def get_status(self) -> Dict[str, Any]:
        """Get augmenter status for observability."""
        return {
            "total_augmentations": self._total_augmentations,
            "total_synthetic_generated": self._total_synthetic_generated,
            "synthetic_ratio": self.synthetic_ratio,
            "recent_stats": self._stats_log[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_synthetic(
        self,
        n: int,
        pair: str,
        current_regime: str,
        regime_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Generate N synthetic trade scenarios."""
        synthetic = []

        # Identify underrepresented regimes
        total = sum(regime_counts.values()) or 1
        regime_proportions = {r: c / total for r, c in regime_counts.items()}
        underrep_regimes = [
            r for r, p in regime_proportions.items()
            if p < REGIME_BALANCE_TARGET
        ]

        # Add missing regimes
        all_regimes = ["LOW", "NORMAL", "HIGH", "EXTREME"]
        for r in all_regimes:
            if r not in regime_proportions:
                underrep_regimes.append(r)

        for i in range(n):
            # Select scenario template
            scenario_name = random.choice(list(SCENARIO_TEMPLATES.keys()))
            template = SCENARIO_TEMPLATES[scenario_name]

            # Select regime (prioritize underrepresented)
            if underrep_regimes and random.random() < 0.60:
                regime = random.choice(underrep_regimes)
            else:
                regime = current_regime

            # Generate synthetic trade
            confidence = random.uniform(*template["confidence_range"])
            is_win = random.random() < template["win_rate"]

            trade = {
                "pair": pair,
                "direction": "BUY" if template["direction_bias"] >= 0 else "SELL",
                "confidence": round(confidence, 4),
                "regime": regime,
                "sl_atr_mult": template["sl_atr_mult"],
                "tp_atr_mult": template["tp_atr_mult"],
                "outcome": "win" if is_win else "loss",
                "pnl_pips": round(
                    random.uniform(5, 30) if is_win else random.uniform(-20, -5), 1
                ),
                "scenario_type": scenario_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            synthetic.append(trade)

        return synthetic

    def _count_regimes(self, trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count regime distribution in trades."""
        counts: Dict[str, int] = {}
        for t in trades:
            regime = t.get("regime", "NORMAL")
            counts[regime] = counts.get(regime, 0) + 1
        return counts

    def _count_scenarios(self, trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count scenario type distribution in synthetic trades."""
        counts: Dict[str, int] = {}
        for t in trades:
            scenario = t.get("scenario_type", "unknown")
            counts[scenario] = counts.get(scenario, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist augmentation statistics."""
        try:
            data = {
                "version": 1,
                "total_augmentations": self._total_augmentations,
                "total_synthetic_generated": self._total_synthetic_generated,
                "stats_log": self._stats_log[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"TrainingAugmenter: Failed to save: {e}")

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
                self._total_augmentations = data.get("total_augmentations", 0)
                self._total_synthetic_generated = data.get("total_synthetic_generated", 0)
                self._stats_log = data.get("stats_log", [])
        except Exception as e:
            logger.debug(f"TrainingAugmenter: Failed to load: {e}")
