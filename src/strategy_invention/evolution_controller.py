"""High-level orchestrator for offline strategy evolution.

Manages the full lifecycle:
1. Load existing strategy population from disk
2. Run evolution cycle (GA + backtest)
3. Save results
4. Promote shadow strategies to active
5. Retire underperforming strategies
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .blocks import StrategyBlock, BlockType
from .template import StrategyTemplate, BacktestStats
from .grammar import StrategyGrammar
from .genetic_algorithm import GeneticAlgorithm

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_STRATEGIES_DIR = _PROJECT_ROOT / "trained_data" / "strategies"

SHADOW_TRADES_REQUIRED = 50      # trades in shadow before promotion
SHADOW_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement over baseline
MIN_FITNESS_TO_DEPLOY = 1.1      # minimum profit factor
RETIRE_SHARPE_THRESHOLD = 0.5   # retire if Sharpe < this
RETIRE_PF_THRESHOLD = 0.9       # retire if profit factor < this


class EvolutionController:
    """Orchestrates strategy evolution: GA + backtest + registry.

    Usage (offline, non-blocking):
        ctrl = EvolutionController()
        ctrl.load()
        top_strategies = ctrl.run_evolution_cycle(fitness_fn)
        ctrl.save()
    """

    def __init__(self, strategies_dir: Optional[Path] = None):
        self._dir = strategies_dir or _STRATEGIES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, StrategyTemplate] = {}   # id → template

    def load(self) -> None:
        """Load strategy registry from disk."""
        manifest = self._dir / "manifest.json"
        if not manifest.exists():
            logger.info("EvolutionController: no manifest found — starting fresh")
            return
        try:
            with open(manifest) as f:
                data = json.load(f)
            for entry in data.get("strategies", []):
                strategy_path = self._dir / f"{entry['id']}.json"
                if not strategy_path.exists():
                    continue
                try:
                    with open(strategy_path) as sf:
                        s = StrategyTemplate.from_dict(json.load(sf))
                        self._registry[s.id] = s
                except Exception as e:
                    logger.error("EvolutionController: failed to load strategy %s: %s", entry['id'], e)
            logger.info("EvolutionController: loaded %d strategies", len(self._registry))
        except Exception as e:
            logger.error("EvolutionController: load failed: %s — starting fresh", e)

    def save(self) -> None:
        """Persist all strategies to disk atomically."""
        try:
            manifest_entries = []
            for strategy in self._registry.values():
                path = self._dir / f"{strategy.id}.json"
                payload = json.dumps(strategy.to_dict(), indent=2, sort_keys=True)
                fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
                try:
                    os.write(fd, payload.encode())
                    os.fsync(fd)
                    os.close(fd)
                    os.rename(tmp, str(path))
                except Exception as e:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                    logger.error("EvolutionController: failed to save strategy %s: %s", strategy.id, e)
                    raise
                manifest_entries.append({"id": strategy.id, "status": strategy.status, "fitness": strategy.fitness})

            manifest_payload = json.dumps({"version": "1.0", "strategies": manifest_entries}, indent=2, sort_keys=True)
            mpath = self._dir / "manifest.json"
            fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
            try:
                os.write(fd, manifest_payload.encode())
                os.fsync(fd)
                os.close(fd)
                os.rename(tmp, str(mpath))
            except Exception as e:
                try:
                    os.close(fd)
                except Exception:
                    pass
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                logger.error("EvolutionController: failed to save manifest: %s", e)
                raise
            logger.info("EvolutionController: saved %d strategies", len(self._registry))
        except Exception as e:
            logger.error("EvolutionController: save failed: %s", e)

    def run_evolution_cycle(
        self,
        fitness_fn: Callable[[StrategyTemplate], float],
        generations: int = 5,
        population_size: int = 30,
        seed_from_registry: bool = True,
    ) -> List[StrategyTemplate]:
        """Run full evolution cycle. Returns top 5 strategies."""
        ga = GeneticAlgorithm(
            population_size=population_size,
            generations=generations,
        )

        # Seed with existing good strategies
        seeds: List[StrategyTemplate] = []
        if seed_from_registry:
            ranked = sorted(self._registry.values(), key=lambda s: s.fitness, reverse=True)
            seeds = ranked[:10]

        top_strategies = ga.evolve(fitness_fn, seed_strategies=seeds)

        # Add top strategies to registry
        for s in top_strategies:
            s.status = "candidate"
            self._registry[s.id] = s

        logger.info(
            "EvolutionController: evolution complete | top fitness=%.3f",
            top_strategies[0].fitness if top_strategies else 0.0,
        )
        return top_strategies

    def promote_to_shadow(self, strategy_id: str) -> bool:
        """Promote candidate strategy to shadow mode for live validation."""
        if strategy_id not in self._registry:
            logger.warning("EvolutionController: strategy %s not found", strategy_id[:8])
            return False
        s = self._registry[strategy_id]
        if s.fitness < MIN_FITNESS_TO_DEPLOY:
            logger.info("EvolutionController: strategy %s fitness %.3f below threshold", strategy_id[:8], s.fitness)
            return False
        s.status = "shadow"
        logger.info("EvolutionController: promoted %s to shadow (fitness=%.3f)", strategy_id[:8], s.fitness)
        return True

    def promote_to_active(self, strategy_id: str) -> bool:
        """Promote shadow strategy to active trading."""
        if strategy_id not in self._registry:
            return False
        s = self._registry[strategy_id]
        if s.status != "shadow":
            logger.warning("EvolutionController: strategy %s is not in shadow mode", strategy_id[:8])
            return False
        s.status = "active"
        logger.info("EvolutionController: promoted %s to ACTIVE", strategy_id[:8])
        return True

    def retire_underperforming(self) -> int:
        """Retire strategies below fitness thresholds. Returns count retired."""
        retired = 0
        for s in list(self._registry.values()):
            if s.status not in ("shadow", "active"):
                continue
            if s.backtest_stats is None:
                continue
            if (s.backtest_stats.sharpe < RETIRE_SHARPE_THRESHOLD or
                    s.backtest_stats.profit_factor < RETIRE_PF_THRESHOLD):
                s.status = "retired"
                retired += 1
                logger.info("EvolutionController: retired %s (sharpe=%.2f pf=%.2f)",
                            s.id[:8], s.backtest_stats.sharpe, s.backtest_stats.profit_factor)
        return retired

    def active_strategies(self) -> List[StrategyTemplate]:
        return [s for s in self._registry.values() if s.status == "active"]

    def shadow_strategies(self) -> List[StrategyTemplate]:
        return [s for s in self._registry.values() if s.status == "shadow"]

    def get_status_summary(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for s in self._registry.values():
            by_status[s.status] = by_status.get(s.status, 0) + 1
        best = max(self._registry.values(), key=lambda s: s.fitness, default=None)
        return {
            "total_strategies": len(self._registry),
            "by_status": by_status,
            "best_fitness": round(best.fitness, 4) if best else 0.0,
            "best_id": best.id[:8] if best else None,
        }
