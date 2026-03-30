"""Genetic algorithm engine for strategy evolution.

Components:
- CrossoverEngine: breeds new strategies from parents (1-point block swap)
- MutationEngine: introduces random variation (block type or param mutation)
- PopulationManager: maintains population sorted by fitness
- GeneticAlgorithm: orchestrates the full evolution loop

Design:
- Population size: 30-50 strategies
- Elite preservation: top 20% always carry to next generation
- Crossover: swap entry/exit/sizing blocks between parents
- Mutation: 30% of population gets random param perturbation
- Grammar: all children validated before added to population
- Fitness: profit_factor * 0.6 + sharpe * 0.4
"""
from __future__ import annotations

import copy
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .blocks import (
    BlockType, StrategyBlock, BLOCK_PARAM_SCHEMAS,
    BlockCategory, BLOCK_CATEGORIES,
)
from .template import StrategyTemplate
from .grammar import StrategyGrammar, VALID_ENTRY_EXIT_PAIRS, ALL_ENTRY_TYPES, ALL_EXIT_TYPES, ALL_SIZING_TYPES

logger = logging.getLogger(__name__)


class CrossoverEngine:
    """Breeds new strategy templates from two parents."""

    @staticmethod
    def single_point(
        parent1: StrategyTemplate,
        parent2: StrategyTemplate,
        rng: Optional[np.random.Generator] = None,
    ) -> List[StrategyTemplate]:
        """Swap one block (entry/exit/sizing) between parents.

        Returns 2 children. Validates before returning; falls back to parents if invalid.
        """
        rng = rng or np.random.default_rng()
        block_attr = rng.choice(["entry", "exit_block", "sizing"])

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.id = uuid.uuid4().hex
        child2.id = uuid.uuid4().hex
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        child1.generation = max(parent1.generation, parent2.generation) + 1
        child2.generation = child1.generation
        child1.fitness = 0.0
        child2.fitness = 0.0
        child1.backtest_stats = None
        child2.backtest_stats = None
        child1.status = "candidate"
        child2.status = "candidate"

        # Swap the chosen block
        setattr(child1, block_attr, copy.deepcopy(getattr(parent2, block_attr)))
        setattr(child2, block_attr, copy.deepcopy(getattr(parent1, block_attr)))

        results = []
        for child in [child1, child2]:
            ok, reason = StrategyGrammar.is_valid(child)
            if ok:
                results.append(child)
            else:
                logger.debug("CrossoverEngine: invalid child discarded (%s)", reason)

        return results if results else [copy.deepcopy(parent1)]

    @staticmethod
    def perturb_params(
        block: StrategyBlock,
        strength: float = 0.15,
        rng: Optional[np.random.Generator] = None,
    ) -> StrategyBlock:
        """Perturb block params by Gaussian noise within schema bounds."""
        rng = rng or np.random.default_rng()
        schema = BLOCK_PARAM_SCHEMAS.get(block.block_type, {})
        new_params = {}
        for param_name, (lo, hi, _) in schema.items():
            val = block.params.get(param_name, (lo + hi) / 2)
            noise = rng.normal(0, strength * (hi - lo))
            new_params[param_name] = float(np.clip(val + noise, lo, hi))
        return StrategyBlock(block_type=block.block_type, params=new_params)


class MutationEngine:
    """Mutates strategy templates."""

    @staticmethod
    def mutate_block_type(
        block: StrategyBlock,
        entry_type: Optional[str] = None,  # needed for exit mutation
        rng: Optional[np.random.Generator] = None,
    ) -> StrategyBlock:
        """Change block_type to a compatible alternative."""
        rng = rng or np.random.default_rng()

        cat = BLOCK_CATEGORIES.get(block.block_type)
        if cat == BlockCategory.ENTRY:
            candidates = [b for b in ALL_ENTRY_TYPES if str(b) != str(block.block_type)]
        elif cat == BlockCategory.EXIT:
            if entry_type:
                candidates = [b for b in VALID_ENTRY_EXIT_PAIRS.get(entry_type, ALL_EXIT_TYPES)
                              if str(b) != str(block.block_type)]
            else:
                candidates = [b for b in ALL_EXIT_TYPES if str(b) != str(block.block_type)]
        elif cat == BlockCategory.SIZING:
            candidates = [b for b in ALL_SIZING_TYPES if str(b) != str(block.block_type)]
        else:
            return block

        if not candidates:
            return block

        new_type = rng.choice(candidates)
        return StrategyBlock.default_for(new_type)

    @staticmethod
    def apply(
        template: StrategyTemplate,
        mutation_rate: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> StrategyTemplate:
        """Randomly mutate params or block type. Validates result."""
        rng = rng or np.random.default_rng()
        mutated = copy.deepcopy(template)
        mutated.id = uuid.uuid4().hex
        mutated.parent_ids = [template.id]
        mutated.generation = template.generation + 1
        mutated.fitness = 0.0
        mutated.backtest_stats = None
        mutated.status = "candidate"

        # Randomly choose mutation type
        roll = float(rng.random())
        if roll < mutation_rate * 0.5:
            # Mutate block type
            target = rng.choice(["entry", "exit_block", "sizing"])
            block = getattr(mutated, target)
            if block is not None:
                entry_type = mutated.entry.block_type if mutated.entry else None
                new_block = MutationEngine.mutate_block_type(
                    block, entry_type=entry_type if target == "exit_block" else None, rng=rng
                )
                setattr(mutated, target, new_block)
        else:
            # Mutate params of a random block
            target = rng.choice(["entry", "exit_block", "sizing"])
            block = getattr(mutated, target)
            if block is not None:
                setattr(mutated, target, CrossoverEngine.perturb_params(block, strength=0.2, rng=rng))

        # Validate; revert if invalid
        ok, reason = StrategyGrammar.is_valid(mutated)
        if not ok:
            logger.debug("MutationEngine: invalid mutation reverted (%s)", reason)
            return template  # Return original unchanged
        return mutated


class PopulationManager:
    """Maintains a ranked population of strategy templates."""

    def __init__(self, max_size: int = 40):
        self.max_size = max_size
        self.population: List[StrategyTemplate] = []
        self.generation: int = 0

    def add(self, strategy: StrategyTemplate) -> None:
        """Add strategy to population. Trims to max_size by dropping lowest fitness."""
        self.population.append(strategy)
        if len(self.population) > self.max_size:
            self.population = self.ranked()[:self.max_size]

    def ranked(self) -> List[StrategyTemplate]:
        """Return population sorted by fitness (highest first)."""
        return sorted(self.population, key=lambda s: s.fitness, reverse=True)

    def best(self) -> Optional[StrategyTemplate]:
        """Return highest-fitness strategy."""
        if not self.population:
            return None
        return max(self.population, key=lambda s: s.fitness)

    def select_parents(self, ratio: float = 0.5) -> List[StrategyTemplate]:
        """Select top `ratio` of population for breeding."""
        ranked = self.ranked()
        n = max(2, int(len(ranked) * ratio))
        return ranked[:n]

    def elite(self) -> List[StrategyTemplate]:
        """Top 20% — always preserved in next generation."""
        ranked = self.ranked()
        n = max(1, len(ranked) // 5)
        return ranked[:n]

    def diversity_score(self) -> float:
        """0-1 measure of strategy diversity (ratio of unique entry types)."""
        if not self.population:
            return 0.0
        entry_types = set(s.entry.block_type for s in self.population if s.entry)
        return len(entry_types) / 4.0  # 4 possible entry types


class GeneticAlgorithm:
    """Main GA loop: evolve a population of strategy templates."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 5,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.population_manager = PopulationManager(max_size=population_size)
        self.crossover = CrossoverEngine()
        self.mutation = MutationEngine()
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self._rng = np.random.default_rng(seed)

    def seed_population(self, seed_strategies: Optional[List[StrategyTemplate]] = None) -> None:
        """Initialize population with seeds + random templates."""
        if seed_strategies:
            for s in seed_strategies:
                self.population_manager.add(s)

        # Fill remaining with random valid templates
        n_random = self.population_manager.max_size - len(self.population_manager.population)
        for _ in range(max(0, n_random)):
            t = StrategyGrammar.random_valid_template(self._rng)
            self.population_manager.add(t)

    def evolve_one_generation(
        self,
        fitness_fn: Callable[[StrategyTemplate], float],
    ) -> List[StrategyTemplate]:
        """Run one generation. Returns updated population ranked by fitness.

        fitness_fn: takes StrategyTemplate, returns float fitness score.
        """
        # Step 1: Evaluate fitness for all un-evaluated strategies
        unevaluated = [s for s in self.population_manager.population if s.fitness == 0.0]
        if unevaluated:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(fitness_fn, s): s for s in unevaluated}
                for future in as_completed(futures):
                    s = futures[future]
                    try:
                        s.fitness = float(future.result())
                    except Exception as e:
                        s.fitness = 0.0
                        logger.warning("GA: fitness evaluation failed for %s: %s", s.id[:8], e)

        # Step 2: Preserve elites
        elites = self.population_manager.elite()
        new_population = list(elites)

        # Step 3: Select parents
        parents = self.population_manager.select_parents(ratio=0.5)

        # Step 4: Crossover
        n_crossover = int(self.population_manager.max_size * self.crossover_rate)
        for _ in range(n_crossover // 2):
            if len(parents) < 2:
                break
            idx = self._rng.choice(len(parents), size=2, replace=False)
            children = CrossoverEngine.single_point(parents[idx[0]], parents[idx[1]], rng=self._rng)
            new_population.extend(children)

        # Step 5: Mutation
        n_mutate = int(self.population_manager.max_size * self.mutation_rate)
        candidates = self.population_manager.ranked()
        # Don't mutate elites
        non_elites = [s for s in candidates if s.id not in {e.id for e in elites}]
        for i in range(min(n_mutate, len(non_elites))):
            mutated = MutationEngine.apply(non_elites[i], mutation_rate=self.mutation_rate, rng=self._rng)
            new_population.append(mutated)

        # Step 6: Replace population
        self.population_manager.population = []
        for s in new_population:
            self.population_manager.add(s)

        self.population_manager.generation += 1

        ranked = self.population_manager.ranked()
        best_fitness = ranked[0].fitness if ranked else 0.0
        logger.info(
            "GA: generation %d complete | pop=%d | best_fitness=%.3f | diversity=%.2f",
            self.population_manager.generation,
            len(ranked),
            best_fitness,
            self.population_manager.diversity_score(),
        )
        return ranked

    def evolve(
        self,
        fitness_fn: Callable[[StrategyTemplate], float],
        seed_strategies: Optional[List[StrategyTemplate]] = None,
    ) -> List[StrategyTemplate]:
        """Full evolution run for N generations. Returns top 5 strategies."""
        self.seed_population(seed_strategies)
        for gen in range(self.generations):
            ranked = self.evolve_one_generation(fitness_fn)
        return self.population_manager.ranked()[:5]
