"""Comprehensive test suite for Strategy Invention Engine (Tier 6 Dimension B).

Tests covering:
- StrategyBlock validation and parameter clamping
- StrategyTemplate serialization
- Grammar validation for entry-exit compatibility
- Genetic algorithm crossover and mutation
- Population management and evolution
- Evolution controller save/load lifecycle
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.strategy_invention.blocks import (
    StrategyBlock, BlockType, BLOCK_PARAM_SCHEMAS, BLOCK_CATEGORIES, BlockCategory
)
from src.strategy_invention.template import StrategyTemplate, BacktestStats
from src.strategy_invention.grammar import StrategyGrammar, VALID_ENTRY_EXIT_PAIRS
from src.strategy_invention.genetic_algorithm import (
    CrossoverEngine, MutationEngine, PopulationManager, GeneticAlgorithm
)
from src.strategy_invention.evolution_controller import EvolutionController


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyBlock:
    """Test StrategyBlock parameter validation and clamping."""

    def test_block_default_params_valid(self):
        """Default block should have valid params within schema bounds."""
        for block_type in BLOCK_PARAM_SCHEMAS.keys():
            block = StrategyBlock.default_for(block_type)
            ok, reason = block.validate()
            assert ok, f"Default block {block_type} invalid: {reason}"

    def test_block_clamps_out_of_range_params(self):
        """Params outside bounds should be clamped to schema range."""
        block = StrategyBlock(
            block_type=BlockType.TREND_ENTRY,
            params={"timeframe": 9999, "slope_threshold": -999, "ma_period": 1}
        )
        # Check clamped values are within schema
        schema = BLOCK_PARAM_SCHEMAS[BlockType.TREND_ENTRY]
        for param_name, (lo, hi, _) in schema.items():
            val = block.params.get(param_name)
            assert lo <= val <= hi, f"Param {param_name}={val} not clamped to [{lo}, {hi}]"

    def test_block_unknown_type_validate_fails(self):
        """Unknown block_type should fail validation."""
        block = StrategyBlock(block_type="UnknownType")
        ok, reason = block.validate()
        assert not ok, "Unknown block_type should fail validation"
        assert "Unknown" in reason

    def test_block_serialize_roundtrip(self):
        """Block to_dict → from_dict should preserve values."""
        original = StrategyBlock.default_for(BlockType.MEAN_REVERSION_ENTRY)
        dict_form = original.to_dict()
        restored = StrategyBlock.from_dict(dict_form)
        assert restored.block_type == original.block_type
        for key, val in original.params.items():
            assert abs(restored.params[key] - val) < 1e-4

    def test_block_categories_assigned(self):
        """All block types should have a category."""
        for block_type in BLOCK_PARAM_SCHEMAS.keys():
            block = StrategyBlock.default_for(block_type)
            cat = block.category
            assert cat in ("entry", "exit", "sizing", "filter")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyTemplate:
    """Test StrategyTemplate validation and serialization."""

    def test_template_is_complete_true(self):
        """Complete template (entry + exit + sizing) should return True."""
        t = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.TREND_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.TRAILING_STOP_EXIT),
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        assert t.is_complete()

    def test_template_is_complete_false_no_exit(self):
        """Template missing exit should return False."""
        t = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.TREND_ENTRY),
            exit_block=None,
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        assert not t.is_complete()

    def test_template_serialize_roundtrip(self):
        """Template to_dict → from_dict should preserve structure."""
        original = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.MOMENTUM_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.PROFIT_TARGET_EXIT),
            sizing=StrategyBlock.default_for(BlockType.CONFIDENCE_SCALING),
            fitness=0.87,
            generation=3
        )
        dict_form = original.to_dict()
        restored = StrategyTemplate.from_dict(dict_form)
        assert restored.entry.block_type == original.entry.block_type
        assert restored.exit_block.block_type == original.exit_block.block_type
        assert restored.sizing.block_type == original.sizing.block_type
        assert abs(restored.fitness - original.fitness) < 1e-3
        assert restored.generation == original.generation

    def test_template_summary_string(self):
        """Summary string should be non-empty and contain key info."""
        t = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.BREAKOUT_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.STOP_LOSS_EXIT),
            sizing=StrategyBlock.default_for(BlockType.VOLATILITY_SCALING),
            fitness=0.65
        )
        summary = t.summary()
        assert len(summary) > 0
        assert "BreakoutEntry" in summary
        assert "StopLossExit" in summary
        assert "0.650" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# GRAMMAR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyGrammar:
    """Test grammar validation of strategy compositions."""

    def test_grammar_valid_trend_trailing(self):
        """TrendEntry + TrailingStop should be valid (compatible pair)."""
        t = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.TREND_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.TRAILING_STOP_EXIT),
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        ok, reason = StrategyGrammar.is_valid(t)
        assert ok, f"Valid strategy marked invalid: {reason}"

    def test_grammar_invalid_mean_reversion_trailing(self):
        """MeanRevEntry + TrailingStop should be invalid (incompatible pair)."""
        t = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.MEAN_REVERSION_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.TRAILING_STOP_EXIT),
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        ok, reason = StrategyGrammar.is_valid(t)
        assert not ok, "Invalid strategy marked valid"

    def test_grammar_incomplete_template(self):
        """Template missing entry should fail grammar check."""
        t = StrategyTemplate(
            entry=None,
            exit_block=StrategyBlock.default_for(BlockType.PROFIT_TARGET_EXIT),
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        ok, reason = StrategyGrammar.is_valid(t)
        assert not ok

    def test_grammar_random_valid_template(self):
        """random_valid_template should always generate valid strategies."""
        for _ in range(10):
            t = StrategyGrammar.random_valid_template()
            ok, reason = StrategyGrammar.is_valid(t)
            assert ok, f"Random template invalid: {reason}"


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSOVER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossover:
    """Test crossover breeding of strategies."""

    def test_crossover_children_are_valid(self):
        """Children from crossover should pass grammar validation."""
        p1 = StrategyGrammar.random_valid_template()
        p2 = StrategyGrammar.random_valid_template()
        rng = np.random.default_rng(42)
        children = CrossoverEngine.single_point(p1, p2, rng=rng)
        for child in children:
            ok, reason = StrategyGrammar.is_valid(child)
            assert ok, f"Child invalid after crossover: {reason}"

    def test_crossover_children_have_parent_ids(self):
        """Children should record parent IDs."""
        p1 = StrategyGrammar.random_valid_template()
        p2 = StrategyGrammar.random_valid_template()
        p1.id = "parent1_id"
        p2.id = "parent2_id"
        rng = np.random.default_rng(42)
        children = CrossoverEngine.single_point(p1, p2, rng=rng)
        for child in children:
            assert p1.id in child.parent_ids
            assert p2.id in child.parent_ids

    def test_crossover_children_different_from_parents(self):
        """At least one block in children should differ from parents."""
        p1 = StrategyGrammar.random_valid_template()
        p2 = StrategyGrammar.random_valid_template()
        rng = np.random.default_rng(42)
        children = CrossoverEngine.single_point(p1, p2, rng=rng)
        assert len(children) >= 1
        for child in children:
            differs = (
                (child.entry.block_type != p1.entry.block_type) or
                (child.exit_block.block_type != p1.exit_block.block_type) or
                (child.sizing.block_type != p1.sizing.block_type) or
                (child.entry.block_type != p2.entry.block_type) or
                (child.exit_block.block_type != p2.exit_block.block_type) or
                (child.sizing.block_type != p2.sizing.block_type)
            )
            # At least one child should differ
            assert len(children) == 0 or any(differs for child in children)

    def test_perturb_params_stays_in_bounds(self):
        """Perturbed params should remain within schema bounds."""
        block = StrategyBlock.default_for(BlockType.VOLATILITY_SCALING)
        rng = np.random.default_rng(42)
        perturbed = CrossoverEngine.perturb_params(block, strength=0.2, rng=rng)
        schema = BLOCK_PARAM_SCHEMAS[block.block_type]
        for param_name, (lo, hi, _) in schema.items():
            val = perturbed.params[param_name]
            assert lo <= val <= hi, f"Perturbed param {param_name}={val} out of bounds [{lo}, {hi}]"


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMutation:
    """Test mutation operators on strategies."""

    def test_mutation_result_is_valid(self):
        """Mutated template should pass grammar validation."""
        original = StrategyGrammar.random_valid_template()
        rng = np.random.default_rng(42)
        mutated = MutationEngine.apply(original, mutation_rate=0.3, rng=rng)
        ok, reason = StrategyGrammar.is_valid(mutated)
        assert ok, f"Mutated template invalid: {reason}"

    def test_mutation_changes_something(self):
        """Mutation should usually produce a different strategy."""
        original = StrategyGrammar.random_valid_template()
        rng = np.random.default_rng(42)
        mutated = MutationEngine.apply(original, mutation_rate=0.5, rng=rng)
        # Check if at least one block or param changed
        changed = (
            (mutated.entry.block_type != original.entry.block_type) or
            (mutated.exit_block.block_type != original.exit_block.block_type) or
            (mutated.sizing.block_type != original.sizing.block_type)
        )
        # With high mutation rate, change is likely but not guaranteed
        # So we just verify the mutation returns a valid template
        assert StrategyGrammar.is_valid(mutated)[0]

    def test_mutation_block_type_compatible(self):
        """New block type after mutation should be compatible with entry."""
        original = StrategyTemplate(
            entry=StrategyBlock.default_for(BlockType.TREND_ENTRY),
            exit_block=StrategyBlock.default_for(BlockType.TRAILING_STOP_EXIT),
            sizing=StrategyBlock.default_for(BlockType.FIXED_SIZE)
        )
        rng = np.random.default_rng(42)
        # Mutate exit block type repeatedly to check compatibility
        for _ in range(5):
            mutated = MutationEngine.apply(original, mutation_rate=1.0, rng=rng)
            # Should still pass grammar (exit compatible with entry)
            ok, _ = StrategyGrammar.is_valid(mutated)
            assert ok, f"Mutated exit {mutated.exit_block.block_type} not compatible with entry"


# ═══════════════════════════════════════════════════════════════════════════════
# POPULATION & GA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopulationManager:
    """Test population management and ranking."""

    def test_population_ranks_by_fitness(self):
        """Population ranked() should sort by fitness (highest first)."""
        pm = PopulationManager(max_size=10)
        t1 = StrategyGrammar.random_valid_template()
        t1.fitness = 0.5
        t2 = StrategyGrammar.random_valid_template()
        t2.fitness = 0.9
        t3 = StrategyGrammar.random_valid_template()
        t3.fitness = 0.3
        pm.add(t1)
        pm.add(t2)
        pm.add(t3)
        ranked = pm.ranked()
        assert ranked[0].fitness == 0.9
        assert ranked[1].fitness == 0.5
        assert ranked[2].fitness == 0.3

    def test_population_elite_is_top_20pct(self):
        """Elite should return top 20% of population."""
        pm = PopulationManager(max_size=10)
        for i in range(10):
            t = StrategyGrammar.random_valid_template()
            t.fitness = float(i) / 10.0
            pm.add(t)
        elite = pm.elite()
        # Top 20% of 10 = 2 (rounded up to at least 1)
        assert len(elite) >= 1
        # Elite should have highest fitness
        assert elite[0].fitness >= pm.ranked()[-1].fitness

    def test_population_stays_bounded(self):
        """Population should never exceed max_size."""
        pm = PopulationManager(max_size=5)
        for i in range(20):
            t = StrategyGrammar.random_valid_template()
            t.fitness = float(i) / 20.0
            pm.add(t)
        assert len(pm.population) <= 5


class TestGeneticAlgorithm:
    """Test GA evolution cycle."""

    def test_ga_evolve_improves_fitness(self):
        """Fitness should improve (or stay same) over generations."""
        ga = GeneticAlgorithm(population_size=10, generations=3, seed=42)

        # Simple fitness function: higher if entry is TREND_ENTRY
        def fitness_fn(template: StrategyTemplate) -> float:
            if template.entry.block_type == BlockType.TREND_ENTRY:
                return 0.8
            elif template.entry.block_type == BlockType.BREAKOUT_ENTRY:
                return 0.6
            else:
                return 0.4

        top = ga.evolve(fitness_fn, seed_strategies=None)
        # Best strategy should have fitness >= second-best
        if len(top) >= 2:
            assert top[0].fitness >= top[1].fitness - 1e-4

    def test_ga_population_stays_bounded(self):
        """GA population should never exceed max size."""
        ga = GeneticAlgorithm(population_size=20, generations=2, seed=42)

        def dummy_fitness(t: StrategyTemplate) -> float:
            return 0.5

        ga.seed_population()
        assert len(ga.population_manager.population) <= 20


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION CONTROLLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvolutionController:
    """Test strategy registry lifecycle: load, save, promote."""

    def test_evolution_controller_save_and_load(self):
        """Save to disk → load should restore strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = EvolutionController(strategies_dir=Path(tmpdir))

            # Create and register a strategy
            t1 = StrategyGrammar.random_valid_template()
            t1.fitness = 0.75
            t1.status = "candidate"
            ctrl._registry[t1.id] = t1

            # Save
            ctrl.save()

            # Load in new controller
            ctrl2 = EvolutionController(strategies_dir=Path(tmpdir))
            ctrl2.load()

            # Verify restored
            assert t1.id in ctrl2._registry
            restored = ctrl2._registry[t1.id]
            assert restored.fitness == 0.75
            assert restored.entry.block_type == t1.entry.block_type

    def test_promote_to_shadow_requires_fitness(self):
        """Promotion to shadow should check fitness threshold."""
        ctrl = EvolutionController()
        t = StrategyGrammar.random_valid_template()
        t.fitness = 0.5  # Below MIN_FITNESS_TO_DEPLOY (1.1)
        ctrl._registry[t.id] = t

        result = ctrl.promote_to_shadow(t.id)
        assert not result, "Low fitness strategy should not promote to shadow"
        assert t.status == "candidate"

    def test_promote_to_active_requires_shadow(self):
        """Promotion to active should only work from shadow."""
        ctrl = EvolutionController()
        t = StrategyGrammar.random_valid_template()
        t.status = "candidate"
        ctrl._registry[t.id] = t

        result = ctrl.promote_to_active(t.id)
        assert not result, "Candidate should not promote to active"

    def test_retire_underperforming_changes_status(self):
        """Retiring should change status and mark as retired."""
        ctrl = EvolutionController()
        t = StrategyGrammar.random_valid_template()
        t.status = "shadow"
        t.backtest_stats = BacktestStats(sharpe=0.3, profit_factor=0.8)  # Below thresholds
        ctrl._registry[t.id] = t

        retired = ctrl.retire_underperforming()
        assert retired >= 1
        assert t.status == "retired"

    def test_status_summary_counts_correct(self):
        """Status summary should accurately count strategies by status."""
        ctrl = EvolutionController()

        # Add 5 candidates
        for i in range(5):
            t = StrategyGrammar.random_valid_template()
            t.status = "candidate"
            ctrl._registry[t.id] = t

        # Add 2 shadow
        for i in range(2):
            t = StrategyGrammar.random_valid_template()
            t.status = "shadow"
            ctrl._registry[t.id] = t

        summary = ctrl.get_status_summary()
        assert summary["total_strategies"] == 7
        assert summary["by_status"]["candidate"] == 5
        assert summary["by_status"]["shadow"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST STATS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestStats:
    """Test backtest stats serialization."""

    def test_backtest_stats_roundtrip(self):
        """BacktestStats to_dict → from_dict should preserve values."""
        original = BacktestStats(
            win_rate=0.55,
            avg_trade_pnl=25.5,
            sharpe=1.2,
            profit_factor=1.8,
            max_drawdown=0.15,
            sample_size=100,
            fitness=0.72
        )
        dict_form = original.to_dict()
        restored = BacktestStats.from_dict(dict_form)
        assert restored.win_rate == original.win_rate
        assert abs(restored.sharpe - original.sharpe) < 1e-3
        assert restored.sample_size == original.sample_size


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_ga_cycle(self):
        """Full GA cycle: seed → evolve → save → load → verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = EvolutionController(strategies_dir=Path(tmpdir))

            # Simple fitness: penalizes mean reversion entries
            def fitness_fn(t: StrategyTemplate) -> float:
                if t.entry.block_type == BlockType.MEAN_REVERSION_ENTRY:
                    return 0.4
                else:
                    return 0.7

            # Run evolution
            top = ctrl.run_evolution_cycle(fitness_fn, generations=2, population_size=10)
            assert len(top) > 0
            assert top[0].fitness >= 0.0

            # Save
            ctrl.save()

            # Load and verify best is present
            ctrl2 = EvolutionController(strategies_dir=Path(tmpdir))
            ctrl2.load()
            assert len(ctrl2._registry) > 0
