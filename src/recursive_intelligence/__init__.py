"""Recursive Intelligence Framework.

Domain-agnostic self-improvement engine extracted from Buddy's architecture.
Both Buddy (market intelligence) and Aura (human intelligence) instantiate
this framework with different data types.

Core loop:
    observe(data_stream) → model(signals) → detect_patterns(model)
    → validate(patterns) → promote(confirmed) → tune(rules, config)
    → bound_check(adjustments) → rollback_if_degraded()
"""

from src.recursive_intelligence.learner import RecursiveLearner, LearningEntry
from src.recursive_intelligence.tuner import AdaptiveTuner
from src.recursive_intelligence.persistence import SessionPersistence
from src.recursive_intelligence.weight_learner import WeightLearner
from src.recursive_intelligence.meta_parameters import MetaParametersRegistry, HyperparameterSet
from src.recursive_intelligence.bayesian_adapter import OnlineBayesianAdapter
from src.recursive_intelligence.meta_learner import MetaLearner
from src.recursive_intelligence.ensemble_weighting import EnsembleWeighter, WeighterConfig
from src.recursive_intelligence.maml_benchmark import MAMLBenchmark, BenchmarkConfig

__all__ = [
    "RecursiveLearner",
    "LearningEntry",
    "AdaptiveTuner",
    "SessionPersistence",
    "WeightLearner",
    "MetaParametersRegistry",
    "HyperparameterSet",
    "OnlineBayesianAdapter",
    "MetaLearner",
    "EnsembleWeighter",
    "WeighterConfig",
    "MAMLBenchmark",
    "BenchmarkConfig",
]
