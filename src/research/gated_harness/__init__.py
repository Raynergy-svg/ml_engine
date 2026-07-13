"""Canonical, deterministic research evaluation controls.

Lane modules own strategy construction.  This package owns the controls that
decide whether the resulting return stream is admissible evidence.
"""

from .preregistration import ResearchSpecification
from .report import EvaluationReport, ResearchEvaluationReport, evaluate_research
from .verifier import IndependentReplayVerifier, ReplayVerdict

__all__ = [
    # ``ResearchEvaluationReport`` is the canonical name; ``EvaluationReport``
    # is a backward-compat alias. The formal per-head evidence contract with
    # the same short name lives in ``src.evidence.contracts``.
    "EvaluationReport",
    "ResearchEvaluationReport",
    "IndependentReplayVerifier",
    "ReplayVerdict",
    "ResearchSpecification",
    "evaluate_research",
]
