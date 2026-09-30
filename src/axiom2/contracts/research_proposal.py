"""Versioned, provider-neutral inputs to Axiom's research admission boundary.

This module validates structure, not truth: referenced datasets, universes,
models, costs and promotion rules still require authoritative resolution.
No proposal grants registry, holdout, promotion or broker privileges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


RESEARCH_PROPOSAL_SCHEMA_VERSION = "1.0.0"
PHASE1_FEATURE_FAMILIES = frozenset(
    {
        "price_return",
        "volume_liquidity",
        "cross_sectional_relative_strength",
        "market_sector_regime",
    }
)
_REFERENCE_FIELDS = (
    "experiment_id",
    "universe_id",
    "label_id",
    "benchmark_id",
    "cost_model_id",
    "model_id",
    "promotion_rule_id",
    "code_commit",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DataDependency:
    """Expected immutable dataset reference, not a provider client or data proof.

    ``dataset_id`` identifies the expected snapshot. ``source`` names its
    provenance source, and ``schema_version`` names that dataset's schema.
    Existence and point-in-time eligibility are checked by later authorities.
    """

    dataset_id: str
    source: str
    schema_version: str

    def __post_init__(self) -> None:
        _validate_data_dependency(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchProposal:
    """Immutable experiment inputs independent of the producing agent.

    ``holding_horizon`` counts observation bars as defined by ``label_id``;
    it is not implicitly calendar days. Collections must be nonempty tuples
    of plain values, so mutable containers and SDK objects cannot enter.

    Frozen records prevent ordinary mutation, not hostile Python reflection.
    Consumers must call ``validate_research_proposal`` at admission boundaries.
    """

    experiment_id: str
    hypothesis: str
    universe_id: str
    feature_families: tuple[str, ...]
    label_id: str
    holding_horizon: int
    benchmark_id: str
    cost_model_id: str
    model_id: str
    baseline_ids: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    promotion_rule_id: str
    code_commit: str
    data_dependencies: tuple[DataDependency, ...]
    schema_version: str = RESEARCH_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_research_proposal(self)


def _require_text(value: object, field: str, *, reference: bool = True) -> None:
    # Exact types avoid invoking provider-defined coercion/equality hooks.
    if type(value) is not str:
        raise TypeError(f"{field} must be a plain string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    if reference and value != value.strip():
        raise ValueError(f"{field} must not contain surrounding whitespace")


def _validate_string_tuple(value: object, field: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be an immutable tuple")
    if not value:
        raise ValueError(f"{field} must not be empty")
    for item in value:
        _require_text(item, field)
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")


def _validate_data_dependency(dependency: DataDependency) -> None:
    if type(dependency) is not DataDependency:
        raise TypeError("data_dependencies entries must be DataDependency records")
    for field in ("dataset_id", "source", "schema_version"):
        _require_text(getattr(dependency, field), f"data_dependencies.{field}")


def validate_research_proposal(proposal: ResearchProposal) -> None:
    """Reject unsupported or malformed proposals; return None on success.

    Validation neither coerces values nor imports provider, broker, evidence
    or training modules. Errors name fields without rendering rejected values.
    A valid reference is not proof that the referenced artifact is eligible.
    """
    if type(proposal) is not ResearchProposal:
        raise TypeError("proposal must be a ResearchProposal record")

    _require_text(proposal.schema_version, "schema_version")
    if proposal.schema_version != RESEARCH_PROPOSAL_SCHEMA_VERSION:
        raise ValueError("schema_version is unsupported")

    for field in _REFERENCE_FIELDS:
        _require_text(getattr(proposal, field), field)
    _require_text(proposal.hypothesis, "hypothesis", reference=False)

    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", proposal.code_commit) is None:
        raise ValueError("code_commit must be a full lowercase Git object ID")
    if type(proposal.holding_horizon) is not int:
        raise TypeError("holding_horizon must be an integer bar count")
    if proposal.holding_horizon <= 0:
        raise ValueError("holding_horizon must be positive")

    for field in ("feature_families", "baseline_ids", "primary_metrics"):
        _validate_string_tuple(getattr(proposal, field), field)
    if any(family not in PHASE1_FEATURE_FAMILIES for family in proposal.feature_families):
        raise ValueError("feature_families contains a quarantined or unknown Phase-1 family")

    if type(proposal.data_dependencies) is not tuple:
        raise TypeError("data_dependencies must be an immutable tuple")
    if not proposal.data_dependencies:
        raise ValueError("data_dependencies must not be empty")
    dataset_ids: set[str] = set()
    for dependency in proposal.data_dependencies:
        _validate_data_dependency(dependency)
        if dependency.dataset_id in dataset_ids:
            raise ValueError("data_dependencies must not repeat a dataset_id")
        dataset_ids.add(dependency.dataset_id)
