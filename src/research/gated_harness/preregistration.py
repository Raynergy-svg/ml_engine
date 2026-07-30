"""Frozen research specifications and hypothesis identities."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field, model_validator

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts.base import StrictContract
from src.research.trial_budget import resolve_trial_budget


class ResearchSpecification(StrictContract):
    """The fields that must be frozen before confirmatory evaluation begins."""

    experiment_id: str = Field(min_length=1)
    lane_id: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    mode: Literal["confirmatory", "exploratory"]
    target: str = Field(min_length=1)
    features: tuple[str, ...] = Field(min_length=1)
    causal_lag: int = Field(ge=0)
    universe: str = Field(min_length=1)
    evaluation_windows: tuple[str, ...] = Field(min_length=1)
    search_space: dict[str, Any]
    trial_budget: int = Field(ge=1)
    correction: Literal["bonferroni", "holm", "none"]
    metrics: tuple[str, ...] = Field(min_length=1)
    promotion_criteria: dict[str, Any]
    cost_scenarios_bps: tuple[float, ...] = Field(min_length=1)
    minimum_history_observations: int = Field(ge=2)
    minimum_effective_n: float = Field(gt=0)
    maximum_absolute_placebo_sharpe: float = Field(ge=0)
    point_in_time_policy: str = Field(min_length=1)
    survivorship_policy: str = Field(min_length=1)
    use_cpcv: bool = False
    #: ``"authoritative"`` (default) binds ``trial_budget`` to the campaign
    #: register in :mod:`src.research.trial_budget`. ``"frozen_replay"`` is the
    #: ONLY way to declare a historical budget, and only one an archived run
    #: actually used — it exists to re-derive a recorded artifact exactly as
    #: recorded, never to issue a new verdict at a looser alpha.
    budget_scope: Literal["authoritative", "frozen_replay"] = "authoritative"

    @model_validator(mode="after")
    def _confirmatory_requires_correction(self) -> "ResearchSpecification":
        if self.mode == "confirmatory" and self.trial_budget > 1 and self.correction == "none":
            raise ValueError("confirmatory multi-trial research requires a declared correction")
        if tuple(sorted(set(self.cost_scenarios_bps))) != self.cost_scenarios_bps:
            raise ValueError("cost_scenarios_bps must be unique and ascending")
        if self.mode == "confirmatory":
            # A confirmatory verdict binds to the CAMPAIGN's family-wise budget.
            # Per-lane literals are how N_TRIALS=3 (alpha 0.01667) and N_TRIALS=24
            # (alpha 0.00208) came to coexist — an 8x difference in alpha decided
            # by which file you called. Exploratory specs may declare anything;
            # they are not verdicts.
            resolve_trial_budget(
                self.trial_budget,
                context=f"ResearchSpecification({self.experiment_id!r}, lane={self.lane_id!r})",
                frozen_replay=self.budget_scope == "frozen_replay",
            )
        return self

    @property
    def hypothesis_digest(self) -> str:
        """Content identity of the complete frozen specification."""
        return hashlib.sha256(canonical_bytes(self)).hexdigest()
