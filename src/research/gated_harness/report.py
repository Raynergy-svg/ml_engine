"""One deterministic evaluation report and binding gate for new research."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from pydantic import Field

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts.base import StrictContract

from .backtest import hard_gate, summarize_returns
from .preregistration import ResearchSpecification
from .regimes import stratify_returns
from .significance import corrected_significance
from .stress import placebo_control, stress_matrix


class ResearchEvaluationReport(StrictContract):
    """Research-harness evaluation report (return-stream admissibility).

    Renamed from ``EvaluationReport`` (2026-07-13) to remove a name collision
    with the formal per-head evidence contract
    :class:`src.evidence.contracts.EvaluationReport`. The two are distinct
    shapes for distinct jobs: this one carries a research strategy's
    significance / regime / stress / control battery; the evidence contract
    carries one model head's holdout metrics and gates inside an
    ``EvidencePackage``. A module-level ``EvaluationReport`` alias is retained
    below for backward compatibility with existing gated-harness callers.
    """

    experiment_id: str
    hypothesis_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer_id: str
    metrics: dict[str, Any]
    significance: dict[str, Any]
    regimes: dict[str, Any]
    stress: dict[str, Any]
    controls: dict[str, Any]
    gates: dict[str, bool]
    effective_n: float
    point_in_time_policy: str
    survivorship_policy: str
    exploratory: bool

    @property
    def report_digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


def evaluate_research(
    specification: ResearchSpecification,
    returns: pd.Series,
    turnover: pd.Series,
    regimes: pd.Series,
    *,
    producer_id: str,
    effective_n: float,
    parameter_returns: dict[str, pd.Series],
    placebo_returns: pd.Series,
    causality_passed: bool,
    untouched_oos: bool,
    temporal_split_digest: str,
    cpcv_path_count: int,
    maximum_drawdown: float,
    minimum_sharpe: float,
) -> ResearchEvaluationReport:
    """Run the shared report path; lane code supplies returns, never verdict math."""
    metrics = summarize_returns(returns)
    significance = corrected_significance(returns, n_trials=specification.trial_budget)
    if len(temporal_split_digest) != 64 or any(
        char not in "0123456789abcdef" for char in temporal_split_digest
    ):
        raise ValueError("temporal_split_digest must be a lowercase SHA-256 digest")
    if cpcv_path_count < 0:
        raise ValueError("cpcv_path_count must be non-negative")
    placebo = placebo_control(
        returns,
        placebo_returns,
        maximum_absolute_placebo_sharpe=specification.maximum_absolute_placebo_sharpe,
    )
    controls = {
        "untouched_oos": untouched_oos,
        "temporal_split_digest": temporal_split_digest,
        "causality_passed": causality_passed,
        "placebo": placebo,
        "cpcv_required": specification.use_cpcv,
        "cpcv_path_count": cpcv_path_count,
    }
    gates = hard_gate(
        metrics,
        minimum_history=specification.minimum_history_observations,
        maximum_drawdown=maximum_drawdown,
        minimum_sharpe=minimum_sharpe,
    )
    gates["minimum_effective_n"] = effective_n >= specification.minimum_effective_n
    gates["untouched_oos"] = untouched_oos
    gates["causality"] = causality_passed
    gates["placebo"] = bool(placebo["passed"])
    gates["cpcv"] = not specification.use_cpcv or cpcv_path_count > 0
    gates["significance"] = (
        significance["passes_significance"] is True
        if specification.mode == "confirmatory"
        else True
    )
    gates["passed"] = all(value for key, value in gates.items() if key != "passed")
    return ResearchEvaluationReport(
        experiment_id=specification.experiment_id,
        hypothesis_digest=specification.hypothesis_digest,
        producer_id=producer_id,
        metrics=metrics,
        significance=significance,
        regimes=stratify_returns(returns, regimes),
        stress=stress_matrix(
            returns,
            turnover,
            cost_scenarios_bps=specification.cost_scenarios_bps,
            parameter_returns=parameter_returns,
        ),
        controls=controls,
        gates=gates,
        effective_n=effective_n,
        point_in_time_policy=specification.point_in_time_policy,
        survivorship_policy=specification.survivorship_policy,
        exploratory=specification.mode == "exploratory",
    )


# Backward-compatibility alias. Prefer ``ResearchEvaluationReport`` in new code;
# the formal per-head evidence contract is ``src.evidence.contracts.EvaluationReport``.
EvaluationReport = ResearchEvaluationReport
