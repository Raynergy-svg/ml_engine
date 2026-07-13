"""Phase E tests for the canonical cross-lane gated research harness."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from src.evidence.canonical import canonical_bytes
from src.research.gated_harness.backtest import hard_gate, summarize_returns
from src.research.gated_harness.costs import apply_turnover_costs
from src.research.gated_harness.cpcv import combinatorial_purged_splits
from src.research.gated_harness.preregistration import ResearchSpecification
from src.research.gated_harness.purging import purge_train_indices, purged_kfold
from src.research.gated_harness.report import evaluate_research
from src.research.gated_harness.significance import corrected_significance
from src.research.gated_harness.stress import assert_prefix_causality, placebo_control
from src.research.gated_harness.temporal_splits import chronological_holdout
from src.research.gated_harness.verifier import IndependentReplayVerifier


def _spec(*, mode: str = "exploratory", use_cpcv: bool = False) -> ResearchSpecification:
    return ResearchSpecification(
        experiment_id="phase-e-fixture",
        lane_id="equity",
        hypothesis="lagged value scores improve after-cost returns",
        mode=mode,
        target="next-period net return",
        features=("pit_value",),
        causal_lag=1,
        universe="survivorship-aware US equities",
        evaluation_windows=("2010-01-01/2020-01-01",),
        search_space={"rebalance_days": (21,)},
        trial_budget=2,
        correction="bonferroni",
        metrics=("sharpe", "max_drawdown"),
        promotion_criteria={"minimum_sharpe": 0.1},
        cost_scenarios_bps=(0.0, 2.0, 5.0),
        minimum_history_observations=50,
        minimum_effective_n=5.0,
        maximum_absolute_placebo_sharpe=0.15,
        point_in_time_policy="filing timestamp <= rebalance timestamp",
        survivorship_policy="delisted names retained",
        use_cpcv=use_cpcv,
    )


def _series(n: int = 252, *, mean: float = 0.0005, seed: int = 8) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.005, n), index=index)


def test_preregistration_digest_is_frozen_and_content_addressed():
    spec = _spec()
    assert len(spec.hypothesis_digest) == 64
    with pytest.raises(TypeError):
        spec.search_space["rebalance_days"] = (5,)
    changed = spec.model_copy(update={"hypothesis": "a different hypothesis"})
    assert changed.hypothesis_digest != spec.hypothesis_digest


def test_confirmatory_multitrial_spec_requires_correction():
    with pytest.raises(ValueError, match="requires a declared correction"):
        payload = _spec(mode="confirmatory").model_dump()
        payload["correction"] = "none"
        ResearchSpecification.model_validate(payload)


def test_chronological_holdout_has_disjoint_embargoed_tail():
    split = chronological_holdout(100, holdout_fraction=0.2, embargo_observations=5)
    assert list(split.train) == list(range(75))
    assert list(split.embargoed) == list(range(75, 80))
    assert list(split.holdout) == list(range(80, 100))
    assert not set(split.train) & set(split.holdout)
    assert len(split.split_digest) == 64


def test_chronological_holdout_rejects_undersized_history():
    with pytest.raises(ValueError, match="insufficient observations"):
        chronological_holdout(8, embargo_observations=5, minimum_train=2, minimum_holdout=2)


def test_purging_removes_both_label_overlap_and_post_test_embargo():
    train = np.array([0, 1, 2, 3, 7, 8, 9, 10])
    test = np.array([4, 5, 6])
    out = purge_train_indices(train, test, purge_gap=1, embargo_gap=2)
    assert out.tolist() == [0, 1, 2, 10]


def test_purged_kfold_never_overlaps_test():
    for train, test in purged_kfold(30, n_splits=3, purge_gap=2, embargo_gap=1):
        assert not set(train) & set(test)
        assert len(test) == 10


def test_cpcv_yields_every_declared_combination_and_purges():
    splits = list(
        combinatorial_purged_splits(
            60, n_groups=5, n_test_groups=2, purge_gap=2, embargo_gap=1
        )
    )
    assert len(splits) == comb(5, 2)
    for train, test in splits:
        assert not set(train) & set(test)


def test_cost_model_charges_declared_bps_per_turnover():
    returns = pd.Series([0.01, 0.02], index=[0, 1])
    turnover = pd.Series([1.0, 0.5], index=[0, 1])
    result = apply_turnover_costs(returns, turnover, cost_bps=10.0)
    assert result.total_cost == pytest.approx(0.0015)
    assert result.net_returns.tolist() == pytest.approx([0.009, 0.0195])


def test_return_summary_and_drawdown_gate_use_real_history():
    returns = pd.Series([0.1, -0.2, 0.05, 0.02])
    metrics = summarize_returns(returns)
    assert metrics["n_observations"] == 4
    assert metrics["max_drawdown"] == pytest.approx(-0.2)
    verdict = hard_gate(metrics, minimum_history=10, maximum_drawdown=0.15, minimum_sharpe=-10)
    assert verdict["minimum_history"] is False
    assert verdict["maximum_drawdown"] is False
    assert verdict["passed"] is False


def test_significance_abstains_when_history_is_too_short():
    result = corrected_significance(pd.Series([0.1] * 5), n_trials=22)
    assert result["passes_significance"] is None
    assert result["insufficient_data"] is True


def test_significance_is_deterministic_and_trial_adjusted():
    returns = _series(500, mean=0.002, seed=2)
    first = corrected_significance(returns, n_trials=22)
    second = corrected_significance(returns, n_trials=22)
    assert first == second
    assert first["bonferroni_alpha"] == pytest.approx(0.05 / 22)


def test_placebo_control_is_binding():
    observed = _series(252, mean=0.001, seed=4)
    dirty_placebo = _series(252, mean=0.001, seed=5)
    result = placebo_control(
        observed, dirty_placebo, maximum_absolute_placebo_sharpe=0.15
    )
    assert result["passed"] is False


def test_causality_control_detects_future_leakage():
    index = pd.date_range("2020-01-01", periods=10, tz="UTC")
    original = pd.DataFrame({"x": np.arange(10.0)}, index=index)
    mutated = original.copy()
    mutated.loc[index[6]:, "x"] = 1000.0
    def causal(frame):
        return frame.rolling(2).mean()

    def leaking(frame):
        return pd.DataFrame(
            {"x": np.repeat(frame["x"].iloc[-1], len(frame))}, index=frame.index
        )
    assert assert_prefix_causality(causal, original, mutated, cutoff=index[5]) is True
    assert assert_prefix_causality(leaking, original, mutated, cutoff=index[5]) is False


def test_report_records_pit_survivorship_regimes_costs_and_effective_n():
    spec = _spec()
    returns = _series()
    turnover = pd.Series(0.1, index=returns.index)
    regimes = pd.Series(
        np.where(np.arange(len(returns)) % 2, "risk_on", "risk_off"),
        index=returns.index,
    )
    report = evaluate_research(
        spec,
        returns,
        turnover,
        regimes,
        producer_id="worker-a",
        effective_n=10.0,
        parameter_returns={"base": returns, "slower": returns * 0.9},
        placebo_returns=_series(mean=0.0, seed=21),
        causality_passed=True,
        untouched_oos=True,
        temporal_split_digest=chronological_holdout(252).split_digest,
        cpcv_path_count=0,
        maximum_drawdown=0.5,
        minimum_sharpe=-10.0,
    )
    assert report.point_in_time_policy == spec.point_in_time_policy
    assert report.survivorship_policy == spec.survivorship_policy
    assert set(report.regimes) == {"risk_off", "risk_on"}
    assert set(report.stress["cost_scenarios"]) == {"0bps", "2bps", "5bps"}
    assert report.gates["minimum_effective_n"] is True
    assert report.gates["untouched_oos"] is True
    assert report.controls["temporal_split_digest"] == chronological_holdout(252).split_digest
    assert canonical_bytes(report) == canonical_bytes(report)


def test_independent_replay_requires_distinct_identity_and_exact_report():
    spec = _spec()
    returns = _series()
    turnover = pd.Series(0.1, index=returns.index)
    regimes = pd.Series("normal", index=returns.index)

    def run():
        return evaluate_research(
            spec,
            returns,
            turnover,
            regimes,
            producer_id="worker-a",
            effective_n=10.0,
            parameter_returns={"base": returns},
            placebo_returns=_series(mean=0.0, seed=21),
            causality_passed=True,
            untouched_oos=True,
            temporal_split_digest=chronological_holdout(252).split_digest,
            cpcv_path_count=0,
            maximum_drawdown=0.5,
            minimum_sharpe=-10.0,
        )

    produced = run()
    with pytest.raises(ValueError, match="cannot independently verify"):
        IndependentReplayVerifier("worker-a").verify(produced, run)
    verdict = IndependentReplayVerifier("verifier-b").verify(produced, run)
    assert verdict.accepted is True
    assert verdict.expected_digest == verdict.replayed_digest


def test_report_fails_closed_when_methodology_controls_are_missing():
    spec = _spec(use_cpcv=True)
    returns = _series()
    report = evaluate_research(
        spec,
        returns,
        pd.Series(0.1, index=returns.index),
        pd.Series("normal", index=returns.index),
        producer_id="worker-a",
        effective_n=10.0,
        parameter_returns={"base": returns},
        placebo_returns=_series(mean=0.001, seed=22),
        causality_passed=False,
        untouched_oos=False,
        temporal_split_digest=chronological_holdout(252).split_digest,
        cpcv_path_count=0,
        maximum_drawdown=0.5,
        minimum_sharpe=-10.0,
    )
    assert report.gates["untouched_oos"] is False
    assert report.gates["causality"] is False
    assert report.gates["placebo"] is False
    assert report.gates["cpcv"] is False
    assert report.gates["passed"] is False
