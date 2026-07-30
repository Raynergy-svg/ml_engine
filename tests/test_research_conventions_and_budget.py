"""Adversarial coverage for the two cross-lane research invariants.

Both defects these tests cover were live-but-untriggered on 2026-07-30, and
both are of the "silently satisfied guard" family the repo has been bitten by
before (docs/incidents.md — "$3,527 dead-write", "No-Mock catastrophe"):

1. **Drawdown sign convention.** Six producers emitted a positive magnitude
   under the key ``max_drawdown`` while ``gated_harness.hard_gate`` compared
   with ``>= -abs(limit)``, so a positive-convention 0.863 (86.3% drawdown)
   returned ``True`` from a 25% gate. Nothing cross-fed the two, but Phase E's
   exit gate is to route every lane through that function.
2. **Trial inflation.** ``contracts.py`` ran N_TRIALS=24 (alpha 0.00208) while
   the crypto/EDGAR scripts hardcoded N_TRIALS=3 (alpha 0.01667) — an 8x
   difference in alpha decided by which file you called. There were zero tests
   for trial inflation before this file (`grep 'trial.*inflation'` over
   ``tests/`` returned 0 test names).

No mocks, no test doubles: real producers, real gates, real pydantic contracts,
real return series, real on-disk artifacts (project No-Mock rule).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.crypto.carry_scorecard import _max_drawdown as carry_max_drawdown
from src.crypto.momentum_scorecard import _max_drawdown as momentum_max_drawdown
from src.equity.backtest import stats as equity_stats
from src.equity.research import contracts as equity_contracts
from src.equity.research import harness as equity_harness
from src.factor.backtest import _max_drawdown as factor_max_drawdown
from src.factor.ship_gate import MAX_DRAWDOWN, evaluate_gate
from src.hedge.hedge_scorecard import _max_drawdown as hedge_max_drawdown
from src.research.drawdown_convention import (
    DRAWDOWN_CONVENTION,
    DrawdownConventionError,
    drawdown_fraction,
)
from src.research.gated_harness.backtest import (
    hard_gate,
    max_drawdown as harness_max_drawdown,
    summarize_returns,
)
from src.research.gated_harness.preregistration import ResearchSpecification
from src.research.gated_harness.significance import (
    corrected_significance,
    deflated_sharpe_ratio,
)
from src.research.trial_budget import (
    BONFERRONI_ALPHA,
    DIVERGENT_BUDGETS,
    FAMILY_ALPHA,
    N_TRIALS,
    TRIAL_LEDGER,
    TrialBudgetError,
    resolve_trial_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Fixtures — real return series, deterministic, no doubles
# --------------------------------------------------------------------------

def _crash_returns() -> pd.Series:
    """A series whose compounded equity curve draws down 86.3% peak-to-trough.

    This is the exact magnitude the Phase E audit demonstrated the gate bypass
    with, reproduced here from real arithmetic rather than a hardcoded number.
    """
    index = pd.bdate_range("2020-01-01", periods=6, tz="UTC")
    # 1.0 -> 1.10 (peak) -> ... -> 0.15034 : (1.10 - 0.15034)/1.10 = 0.86331
    return pd.Series([0.10, -0.55, -0.50, -0.3925, 0.05, 0.02], index=index)


def _budget_sensitive_returns() -> pd.Series:
    """A real series (annualized Sharpe 2.01) that PASSES DSR>=0.95 at N=3
    and FAILS it at N=24 — inside the 1.74-2.10 band the audit measured.

    Nothing is asserted about the seed itself — the test asserts the pair of
    DSR values, so if the arithmetic ever changes the test tells you.
    """
    index = pd.bdate_range("2021-01-04", periods=520, tz="UTC")
    rng = np.random.default_rng(20260730)
    return pd.Series(rng.normal(0.00085, 0.0060, 520), index=index)


# --------------------------------------------------------------------------
# DEFECT 1 — drawdown sign convention
# --------------------------------------------------------------------------

def test_shared_gate_refuses_a_wrong_convention_metrics_dict():
    """THE regression test for the 86.3%-passes-a-25%-gate bypass.

    The canonical convention is the positive fraction (six of seven producers
    already emitted it, so no lane's reported number changed). A dict carrying
    the opposite convention must therefore be REFUSED at the gate boundary, not
    compared — refusal is the whole point, because both signs are numerically
    comparable and one of them silently satisfies the guard.
    """
    wrong_convention = {
        "n_observations": 500,
        "sharpe": 1.0,
        "max_drawdown": -0.8633,  # negative convention: 86.3% decline
    }
    with pytest.raises(DrawdownConventionError) as excinfo:
        hard_gate(
            wrong_convention,
            minimum_history=10,
            maximum_drawdown=0.25,
            minimum_sharpe=0.0,
        )
    message = str(excinfo.value)
    assert "max_drawdown" in message
    assert "-0.8633" in message
    assert DRAWDOWN_CONVENTION in message


def test_the_exact_pre_fix_bypass_is_now_a_failed_gate_not_a_pass():
    """86.3% drawdown vs a 25% budget: the old comparison passed, the gate fails.

    The pre-fix expression is written out literally so the test documents what
    it is preventing rather than describing it.
    """
    returns = _crash_returns()
    metrics = summarize_returns(returns)
    observed = metrics["max_drawdown"]
    assert observed == pytest.approx(0.8633, abs=5e-4)

    # The pre-fix gate expression, verbatim: `metrics >= -abs(limit)`.
    assert (observed >= -abs(0.25)) is True, "pre-fix comparison did pass — that was the bug"

    verdict = hard_gate(
        metrics, minimum_history=1, maximum_drawdown=0.25, minimum_sharpe=-10.0
    )
    assert verdict["maximum_drawdown"] is False
    assert verdict["passed"] is False


def test_summarize_returns_emits_the_canonical_positive_convention():
    metrics = summarize_returns(_crash_returns())
    assert metrics["drawdown_convention"] == DRAWDOWN_CONVENTION
    assert metrics["max_drawdown"] > 0.0


def test_every_drawdown_producer_agrees_on_sign_and_magnitude():
    """Cross-producer agreement — the check that did not exist before.

    All six live producers are handed the same crash and must return the same
    non-negative fraction. A future sign flip in any one of them fails here.
    """
    returns = _crash_returns()
    values = [float(r) for r in returns]
    equity = np.cumprod(1.0 + np.asarray(values))

    produced = {
        "gated_harness.backtest.max_drawdown": harness_max_drawdown(returns),
        "factor.backtest._max_drawdown": factor_max_drawdown(equity),
        "equity.backtest.stats[max_dd]": float(equity_stats(returns)["max_dd"]),
        "crypto.momentum_scorecard._max_drawdown": momentum_max_drawdown(values),
        "crypto.carry_scorecard._max_drawdown": carry_max_drawdown(values),
        "hedge.hedge_scorecard._max_drawdown": hedge_max_drawdown(values),
    }
    for name, value in produced.items():
        assert value >= 0.0, f"{name} emitted a negative drawdown: {value!r}"
    # equity.backtest.stats rounds to 3dp; compare at that tolerance.
    assert produced == pytest.approx(
        {name: 0.8633 for name in produced}, abs=1e-3
    )


def test_factor_ship_gate_refuses_the_mirror_image_collision():
    """``factor.ship_gate`` reads the same key with the opposite comparison.

    ``max_dd <= MAX_DRAWDOWN`` is trivially satisfied by any negative value, so
    the same collision exists on the consumer side and must fail closed too.
    """
    report = {
        "net_sharpe": 1.0,
        "positive_years": 10,
        "total_years": 12,
        "max_drawdown": -0.8633,
        "walk_forward": True,
    }
    with pytest.raises(DrawdownConventionError):
        evaluate_gate(report)

    report["max_drawdown"] = 0.8633
    verdict = evaluate_gate(report)
    assert verdict.criteria["max_drawdown"]["passed"] is False
    assert verdict.passed is False

    report["max_drawdown"] = MAX_DRAWDOWN - 0.01
    assert evaluate_gate(report).criteria["max_drawdown"]["passed"] is True


@pytest.mark.parametrize(
    "artifact",
    [
        "trained_data/backtests/crypto_xs_orderflow_20260629T223849Z.json",
        "trained_data/backtests/crypto_funding_carry_20260629T213504Z.json",
        "trained_data/backtests/crypto_cash_and_carry_20260707T015642Z.json",
        "trained_data/backtests/crypto_xs_momentum_20260629T220045Z.json",
    ],
)
def test_real_archived_negative_convention_artifacts_are_refused_not_passed(artifact):
    """The collision is NOT hypothetical — wrong-convention data is on disk.

    Four archived crypto artifacts carry NEGATIVE ``max_drawdown`` values,
    written by ``scripts/experiment_crypto_*.py`` helpers that use the opposite
    convention (their own gates hide it behind ``abs()``). Handing one of those
    blocks to the equity/factor lane's canonical gate used to satisfy
    ``max_dd <= 0.25`` trivially — ``crypto_xs_orderflow`` records -0.995, a
    99.5% drawdown, which passed a 25% budget. It now raises.
    """
    path = REPO_ROOT / artifact
    if not path.exists():
        pytest.skip(f"archived artifact not present: {artifact}")
    blocks = json.loads(path.read_text()).get("blocks", {})
    negatives = {
        name: block["max_drawdown"]
        for name, block in blocks.items()
        if isinstance(block, dict)
        and isinstance(block.get("max_drawdown"), (int, float))
        and block["max_drawdown"] < 0
    }
    assert negatives, f"{artifact} no longer carries negative-convention values"

    for name, value in negatives.items():
        # The pre-fix consumer comparison, verbatim — it did pass.
        assert (value <= 0.25) is True, f"{name}: pre-fix comparison should have passed"
        with pytest.raises(DrawdownConventionError):
            evaluate_gate(
                {
                    "net_sharpe": 1.0,
                    "positive_years": 10,
                    "total_years": 12,
                    "max_drawdown": value,
                    "walk_forward": True,
                }
            )


def test_validator_refuses_percent_scaled_and_non_finite_drawdowns():
    with pytest.raises(DrawdownConventionError, match="FRACTION, not a percentage"):
        drawdown_fraction(86.3, source="test")
    with pytest.raises(DrawdownConventionError, match="not finite"):
        drawdown_fraction(float("nan"), source="test")
    with pytest.raises(DrawdownConventionError, match="not a real number"):
        drawdown_fraction("0.25", source="test")
    # 0.0 is the one value both conventions share and it is admissible.
    assert drawdown_fraction(0.0, source="test") == 0.0
    assert drawdown_fraction(1.0, source="test") == 1.0


def test_gate_refuses_a_metrics_dict_with_no_drawdown_at_all():
    with pytest.raises(DrawdownConventionError, match="no 'max_drawdown' key"):
        hard_gate(
            {"n_observations": 100, "sharpe": 1.0},
            minimum_history=10,
            maximum_drawdown=0.25,
            minimum_sharpe=0.0,
        )


def test_gate_refuses_a_wrong_signed_budget_too():
    """The limit is validated on the same terms as the observation.

    The pre-fix gate wrote ``-abs(maximum_drawdown)``, which quietly accepted a
    budget of either sign and so hid the collision from the caller side as well.
    """
    metrics = summarize_returns(_crash_returns())
    with pytest.raises(DrawdownConventionError):
        hard_gate(
            metrics, minimum_history=1, maximum_drawdown=-0.25, minimum_sharpe=-10.0
        )


# --------------------------------------------------------------------------
# DEFECT 2 — multiple-testing budget / trial inflation
# --------------------------------------------------------------------------

def test_authoritative_budget_is_derived_from_the_ledger_not_a_literal():
    assert N_TRIALS == sum(a.trials for a in TRIAL_LEDGER)
    assert N_TRIALS == 24, "campaign budget changed — update the ledger, not this test"
    assert BONFERRONI_ALPHA == pytest.approx(FAMILY_ALPHA / N_TRIALS)
    # Every allocation carries provenance; an unsourced trial is not a trial.
    for allocation in TRIAL_LEDGER:
        assert allocation.source
        assert allocation.registered_utc


def test_equity_contracts_budget_is_the_shared_source_not_its_own_constant():
    """``contracts.py:52`` used to be a hand-edited ``N_TRIALS = 24``."""
    assert equity_contracts.N_TRIALS is N_TRIALS
    assert equity_contracts.BONFERRONI_ALPHA == pytest.approx(BONFERRONI_ALPHA)
    assert equity_contracts.BUDGET_SOURCE == "src.research.trial_budget.TRIAL_LEDGER"


def test_trial_inflation_flips_this_strategy_and_the_authoritative_budget_binds():
    """Same series, two budgets, two opposite verdicts — the measured defect.

    At the divergent N=3 budget this strategy clears DSR>=0.95; at the campaign's
    N=24 it does not. Which one applies must be decided by the register, not by
    which script the returns were passed to.
    """
    returns = _budget_sensitive_returns()

    at_three = deflated_sharpe_ratio(returns, 3)
    at_authoritative = deflated_sharpe_ratio(returns, N_TRIALS)
    assert at_three is not None and at_authoritative is not None

    assert at_three["dsr"] >= 0.95, (
        "fixture no longer passes at the inflated budget: " f"{at_three['dsr']:.4f}"
    )
    assert at_authoritative["dsr"] < 0.95, (
        "fixture no longer fails at the authoritative budget: "
        f"{at_authoritative['dsr']:.4f}"
    )

    # The authoritative budget is the one a report applies and reports.
    applied = corrected_significance(returns, n_trials=N_TRIALS)
    assert applied["n_trials"] == N_TRIALS
    assert applied["budget_is_authoritative"] is True
    assert applied["bonferroni_alpha"] == pytest.approx(BONFERRONI_ALPHA)

    # And a run at the divergent budget is self-incriminating in the artifact.
    inflated = corrected_significance(returns, n_trials=3)
    assert inflated["budget_is_authoritative"] is False
    assert inflated["authoritative_n_trials"] == N_TRIALS
    assert inflated["bonferroni_alpha"] == pytest.approx(FAMILY_ALPHA / 3)
    assert inflated["bonferroni_alpha"] > applied["bonferroni_alpha"] * 7


def test_confirmatory_specification_refuses_a_self_declared_trial_budget():
    """A lane declaring its own budget for a binding verdict must fail closed."""
    with pytest.raises(ValueError) as excinfo:
        _confirmatory_spec(trial_budget=3)
    message = str(excinfo.value)
    assert "authoritative campaign budget" in message
    assert "N_TRIALS=24" in message
    assert "8.0x looser" in message


def test_confirmatory_specification_accepts_the_authoritative_budget():
    spec = _confirmatory_spec(trial_budget=N_TRIALS)
    assert spec.trial_budget == N_TRIALS
    assert spec.budget_scope == "authoritative"


def test_exploratory_specification_may_declare_its_own_budget():
    """Exploratory runs are not verdicts, so they are not budget-bound."""
    spec = _confirmatory_spec(trial_budget=2, mode="exploratory")
    assert spec.trial_budget == 2


def test_frozen_replay_admits_only_budgets_an_archived_run_actually_used():
    # 22 = Track B before the 2026-07-04 FINRA bump — an archived artifact
    # (trained_data/research/track_b_*/harness_result_*.json) cites exactly it.
    assert (
        resolve_trial_budget(22, context="test", frozen_replay=True) == 22
    )
    # 3 = the divergent crypto/EDGAR budget: replayable, never new.
    assert resolve_trial_budget(3, context="test", frozen_replay=True) == 3
    assert 3 in DIVERGENT_BUDGETS
    # A budget nobody ever ran is refused even under replay.
    with pytest.raises(TrialBudgetError, match="no archived run used that budget"):
        resolve_trial_budget(7, context="test", frozen_replay=True)
    # And replay is opt-in: without the flag the same value is refused.
    with pytest.raises(TrialBudgetError):
        resolve_trial_budget(22, context="test")


def test_equity_harness_derives_alpha_from_the_declared_budget():
    """``_dsr_oos_n22`` used the module alpha regardless of its ``n_trials`` arg.

    At the default budget the derived alpha is identical to the old constant —
    no recorded verdict moves — but the two can no longer disagree.
    """
    returns = _budget_sensitive_returns()
    result = equity_harness._dsr_oos_n22(returns)
    assert result["n_trials"] == N_TRIALS
    assert result["bonferroni_alpha"] == pytest.approx(FAMILY_ALPHA / N_TRIALS)
    assert result["bonferroni_alpha"] == pytest.approx(equity_contracts.BONFERRONI_ALPHA)
    assert result["budget_is_authoritative"] is True

    with pytest.raises(TrialBudgetError):
        equity_harness._dsr_oos_n22(returns, n_trials=3)

    replayed = equity_harness._dsr_oos_n22(returns, n_trials=22, frozen_replay=True)
    assert replayed["bonferroni_alpha"] == pytest.approx(FAMILY_ALPHA / 22)
    assert replayed["budget_is_authoritative"] is False


@pytest.mark.parametrize(
    "artifact, pointers",
    [
        (
            "trained_data/backtests/crypto_funding_carry_20260629T213504Z.json",
            [("gate", "multiple_testing(DSR>=.95 & p<.0167)")],
        ),
        (
            "trained_data/backtests/crypto_xs_momentum_20260629T220045Z.json",
            [("gate", "multiple_testing(DSR>=.95 & p<.0167)")],
        ),
        (
            "trained_data/backtests/crypto_cash_and_carry_20260707T015642Z.json",
            [("gate", "multiple_testing(DSR>=.95 & p<.0167)")],
        ),
        (
            "trained_data/backtests/crypto_round2_20260630T011421Z.json",
            [
                ("H4", "gate", "mt(DSR_N15>=.95 & p<.0033)"),
                ("H5", "gate", "mt(DSR_N15>=.95 & p<.0033)"),
            ],
        ),
        (
            "trained_data/backtests/edgar_value_accruals_bakeoff.json",
            [
                ("significance", "value_oos", "significant_at_bar"),
                ("significance", "accruals_oos", "significant_at_bar"),
            ],
        ),
    ],
)
def test_no_archived_verdict_flips_when_the_budget_is_unified(artifact, pointers):
    """Executable form of the "does unifying the budget change a verdict?" answer.

    Every significance verdict recorded at a divergent budget is already
    NEGATIVE, so tightening alpha cannot flip it — a result that fails at the
    loose budget also fails at the strict one. If a future artifact records a
    PASS at a divergent budget this test fails and the flip must be surfaced
    explicitly rather than absorbed silently.
    """
    path = REPO_ROOT / artifact
    if not path.exists():
        pytest.skip(f"archived artifact not present: {artifact}")
    payload = json.loads(path.read_text())
    for pointer in pointers:
        node = payload
        for key in pointer:
            assert key in node, f"{artifact}: missing {pointer!r}"
            node = node[key]
        assert node is False, (
            f"{artifact}{list(pointer)} recorded a PASS at a divergent trial "
            f"budget. Unifying the budget to N_TRIALS={N_TRIALS} would flip it "
            f"to FAIL — surface this explicitly, do not re-run silently."
        )


# --------------------------------------------------------------------------
# helper
# --------------------------------------------------------------------------

def _confirmatory_spec(
    *, trial_budget: int, mode: str = "confirmatory"
) -> ResearchSpecification:
    return ResearchSpecification(
        experiment_id="trial-inflation-fixture",
        lane_id="crypto",
        hypothesis="funding carry survives after-cost",
        mode=mode,
        target="next-period net return",
        features=("funding_rate",),
        causal_lag=1,
        universe="perp majors",
        evaluation_windows=("2021-01-01/2024-01-01",),
        search_space={"holding_days": (1,)},
        trial_budget=trial_budget,
        correction="bonferroni",
        metrics=("sharpe", "max_drawdown"),
        promotion_criteria={"minimum_sharpe": 0.4},
        cost_scenarios_bps=(0.0, 10.0),
        minimum_history_observations=200,
        minimum_effective_n=3.0,
        maximum_absolute_placebo_sharpe=0.15,
        point_in_time_policy="funding stamped at settlement",
        survivorship_policy="delisted perps retained",
    )
