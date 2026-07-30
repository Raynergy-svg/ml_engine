"""THE family-wise multiple-testing budget for this repository's research.

Why this module exists
----------------------
Before 2026-07-30 the repo ran two mutually inconsistent family-wise error
budgets at the same time:

* ``src/equity/research/contracts.py:52`` — a hand-maintained ``N_TRIALS = 24``
  (Bonferroni alpha 0.05/24 = 0.00208), documented as edge-round-4 = 21 →
  Track B = 22 → FINRA +2 = 24.
* the crypto and EDGAR experiment scripts — their own ``N_TRIALS = 3``
  literals (alpha 0.05/3 = 0.01667), and ``experiment_crypto_round2.py`` a
  third value, 15.

That is an **8x difference in alpha** decided by which file you happened to
call. Measured on the same return series, a strategy at annualized Sharpe
1.74-2.10 lands DSR >= 0.95 at N=3 and DSR < 0.95 at N=24 — the budget alone
flips the verdict. This is trial inflation: the exact failure the campaign's
trial-count accounting exists to prevent, arriving through per-script literals
that nothing checked against each other.

The fix
-------
The budget is now **derived from one ledger**, not typed into files. Adding a
trial means appending a :class:`TrialAllocation`; ``N_TRIALS`` follows. There
is no hand-edited count left to drift.

Historical budgets are preserved, not rewritten
-----------------------------------------------
``contracts.py:51`` states the intent explicitly: "Historical Track B reports
that cited N_TRIALS=22 / alpha=0.00227 remain valid as-of-then; only NEW calls
use the new budget." That is honoured here. Every budget a lane actually ran
under is registered in :data:`REPLAYABLE_BUDGETS` so an auditor can re-derive
an archived artifact exactly as recorded. What is *refused* is using one of
them for a NEW verdict: :func:`resolve_trial_budget` fail-closes unless the
caller explicitly asks for a frozen replay and names the archived run.

Budgets that were campaign-consistent when declared are separated from budgets
that were not (:data:`DIVERGENT_BUDGETS`) — the N=3 / N=15 lanes never joined
the campaign counter, and calling that out is the finding, not a footnote.

Verdict impact of unifying the budget (checked against every artifact on disk,
2026-07-30): **no recorded verdict flips.** Every significance verdict recorded
at the divergent budgets is already negative —
``trained_data/backtests/crypto_{funding_carry,xs_momentum,xs_orderflow,
cash_and_carry}*.json`` all record
``gate["multiple_testing(DSR>=.95 & p<.0167)"] = False``;
``crypto_round2_*.json`` records ``mt(DSR_N15>=.95 & p<.0033) = False`` for both
H4 and H5; ``edgar_value_accruals_bakeoff.json`` records
``significant_at_bar = false`` for both value and accruals. A verdict that
fails at the loose budget also fails at the strict one. The defect was live and
armed, not yet triggered.

Deliberately stdlib-only so ``src/equity/research/contracts.py`` — which
promises "No pandas / network / LLM imports here on purpose" — can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

#: Family-wise error rate the campaign controls to.
FAMILY_ALPHA: float = 0.05


@dataclass(frozen=True)
class TrialAllocation:
    """One registered increment of the campaign's multiple-testing budget."""

    lane: str
    trials: int
    registered_utc: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError(f"TrialAllocation({self.lane!r}).trials must be >= 1")


#: THE campaign register. Append-only; ``N_TRIALS`` is ``sum(a.trials)``.
#: Reconstructed from the provenance chain documented at
#: ``src/equity/research/contracts.py:48-51`` and the frozen budgets declared in
#: the experiment scripts / pre-registration docs cited per row.
TRIAL_LEDGER: tuple[TrialAllocation, ...] = (
    TrialAllocation(
        lane="edge-hunt-rounds-1-3",
        trials=20,
        registered_utc="2026-06-30",
        source="scripts/experiment_edge_round3_leadA.py:37 (N_TRIALS = 20)",
        note="cumulative search budget carried into round 3",
    ),
    TrialAllocation(
        lane="edge-hunt-round-4",
        trials=1,
        registered_utc="2026-06-30",
        source="scripts/experiment_edge_round4.py:8,35 (N_TRIALS 20 -> 21)",
        note="breadth+history expansion",
    ),
    TrialAllocation(
        lane="equity-track-b",
        trials=1,
        registered_utc="2026-06-30",
        source=(
            "src/equity/research/contracts.py:48 + "
            "docs/experiment-equity-research-alpha-prereg-2026-06-30.md"
        ),
        note="Track B research-alpha pilot = trial 22",
    ),
    TrialAllocation(
        lane="finra-short-volume",
        trials=2,
        registered_utc="2026-07-04",
        source="docs/prereg-finra-short-volume-2026-07-04.md:112-115",
        note="H1 + H2; bumped the campaign count 22 -> 24",
    ),
)

#: The authoritative budget. DERIVED — never hand-edited.
N_TRIALS: int = sum(allocation.trials for allocation in TRIAL_LEDGER)

#: Bonferroni alpha at the authoritative budget (0.05 / 24 = 0.00208...).
BONFERRONI_ALPHA: float = FAMILY_ALPHA / N_TRIALS

#: Budgets that were campaign-consistent at the time they were used. An
#: archived artifact citing one of these is re-derivable exactly as recorded.
FROZEN_AS_OF_BUDGETS: Mapping[int, str] = {
    20: "edge-hunt round 3 (scripts/experiment_edge_round3_leadA.py:37)",
    21: "edge-hunt round 4 (scripts/experiment_edge_round4.py:35)",
    22: "equity Track B before the 2026-07-04 FINRA bump (contracts.py:51)",
    N_TRIALS: "current authoritative campaign budget",
}

#: Budgets that were declared OUTSIDE the campaign register. These lanes never
#: joined the counter; their alpha was up to 8x looser than the campaign's.
#: Registered so archived artifacts stay re-derivable — NOT admissible for any
#: new verdict.
DIVERGENT_BUDGETS: Mapping[int, str] = {
    3: (
        "crypto H1-H3 / cash-and-carry / EDGAR value+accruals "
        "(scripts/experiment_crypto_funding_carry.py:48, "
        "scripts/experiment_crypto_xs_signals.py:44, "
        "scripts/experiment_crypto_cash_and_carry.py:48, "
        "scripts/experiment_edgar_value_accruals_2026_07_02.py:83; "
        "docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md:78) "
        "-- alpha 0.01667, 8x looser than the campaign's 0.00208"
    ),
    15: (
        "crypto round 2 H4/H5 (scripts/experiment_crypto_round2.py:35) "
        "-- alpha 0.00333"
    ),
}

#: Every budget an archived run may legitimately be replayed at.
REPLAYABLE_BUDGETS: Mapping[int, str] = {**DIVERGENT_BUDGETS, **FROZEN_AS_OF_BUDGETS}


class TrialBudgetError(ValueError):
    """A trial budget was declared that the campaign register does not admit."""


def bonferroni_alpha(n_trials: Optional[int] = None) -> float:
    """Bonferroni alpha for ``n_trials`` (default: the authoritative budget)."""
    trials = N_TRIALS if n_trials is None else int(n_trials)
    if trials < 1:
        raise TrialBudgetError(f"n_trials must be >= 1, got {trials!r}")
    return FAMILY_ALPHA / trials


def resolve_trial_budget(
    declared: Optional[int] = None,
    *,
    context: str,
    frozen_replay: bool = False,
) -> int:
    """Return the trial budget to apply, fail-closed.

    * ``declared is None`` -> the authoritative budget.
    * ``declared == N_TRIALS`` -> accepted.
    * ``frozen_replay=True`` and ``declared`` in :data:`REPLAYABLE_BUDGETS` ->
      accepted, for re-deriving an archived artifact exactly as recorded.
    * anything else -> :class:`TrialBudgetError`, naming the declared value,
      the authoritative value and how much looser the declared alpha is.

    ``context`` identifies the caller in the error (lane, experiment id, …).
    """
    if declared is None:
        return N_TRIALS
    trials = int(declared)
    if trials < 1:
        raise TrialBudgetError(
            f"{context}: trial budget {trials!r} must be >= 1"
        )
    if trials == N_TRIALS:
        return trials
    if frozen_replay:
        if trials in REPLAYABLE_BUDGETS:
            return trials
        raise TrialBudgetError(
            f"{context}: frozen replay requested at N_TRIALS={trials}, but no "
            f"archived run used that budget. Replayable budgets: "
            f"{sorted(REPLAYABLE_BUDGETS)!r}"
        )
    looseness = trials and (N_TRIALS / trials)
    divergent = DIVERGENT_BUDGETS.get(trials)
    detail = f" That budget is a known divergent one: {divergent}." if divergent else ""
    raise TrialBudgetError(
        f"{context}: declared trial budget N_TRIALS={trials} does not match the "
        f"authoritative campaign budget N_TRIALS={N_TRIALS} "
        f"(alpha {bonferroni_alpha(trials):.5f} vs {BONFERRONI_ALPHA:.5f} — "
        f"{looseness:.1f}x looser).{detail} A new verdict must use the campaign "
        f"budget from src/research/trial_budget.TRIAL_LEDGER. To re-derive an "
        f"archived result exactly as recorded, pass frozen_replay=True."
    )


def assert_authoritative_budget(n_trials: int, *, context: str) -> int:
    """Assert ``n_trials`` is the authoritative budget; raise otherwise."""
    return resolve_trial_budget(n_trials, context=context, frozen_replay=False)


def ledger_summary() -> dict[str, object]:
    """Machine-readable provenance for embedding in a report artifact."""
    return {
        "n_trials": N_TRIALS,
        "family_alpha": FAMILY_ALPHA,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "allocations": [
            {
                "lane": a.lane,
                "trials": a.trials,
                "registered_utc": a.registered_utc,
                "source": a.source,
                "note": a.note,
            }
            for a in TRIAL_LEDGER
        ],
    }
