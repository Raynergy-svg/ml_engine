"""Frozen interface contracts for the equity research-alpha pipeline (Track B).

This module is the COMPATIBILITY SPINE for the pipeline pre-registered in
``docs/experiment-equity-research-alpha-prereg-2026-06-30.md``. Every other
module in :mod:`src.equity.research` builds against the dataclasses / Protocol
here, so the loader, blinder, scorer and harness can be developed independently
and still interoperate.

Design invariants that are LOAD-BEARING for the experiment's validity:

* **Point-in-time (PIT).** A :class:`FilingText` carries the ``as_of`` date on
  which its text first became public (``filed`` <= ``as_of``). The loader must
  never return text filed after ``as_of`` — that is lookahead.
* **Entity-blinding is the primary lookahead control.** A :class:`BlindedText`
  deliberately DROPS ``ticker`` / ``cik`` — the blinded text fed to the research
  scorer must not let the model recall the company's known future. The mapping
  back to the issuer lives only in the harness, never in what the scorer sees.
* **Frozen composite weights.** ``PRIMARY_WEIGHTS`` is the pre-registered scoring
  composite (§3.2). It is FROZEN — changing it after seeing results is the L-018
  dredging trap and must not be done.
* **Multiple-testing budget.** ``N_TRIALS`` / ``BONFERRONI_ALPHA`` continue the
  campaign's count (edge-round-4 was 21; this experiment is trial 22). As of
  2026-07-30 they are DERIVED from the append-only ledger in
  :mod:`src.research.trial_budget` rather than hand-edited here — one lane
  hardcoding its own budget (the crypto/EDGAR scripts ran ``N_TRIALS=3``,
  alpha 8x looser) is now a fail-closed error, not a silent divergence.

No pandas / network / LLM imports here on purpose — contracts must stay importable
by every builder with zero heavy deps. ``src.research.trial_budget`` is
stdlib-only for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable

# Re-exported below as this module's public budget surface (see the
# "Multiple-testing budget" block) — every existing consumer imports them
# from here, e.g. src/equity/research/harness.py:67-68.
from src.research.trial_budget import (  # noqa: F401  (intentional re-export)
    BONFERRONI_ALPHA,
    N_TRIALS,
)

# Bumped on ANY change to the loader column set, blinder ruleset, scorer schema,
# or composite weights — artifacts tagged with a different version must refuse to
# load (mirrors FEATURE_PIPELINE_VERSION discipline elsewhere in the repo).
RESEARCH_PIPELINE_VERSION = "1.0.0"

# Pre-registered composite (docs/...prereg-2026-06-30.md §3.2). FROZEN.
# composite = 0.5*quality - 0.5*red_flags + 0.0*outlook  (outlook logged,
# zero-weighted in the PRIMARY arm to keep the most lookahead-prone field from
# driving the book). Do NOT retune after seeing results.
PRIMARY_WEIGHTS: Dict[str, float] = {
    "fundamental_quality": 0.5,
    "accounting_red_flags": -0.5,
    "forward_outlook": 0.0,
}

# Multiple-testing budget — campaign count continues (edge-round-4 = 21; Track B = 22).
# 2026-07-04: bumped 22 -> 24 for the FINRA short-volume pre-reg (2 new trials: H1 + H2).
# See docs/prereg-finra-short-volume-2026-07-04.md. Historical Track B reports that cited
# N_TRIALS=22 / alpha=0.00227 remain valid as-of-then; only NEW calls use the new budget.
#
# 2026-07-30: these are no longer hand-edited constants. They are RE-EXPORTS of the one
# authoritative source, ``src.research.trial_budget``, whose append-only TRIAL_LEDGER
# derives the count (20 + 1 + 1 + 2 = 24 — same value, same alpha, no verdict moves).
# The as-of-then rule above is preserved and made executable: prior budgets live in
# ``trial_budget.REPLAYABLE_BUDGETS`` and are reachable only via an explicit
# ``frozen_replay=True``, while any NEW call declaring its own budget fails closed.
# This is what stops another lane hardcoding N_TRIALS=3 (alpha 8x looser) undetected.
# ``N_TRIALS`` / ``BONFERRONI_ALPHA`` are imported at the top of this module and
# re-exported here for every existing consumer (harness.py:67-68 and friends).
BUDGET_SOURCE = "src.research.trial_budget.TRIAL_LEDGER"

# Arm identifiers for the four pre-registered lookahead-control arms (§1).
ARM_FULL = "full"              # whole sample — EXPECTED contaminated by pretraining
ARM_PRE_CUTOFF = "pre_cutoff"  # before the research model's training cutoff
ARM_POST_CUTOFF = "post_cutoff"  # after cutoff — the only clean backtest arm
ARM_PLACEBO = "placebo"        # entity-blinded text, fundamentals scrambled → ~0
ALL_ARMS = (ARM_FULL, ARM_PRE_CUTOFF, ARM_POST_CUTOFF, ARM_PLACEBO)


@dataclass(frozen=True)
class FilingText:
    """One PIT primary-document text for a single issuer.

    ``as_of`` is the date the text became public knowledge; ``filed`` is the SEC
    filing date and MUST be <= ``as_of``. ``text`` is the raw (pre-blinding)
    primary document. The loader is responsible for the PIT guarantee.
    """

    ticker: str
    cik: str
    as_of: str   # ISO date (UTC) — the PIT knowledge timestamp
    form: str    # "10-K" / "10-Q" / "8-K"
    filed: str   # ISO date (UTC) — SEC filing date; invariant: filed <= as_of
    text: str    # raw primary-document text, pre-blinding

    def __post_init__(self) -> None:
        if self.filed > self.as_of:
            raise ValueError(
                f"PIT violation: filed={self.filed!r} > as_of={self.as_of!r} "
                f"for {self.ticker} ({self.form}) — that is lookahead"
            )


@dataclass(frozen=True)
class BlindedText:
    """Entity-blinded text + a re-identification audit.

    Deliberately carries NO ticker / cik — the scorer must not be able to map
    this text back to a known issuer (the primary lookahead control). ``audit``
    records what was redacted and any residual re-identification risk flags so a
    verifier can sample for leaks.
    """

    as_of: str
    form: str
    filed: str
    text: str                                   # blinded
    audit: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchScore:
    """Structured per-(issuer, as_of) score the scorer emits.

    Ranges are pre-registered (§3.1). A scorer that cannot produce a schema-valid
    score must ABSTAIN (the harness drops that name for that rebalance — never
    zero-fills, mirroring the inference-contract refusal rule).
    """

    ticker: str
    as_of: str
    fundamental_quality: float   # [-1, 1]
    accounting_red_flags: float  # [0, 1]
    forward_outlook: float       # [-1, 1]
    conviction: float            # [0, 1]
    rationale: str
    spans_used: List[str] = field(default_factory=list)

    def composite(self, weights: Dict[str, float] = PRIMARY_WEIGHTS) -> float:
        """Pre-registered linear composite of the score fields."""
        return (
            weights.get("fundamental_quality", 0.0) * self.fundamental_quality
            + weights.get("accounting_red_flags", 0.0) * self.accounting_red_flags
            + weights.get("forward_outlook", 0.0) * self.forward_outlook
        )

    def validate(self) -> None:
        """Raise ValueError if any field is out of its pre-registered range."""
        checks = (
            ("fundamental_quality", self.fundamental_quality, -1.0, 1.0),
            ("accounting_red_flags", self.accounting_red_flags, 0.0, 1.0),
            ("forward_outlook", self.forward_outlook, -1.0, 1.0),
            ("conviction", self.conviction, 0.0, 1.0),
        )
        for name, val, lo, hi in checks:
            fval = float(val)
            if not (lo <= fval <= hi):
                raise ValueError(
                    f"ResearchScore.{name}={fval!r} out of [{lo},{hi}] "
                    f"for {self.ticker} @ {self.as_of}"
                )


@runtime_checkable
class ResearchScorer(Protocol):
    """A scorer maps one BlindedText to one ResearchScore (or raises to abstain).

    Implementations: a deterministic baseline (for plumbing / placebo tests) and
    the offline LLM research pass (scores produced by Claude / subagents and read
    back from a versioned artifact). The harness depends ONLY on this Protocol.
    """

    def score(self, blinded: BlindedText, *, ticker: str) -> ResearchScore:
        ...
