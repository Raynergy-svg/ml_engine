"""AXIOM Hedge Layer — Phase 3: hedge candidate generator (SHADOW / ANALYSIS-ONLY).

Given a candidate trade and the netted exposure picture it produces (Phase 1
tagging + Phase 2 netting, ``src/hedge/exposure_tags.py`` +
``src/hedge/portfolio_exposure.py``), propose ranked hedge options that would
reduce a flagged concentration bucket — never a bare "yes, trade it" verdict
with no accounting for what it does to the book.

This module NEVER places, modifies, or cancels an order; never unhalts a
lane; never arms a live gate; never bypasses a gate; never changes leverage.
It is a pure decision-support layer: every output carries the fixed safety
triplet ``runtime_allowed=False, paper_only=True, human_review_required=True``
and every ``decision`` is advisory text for a human to act on, not an
executable instruction.

Design choices worth stating explicitly (so a reader doesn't mistake a
deliberate simplification for a bug):

* Hedge legs are sized to FULLY offset the flagged bucket by construction
  (direct-offset / market / sector styles). This keeps every generated
  proposal's ``exposure_reduction_pct`` == 1.0 and testable; capital-
  efficiency tradeoffs (is a full hedge worth its cost?) are handled by the
  cost-ratio decision tiers below, not by partial sizing.
* Relative-value proposals (FX: long strongest counter-currency / short
  weakest; equity: long strongest basket / short weakest) are reported with
  ``exposure_reduction=0.0`` and ``correlation_to_target=None`` — they
  isolate a *different* risk (relative/idiosyncratic view) rather than
  reduce the flagged bucket's dollar exposure, and this module has no live
  return-series data to prove otherwise. Claiming a dollar reduction it
  can't verify would be fabrication (see ``docs/incidents.md``); BOTH the
  FX and equity RV styles require the caller to pass a ``relative_strength``
  ranking (e.g. trailing returns) — without one there is no data to back a
  "strongest/weakest" claim, so the style fails closed
  (``no_relative_strength_data_supplied``) rather than picking an arbitrary
  pair (e.g. alphabetically) and calling it a ranked view.
* Cost is read from the real P2 execution-cost model
  (``src.data.execution_cost_model.estimate_execution_cost``), not forked.
  A hedge's dollar cost is approximated as
  ``half_spread_cost_per_unit * risk_home * 2`` (round-trip) — this treats
  ``risk_home`` (an abstract dollar-risk magnitude, per Phase 2) as
  interchangeable with the cost model's per-unit dollar figure. That is a
  simplification appropriate for a shadow analysis layer, not an exact
  position-sizing conversion; documented here so it is never mistaken for
  one. When the cost model reports ``insufficient_data`` (no fill/tick
  history for that instrument), cost is UNKNOWN — never assumed cheap.
* ``correlation_to_target`` is a structural/definitional proxy (1.0 for a
  hedge instrument that IS the target factor by design — e.g. SPY for
  market beta, a sector ETF for its own sector, a direct FX currency
  offset), not a fitted historical price correlation. No live correlation
  matrix exists in this repo; a fabricated fitted number would be worse
  than an honest, clearly-labeled definitional one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.data.execution_cost_model import CostEstimate, estimate_execution_cost
from src.hedge.exposure_tags import (
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
    tag_fx_exposure,
)
from src.hedge.portfolio_exposure import ExposurePosition, ExposureReport

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Fixed safety fields — SHADOW/ANALYSIS-ONLY. Module-level constants,   #
# never parameters, so no call site can ever override them.             #
# --------------------------------------------------------------------- #
RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

DECISION_APPROVE_RAW = "APPROVE_RAW"
DECISION_APPROVE_HEDGED = "APPROVE_HEDGED"
DECISION_REDUCE_SIZE = "REDUCE_SIZE"
DECISION_BLOCK_CORRELATION = "BLOCK_CORRELATION"
DECISION_BLOCK_COST = "BLOCK_COST"
DECISION_BLOCK_NO_VALID_HEDGE = "BLOCK_NO_VALID_HEDGE"
DECISION_SHADOW_ONLY = "SHADOW_ONLY"

MARKET_HEDGE_INSTRUMENT = "SPY"

# A hedge whose approximate round-trip cost exceeds this fraction of the
# dollar exposure it removes is not worth putting on.
DEFAULT_MAX_COST_TO_REDUCTION_RATIO = 0.5
# Below this fraction, the hedge is cheap enough to just put on outright.
# Between low and max, a full hedge works but isn't cheap — trimming the
# candidate's own size is the more capital-efficient fix (REDUCE_SIZE).
DEFAULT_LOW_COST_TO_REDUCTION_RATIO = 0.15
# A hedge whose correlation to the flagged bucket is below this floor isn't
# trustworthy as a hedge for that bucket, regardless of cost.
DEFAULT_MIN_CORRELATION_TO_TARGET = 0.3

CostLookup = Callable[[str], CostEstimate]


# --------------------------------------------------------------------- #
# Data shapes                                                           #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class HedgeLeg:
    instrument: str
    direction: str  # "long" | "short"
    risk_home: float

    def to_dict(self) -> dict:
        return {"instrument": self.instrument, "direction": self.direction,
                "risk_home": self.risk_home}


@dataclass(frozen=True)
class HedgeProposal:
    style: str  # "direct_offset" | "relative_value" | "market" | "sector"
    legs: Tuple[HedgeLeg, ...]
    neutralizes: str            # which exposure bucket this targets, e.g. "currency:USD"
    isolates: str                # narrative: what risk survives / what view this expresses
    exposure_reduction: float    # dollar risk_home removed from the flagged bucket
    exposure_reduction_pct: Optional[float]
    correlation_to_target: Optional[float]  # structural proxy in [0,1]; None if not applicable/unknown
    cost_known: bool
    cost_estimate_dollars: Optional[float]
    cost_to_reduction_ratio: Optional[float]
    cost_source: str
    cost_confidence: float
    fail_closed: bool = False
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "legs": [leg.to_dict() for leg in self.legs],
            "neutralizes": self.neutralizes,
            "isolates": self.isolates,
            "exposure_reduction": self.exposure_reduction,
            "exposure_reduction_pct": self.exposure_reduction_pct,
            "correlation_to_target": self.correlation_to_target,
            "cost_known": self.cost_known,
            "cost_estimate_dollars": self.cost_estimate_dollars,
            "cost_to_reduction_ratio": self.cost_to_reduction_ratio,
            "cost_source": self.cost_source,
            "cost_confidence": self.cost_confidence,
            "fail_closed": self.fail_closed,
            "reasons": list(self.reasons),
        }


def _no_valid_hedge_proposal(style: str, neutralizes: str, reasons: Tuple[str, ...]) -> HedgeProposal:
    return HedgeProposal(
        style=style, legs=(), neutralizes=neutralizes,
        isolates="none — no structurally valid hedge could be constructed",
        exposure_reduction=0.0, exposure_reduction_pct=None, correlation_to_target=None,
        cost_known=False, cost_estimate_dollars=None, cost_to_reduction_ratio=None,
        cost_source="n/a", cost_confidence=0.0, fail_closed=True, reasons=reasons)


# --------------------------------------------------------------------- #
# Cost evaluation — reuses src.data.execution_cost_model, does not fork #
# --------------------------------------------------------------------- #
def _evaluate_cost(instrument: str, risk_home: float, exposure_reduction: float,
                   cost_lookup: CostLookup) -> Tuple[bool, Optional[float], Optional[float], str, float]:
    """Returns (cost_known, cost_estimate_dollars, cost_to_reduction_ratio, source, confidence)."""
    try:
        estimate = cost_lookup(instrument)
    except Exception as exc:  # cost model itself is offline/read-only; never let a lookup failure
        logger.warning("hedge: cost lookup failed for %s (%s) — treating cost as unknown", instrument, exc)
        return False, None, None, "lookup_error", 0.0

    per_unit = estimate.half_spread_cost_per_unit
    if per_unit is None or estimate.source == "insufficient_data":
        return False, None, None, estimate.source, estimate.confidence

    cost_dollars = abs(per_unit) * abs(risk_home) * 2.0  # round trip
    ratio = (cost_dollars / exposure_reduction) if exposure_reduction > 0 else None
    return True, cost_dollars, ratio, estimate.source, estimate.confidence


# --------------------------------------------------------------------- #
# FX hedges                                                             #
# --------------------------------------------------------------------- #
def _pick_counter_currency(flagged_ccy: str, net_currency: Dict[str, float],
                           currency_bucket_map: Dict[str, dict]) -> Optional[str]:
    """Prefer a currency already in the book with the smallest |net| (least
    disruptive to add to); fall back to any other mapped currency."""
    in_book = sorted((c for c in net_currency if c != flagged_ccy),
                     key=lambda c: (abs(net_currency[c]), c))
    if in_book:
        return in_book[0]
    others = sorted(c for c in currency_bucket_map if c != flagged_ccy)
    return others[0] if others else None


def generate_fx_hedges(candidate: ExposurePosition, portfolio_exposure: Optional[ExposureReport], *,
                       currency_bucket_map: Optional[Dict[str, dict]] = None,
                       cost_lookup: Optional[CostLookup] = None,
                       relative_strength: Optional[Dict[str, float]] = None) -> List[HedgeProposal]:
    """Propose FX hedges that reduce the largest-magnitude net currency
    bucket in ``portfolio_exposure`` (the book netted with ``candidate``
    already folded in, per Phase 2 convention). Fails closed (a single
    fail_closed proposal, never a silent empty list standing in for
    "nothing wrong") when the exposure data itself is unresolved.

    The relative-value style requires a caller-supplied ``relative_strength``
    ranking (e.g. trailing returns per currency) — same contract as
    ``generate_equity_hedges``. Without one, this module has no data to back
    a "does X outperform Y" claim, so it fails closed rather than picking an
    arbitrary pair and calling it a ranked view."""
    if candidate.asset_class != "fx":
        return []
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    if cost_lookup is None:
        cost_lookup = estimate_execution_cost

    if portfolio_exposure is None or portfolio_exposure.fail_closed:
        reasons = portfolio_exposure.fail_reasons if portfolio_exposure else ("missing_portfolio_exposure",)
        return [_no_valid_hedge_proposal("direct_offset", "currency:unknown", tuple(reasons) or
                                         ("portfolio_exposure_fail_closed",))]

    net_currency = portfolio_exposure.net_currency_exposure
    if not net_currency:
        return []  # flat book — nothing to hedge

    flagged_ccy, flagged_net = max(net_currency.items(), key=lambda kv: (abs(kv[1]), kv[0]))
    if abs(flagged_net) <= 0:
        return []

    proposals: List[HedgeProposal] = []

    # --- style: direct offset -------------------------------------------------
    counter_ccy = _pick_counter_currency(flagged_ccy, net_currency, currency_bucket_map)
    if counter_ccy is None:
        proposals.append(_no_valid_hedge_proposal(
            "direct_offset", f"currency:{flagged_ccy}",
            (f"no_counter_currency_available_for:{flagged_ccy}",)))
    else:
        instrument = f"{flagged_ccy}_{counter_ccy}"
        direction = "short" if flagged_net > 0 else "long"
        risk_home = abs(flagged_net)
        tag = tag_fx_exposure(instrument, direction, risk_home, currency_bucket_map)
        if tag.fail_closed:
            proposals.append(_no_valid_hedge_proposal(
                "direct_offset", f"currency:{flagged_ccy}", tag.reasons))
        else:
            cost_known, cost_dollars, ratio, source, confidence = _evaluate_cost(
                instrument, risk_home, risk_home, cost_lookup)
            proposals.append(HedgeProposal(
                style="direct_offset",
                legs=(HedgeLeg(instrument, direction, risk_home),),
                neutralizes=f"currency:{flagged_ccy}",
                isolates=(
                    f"Sells the {flagged_ccy} macro bet outright against {counter_ccy}; "
                    f"removes the {flagged_ccy} concentration but adds fresh outright "
                    f"{counter_ccy} directional risk in its place (not itself hedged)."),
                exposure_reduction=risk_home,
                exposure_reduction_pct=1.0,
                correlation_to_target=1.0,
                cost_known=cost_known, cost_estimate_dollars=cost_dollars,
                cost_to_reduction_ratio=ratio, cost_source=source, cost_confidence=confidence))

    # --- style: relative value --------------------------------------------
    counter_currencies = [c for c in net_currency if c != flagged_ccy]
    ranked = (relative_strength or {})
    ranked_counters = sorted((c for c in counter_currencies if c in ranked),
                             key=lambda c: (ranked[c], c))
    if len(ranked_counters) < 2:
        if len(counter_currencies) >= 2:
            # A relative-value leg IS structurally possible here -- the gap
            # is missing ranking data, not missing currencies. Record it
            # rather than silently omitting the style or fabricating a pick.
            proposals.append(_no_valid_hedge_proposal(
                "relative_value", f"currency:{flagged_ccy}",
                ("no_relative_strength_data_supplied",)))
    else:
        weak, strong = ranked_counters[0], ranked_counters[-1]
        leg_size = abs(flagged_net) / 2.0
        long_inst = f"{strong}_{flagged_ccy}"
        short_inst = f"{weak}_{flagged_ccy}"
        long_tag = tag_fx_exposure(long_inst, "long", leg_size, currency_bucket_map)
        short_tag = tag_fx_exposure(short_inst, "short", leg_size, currency_bucket_map)
        if long_tag.fail_closed or short_tag.fail_closed:
            proposals.append(_no_valid_hedge_proposal(
                "relative_value", f"currency:{flagged_ccy}",
                tuple(long_tag.reasons) + tuple(short_tag.reasons)))
        else:
            cost_known_l, cost_l, _, source_l, conf_l = _evaluate_cost(long_inst, leg_size, 1.0, cost_lookup)
            cost_known_s, cost_s, _, source_s, conf_s = _evaluate_cost(short_inst, leg_size, 1.0, cost_lookup)
            both_known = cost_known_l and cost_known_s
            total_cost = (cost_l + cost_s) if both_known else None
            proposals.append(HedgeProposal(
                style="relative_value",
                legs=(HedgeLeg(long_inst, "long", leg_size), HedgeLeg(short_inst, "short", leg_size)),
                neutralizes=f"currency:{flagged_ccy}",
                isolates=(
                    f"Long {strong}/{flagged_ccy}, short {weak}/{flagged_ccy} — does {strong} "
                    f"outperform {weak} net of {flagged_ccy} noise. Structurally "
                    f"{flagged_ccy}-neutral (legs cancel in {flagged_ccy}), so it does NOT "
                    f"reduce the flagged {flagged_ccy} dollar exposure — reported as 0 "
                    "reduction rather than an unverified estimate; it re-expresses a "
                    f"directional view as a relative one instead of accumulating more "
                    f"{flagged_ccy} risk."),
                exposure_reduction=0.0,
                exposure_reduction_pct=0.0,
                correlation_to_target=None,
                cost_known=both_known, cost_estimate_dollars=total_cost,
                cost_to_reduction_ratio=None,  # exposure_reduction is 0 -- ratio undefined, not infinite
                cost_source=f"{source_l}+{source_s}", cost_confidence=min(conf_l, conf_s)))

    return proposals


# --------------------------------------------------------------------- #
# Equity hedges                                                         #
# --------------------------------------------------------------------- #
def _sector_benchmark_hedge(sector: str, sector_bucket_map: Dict[str, dict]) -> Optional[str]:
    for entry in sector_bucket_map.values():
        if entry.get("sector") == sector and entry.get("benchmark_hedge"):
            return entry["benchmark_hedge"]
    return None


def generate_equity_hedges(candidate: ExposurePosition, portfolio_exposure: Optional[ExposureReport], *,
                           sector_bucket_map: Optional[Dict[str, dict]] = None,
                           cost_lookup: Optional[CostLookup] = None,
                           relative_strength: Optional[Dict[str, float]] = None) -> List[HedgeProposal]:
    """Propose equity hedges in three styles: market (short SPY, isolates
    "beating the market"), sector (short the sector ETF, isolates "beating
    its sector"), relative-value (long strongest / short weakest of a
    caller-supplied ``relative_strength`` ranking — isolates "which name
    wins"). Fails closed on unresolved exposure data, same as
    ``generate_fx_hedges``."""
    if candidate.asset_class != "equity":
        return []
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    if cost_lookup is None:
        cost_lookup = estimate_execution_cost

    if portfolio_exposure is None or portfolio_exposure.fail_closed:
        reasons = portfolio_exposure.fail_reasons if portfolio_exposure else ("missing_portfolio_exposure",)
        return [_no_valid_hedge_proposal("market", "sector:unknown", tuple(reasons) or
                                         ("portfolio_exposure_fail_closed",))]

    proposals: List[HedgeProposal] = []

    # --- style: market ------------------------------------------------------
    net_beta = portfolio_exposure.net_beta_exposure
    if net_beta is None:
        proposals.append(_no_valid_hedge_proposal("market", "beta:portfolio", ("beta_unresolved",)))
    elif abs(net_beta) > 0:
        direction = "short" if net_beta > 0 else "long"
        risk_home = abs(net_beta)
        cost_known, cost_dollars, ratio, source, confidence = _evaluate_cost(
            MARKET_HEDGE_INSTRUMENT, risk_home, risk_home, cost_lookup)
        proposals.append(HedgeProposal(
            style="market",
            legs=(HedgeLeg(MARKET_HEDGE_INSTRUMENT, direction, risk_home),),
            neutralizes="beta:portfolio",
            isolates=(
                f"{'Short' if direction == 'short' else 'Long'} {MARKET_HEDGE_INSTRUMENT} against the "
                "book's net beta-weighted exposure — removes broad market direction; what's left is "
                "'does this book beat the market', not 'did the market go up'."),
            exposure_reduction=risk_home, exposure_reduction_pct=1.0,
            correlation_to_target=1.0,
            cost_known=cost_known, cost_estimate_dollars=cost_dollars,
            cost_to_reduction_ratio=ratio, cost_source=source, cost_confidence=confidence))

    # --- style: sector --------------------------------------------------------
    net_sector = portfolio_exposure.net_sector_exposure
    if net_sector:
        flagged_sector, flagged_net = max(net_sector.items(), key=lambda kv: (abs(kv[1]), kv[0]))
        if abs(flagged_net) > 0:
            hedge_ticker = _sector_benchmark_hedge(flagged_sector, sector_bucket_map)
            if hedge_ticker is None:
                proposals.append(_no_valid_hedge_proposal(
                    "sector", f"sector:{flagged_sector}",
                    (f"no_benchmark_hedge_mapped_for_sector:{flagged_sector}",)))
            else:
                direction = "short" if flagged_net > 0 else "long"
                risk_home = abs(flagged_net)
                cost_known, cost_dollars, ratio, source, confidence = _evaluate_cost(
                    hedge_ticker, risk_home, risk_home, cost_lookup)
                proposals.append(HedgeProposal(
                    style="sector",
                    legs=(HedgeLeg(hedge_ticker, direction, risk_home),),
                    neutralizes=f"sector:{flagged_sector}",
                    isolates=(
                        f"{'Short' if direction == 'short' else 'Long'} {hedge_ticker} against the "
                        f"{flagged_sector} concentration — removes sector direction; what's left is "
                        f"'does this name beat/lag {flagged_sector}', not the sector's own move."),
                    exposure_reduction=risk_home, exposure_reduction_pct=1.0,
                    correlation_to_target=1.0,
                    cost_known=cost_known, cost_estimate_dollars=cost_dollars,
                    cost_to_reduction_ratio=ratio, cost_source=source, cost_confidence=confidence))

    # --- style: relative value --------------------------------------------
    if relative_strength and len(relative_strength) >= 2:
        strongest = max(relative_strength.items(), key=lambda kv: (kv[1], kv[0]))[0]
        weakest = min(relative_strength.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if strongest != weakest:
            leg_size = abs(candidate.risk_home) if candidate.risk_home else 1.0
            cost_known_l, cost_l, _, source_l, conf_l = _evaluate_cost(strongest, leg_size, 1.0, cost_lookup)
            cost_known_s, cost_s, _, source_s, conf_s = _evaluate_cost(weakest, leg_size, 1.0, cost_lookup)
            both_known = cost_known_l and cost_known_s
            total_cost = (cost_l + cost_s) if both_known else None
            proposals.append(HedgeProposal(
                style="relative_value",
                legs=(HedgeLeg(strongest, "long", leg_size), HedgeLeg(weakest, "short", leg_size)),
                neutralizes="beta:portfolio+sector",
                isolates=(
                    f"Long {strongest}, short {weakest} (caller-supplied relative-strength ranking) — "
                    "market/sector beta approximately cancels between two same-book names by "
                    "construction; net dollar reduction of the flagged bucket is reported "
                    "conservatively as 0 pending real correlation data, not an unverified estimate. "
                    "What's left is pure relative ranking risk."),
                exposure_reduction=0.0, exposure_reduction_pct=0.0,
                correlation_to_target=None,
                cost_known=both_known, cost_estimate_dollars=total_cost,
                cost_to_reduction_ratio=None,
                cost_source=f"{source_l}+{source_s}", cost_confidence=min(conf_l, conf_s)))
    elif net_sector or (net_beta is not None and abs(net_beta) > 0):
        # Something WAS flagged, but no relative-strength data was supplied to
        # rank a relative-value leg -- record the gap, don't silently omit it.
        proposals.append(_no_valid_hedge_proposal(
            "relative_value", "sector:relative_ranking", ("no_relative_strength_data_supplied",)))

    return proposals


# --------------------------------------------------------------------- #
# Ranking                                                               #
# --------------------------------------------------------------------- #
def rank_hedges_by_cost(proposals: Sequence[HedgeProposal]) -> List[HedgeProposal]:
    """Cheapest-known-cost-per-dollar-of-reduction first. Unknown cost is
    NEVER assumed cheap — sorted after every known-cost proposal, ahead only
    of fail-closed (no-hedge) placeholders."""
    def key(p: HedgeProposal):
        if p.fail_closed:
            return (2, 0.0)
        if p.cost_to_reduction_ratio is None:
            return (1, 0.0)
        return (0, p.cost_to_reduction_ratio)
    return sorted(proposals, key=key)


def rank_hedges_by_correlation(proposals: Sequence[HedgeProposal]) -> List[HedgeProposal]:
    """Highest structural correlation-to-target first; unknown/inapplicable
    correlation sorted last (never assumed to be a good hedge)."""
    def key(p: HedgeProposal):
        if p.fail_closed or p.correlation_to_target is None:
            return (1, 0.0)
        return (0, -p.correlation_to_target)
    return sorted(proposals, key=key)


def rank_hedges_by_exposure_reduction(proposals: Sequence[HedgeProposal]) -> List[HedgeProposal]:
    """Largest dollar exposure removed from the flagged bucket first."""
    return sorted(proposals, key=lambda p: (p.fail_closed, -p.exposure_reduction))


# --------------------------------------------------------------------- #
# Decision + structured report                                          #
# --------------------------------------------------------------------- #
def _decide(portfolio_exposure: Optional[ExposureReport], ranked_by_cost: List[HedgeProposal], *,
            max_cost_to_reduction_ratio: float, low_cost_to_reduction_ratio: float,
            min_correlation_to_target: float) -> str:
    if portfolio_exposure is None or portfolio_exposure.fail_closed:
        return DECISION_SHADOW_ONLY
    if not portfolio_exposure.concentration_warnings:
        return DECISION_APPROVE_RAW

    structurally_valid = [p for p in ranked_by_cost if not p.fail_closed]
    if not structurally_valid:
        return DECISION_BLOCK_NO_VALID_HEDGE

    best = structurally_valid[0]
    if not best.cost_known or best.cost_to_reduction_ratio is None:
        return DECISION_BLOCK_COST
    if best.cost_to_reduction_ratio > max_cost_to_reduction_ratio:
        return DECISION_BLOCK_COST
    if best.correlation_to_target is not None and best.correlation_to_target < min_correlation_to_target:
        return DECISION_BLOCK_CORRELATION
    if best.cost_to_reduction_ratio > low_cost_to_reduction_ratio:
        return DECISION_REDUCE_SIZE
    return DECISION_APPROVE_HEDGED


def _candidate_to_dict(candidate: ExposurePosition) -> dict:
    return {
        "asset_class": candidate.asset_class,
        "instrument": candidate.instrument,
        "direction": candidate.direction,
        "risk_home": candidate.risk_home,
        "source": candidate.source,
    }


def _risk_issue(portfolio_exposure: Optional[ExposureReport]) -> str:
    if portfolio_exposure is None:
        return "no exposure data supplied — cannot evaluate, fail-closed."
    if portfolio_exposure.fail_closed:
        return (f"exposure data partially unresolved ({list(portfolio_exposure.fail_reasons)}) — "
                "report is incomplete, fail-closed rather than treating gaps as zero exposure.")
    if portfolio_exposure.concentration_warnings:
        return portfolio_exposure.concentration_warnings[0]
    return "no concentration detected — candidate does not appear to be a disguised correlated bet."


def build_hedge_report(candidate: ExposurePosition, portfolio_exposure: Optional[ExposureReport], *,
                       currency_bucket_map: Optional[Dict[str, dict]] = None,
                       sector_bucket_map: Optional[Dict[str, dict]] = None,
                       cost_lookup: Optional[CostLookup] = None,
                       relative_strength: Optional[Dict[str, float]] = None,
                       max_cost_to_reduction_ratio: float = DEFAULT_MAX_COST_TO_REDUCTION_RATIO,
                       low_cost_to_reduction_ratio: float = DEFAULT_LOW_COST_TO_REDUCTION_RATIO,
                       min_correlation_to_target: float = DEFAULT_MIN_CORRELATION_TO_TARGET) -> dict:
    """Full structured hedge-proposal report for one candidate trade.

    Routes to ``generate_fx_hedges`` or ``generate_equity_hedges`` by
    ``candidate.asset_class``, ranks by cost (the primary ordering — a hedge
    that costs more than the noise it removes is not a hedge), and derives a
    single advisory ``decision``. Every output carries the fixed
    SHADOW/ANALYSIS-ONLY safety triplet; nothing here is executable.
    """
    if candidate.asset_class == "fx":
        proposals = generate_fx_hedges(candidate, portfolio_exposure,
                                       currency_bucket_map=currency_bucket_map, cost_lookup=cost_lookup,
                                       relative_strength=relative_strength)
    elif candidate.asset_class == "equity":
        proposals = generate_equity_hedges(candidate, portfolio_exposure,
                                           sector_bucket_map=sector_bucket_map, cost_lookup=cost_lookup,
                                           relative_strength=relative_strength)
    else:
        proposals = [_no_valid_hedge_proposal(
            "direct_offset", "unknown", (f"unsupported_asset_class:{candidate.asset_class}",))]

    ranked = rank_hedges_by_cost(proposals)
    decision = _decide(portfolio_exposure, ranked,
                       max_cost_to_reduction_ratio=max_cost_to_reduction_ratio,
                       low_cost_to_reduction_ratio=low_cost_to_reduction_ratio,
                       min_correlation_to_target=min_correlation_to_target)

    detected_exposures = {
        "concentration_warnings": list(portfolio_exposure.concentration_warnings) if portfolio_exposure else [],
        "fail_closed": bool(portfolio_exposure.fail_closed) if portfolio_exposure else True,
        "fail_reasons": list(portfolio_exposure.fail_reasons) if portfolio_exposure else ("missing_portfolio_exposure",),
    }
    portfolio_after_trade = portfolio_exposure.to_dict() if portfolio_exposure else None

    return {
        "candidate_trade": _candidate_to_dict(candidate),
        "detected_exposures": detected_exposures,
        "portfolio_after_trade": portfolio_after_trade,
        "risk_issue": _risk_issue(portfolio_exposure),
        "hedge_action": [p.to_dict() for p in ranked],
        "decision": decision,
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }
