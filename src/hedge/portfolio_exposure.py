"""AXIOM Hedge Layer — Phase 2: portfolio exposure engine (SHADOW / ANALYSIS-ONLY).

Nets exposure across all (shadow) open positions plus an optional
hypothetical candidate, so the book can be described as EXPOSURE BUCKETS
instead of a list of individual trades — the read the 2026-xx-xx FX incident
needed: "Long USD_CAD + USD_CHF + USD_JPY looked like 3 trades but was one
USD macro bet."

No execution path. This module never places/cancels an order, never touches
``.claude/state.json``, never arms a live gate, never changes leverage. It
only consumes ``ExposurePosition`` records the CALLER constructs from
whatever shadow ledger/journal it already reads (crypto_carry, track_b,
equity harvester, OANDA open trades, ...) — decoupled from any one lane's
schema on purpose, so this module has zero import-time coupling to
execution.py / any broker client.

Bucket concept, reused not forked:
* FX correlation bucket == the currency itself, exactly as
  ``src.equity.trend_risk_gates`` already defines it (``currency_legs``).
  This module does not invent a second FX bucket taxonomy.
* Equity correlation bucket == a sector cluster from
  ``config/correlation_bucket_map.json``, since a single sector isn't always
  a sufficient concentration key (multiple sectors can share one macro
  driver — rates, commodities, growth).

Concentration-warning threshold/instrument-count constants are the exact
values reused from ``trend_risk_gates.DEFAULT_BIAS_SHARE_THRESHOLD`` /
``DEFAULT_BIAS_MIN_INSTRUMENTS`` (0.90 share, >=2 distinct instruments) —
same reasoning: a single-instrument bucket isn't a "disguised" concentration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.equity.trend_risk_gates import (
    DEFAULT_BIAS_MIN_INSTRUMENTS,
    DEFAULT_BIAS_SHARE_THRESHOLD,
)
from src.hedge.exposure_tags import (
    CORRELATION_BUCKET_MAP_PATH,
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
    tag_equity_exposure,
    tag_fx_exposure,
)

logger = logging.getLogger(__name__)

VALID_ASSET_CLASSES = ("fx", "equity")


@dataclass(frozen=True)
class ExposurePosition:
    """One shadow-lane open position OR a hypothetical candidate to net
    against the book. ``risk_home`` is a non-negative dollar-risk magnitude;
    sign comes from ``direction``. Decoupled from any lane's native schema —
    callers translate their own position/ledger records into this shape."""
    asset_class: str      # "fx" | "equity"
    instrument: str       # e.g. "USD_CAD" or "AAPL"
    direction: str        # "long" | "short"
    risk_home: float
    source: str = "shadow"  # provenance tag, e.g. "shadow_open" | "candidate"


@dataclass(frozen=True)
class NetExposureResult:
    net: Dict[str, float]
    bucket_instruments: Dict[str, Tuple[str, ...]]
    resolved_risk_total: float
    fail_reasons: Tuple[str, ...] = ()
    skipped_positions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BetaExposureResult:
    net_beta_home: Optional[float]
    resolved_risk_total: float
    fail_reasons: Tuple[str, ...] = ()
    skipped_positions: Tuple[str, ...] = ()


def _validate_asset_classes(positions: Sequence[ExposurePosition]) -> Tuple[List[str], List[str]]:
    fail_reasons, skipped = [], []
    for p in positions:
        if p.asset_class not in VALID_ASSET_CLASSES:
            fail_reasons.append(f"unknown_asset_class:{p.instrument}:{p.asset_class}")
            skipped.append(p.instrument)
    return fail_reasons, skipped


def net_currency_exposure(positions: Sequence[ExposurePosition],
                          currency_bucket_map: Optional[Dict[str, dict]] = None
                          ) -> NetExposureResult:
    """Rule extension of trend_risk_gates' correlation-bucket cap: instead of
    a pass/fail cap check, return the full SIGNED net per currency across
    every FX position in ``positions`` (open book + optional candidate).
    Non-FX positions are out of scope here and silently skipped (not a
    failure); an FX position with an unresolvable instrument or an
    unclassified currency IS a failure — recorded, never silently zeroed."""
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    net: Dict[str, float] = {}
    bucket_insts: Dict[str, set] = {}
    fail_reasons: List[str] = []
    skipped: List[str] = []
    resolved_total = 0.0
    for p in positions:
        if p.asset_class != "fx":
            continue
        tag = tag_fx_exposure(p.instrument, p.direction, p.risk_home, currency_bucket_map)
        if tag.fail_closed:
            fail_reasons.extend(tag.reasons)
            skipped.append(p.instrument)
            continue
        resolved_total += tag.risk_home
        for leg in tag.legs:
            net[leg.currency] = net.get(leg.currency, 0.0) + leg.signed_home
            bucket_insts.setdefault(leg.currency, set()).add(p.instrument)
    return NetExposureResult(
        net=net, bucket_instruments={k: tuple(sorted(v)) for k, v in bucket_insts.items()},
        resolved_risk_total=resolved_total, fail_reasons=tuple(fail_reasons),
        skipped_positions=tuple(skipped))


def net_sector_exposure(positions: Sequence[ExposurePosition],
                        sector_bucket_map: Optional[Dict[str, dict]] = None
                        ) -> NetExposureResult:
    """Signed net dollar-risk per equity sector. Non-equity positions
    silently skipped (out of scope); an unknown ticker fails closed."""
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    net: Dict[str, float] = {}
    bucket_insts: Dict[str, set] = {}
    fail_reasons: List[str] = []
    skipped: List[str] = []
    resolved_total = 0.0
    for p in positions:
        if p.asset_class != "equity":
            continue
        tag = tag_equity_exposure(p.instrument, p.direction, p.risk_home, sector_bucket_map)
        if tag.fail_closed or tag.sector is None:
            fail_reasons.extend(tag.reasons or (f"unresolvable_sector:{p.instrument}",))
            skipped.append(p.instrument)
            continue
        resolved_total += tag.risk_home
        net[tag.sector] = net.get(tag.sector, 0.0) + tag.signed_home
        bucket_insts.setdefault(tag.sector, set()).add(p.instrument)
    return NetExposureResult(
        net=net, bucket_instruments={k: tuple(sorted(v)) for k, v in bucket_insts.items()},
        resolved_risk_total=resolved_total, fail_reasons=tuple(fail_reasons),
        skipped_positions=tuple(skipped))


def net_beta_exposure(positions: Sequence[ExposurePosition],
                      sector_bucket_map: Optional[Dict[str, dict]] = None
                      ) -> BetaExposureResult:
    """Beta-weighted signed net dollar exposure across resolvable equity
    positions (``sum(signed_home * market_beta)``). ``None`` only when the
    equity subset is non-empty but every member fails to resolve; an empty
    (no-equity) book correctly reports ``0.0``, not a failure."""
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    equity_positions = [p for p in positions if p.asset_class == "equity"]
    if not equity_positions:
        return BetaExposureResult(net_beta_home=0.0, resolved_risk_total=0.0)

    total_beta_home = 0.0
    resolved_total = 0.0
    fail_reasons: List[str] = []
    skipped: List[str] = []
    any_resolved = False
    for p in equity_positions:
        tag = tag_equity_exposure(p.instrument, p.direction, p.risk_home, sector_bucket_map)
        if tag.fail_closed or tag.signed_beta_home is None:
            fail_reasons.extend(tag.reasons or (f"missing_beta:{p.instrument}",))
            skipped.append(p.instrument)
            continue
        total_beta_home += tag.signed_beta_home
        resolved_total += tag.risk_home
        any_resolved = True
    net_beta_home = total_beta_home if any_resolved else None
    return BetaExposureResult(net_beta_home=net_beta_home, resolved_risk_total=resolved_total,
                              fail_reasons=tuple(fail_reasons), skipped_positions=tuple(skipped))


def net_correlation_bucket_exposure(positions: Sequence[ExposurePosition],
                                    sector_bucket_map: Optional[Dict[str, dict]] = None,
                                    correlation_bucket_map: Optional[Dict[str, dict]] = None,
                                    currency_bucket_map: Optional[Dict[str, dict]] = None
                                    ) -> NetExposureResult:
    """Unified cross-asset correlation-bucket net. FX buckets ARE currencies
    (namespaced ``fx_currency:<CCY>`` — same definition as
    ``net_currency_exposure``, just prefixed so it can share a dict with
    equity clusters without key collisions); equity buckets are sector
    clusters from ``correlation_bucket_map.json`` (namespaced
    ``equity_cluster:<cluster>``)."""
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    if correlation_bucket_map is None:
        correlation_bucket_map = load_bucket_map(CORRELATION_BUCKET_MAP_PATH)
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    net: Dict[str, float] = {}
    bucket_insts: Dict[str, set] = {}
    fail_reasons: List[str] = []
    skipped: List[str] = []
    resolved_total = 0.0

    for p in positions:
        if p.asset_class == "fx":
            tag = tag_fx_exposure(p.instrument, p.direction, p.risk_home, currency_bucket_map)
            if tag.fail_closed:
                fail_reasons.extend(tag.reasons)
                skipped.append(p.instrument)
                continue
            resolved_total += tag.risk_home
            for leg in tag.legs:
                key = f"fx_currency:{leg.currency}"
                net[key] = net.get(key, 0.0) + leg.signed_home
                bucket_insts.setdefault(key, set()).add(p.instrument)
        elif p.asset_class == "equity":
            etag = tag_equity_exposure(p.instrument, p.direction, p.risk_home, sector_bucket_map)
            if etag.fail_closed or etag.sector is None:
                fail_reasons.extend(etag.reasons or (f"unresolvable_sector:{p.instrument}",))
                skipped.append(p.instrument)
                continue
            cluster = correlation_bucket_map.get(etag.sector)
            if cluster is None:
                fail_reasons.append(f"unmapped_correlation_bucket:{etag.sector}")
                skipped.append(p.instrument)
                continue
            resolved_total += etag.risk_home
            key = f"equity_cluster:{cluster}"
            net[key] = net.get(key, 0.0) + etag.signed_home
            bucket_insts.setdefault(key, set()).add(p.instrument)
        else:
            fail_reasons.append(f"unknown_asset_class:{p.instrument}:{p.asset_class}")
            skipped.append(p.instrument)

    return NetExposureResult(
        net=net, bucket_instruments={k: tuple(sorted(v)) for k, v in bucket_insts.items()},
        resolved_risk_total=resolved_total, fail_reasons=tuple(fail_reasons),
        skipped_positions=tuple(skipped))


def _concentration_warnings(label: str, result: NetExposureResult, *,
                            share_threshold: float = DEFAULT_BIAS_SHARE_THRESHOLD,
                            min_instruments: int = DEFAULT_BIAS_MIN_INSTRUMENTS) -> List[str]:
    """Mirrors trend_risk_gates.bias_detector's share-of-total-risk heuristic
    (same threshold/min-instrument constants, reused directly), adapted for
    this module's cross-asset ``risk_home``-only position shape rather than
    bias_detector's OANDA-trade-dict + per-trade-ATR shape."""
    if result.resolved_risk_total <= 0:
        return []
    warnings = []
    for key, net_value in result.net.items():
        insts = result.bucket_instruments.get(key, ())
        if len(insts) < min_instruments:
            continue
        share = abs(net_value) / result.resolved_risk_total
        if share >= share_threshold:
            direction = "long" if net_value > 0 else "short"
            warnings.append(
                f"{label} concentration: {key} net {direction} {abs(net_value):.2f} "
                f"({share:.0%} of resolved risk) across {len(insts)} instruments "
                f"{list(insts)} — one concentrated bet wearing multiple tickets, "
                f"not {len(insts)} independent positions")
    return warnings


@dataclass(frozen=True)
class ExposureReport:
    net_currency_exposure: Dict[str, float]
    net_sector_exposure: Dict[str, float]
    net_beta_exposure: Optional[float]
    net_correlation_bucket_exposure: Dict[str, float]
    concentration_warnings: Tuple[str, ...]
    fail_closed: bool
    fail_reasons: Tuple[str, ...]
    skipped_positions: Tuple[str, ...]
    position_count: int
    resolved_position_count: int

    def narrative(self) -> Tuple[str, ...]:
        lines: List[str] = []
        if self.position_count == 0:
            return ("empty book — no open positions or candidate, zero exposure.",)
        lines.append(
            f"{self.position_count} position(s) net to "
            f"{len(self.net_correlation_bucket_exposure)} correlation bucket(s).")
        if self.concentration_warnings:
            lines.extend(self.concentration_warnings)
        if self.fail_closed:
            lines.append(
                f"FAIL-CLOSED: {len(self.skipped_positions)} position(s) could not be "
                f"resolved ({list(self.fail_reasons)}) — report is PARTIAL, do not treat "
                f"missing legs as zero exposure.")
        return tuple(lines)

    def to_dict(self) -> dict:
        return {
            "net_currency_exposure": dict(self.net_currency_exposure),
            "net_sector_exposure": dict(self.net_sector_exposure),
            "net_beta_exposure": self.net_beta_exposure,
            "net_correlation_bucket_exposure": dict(self.net_correlation_bucket_exposure),
            "concentration_warnings": list(self.concentration_warnings),
            "fail_closed": self.fail_closed,
            "fail_reasons": list(self.fail_reasons),
            "skipped_positions": list(self.skipped_positions),
            "position_count": self.position_count,
            "resolved_position_count": self.resolved_position_count,
            "narrative": list(self.narrative()),
        }


def build_exposure_report(open_positions: Sequence[ExposurePosition],
                          candidate: Optional[ExposurePosition] = None, *,
                          currency_bucket_map: Optional[Dict[str, dict]] = None,
                          sector_bucket_map: Optional[Dict[str, dict]] = None,
                          correlation_bucket_map: Optional[Dict[str, dict]] = None
                          ) -> ExposureReport:
    """Net a (shadow) open book plus an optional hypothetical candidate into
    the four exposure dimensions, and produce an honest, fail-closed report.
    Loads config maps once (real disk read) if not supplied, so callers
    running many candidates through the same book don't re-read the JSON
    files per call."""
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    if correlation_bucket_map is None:
        correlation_bucket_map = load_bucket_map(CORRELATION_BUCKET_MAP_PATH)

    all_positions = list(open_positions) + ([candidate] if candidate is not None else [])

    invalid_reasons, invalid_skipped = _validate_asset_classes(all_positions)

    currency_result = net_currency_exposure(all_positions, currency_bucket_map)
    sector_result = net_sector_exposure(all_positions, sector_bucket_map)
    beta_result = net_beta_exposure(all_positions, sector_bucket_map)
    corr_result = net_correlation_bucket_exposure(
        all_positions, sector_bucket_map, correlation_bucket_map, currency_bucket_map)

    fail_reasons: List[str] = []
    skipped: List[str] = []
    for reasons, skips in (
        (invalid_reasons, invalid_skipped),
        (currency_result.fail_reasons, currency_result.skipped_positions),
        (sector_result.fail_reasons, sector_result.skipped_positions),
        (beta_result.fail_reasons, beta_result.skipped_positions),
        (corr_result.fail_reasons, corr_result.skipped_positions),
    ):
        for r in reasons:
            if r not in fail_reasons:
                fail_reasons.append(r)
        for s in skips:
            if s not in skipped:
                skipped.append(s)

    concentration_warnings: List[str] = []
    concentration_warnings.extend(_concentration_warnings("currency", currency_result))
    concentration_warnings.extend(_concentration_warnings("sector", sector_result))
    concentration_warnings.extend(_concentration_warnings("correlation_bucket", corr_result))

    return ExposureReport(
        net_currency_exposure=currency_result.net,
        net_sector_exposure=sector_result.net,
        net_beta_exposure=beta_result.net_beta_home,
        net_correlation_bucket_exposure=corr_result.net,
        concentration_warnings=tuple(concentration_warnings),
        fail_closed=bool(fail_reasons),
        fail_reasons=tuple(fail_reasons),
        skipped_positions=tuple(skipped),
        position_count=len(all_positions),
        resolved_position_count=len(all_positions) - len(skipped),
    )
