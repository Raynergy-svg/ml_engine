"""Historical universe membership from explicit, bounded evidence manifests.

This is a membership selector, not a market-data or broker lookup. Upstream
builders must prove coverage/completeness (including delisted securities), source
availability and stable instrument identity. A manifest reference is not proof.
Liquidity, corporate-action and execution eligibility remain separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .temporal import _as_utc


_BASIS_ASSET_CLASS = {"sp500": "equity", "curated_etf": "etf"}


@dataclass(frozen=True, slots=True, kw_only=True)
class UniverseMembership:
    """One historical [member_from, member_until) interval for a stable ID.

    Removal/delisting closes an interval; it never erases earlier membership.
    Instrument IDs are provider-neutral security identities, not present-day
    tickers. Ticker reuse/mapping must be resolved upstream. None means an
    UNKNOWN bound, never infinity; such records cannot enter a universe build.
    Still-active membership must be explicitly bounded by verified coverage.
    """

    instrument_id: str
    asset_class: str
    member_from: datetime | None
    member_until: datetime | None

    def __post_init__(self) -> None:
        start, end = _membership_times(self)
        object.__setattr__(self, "member_from", start)
        object.__setattr__(self, "member_until", end)


@dataclass(frozen=True, slots=True, kw_only=True)
class UniverseManifest:
    """Explicit historical membership over [coverage_start, coverage_end).

    Source/evidence/schema IDs reference the source dataset, not its verification.
    Empty memberships is an explicit empty universe, not permission to fall back.
    All bounds are aware instants, normalized to UTC. Queries outside coverage
    fail, even if a supplied membership interval happens to extend further.

    Basis is sp500 (equity rows) or curated_etf (ETF rows). An sp500 manifest may
    attach exactly one curated_etf manifest as etf_exceptions. ETF coverage and
    intervals are checked independently; exceptions cannot themselves nest.
    Input order is immaterial. Disjoint re-entry is supported; overlap is invalid.
    """

    universe_id: str
    source: str
    evidence_id: str
    schema_version: str
    membership_basis: str
    coverage_start: datetime
    coverage_end: datetime
    memberships: tuple[UniverseMembership, ...]
    etf_exceptions: UniverseManifest | None = None

    def __post_init__(self) -> None:
        start, end = _validate_manifest(self)
        object.__setattr__(self, "coverage_start", start)
        object.__setattr__(self, "coverage_end", end)


def _require_text(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a plain string")
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be nonempty without surrounding whitespace")


def _membership_times(member: UniverseMembership) -> tuple[datetime | None, datetime | None]:
    if type(member) is not UniverseMembership:
        raise TypeError("memberships entries must be UniverseMembership records")
    _require_text(member.instrument_id, "instrument_id")
    _require_text(member.asset_class, "asset_class")
    if member.asset_class not in ("equity", "etf"):
        raise ValueError("asset_class must be equity or etf")
    start = None if member.member_from is None else _as_utc(member.member_from, "member_from")
    end = None if member.member_until is None else _as_utc(member.member_until, "member_until")
    if start is not None and end is not None and end <= start:
        raise ValueError("member_until must be after member_from")
    return start, end


def _validate_manifest(
    manifest: UniverseManifest, *, require_complete: bool = False, nested: bool = False
) -> tuple[datetime, datetime]:
    if type(manifest) is not UniverseManifest:
        raise TypeError("manifest must be a UniverseManifest record")
    for field in ("universe_id", "source", "evidence_id", "schema_version", "membership_basis"):
        _require_text(getattr(manifest, field), field)
    if manifest.membership_basis not in _BASIS_ASSET_CLASS:
        raise ValueError("membership_basis must be sp500 or curated_etf")
    if nested and manifest.membership_basis != "curated_etf":
        raise ValueError("etf_exceptions must use a curated_etf manifest")
    start = _as_utc(manifest.coverage_start, "coverage_start")
    end = _as_utc(manifest.coverage_end, "coverage_end")
    if end <= start:
        raise ValueError("coverage_end must be after coverage_start")
    if type(manifest.memberships) is not tuple:
        raise TypeError("memberships must be an immutable tuple")

    intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    for member in manifest.memberships:
        first, last = _membership_times(member)
        if member.asset_class != _BASIS_ASSET_CLASS[manifest.membership_basis]:
            raise ValueError("asset_class disagrees with the manifest membership_basis")
        if first is None or last is None:
            if require_complete:
                raise ValueError("membership interval has an unknown bound")
            continue
        intervals.setdefault(member.instrument_id, []).append((first, last))
    for periods in intervals.values():
        ordered = sorted(periods)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] < previous[1]:
                raise ValueError("memberships contain overlapping intervals for an instrument_id")

    extra = manifest.etf_exceptions
    if extra is not None:
        if nested or manifest.membership_basis != "sp500":
            raise ValueError("ETF exception manifests must not nest")
        if type(extra) is not UniverseManifest:
            raise TypeError("etf_exceptions must be a UniverseManifest")
        _validate_manifest(extra, require_complete=require_complete, nested=True)
        own_ids = {member.instrument_id for member in manifest.memberships}
        extra_ids = {member.instrument_id for member in extra.memberships}
        if own_ids & extra_ids:
            raise ValueError("instrument_id has conflicting equity and ETF classifications")
    return start, end


def build_universe_as_of(timestamp: datetime, manifest: UniverseManifest) -> tuple[str, ...]:
    """Return sorted historical instrument IDs, or fail closed on incomplete data.

    Validate the entire supplied manifest, including unselected rows and ETF
    exceptions, before returning anything. No current-constituent fallback,
    broker access, clock lookup, or dropped unknown intervals is permitted.
    Revalidation detects ordinary/reflected malformed state; frozen records are
    not a security sandbox. Passing this selector grants no trading authority.
    """
    instant = _as_utc(timestamp, "timestamp")
    _validate_manifest(manifest, require_complete=True)
    parts = (manifest,) if manifest.etf_exceptions is None else (manifest, manifest.etf_exceptions)
    selected: set[str] = set()
    for part in parts:
        start = _as_utc(part.coverage_start, "coverage_start")
        end = _as_utc(part.coverage_end, "coverage_end")
        if not start <= instant < end:
            raise ValueError("timestamp is outside manifest coverage")
        for member in part.memberships:
            first, last = _membership_times(member)
            if first <= instant < last:
                selected.add(member.instrument_id)
    return tuple(sorted(selected))
