"""Conservative temporal eligibility for completed research observations.

Timestamps describe this exact source revision, not the latest mutable view.
This contract checks chronology, not authenticity: upstream evidence must prove
availability. Neither ingestion time nor a revision label supplies that proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


_TIME_FIELDS = ("event_time", "published_time", "available_to_axiom_time", "ingested_time")
_OPTIONAL_TIMES = frozenset({"published_time", "available_to_axiom_time"})
_CHRONOLOGY = (
    ("event_time", "published_time"),
    ("event_time", "available_to_axiom_time"),
    ("event_time", "ingested_time"),
    ("published_time", "available_to_axiom_time"),
    ("published_time", "ingested_time"),
    ("available_to_axiom_time", "ingested_time"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalRecord:
    """Immutable metadata for one completed observation and source revision.

    All fields are explicit. ``None`` means unknown/not applicable for publication
    or revision; unknown availability is representable but NEVER trainable.
    Known times follow event -> publication -> availability -> ingestion,
    omitting unknown optional times. Archival ingestion may be after the cutoff.

    ``schema_version`` names the source data schema, as in ``DataDependency``.
    Source adapters provide plain aware datetimes using fixed offsets or stdlib
    ZoneInfo; records store UTC. A completed bar's event_time is its completion,
    not its opening. Scheduled future events are outside this Phase-1 contract.
    Revision history and proof resolution are external responsibilities; this
    class does not detect dishonest timestamps.
    """

    event_time: datetime
    published_time: datetime | None
    available_to_axiom_time: datetime | None
    ingested_time: datetime
    source: str
    revision: str | None
    schema_version: str

    def __post_init__(self) -> None:
        for field, value in _validated_times(self).items():
            object.__setattr__(self, field, value)


def _require_text(value: object, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a plain string")
    if not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be nonempty without surrounding whitespace")


def _as_utc(value: object, field: str) -> datetime:
    # Exact types avoid provider-defined datetime/tzinfo hooks at admission.
    if type(value) is not datetime:
        raise TypeError(f"{field} must be a plain datetime")
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    if type(value.tzinfo) not in (timezone, ZoneInfo):
        raise TypeError(f"{field} must use a stdlib timezone or ZoneInfo")
    try:
        normalized = value.astimezone(timezone.utc)
        roundtrip = normalized.astimezone(value.tzinfo)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} cannot be represented in UTC") from None
    if roundtrip.replace(tzinfo=None) != value.replace(tzinfo=None):
        raise ValueError(f"{field} is a nonexistent local time")
    return normalized


def _validated_times(record: TemporalRecord) -> dict[str, datetime | None]:
    if type(record) is not TemporalRecord:
        raise TypeError("record must be a TemporalRecord")
    for field in ("source", "schema_version"):
        _require_text(getattr(record, field), field)
    if record.revision is not None:
        _require_text(record.revision, "revision")

    times: dict[str, datetime | None] = {}
    for field in _TIME_FIELDS:
        value = getattr(record, field)
        times[field] = None if value is None and field in _OPTIONAL_TIMES else _as_utc(value, field)
    for earlier, later in _CHRONOLOGY:
        first, second = times[earlier], times[later]
        if first is not None and second is not None and second < first:
            raise ValueError(f"{later} must not be before {earlier}")
    return times


def assert_trainable(record: TemporalRecord, feature_cutoff: datetime) -> None:
    """Accept only explicitly known availability at or before the aware cutoff.

    Revalidate the full record on every admission; frozen dataclasses prevent
    ordinary mutation, not hostile Python reflection. This function has no clock,
    I/O, implicit parsing, provider imports, ingestion fallback or side effects.
    Passing this temporal check grants no model, promotion or execution authority.
    """
    times = _validated_times(record)
    cutoff = _as_utc(feature_cutoff, "feature_cutoff")
    available = times["available_to_axiom_time"]
    if available is None:
        raise ValueError("available_to_axiom_time is unknown; record is not trainable")
    if available > cutoff:
        raise ValueError("available_to_axiom_time is after feature_cutoff")
