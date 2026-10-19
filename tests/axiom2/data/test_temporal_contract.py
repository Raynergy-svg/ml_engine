"""Task 3: explicit, immutable, point-in-time observation eligibility."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest


MODULE = "src.axiom2.data.temporal"
ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
T0 = datetime(2024, 6, 3, 14, 0, tzinfo=UTC)
TIME_FIELDS = ("event_time", "published_time", "available_to_axiom_time", "ingested_time")
FIELDS = TIME_FIELDS + ("source", "revision", "schema_version")


def test_temporal_contract_module_exists() -> None:
    assert (ROOT / "src/axiom2/data/temporal.py").is_file(), "Task 3 temporal contract is missing"


@pytest.fixture(scope="module")
def api():
    return importlib.import_module(MODULE)


@pytest.fixture
def values():
    return dict(
        event_time=T0,
        published_time=T0 + timedelta(minutes=1),
        available_to_axiom_time=T0 + timedelta(minutes=2),
        ingested_time=T0 + timedelta(days=30),
        source="historical-bars",
        revision="vendor-revision-1",
        schema_version="ohlcv.v1",
    )


def test_valid_record_is_trainable_without_mutation(api, values) -> None:
    record = api.TemporalRecord(**values)
    before = asdict(record)
    assert api.assert_trainable(record, T0 + timedelta(minutes=3)) is None
    assert asdict(record) == before


@pytest.mark.parametrize("field", FIELDS)
def test_all_record_fields_are_explicit(api, values, field) -> None:
    values.pop(field)
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**values)


@pytest.mark.parametrize("published", [None, T0 + timedelta(minutes=1)])
@pytest.mark.parametrize("revision", [None, "revision-1"])
def test_publication_and_revision_may_be_unavailable(api, values, published, revision) -> None:
    record = api.TemporalRecord(**{**values, "published_time": published, "revision": revision})
    assert api.assert_trainable(record, T0 + timedelta(minutes=2)) is None
    assert record.published_time == published
    assert record.revision == revision


@pytest.mark.parametrize("delta_us", [-1, 0, 1])
def test_availability_cutoff_is_inclusive_and_microsecond_precise(api, values, delta_us) -> None:
    record = api.TemporalRecord(**values)
    cutoff = values["available_to_axiom_time"] + timedelta(microseconds=delta_us)
    if delta_us < 0:
        with pytest.raises(ValueError, match="feature_cutoff"):
            api.assert_trainable(record, cutoff)
    else:
        assert api.assert_trainable(record, cutoff) is None


@pytest.mark.parametrize("ingestion_days", [0, 30])
@pytest.mark.parametrize("published", [None, T0])
def test_unknown_availability_never_falls_back_to_ingestion_or_publication(
    api, values, ingestion_days, published
) -> None:
    record = api.TemporalRecord(
        **{
            **values,
            "published_time": published,
            "available_to_axiom_time": None,
            "ingested_time": T0 + timedelta(days=ingestion_days),
        }
    )
    with pytest.raises(ValueError, match="available_to_axiom_time"):
        api.assert_trainable(record, T0 + timedelta(days=90))
    assert record.available_to_axiom_time is None


def test_late_archival_ingestion_is_not_the_historical_cutoff(api, values) -> None:
    record = api.TemporalRecord(**values)
    cutoff = values["available_to_axiom_time"]
    assert record.ingested_time > cutoff
    assert api.assert_trainable(record, cutoff) is None


@pytest.mark.parametrize("offset_minutes", [-720, -300, -210, 0, 345, 840])
def test_all_timestamps_and_cutoff_normalize_to_utc(api, values, offset_minutes) -> None:
    zone = timezone(timedelta(minutes=offset_minutes))
    converted = {**values, **{field: values[field].astimezone(zone) for field in TIME_FIELDS}}
    record = api.TemporalRecord(**converted)
    for field in TIME_FIELDS:
        assert getattr(record, field) == values[field]
        assert getattr(record, field).tzinfo is UTC
    cutoff = values["available_to_axiom_time"].astimezone(zone)
    assert api.assert_trainable(record, cutoff) is None


@pytest.mark.parametrize("field", TIME_FIELDS)
@pytest.mark.parametrize("bad", ["2024-06-03T14:00:00Z", 0, True, date(2024, 6, 3)])
def test_timestamps_are_not_implicitly_parsed_or_coerced(api, values, field, bad) -> None:
    values[field] = bad
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**values)


@pytest.mark.parametrize("field", TIME_FIELDS)
def test_naive_timestamps_are_rejected_not_assumed_utc(api, values, field) -> None:
    values[field] = values[field].replace(tzinfo=None)
    with pytest.raises(ValueError, match=field):
        api.TemporalRecord(**values)


@pytest.mark.parametrize("field", ["event_time", "ingested_time"])
def test_event_and_ingestion_are_required_timestamps(api, values, field) -> None:
    values[field] = None
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**values)


@pytest.mark.parametrize("cutoff", [None, True, "2024-06-03", date(2024, 6, 3), T0.replace(tzinfo=None)])
def test_cutoff_requires_an_explicit_aware_datetime(api, values, cutoff) -> None:
    with pytest.raises((TypeError, ValueError), match="feature_cutoff"):
        api.assert_trainable(api.TemporalRecord(**values), cutoff)


@pytest.mark.parametrize(
    "overrides",
    [
        {"published_time": T0 - timedelta(microseconds=1)},
        {"published_time": None, "available_to_axiom_time": T0 - timedelta(microseconds=1)},
        {"published_time": None, "available_to_axiom_time": None, "ingested_time": T0 - timedelta(microseconds=1)},
        {"available_to_axiom_time": T0},
        {"available_to_axiom_time": None, "ingested_time": T0},
        {"ingested_time": T0 + timedelta(minutes=1)},
    ],
)
def test_inconsistent_observation_chronology_is_rejected(api, values, overrides) -> None:
    with pytest.raises(ValueError, match="must not be before"):
        api.TemporalRecord(**{**values, **overrides})


def test_simultaneous_observation_release_availability_and_ingestion_is_valid(api, values) -> None:
    record = api.TemporalRecord(**{**values, **dict.fromkeys(TIME_FIELDS, T0)})
    assert api.assert_trainable(record, T0) is None


@pytest.mark.parametrize("field", ["source", "revision", "schema_version"])
@pytest.mark.parametrize("bad", ["", " \t", " padded ", 42, []])
def test_metadata_rejects_empty_ambiguous_and_nonstring_values(api, values, field, bad) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        api.TemporalRecord(**{**values, field: bad})


@pytest.mark.parametrize("field", ["source", "schema_version"])
def test_source_and_schema_cannot_be_unknown(api, values, field) -> None:
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**{**values, field: None})


def test_revised_record_uses_its_own_availability_not_original_event(api, values) -> None:
    original = api.TemporalRecord(**values)
    revised = replace(
        original,
        revision="vendor-revision-2",
        published_time=T0 + timedelta(days=2),
        available_to_axiom_time=T0 + timedelta(days=3),
    )
    assert original.event_time == revised.event_time
    assert original.revision != revised.revision
    assert api.assert_trainable(original, T0 + timedelta(days=1)) is None
    with pytest.raises(ValueError, match="feature_cutoff"):
        api.assert_trainable(revised, T0 + timedelta(days=1))
    assert api.assert_trainable(revised, T0 + timedelta(days=3)) is None


def test_revision_cannot_reuse_old_availability_before_its_publication(api, values) -> None:
    original = api.TemporalRecord(**values)
    with pytest.raises(ValueError, match="must not be before"):
        replace(original, revision="vendor-revision-2", published_time=T0 + timedelta(days=2))


def test_revision_with_unknown_availability_is_not_trainable(api, values) -> None:
    revised = api.TemporalRecord(**{**values, "revision": "vendor-revision-2", "available_to_axiom_time": None})
    with pytest.raises(ValueError, match="available_to_axiom_time"):
        api.assert_trainable(revised, T0 + timedelta(days=60))


@pytest.mark.parametrize("field", FIELDS)
def test_records_are_frozen_and_slotted(api, values, field) -> None:
    record = api.TemporalRecord(**values)
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, None)
    assert not hasattr(record, "__dict__")


def test_imports_load_only_checkout_temporal_module_without_runtime_dependencies() -> None:
    script = """
import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
module = importlib.import_module('src.axiom2.data.temporal')
assert Path(module.__file__).resolve() == Path(sys.argv[1])/'src/axiom2/data/temporal.py'
for prefix in ('src.brokers','src.scanner','src.evidence','requests','pandas','numpy','tensorflow','torch'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(ROOT)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class ProviderObject:
    def __repr__(self):
        raise AssertionError("provider representation must not be invoked")

    def __str__(self):
        raise AssertionError("provider coercion must not be invoked")


@pytest.mark.parametrize("field", FIELDS)
def test_provider_objects_cannot_enter_any_record_field(api, values, field) -> None:
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**{**values, field: ProviderObject()})


@pytest.mark.parametrize("field", FIELDS)
def test_admission_revalidates_reflectively_tampered_fields(api, values, field) -> None:
    record = api.TemporalRecord(**values)
    object.__setattr__(record, field, ProviderObject())
    with pytest.raises(TypeError, match=field):
        api.assert_trainable(record, T0 + timedelta(days=60))


def test_arbitrary_provider_object_is_not_a_temporal_record(api) -> None:
    with pytest.raises(TypeError, match="TemporalRecord"):
        api.assert_trainable(ProviderObject(), T0)


def test_record_subclasses_cannot_change_validation_semantics(api, values) -> None:
    class ExtendedRecord(api.TemporalRecord):
        pass

    with pytest.raises(TypeError, match="TemporalRecord"):
        ExtendedRecord(**values)


@pytest.mark.parametrize("field", TIME_FIELDS)
def test_datetime_subclasses_are_rejected_before_their_hooks(api, values, field) -> None:
    class ProviderDateTime(datetime):
        def astimezone(self, *args):
            raise AssertionError("provider hook must not run")

    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**{**values, field: ProviderDateTime(2024, 6, 3, tzinfo=UTC)})


@pytest.mark.parametrize("field", ["source", "revision", "schema_version"])
def test_metadata_subclasses_are_rejected_before_their_hooks(api, values, field) -> None:
    class ProviderText(str):
        def strip(self, *args):
            raise AssertionError("provider hook must not run")

    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**{**values, field: ProviderText("value")})


@pytest.mark.parametrize("field", TIME_FIELDS)
def test_custom_timezone_hooks_are_not_executed(api, values, field) -> None:
    from datetime import tzinfo

    class ProviderZone(tzinfo):
        def utcoffset(self, dt):
            raise AssertionError("provider timezone hook must not run")

    bad = T0.replace(tzinfo=ProviderZone())
    with pytest.raises(TypeError, match=field):
        api.TemporalRecord(**{**values, field: bad})


@pytest.mark.parametrize("field", TIME_FIELDS + ("feature_cutoff",))
def test_nonexistent_spring_clock_time_is_rejected(api, values, field) -> None:
    from zoneinfo import ZoneInfo

    gap = datetime(2024, 3, 10, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    instant = datetime(2024, 3, 10, 7, 30, tzinfo=UTC)
    consistent = {**values, **dict.fromkeys(TIME_FIELDS, instant)}
    with pytest.raises(ValueError, match=f"{field} is a nonexistent local time"):
        if field == "feature_cutoff":
            api.assert_trainable(api.TemporalRecord(**consistent), gap)
        else:
            api.TemporalRecord(**{**consistent, field: gap})


def test_fall_clock_fold_is_compared_as_an_instant_not_a_wall_clock(api, values) -> None:
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    first = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=0)
    second = first.replace(fold=1)
    record = api.TemporalRecord(
        **{
            **values,
            "event_time": first,
            "published_time": first,
            "available_to_axiom_time": second,
            "ingested_time": second,
        }
    )
    assert record.available_to_axiom_time - record.event_time == timedelta(hours=1)
    with pytest.raises(ValueError, match="feature_cutoff"):
        api.assert_trainable(record, first.replace(minute=30))
    assert api.assert_trainable(record, second) is None


def test_clock_gap_is_rejected_even_without_any_chronology_conflict(api, values) -> None:
    from zoneinfo import ZoneInfo

    gap = datetime(2024, 3, 10, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    instant = datetime(2024, 3, 10, 7, 30, tzinfo=UTC)
    record = api.TemporalRecord(**{**values, **dict.fromkeys(TIME_FIELDS, instant)})
    with pytest.raises(ValueError, match="nonexistent local time"):
        api.assert_trainable(record, gap)


@pytest.mark.parametrize(
    "cutoff",
    [
        datetime.min.replace(tzinfo=timezone(timedelta(hours=14))),
        datetime.max.replace(tzinfo=timezone(timedelta(hours=-14))),
    ],
)
def test_utc_conversion_overflow_fails_closed(api, values, cutoff) -> None:
    with pytest.raises(ValueError, match="feature_cutoff cannot be represented in UTC"):
        api.assert_trainable(api.TemporalRecord(**values), cutoff)


def test_admission_rechecks_late_availability_after_reflective_tampering(api, values) -> None:
    record = api.TemporalRecord(**values)
    object.__setattr__(record, "available_to_axiom_time", T0 + timedelta(days=10))
    with pytest.raises(ValueError, match="feature_cutoff"):
        api.assert_trainable(record, T0 + timedelta(days=1))


def test_admission_rechecks_chronology_after_reflective_tampering(api, values) -> None:
    record = api.TemporalRecord(**values)
    object.__setattr__(record, "event_time", T0 + timedelta(days=1))
    with pytest.raises(ValueError, match="must not be before"):
        api.assert_trainable(record, T0 + timedelta(days=60))


def test_no_implicit_ingestion_override_field_is_accepted(api, values) -> None:
    with pytest.raises(TypeError, match="assume_available_at_ingestion"):
        api.TemporalRecord(**values, assume_available_at_ingestion=True)
