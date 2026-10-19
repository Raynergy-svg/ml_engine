"""Task 4: explicit historical membership, never today's constituent list."""

from __future__ import annotations

import importlib
import itertools
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MODULE = "src.axiom2.data.universe"
ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
T0 = datetime(2020, 1, 1, tzinfo=UTC)
MEMBER_FIELDS = ("instrument_id", "asset_class", "member_from", "member_until")
MANIFEST_FIELDS = (
    "universe_id",
    "source",
    "evidence_id",
    "schema_version",
    "membership_basis",
    "coverage_start",
    "coverage_end",
    "memberships",
)


def at(day: int) -> datetime:
    return T0 + timedelta(days=day)


def test_universe_module_exists() -> None:
    assert (ROOT / "src/axiom2/data/universe.py").is_file(), "Task 4 universe implementation is missing"


@pytest.fixture(scope="module")
def api():
    return importlib.import_module(MODULE)


@pytest.fixture
def member(api):
    def make(instrument_id="SEC:A", start=0, end=100, asset_class="equity"):
        return api.UniverseMembership(
            instrument_id=instrument_id,
            asset_class=asset_class,
            member_from=None if start is None else at(start),
            member_until=None if end is None else at(end),
        )

    return make


@pytest.fixture
def manifest(api):
    def make(memberships=(), **updates):
        values = dict(
            universe_id="liquid-us.pit.v1",
            source="historical-membership",
            evidence_id="membership-snapshot-001",
            schema_version="membership.v1",
            membership_basis="sp500",
            coverage_start=at(0),
            coverage_end=at(100),
            memberships=memberships,
        )
        return api.UniverseManifest(**{**values, **updates})

    return make


@pytest.mark.parametrize("day,expected", [(0, ("SEC:OLD",)), (9, ("SEC:OLD",)), (10, ("SEC:NEW",)), (99, ("SEC:NEW",))])
def test_historical_membership_start_inclusive_end_exclusive(api, member, manifest, day, expected) -> None:
    history = manifest((member("SEC:OLD", 0, 10), member("SEC:NEW", 10, 100)))
    assert api.build_universe_as_of(at(day), history) == expected


def test_current_constituent_is_never_projected_backwards(api, member, manifest) -> None:
    current = manifest((member("SEC:NEW", 50, 100),))
    assert api.build_universe_as_of(at(49), current) == ()
    assert api.build_universe_as_of(at(50), current) == ("SEC:NEW",)


def test_delisted_security_remains_in_historical_universe(api, member, manifest) -> None:
    history = manifest((member("SEC:DELISTED", 0, 10),))
    assert api.build_universe_as_of(at(5), history) == ("SEC:DELISTED",)
    assert api.build_universe_as_of(at(10), history) == ()


def test_disjoint_reentry_preserves_absence_between_memberships(api, member, manifest) -> None:
    history = manifest((member("SEC:A", 0, 10), member("SEC:A", 20, 100)))
    assert api.build_universe_as_of(at(5), history) == ("SEC:A",)
    assert api.build_universe_as_of(at(15), history) == ()
    assert api.build_universe_as_of(at(20), history) == ("SEC:A",)


def test_adjacent_intervals_are_not_overlapping(api, member, manifest) -> None:
    history = manifest((member("SEC:A", 0, 10), member("SEC:A", 10, 100)))
    assert api.build_universe_as_of(at(10), history) == ("SEC:A",)


@pytest.mark.parametrize("intervals", [((0, 10), (9, 20)), ((0, 100), (1, 2), (3, 4)), ((0, 10), (0, 10))])
def test_overlapping_or_duplicate_intervals_fail_closed(api, member, manifest, intervals) -> None:
    with pytest.raises(ValueError, match="overlap"):
        manifest(tuple(member("SEC:A", start, end) for start, end in intervals))


def test_output_is_sorted_unique_and_independent_of_input_order(api, member, manifest) -> None:
    rows = (member("SEC:C"), member("SEC:A"), member("SEC:B"))
    for ordered in itertools.permutations(rows):
        history = manifest(ordered)
        before = asdict(history)
        assert api.build_universe_as_of(at(5), history) == ("SEC:A", "SEC:B", "SEC:C")
        assert asdict(history) == before


def test_adding_known_future_members_does_not_change_past(api, member, manifest) -> None:
    history = manifest((member("SEC:A"),))
    extended = replace(history, memberships=history.memberships + (member("SEC:FUTURE", 90, 100),))
    assert api.build_universe_as_of(at(5), extended) == api.build_universe_as_of(at(5), history)


@pytest.mark.parametrize("bounds", [(None, 10), (0, None), (None, None), (90, None)])
def test_unknown_bounds_block_the_build_even_on_unselected_rows(api, member, manifest, bounds) -> None:
    history = manifest((member("SEC:KNOWN"), member("SEC:UNKNOWN", *bounds)))
    with pytest.raises(ValueError, match="unknown"):
        api.build_universe_as_of(at(5), history)


@pytest.mark.parametrize("day", [-1, 100, 101])
def test_queries_outside_verified_coverage_fail_instead_of_extrapolating(api, manifest, day) -> None:
    with pytest.raises(ValueError, match="coverage"):
        api.build_universe_as_of(at(day), manifest())


def test_explicit_empty_universe_does_not_fall_back_to_a_current_list(api, manifest) -> None:
    assert api.build_universe_as_of(at(5), manifest()) == ()


@pytest.mark.parametrize("timestamp", [None, True, 0, "2020-01-01", T0.replace(tzinfo=None)])
def test_query_requires_explicit_aware_datetime(api, manifest, timestamp) -> None:
    with pytest.raises((TypeError, ValueError), match="timestamp"):
        api.build_universe_as_of(timestamp, manifest())


def test_etfs_require_their_own_explicit_curated_manifest(api, member, manifest) -> None:
    etf = member("FUND:ONE", 0, 100, "etf")
    with pytest.raises(ValueError, match="asset_class"):
        manifest((etf,))
    explicit = manifest((etf,), membership_basis="curated_etf", universe_id="curated-etfs.v1", evidence_id="etfs-001")
    combined = manifest((member("SEC:A"),), etf_exceptions=explicit)
    assert api.build_universe_as_of(at(5), combined) == ("FUND:ONE", "SEC:A")
    assert api.build_universe_as_of(at(5), explicit) == ("FUND:ONE",)


def test_etfs_are_not_implicitly_added_when_no_exception_manifest_exists(api, member, manifest) -> None:
    assert api.build_universe_as_of(at(5), manifest((member(),))) == ("SEC:A",)


def test_equity_rows_cannot_masquerade_as_etf_exceptions(api, member, manifest) -> None:
    with pytest.raises(ValueError, match="asset_class"):
        manifest((member(),), membership_basis="curated_etf")


def test_equity_manifest_cannot_be_used_as_etf_exception(api, manifest) -> None:
    with pytest.raises(ValueError, match="curated_etf"):
        manifest(etf_exceptions=manifest())


def test_etf_exception_coverage_must_cover_the_requested_instant(api, member, manifest) -> None:
    etfs = manifest(
        (member("FUND:ONE", 10, 20, "etf"),), membership_basis="curated_etf", coverage_start=at(10), coverage_end=at(20)
    )
    combined = manifest((member(),), etf_exceptions=etfs)
    assert api.build_universe_as_of(at(10), combined) == ("FUND:ONE", "SEC:A")
    for day in (9, 20):
        with pytest.raises(ValueError, match="coverage"):
            api.build_universe_as_of(at(day), combined)


def test_etf_unknown_interval_cannot_be_silently_dropped(api, member, manifest) -> None:
    etfs = manifest((member("FUND:UNKNOWN", 0, None, "etf"),), membership_basis="curated_etf")
    with pytest.raises(ValueError, match="unknown"):
        api.build_universe_as_of(at(5), manifest((member(),), etf_exceptions=etfs))


def test_nested_etf_manifests_are_rejected(api, manifest) -> None:
    etfs = manifest(membership_basis="curated_etf")
    with pytest.raises(ValueError, match="nest"):
        manifest(membership_basis="curated_etf", etf_exceptions=etfs)


def test_conflicting_asset_identity_across_manifests_is_not_deduplicated(api, member, manifest) -> None:
    etfs = manifest((member("SEC:A", 0, 100, "etf"),), membership_basis="curated_etf")
    with pytest.raises(ValueError, match="instrument_id"):
        manifest((member(),), etf_exceptions=etfs)


@pytest.mark.parametrize("field", MEMBER_FIELDS)
def test_membership_fields_are_explicit(api, member, field) -> None:
    values = asdict(member())
    values.pop(field)
    with pytest.raises(TypeError, match=field):
        api.UniverseMembership(**values)


@pytest.mark.parametrize("field", MANIFEST_FIELDS)
def test_manifest_fields_are_explicit(api, manifest, field) -> None:
    values = asdict(manifest())
    values.pop(field)
    with pytest.raises(TypeError, match=field):
        api.UniverseManifest(**values)


@pytest.mark.parametrize("field", ["universe_id", "source", "evidence_id", "schema_version"])
@pytest.mark.parametrize("bad", ["", " padded ", None, 42])
def test_manifest_provenance_identifiers_cannot_be_empty_or_coerced(api, manifest, field, bad) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        manifest(**{field: bad})


@pytest.mark.parametrize("basis", ["current", "all_equities", "SP500", "", None, True])
def test_unknown_manifest_basis_is_rejected(api, manifest, basis) -> None:
    with pytest.raises((TypeError, ValueError), match="membership_basis"):
        manifest(membership_basis=basis)


@pytest.mark.parametrize("asset", ["crypto", "options", "fx", "Equity", "", None])
def test_unsupported_asset_classes_are_rejected(api, member, asset) -> None:
    with pytest.raises((TypeError, ValueError), match="asset_class"):
        member(asset_class=asset)


@pytest.mark.parametrize("bad", ["", " \t", " padded ", None, 42])
def test_instrument_identity_is_explicit_plain_text(api, member, bad) -> None:
    with pytest.raises((TypeError, ValueError), match="instrument_id"):
        member(instrument_id=bad)


@pytest.mark.parametrize("bounds", [(10, 10), (11, 10)])
def test_empty_or_reversed_membership_intervals_are_invalid(api, member, bounds) -> None:
    with pytest.raises(ValueError, match="member_until"):
        member(start=bounds[0], end=bounds[1])


@pytest.mark.parametrize("end", [0, -1])
def test_empty_or_reversed_coverage_is_invalid(api, manifest, end) -> None:
    with pytest.raises(ValueError, match="coverage_end"):
        manifest(coverage_end=at(end))


@pytest.mark.parametrize("bad", [[], None, {}, "SEC:A", (object(),)])
def test_manifest_memberships_require_immutable_records(api, manifest, bad) -> None:
    with pytest.raises(TypeError, match="memberships"):
        manifest(bad)


def test_nonempty_mutable_member_list_is_rejected(api, member, manifest) -> None:
    with pytest.raises(TypeError, match="memberships"):
        manifest([member()])


@pytest.mark.parametrize("bad", ["etf-manifest-id", {}, [], True])
def test_etf_exception_requires_a_manifest_not_an_id_or_provider_payload(api, manifest, bad) -> None:
    with pytest.raises(TypeError, match="etf_exceptions"):
        manifest(etf_exceptions=bad)


@pytest.mark.parametrize("offset", [-720, 345, 840])
def test_membership_coverage_and_query_use_utc_instants(api, member, manifest, offset) -> None:
    zone = timezone(timedelta(minutes=offset))
    row = replace(member(), member_from=at(0).astimezone(zone), member_until=at(100).astimezone(zone))
    history = manifest((row,), coverage_start=at(0).astimezone(zone), coverage_end=at(100).astimezone(zone))
    assert row.member_from.tzinfo is UTC and row.member_until.tzinfo is UTC
    assert history.coverage_start.tzinfo is UTC and history.coverage_end.tzinfo is UTC
    assert api.build_universe_as_of(at(5).astimezone(zone), history) == ("SEC:A",)


def test_records_reject_ordinary_mutation(api, member, manifest) -> None:
    row = member()
    history = manifest((row,))
    for item, field, value in [
        (row, "member_until", at(1)),
        (history, "memberships", ()),
        (history, "etf_exceptions", None),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, value)
        assert not hasattr(item, "__dict__")


def test_no_current_member_or_delisting_override_is_accepted(api, member, manifest) -> None:
    for field in ("current_members", "include_current", "exclude_delisted"):
        with pytest.raises(TypeError, match=field):
            manifest((member(),), **{field: True})


def test_universe_import_has_no_broker_provider_or_dataset_side_effects() -> None:
    script = """
import importlib,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
module=importlib.import_module('src.axiom2.data.universe')
assert Path(module.__file__).resolve()==Path(sys.argv[1])/'src/axiom2/data/universe.py'
for prefix in ('src.brokers','src.scanner','src.evidence','requests','pandas','numpy','tensorflow','torch'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script, str(ROOT)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


class ProviderObject:
    def __repr__(self):
        raise AssertionError("provider representation must not execute")

    def __str__(self):
        raise AssertionError("provider coercion must not execute")


@pytest.mark.parametrize("field", MEMBER_FIELDS)
def test_provider_objects_cannot_enter_membership_fields(api, member, field) -> None:
    values = asdict(member())
    with pytest.raises(TypeError, match=field):
        api.UniverseMembership(**{**values, field: ProviderObject()})


@pytest.mark.parametrize("field", MANIFEST_FIELDS + ("etf_exceptions",))
def test_provider_objects_cannot_enter_manifest_fields(api, manifest, field) -> None:
    with pytest.raises(TypeError, match=field):
        manifest(**{field: ProviderObject()})


def test_provider_object_is_not_a_manifest(api) -> None:
    with pytest.raises(TypeError, match="UniverseManifest"):
        api.build_universe_as_of(at(5), ProviderObject())


@pytest.mark.parametrize("field", MEMBER_FIELDS)
def test_membership_is_revalidated_after_reflective_tampering(api, member, manifest, field) -> None:
    row = member()
    history = manifest((row,))
    object.__setattr__(row, field, ProviderObject())
    with pytest.raises(TypeError, match=field):
        api.build_universe_as_of(at(5), history)


@pytest.mark.parametrize("field", MANIFEST_FIELDS + ("etf_exceptions",))
def test_manifest_is_revalidated_after_reflective_tampering(api, member, manifest, field) -> None:
    history = manifest((member(),))
    object.__setattr__(history, field, ProviderObject())
    with pytest.raises(TypeError, match=field):
        api.build_universe_as_of(at(5), history)


@pytest.mark.parametrize("field", ["member_from", "member_until"])
def test_unknown_bounds_after_tampering_never_silently_disappear(api, member, manifest, field) -> None:
    row = member()
    history = manifest((row,))
    object.__setattr__(row, field, None)
    with pytest.raises(ValueError, match="unknown"):
        api.build_universe_as_of(at(5), history)


def test_nested_etf_is_revalidated_after_tampering(api, member, manifest) -> None:
    row = member("FUND:A", 0, 100, "etf")
    etfs = manifest((row,), membership_basis="curated_etf")
    history = manifest((member(),), etf_exceptions=etfs)
    object.__setattr__(row, "member_until", None)
    with pytest.raises(ValueError, match="unknown"):
        api.build_universe_as_of(at(5), history)


def test_overlapping_intervals_after_tampering_fail_before_selection(api, member, manifest) -> None:
    row = member("SEC:A", 20, 30)
    history = manifest((member("SEC:A", 0, 10), row))
    object.__setattr__(row, "member_from", at(5))
    with pytest.raises(ValueError, match="overlap"):
        api.build_universe_as_of(at(99), history)


def test_cyclic_etf_exception_fails_without_recursion(api, manifest) -> None:
    etfs = manifest(membership_basis="curated_etf")
    history = manifest(etf_exceptions=etfs)
    object.__setattr__(etfs, "etf_exceptions", history)
    with pytest.raises(ValueError, match="nest"):
        api.build_universe_as_of(at(5), history)


def test_manifest_subclasses_cannot_change_validation_semantics(api, manifest) -> None:
    class Extended(api.UniverseManifest):
        pass

    with pytest.raises(TypeError, match="UniverseManifest"):
        Extended(**asdict(manifest()))


def test_membership_subclasses_cannot_change_validation_semantics(api, member) -> None:
    class Extended(api.UniverseMembership):
        pass

    with pytest.raises(TypeError, match="UniverseMembership"):
        Extended(**asdict(member()))


@pytest.mark.parametrize("field", ["instrument_id", "asset_class"])
def test_member_string_subclass_hooks_are_not_called(api, member, field) -> None:
    class ProviderText(str):
        def strip(self, *args):
            raise AssertionError("provider hook must not run")

    values = asdict(member())
    values[field] = ProviderText(values[field])
    with pytest.raises(TypeError, match=field):
        api.UniverseMembership(**values)


@pytest.mark.parametrize("field", ["universe_id", "source", "evidence_id", "schema_version", "membership_basis"])
def test_manifest_string_subclass_hooks_are_not_called(api, manifest, field) -> None:
    class ProviderText(str):
        def strip(self, *args):
            raise AssertionError("provider hook must not run")

    values = asdict(manifest())
    values[field] = ProviderText(values[field])
    with pytest.raises(TypeError, match=field):
        api.UniverseManifest(**values)


def test_membership_tuple_subclasses_are_rejected_before_iteration(api, member, manifest) -> None:
    class ProviderTuple(tuple):
        def __iter__(self):
            raise AssertionError("provider iteration must not run")

    with pytest.raises(TypeError, match="memberships"):
        manifest(ProviderTuple((member(),)))


@pytest.mark.parametrize("field", ["member_from", "member_until", "coverage_start", "coverage_end", "timestamp"])
@pytest.mark.parametrize("bad", [None, "2020-01-01", True, T0.replace(tzinfo=None)])
def test_every_time_boundary_is_explicit_and_aware(api, member, manifest, field, bad) -> None:
    if field.startswith("member_") and bad is None:
        # Unknown member bounds are representable but must fail admission.
        row = replace(member(), **{field: None})
        with pytest.raises(ValueError, match="unknown"):
            api.build_universe_as_of(at(5), manifest((row,)))
        return
    with pytest.raises((TypeError, ValueError), match=field):
        if field.startswith("member_"):
            replace(member(), **{field: bad})
        elif field.startswith("coverage_"):
            manifest(**{field: bad})
        else:
            api.build_universe_as_of(bad, manifest())


@pytest.mark.parametrize("field", ["member_from", "member_until", "coverage_start", "coverage_end", "timestamp"])
def test_datetime_subclass_hooks_are_not_executed(api, member, manifest, field) -> None:
    class ProviderDateTime(datetime):
        def astimezone(self, *args):
            raise AssertionError("provider datetime hook must not run")

    bad = ProviderDateTime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(TypeError, match=field):
        if field.startswith("member_"):
            replace(member(), **{field: bad})
        elif field.startswith("coverage_"):
            manifest(**{field: bad})
        else:
            api.build_universe_as_of(bad, manifest())


def test_fall_clock_fold_does_not_backfill_the_second_hour(api, member, manifest) -> None:
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    first = datetime(2020, 11, 1, 1, 15, tzinfo=zone, fold=0)
    second = first.replace(fold=1)
    row = replace(member(), member_from=second, member_until=second + timedelta(hours=2))
    history = manifest(
        (row,), coverage_start=datetime(2020, 11, 1, tzinfo=UTC), coverage_end=datetime(2020, 11, 2, tzinfo=UTC)
    )
    assert api.build_universe_as_of(first, history) == ()
    assert api.build_universe_as_of(second, history) == ("SEC:A",)


def test_nonexistent_clock_time_cannot_be_a_membership_start(api, member) -> None:
    from zoneinfo import ZoneInfo

    gap = datetime(2020, 3, 8, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="member_from is a nonexistent local time"):
        replace(member(), member_from=gap, member_until=datetime(2020, 3, 9, tzinfo=UTC))


def test_microsecond_precision_at_membership_boundary(api, member, manifest) -> None:
    start = at(5)
    end = start + timedelta(microseconds=1)
    row = replace(member(), member_from=start, member_until=end)
    history = manifest((row,))
    assert api.build_universe_as_of(start - timedelta(microseconds=1), history) == ()
    assert api.build_universe_as_of(start, history) == ("SEC:A",)
    assert api.build_universe_as_of(end, history) == ()


def test_declared_coverage_limits_even_members_extending_beyond_it(api, member, manifest) -> None:
    history = manifest((member("SEC:A", -10, 110),))
    assert api.build_universe_as_of(at(0), history) == ("SEC:A",)
    for query in (at(-1), at(100)):
        with pytest.raises(ValueError, match="coverage"):
            api.build_universe_as_of(query, history)


def test_stable_security_ids_do_not_collapse_reused_tickers(api, member, manifest) -> None:
    # Symbols may be reused; explicit instrument identities stay distinct.
    history = manifest((member("ISSUER:OLD-ABC", 0, 10), member("ISSUER:NEW-ABC", 10, 100)))
    assert api.build_universe_as_of(at(5), history) == ("ISSUER:OLD-ABC",)
    assert api.build_universe_as_of(at(15), history) == ("ISSUER:NEW-ABC",)


def test_reference_errors_do_not_render_rejected_payloads(api, member) -> None:
    payload = " provider-private-value "
    with pytest.raises(ValueError) as caught:
        member(instrument_id=payload)
    assert payload.strip() not in str(caught.value)


def test_selection_matches_independent_seeded_interval_oracle(api, member, manifest) -> None:
    import random

    rng = random.Random(404)
    raw = []
    for index in range(25):
        points = sorted(rng.sample(range(100), 4))
        raw.extend((f"SEC:{index:03}", points[pos], points[pos + 1]) for pos in (0, 2))
    rng.shuffle(raw)
    history = manifest(tuple(member(identifier, start, end) for identifier, start, end in raw))
    for day in range(100):
        expected = tuple(sorted({identifier for identifier, start, end in raw if day in range(start, end)}))
        assert api.build_universe_as_of(at(day), history) == expected
