"""Review-driven tests of boundary values, cost coherence and state precedence."""
from dataclasses import asdict, replace
from datetime import timedelta, timezone
import hashlib
from pathlib import Path
import runpy

import pytest


def case():
    fixture = runpy.run_path(str(Path(__file__).with_name("test_portfolio_authority.py")))
    return fixture["case"]()


def evaluate(request, policy):
    from src.axiom2.portfolio.authority import evaluate_portfolio
    return evaluate_portfolio(request, policy)


@pytest.mark.parametrize("buffer_bps", [0, 1, 4])
def test_cost_buffer_must_cover_observable_entry_half_spread(buffer_bps):
    _, request, policy = case()
    result = evaluate(request, replace(policy, cost_buffer_bps=buffer_bps))
    assert result.status == "REJECTED"
    assert "COST_BUFFER_BELOW_SPREAD" in result.reasons
    assert result.targets == ()


def test_spread_cost_boundary_uses_exact_cross_products():
    _, request, policy = case()
    request = replace(request, candidates=tuple(
        replace(o, bid_micros=199_000_000, ask_micros=201_000_000)
        for o in request.candidates
    ))
    limits = replace(policy, max_spread_bps=100, cost_buffer_bps=50)
    assert evaluate(request, limits).status == "READY_SHADOW"
    blocked = evaluate(request, replace(limits, cost_buffer_bps=49))
    assert blocked.status == "REJECTED" and "COST_BUFFER_BELOW_SPREAD" in blocked.reasons


@pytest.mark.parametrize("target,ttl,reason", [
    ("quoted_at", 30, "QUOTE_STALE"),
    ("metadata_at", 86400, "METADATA_STALE"),
])
def test_freshness_limit_is_inclusive_but_next_microsecond_is_stale(target, ttl, reason):
    _, request, policy = case()
    boundary = request.as_of - timedelta(seconds=ttl)
    first = replace(request.candidates[0], **{target: boundary})
    request = replace(request, candidates=(first,) + request.candidates[1:])
    assert evaluate(request, policy).status == "READY_SHADOW"
    expired = replace(first, **{target: boundary - timedelta(microseconds=1)})
    result = evaluate(replace(request, candidates=(expired,) + request.candidates[1:]), policy)
    assert result.status == "REJECTED" and reason in result.reasons


def test_cost_rounds_up_and_equal_weight_remainder_stays_cash():
    _, request, policy = case()
    request = replace(request, snapshot=replace(request.snapshot, nav_cents=5003,
        cash_cents=5003, settled_cash_cents=5003, peak_nav_cents=5003))
    result = evaluate(request, replace(policy, cost_buffer_bps=21))
    assert result.status == "READY_SHADOW"
    assert [p.target_value_cents for p in result.targets] == [800] * 5
    assert result.gross_trade_cents == 4000 and result.estimated_cost_cents == 9
    assert result.residual_cash_cents == 994


def test_target_cannot_be_rounded_up_from_zero():
    _, request, policy = case()
    request = replace(request, snapshot=replace(request.snapshot, nav_cents=11,
        cash_cents=11, settled_cash_cents=11, peak_nav_cents=11))
    result = evaluate(request, replace(policy, allocation_bps=1))
    assert result.status == "REJECTED" and "INSUFFICIENT_ALLOCATION" in result.reasons
    assert result.targets == ()


def test_cash_reserve_is_not_funded_with_unsettled_cash():
    _, request, policy = case()
    current = replace(request, snapshot=replace(request.snapshot, settled_cash_cents=850_000))
    result = evaluate(current, policy)
    assert result.status == "REJECTED" and "CASH_RESERVE" in result.reasons


def test_output_digest_is_independently_recomputable():
    from src.evidence.canonical import canonical_bytes
    _, request, policy = case()
    result = evaluate(request, policy)
    body = asdict(result)
    digest = body.pop("decision_digest")
    assert hashlib.sha256(canonical_bytes(body)).hexdigest() == digest


def test_equivalent_timezones_have_identical_decision_identity():
    _, request, policy = case()
    zone = timezone(timedelta(hours=-4))
    changed = replace(request,
        as_of=request.as_of.astimezone(zone), signal_at=request.signal_at.astimezone(zone),
        session_open=request.session_open.astimezone(zone), session_close=request.session_close.astimezone(zone),
        snapshot=replace(request.snapshot, observed_at=request.snapshot.observed_at.astimezone(zone)),
        candidates=tuple(replace(o, quoted_at=o.quoted_at.astimezone(zone),
                                metadata_at=o.metadata_at.astimezone(zone)) for o in request.candidates))
    assert evaluate(request, policy) == evaluate(changed, policy)


def test_dst_fold_uses_elapsed_utc_time_not_wall_time():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.axiom2.portfolio.authority import _age_reason
    zone = ZoneInfo("America/New_York")
    before = datetime(2026, 11, 1, 1, 50, tzinfo=zone, fold=0)
    after = datetime(2026, 11, 1, 1, 10, tzinfo=zone, fold=1)
    assert _age_reason(before, after, 30, "SNAPSHOT") == "SNAPSHOT_STALE"
    assert _age_reason(before, after, 1200, "SNAPSHOT") is None


@pytest.mark.parametrize("kind", ["SYNTHETIC", "DEVELOPMENT", "LIVE_SHADOW"])
def test_no_input_class_can_authorize_capital(kind):
    _, request, policy = case()
    result = evaluate(replace(request, source_kind=kind), policy)
    assert result.status == "READY_SHADOW" and result.source_kind == kind
    assert not result.execution_enabled and not result.capital_authorized


def test_kill_precedes_open_order_management_but_never_disables_reconciliation_elsewhere():
    _, request, policy = case()
    request = replace(request, risk_state="HALTED", snapshot=replace(request.snapshot, open_order_count=1))
    result = evaluate(request, policy)
    assert result.status == "HALTED" and result.reasons == ("KILL_STATE",)
    assert not result.targets  # A separate reconciliation worker continues observing.


@pytest.mark.parametrize("entry", [None, 95, 100, 101, 105])
def test_flat_account_with_unclosed_cohort_identity_never_creates_replacement(entry):
    _, request, policy = case()
    result = evaluate(replace(request, snapshot=replace(request.snapshot, cohort_entry_session=entry)), policy)
    if entry is None:
        assert result.status == "READY_SHADOW"
    else:
        assert result.status == "RECONCILE" and "COHORT_STATE_UNKNOWN" in result.reasons
        assert not result.targets


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("quote_ttl_seconds", 0),
    ("max_name_bps", 10001), ("cost_buffer_bps", -1),
])
def test_reflected_policy_mutation_cannot_skip_admission_validation(field, value):
    _, request, policy = case()
    object.__setattr__(policy, field, value)
    with pytest.raises((TypeError, ValueError), match=field):
        evaluate(request, policy)


def test_provider_numeric_conversion_hooks_are_never_executed():
    class ProviderNumber:
        def __int__(self):
            raise AssertionError("provider conversion executed")
        def __str__(self):
            raise AssertionError("provider stringification executed")
    _, _, policy = case()
    with pytest.raises(TypeError, match="allocation_bps"):
        replace(policy, allocation_bps=ProviderNumber())
