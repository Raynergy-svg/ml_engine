"""Offline portfolio/risk contracts; no real account, capital or holdout."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import random

import pytest

NOW = datetime(2026, 10, 1, 14, tzinfo=timezone.utc)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def case():
    c = importlib.import_module("src.axiom2.portfolio.contracts")
    policy = c.RiskPolicy(
        allocation_bps=8000, max_name_bps=2000, max_sector_bps=4000,
        max_correlation_group_bps=4000, max_gross_bps=9000,
        cash_reserve_bps=1000, max_turnover_bps=5000,
        max_participation_bps=100, min_net_edge_bps=10,
        max_spread_bps=50, cost_buffer_bps=20, max_drawdown_bps=2000,
        quote_ttl_seconds=30, snapshot_ttl_seconds=30,
        signal_ttl_seconds=300, metadata_ttl_seconds=86400,
        rebalance_anchor_session=100,
    )
    snapshot = c.PortfolioSnapshot(
        observed_at=NOW - timedelta(seconds=1), nav_cents=1_000_000,
        cash_cents=1_000_000, settled_cash_cents=1_000_000,
        reserved_cash_cents=0, positions=(), open_order_count=0,
        unresolved_orders=False, reconciled=True, corporate_actions_clear=True,
        peak_nav_cents=1_000_000,
        cohort_entry_session=None, source_digest=digest("snapshot"),
    )
    candidates = tuple(c.RankedOpportunity(
        instrument_id=f"EQ{i}", asset_class="EQUITY", rank=i + 1,
        sector_id=f"S{i // 2}", correlation_group_id=f"G{i // 2}",
        expected_net_edge_bps=50, adv_cents=100_000_000,
        bid_micros=100_000_000, ask_micros=100_100_000,
        quoted_at=NOW - timedelta(seconds=1), metadata_at=NOW - timedelta(seconds=60),
        eligible=True, membership_known=True, corporate_actions_clear=True,
    ) for i in range(5))
    request = c.PortfolioRequest(
        decision_id="synthetic-decision-1", account_alias="shadow-account",
        as_of=NOW, session_index=100,
        session_open=NOW.replace(hour=13, minute=30), session_close=NOW.replace(hour=20),
        risk_state="RUN", source_kind="SYNTHETIC", signal_at=NOW - timedelta(seconds=2),
        model_artifact_hash=digest("model"), promotion_decision_hash=digest("promotion"),
        signal_digest=digest("signal"), economic_evidence_digest=digest("economics"),
        universe_digest=digest("universe"), calendar_digest=digest("calendar"),
        campaign_portfolio_digest=c.PHASE1_PORTFOLIO_DIGEST,
        snapshot=snapshot, candidates=candidates,
    )
    return c, request, policy


def evaluate(request, policy):
    return importlib.import_module("src.axiom2.portfolio.authority").evaluate_portfolio(request, policy)


def update_first(request, **changes):
    return replace(request, candidates=(replace(request.candidates[0], **changes),) + request.candidates[1:])


def held(c, request):
    positions = tuple(c.Position(
        instrument_id=o.instrument_id, market_value_cents=160_000,
        sector_id=o.sector_id, correlation_group_id=o.correlation_group_id,
    ) for o in request.candidates)
    return replace(request, snapshot=replace(
        request.snapshot, positions=positions, cash_cents=200_000,
        settled_cash_cents=200_000, cohort_entry_session=100,
    ))


def test_portfolio_core_is_present():
    assert importlib.util.find_spec("src.axiom2.portfolio") is not None


def test_equal_weight_targets_turnover_and_cash():
    _, request, policy = case()
    result = evaluate(request, policy)
    assert result.status == "READY_SHADOW" and result.reasons == ()
    assert [(x.instrument_id, x.target_value_cents) for x in result.targets] == [
        (f"EQ{i}", 160_000) for i in range(5)
    ]
    assert result.gross_trade_cents == 800_000
    assert result.turnover_bps_ceiling == 4000  # 0.5 * actual traded notional / NAV
    assert result.estimated_cost_cents == 1600
    assert result.residual_cash_cents == 198_400
    assert result.execution_enabled is False and result.capital_authorized is False
    assert result.source_kind == "SYNTHETIC"


def test_phase1_policy_identity_matches_existing_research_contract():
    c, _, _ = case()
    from src.axiom2.research.ranker import PortfolioPolicy, portfolio_policy_digest
    assert c.PHASE1_PORTFOLIO_DIGEST == portfolio_policy_digest(PortfolioPolicy(5, 5, 5))


def test_permutation_and_exact_replay_are_identical():
    _, request, policy = case()
    first = evaluate(request, policy)
    assert first == evaluate(request, policy)
    assert first == evaluate(replace(request, candidates=tuple(reversed(request.candidates))), policy)


@pytest.mark.parametrize("field", [
    "model_artifact_hash", "promotion_decision_hash", "signal_digest",
    "economic_evidence_digest", "universe_digest", "calendar_digest",
])
def test_every_evidence_reference_is_bound(field):
    _, request, policy = case()
    changed = replace(request, **{field: digest("changed-" + field)})
    assert evaluate(changed, policy).decision_digest != evaluate(request, policy).decision_digest


def test_policy_changes_change_receipt_not_just_target_values():
    _, request, policy = case()
    changed = replace(policy, quote_ttl_seconds=31)
    a, b = evaluate(request, policy), evaluate(request, changed)
    assert a.targets == b.targets and a.policy_digest != b.policy_digest
    assert a.decision_digest != b.decision_digest


@pytest.mark.parametrize("field,value,reason", [
    ("max_name_bps", 1599, "NAME_LIMIT"),
    ("max_sector_bps", 3199, "SECTOR_LIMIT"),
    ("max_correlation_group_bps", 3199, "CORRELATION_LIMIT"),
    ("max_gross_bps", 7999, "GROSS_LIMIT"),
    ("max_turnover_bps", 3999, "TURNOVER_LIMIT"),
    ("max_participation_bps", 15, "PARTICIPATION_LIMIT"),
    ("min_net_edge_bps", 51, "EDGE_FLOOR"),
    ("max_spread_bps", 9, "SPREAD_LIMIT"),
])
def test_each_risk_limit_rejects_the_whole_cohort(field, value, reason):
    _, request, policy = case()
    result = evaluate(request, replace(policy, **{field: value}))
    assert result.status == "REJECTED" and reason in result.reasons
    assert result.targets == () and result.execution_enabled is False


def test_exact_exposure_and_turnover_limits_are_allowed():
    _, request, policy = case()
    policy = replace(policy, max_name_bps=1600, max_sector_bps=3200,
                     max_correlation_group_bps=3200, max_gross_bps=8000, max_turnover_bps=4000)
    assert evaluate(request, policy).status == "READY_SHADOW"


@pytest.mark.parametrize("field,value,reason", [
    ("eligible", False, "INELIGIBLE"),
    ("membership_known", False, "UNKNOWN_MEMBERSHIP"),
    ("corporate_actions_clear", False, "CORPORATE_ACTION_UNRESOLVED"),
    ("bid_micros", 100_200_000, "CROSSED_QUOTE"),
    ("expected_net_edge_bps", -1, "EDGE_FLOOR"),
    ("quoted_at", NOW - timedelta(seconds=31), "QUOTE_STALE"),
    ("quoted_at", NOW + timedelta(seconds=1), "QUOTE_FUTURE"),
    ("metadata_at", NOW - timedelta(seconds=86401), "METADATA_STALE"),
    ("metadata_at", NOW + timedelta(seconds=1), "METADATA_FUTURE"),
])
def test_market_and_economic_failures_do_not_trigger_reselection(field, value, reason):
    _, request, policy = case()
    result = evaluate(update_first(request, **{field: value}), policy)
    assert result.status == "REJECTED" and reason in result.reasons and result.targets == ()


@pytest.mark.parametrize("field,value,reason", [
    ("signal_at", NOW - timedelta(seconds=301), "SIGNAL_STALE"),
    ("signal_at", NOW + timedelta(seconds=1), "SIGNAL_FUTURE"),
    ("campaign_portfolio_digest", "f" * 64, "PORTFOLIO_POLICY_MISMATCH"),
])
def test_input_semantic_binding_failures(field, value, reason):
    _, request, policy = case()
    result = evaluate(replace(request, **{field: value}), policy)
    assert result.status == "REJECTED" and reason in result.reasons and not result.targets


def test_missing_top_five_is_not_backfilled_or_reweighted():
    _, request, policy = case()
    result = evaluate(replace(request, candidates=request.candidates[:4]), policy)
    assert result.status == "REJECTED" and "COHORT_INCOMPLETE" in result.reasons
    assert not result.targets


@pytest.mark.parametrize("changes,reason", [
    ({"settled_cash_cents": 790_000}, "INSUFFICIENT_SETTLED_CASH"),
    ({"reserved_cash_cents": 210_000}, "INSUFFICIENT_SETTLED_CASH"),
])
def test_unsettled_or_reserved_cash_cannot_fund_positions(changes, reason):
    _, request, policy = case()
    result = evaluate(replace(request, snapshot=replace(request.snapshot, **changes)), policy)
    assert result.status == "REJECTED" and reason in result.reasons


def test_costs_cannot_spend_the_cash_reserve():
    _, request, policy = case()
    result = evaluate(request, replace(policy, cost_buffer_bps=1000, cash_reserve_bps=1900))
    assert result.status == "REJECTED" and "CASH_RESERVE" in result.reasons


@pytest.mark.parametrize("changes,reason", [
    ({"reconciled": False}, "ACCOUNT_UNRECONCILED"),
    ({"open_order_count": 1}, "OPEN_ORDERS"),
    ({"unresolved_orders": True}, "ORDER_STATE_UNKNOWN"),
    ({"observed_at": NOW - timedelta(seconds=31)}, "SNAPSHOT_STALE"),
    ({"observed_at": NOW + timedelta(seconds=1)}, "SNAPSHOT_FUTURE"),
])
def test_account_uncertainty_requires_reconciliation(changes, reason):
    _, request, policy = case()
    result = evaluate(replace(request, snapshot=replace(request.snapshot, **changes)), policy)
    assert result.status == "RECONCILE" and reason in result.reasons and not result.targets


def test_holding_and_maturity_do_not_open_an_overlapping_cohort():
    c, request, policy = case()
    request = held(c, request)
    before = evaluate(replace(request, session_index=104), policy)
    due = evaluate(replace(request, session_index=105), policy)
    assert before.status == "HOLD" and before.targets == ()
    assert due.status == "EXIT_REQUIRED" and "COHORT_MATURED" in due.reasons and not due.targets
    assert due.capital_authorized is False


def test_positions_without_cohort_identity_are_not_silently_liquidated():
    c, request, policy = case()
    request = held(c, request)
    result = evaluate(replace(request, snapshot=replace(request.snapshot, cohort_entry_session=None)), policy)
    assert result.status == "RECONCILE" and "COHORT_STATE_UNKNOWN" in result.reasons
    assert not result.targets


def test_risk_off_directs_existing_position_management_but_never_new_exposure():
    c, request, policy = case()
    assert evaluate(replace(request, risk_state="RISK_OFF"), policy).status == "HALTED"
    result = evaluate(replace(held(c, request), risk_state="RISK_OFF"), policy)
    assert result.status == "EXIT_REQUIRED" and "RISK_OFF" in result.reasons and not result.targets


def test_kill_blocks_every_target_even_with_positions():
    c, request, policy = case()
    for current in (request, held(c, request)):
        result = evaluate(replace(current, risk_state="HALTED"), policy)
        assert result.status == "HALTED" and not result.targets and not result.execution_enabled


def test_drawdown_and_drift_require_existing_position_management():
    c, request, policy = case()
    current = held(c, request)
    drawdown = replace(current, snapshot=replace(current.snapshot, peak_nav_cents=1_250_000))
    assert evaluate(drawdown, policy).status == "EXIT_REQUIRED"
    position = replace(current.snapshot.positions[0], market_value_cents=800_000)
    drift = replace(current, snapshot=replace(current.snapshot, positions=(position,)))
    assert evaluate(drift, policy).status == "EXIT_REQUIRED"


@pytest.mark.parametrize("index", [99, 101, 104, 106])
def test_entry_only_on_preregistered_five_session_schedule(index):
    _, request, policy = case()
    result = evaluate(replace(request, session_index=index), policy)
    assert result.status == "HOLD" and not result.targets


@pytest.mark.parametrize("at", [NOW.replace(hour=13, minute=29), NOW.replace(hour=20)])
def test_outside_regular_market_session_is_not_executable(at):
    _, request, policy = case()
    request = replace(request, as_of=at, snapshot=replace(request.snapshot, observed_at=at))
    result = evaluate(request, policy)
    assert result.status == "HOLD" and "OUTSIDE_REGULAR_SESSION" in result.reasons
    assert not result.targets


@pytest.mark.parametrize("value", [True, 8000.0, float("nan"), float("inf"), "8000"])
def test_policy_rejects_noninteger_values_without_coercion(value):
    _, _, policy = case()
    with pytest.raises((TypeError, ValueError), match="allocation_bps"):
        replace(policy, allocation_bps=value)


@pytest.mark.parametrize("field,value", [
    ("nav_cents", True), ("cash_cents", float("nan")),
    ("settled_cash_cents", 1.0), ("reserved_cash_cents", -1),
    ("open_order_count", True), ("reconciled", 1),
])
def test_snapshot_rejects_invalid_types_and_ranges(field, value):
    _, request, _ = case()
    with pytest.raises((TypeError, ValueError), match=field):
        replace(request.snapshot, **{field: value})


def test_nav_must_reconcile_and_settled_cash_cannot_exceed_total_cash():
    _, request, _ = case()
    with pytest.raises(ValueError, match="nav_cents"):
        replace(request.snapshot, nav_cents=999_999)
    with pytest.raises(ValueError, match="settled_cash_cents"):
        replace(request.snapshot, settled_cash_cents=1_000_001)


@pytest.mark.parametrize("field,value", [
    ("expected_net_edge_bps", None), ("rank", True), ("adv_cents", 0),
    ("bid_micros", float("nan")), ("asset_class", "FX"),
    ("metadata_at", NOW.replace(tzinfo=None)),
])
def test_unknown_economic_and_market_values_are_not_coerced(field, value):
    _, request, _ = case()
    with pytest.raises((TypeError, ValueError), match=field):
        replace(request.candidates[0], **{field: value})


@pytest.mark.parametrize("field,value", [
    ("model_artifact_hash", "bad"), ("promotion_decision_hash", None),
    ("economic_evidence_digest", ""), ("calendar_digest", "A" * 64),
    ("as_of", NOW.replace(tzinfo=None)), ("source_kind", "PRODUCTION"),
    ("risk_state", "UNKNOWN"), ("session_index", True),
])
def test_malformed_requests_cannot_become_plans(field, value):
    _, request, _ = case()
    with pytest.raises((TypeError, ValueError), match=field):
        replace(request, **{field: value})


def test_mutable_collections_and_duplicate_identities_are_rejected():
    _, request, _ = case()
    with pytest.raises(TypeError, match="candidates"):
        replace(request, candidates=list(request.candidates))
    with pytest.raises(ValueError, match="duplicate"):
        replace(request, candidates=(request.candidates[0],) * 5)
    duplicates = (replace(request.candidates[0], rank=2),) + request.candidates[1:]
    with pytest.raises(ValueError, match="duplicate"):
        replace(request, candidates=duplicates)


def test_boundary_revalidates_reflectively_mutated_frozen_records():
    _, request, policy = case()
    object.__setattr__(request.snapshot, "settled_cash_cents", float("nan"))
    with pytest.raises((TypeError, ValueError), match="settled_cash_cents"):
        evaluate(request, policy)


def test_integer_rounding_invariants_over_200_seeded_portfolios():
    _, request, policy = case()
    rng = random.Random(20261001)
    for _ in range(200):
        nav = rng.randint(10_000, 50_000_000)
        allocation = rng.randint(100, 9000)
        current = replace(request, snapshot=replace(request.snapshot,
            nav_cents=nav, cash_cents=nav, settled_cash_cents=nav, peak_nav_cents=nav),
            candidates=tuple(replace(x, adv_cents=10**15) for x in request.candidates))
        limits = replace(policy, allocation_bps=allocation, cash_reserve_bps=0,
            max_name_bps=10000, max_sector_bps=10000,
            max_correlation_group_bps=10000, max_gross_bps=10000, max_turnover_bps=10000)
        result = evaluate(current, limits)
        assert result.status == "READY_SHADOW"
        values = [x.target_value_cents for x in result.targets]
        assert len(values) == 5 and len(set(values)) == 1 and min(values) > 0
        assert sum(values) * 10000 <= nav * allocation
        assert result.estimated_cost_cents == (sum(values) * limits.cost_buffer_bps + 9999) // 10000
        assert result.residual_cash_cents == nav - sum(values) - result.estimated_cost_cents
        assert result.residual_cash_cents >= 0 and not result.capital_authorized


def test_account_alias_and_snapshot_bytes_are_bound_to_decision():
    _, request, policy = case()
    original = evaluate(request, policy)
    for changed in (
        replace(request, account_alias="other-shadow-account"),
        replace(request, snapshot=replace(request.snapshot, source_digest=digest("other-snapshot"))),
    ):
        assert evaluate(changed, policy).decision_digest != original.decision_digest


def test_unresolved_account_corporate_action_requires_reconciliation():
    _, request, policy = case()
    current = replace(request, snapshot=replace(request.snapshot, corporate_actions_clear=False))
    result = evaluate(current, policy)
    assert result.status == "RECONCILE"
    assert "ACCOUNT_CORPORATE_ACTION_UNRESOLVED" in result.reasons
    assert not result.targets


def test_portfolio_decision_cannot_be_constructed_as_capital_authorization():
    _, request, policy = case()
    result = evaluate(request, policy)
    with pytest.raises(ValueError):
        replace(result, capital_authorized=True)
    with pytest.raises(ValueError):
        replace(result, execution_enabled=True)
