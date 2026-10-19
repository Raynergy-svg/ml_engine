from dataclasses import replace
from pathlib import Path
import runpy

import pytest

_FIXTURE=runpy.run_path(str(Path(__file__).with_name("test_portfolio_authority.py")))
case=_FIXTURE["case"]
held=_FIXTURE["held"]

def evaluate(request,policy):
    from src.axiom2.portfolio.authority import evaluate_portfolio
    return evaluate_portfolio(request,policy)

def test_zero_valued_held_security_remains_representable_for_reconciliation():
    c,request,policy=case()
    position=c.Position(
        instrument_id="DELISTED",market_value_cents=0,sector_id="legacy",
        correlation_group_id="legacy",
    )
    snapshot=replace(
        request.snapshot,positions=(position,),cash_cents=request.snapshot.nav_cents,
        cohort_entry_session=100,reconciled=False,
    )
    result=evaluate(replace(request,snapshot=snapshot),policy)
    assert result.status=="RECONCILE"
    assert "ACCOUNT_UNRECONCILED" in result.reasons

def test_external_cash_flow_does_not_create_trading_drawdown():
    c,request,policy=case()
    snapshot=replace(
        request.snapshot,nav_cents=500_000,cash_cents=500_000,
        settled_cash_cents=500_000,peak_nav_cents=1_000_000,
        net_external_cash_flow_since_peak_cents=-500_000,
    )
    result=evaluate(replace(request,snapshot=snapshot),policy)
    assert result.status=="READY_SHADOW"
    assert "DRAWDOWN_LIMIT" not in result.reasons

def test_risk_off_management_remains_visible_when_routing_is_closed():
    c,request,policy=case()
    current=held(c,request)
    outside=replace(
        current,risk_state="RISK_OFF",
        as_of=current.session_close,
        snapshot=replace(current.snapshot,observed_at=current.session_close),
    )
    result=evaluate(outside,policy)
    assert result.status=="EXIT_REQUIRED"
    assert "RISK_OFF" in result.reasons
    assert "ROUTING_CLOSED" in result.reasons
    assert result.targets==()

def test_risk_decision_integrity_verifier_recomputes_digest_and_denies_mutation():
    from src.axiom2.portfolio.authority import verify_risk_decision_integrity
    _,request,policy=case()
    decision=evaluate(request,policy)
    assert verify_risk_decision_integrity(decision) is None
    object.__setattr__(decision,"residual_cash_cents",decision.residual_cash_cents+1)
    with pytest.raises(ValueError,match="decision digest"):
        verify_risk_decision_integrity(decision)


def test_external_deposit_above_raw_peak_is_representable_when_flow_adjusted_equity_is_not():
    _,request,policy=case()
    snapshot=replace(
        request.snapshot,nav_cents=1_500_000,cash_cents=1_500_000,
        settled_cash_cents=1_500_000,peak_nav_cents=1_000_000,
        net_external_cash_flow_since_peak_cents=500_000,
    )
    assert evaluate(replace(request,snapshot=snapshot),policy).status=="READY_SHADOW"


def test_external_flow_state_cannot_imply_negative_flow_adjusted_equity():
    _,request,_=case()
    with pytest.raises(ValueError,match="external cash flow"):
        replace(request.snapshot,net_external_cash_flow_since_peak_cents=1_000_001)
