"""Task 12 broker-neutral intent boundary; synthetic/non-capital only."""
from dataclasses import replace
from datetime import timedelta
import hashlib, runpy
from pathlib import Path
import pytest

def base():
    fixture=runpy.run_path(str(Path(__file__).with_name("test_portfolio_authority.py")))
    c,request,policy=fixture["case"]()
    from src.axiom2.portfolio.authority import evaluate_portfolio
    decision=evaluate_portfolio(request,policy)
    return c,request,policy,decision

def refs(request):
    from src.axiom2.contracts.equity_orders import AccountObservationRef,BrokerCapabilities
    account=AccountObservationRef(account_alias=request.account_alias,revision="r1",
        observed_at=request.as_of,expires_at=request.as_of+timedelta(seconds=30),source_digest=request.snapshot.source_digest,
        settled_cash_cents=request.snapshot.settled_cash_cents,reserved_cash_cents=0,
        positions=())
    caps=BrokerCapabilities(capability_digest=hashlib.sha256(b"caps").hexdigest(),
        whole_shares=True,regular_session_day_limit=True,fractional=False,
        notional=False,market_orders=False,extended_hours=False)
    return account,caps

def test_ready_shadow_builds_deterministic_noncapital_whole_share_day_limits():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    first=build_order_intents(decision,request,account,caps,mode="SHADOW")
    second=build_order_intents(decision,request,account,caps,mode="SHADOW")
    assert first==second and len(first)==5
    assert all(x.side=="BUY" and x.order_type=="LIMIT" and x.time_in_force=="DAY" for x in first)
    assert all(type(x.quantity) is int and x.quantity>0 and x.notional_cents is None for x in first)
    assert all(x.execution_enabled is False and x.capital_authorized is False for x in first)
def test_tampered_risk_decision_fails_before_intent_creation():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    object.__setattr__(decision,"gross_trade_cents",decision.gross_trade_cents+1)
    with pytest.raises(ValueError,match="risk decision digest mismatch"):
        build_order_intents(decision,request,account,caps,mode="SHADOW")

@pytest.mark.parametrize("mode",["LIVE","PRODUCTION","PAPER"])
def test_task12_refuses_nonshadow_modes(mode):
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    with pytest.raises(ValueError,match="mode"):
        build_order_intents(decision,request,account,caps,mode=mode)

def test_wrong_account_or_stale_revision_fails_closed():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    with pytest.raises(ValueError,match="account"):
        build_order_intents(decision,request,replace(account,account_alias="other"),caps,mode="SHADOW")
    with pytest.raises(ValueError,match="revision"):
        build_order_intents(decision,request,replace(account,revision=""),caps,mode="SHADOW")

def test_capabilities_are_enablement_not_advertising():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    for changed in (replace(caps,whole_shares=False),replace(caps,regular_session_day_limit=False)):
        with pytest.raises(ValueError,match="capabil"):
            build_order_intents(decision,request,account,changed,mode="SHADOW")
def test_stale_account_observation_and_changed_snapshot_are_rejected():
    _,request,policy,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    stale=replace(account,observed_at=request.as_of-timedelta(seconds=2),expires_at=request.as_of-timedelta(microseconds=1))
    with pytest.raises(ValueError,match="expired"):
        build_order_intents(decision,request,stale,caps,mode="SHADOW")
    changed=replace(account,source_digest=hashlib.sha256(b"other").hexdigest())
    with pytest.raises(ValueError,match="snapshot"):
        build_order_intents(decision,request,changed,caps,mode="SHADOW")

def test_nonready_decision_cannot_be_interpreted_as_liquidation():
    _,request,policy,_=base(); account,caps=refs(request)
    from src.axiom2.portfolio.authority import evaluate_portfolio
    from src.axiom2.portfolio.intents import build_order_intents
    hold=evaluate_portfolio(replace(request,session_index=101),policy)
    assert hold.status=="HOLD"
    assert build_order_intents(hold,replace(request,session_index=101),account,caps,mode="SHADOW")==()

def test_exit_required_only_sells_observed_inventory():
    c,request,policy,_=base()
    fixture=runpy.run_path(str(Path(__file__).with_name("test_portfolio_authority.py")))
    held=fixture["held"](c,request)
    from src.axiom2.portfolio.authority import evaluate_portfolio
    from src.axiom2.portfolio.intents import build_order_intents
    exit_decision=evaluate_portfolio(replace(held,session_index=105),policy)
    account,caps=refs(held)
    account=replace(account,positions=tuple((o.instrument_id,7) for o in request.candidates))
    intents=build_order_intents(exit_decision,replace(held,session_index=105),account,caps,mode="SHADOW")
    assert len(intents)==5
    assert all(x.side=="SELL" and x.quantity==7 for x in intents)
    assert all(x.notional_cents is None and not x.capital_authorized for x in intents)

def test_signed_receipt_binds_exact_intent_set():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents,sign_risk_intent_receipt,verify_risk_intent_receipt
    from src.evidence.signing import Ed25519Signer,TrustStore
    signer=Ed25519Signer.generate(); trust=TrustStore()
    trust.add(signer.trusted_key(valid_from=request.as_of-timedelta(seconds=1)))
    intents=build_order_intents(decision,request,account,caps,mode="SHADOW")
    envelope=sign_risk_intent_receipt(decision,intents,account,caps,signer=signer,created_at=request.as_of)
    payload=verify_risk_intent_receipt(envelope,intents,trust_store=trust)
    assert payload.decision_digest==decision.decision_digest and payload.intent_count==5
    with pytest.raises(ValueError):
        verify_risk_intent_receipt(None,intents,trust_store=trust)
    with pytest.raises(ValueError):
        verify_risk_intent_receipt(envelope,intents[:-1],trust_store=trust)

def test_price_rounding_never_exceeds_target_or_uses_sale_proceeds():
    _,request,_,decision=base(); account,caps=refs(request)
    from src.axiom2.portfolio.intents import build_order_intents
    intents=build_order_intents(decision,request,account,caps,mode="SHADOW")
    by_id={o.instrument_id:o for o in request.candidates}
    for intent in intents:
        ask=by_id[intent.instrument_id].ask_micros
        assert intent.limit_price_micros>=ask
        assert intent.quantity*intent.limit_price_micros<=160_000*10_000
    assert sum((x.quantity*x.limit_price_micros+9999)//10000 for x in intents)<=account.settled_cash_cents-account.reserved_cash_cents


def test_duplicate_account_inventory_is_rejected_at_construction():
    _, request, _, _ = base()
    account, _ = refs(request)
    with pytest.raises(ValueError, match="duplicate position"):
        replace(account, positions=(("equity:A", 7), ("equity:A", 7)))

def test_reflected_duplicate_inventory_is_rejected_before_exit_intents():
    c, request, policy, _ = base()
    fixture = runpy.run_path(str(Path(__file__).with_name("test_portfolio_authority.py")))
    held = replace(fixture["held"](c, request), session_index=105)
    from src.axiom2.portfolio.authority import evaluate_portfolio
    from src.axiom2.portfolio.intents import build_order_intents
    decision = evaluate_portfolio(held, policy)
    assert decision.status == "EXIT_REQUIRED"
    account, caps = refs(held)
    instrument = request.candidates[0].instrument_id
    object.__setattr__(account, "positions", ((instrument, 7), (instrument, 7)))
    with pytest.raises(ValueError, match="duplicate position"):
        build_order_intents(decision, held, account, caps, mode="SHADOW")
