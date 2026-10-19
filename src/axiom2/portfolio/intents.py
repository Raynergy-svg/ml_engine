"""Deterministic shadow-only conversion of risk decisions to broker-neutral intents."""
from dataclasses import asdict
from datetime import timezone
import hashlib
from src.evidence.canonical import canonical_bytes
from src.axiom2.portfolio.authority import verify_risk_decision_integrity
from src.axiom2.portfolio.contracts import PortfolioRequest,validate_request
from src.axiom2.contracts.equity_orders import AccountObservationRef,BrokerCapabilities,EquityOrderIntent,RiskIntentReceipt
from src.evidence.signing import verify_envelope

def _id(body):
    return hashlib.sha256(canonical_bytes(body)).hexdigest()

def build_order_intents(decision,request,account,capabilities,*,mode):
    verify_risk_decision_integrity(decision)
    validate_request(request)
    if type(account) is not AccountObservationRef: raise TypeError("account must be AccountObservationRef")
    if type(capabilities) is not BrokerCapabilities: raise TypeError("capabilities must be BrokerCapabilities")
    account.__post_init__(); capabilities.__post_init__()
    if mode!="SHADOW": raise ValueError("mode must be SHADOW")
    if decision.account_alias!=request.account_alias or account.account_alias!=request.account_alias:
        raise ValueError("account mismatch")
    if not account.revision: raise ValueError("account revision is required")
    if decision.input_digest!=_request_digest(request):
        raise ValueError("risk decision input mismatch")
    if account.source_digest!=request.snapshot.source_digest:
        raise ValueError("account snapshot mismatch")
    now=request.as_of.astimezone(timezone.utc)
    observed=account.observed_at.astimezone(timezone.utc)
    expires=account.expires_at.astimezone(timezone.utc)
    if observed>now: raise ValueError("account observation is future")
    if now>expires: raise ValueError("account observation is expired")
    if not capabilities.whole_shares or not capabilities.regular_session_day_limit:
        raise ValueError("required broker capabilities unavailable")
    if decision.status not in {"READY_SHADOW","EXIT_REQUIRED"}:
        return ()
    opportunities={o.instrument_id:o for o in request.candidates}
    intents=[]; cash=account.settled_cash_cents-account.reserved_cash_cents
    if decision.status=="EXIT_REQUIRED":
        for instrument_id,quantity in sorted(account.positions):
            if quantity<=0: continue
            o=opportunities.get(instrument_id)
            if o is None: raise ValueError("exit inventory lacks bound opportunity")
            price=o.bid_micros
            body={"decision_digest":decision.decision_digest,"account_alias":account.account_alias,
                  "account_revision":account.revision,"capability_digest":capabilities.capability_digest,
                  "mode":mode,"instrument_id":instrument_id,"side":"SELL","revision":1}
            intents.append(EquityOrderIntent(intent_id="i:"+_id(body)[:48],
                decision_digest=decision.decision_digest,account_alias=account.account_alias,
                account_revision=account.revision,capability_digest=capabilities.capability_digest,
                mode=mode,instrument_id=instrument_id,side="SELL",order_type="LIMIT",
                time_in_force="DAY",quantity=quantity,notional_cents=None,
                limit_price_micros=price,expires_at=request.session_close))
        return tuple(intents)
    for target in decision.targets:
        o=opportunities.get(target.instrument_id)
        if o is None: raise ValueError("target lacks bound opportunity")
        # Use the observed ask as the conservative DAY limit for a buy. Quantity
        # floors so price * quantity never exceeds the desired target value.
        price=o.ask_micros
        quantity=(target.target_value_cents*10_000)//price
        if quantity<=0: raise ValueError("target cannot fund one whole share")
        cost=(quantity*price+9999)//10_000
        if cost>cash: raise ValueError("insufficient settled cash")
        cash-=cost
        body={"decision_digest":decision.decision_digest,"account_alias":account.account_alias,
              "account_revision":account.revision,"capability_digest":capabilities.capability_digest,
              "mode":mode,"instrument_id":target.instrument_id,"side":"BUY","revision":1}
        intents.append(EquityOrderIntent(intent_id="i:"+_id(body)[:48],
            decision_digest=decision.decision_digest,account_alias=account.account_alias,
            account_revision=account.revision,capability_digest=capabilities.capability_digest,
            mode=mode,instrument_id=target.instrument_id,side="BUY",order_type="LIMIT",
            time_in_force="DAY",quantity=quantity,notional_cents=None,
            limit_price_micros=price,expires_at=request.session_close))
    return tuple(intents)

def sign_risk_intent_receipt(decision,intents,account,capabilities,*,signer,created_at):
    verify_risk_decision_integrity(decision)
    if type(intents) is not tuple or any(type(x) is not EquityOrderIntent for x in intents):
        raise TypeError("intents must be an immutable exact tuple")
    payload=RiskIntentReceipt(decision_digest=decision.decision_digest,
        intents_digest=_id(tuple(asdict(x) for x in intents)),
        account_alias=account.account_alias,account_revision=account.revision,
        capability_digest=capabilities.capability_digest,mode="SHADOW",intent_count=len(intents))
    return signer.sign(payload,created_at=created_at)

def verify_risk_intent_receipt(envelope,intents,*,trust_store):
    if envelope is None: raise ValueError("signed risk intent receipt is required")
    payload=verify_envelope(envelope,RiskIntentReceipt,trust_store)
    if payload.mode!="SHADOW": raise ValueError("risk intent receipt mode mismatch")
    if payload.intent_count!=len(intents) or payload.intents_digest!=_id(tuple(asdict(x) for x in intents)):
        raise ValueError("risk intent receipt does not bind exact intents")
    return payload

def _request_digest(request):
    normalized=asdict(request)
    normalized["candidates"]=sorted(normalized["candidates"],key=lambda o:(o["rank"],o["instrument_id"]))
    normalized["snapshot"]["positions"]=sorted(normalized["snapshot"]["positions"],key=lambda p:p["instrument_id"])
    return _id(normalized)
