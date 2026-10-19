"""Replay-safe offline engine binding inputs to durable intent reservations."""
from dataclasses import asdict, dataclass, field
import hashlib
from datetime import timezone
from src.evidence.canonical import canonical_bytes
from src.axiom2.portfolio.contracts import validate_request, _digest
from src.axiom2.portfolio.intents import build_order_intents
from src.axiom2.contracts.equity_orders import BrokerCapabilities
from .fills import FillPolicy, MarketSnapshot, HypotheticalFill, hypothetical_fill

def digest(value): return hashlib.sha256(canonical_bytes(value)).hexdigest()

@dataclass(frozen=True,slots=True,kw_only=True)
class ShadowReceipt:
    decision_digest: str
    context_digest: str
    fills: tuple[HypotheticalFill,...]
    hypothetical: bool=field(default=True,init=False)
    execution_enabled: bool=field(default=False,init=False)
    capital_authorized: bool=field(default=False,init=False)

    def __post_init__(self):
        _digest(self.decision_digest,'decision_digest'); _digest(self.context_digest,'context_digest')
        if self.hypothetical is not True or self.execution_enabled is not False or self.capital_authorized is not False:
            raise ValueError('shadow receipt authority flags invalid')
        if type(self.fills) is not tuple: raise TypeError('immutable fills required')
        for fill in self.fills:
            if type(fill) is not HypotheticalFill: raise TypeError('exact HypotheticalFill required')
            fill.__post_init__()
        if len({fill.intent_id for fill in self.fills})!=len(self.fills): raise ValueError('duplicate fill')

class ShadowEngine:
    def __init__(self,*,request,capabilities,policy,journal,portfolio=None,risk_policy=None):
        validate_request(request)
        if type(capabilities) is not BrokerCapabilities: raise TypeError('exact BrokerCapabilities required')
        if type(policy) is not FillPolicy: raise TypeError('exact FillPolicy required')
        capabilities.__post_init__(); policy.__post_init__()
        if (portfolio is None)!=(risk_policy is None):raise ValueError('portfolio and risk policy required together')
        if portfolio is not None:
            from src.axiom2.shadow.portfolio import ShadowPortfolioLedger
            from src.axiom2.portfolio.contracts import RiskPolicy
            if type(portfolio) is not ShadowPortfolioLedger or type(risk_policy) is not RiskPolicy:
                raise TypeError('exact durable portfolio and frozen risk policy required')
            if portfolio.journal is not journal:raise ValueError('portfolio must share execution journal')
        self.portfolio=portfolio;self.risk_policy=risk_policy
        self.request=request; self.capabilities=capabilities; self.policy=policy; self.journal=journal

    def step(self,risk_decision,market_snapshot,account_snapshot):
        validate_request(self.request); self.capabilities.__post_init__(); self.policy.__post_init__()
        if type(market_snapshot) is not MarketSnapshot: raise TypeError('exact MarketSnapshot required')
        market_snapshot.__post_init__()
        if market_snapshot.as_of.astimezone(timezone.utc)!=self.request.as_of.astimezone(timezone.utc): raise ValueError('market session decision time mismatch')
        if (market_snapshot.session_open.astimezone(timezone.utc),market_snapshot.session_close.astimezone(timezone.utc),market_snapshot.calendar_digest)!=(
                self.request.session_open.astimezone(timezone.utc),self.request.session_close.astimezone(timezone.utc),self.request.calendar_digest):
            raise ValueError('market session/calendar mismatch')
        if not market_snapshot.session_open.astimezone(timezone.utc)<=market_snapshot.as_of.astimezone(timezone.utc)<market_snapshot.session_close.astimezone(timezone.utc):
            raise ValueError('market session closed')
        intents=build_order_intents(risk_decision,self.request,account_snapshot,self.capabilities,mode='SHADOW')
        context=digest({'request':asdict(self.request),'decision':asdict(risk_decision),
            'account':asdict(account_snapshot),'capabilities':asdict(self.capabilities),
            'fill_policy':asdict(self.policy),'market':asdict(market_snapshot),'mode':'SHADOW'})
        if self.portfolio is not None:self.portfolio.validate_step(self.request,account_snapshot,context=context,decision=risk_decision,risk_policy=self.risk_policy,fill_policy=self.policy,market=market_snapshot)
        quotes={q.instrument_id:q for q in market_snapshot.quotes}
        computed_fills=tuple(hypothetical_fill(intent,quotes.get(intent.instrument_id),market_snapshot,self.policy)
            for intent in intents)
        buy_debit=sum((fill.quantity*fill.price_micros+9999)//10000+fill.fee_cents
            for intent,fill in zip(intents,computed_fills) if intent.side=='BUY' and fill.quantity)
        available_cash=account_snapshot.settled_cash_cents-account_snapshot.reserved_cash_cents
        if buy_debit>available_cash:
            raise ValueError('hypothetical cash and fee budget exceeded')
        if risk_decision.residual_cash_cents is not None and available_cash-buy_debit<risk_decision.residual_cash_cents:
            raise ValueError('hypothetical cash reserve bound breached')
        modeled_cost=0
        for intent,fill in zip(intents,computed_fills):
            if not fill.quantity: continue
            quote=quotes[intent.instrument_id]
            adverse_twice_price=(2*fill.price_micros-quote.bid_micros-quote.ask_micros) if intent.side=='BUY' else (quote.bid_micros+quote.ask_micros-2*fill.price_micros)
            modeled_cost+=fill.fee_cents+(max(0,adverse_twice_price)*fill.quantity+19999)//20000
        if risk_decision.estimated_cost_cents is not None and modeled_cost>risk_decision.estimated_cost_cents:
            raise ValueError('hypothetical cost bound breached')
        fills=[]
        for intent,computed in zip(intents,computed_fills):
            reservation=self.journal.reserve(intent,context_digest=context)
            result=asdict(computed)
            if reservation.state=='COMPLETED':
                if canonical_bytes(reservation.result)!=canonical_bytes(result): raise ValueError('journal fill replay mismatch')
            else:
                reservation=self.journal.complete(intent.intent_id,result=result,expected_head=self.journal.head)
                if canonical_bytes(reservation.result)!=canonical_bytes(result): raise ValueError('journal fill completion mismatch')
            fills.append(computed)
        if self.portfolio is not None:self.portfolio.apply_step(self.request,risk_decision,account_snapshot,self.capabilities,self.policy,market_snapshot,risk_policy=self.risk_policy,context=context)
        return ShadowReceipt(decision_digest=risk_decision.decision_digest,context_digest=context,fills=tuple(fills))
