"""Pure normalized reconciliation. No provider schema or transport authority.

Signed fill quantity is buy-positive/sell-negative. All prices/money are cents.
Normalized correction chains and period-opening splits/dividend cash are explicit.
Provider authenticity, event ordering and lifecycle settlement remain unverified.
"""
from dataclasses import dataclass
from typing import ClassVar
from .costs import CostReceipt
@dataclass(frozen=True)
class Fill:
    event_id: str
    order_id: str
    instrument: str
    quantity: int
    price_cents: int
    fee_cents: int
    correction_of: str = ''
@dataclass(frozen=True)
class CorporateAction:
    event_id: str
    instrument: str
    kind: str
    numerator: int
    denominator: int
    cash_cents: int

@dataclass(frozen=True)
class Ledger:
    account: str
    mode: str
    initial_cash_cents: int
    initial_positions: tuple
    known_orders: tuple
    expected_fills: tuple
    peak_equity_cents: int
    external_cashflow_cents: int
    expected_corporate_actions: tuple = ()
@dataclass(frozen=True)
class Observations:
    account: str
    mode: str
    complete: bool
    fills: tuple
    timed_out: bool = False
    corporate_actions: tuple = ()
    unresolved_orders: tuple = ()
@dataclass(frozen=True)
class Account:
    account: str
    total_cash_cents: int
    settled_cash_cents: int
    positions: tuple
    complete: bool
    equity_cents: int
    external_cashflow_cents: int
    mode: str = 'observed'
    valuation_marks_micros: tuple = ()
@dataclass(frozen=True)
class Policy:
    policy_digest: str
    cash_tolerance_cents: int
    fee_tolerance_cents: int
@dataclass(frozen=True)
class ReconciliationDecision:
    state: str
    reason: str
    costs: CostReceipt | None = None
    flow_adjusted_peak_cents: int | None = None
    execution_enabled: bool = False
    capital_authorized: bool = False
@dataclass(frozen=True)
class Acceptance:
    minimum_regular_sessions: ClassVar[int] = 20
    minimum_non_overlapping_cycles: ClassVar[int] = 4
    sessions_per_cycle: ClassVar[int] = 5
    maximum_forbidden_effects: ClassVar[int] = 0
    maximum_unresolved_breaks: ClassVar[int] = 0
    policy_digest: str
    cost_tolerance_cents: int
    risk_tolerance_bps: int
    def __post_init__(self):
        _digest(self.policy_digest)
        _integer(self.cost_tolerance_cents,0)
        _integer(self.risk_tolerance_bps,0)

    def qualified(self, regular_session_indices, completed_cycles, *, synthetic, **kwargs):
        # Normalized caller claims cannot authenticate calendar passage or audits.
        # This offline implementation deliberately never issues qualification.
        self.__post_init__()
        return False

def _integer(v, minimum=None):
    if type(v) is not int or (minimum is not None and v<minimum): raise ValueError('INVALID_INTEGER')
def _digest(v):
    if type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v): raise ValueError('INVALID_POLICY')
def _positions(values):
    if type(values) is not tuple: raise ValueError('MUTABLE_POSITIONS')
    result={}
    for pair in values:
        if type(pair) is not tuple or len(pair)!=2: raise ValueError('INVALID_POSITION')
        name,q=pair
        if type(name) is not str or not name or name in result: raise ValueError('INVALID_POSITION')
        _integer(q,0); result[name]=q
    return result
def _fills(values):
    if type(values) is not tuple: raise ValueError('MUTABLE_FILLS')
    result={}
    for f in values:
        if type(f) is not Fill: raise ValueError('INVALID_FILL')
        if any(type(v) is not str or not v for v in (f.event_id,f.order_id,f.instrument)): raise ValueError('LOST_ID')
        _integer(f.quantity); _integer(f.price_cents,1); _integer(f.fee_cents,0)
        if not f.quantity or type(f.correction_of) is not str: raise ValueError('INVALID_FILL')
        if f.event_id in result and result[f.event_id]!=f: raise ValueError('CONFLICTING_EVENT')
        result[f.event_id]=f
    superseded={}
    for f in result.values():
        if not f.correction_of: continue
        prior=result.get(f.correction_of)
        if prior is None or prior.event_id==f.event_id or prior.order_id!=f.order_id or prior.instrument!=f.instrument or (prior.quantity>0)!=(f.quantity>0):
            raise ValueError('INVALID_CORRECTION')
        if prior.event_id in superseded: raise ValueError('BRANCHED_CORRECTION')
        superseded[prior.event_id]=f.event_id
        seen={f.event_id}; cursor=f
        while cursor.correction_of:
            if cursor.correction_of in seen: raise ValueError('CYCLIC_CORRECTION')
            seen.add(cursor.correction_of)
            cursor=result.get(cursor.correction_of)
            if cursor is None: raise ValueError('DANGLING_CORRECTION')
    return result

def _effective_fills(values):
    superseded={fill.correction_of for fill in values.values() if fill.correction_of}
    return {key:fill for key,fill in values.items() if key not in superseded}

def _actions(values):
    if type(values) is not tuple: raise ValueError('MUTABLE_ACTIONS')
    result={}
    for action in values:
        if type(action) is not CorporateAction: raise ValueError('UNRESOLVED_ACTION')
        if any(type(v) is not str or not v for v in (action.event_id,action.instrument,action.kind)):
            raise ValueError('INVALID_ACTION')
        for v in (action.numerator,action.denominator): _integer(v,1)
        _integer(action.cash_cents,0)
        if action.kind not in ('SPLIT','DIVIDEND'): raise ValueError('UNSUPPORTED_ACTION')
        if action.kind=='SPLIT' and action.cash_cents or action.kind=='DIVIDEND' and (action.numerator,action.denominator)!=(1,1):
            raise ValueError('ACTION_SEMANTICS')
        if action.event_id in result and result[action.event_id]!=action: raise ValueError('CONFLICTING_ACTION')
        result[action.event_id]=action
    return result

def reconcile(journal_state, broker_observations, account_snapshot, policy):
    l,o,a,p=journal_state,broker_observations,account_snapshot,policy
    try:
        if (type(l),type(o),type(a),type(p))!=(Ledger,Observations,Account,Policy): raise ValueError('INVALID_CONTRACT')
        _digest(p.policy_digest); _integer(p.cash_tolerance_cents,0); _integer(p.fee_tolerance_cents,0)
        if any(type(v) is not str or not v for v in (l.account,o.account,a.account)) or l.account!=o.account or l.account!=a.account: raise ValueError('ACCOUNT_MISMATCH')
        if type(l.mode) is not str or type(o.mode) is not str or type(a.mode) is not str or l.mode not in ('observed','hypothetical') or l.mode!=o.mode or l.mode!=a.mode: raise ValueError('MODE_MISMATCH')
        for v in (l.initial_cash_cents,l.peak_equity_cents,a.total_cash_cents,a.settled_cash_cents,a.equity_cents): _integer(v,0)
        for v in (l.external_cashflow_cents,a.external_cashflow_cents): _integer(v)
        if any(type(v) is not bool for v in (o.complete,o.timed_out,a.complete)): raise ValueError('INVALID_COMPLETENESS')
        for values in (l.known_orders,o.unresolved_orders):
            if type(values) is not tuple or any(type(v) is not str or not v for v in values) or len(set(values))!=len(values): raise ValueError('INVALID_IDENTITIES')
        pos=_positions(l.initial_positions); actual=_positions(a.positions)
        marks=_positions(a.valuation_marks_micros)
        if set(marks)!=set(actual): raise ValueError('INCOMPLETE_VALUATIONS')
        if a.equity_cents != a.total_cash_cents + sum(actual[name]*marks[name] for name in actual)//10000:
            raise ValueError('EQUITY_DISCREPANCY')
        expected=_fills(l.expected_fills); observed=_fills(o.fills)
        expected_actions=_actions(l.expected_corporate_actions); actions=_actions(o.corporate_actions)
        if expected_actions!=actions: raise ValueError('ACTION_DISCREPANCY')
        corporate_cash=0; split_instruments=set()
        for action in actions.values():
            if action.instrument not in pos: raise ValueError('UNKNOWN_ACTION_HOLDING')
            if action.kind=='DIVIDEND':
                corporate_cash+=action.cash_cents
            else:
                if action.instrument in split_instruments or any(f.instrument==action.instrument for f in observed.values()):
                    raise ValueError('AMBIGUOUS_ACTION_ORDERING')
                split_instruments.add(action.instrument)
                shares=pos[action.instrument]*action.numerator
                if shares%action.denominator: raise ValueError('FRACTIONAL_SPLIT_REQUIRES_RESOLUTION')
                pos[action.instrument]=shares//action.denominator
        if any(f.order_id not in l.known_orders for f in (*observed.values(),*expected.values())): raise ValueError('UNKNOWN_ORDER')
        if a.external_cashflow_cents!=l.external_cashflow_cents: raise ValueError('CASHFLOW_MISMATCH')
        if not o.complete or not a.complete or o.timed_out or o.unresolved_orders: return ReconciliationDecision('PENDING','INCOMPLETE_OBSERVATIONS')
        if set(expected)-set(observed): return ReconciliationDecision('PENDING','UNOBSERVED_FILL')
        if set(observed)-set(expected): raise ValueError('UNJOURNALED_FILL')
        for key,f in observed.items():
            e=expected[key]
            if (f.order_id,f.instrument,f.quantity,f.price_cents,f.correction_of)!=(e.order_id,e.instrument,e.quantity,e.price_cents,e.correction_of) or abs(f.fee_cents-e.fee_cents)>p.fee_tolerance_cents: raise ValueError('FILL_DISCREPANCY')
        cash=l.initial_cash_cents+l.external_cashflow_cents+corporate_cash; turnover=fees=0
        for f in _effective_fills(observed).values():
            pos[f.instrument]=pos.get(f.instrument,0)+f.quantity
            cash-=f.quantity*f.price_cents+f.fee_cents
            turnover+=abs(f.quantity)*f.price_cents; fees+=f.fee_cents
        if cash<0 or any(q<0 for q in pos.values()): raise ValueError('NEGATIVE_BALANCE')
        if pos!=actual: raise ValueError('POSITION_DISCREPANCY')
        if abs(cash-a.total_cash_cents)>p.cash_tolerance_cents: raise ValueError('CASH_DISCREPANCY')
        if a.settled_cash_cents>a.total_cash_cents: raise ValueError('INVALID_SETTLED_CASH')
        if a.settled_cash_cents!=a.total_cash_cents: return ReconciliationDecision('PENDING','UNSETTLED_CASH')
        peak=max(l.peak_equity_cents,a.equity_cents-l.external_cashflow_cents)
        return ReconciliationDecision('MATCHED','COMPLETE_NORMALIZED_MATCH',CostReceipt(turnover,fees,l.mode),peak)
    except (ValueError,TypeError,AttributeError):
        return ReconciliationDecision('HALTED','INVALID_OR_UNRESOLVED_NORMALIZED_FACTS')
