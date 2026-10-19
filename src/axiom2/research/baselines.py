"""Cost-aware naive Phase-1 baselines."""
from dataclasses import dataclass
import math
import pandas as pd

@dataclass(frozen=True)
class CostAssumption:
    evidence_id: str
    round_trip_bps: float
    def __post_init__(self):
        if not isinstance(self.evidence_id,str) or not self.evidence_id.strip():
            raise ValueError("valid cost evidence is required")
        if isinstance(self.round_trip_bps,bool) or not isinstance(self.round_trip_bps,(int,float)):
            raise TypeError("round_trip_bps must be numeric")
        if not math.isfinite(float(self.round_trip_bps)):
            raise ValueError("round_trip_bps must be finite")
        if self.round_trip_bps<0:
            raise ValueError("round_trip_bps must be nonnegative")

@dataclass(frozen=True)
class BaselineResult:
    gross_return: float
    net_return: float
    turnover: float
    cost_evidence_id: str

def _turnover_to_equal(previous_weights, current_ids):
    current=tuple(current_ids)
    target={identifier:1.0/len(current) for identifier in current}
    keys=set(previous_weights)|set(target)
    return .5*sum(abs(target.get(key,0.0)-previous_weights.get(key,0.0)) for key in keys)

def _post_return_weights(ids, returns):
    values={identifier:(1.0+float(ret))/len(ids) for identifier,ret in zip(ids,returns)}
    total=sum(values.values())
    if total<=0:
        return {}
    return {identifier:value/total for identifier,value in values.items()}

def evaluate_momentum_baseline(
    features:pd.DataFrame, labels:pd.DataFrame, *, costs:CostAssumption,
    top_n:int=1, evaluation_dates:tuple[pd.Timestamp,...]|None=None,
)->BaselineResult:
    if type(top_n) is not int or top_n<=0:
        raise ValueError("top_n must be positive")
    required={"forward_excess_return","forward_return"}
    if not required.issubset(labels.columns):
        raise ValueError("baseline labels require forward_return for actual weight-drift turnover")
    joined=features[["return_5"]].join(labels[sorted(required)],how="inner").dropna()
    if evaluation_dates is not None:
        allowed=frozenset(pd.DatetimeIndex(evaluation_dates))
        joined=joined[joined.index.get_level_values("timestamp").isin(allowed)]
    gross=[]
    turnovers=[]
    previous_weights={}
    for _,group in joined.groupby(level="timestamp",sort=True):
        if len(group)<top_n:
            raise ValueError("baseline top_n cohort is incomplete")
        pick=group.nlargest(top_n,"return_5")
        ids=tuple(pick.index.get_level_values("instrument_id"))
        turnovers.append(_turnover_to_equal(previous_weights,ids))
        gross.append(float(pick["forward_excess_return"].mean()))
        if "forward_return" in pick.columns:
            previous_weights=_post_return_weights(ids,pick["forward_return"].to_numpy())
        else:
            previous_weights={identifier:1.0/top_n for identifier in ids}
    if not gross:
        raise ValueError("baseline has no evaluable observations")
    gross_return=float(sum(gross)/len(gross))
    avg_turnover=float(sum(turnovers)/len(turnovers))
    net=gross_return-(float(costs.round_trip_bps)/10000.0)*avg_turnover
    return BaselineResult(gross_return,net,avg_turnover,costs.evidence_id)
