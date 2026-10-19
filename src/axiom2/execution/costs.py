"""Broker-neutral integer cost receipt; observed and hypothetical remain distinct."""
from dataclasses import dataclass
@dataclass(frozen=True)
class CostReceipt:
    turnover_cents: int
    fees_cents: int
    mode: str
    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.turnover_cents,self.fees_cents)) or self.mode not in ('observed','hypothetical'):
            raise ValueError('INVALID_COST')
