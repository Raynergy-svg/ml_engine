"""Purged walk-forward split manifests."""
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class WalkForwardSplit:
    train: tuple[pd.Timestamp,...]
    test: tuple[pd.Timestamp,...]

def purged_walk_forward_splits(dates, *, train_size:int,test_size:int,horizon:int,embargo:int):
    d=tuple(pd.DatetimeIndex(dates).sort_values().unique())
    if min(train_size,test_size,horizon)<=0 or embargo<0: raise ValueError("invalid split parameters")
    out=[]; test_start=train_size
    while test_start+test_size<=len(d):
        train_end=max(0,test_start-horizon); train_start=max(0,train_end-train_size)
        train=tuple(d[train_start:train_end]); test=tuple(d[test_start:test_start+test_size])
        if train: out.append(WalkForwardSplit(train,test))
        test_start += test_size+embargo
    return tuple(out)
