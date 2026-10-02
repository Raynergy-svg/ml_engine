import pandas as pd

def test_purged_walk_forward_removes_overlap_and_embargoes_after_test():
    from src.axiom2.research.splits import purged_walk_forward_splits
    dates=pd.date_range("2025-01-01",periods=20,tz="UTC")
    splits=purged_walk_forward_splits(dates,train_size=8,test_size=4,horizon=3,embargo=2)
    assert splits
    first=splits[0]
    assert max(first.train)<min(first.test)-pd.Timedelta(days=2)
    if len(splits)>1: assert min(splits[1].test)>max(first.test)+pd.Timedelta(days=2)


def test_split_manifests_are_deterministic_and_have_no_train_label_overlap():
    from src.axiom2.research.splits import purged_walk_forward_splits
    dates=pd.date_range("2025-01-01",periods=30,tz="UTC")
    a=purged_walk_forward_splits(dates,train_size=10,test_size=5,horizon=4,embargo=2)
    b=purged_walk_forward_splits(list(reversed(dates)),train_size=10,test_size=5,horizon=4,embargo=2)
    assert a==b
    for split in a:
        assert max(split.train)+pd.Timedelta(days=4)<min(split.test)+pd.Timedelta(days=1)
