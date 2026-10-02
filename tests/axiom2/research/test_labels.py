import pandas as pd
import pytest

def prices():
    idx=pd.date_range("2025-01-01",periods=8,tz="UTC"); mi=pd.MultiIndex.from_product([idx,["A","B"]],names=["timestamp","instrument_id"])
    return pd.Series([100+i+(0 if s=="A" else i) for i,d in enumerate(idx) for s in ("A","B")],index=mi,dtype=float)

def test_forward_labels_use_future_only_and_align_benchmark():
    from src.axiom2.research.labels import build_forward_rank_labels
    p=prices(); idx=p.index.levels[0]; b=pd.Series(range(100,108),index=idx,dtype=float)
    out=build_forward_rank_labels(p,b,horizon=2)
    assert out.index.equals(p.index)
    assert out.loc[(idx[0],"A"),"forward_return"]==pytest.approx(p.loc[(idx[2],"A")]/p.loc[(idx[0],"A")]-1)
    assert pd.isna(out.loc[(idx[-1],"A"),"forward_return"])
    with pytest.raises(ValueError,match="benchmark alignment"): build_forward_rank_labels(p,b.iloc[:-1],horizon=2)


def test_future_price_mutation_cannot_change_earlier_labels():
    from src.axiom2.research.labels import build_forward_rank_labels
    p=prices(); idx=p.index.levels[0]; b=pd.Series(range(100,108),index=idx,dtype=float)
    original=build_forward_rank_labels(p,b,horizon=2)
    mutated=p.copy(); mutated.loc[(idx[-1],"A")]=99999
    changed=build_forward_rank_labels(mutated,b,horizon=2)
    cutoff=idx[-4]
    pd.testing.assert_frame_equal(original.loc[:cutoff],changed.loc[:cutoff])
