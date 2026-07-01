"""No-mock tests for the point-in-time equity universe builder (US-002).

Real classes, real disk via ``tmp_path``. The critical invariant under test:
**no future leak** — a known later-IPO ticker must NOT appear in membership on
dates before its first listing window completes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import EQUITY_PIPELINE_VERSION
from src.equity.universe import (
    UniverseError,
    UniverseRules,
    UniverseSnapshot,
    build_universe,
    compute_universe_hash,
)


def _synth_prices(
    start: str,
    end: str,
    close: float,
    daily_volume: float,
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end, tz="UTC")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, close),
            "high": np.full(n, close * 1.01),
            "low": np.full(n, close * 0.99),
            "close": np.full(n, close),
            "volume": np.full(n, daily_volume),
        },
        index=idx,
    )


def _wide_panel(top_count: int = 5):
    """5 long-history liquid names + 1 later-IPO ticker."""
    prices = {
        "AAA": _synth_prices("2015-01-01", "2026-06-01", 100.0, 5_000_000.0),
        "BBB": _synth_prices("2015-01-01", "2026-06-01", 100.0, 4_000_000.0),
        "CCC": _synth_prices("2015-01-01", "2026-06-01", 100.0, 3_000_000.0),
        "DDD": _synth_prices("2015-01-01", "2026-06-01", 100.0, 2_000_000.0),
        "EEE": _synth_prices("2015-01-01", "2026-06-01", 100.0, 1_500_000.0),
        # IPO'd 2024-01-02, so has < min_history_days listing on any 2020 date.
        "LATEIPO": _synth_prices(
            "2024-01-02", "2026-06-01", 100.0, 9_999_999_999.0
        ),
    }
    return prices, top_count


def test_build_universe_returns_snapshot_with_hash():
    prices, top_n = _wide_panel(top_count=3)
    snap = build_universe(prices, top_n=top_n)
    assert isinstance(snap, UniverseSnapshot)
    assert isinstance(snap.universe_hash, str)
    assert len(snap.universe_hash) == 64
    assert snap.pipeline_version == EQUITY_PIPELINE_VERSION
    assert snap.rules.top_n == top_n
    assert snap.rules.ranking_metric == "adv20_dollar"
    assert len(snap.reconstitution_dates) > 0


def test_no_future_leak_excludes_later_ipo_on_pre_ipo_date():
    """**The headline acceptance criterion.** On a 2020 date, the universe
    must NOT include LATEIPO (IPO 2024-01-02)."""
    prices, _ = _wide_panel(top_count=5)
    snap = build_universe(prices, top_n=5)

    pre_ipo = pd.Timestamp("2020-06-15", tz="UTC")
    members_pre = snap.members_on(pre_ipo)
    assert "LATEIPO" not in members_pre, (
        f"Future leak: LATEIPO must not appear on {pre_ipo.date()} "
        f"(IPO was 2024-01-02). Got {members_pre}."
    )
    # Sanity: liquid long-history names ARE there.
    assert "AAA" in members_pre
    assert "BBB" in members_pre


def test_top_n_respected_and_ranked_by_dollar_volume():
    prices, _ = _wide_panel(top_count=3)
    snap = build_universe(prices, top_n=3)
    post_warmup = pd.Timestamp("2017-01-15", tz="UTC")
    members = snap.members_on(post_warmup)
    assert len(members) == 3
    # AAA/BBB/CCC have the highest $-vol of the long-history names; LATEIPO is
    # still pre-IPO here.
    assert set(members) == {"AAA", "BBB", "CCC"}


def test_later_ipo_enters_universe_after_history_window():
    prices, _ = _wide_panel(top_count=5)
    snap = build_universe(prices, top_n=5, min_history_days=252)

    # LATEIPO IPO'd 2024-01-02; with 252 sessions of history + monthly recon
    # it should be visible by, say, early 2025. It has the highest $-vol so
    # it must enter once eligible.
    post_window = pd.Timestamp("2025-06-02", tz="UTC")
    members = snap.members_on(post_window)
    assert "LATEIPO" in members, (
        "LATEIPO should have entered the universe once "
        "min_history_days has elapsed."
    )


def test_universe_hash_is_deterministic():
    prices, _ = _wide_panel(top_count=3)
    a = build_universe(prices, top_n=3)
    b = build_universe(prices, top_n=3)
    assert a.universe_hash == b.universe_hash
    assert a.membership.equals(b.membership)


def test_universe_hash_changes_when_membership_changes():
    prices, _ = _wide_panel(top_count=3)
    a = build_universe(prices, top_n=3)
    b = build_universe(prices, top_n=5)
    assert a.universe_hash != b.universe_hash


def test_atomic_save_load_round_trip(tmp_path):
    prices, _ = _wide_panel(top_count=3)
    snap = build_universe(prices, top_n=3)
    out = tmp_path / "universe.json"
    snap.save(out)
    assert out.exists()
    # Atomic write: no leftover .tmp file.
    assert not list(tmp_path.glob("*.tmp"))

    loaded = UniverseSnapshot.load(out)
    assert loaded.universe_hash == snap.universe_hash
    assert loaded.rules.top_n == 3
    # Spot-check a date present in the saved snapshot.
    sample_date = pd.Timestamp("2020-06-15", tz="UTC")
    assert loaded.members_on(sample_date) == snap.members_on(sample_date)


def test_load_detects_hash_tampering(tmp_path):
    import json
    prices, _ = _wide_panel(top_count=3)
    snap = build_universe(prices, top_n=3)
    out = tmp_path / "universe.json"
    snap.save(out)
    payload = json.loads(out.read_text())
    payload["universe_hash"] = "0" * 64
    out.write_text(json.dumps(payload))
    with pytest.raises(UniverseError, match="hash mismatch"):
        UniverseSnapshot.load(out)


def test_build_rejects_empty_input():
    with pytest.raises(UniverseError, match="empty"):
        build_universe({})


def test_build_rejects_bad_top_n():
    prices, _ = _wide_panel(top_count=3)
    with pytest.raises(UniverseError, match="top_n"):
        build_universe(prices, top_n=0)


def test_build_rejects_missing_close_column():
    bad = _synth_prices("2015-01-01", "2026-06-01", 100.0, 1_000_000.0)
    bad = bad.drop(columns=["close"])
    with pytest.raises(UniverseError, match="close"):
        build_universe({"X": bad})


def test_rules_serialize_round_trip():
    rules = UniverseRules(top_n=50, lookback_days=30)
    d = rules.to_dict()
    rebuilt = UniverseRules(**d)
    assert rebuilt == rules


def test_compute_universe_hash_empty_frame_is_stable():
    a = compute_universe_hash(pd.DataFrame())
    b = compute_universe_hash(pd.DataFrame())
    assert a == b
    assert len(a) == 64


def test_members_on_before_history_returns_empty():
    prices, _ = _wide_panel(top_count=3)
    snap = build_universe(prices, top_n=3, min_history_days=252)
    # Day 5 of history is far before min_history_days; nobody qualifies yet.
    pre_warmup = pd.Timestamp("2015-01-09", tz="UTC")
    assert snap.members_on(pre_warmup) == []
