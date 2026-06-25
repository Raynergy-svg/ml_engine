"""PIT quality-panel builder — lookahead-proof construction (offline; running:NO).

The load-bearing guarantee: a quarterly fundamental for the period ENDING on date
P only influences the panel on/after its public-availability date (P + lag_days) —
never at P itself (that would be announcement-lag lookahead, the L-018 lie vector).
Tests use a real JSON dump on tmp_path (no mocks) + the real builder/validator.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.equity.quality_data import (
    DEFAULT_LAG_DAYS,
    availability_date,
    build_quality_panel,
    load_metrics_dump,
    panel_coverage,
    run_quality_bakeoff,
    validate_panel_pit,
)


def _dump(tmp_path: Path) -> Path:
    """Two tickers, one quarterly filing each, period ending 2008-09-30."""
    data = {
        "AAA": [{"report_period": "2008-09-30", "gross_margin": 0.50,
                 "net_margin": 0.25, "debt_to_equity": 0.5}],
        "BBB": [{"report_period": "2008-09-30", "gross_margin": 0.20,
                 "net_margin": 0.05, "debt_to_equity": 3.0}],
    }
    p = tmp_path / "quality_metrics.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _grid():
    return pd.DatetimeIndex(pd.bdate_range("2008-01-01", "2009-06-30", tz="UTC")).normalize()


def test_availability_lag_is_conservative():
    av = availability_date("2008-09-30")
    assert av == pd.Timestamp("2008-09-30", tz="UTC") + pd.Timedelta(days=DEFAULT_LAG_DAYS)
    assert av == pd.Timestamp("2008-12-29", tz="UTC")  # period end + 90d


def test_no_score_before_filing_is_public(tmp_path):
    dump = load_metrics_dump(_dump(tmp_path))
    idx = _grid()
    panel = build_quality_panel(dump, idx, ["AAA", "BBB"], lag_days=DEFAULT_LAG_DAYS)
    cutoff = pd.Timestamp("2008-12-29", tz="UTC")
    # Strictly NaN before the filing could have been known...
    assert panel.loc[panel.index < cutoff].isna().all().all()
    # ...and present on/after the availability date (forward-filled).
    assert panel.loc[panel.index >= cutoff].notna().all().all()


def test_quality_ordering_is_correct(tmp_path):
    # AAA: high margin, low leverage -> higher quality than BBB on every component.
    dump = load_metrics_dump(_dump(tmp_path))
    idx = _grid()
    panel = build_quality_panel(dump, idx, ["AAA", "BBB"], lag_days=DEFAULT_LAG_DAYS)
    last = panel.iloc[-1]
    assert last["AAA"] > last["BBB"]


def test_validator_passes_on_clean_panel(tmp_path):
    dump = load_metrics_dump(_dump(tmp_path))
    panel = build_quality_panel(dump, _grid(), ["AAA", "BBB"])
    validate_panel_pit(panel, dump)  # must not raise


def test_validator_catches_planted_lookahead(tmp_path):
    dump = load_metrics_dump(_dump(tmp_path))
    idx = _grid()
    panel = build_quality_panel(dump, idx, ["AAA", "BBB"])
    # Inject a score BEFORE the filing was public (the bug indexing by report_period).
    panel.loc[pd.Timestamp("2008-09-30", tz="UTC"), "AAA"] = 1.0
    with pytest.raises(ValueError, match="PIT violation"):
        validate_panel_pit(panel, dump)


def test_missing_filing_is_nan_not_zero(tmp_path):
    # A ticker with no record must be NaN (never a fabricated 0 score).
    dump = load_metrics_dump(_dump(tmp_path))
    idx = _grid()
    panel = build_quality_panel(dump, idx, ["AAA", "BBB", "CCC"])
    assert panel["CCC"].isna().all()


def test_coverage_reflects_active_cells(tmp_path):
    dump = load_metrics_dump(_dump(tmp_path))
    idx = _grid()
    members = ["AAA", "BBB"]
    panel = build_quality_panel(dump, idx, members)
    membership = pd.DataFrame(True, index=idx, columns=members)
    cov = panel_coverage(panel, membership)
    # ~half the window is pre-availability (NaN), so coverage is well below 1.0.
    assert 0.3 < cov < 0.7


def test_corrupt_dump_yields_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_metrics_dump(p) == {}


def test_bakeoff_refuses_when_no_dump(tmp_path):
    # The honest fail-closed default: no purchased data on disk -> NOT_EVALUATED,
    # never a fabricated quality result (L-018). Does NOT hit the network.
    res = run_quality_bakeoff(dump_path=tmp_path / "absent.json",
                              out_path=tmp_path / "out.json")
    assert res["status"] == "NOT_EVALUATED"
    assert "not yet sourced" in res["reason"]
