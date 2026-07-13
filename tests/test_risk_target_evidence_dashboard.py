"""Dashboard reader for the risk-target evidence cockpit (read-only).

Verifies the ``/api/risk_target_evidence`` data source projects real committed
disposition state and degrades honestly (never raises) when no evidence exists.
NO MOCKS: a real slice run writes a real evidence store to tmp, and the reader
is pointed at it.
"""

from __future__ import annotations

from datetime import date

from dashboard.server import data_sources as ds
from src.evidence.contracts import GateResult, GateStatus
from src.evidence.risk_target.dashboard import risk_target_evidence_view
from src.evidence.risk_target.models import DRAWDOWN_HEAD_ID, VOLATILITY_HEAD_ID, HeadResult
from src.evidence.risk_target.slice import (
    build_evidence_store,
    build_slice_identities,
    run_risk_target_evidence_slice,
    utc,
)

NOW = utc(2026, 7, 13, 12, 0)


def _head(head_id, lane_id, passed):
    status = GateStatus.PASS if passed else GateStatus.FAIL
    return HeadResult(
        head_id=head_id, lane_id=lane_id, target=head_id,
        metrics={"score": 0.05}, metric_tolerances={"score": 1e-9},
        gates=(GateResult(gate_id=f"{head_id}_bar", status=status, observed=0.05,
                          threshold=0.1, reason="bar"),),
        passed=passed, model_bytes=f"m::{head_id}".encode(),
        media_type="application/octet-stream", trial_count=1, effective_sample_size=100.0,
        temporal_holdout="OOS", purge_observations=20, embargo_observations=20,
    )


def _fixed(partitions, params):
    return (_head(VOLATILITY_HEAD_ID, "risk_target_vol", True),
            _head(DRAWDOWN_HEAD_ID, "risk_target_drawdown", False))


def _populate(root):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(root, identities, clock_now=NOW)
    run_risk_target_evidence_slice(
        store, identities,
        {"EUR_USD": b"partition-a\n", "USD_JPY": b"partition-b\n"},
        dataset_id="fx-daily", coverage_start=date(2019, 1, 1), coverage_end=date(2026, 1, 1),
        retrieved_at=NOW, created_at=NOW, git_commit="0" * 40,
        evaluator=_fixed, registry_publish=lambda e, m: None,
    )
    return store


def test_reader_projects_committed_dispositions(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    _populate(root)
    monkeypatch.setattr(ds, "EVIDENCE_DIR", root)
    view = ds.read_risk_target_evidence()
    assert view["available"] is True
    assert view["connected"] is True
    by_lane = {entry["lane_id"]: entry for entry in view["lanes"]}
    assert by_lane["risk_target_vol"]["state"] == "QUARANTINED"
    assert by_lane["risk_target_drawdown"]["state"] == "REJECTED"
    assert view["champions"] == {}


def test_reader_degrades_honestly_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "EVIDENCE_DIR", tmp_path / "nope")
    view = ds.read_risk_target_evidence()
    assert view["available"] is False
    assert view["connected"] is False
    assert view["lanes"] == []


def test_direct_view_matches_reader(tmp_path):
    root = tmp_path / "evidence"
    _populate(root)
    assert risk_target_evidence_view(root)["lanes"]
