"""Tests for AuraQuery protocol and AuraQueryGate."""
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.aura.bridge.signals import (
    AuraQuery, AuraQueryResponse, FeedbackBridge,
)
from src.aura.bridge.query_gate import AuraQueryGate


@pytest.fixture
def tmp_bridge(tmp_path):
    bridge_dir = tmp_path / ".aura" / "bridge"
    bridge_dir.mkdir(parents=True)
    return FeedbackBridge(bridge_dir=bridge_dir)


def make_query(bridge_dir=None) -> AuraQuery:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    return AuraQuery(
        query_id=uuid.uuid4().hex,
        query_type="readiness_check",
        context={"pair": "EUR_USD", "confidence": 0.55},
        timestamp=now.isoformat(),
        expires_at=(now + timedelta(seconds=30)).isoformat(),
    )


def test_submit_and_read_query(tmp_bridge):
    """Test basic query submission and retrieval."""
    q = make_query()
    tmp_bridge.submit_query(q)
    pending = tmp_bridge.get_pending_queries()
    assert len(pending) == 1
    assert pending[0].query_id == q.query_id
    assert pending[0].query_type == "readiness_check"


def test_response_roundtrip(tmp_bridge):
    """Test query response submission and reading."""
    q = make_query()
    tmp_bridge.submit_query(q)
    resp = AuraQueryResponse(
        query_id=q.query_id,
        proceed=True,
        readiness_score=0.75,
        confidence_modifier=0.05,
        message="All clear",
    )
    tmp_bridge.submit_response(resp)
    result = tmp_bridge.read_response(q.query_id)
    assert result is not None
    assert result.proceed is True
    assert abs(result.readiness_score - 0.75) < 0.01
    assert abs(result.confidence_modifier - 0.05) < 0.01


def test_wait_for_response_timeout(tmp_bridge):
    """Test timeout when Aura doesn't respond in time."""
    q = make_query()
    tmp_bridge.submit_query(q)
    start = time.monotonic()
    result = tmp_bridge.wait_for_response(q.query_id, timeout_sec=0.6)
    elapsed = time.monotonic() - start
    assert result is None
    assert elapsed >= 0.5


def test_gate_passes_when_aura_offline(tmp_path):
    """Test gate passes (non-blocking) when Aura is offline."""
    bridge_dir = tmp_path / ".aura" / "bridge"
    bridge_dir.mkdir(parents=True)
    bridge = FeedbackBridge(bridge_dir=bridge_dir)
    gate = AuraQueryGate(bridge=bridge, timeout_sec=1.0)
    trade_ctx = {
        "pair": "EUR_USD",
        "confidence": 0.50,  # below threshold → would query
        "regime": "NORMAL",
        "weighted_vote_score": 0.60,
        "portfolio_risk_pct": 0.05,
        "drawdown_today_pct": 0.02,
    }
    result = gate.evaluate(trade_ctx)
    # Aura offline (no readiness file) → gate passes
    assert result["gate_passed"] is True
    assert result["aura_available"] is False


def test_gate_hard_veto_blocks_trade(tmp_bridge):
    """Test that Aura's hard_veto blocks the trade."""
    gate = AuraQueryGate(bridge=tmp_bridge, timeout_sec=25.0)

    # Write readiness file so gate thinks Aura is online
    readiness_path = tmp_bridge.bridge_dir / "readiness_signal.json"
    readiness_path.write_text(json.dumps({"readiness_score": 0.1, "timestamp": "2026-01-01T00:00:00Z"}))

    def fake_submit(query: AuraQuery):
        # Write directly without recursion
        qdir = tmp_bridge.bridge_dir / "queries"
        qdir.mkdir(exist_ok=True)
        p = qdir / f"{query.query_id}.json"
        p.write_text(json.dumps(query.to_dict()))
        # Immediately write veto response
        rdir = tmp_bridge.bridge_dir / "responses"
        rdir.mkdir(exist_ok=True)
        resp = AuraQueryResponse(
            query_id=query.query_id,
            proceed=False,
            readiness_score=0.1,
            confidence_modifier=-0.3,
            message="Critical stress detected",
            hard_veto=True,
        )
        rp = rdir / f"{query.query_id}.json"
        rp.write_text(json.dumps(resp.to_dict()))

    with patch.object(tmp_bridge, 'submit_query', side_effect=fake_submit):
        trade_ctx = {
            "pair": "EUR_USD",
            "confidence": 0.50,
            "regime": "NORMAL",
            "weighted_vote_score": 0.60,
            "portfolio_risk_pct": 0.05,
            "drawdown_today_pct": 0.02,
        }
        result = gate.evaluate(trade_ctx)

    assert result["hard_veto"] is True
    assert result["gate_passed"] is False


def test_gate_confidence_modifier_returned(tmp_bridge):
    """Test that confidence modifier is applied correctly."""
    # Write readiness to make Aura appear online
    readiness_path = tmp_bridge.bridge_dir / "readiness_signal.json"
    readiness_path.write_text(json.dumps({"readiness_score": 0.65, "timestamp": "2026-01-01T00:00:00Z"}))

    def fake_submit(query: AuraQuery):
        # Immediately write response with modifier
        rdir = tmp_bridge.bridge_dir / "responses"
        rdir.mkdir(exist_ok=True)
        resp = AuraQueryResponse(
            query_id=query.query_id,
            proceed=False,
            readiness_score=0.4,
            confidence_modifier=-0.15,
            message="Mildly stressed",
        )
        (rdir / f"{query.query_id}.json").write_text(json.dumps(resp.to_dict()))

    gate = AuraQueryGate(bridge=tmp_bridge, timeout_sec=25.0)
    with patch.object(tmp_bridge, 'submit_query', side_effect=fake_submit):
        result = gate.evaluate({
            "pair": "GBP_USD", "confidence": 0.55,
            "weighted_vote_score": 0.60, "portfolio_risk_pct": 0.05,
            "drawdown_today_pct": 0.02, "regime": "NORMAL",
        })

    assert abs(result["confidence_modifier"] - (-0.15)) < 0.01
    assert result["gate_passed"] is True  # not a hard veto


def test_stale_query_cleanup(tmp_bridge):
    """Test that stale queries are cleaned up."""
    import os
    q = make_query()
    tmp_bridge.submit_query(q)
    # Backdate the file's mtime to be older than max_age_sec
    qpath = tmp_bridge.bridge_dir / "queries" / f"{q.query_id}.json"
    old_time = time.time() - 200
    os.utime(str(qpath), (old_time, old_time))
    removed = tmp_bridge.cleanup_stale_queries(max_age_sec=120.0)
    assert removed >= 1
    assert not qpath.exists()


def test_should_query_triggers_on_low_confidence():
    """Test query trigger conditions - low confidence."""
    gate = AuraQueryGate(timeout_sec=1.0)
    assert gate.should_query({
        "confidence": 0.50, "weighted_vote_score": 0.70,
        "portfolio_risk_pct": 0.05, "drawdown_today_pct": 0.01,
        "regime": "NORMAL"
    }) is True


def test_should_query_false_on_strong_signal():
    """Test query trigger conditions - strong signal."""
    gate = AuraQueryGate(timeout_sec=1.0)
    assert gate.should_query({
        "confidence": 0.80, "weighted_vote_score": 0.75,
        "portfolio_risk_pct": 0.04, "drawdown_today_pct": 0.01,
        "regime": "NORMAL"
    }) is False


def test_query_id_uniqueness():
    """Test that query IDs are unique."""
    from datetime import datetime, timedelta, timezone
    ids = set()
    now = datetime.now(timezone.utc)
    for _ in range(20):
        q = AuraQuery(
            query_id=uuid.uuid4().hex,
            query_type="readiness_check",
            context={},
            timestamp=now.isoformat(),
            expires_at=(now + timedelta(seconds=30)).isoformat(),
        )
        ids.add(q.query_id)
    assert len(ids) == 20


def test_confidence_modifier_clamping():
    """Test that confidence modifier is properly clamped."""
    resp = AuraQueryResponse(
        query_id="test-id",
        proceed=True,
        readiness_score=0.75,
        confidence_modifier=0.5,  # Will be clamped to 0.1
        message="Test",
    )
    resp_dict = resp.to_dict()
    assert resp_dict["confidence_modifier"] <= 0.1
    assert resp_dict["confidence_modifier"] >= -0.3
