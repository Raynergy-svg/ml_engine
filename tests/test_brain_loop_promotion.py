"""No-mock tests for src.brain_loop.promotion.

Load-bearing guarantee: promotion NEVER auto-approves live exposure. Shadow
promotion (no capital, no ARM) may auto-approve; live promotion always lands
as PENDING_OPERATOR and this module never mutates SHIP_GATE.json or touches
anything ARM-gated.
"""
from __future__ import annotations

import json

import pytest

from src.brain_loop.promotion import PromotionRefused, list_pending, propose_promotion

_PASS_DECISION = {"decision": "continue", "reasons": [], "actionable": True}
_FAIL_DECISION = {"decision": "no_act", "reasons": ["ship_gate_missing"], "actionable": False}


def test_shadow_promotion_auto_approves(tmp_path):
    requests_dir = tmp_path / "promotion_requests"
    path = propose_promotion(
        requests_dir, hypothesis_id="h-1", decision=_PASS_DECISION, target="shadow"
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "APPROVED_SHADOW"
    assert payload["target"] == "shadow"
    assert payload["hypothesis_id"] == "h-1"


def test_live_promotion_is_always_pending_operator(tmp_path):
    requests_dir = tmp_path / "promotion_requests"
    path = propose_promotion(
        requests_dir, hypothesis_id="h-1", decision=_PASS_DECISION, target="live"
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "PENDING_OPERATOR"
    assert payload["requires_operator_arm"] is True


def test_propose_promotion_refuses_when_gate_did_not_continue(tmp_path):
    requests_dir = tmp_path / "promotion_requests"
    with pytest.raises(PromotionRefused):
        propose_promotion(
            requests_dir, hypothesis_id="h-1", decision=_FAIL_DECISION, target="shadow"
        )
    with pytest.raises(PromotionRefused):
        propose_promotion(
            requests_dir, hypothesis_id="h-1", decision=_FAIL_DECISION, target="live"
        )


def test_propose_promotion_rejects_unknown_target(tmp_path):
    requests_dir = tmp_path / "promotion_requests"
    with pytest.raises(ValueError):
        propose_promotion(
            requests_dir, hypothesis_id="h-1", decision=_PASS_DECISION, target="production"
        )


def test_list_pending_returns_only_pending_operator_entries(tmp_path):
    requests_dir = tmp_path / "promotion_requests"
    propose_promotion(requests_dir, hypothesis_id="h-1", decision=_PASS_DECISION, target="shadow")
    propose_promotion(requests_dir, hypothesis_id="h-2", decision=_PASS_DECISION, target="live")

    pending = list_pending(requests_dir)
    assert [p["hypothesis_id"] for p in pending] == ["h-2"]
    assert pending[0]["status"] == "PENDING_OPERATOR"


def test_list_pending_on_missing_dir_returns_empty(tmp_path):
    assert list_pending(tmp_path / "does_not_exist") == []


def test_promotion_never_touches_ship_gate_file(tmp_path):
    """propose_promotion must not read or write SHIP_GATE.json — the ship gate
    is produced by the harness, never forged by the promotion proposal step."""
    sg_dir = tmp_path / "trained_data" / "backtests"
    sg_dir.mkdir(parents=True)
    sg_path = sg_dir / "SHIP_GATE.json"
    original = json.dumps({"gate_pass": True, "universe_hash": "abc"})
    sg_path.write_text(original)

    requests_dir = tmp_path / "promotion_requests"
    propose_promotion(requests_dir, hypothesis_id="h-1", decision=_PASS_DECISION, target="live")

    assert sg_path.read_text() == original
