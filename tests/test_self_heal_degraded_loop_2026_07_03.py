"""Degraded self-heal loop fix (2026-07-03) — no-mock tests.

The live supervisor reported `self_heal=degraded actions=2` every 30s forever.
Three defects, each covered here:
1. prose advice fed to the action executor (unknown_action_string every tick)
   → prose now travels in `recommendations`, never `recommended_actions`;
2. frozen evidence (halted ⇒ journal tail never changes) re-fired the same
   prescription after every debounce window → evidence-fingerprint suppression;
3. a tick of pure suppressions reported status="degraded" → now "suppressed".

No unittest.mock / no patching: real ScannerConfig, real SelfHeal against real
disk under tmp cwd (SelfHeal's state paths are cwd-relative), real
PostTradeDiagnostics run read-only against the repo.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scanner.config import ScannerConfig  # noqa: E402
from src.scanner.feedback.diagnostics import PostTradeDiagnostics  # noqa: E402
from src.scanner.feedback.self_heal import SelfHeal  # noqa: E402
from src.scanner.automation.maintenance_report import build_maintenance_report  # noqa: E402


def _mk_heal() -> SelfHeal:
    cfg = ScannerConfig()
    cfg.self_heal_max_autonomy_level = 5  # allow the L4 reduce_risk handler
    return SelfHeal(config=cfg)


def _diag(evidence_fp: str, recommendations=None) -> dict:
    return {
        "status": "DEGRADED",
        "issues": [{
            "check": "drawdown_streak", "severity": "critical",
            "detail": "7 consecutive losses at tail", "value": 7, "threshold": 5,
        }],
        "recommended_actions": ["reduce_risk_per_trade_pct"],
        "recommendations": recommendations or [],
        "evidence": {"reduce_risk_per_trade_pct": evidence_fp},
    }


# ---------------------------------------------------------------------------
# 1. Prose never reaches the executor (real diagnostics, read-only)
# ---------------------------------------------------------------------------

def test_diagnostics_actions_are_all_dispatchable_and_joint_free(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)  # diagnostics reads repo-relative artifacts
    result = PostTradeDiagnostics().run()
    actions = result["recommended_actions"]
    assert isinstance(result.get("recommendations"), list)
    assert isinstance(result.get("evidence"), dict)
    for action in actions:
        verb = action.partition(":")[0]
        assert " " not in verb, f"prose leaked into executable actions: {action!r}"
    blob = json.dumps(actions)
    assert "train-joint" not in blob, "deprecated joint retrain resurfaced as an action"
    # If the freshness recommendation fired, it must carry the L-016 caveat.
    for rec in result["recommendations"]:
        if "freshness" in rec.lower() or "STALE" in rec:
            assert "L-016" in rec


# ---------------------------------------------------------------------------
# 2. Evidence-anchored suppression (real SelfHeal on tmp cwd)
# ---------------------------------------------------------------------------

def test_first_fire_applies_then_window_then_evidence_suppression(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    heal = _mk_heal()

    r1 = heal.apply(_diag("streak=7@T1", recommendations=["advice X"]))
    assert r1["status"] == "applied"
    assert any(a["success"] for a in r1["actions_taken"])
    assert r1["recommendations"] == ["advice X"]

    # Immediate re-fire: inside the debounce window.
    r2 = heal.apply(_diag("streak=7@T1"))
    assert r2["status"] == "suppressed"
    details = [a["detail"] for a in r2["actions_taken"]]
    assert any(d.startswith(("debounced", "evidence_unchanged")) for d in details)
    assert not any(a["success"] for a in r2["actions_taken"])

    # Window expired but evidence UNCHANGED -> still suppressed (the fix).
    state_path = Path(".claude/self_heal_debounce.json")
    state = json.loads(state_path.read_text())
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    state["reduce_risk_per_trade_pct"]["ts"] = old_ts
    state_path.write_text(json.dumps(state))
    r3 = heal.apply(_diag("streak=7@T1"))
    assert r3["status"] == "suppressed"
    assert any(a["detail"].startswith("evidence_unchanged") for a in r3["actions_taken"])

    # NEW evidence (a new closed trade changed the tail) -> fires again.
    r4 = heal.apply(_diag("streak=8@T2"))
    assert r4["status"] == "applied"
    assert any(a["success"] for a in r4["actions_taken"])


def test_legacy_debounce_schema_still_read(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    heal = _mk_heal()
    # Legacy v1 schema: bare iso string, older than the window, no evidence.
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    p = Path(".claude")
    p.mkdir(parents=True, exist_ok=True)
    (p / "self_heal_debounce.json").write_text(
        json.dumps({"reduce_risk_per_trade_pct": old_ts}))
    r = heal.apply(_diag("streak=7@T1"))
    assert r["status"] == "applied"          # window expired, no stored evidence
    state = json.loads((p / "self_heal_debounce.json").read_text())
    assert state["reduce_risk_per_trade_pct"]["evidence"] == "streak=7@T1"  # upgraded to v2


# ---------------------------------------------------------------------------
# 3. Real failures still report degraded (we did not hide errors)
# ---------------------------------------------------------------------------

def test_unknown_action_still_degrades(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    heal = _mk_heal()
    diag = {
        "status": "DEGRADED",
        "issues": [{"check": "x", "severity": "warning", "detail": "d",
                    "value": 1, "threshold": 1}],
        "recommended_actions": ["Some prose that is not a verb"],
        "recommendations": [],
        "evidence": {},
    }
    r = heal.apply(diag)
    assert r["status"] == "degraded"
    assert r["actions_taken"][0]["detail"] == "unknown_action_string"


def test_recommendations_land_in_maintenance_report(tmp_path) -> None:
    report = build_maintenance_report(tmp_path, recommendations=["a", "", "b"])
    assert report["recommendations"] == ["a", "b"]
    # Informational only — never inflates needs_attention.
    assert report["needs_attention"] == []


def test_active_alerts_carry_a_route(tmp_path) -> None:
    claude = tmp_path / ".claude"
    claude.mkdir(parents=True)
    (claude / "alert_state.json").write_text(json.dumps({
        "active_alerts": [
            {"alert_type": "consecutive_losses", "severity": "WARNING",
             "message": "7 losses", "timestamp": "t", "acknowledged": False},
            {"alert_type": "brand_new_alert_kind", "severity": "WARNING",
             "message": "x", "timestamp": "t", "acknowledged": False},
        ],
    }))
    report = build_maintenance_report(tmp_path)
    routes = {a["type"]: a["route"] for a in report["alerts"]["active"]}
    assert "self_heal reduce_risk" in routes["consecutive_losses"]
    # Unknown alert types fail loud, never silently unowned.
    assert routes["brand_new_alert_kind"].startswith("UNROUTED")
