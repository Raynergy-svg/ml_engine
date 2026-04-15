"""Tests for model freshness lookup + losing-streak trigger.

The user reported "I'm losing 6+ trades in a row, buddy won't heal on his
own." Root cause was usually stale models — but Claude wasn't seeing the
training dates so couldn't recommend a retrain. These tests guard the new
plumbing that surfaces training metadata to Claude and adds a deep
losing-streak reflection trigger.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def tmp_models(tmp_path, monkeypatch):
    """Build a fake trained_data/models tree under tmp_path."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data" / "models" / "joint").mkdir(parents=True)
    return tmp_path


# ── Model freshness lookup ─────────────────────────────────────────────


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_freshness_fresh_when_under_3_days(tmp_models):
    """New thresholds for forex weekly cadence: FRESH ≤3d."""
    from src.scanner.automation.model_freshness import get_model_freshness

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(2)})
    )
    (tmp_models / "trained_data" / "models" / "joint" / "joint_training_meta.json").write_text(
        json.dumps({"trained_at": _iso(1)})
    )

    f = get_model_freshness()
    assert f["status"] == "FRESH"
    assert f["oldest_age_days"] is not None
    assert f["oldest_age_days"] < 3
    assert f["stale_models"] == []


def test_freshness_aging_at_4_days(tmp_models):
    """AGING band: 3-5d (warn but no auto-trigger)."""
    from src.scanner.automation.model_freshness import get_model_freshness

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(4)})
    )
    (tmp_models / "trained_data" / "models" / "joint" / "joint_training_meta.json").write_text(
        json.dumps({"trained_at": _iso(2)})
    )

    f = get_model_freshness()
    assert f["status"] == "AGING"
    assert f["stale_models"] == []  # AGING is below STALE_DAYS (5)


def test_freshness_stale_at_6_days(tmp_models):
    """STALE band: 5-7d. Now triggers auto-retrain by default (forex)."""
    from src.scanner.automation.model_freshness import get_model_freshness

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(6)})
    )
    (tmp_models / "trained_data" / "models" / "joint" / "joint_training_meta.json").write_text(
        json.dumps({"trained_at": _iso(1)})
    )

    f = get_model_freshness()
    assert f["status"] == "STALE"
    assert any("modular_ensemble" in s for s in f["stale_models"])


def test_freshness_critical_above_7_days(tmp_models):
    """User's exact failure mode — models 27d old (now CRITICAL)."""
    from src.scanner.automation.model_freshness import get_model_freshness

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(27)})
    )

    f = get_model_freshness()
    assert f["status"] == "CRITICAL"
    assert f["oldest_age_days"] > 7


def test_freshness_unknown_when_no_metadata(tmp_models):
    """Empty models dir → status UNKNOWN, doesn't crash."""
    from src.scanner.automation.model_freshness import get_model_freshness

    f = get_model_freshness()
    assert f["status"] == "UNKNOWN"
    assert f["oldest_age_days"] is None


def test_freshness_handles_corrupt_meta_json(tmp_models):
    from src.scanner.automation.model_freshness import get_model_freshness

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        "{not valid json"
    )

    # Should not crash
    f = get_model_freshness()
    # Either UNKNOWN or falls back to file mtime — both are acceptable
    assert f["status"] in {"UNKNOWN", "FRESH", "AGING", "STALE", "CRITICAL"}


def test_freshness_format_for_prompt():
    from src.scanner.automation.model_freshness import format_freshness_for_prompt

    text = format_freshness_for_prompt({
        "status": "STALE",
        "oldest_age_days": 27.4,
        "stale_models": ["modular_ensemble (27d)"],
        "groups": [
            {
                "name": "modular_ensemble",
                "trained_at": "2026-03-18T23:11:39+00:00",
                "age_days": 27.4,
            },
            {
                "name": "joint_gates",
                "trained_at": "2026-04-14T19:00:00+00:00",
                "age_days": 0.5,
            },
        ],
    })
    assert "STALE" in text
    assert "27.4" in text
    assert "modular_ensemble" in text
    assert "joint_gates" in text


# ── Diagnostics integration ────────────────────────────────────────────


def test_diagnostics_flags_stale_models_as_warning(tmp_models):
    """6 days = STALE band → warning issue."""
    from src.scanner.feedback.diagnostics import PostTradeDiagnostics

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(6)})
    )
    diag = PostTradeDiagnostics()
    result = diag.run()
    freshness_issues = [
        i for i in result.get("issues", [])
        if i.get("check") == "model_training_freshness"
    ]
    assert len(freshness_issues) == 1
    assert freshness_issues[0]["severity"] == "warning"


def test_diagnostics_flags_critical_age(tmp_models):
    """8+ days = CRITICAL → critical severity + retrain action."""
    from src.scanner.feedback.diagnostics import PostTradeDiagnostics

    (tmp_models / "trained_data" / "models" / "modular_ensemble.meta.json").write_text(
        json.dumps({"trained_at": _iso(10)})
    )
    result = PostTradeDiagnostics().run()
    freshness_issues = [
        i for i in result.get("issues", [])
        if i.get("check") == "model_training_freshness"
    ]
    assert len(freshness_issues) == 1
    assert freshness_issues[0]["severity"] == "critical"
    # Should recommend retraining
    assert any(
        "train-joint" in a or "retrain" in a.lower()
        for a in result.get("recommended_actions", [])
    )


# ── Losing-streak trigger ──────────────────────────────────────────────


def _trade(trade_id: str, won: bool, ts_offset_minutes: int = 0) -> dict:
    """Build a fake closed journal entry."""
    return {
        "trade_id": trade_id,
        "pair": "EUR_USD",
        "direction": "LONG",
        "confidence": 0.6,
        "timestamp": (
            datetime.now(timezone.utc) - timedelta(minutes=ts_offset_minutes)
        ).isoformat(),
        "regime": {"volatility_regime": "NORMAL"},
        "outcome": {
            "trade_won": won,
            "pnl_pips": 15.0 if won else -10.0,
            "exit_reason": "TP" if won else "SL",
            "close_time": (
                datetime.now(timezone.utc) - timedelta(minutes=ts_offset_minutes)
            ).isoformat(),
        },
    }


@pytest.fixture
def iso_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data").mkdir()
    (tmp_path / ".claude").mkdir()
    return tmp_path


def test_losing_streak_fires_at_threshold(iso_journal):
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    # 3 most recent trades all losses
    journal = [
        _trade("T-001", won=True, ts_offset_minutes=60),
        _trade("T-002", won=False, ts_offset_minutes=45),
        _trade("T-003", won=False, ts_offset_minutes=30),
        _trade("T-004", won=False, ts_offset_minutes=15),
    ]
    (iso_journal / "trained_data" / "trade_journal_rl.json").write_text(
        json.dumps(journal)
    )

    fired = []
    t = CycleAutonomyTriggers(
        brain_callback=lambda m: None,
        periodic_every_n=999,
        rejection_streak_threshold=999,
        losing_streak_threshold=3,
    )
    t._invoke = lambda prompt, trade_id, mode, timeout: fired.append(
        (trade_id, mode, prompt)
    )
    t._should_fire_self_heal = lambda: False

    from types import SimpleNamespace
    t.on_cycle_complete(
        scan_count=1,
        scan_result=SimpleNamespace(tradeable=[], analyses=[]),
        trades_executed=0,
        tradeable_count=0,
    )

    assert len(fired) == 1
    assert fired[0][0].startswith("LOSING-")
    assert fired[0][1] == "deep"  # deep mode for losing streak
    # Prompt must include MODEL_FRESHNESS so Claude can see staleness
    assert "MODEL_FRESHNESS" in fired[0][2] or "freshness" in fired[0][2].lower()


def test_losing_streak_requires_consecutive(iso_journal):
    """A win in the last N breaks the streak."""
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    journal = [
        _trade("T-001", won=False, ts_offset_minutes=60),
        _trade("T-002", won=True, ts_offset_minutes=45),  # win!
        _trade("T-003", won=False, ts_offset_minutes=30),
        _trade("T-004", won=False, ts_offset_minutes=15),
    ]
    (iso_journal / "trained_data" / "trade_journal_rl.json").write_text(
        json.dumps(journal)
    )

    fired = []
    t = CycleAutonomyTriggers(
        brain_callback=lambda m: None,
        periodic_every_n=999,
        rejection_streak_threshold=999,
        losing_streak_threshold=3,
    )
    t._invoke = lambda prompt, trade_id, mode, timeout: fired.append(trade_id)
    t._should_fire_self_heal = lambda: False

    from types import SimpleNamespace
    t.on_cycle_complete(
        scan_count=1,
        scan_result=SimpleNamespace(tradeable=[], analyses=[]),
        trades_executed=0,
        tradeable_count=0,
    )

    losing_calls = [c for c in fired if c.startswith("LOSING-")]
    assert len(losing_calls) == 0


def test_losing_streak_dedupes_until_new_trade(iso_journal):
    """Once fired on a streak, don't re-fire on the same trade IDs."""
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    journal = [
        _trade("T-001", won=False, ts_offset_minutes=45),
        _trade("T-002", won=False, ts_offset_minutes=30),
        _trade("T-003", won=False, ts_offset_minutes=15),
    ]
    (iso_journal / "trained_data" / "trade_journal_rl.json").write_text(
        json.dumps(journal)
    )

    fired = []
    t = CycleAutonomyTriggers(
        brain_callback=lambda m: None,
        periodic_every_n=999,
        rejection_streak_threshold=999,
        losing_streak_threshold=3,
    )
    t._invoke = lambda prompt, trade_id, mode, timeout: fired.append(trade_id)
    t._should_fire_self_heal = lambda: False

    from types import SimpleNamespace
    fake_scan = SimpleNamespace(tradeable=[], analyses=[])
    # Fire 3 cycles in a row — should only spawn once
    for c in range(1, 4):
        t.on_cycle_complete(
            scan_count=c,
            scan_result=fake_scan,
            trades_executed=0,
            tradeable_count=0,
        )

    losing_calls = [c for c in fired if c.startswith("LOSING-")]
    assert len(losing_calls) == 1


def test_losing_streak_priority_below_self_heal(iso_journal):
    """When BOTH self-heal and losing-streak conditions match, self-heal wins."""
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    journal = [
        _trade("T-A", won=False, ts_offset_minutes=45),
        _trade("T-B", won=False, ts_offset_minutes=30),
        _trade("T-C", won=False, ts_offset_minutes=15),
    ]
    (iso_journal / "trained_data" / "trade_journal_rl.json").write_text(
        json.dumps(journal)
    )

    fired = []
    t = CycleAutonomyTriggers(
        brain_callback=lambda m: None,
        periodic_every_n=999,
        rejection_streak_threshold=999,
        losing_streak_threshold=3,
    )
    t._invoke = lambda prompt, trade_id, mode, timeout: fired.append(trade_id)

    with patch("src.scanner.feedback.diagnostics.PostTradeDiagnostics") as MD:
        MD.return_value.run.return_value = {
            "status": "DEGRADED",
            "issues": [{"check": "x", "severity": "warning", "detail": "y"}],
        }
        from types import SimpleNamespace
        t.on_cycle_complete(
            scan_count=1,
            scan_result=SimpleNamespace(tradeable=[], analyses=[]),
            trades_executed=0,
            tradeable_count=0,
        )

    assert len(fired) == 1
    assert fired[0].startswith("SELFHEAL-"), f"expected self-heal to win, got {fired}"
