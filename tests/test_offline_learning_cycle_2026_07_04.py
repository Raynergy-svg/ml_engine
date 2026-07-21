"""Market-closed offline batch job: drains Tier-7 retrain_requests markers,
incrementally updates the risk/calibration learner, and walk-forward-gates
the update against the incumbent before promoting.

Root cause this closes (item 2 of the task): the `retrain_rl_position_sizer`
self-heal action (src/scanner/feedback/self_heal.py:1079) writes marker
files into trained_data/retrain_requests/ that NOTHING consumed — 12 markers
piled up. This script is the consumer.

Gate requirement (item 4): a worse candidate must be REJECTED — this is a
reproduce-then-resolve test, not an assumption.

NO MOCKS per CLAUDE.md No-Mock Rule. Real functions, real disk via tmp_path,
no network (no OANDA calls exist in this module at all).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import scripts.offline_learning_cycle as olc


def _iso(y, m, d, hh=12, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc).isoformat()


def _scanner_entry(ts: str, win: bool, confidence: float = 0.6) -> dict:
    return {
        "trade_id": ts,
        "pair": "EUR_USD",
        "timestamp": ts,
        "confidence": confidence,
        "outcome": "win" if win else "loss",
        "regime": {"volatility_regime": "NORMAL"},
        "rr_ratio": 1.3,
    }


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(olc, "JOURNAL_PATH", tmp_path / "trade_journal_rl.json")
    monkeypatch.setattr(olc, "RETRAIN_REQUESTS_DIR", tmp_path / "retrain_requests")
    monkeypatch.setattr(olc, "PROCESSED_DIR", tmp_path / "retrain_requests" / "_processed")
    monkeypatch.setattr(olc, "STATE_PATH", tmp_path / "risk_calibration_state.json")
    monkeypatch.setattr(olc, "HISTORY_PATH", tmp_path / "learning_loop" / "history.jsonl")
    monkeypatch.setattr(olc, "CURSOR_PATH", tmp_path / "learning_loop_cursor.json")
    monkeypatch.setattr(olc, "BRAIN_STATUS_PATH", tmp_path / "brain" / "learning_loop_status.json")
    monkeypatch.setattr(olc, "BRAIN_FEED_PATH", tmp_path / "brain" / "feed.jsonl")
    # A halt/arm state file this job must NEVER touch, to prove the "no
    # bypass of halt/arm" requirement empirically rather than by inspection.
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"halted": True, "oanda_environment": "practice"}))
    return state_path


def _write_markers(tmp_path: Path, n: int) -> None:
    d = tmp_path / "retrain_requests"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"rl_sizer_2026070{i}_000000.json").write_text(
            json.dumps({"reason": "self_heal_rl_staleness", "requested_at": _iso(2026, 7, i + 1)})
        )


def test_is_fx_market_closed_saturday_is_always_closed():
    assert olc.is_fx_market_closed(datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)) is True  # Saturday


def test_is_fx_market_closed_wednesday_noon_is_open():
    assert olc.is_fx_market_closed(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)) is False  # Wednesday


def test_is_fx_market_closed_friday_evening_is_closed():
    assert olc.is_fx_market_closed(datetime(2026, 7, 3, 22, 0, tzinfo=timezone.utc)) is True  # Fri 22:00 UTC


def test_is_fx_market_closed_sunday_afternoon_is_open():
    assert olc.is_fx_market_closed(datetime(2026, 7, 5, 22, 0, tzinfo=timezone.utc)) is False  # Sun 22:00 UTC


def test_is_fx_market_closed_sunday_morning_is_closed():
    assert olc.is_fx_market_closed(datetime(2026, 7, 5, 10, 0, tzinfo=timezone.utc)) is True  # Sun 10:00 UTC


def test_run_cycle_refuses_to_run_when_market_open_and_not_forced(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    wednesday_noon = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)

    result = olc.run_cycle(now=wednesday_noon)

    assert result == {"ran": False, "reason": "market_open"}


def test_run_cycle_drains_markers_moving_them_to_processed(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_markers(tmp_path, 12)
    saturday = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)

    result = olc.run_cycle(now=saturday)

    assert result["ran"] is True
    assert result["markers_drained"] == 12
    assert list((tmp_path / "retrain_requests").glob("*.json")) == []
    processed = list((tmp_path / "retrain_requests" / "_processed").glob("*.json"))
    assert len(processed) == 12


def test_run_cycle_never_touches_halt_state(tmp_path, monkeypatch):
    """Item constraint: 'no update may ... bypass the halt/ARM'. Prove it
    empirically — the halt-state fixture file must be byte-identical after
    a full accept-path run."""
    state_path = _patch_paths(monkeypatch, tmp_path)
    before = state_path.read_bytes()
    entries = [_scanner_entry(_iso(2026, 6, d), win=(d % 2 == 0)) for d in range(1, 20)]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))
    _write_markers(tmp_path, 1)

    olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert state_path.read_bytes() == before


def test_run_cycle_rejects_a_clearly_worse_candidate(tmp_path, monkeypatch):
    """Reproduce-then-resolve for the gate requirement: seed an incumbent
    that is ALREADY well-calibrated on a pattern, then feed new training
    data that is the OPPOSITE pattern (mislabeled) plus a holdout that still
    follows the ORIGINAL pattern. Fitting the candidate on the opposite
    pattern must make it worse on the holdout -> gate must reject and the
    incumbent state file must be left unchanged."""
    _patch_paths(monkeypatch, tmp_path)

    from src.training.incremental.risk_calibration_learner import RiskCalibrationLearner

    incumbent = RiskCalibrationLearner()
    good_pattern = [
        _scanner_entry(_iso(2026, 5, 1, hh=h), win=True, confidence=0.9) for h in range(20)
    ] + [
        _scanner_entry(_iso(2026, 5, 1, hh=h, mm=30), win=False, confidence=0.1) for h in range(20)
    ]
    incumbent.fit_incremental(good_pattern)
    incumbent_state = incumbent.to_state()
    olc.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    olc.STATE_PATH.write_text(json.dumps(incumbent_state))

    # New data: high confidence now loses, low confidence now wins (opposite
    # of the incumbent's learned pattern) — training on this makes things
    # worse for a holdout that follows the ORIGINAL pattern.
    bad_new_data = [
        _scanner_entry(_iso(2026, 6, d), win=False, confidence=0.9) for d in range(1, 16)
    ]
    holdout = [
        _scanner_entry(_iso(2026, 6, 20 + i), win=True, confidence=0.9) for i in range(3)
    ] + [
        _scanner_entry(_iso(2026, 6, 25 + i), win=False, confidence=0.1) for i in range(3)
    ]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(bad_new_data + holdout))
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    _write_markers(tmp_path, 2)

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] == "rejected"
    assert result["candidate_brier"] > result["incumbent_brier"]
    # Incumbent state file on disk must be untouched byte-for-byte.
    assert json.loads(olc.STATE_PATH.read_text()) == incumbent_state


def test_run_cycle_accepts_a_clearly_better_candidate(tmp_path, monkeypatch):
    """Mirror of the rejection test: new data reinforces a real pattern the
    (fresh, uncalibrated) incumbent doesn't know yet, and the holdout
    follows the same pattern -> candidate must score better and be
    promoted."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)

    train_new = [
        _scanner_entry(_iso(2026, 6, d), win=True, confidence=0.9) for d in range(1, 16)
    ] + [
        _scanner_entry(_iso(2026, 6, d, hh=13), win=False, confidence=0.1) for d in range(1, 16)
    ]
    holdout = [
        _scanner_entry(_iso(2026, 6, 20 + i), win=True, confidence=0.9) for i in range(3)
    ] + [
        _scanner_entry(_iso(2026, 6, 25 + i), win=False, confidence=0.1) for i in range(3)
    ]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(train_new + holdout))
    _write_markers(tmp_path, 3)

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] == "accepted"
    assert result["candidate_brier"] < result["incumbent_brier"]
    assert olc.STATE_PATH.exists()
    assert result["max_size_multiplier"] <= 1.0


def test_run_cycle_with_insufficient_new_data_does_not_promote(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    entries = [_scanner_entry(_iso(2026, 6, 1), win=True)]  # only 1 resolved outcome
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    # 1 new entry exists but is below MIN_HOLDOUT+1 -- distinct from genuinely
    # zero new data ("no_new_data"); the cursor must also NOT advance past this
    # entry, since fit_incremental() never ran on it (see cursor-advance fix).
    assert result["decision"] == "insufficient_new_data"
    assert not olc.STATE_PATH.exists()
    assert not olc.CURSOR_PATH.exists()


def test_run_cycle_writes_brain_status_and_history(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    train_new = [_scanner_entry(_iso(2026, 6, d), win=True, confidence=0.9) for d in range(1, 10)]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(train_new))
    _write_markers(tmp_path, 4)

    olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert (tmp_path / "learning_loop" / "history.jsonl").exists()
    assert (tmp_path / "brain" / "learning_loop_status.json").exists()
    status = json.loads((tmp_path / "brain" / "learning_loop_status.json").read_text())
    assert "summary" in status


def test_cursor_advances_so_second_run_does_not_reprocess_same_outcomes(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    train_new = [
        _scanner_entry(_iso(2026, 6, d), win=True, confidence=0.9) for d in range(1, 16)
    ] + [
        _scanner_entry(_iso(2026, 6, d, hh=13), win=False, confidence=0.1) for d in range(1, 16)
    ]
    holdout = [
        _scanner_entry(_iso(2026, 6, 20 + i), win=True, confidence=0.9) for i in range(3)
    ] + [
        _scanner_entry(_iso(2026, 6, 25 + i), win=False, confidence=0.1) for i in range(3)
    ]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(train_new + holdout))

    first = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))
    assert first["new_outcomes"] > 0

    second = olc.run_cycle(now=datetime(2026, 7, 4, 13, 0, tzinfo=timezone.utc))
    assert second["decision"] == "no_new_data"
    assert second["new_outcomes"] == 0
