"""Adversarial-review fixes for the market-closed learning cycle (2026-07-04).

Reproduce-then-resolve tests for defects an independent adversarial review
found in commit 51b85bf:

- #1: the script crashed with ModuleNotFoundError when run by path (the
  documented + plist invocation) because it never inserted the repo root on
  sys.path.
- #2/#3/#8: the walk-forward gate was invalid on the REAL journal — 90% of
  entries are trend-lane bare-string outcomes with all-None risk features,
  so the holdout collapsed to a single repeated feature vector, and a
  single-class (all-win / all-loss) journal force-promoted a degenerate
  constant predictor. The learner must (a) only train/score entries that
  actually carry risk/calibration features, and (b) refuse to promote on a
  single-class holdout or a feature-degenerate training set.
- #9: the cursor used ``str > cursor``, permanently dropping equal-timestamp
  entries and mis-sorting ``Z`` vs ``+00:00`` suffixes.

NO MOCKS per CLAUDE.md No-Mock Rule. Real functions, real disk via tmp_path,
no network.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts.offline_learning_cycle as olc
from src.training.incremental.risk_calibration_learner import is_calibration_scoreable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _iso(y, m, d, hh=12, mm=0, ss=0, us=0):
    return datetime(y, m, d, hh, mm, ss, us, tzinfo=timezone.utc).isoformat()


def _scanner_entry(ts, win, confidence=0.6, tid=None):
    return {
        "trade_id": tid or ts,
        "pair": "EUR_USD",
        "timestamp": ts,
        "confidence": confidence,
        "outcome": "win" if win else "loss",
        "regime": {"volatility_regime": "NORMAL"},
        "rr_ratio": 1.3,
        "sl_pips": 12.0,
        "tp_pips": 16.0,
    }


def _trend_entry(ts, win, tid=None):
    """The real-journal shape: trend-lane, rl_eligible=False, no risk features."""
    return {
        "trade_id": tid or ts,
        "pair": "USD_JPY",
        "timestamp": ts,
        "lane": "trend",
        "rl_eligible": False,
        "outcome": "win" if win else "loss",
        "realized_pl": 10.0 if win else -10.0,
        "regime": None,
    }


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(olc, "JOURNAL_PATH", tmp_path / "trade_journal_rl.json")
    monkeypatch.setattr(olc, "RETRAIN_REQUESTS_DIR", tmp_path / "retrain_requests")
    monkeypatch.setattr(olc, "PROCESSED_DIR", tmp_path / "retrain_requests" / "_processed")
    monkeypatch.setattr(olc, "STATE_PATH", tmp_path / "risk_calibration_state.json")
    monkeypatch.setattr(olc, "HISTORY_PATH", tmp_path / "learning_loop" / "history.jsonl")
    monkeypatch.setattr(olc, "CURSOR_PATH", tmp_path / "learning_loop_cursor.json")
    monkeypatch.setattr(olc, "BRAIN_STATUS_PATH", tmp_path / "brain" / "status.json")
    monkeypatch.setattr(olc, "BRAIN_FEED_PATH", tmp_path / "brain" / "feed.jsonl")


# ── #1: runs by path without ModuleNotFoundError ──────────────────────────
def test_script_runs_by_path_without_import_error(tmp_path):
    """The plist and docs invoke `python scripts/offline_learning_cycle.py`.
    That must not die at `import src.training...`. --force so it doesn't
    early-return on market-open. Runs under OLC_DATA_ROOT=tmp so it exercises
    the real entrypoint in FULL isolation — never mutates the live journal /
    markers / model state / cursor. We assert it EXITS 0 and prints a JSON
    result, i.e. the run-by-path imports resolved (the #1 bug)."""
    env = {**os.environ, "OLC_DATA_ROOT": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "scripts/offline_learning_cycle.py", "--force"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "ModuleNotFoundError" not in proc.stderr
    # main() prints the result dict as pretty (multi-line) JSON to stdout;
    # logging goes to stderr, so stdout is pure JSON.
    parsed = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert "ran" in parsed


# ── #2/#8: trend-lane featureless entries are excluded from the learner ────
def test_trend_lane_bare_entries_are_not_calibration_scoreable():
    assert is_calibration_scoreable(_trend_entry(_iso(2026, 6, 1), True)) is False
    assert is_calibration_scoreable(_scanner_entry(_iso(2026, 6, 1), True)) is True


def test_run_cycle_ignores_trend_lane_entries_as_new_data(tmp_path, monkeypatch):
    """A journal of ONLY trend-lane bare entries has no calibration signal —
    the cycle must treat it as no-new-data, never build a degenerate gate on
    the collapsed single-feature-vector holdout."""
    _patch_paths(monkeypatch, tmp_path)
    entries = [_trend_entry(_iso(2026, 6, d), win=(d % 2 == 0)) for d in range(1, 25)]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] == "no_new_data"
    assert result["new_outcomes"] == 0
    assert not olc.STATE_PATH.exists()


# ── #3: single-class / degenerate holdout must NOT force a promotion ───────
def test_all_loss_journal_is_refused_not_promoted(tmp_path, monkeypatch):
    """An all-loss journal used to force ACCEPT (candidate Brier ~0 on the
    one-sided holdout). A single-class holdout cannot validate calibration —
    it must be refused."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    entries = [_scanner_entry(_iso(2026, 6, d), win=False, confidence=0.9) for d in range(1, 25)]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] != "accepted"
    assert not olc.STATE_PATH.exists()


def test_severely_imbalanced_holdout_is_refused(tmp_path, monkeypatch):
    """The real-journal shape (1 win / 17 losses): the lone win in the holdout
    makes it technically two-class, but a majority-class 'always predict loss'
    model beats the 0.5 baseline without any calibration skill. A holdout with
    fewer than MIN_MINORITY_HOLDOUT of either class must be refused."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 15)
    entries = [_scanner_entry(_iso(2026, 6, d), win=False, confidence=0.5, tid=f"L{d}") for d in range(1, 18)]
    entries.append(_scanner_entry(_iso(2026, 6, 20), win=True, confidence=0.9, tid="W1"))
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] == "insufficient_holdout_signal"
    assert not olc.STATE_PATH.exists()


def test_all_win_journal_is_refused_not_promoted(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    entries = [_scanner_entry(_iso(2026, 6, d), win=True, confidence=0.9) for d in range(1, 25)]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(entries))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] != "accepted"
    assert not olc.STATE_PATH.exists()


def test_balanced_scanner_journal_still_gates_normally(tmp_path, monkeypatch):
    """Regression guard: the new filters must not break the happy path — a
    balanced scanner-lane journal with real features still reaches a real
    accept/reject decision (not a spurious no_new_data)."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 6)
    # Realistic scanner-lane diversity: varied confidence + regime so the
    # training set has many distinct feature vectors (real journals do).
    regimes = ["NORMAL", "LOW", "HIGH", "EXTREME"]
    train = []
    for d in range(1, 16):
        e = _scanner_entry(_iso(2026, 6, d, hh=9), win=True, confidence=0.85 + d * 0.005, tid=f"w{d}")
        e["regime"] = {"volatility_regime": regimes[d % 4]}
        e["rr_ratio"] = 1.2 + d * 0.05
        train.append(e)
    for d in range(1, 16):
        e = _scanner_entry(_iso(2026, 6, d, hh=13), win=False, confidence=0.10 + d * 0.005, tid=f"l{d}")
        e["regime"] = {"volatility_regime": regimes[(d + 1) % 4]}
        e["rr_ratio"] = 1.3 + d * 0.04
        train.append(e)
    holdout = [
        _scanner_entry(_iso(2026, 6, 20 + i), win=True, confidence=0.9, tid=f"hw{i}") for i in range(3)
    ] + [
        _scanner_entry(_iso(2026, 6, 25 + i), win=False, confidence=0.1, tid=f"hl{i}") for i in range(3)
    ]
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(train + holdout))

    result = olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))

    assert result["decision"] in ("accepted", "rejected")
    assert result["new_outcomes"] == len(train) + len(holdout)


# ── #9: cursor must not drop equal-timestamp entries ──────────────────────
def test_equal_timestamp_entries_are_not_permanently_dropped(tmp_path, monkeypatch):
    """Two closes at the exact same microsecond straddling a run boundary
    must both eventually be trained on — the old `str > cursor` dropped the
    one whose timestamp equalled the cursor forever."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(olc, "MIN_HOLDOUT", 3)
    # A 3-entry holdout can never satisfy the default MIN_MINORITY_HOLDOUT=2 for
    # BOTH classes at once (needs >=4 total) — relax it here so the first cycle
    # actually reaches fit_incremental() (and legitimately advances the cursor)
    # instead of refusing on "insufficient_holdout_signal", which — after the
    # cursor-advance-on-no-fit fix — correctly no longer writes a cursor at all.
    monkeypatch.setattr(olc, "MIN_MINORITY_HOLDOUT", 1)
    ts = _iso(2026, 6, 10, 5, 1, 50, 123456)
    # first run establishes a cursor
    first_batch = (
        [_scanner_entry(_iso(2026, 6, d, hh=9), win=True, confidence=0.9, tid=f"a{d}") for d in range(1, 6)]
        + [_scanner_entry(_iso(2026, 6, d, hh=13), win=False, confidence=0.1, tid=f"b{d}") for d in range(1, 6)]
    )
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(first_batch))
    olc.run_cycle(now=datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc))
    cursor_after_first = json.loads(olc.CURSOR_PATH.read_text())

    # Now two NEW entries share the exact same timestamp; one of them equals
    # nothing in the cursor but both are strictly newer than the first batch.
    twin_a = _scanner_entry(ts, win=True, confidence=0.9, tid="twin_a")
    twin_b = _scanner_entry(ts, win=False, confidence=0.1, tid="twin_b")
    (tmp_path / "trade_journal_rl.json").write_text(json.dumps(first_batch + [twin_a, twin_b]))

    # Rewind the cursor to exactly the twins' timestamp with only twin_a's id
    # processed — twin_b (same ts) must NOT be silently excluded.
    olc._write_cursor_tuple(ts, "twin_a")
    resolved = olc._new_entries_since_cursor(first_batch + [twin_a, twin_b])
    ids = {e["trade_id"] for e in resolved}
    assert "twin_b" in ids
    assert "twin_a" not in ids  # already processed, correctly excluded
    assert cursor_after_first  # sanity: first run wrote a cursor
