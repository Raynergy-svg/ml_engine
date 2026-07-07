#!/usr/bin/env python3
"""Market-closed offline learning cycle — the single HEADLESS LEARNING
SUPERVISOR for the risk/execution/calibration layer AND the agent-weight RL
sync. The real consumer for the Tier-7 ``retrain_rl_position_sizer`` self-heal
markers, and (2026-07-06) the sole headless entrypoint for
``ExecutionManager.apply_pending_rl_weight_updates``.

Root cause closed here (2026-07-04): ``self_heal.py``'s
``_handle_retrain_rl_position_sizer`` action wrote marker files into
``trained_data/retrain_requests/`` that NOTHING consumed. This script:

1. Refuses to run while the FX market is open (unless ``--force``), so it
   never competes with the live scan loop for CPU/IO.
2. Drains every pending marker into ``retrain_requests/_processed/``
   (atomic rename, audit trail preserved — never deleted).
3. Runs ``ExecutionManager.apply_pending_rl_weight_updates()`` — scores any
   journal entries whose outcome was resolved (by ``OutcomeBackfill`` or
   ``TrendJournalSync``) but never fed into ``ScannerAgentTeam`` agent-weight
   learning (commit 51b85bf, 2026-07-04). Before 2026-07-06 this method's ONLY
   production caller was ``embedded_scanner.py`` (the TUI) — i.e. it was
   welded to a process that is dormant now, the exact producer/consumer
   dead-write asymmetry docs/ENGINEERING_BRAIN.md's P1 item names. Folding it
   into this already-headless, already-scheduled batch job gives it a live,
   TUI-independent caller without standing up a second daemon. Idempotent via
   the ``rl_weights_applied`` flag, so it is safe to run from here AND the TUI
   (if ever reopened) without double-scoring a trade.
4. Incrementally fits a candidate ``RiskCalibrationLearner`` update on
   journal outcomes since the last processed cursor. Scope is RISK/
   EXECUTION/CALIBRATION only (regime, rr_ratio, sl/tp, mae/mfe, lane,
   existing confidence) — see risk_calibration_learner.py docstring for why
   directional/price features are explicitly excluded.
5. Walk-forward-gates the candidate against the incumbent on a held-out,
   chronologically-later slice (Brier score). Promotes ONLY if the
   candidate beats the incumbent by a margin AND its size multiplier never
   exceeds 1.0 (risk-decreasing-only, by construction). A worse candidate
   is rejected and logged — the incumbent is never overwritten.
6. Logs every cycle to ``trained_data/learning_loop/history.jsonl`` and
   surfaces a summary to ``.claude/brain/learning_loop_status.json`` +
   ``.claude/brain/feed.jsonl`` for AXIOM / the brain loop.

Never touches OANDA. Never reads or writes ``.claude/state.json`` (halt/arm)
or any broker order endpoint. Never trains a directional-alpha model. Safe
to run regardless of halted state, because it does not trade — it only
reads past resolved outcomes and updates an offline model file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
# Run-by-path safety: `python scripts/offline_learning_cycle.py` (the plist +
# documented invocation) puts scripts/ on sys.path[0], NOT the repo root, so
# `import src.training...` fails with ModuleNotFoundError. WorkingDirectory does
# not fix this. Insert the repo root explicitly. (Adversarial review 2026-07-04.)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Data root. Defaults to the repo root (production). `OLC_DATA_ROOT` relocates
# every read/write under a sandbox — used by the run-by-path smoke test so it
# can exercise the real entrypoint WITHOUT mutating live journal / markers /
# model state / cursor. (Added 2026-07-04 after a --force smoke run polluted
# real state.)
_DATA_ROOT = Path(os.environ.get("OLC_DATA_ROOT", str(ROOT)))
JOURNAL_PATH = _DATA_ROOT / "trained_data" / "trade_journal_rl.json"
RETRAIN_REQUESTS_DIR = _DATA_ROOT / "trained_data" / "retrain_requests"
PROCESSED_DIR = RETRAIN_REQUESTS_DIR / "_processed"
STATE_PATH = _DATA_ROOT / "trained_data" / "models" / "risk_calibration_state.json"
HISTORY_PATH = _DATA_ROOT / "trained_data" / "learning_loop" / "history.jsonl"
CURSOR_PATH = _DATA_ROOT / ".claude" / "learning_loop_cursor.json"
BRAIN_STATUS_PATH = _DATA_ROOT / ".claude" / "brain" / "learning_loop_status.json"
BRAIN_FEED_PATH = _DATA_ROOT / ".claude" / "brain" / "feed.jsonl"

MIN_HOLDOUT = 15
PROMOTION_MARGIN = 0.005  # candidate must beat incumbent Brier score by this much
# A holdout that is single-class OR severely class-imbalanced cannot validate
# calibration: a constant / majority-class predictor scores well on it and would
# auto-promote WITHOUT learning anything from features (adversarial review #3 —
# on the real journal, 1 win / 17 losses let an "always predict loss" model beat
# the 0.5 baseline). Require at least this many of BOTH classes in the holdout,
# and at least one of each in the training set, before trusting a promotion.
# The companion review #2 (all-trend featureless holdout collapsing to one
# vector) is handled upstream by is_calibration_scoreable excluding trend-lane
# entries from the population entirely.
MIN_MINORITY_HOLDOUT = 2


def is_fx_market_closed(now: Optional[datetime] = None) -> bool:
    """OANDA FX practice market is closed roughly Fri 21:00 UTC -> Sun 21:00 UTC."""
    now = now or datetime.now(timezone.utc)
    weekday = now.weekday()  # Mon=0 ... Sun=6
    if weekday == 5:  # Saturday: closed all day
        return True
    if weekday == 4 and now.hour >= 21:  # Friday evening
        return True
    if weekday == 6 and now.hour < 21:  # Sunday before evening reopen
        return True
    return False


def drain_retrain_requests() -> List[Dict[str, Any]]:
    """Move every pending marker into ``_processed/`` (atomic rename) and
    return their parsed contents. Never deletes — full audit trail."""
    if not RETRAIN_REQUESTS_DIR.exists():
        return []
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    drained: List[Dict[str, Any]] = []
    for marker in sorted(RETRAIN_REQUESTS_DIR.glob("*.json")):
        try:
            payload = json.loads(marker.read_text())
        except Exception as e:  # noqa: BLE001
            logger.warning("offline_learning_cycle: unreadable marker %s: %s", marker, e)
            payload = {"error": str(e)}
        dest = PROCESSED_DIR / marker.name
        os.replace(marker, dest)
        drained.append({"file": marker.name, **payload})
    return drained


def _load_journal() -> List[Dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    try:
        data = json.loads(JOURNAL_PATH.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("offline_learning_cycle: journal parse failed: %s", e)
        return []
    return data if isinstance(data, list) else []


_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


def _parse_ts(value: Any) -> datetime:
    """Parse an ISO timestamp tolerant of both ``...Z`` and ``...+00:00``.
    Unparseable/missing -> epoch-min so it sorts first (never dropped)."""
    s = str(value or "")
    if not s:
        return _MIN_DT
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return _MIN_DT


def _entry_key(entry: Dict[str, Any]) -> Tuple[datetime, str]:
    """Total order over journal entries: (parsed timestamp, trade_id). The
    trade_id tiebreaker is why equal-timestamp entries are never dropped
    (adversarial review #9) and why the comparison is format-agnostic."""
    return (_parse_ts(entry.get("timestamp")), str(entry.get("trade_id", "")))


def _read_cursor() -> Tuple[datetime, str]:
    if not CURSOR_PATH.exists():
        return (_MIN_DT, "")
    try:
        data = json.loads(CURSOR_PATH.read_text())
        return (_parse_ts(data.get("last_processed_timestamp")),
                str(data.get("last_processed_trade_id", "")))
    except Exception:  # noqa: BLE001
        return (_MIN_DT, "")


def _new_entries_since_cursor(journal: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calibration-scoreable, resolved entries strictly after the cursor,
    chronologically sorted. Uses the (timestamp, trade_id) total order so an
    entry whose timestamp equals the cursor's is only excluded when it IS the
    cursor's trade (or older), never merely because the timestamp matched."""
    from src.training.incremental.risk_calibration_learner import (
        _label,
        is_calibration_scoreable,
    )
    cur = _read_cursor()
    scoreable = [
        e for e in journal
        if isinstance(e, dict) and is_calibration_scoreable(e) and _label(e) is not None
    ]
    scoreable.sort(key=_entry_key)
    return [e for e in scoreable if _entry_key(e) > cur]


def _write_cursor_tuple(ts: Any, trade_id: Any) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURSOR_PATH.with_suffix(CURSOR_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "last_processed_timestamp": str(ts or ""),
        "last_processed_trade_id": str(trade_id or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    os.replace(tmp, CURSOR_PATH)


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _write_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_PATH)


def _append_history(event: Dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def _write_brain_status(event: Dict[str, Any]) -> None:
    BRAIN_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BRAIN_STATUS_PATH.with_suffix(BRAIN_STATUS_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(event, indent=2, default=str))
    os.replace(tmp, BRAIN_STATUS_PATH)
    try:
        BRAIN_FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BRAIN_FEED_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": event.get("timestamp"),
                "source": "offline_learning_cycle",
                "text": event.get("summary", ""),
            }) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.debug("offline_learning_cycle: brain feed append skipped: %s", e)


def _run_rl_weight_sync() -> Dict[str, Any]:
    """Headless call to ``ExecutionManager.apply_pending_rl_weight_updates``
    (the single-feedback-path anchor, commit 51b85bf). That method reads/
    writes journal + agent-weight paths RELATIVE TO CWD, so this chdirs for
    the duration of the call — into ``JOURNAL_PATH.parent.parent`` (read at
    CALL TIME, not captured at import), NOT ``_DATA_ROOT`` directly. This
    deliberately reuses whatever the module's ``JOURNAL_PATH`` constant
    currently resolves to, so tests that sandbox via
    ``monkeypatch.setattr(olc, "JOURNAL_PATH", tmp_path / ...)`` (the
    existing convention in ``tests/test_offline_learning_cycle_2026_07_04.py``)
    automatically sandbox this call too — chdir-ing off of the separate
    ``OLC_DATA_ROOT``-derived ``_DATA_ROOT`` constant would silently operate
    on the REAL production journal/agent_weights.json in any test that
    reaches this call without also exporting ``OLC_DATA_ROOT`` (a real
    safety gap, caught before shipping — see the commit message). Never
    touches OANDA/halt/arm (see the method's own docstring); any failure is
    caught and reported, never allowed to block the calibration cycle below.
    """
    from src.scanner.execution import ExecutionConfig, ExecutionManager

    data_root = JOURNAL_PATH.parent.parent
    original_cwd = os.getcwd()
    os.chdir(data_root)
    try:
        mgr = ExecutionManager(config=ExecutionConfig())
        return mgr.apply_pending_rl_weight_updates()
    except Exception as exc:  # noqa: BLE001 — RL sync failure must never block calibration
        logger.warning("offline_learning_cycle: rl weight sync failed: %s", exc)
        return {"applied": 0, "weights_updated": False, "detail": f"error: {exc}"}
    finally:
        os.chdir(original_cwd)


def run_cycle(*, force: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
    from src.training.incremental.risk_calibration_learner import (
        RiskCalibrationLearner,
        _label,
        brier_score,
        build_features,
    )

    now = now or datetime.now(timezone.utc)
    if not force and not is_fx_market_closed(now):
        return {"ran": False, "reason": "market_open"}

    markers = drain_retrain_requests()
    rl_weight_sync = _run_rl_weight_sync()

    journal = _load_journal()
    # Population = calibration-scoreable, resolved, strictly-after-cursor,
    # chronologically ordered. Trend-lane featureless entries are excluded
    # here (review #2) so they can't collapse the holdout.
    new_entries = _new_entries_since_cursor(journal)

    result: Dict[str, Any] = {
        "ran": True,
        "timestamp": now.isoformat(),
        "markers_drained": len(markers),
        "rl_weight_sync": rl_weight_sync,
        "new_outcomes": len(new_entries),
        "decision": "no_new_data",
    }

    def _advance_cursor_and_return(
        res: Dict[str, Any], summary: str, *, advance_cursor: bool = True
    ) -> Dict[str, Any]:
        _append_history(res)
        _write_brain_status({**res, "summary": summary, "consumed_by_live_sizing": False})
        if advance_cursor and new_entries:
            last = new_entries[-1]
            _write_cursor_tuple(last.get("timestamp", ""), last.get("trade_id", ""))
        return res

    if len(new_entries) < MIN_HOLDOUT + 1:
        # Do NOT advance the cursor here: fit_incremental() was never called on
        # these entries (that happens further below, only once MIN_HOLDOUT+1 is
        # reached), so marking them "processed" would permanently discard them
        # from training instead of letting them accumulate toward the next cycle.
        result["decision"] = "no_new_data" if not new_entries else "insufficient_new_data"
        return _advance_cursor_and_return(
            result,
            f"offline learning cycle: {len(markers)} marker(s) drained, "
            f"insufficient new data ({len(new_entries)}/{MIN_HOLDOUT + 1} needed)",
            advance_cursor=False,
        )

    holdout = new_entries[-MIN_HOLDOUT:]
    train_new = new_entries[:-MIN_HOLDOUT]

    # Guard #3: refuse on a single-class or severely imbalanced holdout — a
    # constant / majority-class predictor scores well on it WITHOUT learning
    # from features, and would auto-promote. Validating on a holdout that has
    # enough of BOTH outcomes is the principled bar; a candidate that overfit a
    # one-sided training set is then correctly rejected by the Brier comparison
    # below, so no separate training-set-balance guard is needed.
    holdout_wins = sum(1 for e in holdout if _label(e) == 1.0)
    holdout_losses = len(holdout) - holdout_wins
    if min(holdout_wins, holdout_losses) < MIN_MINORITY_HOLDOUT:
        result["decision"] = "insufficient_holdout_signal"
        result["holdout_wins"] = holdout_wins
        result["holdout_losses"] = holdout_losses
        return _advance_cursor_and_return(
            result,
            f"offline learning cycle: refused — holdout {holdout_wins}W/{holdout_losses}L "
            f"< {MIN_MINORITY_HOLDOUT} of a class; cannot validate calibration",
            advance_cursor=False,
        )

    incumbent_state = _load_state()
    incumbent = RiskCalibrationLearner.from_state(incumbent_state)
    candidate = deepcopy(incumbent)
    candidate.fit_incremental(train_new)

    holdout_samples = [(build_features(e), _label(e)) for e in holdout]
    incumbent_metric = brier_score(incumbent.model, holdout_samples)
    candidate_metric = brier_score(candidate.model, holdout_samples)

    # Risk-decreasing check, explicit and redundant with the multiplier's
    # own [0.5, 1.0] cap: a promoted candidate must never be capable of
    # sizing UP beyond what the incumbent already allowed.
    max_multiplier = max(
        (candidate.size_multiplier(x) for x, _ in holdout_samples), default=1.0
    )

    promote = (
        incumbent_metric is not None
        and candidate_metric is not None
        and candidate_metric < incumbent_metric - PROMOTION_MARGIN
        and max_multiplier <= 1.0
    )

    result.update({
        "incumbent_brier": incumbent_metric,
        "candidate_brier": candidate_metric,
        "max_size_multiplier": max_multiplier,
        "decision": "accepted" if promote else "rejected",
    })

    if promote:
        _write_state(candidate.to_state())
        result["model_n_seen"] = candidate.model.n_seen

    summary = (
        f"offline learning cycle: {result['decision']} "
        f"(brier {incumbent_metric:.4f}→{candidate_metric:.4f}, "
        f"{len(markers)} marker(s) drained, {len(new_entries)} new outcome(s))"
    )
    return _advance_cursor_and_return(result, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="run even if the FX market is open (manual/testing invocation only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = run_cycle(force=args.force)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
