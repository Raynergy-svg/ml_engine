#!/usr/bin/env python3
"""Market-closed offline learning cycle for the risk/execution/calibration
layer — the real consumer for the Tier-7 ``retrain_rl_position_sizer``
self-heal markers.

Root cause closed here (2026-07-04): ``self_heal.py``'s
``_handle_retrain_rl_position_sizer`` action wrote marker files into
``trained_data/retrain_requests/`` that NOTHING consumed. This script:

1. Refuses to run while the FX market is open (unless ``--force``), so it
   never competes with the live scan loop for CPU/IO.
2. Drains every pending marker into ``retrain_requests/_processed/``
   (atomic rename, audit trail preserved — never deleted).
3. Incrementally fits a candidate ``RiskCalibrationLearner`` update on
   journal outcomes since the last processed cursor. Scope is RISK/
   EXECUTION/CALIBRATION only (regime, rr_ratio, sl/tp, mae/mfe, lane,
   existing confidence) — see risk_calibration_learner.py docstring for why
   directional/price features are explicitly excluded.
4. Walk-forward-gates the candidate against the incumbent on a held-out,
   chronologically-later slice (Brier score). Promotes ONLY if the
   candidate beats the incumbent by a margin AND its size multiplier never
   exceeds 1.0 (risk-decreasing-only, by construction). A worse candidate
   is rejected and logged — the incumbent is never overwritten.
5. Logs every cycle to ``trained_data/learning_loop/history.jsonl`` and
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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_PATH = ROOT / "trained_data" / "trade_journal_rl.json"
RETRAIN_REQUESTS_DIR = ROOT / "trained_data" / "retrain_requests"
PROCESSED_DIR = RETRAIN_REQUESTS_DIR / "_processed"
STATE_PATH = ROOT / "trained_data" / "models" / "risk_calibration_state.json"
HISTORY_PATH = ROOT / "trained_data" / "learning_loop" / "history.jsonl"
CURSOR_PATH = ROOT / ".claude" / "learning_loop_cursor.json"
BRAIN_STATUS_PATH = ROOT / ".claude" / "brain" / "learning_loop_status.json"
BRAIN_FEED_PATH = ROOT / ".claude" / "brain" / "feed.jsonl"

MIN_HOLDOUT = 15
PROMOTION_MARGIN = 0.005  # candidate must beat incumbent Brier score by this much


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


def _read_cursor() -> str:
    if not CURSOR_PATH.exists():
        return ""
    try:
        return json.loads(CURSOR_PATH.read_text()).get("last_processed_timestamp", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _write_cursor(ts: str) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CURSOR_PATH.with_suffix(CURSOR_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "last_processed_timestamp": ts,
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

    journal = _load_journal()
    cursor = _read_cursor()
    resolved = [e for e in journal if isinstance(e, dict) and _label(e) is not None]
    resolved.sort(key=lambda e: str(e.get("timestamp", "")))
    new_entries = [e for e in resolved if str(e.get("timestamp", "")) > cursor] if cursor else resolved

    result: Dict[str, Any] = {
        "ran": True,
        "timestamp": now.isoformat(),
        "markers_drained": len(markers),
        "new_outcomes": len(new_entries),
        "decision": "no_new_data",
    }

    if len(new_entries) < MIN_HOLDOUT + 1:
        _append_history(result)
        _write_brain_status({
            **result,
            "summary": (
                f"offline learning cycle: {len(markers)} marker(s) drained, "
                f"insufficient new data ({len(new_entries)}/{MIN_HOLDOUT + 1} needed)"
            ),
        })
        if new_entries:
            _write_cursor(str(new_entries[-1].get("timestamp", cursor)))
        return result

    holdout = new_entries[-MIN_HOLDOUT:]
    train_new = new_entries[:-MIN_HOLDOUT]

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

    _write_cursor(str(new_entries[-1].get("timestamp", cursor)))
    _append_history(result)

    if incumbent_metric is not None and candidate_metric is not None:
        summary = (
            f"offline learning cycle: {result['decision']} "
            f"(brier {incumbent_metric:.4f}→{candidate_metric:.4f}, "
            f"{len(markers)} marker(s) drained, {len(new_entries)} new outcome(s))"
        )
    else:
        summary = f"offline learning cycle: {result['decision']}"
    _write_brain_status({**result, "summary": summary})

    return result


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
