"""Trade journal self-healing: repair schema gaps and archive rule-violating entries.

Called periodically from the continuous scan loop (every N cycles) and from
the deployment validator's --auto-fix mode.  Pure data-repair — no side effects
beyond writing the journal and archive files.

Self-heal actions:
1. Compute missing sl_pips / tp_pips from entry/sl/tp prices
2. Archive closed trades with R:R < 1.2:1 out of the active RL journal
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _pip_scale(pair: str) -> float:
    """JPY pairs use 0.01, all others 0.0001."""
    return 0.01 if "JPY" in pair.upper() else 0.0001


def scrub_trade_journal(
    project_root: Path,
    min_rr: float = 1.2,
    journal_rel: str = "trained_data/trade_journal_rl.json",
    archive_rel: str = "trained_data/trade_journal_low_rr_archive.json",
) -> Dict[str, Any]:
    """Repair and scrub the trade journal in-place.

    Returns a summary dict:
        {"repaired": int, "archived": int, "total": int, "error": Optional[str]}
    """
    journal_path = project_root / journal_rel
    archive_path = project_root / archive_rel
    result: Dict[str, Any] = {"repaired": 0, "archived": 0, "total": 0, "error": None}

    if not journal_path.exists():
        return result

    # ── Read journal ──────────────────────────────────────────────────────
    try:
        content = journal_path.read_text().strip()
        if not content:
            return result
        if content.startswith("["):
            journal: List[Dict[str, Any]] = json.loads(content)
        else:
            journal = [json.loads(line) for line in content.split("\n") if line.strip()]
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = str(e)
        logger.warning("journal_scrub: failed to read journal: %s", e)
        return result

    if not isinstance(journal, list):
        return result

    result["total"] = len(journal)

    # ── Scrub loop ────────────────────────────────────────────────────────
    fixed = False
    repaired_journal: List[Dict[str, Any]] = []
    archived_low_rr: List[Dict[str, Any]] = []

    for trade in journal:
        if not isinstance(trade, dict):
            continue

        pair = str(trade.get("pair", ""))
        entry = trade.get("entry_price")
        sl_price = trade.get("sl_price")
        tp_price = trade.get("tp_price")
        pip = _pip_scale(pair)

        # Repair missing sl_pips from price data
        if entry is not None and sl_price is not None and float(trade.get("sl_pips", 0) or 0) <= 0:
            trade["sl_pips"] = round(abs(entry - sl_price) / pip, 1)
            result["repaired"] += 1
            fixed = True

        # Repair missing tp_pips from price data
        if entry is not None and tp_price is not None and float(trade.get("tp_pips", 0) or 0) <= 0:
            trade["tp_pips"] = round(abs(tp_price - entry) / pip, 1)
            result["repaired"] += 1
            fixed = True

        sl_pips = float(trade.get("sl_pips", 0) or 0)
        tp_pips = float(trade.get("tp_pips", 0) or 0)
        rr_ratio = (tp_pips / sl_pips) if sl_pips > 0 and tp_pips > 0 else 0.0

        # Backfill rr_ratio if missing/stale
        if rr_ratio > 0:
            trade["rr_ratio"] = round(rr_ratio, 2)
            trade["risk_reward"] = trade["rr_ratio"]

        # Archive closed trades that violate the R:R floor
        is_closed = isinstance(trade.get("outcome"), dict)
        if is_closed and 0 < rr_ratio < min_rr:
            archived_item = dict(trade)
            archived_item["_archived_reason"] = f"rr_ratio_below_{min_rr}"
            archived_item["_archived_at"] = datetime.utcnow().isoformat()
            archived_low_rr.append(archived_item)
            result["archived"] += 1
            fixed = True
            continue

        repaired_journal.append(trade)

    if not fixed:
        return result

    # ── Write archive (append) ────────────────────────────────────────────
    if archived_low_rr:
        existing_archive: List[Dict[str, Any]] = []
        if archive_path.exists():
            try:
                existing_archive = json.loads(archive_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing_archive = []
        if not isinstance(existing_archive, list):
            existing_archive = []
        existing_archive.extend(archived_low_rr)

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + rename
        tmp_path = archive_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(existing_archive, indent=2))
        tmp_path.rename(archive_path)

    # ── Write repaired journal (atomic) ───────────────────────────────────
    tmp_path = journal_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(repaired_journal, indent=2))
    tmp_path.rename(journal_path)

    if result["archived"] > 0:
        logger.info(
            "journal_scrub: archived %d closed low-R:R trade(s), repaired %d field(s) "
            "(%d trades remaining)",
            result["archived"], result["repaired"], len(repaired_journal),
        )
    elif result["repaired"] > 0:
        logger.info(
            "journal_scrub: repaired %d missing sl_pips/tp_pips field(s)",
            result["repaired"],
        )

    return result
