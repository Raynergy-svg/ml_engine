"""Journal trade loader with archive-salvage support.

Live journal at `trained_data/trade_journal_rl.json` is canonical; archive
files under `trained_data/archive/trade_journal_rl_*.json` are JSON-corrupt
but recoverable by truncating the trailing partial record (cut at the last
record terminator `\\n  },\\n` and append `\\n  }\\n]\\n`).

Returns flat list of trade dicts with normalized fields:
    pair (e.g. EUR_USD; underscore form)
    timestamp (ISO-8601 string from journal `timestamp` field)
    entry_price
    direction
    confidence
    sl_pips, tp_pips (may be None on older archive schemas)
    outcome (dict containing trade_won, exit_reason, pnl_pips, ...)
    trade_won (boolean, hoisted from outcome dict for convenience)

Dedup key: (pair, timestamp_floor_minute, entry_price).

The salvage routine is deliberately conservative: it never throws on a
corrupt file — it returns whatever records it could parse cleanly, and
logs counts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Project layout
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LIVE_JOURNAL = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
_ARCHIVE_DIR = _PROJECT_ROOT / "trained_data" / "archive"
_ARCHIVE_GLOB = "trade_journal_rl_*.json"

# Regex matches end-of-record at indent 2 (the standard journal format)
# Captures the position immediately after the closing `},` of a record so
# we can trim there.
_RECORD_END_RE = re.compile(r"\n  \},\n")


def salvage_corrupt_journal(path: Path) -> List[Dict[str, Any]]:
    """Best-effort tolerant parse of a (possibly corrupt) journal file.

    Strategy: read the raw text, find the last position of `\\n  },\\n`
    (a record terminator at indent 2), truncate after the closing `}` of
    that record, and append `\\n]\\n`. This drops at most one partially
    written trailing record.

    Returns an empty list if the file cannot be salvaged.
    """
    try:
        text = path.read_text()
    except OSError as e:
        logger.warning("salvage: cannot read %s: %s", path, e)
        return []

    # Try clean parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        pass  # Fall through to salvage path

    matches = list(_RECORD_END_RE.finditer(text))
    if not matches:
        logger.warning("salvage: no record terminators in %s; file unrecoverable", path)
        return []

    last = matches[-1].start()
    candidate = text[:last] + "\n  }\n]\n"
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            logger.info(
                "salvage: recovered %d records from %s (truncated %d bytes)",
                len(data),
                path.name,
                len(text) - last,
            )
            return data
    except json.JSONDecodeError as e:
        logger.warning("salvage: tolerant parse of %s failed: %s", path, e)

    return []


def _normalize_pair(s: str) -> str:
    """Normalize pair name to underscore form (EUR_USD)."""
    if not isinstance(s, str):
        return ""
    return s.replace("/", "_").replace("-", "_").upper().strip()


def _hoist_won(t: Dict[str, Any]) -> Optional[bool]:
    """Extract trade_won boolean from a trade record (handles old + new schemas)."""
    out = t.get("outcome")
    if isinstance(out, dict):
        if "trade_won" in out:
            return bool(out["trade_won"])
        # Older schemas might use exit_reason or pnl sign
        reason = str(out.get("exit_reason", "")).lower()
        if reason in ("tp_hit", "take_profit"):
            return True
        if reason in ("sl_hit", "stop_loss", "stopped"):
            return False
        pnl = out.get("realized_pl") or out.get("pnl_pips")
        if isinstance(pnl, (int, float)):
            return pnl > 0
    elif isinstance(out, str):
        s = out.upper()
        if s in ("WIN", "WON", "TP_HIT"):
            return True
        if s in ("LOSS", "LOST", "SL_HIT", "STOPPED"):
            return False
    return None


def _normalize_record(t: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw journal record to a normalized trade dict, or None if unusable."""
    if not isinstance(t, dict):
        return None
    pair = _normalize_pair(t.get("pair", ""))
    ts = t.get("timestamp")
    entry = t.get("entry_price")
    if not pair or not ts or entry is None:
        return None
    won = _hoist_won(t)
    if won is None:
        return None  # Skip open / unknown-outcome
    return {
        "pair": pair,
        "timestamp": ts,
        "entry_price": float(entry),
        "direction": str(t.get("direction", "")).upper(),
        "confidence": t.get("confidence"),
        "sl_price": t.get("sl_price"),
        "tp_price": t.get("tp_price"),
        "sl_pips": t.get("sl_pips"),
        "tp_pips": t.get("tp_pips"),
        "outcome": t.get("outcome"),
        "trade_won": won,
        "regime": (
            t.get("regime", {}).get("volatility_regime")
            if isinstance(t.get("regime"), dict)
            else None
        ),
    }


def _dedup_key(t: Dict[str, Any]) -> Tuple[str, str, float]:
    """Stable dedup key. Floor timestamp to whole minutes to handle resave noise."""
    ts = str(t.get("timestamp", ""))
    # Take prefix up to minute (YYYY-MM-DDTHH:MM)
    ts_floor = ts[:16]
    return (
        str(t.get("pair", "")),
        ts_floor,
        round(float(t.get("entry_price", 0.0) or 0.0), 6),
    )


def load_all_journal_trades(
    include_archives: bool = True,
    live_path: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load and normalize trades from live journal + (optionally) archives.

    Args:
        include_archives: include `trained_data/archive/trade_journal_rl_*.json`.
        live_path: override live journal path (for tests).
        archive_dir: override archive dir (for tests).

    Returns:
        List of normalized trade dicts (closed trades only), de-duplicated.
    """
    live_path = live_path or _LIVE_JOURNAL
    archive_dir = archive_dir or _ARCHIVE_DIR

    raw_trades: List[Dict[str, Any]] = []

    if live_path.exists():
        try:
            data = json.loads(live_path.read_text())
            if isinstance(data, list):
                raw_trades.extend(data)
                logger.info("Loaded %d trades from live journal", len(data))
        except json.JSONDecodeError:
            # Live journal corrupt — try salvage
            recovered = salvage_corrupt_journal(live_path)
            raw_trades.extend(recovered)
            logger.warning(
                "Live journal corrupt; salvaged %d trades", len(recovered)
            )

    if include_archives and archive_dir.exists():
        for f in sorted(archive_dir.glob(_ARCHIVE_GLOB)):
            recovered = salvage_corrupt_journal(f)
            raw_trades.extend(recovered)

    # Normalize + filter to closed trades with usable data
    normalized: List[Dict[str, Any]] = []
    skipped = 0
    for t in raw_trades:
        norm = _normalize_record(t)
        if norm is None:
            skipped += 1
            continue
        normalized.append(norm)

    # Dedup by (pair, ts_minute, entry_price); keep first (live is loaded first
    # so it wins over older archive duplicates).
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for t in normalized:
        k = _dedup_key(t)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(t)

    logger.info(
        "load_all_journal_trades: raw=%d normalized=%d skipped=%d deduped=%d",
        len(raw_trades),
        len(normalized),
        skipped,
        len(deduped),
    )
    return deduped


def trades_to_per_pair_index(
    trades: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Index normalized trades by pair name (underscore form)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        out.setdefault(t["pair"], []).append(t)
    return out
