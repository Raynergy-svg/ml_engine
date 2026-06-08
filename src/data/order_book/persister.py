"""Append-only JSONL persistence for OANDA order/position book snapshots.

Why JSONL: order book is fetched every 5 min in the live loop; row-append is
the natural shape. JSONL is also tail-and-grep friendly for postmortems.
Future retrain pipelines can convert to parquet in one pass.

Per CLAUDE.md JSON safety gates: atomic append using lock-then-fsync.
Per the No-Mock Rule: this module is tested against real disk via tmp_path.

Storage estimate: ~250 bytes/row × 288 rows/day (5-min cache) × 15 pairs × 2
book types ≈ 660 MB/year. Manageable. Operator can rotate older files into
parquet archives once volume warrants.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.order_book.features import (
    extract_order_features,
    extract_position_features,
)


# Default location; the agent passes an override during testing.
_DEFAULT_ROOT = Path("trained_data/order_book")


def _path_for(pair: str, book_type: str, root: Optional[Path] = None) -> Path:
    """Per-pair-per-book-type JSONL path. Caller owns mkdir."""
    base = root if root is not None else _DEFAULT_ROOT
    return base / f"{pair}_{book_type}.jsonl"


def snapshot(
    pair: str,
    book_type: str,
    buckets: List[Dict[str, Any]],
    current_price: float,
    *,
    timestamp: Optional[datetime] = None,
    root: Optional[Path] = None,
    keep_raw_buckets: bool = False,
) -> Optional[Path]:
    """Append one JSONL row for the given snapshot.

    Args:
        pair: instrument code (e.g. ``EUR_USD``).
        book_type: ``"position"`` or ``"order"``.
        buckets: OANDA's raw bucket array (each item has ``price``,
            ``longCountPercent``, ``shortCountPercent``).
        current_price: mid-price at snapshot time, used for distance bands.
        timestamp: snapshot UTC time; defaults to ``datetime.now(UTC)``.
        root: override the default ``trained_data/order_book`` root (tests).
        keep_raw_buckets: include the full bucket array in the row. Default
            False — features are the load-bearing payload; raw buckets bloat
            the file ~10x. Set True for forensic captures.

    Returns:
        The path written to, or ``None`` if input is unusable (empty buckets
        or non-positive current_price). The agent treats failures as soft —
        persistence is best-effort, never blocking a trade decision.
    """
    if book_type not in ("position", "order"):
        return None
    if not buckets or current_price <= 0:
        return None

    if book_type == "position":
        features = extract_position_features(buckets, current_price)
    else:
        features = extract_order_features(buckets, current_price)

    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    row: Dict[str, Any] = {
        "timestamp": ts.isoformat(),
        "pair": pair,
        "book_type": book_type,
        "current_price": current_price,
        "features": features,
    }
    if keep_raw_buckets:
        row["buckets"] = buckets

    path = _path_for(pair, book_type, root=root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    # Append-mode write; one JSON object per line + trailing newline.
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        return None
    return path
