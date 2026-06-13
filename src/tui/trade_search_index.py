"""In-memory SQLite FTS5 index over trade_journal_rl.json (pick #3).

Pattern adapted from buddy-autonomous-loop's sessions.ts FTS5 session search.
The journal is small (low thousands of rows even after years of trading), so
rebuilding the whole index on demand is cheap (~10ms for 1k rows) and avoids
the cache-invalidation problem when new trades close.

Searchable corpus per row (all lowercased, space-joined):
  trade_id · pair · direction · broker · asset_class · volatility_regime ·
  gate_summary · outcome_label · all agent reason texts · all agent reason codes

Typical queries that work:
  "EUR_AUD"          → all EUR_AUD trades
  "loss trend"       → losing trades where trend agent was mentioned
  "ranging"          → LOW/ranging regime trades
  "1220"             → exact trade_id 1220
  "" (empty)         → latest `limit` trades sorted by timestamp desc
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _outcome_label(entry: dict[str, Any]) -> str:
    """'win' / 'loss' / 'open' (used in the searchable corpus)."""
    outcome = entry.get("outcome")
    if not isinstance(outcome, dict) or "realized_pl" not in outcome:
        return "open"
    if outcome.get("trade_won") is True:
        return "win"
    return "loss"


def _entry_to_corpus(entry: dict[str, Any]) -> str:
    """Build the searchable text blob for one trade entry."""
    parts: list[str] = []
    for key in ("trade_id", "pair", "direction", "broker", "asset_class"):
        val = entry.get(key)
        if val:
            parts.append(str(val))

    regime = entry.get("regime")
    if isinstance(regime, dict):
        vol = regime.get("volatility_regime")
        if vol:
            parts.append(str(vol))

    gates = entry.get("gates")
    if isinstance(gates, dict):
        summary = gates.get("gate_summary")
        if summary:
            parts.append(str(summary))

    parts.append(_outcome_label(entry))

    agents = entry.get("agents")
    if isinstance(agents, dict):
        for reason in agents.get("agent_reasons", []) or []:
            if isinstance(reason, dict):
                name = reason.get("name") or reason.get("agent")
                if name:
                    parts.append(str(name))
                code = reason.get("reason_code")
                if code:
                    parts.append(str(code))
                text = reason.get("reason") or reason.get("description")
                if text:
                    parts.append(str(text))

    return " ".join(p.lower() for p in parts if p)


def _safe_timestamp(entry: dict[str, Any]) -> str:
    """Sortable ISO timestamp; missing/bad values sort first (oldest)."""
    ts = entry.get("timestamp")
    return str(ts) if isinstance(ts, str) else ""


class TradeSearchIndex:
    """FTS5-backed search over a list of trade dicts (the source-of-truth format
    used by trade_journal_rl.json).

    Lifecycle:
      idx = TradeSearchIndex.from_entries(entries)
      hits = idx.search("EUR_AUD loss", limit=50)   # ranked by FTS bm25
      latest = idx.search("", limit=20)             # newest-first fallback

    The class owns the SQLite connection; close it via ``.close()`` (also a
    context manager). Connections are thread-confined per the sqlite3 module
    default — don't share across threads.
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE VIRTUAL TABLE trades USING fts5(searchable, tokenize='unicode61')"
        )
        self._entries: list[dict[str, Any]] = []

    @classmethod
    def from_entries(cls, entries: Iterable[dict[str, Any]]) -> "TradeSearchIndex":
        idx = cls()
        idx.build(entries)
        return idx

    @classmethod
    def from_journal_file(cls, path: str | Path) -> "TradeSearchIndex":
        """Read trade_journal_rl.json from disk and build the index.

        Returns an empty index (no entries) if the file is missing or
        malformed — never raises on corrupt journal so the TUI stays alive.
        """
        idx = cls()
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("TradeSearchIndex: cannot read %s: %s", path, exc)
            return idx
        if not isinstance(raw, list):
            logger.warning("TradeSearchIndex: %s is not a list (got %s)", path, type(raw).__name__)
            return idx
        idx.build(raw)
        return idx

    def build(self, entries: Iterable[dict[str, Any]]) -> None:
        """Replace the index contents with the given entries (in order).

        Row ids are 1-based positions in the materialized list, used to map
        FTS hits back to the original dict.
        """
        self._conn.execute("DELETE FROM trades")
        self._entries = [e for e in entries if isinstance(e, dict)]
        rows = [
            (i + 1, _entry_to_corpus(e))
            for i, e in enumerate(self._entries)
        ]
        if rows:
            self._conn.executemany(
                "INSERT INTO trades(rowid, searchable) VALUES (?, ?)", rows
            )
        self._conn.commit()

    def search(self, query: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Run the search.

        Empty/whitespace query → most-recent-first slice of the journal.
        Non-empty → FTS5 MATCH, ranked by bm25 (best first).
        Malformed FTS queries (e.g. lone operators) fall back to a LIKE search
        so users typing partial input don't see crashes.
        """
        if not self._entries:
            return []

        q = query.strip()
        if not q:
            ordered = sorted(
                self._entries, key=_safe_timestamp, reverse=True
            )
            return ordered[:limit]

        fts_query = _to_fts_query(q)
        try:
            cur = self._conn.execute(
                "SELECT rowid FROM trades WHERE trades MATCH ? "
                "ORDER BY bm25(trades) LIMIT ?",
                (fts_query, limit),
            )
            rowids = [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError as exc:
            logger.debug("TradeSearchIndex: FTS parse fallback for %r: %s", q, exc)
            return self._like_fallback(q, limit)

        return [self._entries[r - 1] for r in rowids if 1 <= r <= len(self._entries)]

    def _like_fallback(self, query: str, limit: int) -> list[dict[str, Any]]:
        like = "%" + query.lower().replace("%", "") + "%"
        cur = self._conn.execute(
            "SELECT rowid FROM trades WHERE searchable LIKE ? LIMIT ?",
            (like, limit),
        )
        rowids = [r[0] for r in cur.fetchall()]
        return [self._entries[r - 1] for r in rowids if 1 <= r <= len(self._entries)]

    def close(self) -> None:
        self._conn.close()
        self._entries = []

    def __enter__(self) -> "TradeSearchIndex":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._entries)


_FTS_SPECIALS = set('"():*^')


def _to_fts_query(user_input: str) -> str:
    """Translate a free-form user query to a safe FTS5 MATCH expression.

    Strategy: tokenize on whitespace, drop tokens containing FTS operator
    characters, AND the remainder together as prefix queries (`token*`).
    This makes partial typing usable ("eur_a" matches "eur_aud") while
    avoiding the FTS5 syntax errors that lone quotes / parens would cause.
    """
    safe_tokens: list[str] = []
    for token in user_input.split():
        # Strip FTS-meaningful chars from edges; drop the token if nothing useful is left.
        stripped = "".join(ch for ch in token if ch not in _FTS_SPECIALS)
        if not stripped:
            continue
        safe_tokens.append(f"{stripped.lower()}*")
    if not safe_tokens:
        # Empty after sanitization → match nothing (caller will fall back via LIKE).
        return "zzz_no_match_zzz"
    return " AND ".join(safe_tokens)
