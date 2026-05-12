"""On-disk SQLite + FTS5 audit index over trade_journal_rl.json agent votes.

Each row of the journal expands to one row PER agent that voted on that
trade (typically 12-15 rows per closed trade). The resulting ``votes`` table
plus an FTS5 shadow over ``reason_text`` lets operators ask questions like:

  "show every trade in the last 30 days where mean_reversion voted NO and
   the outcome was loss"
  "find all losses where the agent reason mentioned 'spread' or 'drawdown'"

Unlike :class:`src.tui.trade_search_index.TradeSearchIndex` (which is an
in-memory FTS over whole trades), this index is persistent and structured
because the audit needs to survive process restart and compose multiple
column filters with FTS in a single query.

DB path:    ``trained_data/agent_vote_index.db``
Source:     ``trained_data/trade_journal_rl.json``
Sync model: watermark on the journal's tail index; full rebuild fallback
            when the journal has been rewritten.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _PROJECT_ROOT / "trained_data" / "agent_vote_index.db"
JOURNAL_PATH = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS votes (
  trade_id      TEXT NOT NULL,
  agent_name    TEXT NOT NULL,
  passed        INTEGER NOT NULL,
  score         REAL,
  weight        REAL,
  reason_text   TEXT,
  regime        TEXT,
  outcome       TEXT,
  pnl_usd       REAL,
  closed_ts_iso TEXT,
  PRIMARY KEY (trade_id, agent_name)
);
CREATE INDEX IF NOT EXISTS idx_votes_agent_outcome ON votes(agent_name, outcome);
CREATE INDEX IF NOT EXISTS idx_votes_closed_ts ON votes(closed_ts_iso);

CREATE VIRTUAL TABLE IF NOT EXISTS votes_fts USING fts5(
  reason_text,
  content='votes',
  content_rowid='rowid',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS votes_ai AFTER INSERT ON votes BEGIN
  INSERT INTO votes_fts(rowid, reason_text) VALUES (new.rowid, new.reason_text);
END;
CREATE TRIGGER IF NOT EXISTS votes_ad AFTER DELETE ON votes BEGIN
  INSERT INTO votes_fts(votes_fts, rowid, reason_text) VALUES ('delete', old.rowid, old.reason_text);
END;
CREATE TRIGGER IF NOT EXISTS votes_au AFTER UPDATE ON votes BEGIN
  INSERT INTO votes_fts(votes_fts, rowid, reason_text) VALUES ('delete', old.rowid, old.reason_text);
  INSERT INTO votes_fts(rowid, reason_text) VALUES (new.rowid, new.reason_text);
END;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def _outcome_label(entry: Dict[str, Any]) -> str:
    """'win' / 'loss' / 'breakeven' / 'open' for a journal entry."""
    outcome = entry.get("outcome")
    if not isinstance(outcome, dict) or "realized_pl" not in outcome:
        return "open"
    pnl = outcome.get("realized_pl")
    try:
        pnl_f = float(pnl)
    except (TypeError, ValueError):
        pnl_f = 0.0
    if outcome.get("trade_won") is True:
        return "win"
    if abs(pnl_f) < 1.0:
        return "breakeven"
    return "loss"


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _extract_rows(entry: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    """One tuple per (trade_id, agent_name). Missing fields become None."""
    trade_id = entry.get("trade_id")
    if trade_id is None:
        return []
    trade_id = str(trade_id)

    regime_blob = entry.get("regime")
    regime = (regime_blob.get("volatility_regime") if isinstance(regime_blob, dict) else None) or ""

    outcome_label = _outcome_label(entry)
    outcome_blob = entry.get("outcome") if isinstance(entry.get("outcome"), dict) else {}
    pnl_usd = _safe_float(outcome_blob.get("realized_pl"))
    closed_ts = outcome_blob.get("close_time") or entry.get("timestamp") or ""

    agents = entry.get("agents")
    reasons = agents.get("agent_reasons") if isinstance(agents, dict) else None
    if not isinstance(reasons, list):
        return []

    rows: List[Tuple[Any, ...]] = []
    for r in reasons:
        if not isinstance(r, dict):
            continue
        name = r.get("name") or r.get("agent")
        if not name:
            continue
        passed = 1 if r.get("passed") else 0
        score = _safe_float(r.get("score"))
        weight = _safe_float(r.get("weight"))
        reason_text = str(r.get("reason") or r.get("description") or "")
        rows.append((
            trade_id,
            str(name),
            passed,
            score,
            weight,
            reason_text,
            str(regime),
            outcome_label,
            pnl_usd,
            str(closed_ts),
        ))
    return rows


# ---------------------------------------------------------------------------
# AgentVoteIndex
# ---------------------------------------------------------------------------

class AgentVoteIndex:
    """SQLite+FTS5 audit index over agent votes per closed trade.

    Open one instance per query/sync. Connection is thread-confined; do not
    share across threads.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None gives us explicit BEGIN/COMMIT control via execute.
        self._conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    @classmethod
    def open(cls, db_path: Optional[Path] = None) -> "AgentVoteIndex":
        return cls(db_path=db_path)

    # ------------------------------------------------------------------
    # Schema + meta
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._meta_set("schema_version", str(SCHEMA_VERSION))

    def _meta_get(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row is not None else None

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def _load_journal(self, journal_path: Path) -> List[Dict[str, Any]]:
        """Parse the journal file. Returns empty list on missing/malformed input."""
        if not journal_path.exists():
            return []
        try:
            raw = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("AgentVoteIndex: cannot read %s: %s", journal_path, exc)
            return []
        if not isinstance(raw, list):
            logger.warning("AgentVoteIndex: %s is not a list (got %s)", journal_path, type(raw).__name__)
            return []
        return [e for e in raw if isinstance(e, dict)]

    def sync_from_journal(self, journal_path: Optional[Path] = None) -> Dict[str, int]:
        """Incrementally sync the journal into the index.

        Watermark protocol:
          - meta.watermark_idx: number of journal entries processed.
          - meta.watermark_trade_id: trade_id at watermark_idx-1 (sanity).
        If the journal at watermark_idx-1 no longer matches watermark_trade_id,
        the journal has been rewritten and we full-rebuild.

        Returns ``{"added": N, "updated": M, "skipped": K}``.
        """
        path = Path(journal_path) if journal_path is not None else JOURNAL_PATH
        entries = self._load_journal(path)
        total = len(entries)
        result = {"added": 0, "updated": 0, "skipped": 0}

        try:
            watermark_idx = int(self._meta_get("watermark_idx") or "0")
        except ValueError:
            watermark_idx = 0
        watermark_trade_id = self._meta_get("watermark_trade_id") or ""

        # Sanity: did the journal get rewritten under us?
        if 0 < watermark_idx <= total:
            anchor = entries[watermark_idx - 1]
            if str(anchor.get("trade_id", "")) != watermark_trade_id:
                logger.warning(
                    "AgentVoteIndex: journal watermark mismatch (expected trade_id=%s, "
                    "got %s) — performing full rebuild",
                    watermark_trade_id,
                    anchor.get("trade_id"),
                )
                self.rebuild_full(journal_path=path)
                return {"added": total, "updated": 0, "skipped": 0}

        if watermark_idx > total:
            # Journal shrank (manual edit?). Full rebuild.
            logger.warning(
                "AgentVoteIndex: journal shrank (watermark=%d > total=%d) — full rebuild",
                watermark_idx, total,
            )
            self.rebuild_full(journal_path=path)
            return {"added": total, "updated": 0, "skipped": 0}

        new_entries = entries[watermark_idx:]
        if not new_entries:
            return result

        with self._txn():
            for entry in new_entries:
                rows = _extract_rows(entry)
                for row in rows:
                    # ON CONFLICT REPLACE so re-running over the same entry is idempotent.
                    cur = self._conn.execute(
                        "INSERT INTO votes(trade_id, agent_name, passed, score, weight, "
                        "reason_text, regime, outcome, pnl_usd, closed_ts_iso) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(trade_id, agent_name) DO UPDATE SET "
                        "passed=excluded.passed, score=excluded.score, weight=excluded.weight, "
                        "reason_text=excluded.reason_text, regime=excluded.regime, "
                        "outcome=excluded.outcome, pnl_usd=excluded.pnl_usd, "
                        "closed_ts_iso=excluded.closed_ts_iso",
                        row,
                    )
                    # rowcount=1 for both INSERT and UPDATE on most SQLite versions; we can't
                    # easily distinguish added vs updated here. Aggregate as "added".
                    if cur.rowcount > 0:
                        result["added"] += 1
                    else:
                        result["skipped"] += 1

            # Advance watermark.
            last = new_entries[-1]
            self._meta_set("watermark_idx", str(total))
            self._meta_set("watermark_trade_id", str(last.get("trade_id", "")))
        return result

    def rebuild_full(self, journal_path: Optional[Path] = None) -> int:
        """Truncate the votes table and re-ingest the full journal.

        Used on first run and after a watermark mismatch (journal rewrite).
        Returns the number of journal entries processed.
        """
        path = Path(journal_path) if journal_path is not None else JOURNAL_PATH
        entries = self._load_journal(path)
        with self._txn():
            self._conn.execute("DELETE FROM votes")
            for entry in entries:
                for row in _extract_rows(entry):
                    self._conn.execute(
                        "INSERT OR REPLACE INTO votes(trade_id, agent_name, passed, score, "
                        "weight, reason_text, regime, outcome, pnl_usd, closed_ts_iso) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        row,
                    )
            self._meta_set("watermark_idx", str(len(entries)))
            if entries:
                self._meta_set("watermark_trade_id", str(entries[-1].get("trade_id", "")))
            else:
                self._meta_set("watermark_trade_id", "")
        return len(entries)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_votes(
        self,
        *,
        agent_name: Optional[str] = None,
        passed: Optional[int] = None,
        outcome: Optional[str] = None,
        regime: Optional[str] = None,
        since: Optional[str] = None,
        reason_fts: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Run a filtered query.

        :param agent_name: exact agent name (e.g. ``"mean_reversion"``).
        :param passed: ``0`` (vote was NO / failed) or ``1`` (vote was YES / passed).
        :param outcome: one of ``win``/``loss``/``breakeven``/``open``.
        :param regime: exact volatility regime (``LOW``/``NORMAL``/``HIGH``/``EXTREME``).
        :param since: ISO timestamp lower bound on ``closed_ts_iso``.
        :param reason_fts: FTS5 MATCH expression against ``reason_text``. Falls
            back to a LIKE match on malformed FTS input.
        :param limit: maximum rows.
        :return: list of row dicts, newest-first.
        """
        wheres: List[str] = []
        params: List[Any] = []

        if reason_fts:
            # Sanitize partial input the same way TradeSearchIndex does.
            fts_q = _to_fts_query(reason_fts)
            try:
                self._conn.execute(
                    "SELECT rowid FROM votes_fts WHERE votes_fts MATCH ? LIMIT 1",
                    (fts_q,),
                ).fetchall()
                wheres.append(
                    "votes.rowid IN (SELECT rowid FROM votes_fts WHERE votes_fts MATCH ?)"
                )
                params.append(fts_q)
            except sqlite3.OperationalError:
                logger.debug("AgentVoteIndex: FTS parse fallback for %r", reason_fts)
                wheres.append("LOWER(votes.reason_text) LIKE ?")
                params.append(f"%{reason_fts.lower()}%")

        if agent_name is not None:
            wheres.append("agent_name = ?")
            params.append(str(agent_name))
        if passed is not None:
            wheres.append("passed = ?")
            params.append(int(bool(passed)))
        if outcome is not None:
            wheres.append("outcome = ?")
            params.append(str(outcome))
        if regime is not None:
            wheres.append("regime = ?")
            params.append(str(regime))
        if since is not None:
            wheres.append("closed_ts_iso >= ?")
            params.append(str(since))

        sql = "SELECT * FROM votes"
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY closed_ts_iso DESC LIMIT ?"
        params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @contextmanager
    def _txn(self):
        """Explicit BEGIN/COMMIT/ROLLBACK around a write batch."""
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "AgentVoteIndex":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# FTS query sanitization (mirrors src/tui/trade_search_index._to_fts_query)
# ---------------------------------------------------------------------------

_FTS_SPECIALS = set('"():*^')


def _to_fts_query(user_input: str) -> str:
    """Translate free-form input to a safe FTS5 MATCH expression.

    Tokens with operator chars are stripped of those chars; AND-combined as
    prefix queries (``token*``). Empty result returns a sentinel that matches
    nothing — caller falls back to LIKE.
    """
    safe: List[str] = []
    for token in user_input.split():
        stripped = "".join(ch for ch in token if ch not in _FTS_SPECIALS)
        if not stripped:
            continue
        safe.append(f"{stripped.lower()}*")
    if not safe:
        return "zzz_no_match_zzz"
    return " AND ".join(safe)
