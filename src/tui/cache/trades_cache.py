"""Tier 2 T9: Two-tier trade journal cache — precomputed summary rows.

Maintains a small JSON-on-disk cache of `TradeRow` summaries derived from
``trained_data/trade_journal_rl.json`` so the Trades screen does not need to
re-parse the full journal on every refresh.

Sync semantics:
* First call: load cache from disk if present and valid; otherwise rebuild
  from the journal and persist.
* Subsequent calls: compare current journal byte-size against the size we
  recorded at the last rebuild. If unchanged, no-op. If changed (or unreadable),
  rebuild fully from the journal and re-persist.
* Corrupt cache (invalid JSON, wrong shape) is recovered transparently by
  rebuilding from the journal — never crashes the caller.

Atomic writes: the cache file is written via tmp + ``Path.replace`` to avoid
partial files on crash.

Journal shape handled:
* ``{"trades": [...]}``  (legacy/test shape)
* ``[...]``               (current ``trade_journal_rl.json`` on disk)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeRow:
    """Summary projection of one trade journal entry.

    Frozen so callers cannot mutate cached rows accidentally; rebuild by
    calling ``TradesCache.sync()`` again.
    """

    id: int
    pair: str
    direction: str
    pnl: float
    opened_at: str
    closed_at: str
    outcome: str


class TradesCache:
    """Maintains a precomputed trade-summary index synced incrementally."""

    def __init__(self, *, journal_path: Path, cache_path: Path) -> None:
        self._journal_path = Path(journal_path)
        self._cache_path = Path(cache_path)
        self._rows: List[TradeRow] = []
        self._raw_entries: List[dict] = []
        self._last_journal_size: int = 0
        self.sync_count: int = 0
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_cache(self) -> Optional[List[TradeRow]]:
        """Read the persisted cache, returning ``None`` if missing or corrupt.

        Side effect: also populates ``self._raw_entries`` on hit so the
        Trades screen can render rich detail without re-parsing the
        journal on a cache hit.
        """
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            rows = [TradeRow(**r) for r in data.get("rows", [])]
            self._last_journal_size = int(data.get("journal_size", 0))
            raw = data.get("raw_entries", [])
            self._raw_entries = [r for r in raw if isinstance(r, dict)]
            return rows
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            # ``TypeError`` covers TradeRow(**r) shape mismatches.
            # ``ValueError`` covers int() coercion on a garbage journal_size.
            logger.debug("TradesCache: cache load failed (rebuilding): %s", e)
            return None

    def _persist_cache(self) -> None:
        """Atomically write the cache to disk (tmp + replace)."""
        payload = {
            "rows": [asdict(r) for r in self._rows],
            "raw_entries": self._raw_entries,
            "journal_size": self._last_journal_size,
        }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self._cache_path)
        except OSError as e:
            logger.warning("TradesCache: persist failed: %s", e)

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_trades(raw: Any) -> List[dict]:
        """Tolerate both supported journal shapes.

        ``trade_journal_rl.json`` on disk is a bare list. The plan's test
        fixtures (and some legacy artefacts) use ``{"trades": [...]}``.
        """
        if isinstance(raw, list):
            return [t for t in raw if isinstance(t, dict)]
        if isinstance(raw, dict):
            trades = raw.get("trades", [])
            if isinstance(trades, list):
                return [t for t in trades if isinstance(t, dict)]
        return []

    def _rebuild_from_journal(self) -> None:
        try:
            raw = json.loads(self._journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("TradesCache: journal read failed (%s) — empty rebuild", e)
            self._rows = []
            self._last_journal_size = 0
            return

        trades = self._extract_trades(raw)
        self._raw_entries = trades
        rebuilt: List[TradeRow] = []
        for t in trades:
            try:
                rebuilt.append(
                    TradeRow(
                        id=int(t.get("id", t.get("trade_id", 0)) or 0),
                        pair=str(t.get("pair", "")),
                        direction=str(t.get("direction", "")),
                        pnl=float(t.get("pnl", 0.0) or 0.0),
                        opened_at=str(t.get("opened_at", t.get("timestamp", ""))),
                        closed_at=str(t.get("closed_at", t.get("close_time", ""))),
                        outcome=str(t.get("outcome", "")),
                    )
                )
            except (TypeError, ValueError) as e:
                # Individual row is malformed (e.g. pnl is a dict); skip but
                # keep building the rest of the index.
                logger.debug("TradesCache: skipping malformed trade entry: %s", e)
                continue
        self._rows = rebuilt

        try:
            self._last_journal_size = self._journal_path.stat().st_size
        except OSError:
            self._last_journal_size = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Idempotent sync. Only does work if the journal grew (or no cache)."""
        if not self._loaded:
            cached = self._load_cache()
            if cached is not None:
                self._rows = cached
            else:
                self._rebuild_from_journal()
                self._persist_cache()
                self.sync_count += 1
            self._loaded = True
            return

        try:
            current_size = self._journal_path.stat().st_size
        except OSError:
            return
        if current_size == self._last_journal_size:
            return
        self._rebuild_from_journal()
        self._persist_cache()
        self.sync_count += 1

    def rows(self) -> List[TradeRow]:
        """Return a copy of the cached summary rows (defensive)."""
        return list(self._rows)

    def raw_entries(self) -> List[dict]:
        """Return the raw journal dict entries that the rows were derived from.

        The Trades screen still wants the full per-entry shape for its rich
        rendering (exit_price, realized_pl, trade_won, duration_minutes,
        trade_id) — those fields aren't on ``TradeRow``. Exposing them here
        keeps the screen from re-parsing the journal on every refresh.

        Raw entries are persisted alongside the summary ``rows`` so a
        cache-hit (cache file valid, journal size unchanged) restores them
        without touching the journal.
        """
        return list(self._raw_entries)
