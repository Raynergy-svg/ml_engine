"""
Outcome Backfill — US-605.

On every EmbeddedScanner startup, query the OANDA transactions endpoint since
the last-processed transaction id and backfill ``trade_journal_rl.json`` with
``outcome`` / ``realized_pl`` / ``close_time`` / ``close_reason`` for any entries
that closed while the bot was offline.

The 2026-04-24 audit surfaced 10 OANDA closes that never made it back to the
journal because the RL sync loop only writes outcomes for trades closed during
an active process lifetime. This module is the catch-up pass.

Never fails boot: every OANDA/HTTP/JSON error is caught, logged, and the
startup continues.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 1000
OANDA_TIMEOUT = (5, 30)

_REASON_MAP = {
    "STOP_LOSS_ORDER": "SL",
    "TAKE_PROFIT_ORDER": "TP",
    "TRAILING_STOP_LOSS_ORDER": "SL",
    "MARKET_ORDER_TRADE_CLOSE": "manual",
    "MARKET_ORDER_POSITION_CLOSEOUT": "manual",
    "MARKET_ORDER_MARGIN_CLOSEOUT": "manual",
}


@dataclass
class BackfillResult:
    """Return value from :meth:`OutcomeBackfill.run_once`."""

    matched: int = 0
    unmatched_trade_ids: List[str] = field(default_factory=list)
    processed_tx_ids: List[str] = field(default_factory=list)
    last_tx_id: Optional[str] = None
    pages: int = 0
    error: Optional[str] = None


class OutcomeBackfill:
    """Reads OANDA transactions since last cursor and patches journal entries."""

    def __init__(
        self,
        *,
        journal_path: Optional[Path] = None,
        cursor_path: Optional[Path] = None,
        transactions_fetcher: Optional[
            Callable[[str], Tuple[List[Dict[str, Any]], Optional[str]]]
        ] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self._journal_path = (
            Path(journal_path)
            if journal_path
            else root / "trained_data" / "trade_journal_rl.json"
        )
        self._cursor_path = (
            Path(cursor_path) if cursor_path else root / ".claude" / "last_tx_id.json"
        )
        self._fetch = transactions_fetcher or self._default_fetcher

    # --------------------------------------------------------------- public API

    def run_once(self) -> BackfillResult:
        """Catch-up pass. Never raises.

        1. Read cursor ``last_processed_tx_id`` (default 0)
        2. Paginate ``/transactions/sinceid``
        3. For each ``ORDER_FILL`` with ``tradesClosed``, update the journal entry
        4. Persist journal atomically + advance cursor
        """
        try:
            last_id = self._read_cursor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomeBackfill: cursor read failed (%s) — starting at 0", exc)
            last_id = "0"

        try:
            pages = self._collect_pages(last_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomeBackfill: transaction fetch failed: %s", exc)
            return BackfillResult(last_tx_id=last_id, error=str(exc))

        transactions: List[Dict[str, Any]] = []
        for page_txs, _next in pages:
            transactions.extend(page_txs)

        result = BackfillResult(last_tx_id=last_id, pages=len(pages))

        closures = self._extract_closures(transactions)
        if not closures and not transactions:
            return result

        try:
            journal = self._load_journal()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomeBackfill: journal load failed: %s", exc)
            result.error = str(exc)
            return result

        by_trade = self._index_journal(journal)

        for closure in closures:
            tid = str(closure.get("trade_id") or "")
            if not tid:
                continue
            entry = by_trade.get(tid)
            if entry is None:
                result.unmatched_trade_ids.append(tid)
                logger.info("OutcomeBackfill: trade_id %s not in journal — skipping", tid)
                continue
            self._apply_closure(entry, closure)
            result.matched += 1

        # Advance cursor to the max transaction id we observed (even if nothing matched).
        new_cursor = self._max_tx_id(transactions, fallback=last_id)
        result.processed_tx_ids = [str(t.get("id")) for t in transactions if t.get("id")]
        result.last_tx_id = new_cursor

        try:
            if result.matched:
                self._write_journal(journal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomeBackfill: journal write failed: %s", exc)
            result.error = str(exc)
            return result

        try:
            self._write_cursor(new_cursor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OutcomeBackfill: cursor write failed: %s", exc)
            result.error = str(exc)

        if result.matched:
            logger.info(
                "OutcomeBackfill: backfilled %d trade(s); cursor %s -> %s",
                result.matched,
                last_id,
                new_cursor,
            )
        return result

    # ----------------------------------------------------------------- internal

    def _collect_pages(
        self, last_id: str
    ) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        pages: List[Tuple[List[Dict[str, Any]], Optional[str]]] = []
        cursor = last_id
        seen = 0
        # Safety cap to prevent runaway pagination.
        for _ in range(100):
            txs, next_cursor = self._fetch(cursor)
            pages.append((txs, next_cursor))
            seen += len(txs)
            if not next_cursor or not txs:
                break
            cursor = next_cursor
        return pages

    @staticmethod
    def _extract_closures(transactions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        closures: List[Dict[str, Any]] = []
        for tx in transactions:
            if tx.get("type") != "ORDER_FILL":
                continue
            closed = tx.get("tradesClosed") or []
            if not closed:
                continue
            reason_raw = str(tx.get("reason") or "")
            close_reason = _REASON_MAP.get(reason_raw, "manual")
            close_time = tx.get("time") or ""
            for tc in closed:
                try:
                    realized_pl = float(tc.get("realizedPL") or 0.0)
                except (TypeError, ValueError):
                    realized_pl = 0.0
                tid = tc.get("tradeID") or tc.get("tradeId")
                if not tid:
                    continue
                closures.append(
                    {
                        "trade_id": str(tid),
                        "realized_pl": realized_pl,
                        "close_time": close_time,
                        "close_reason": close_reason,
                        "reason_raw": reason_raw,
                    }
                )
        return closures

    @staticmethod
    def _index_journal(journal: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for entry in journal:
            tid = entry.get("trade_id")
            if tid is None:
                continue
            out[str(tid)] = entry
        return out

    @staticmethod
    def _apply_closure(entry: Dict[str, Any], closure: Dict[str, Any]) -> None:
        realized_pl = float(closure["realized_pl"])
        outcome = "win" if realized_pl > 0 else "loss"
        # Top-level fields per US-605 AC.
        entry["outcome"] = outcome
        entry["realized_pl"] = realized_pl
        entry["close_time"] = closure["close_time"]
        entry["close_reason"] = closure["close_reason"]
        entry["backfilled_at"] = datetime.now(timezone.utc).isoformat()
        entry["backfill_source"] = "oanda_tx_stream"

    @staticmethod
    def _max_tx_id(transactions: Iterable[Dict[str, Any]], fallback: str) -> str:
        best: Optional[int] = None
        for tx in transactions:
            raw = tx.get("id")
            try:
                val = int(str(raw))
            except (TypeError, ValueError):
                continue
            if best is None or val > best:
                best = val
        if best is None:
            return fallback
        try:
            fallback_int = int(str(fallback))
        except (TypeError, ValueError):
            fallback_int = 0
        return str(max(best, fallback_int))

    # ------------------------------------------------------------------- I/O

    def _read_cursor(self) -> str:
        if not self._cursor_path.exists():
            return "0"
        with self._cursor_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        return str(data.get("last_processed_tx_id") or "0")

    def _write_cursor(self, tx_id: str) -> None:
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cursor_path.with_suffix(self._cursor_path.suffix + ".tmp")
        payload = {
            "last_processed_tx_id": str(tx_id),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._cursor_path)

    def _load_journal(self) -> List[Dict[str, Any]]:
        if not self._journal_path.exists():
            return []
        with self._journal_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        return data

    def _write_journal(self, journal: List[Dict[str, Any]]) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._journal_path.with_suffix(self._journal_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(journal, fh, indent=2, sort_keys=True)
        os.replace(tmp, self._journal_path)

    # ----------------------------------------------------------- OANDA fetcher

    def _default_fetcher(
        self, since_id: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """GET /v3/accounts/{id}/transactions/sinceid?id=<since_id>.

        Returns ``(transactions, next_since_id_or_None)``. ``next_since_id``
        is set when the response indicates there are more pages.
        """
        import requests  # type: ignore[import-untyped]  # lazy import

        token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")
        if not token or not acct:
            raise RuntimeError("OANDA_API_TOKEN / OANDA_ACCOUNT_ID not set")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{base}/v3/accounts/{acct}/transactions/sinceid"
        resp = requests.get(
            url,
            headers=headers,
            params={"id": str(since_id), "pageSize": DEFAULT_PAGE_SIZE},
            timeout=OANDA_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json() or {}
        txs = body.get("transactions") or []
        last_tx_id = body.get("lastTransactionID")
        if not txs:
            return [], None
        try:
            max_seen = max(int(str(t.get("id") or 0)) for t in txs)
        except ValueError:
            max_seen = 0
        next_cursor: Optional[str] = None
        if last_tx_id is not None:
            try:
                if int(str(last_tx_id)) > max_seen:
                    next_cursor = str(max_seen)
            except ValueError:
                next_cursor = None
        return txs, next_cursor
