"""TrendJournalSync — RL-journal visibility for the ``com.buddy.trend`` lane.

Root cause (2026-07-02 audit): the trend daemon (``scripts/run_oanda_trend.py`` ->
``src/equity/oanda_trend.run_oanda_trend_cycle``) places real OANDA practice orders
directly against ``OandaPracticeClient`` and never calls
``ExecutionManager.execute_trade()`` — the only code path that writes an
``entry_record`` into ``trade_journal_rl.json``. It never ran the 15-agent
consensus team either (it is explicitly "NON-directional... price vs MA"), so no
``agent_verdicts`` ever existed for these trades. Because the trend lane never wrote
an open-side journal entry, the existing catch-up mechanism
(``src.scanner.automation.outcome_backfill.OutcomeBackfill``) could never help: it
only PATCHES an entry that is already present (looked up by ``trade_id``) — it does
not create new ones. The result: 57 real closed trades (2026-06-25..07-01) landed in
``trained_data/oanda/transactions.jsonl`` (via ``TransactionLedger``) but never in
``trade_journal_rl.json``, and never reached
``ScannerAgentTeam.update_weights_from_outcome()``.

This module closes the *visibility* gap going forward without touching the
delicate, actively-changing risk-gate code in ``oanda_trend.py``: it walks
``ORDER_FILL`` transactions via ``/transactions/sinceid`` (same fetcher contract as
``OutcomeBackfill``), and for each fill:

  * ``tradeOpened`` (trade opened by this lane) -> creates a new journal entry,
    explicitly tagged ``lane="trend"`` and ``rl_eligible=False``, with
    ``agents.agent_votes = None`` (never fabricated).
  * ``tradesClosed`` (trade closed) -> patches ``outcome`` / ``realized_pl`` /
    ``close_time`` / ``close_reason`` on the matching entry, same schema as
    ``OutcomeBackfill._apply_closure``.

HARD RULE: this module MUST NEVER call
``ScannerAgentTeam.update_weights_from_outcome`` — there are no agent verdicts for
trend-lane trades, and applying RL credit/blame to agents that never voted on the
decision would be a category error, not a fix. ``sync_closed_trades_rl`` (the
scanner's own RL-sync path in ``ExecutionManager``) remains the ONLY caller of
``update_weights_from_outcome``.

Never fails the caller: every OANDA/HTTP/JSON error is caught, logged, and
``run_once`` returns a result with ``.error`` set rather than raising.

Enrichment (2026-07-03, P0 from docs/training-architecture-audit-2026-07-03.md):
trend records are nulls-by-design for agents/gates, but they CAN cheaply carry
market context that exists at sync time. Every field below is honestly sourced —
null when unavailable, never fabricated, no new broker calls:

  Open-side (source: the open ORDER_FILL transaction itself):
    * ``entry_bid`` / ``entry_ask`` / ``entry_spread`` — top-of-book from
      ``fullPrice.bids[0]/asks[0]``; spread is ask-bid in RAW PRICE UNITS
      (deliberately NOT converted to pips: pip size varies across FX/JPY/metal/
      CFD instruments and a wrong conversion would be fabrication).
    * ``entry_half_spread_cost`` — OANDA ``halfSpreadCost`` (account home ccy).
    * ``account_balance`` — OANDA ``accountBalance`` at fill time.
    * ``requested_units`` — OANDA ``requestedUnits`` (vs filled ``units``).
    * ``atr_stop_distance`` — joined by trade_id from
      ``trained_data/oanda/risk_state.json`` (``r_distance``: the entry-anchored
      ATR*mult stop distance the trend lane persists at entry, rule-5 bootstrap
      in ``src/equity/trend_risk_gates.py``). Read from disk only.
    * ``regime`` — dict-shaped with ``volatility_regime=None``: the trend lane
      runs NO regime classifier, so the value is honestly null; the dict shape
      (vs bare None) keeps ``entry.get("regime", {}).get(...)``-style consumers
      (e.g. execution.py sync_closed_trades_rl outcome_data) from raising
      AttributeError on lane-crossover reads.

  Close-side (source: the close ORDER_FILL transaction):
    * ``exit_price`` — ``tradesClosed[].price``.
    * ``close_bid`` / ``close_ask`` / ``close_spread`` — top-of-book of the
      CLOSE fill's ``fullPrice`` (instrument spread at close).
    * ``close_half_spread_cost`` — ``tradesClosed[].halfSpreadCost``.
    * ``financing`` — ``tradesClosed[].financing`` (account home ccy).
    * ``close_reason_raw`` — unmapped OANDA reason string.

  Partial closes (``tradeReduced`` fills — the trend lane rebalances position
  sizes, so partial reduces are routine): accumulated idempotently (by fill tx
  id) into ``partial_realized_pl`` / ``partial_close_tx_ids``;
  ``realized_pl_total`` = partials + final ``tradesClosed`` realizedPL, and
  ``outcome`` is derived from the TOTAL (the final fill alone understates P&L
  on partially-closed trades). ``realized_pl`` keeps its original meaning (the
  final fill's realizedPL) for backward compatibility with OutcomeBackfill's
  schema.

  Deliberately NOT added: slippage (the trend lane places market orders with no
  recorded requested price — any slippage number would be fabricated) and
  pip-converted spreads (see above).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.scanner.automation.safe_json import _FileLock

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 1000
OANDA_TIMEOUT = (5, 30)


def _f_or_none(value: Any) -> Optional[float]:
    """float(value) or None — never a fabricated 0.0 for a missing field."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_REASON_MAP = {
    "STOP_LOSS_ORDER": "SL",
    "TAKE_PROFIT_ORDER": "TP",
    "TRAILING_STOP_LOSS_ORDER": "SL",
    "MARKET_ORDER_TRADE_CLOSE": "manual",
    "MARKET_ORDER_POSITION_CLOSEOUT": "manual",
    "MARKET_ORDER_MARGIN_CLOSEOUT": "manual",
}


@dataclass
class TrendSyncResult:
    """Return value from :meth:`TrendJournalSync.run_once`."""

    opened: int = 0
    closed: int = 0
    partials: int = 0
    skipped_existing: List[str] = field(default_factory=list)
    unmatched_close_trade_ids: List[str] = field(default_factory=list)
    last_tx_id: Optional[str] = None
    pages: int = 0
    error: Optional[str] = None


class TrendJournalSync:
    """Creates/patches trade_journal_rl.json entries for the trend lane only.

    Deliberately separate from ``OutcomeBackfill`` (which only patches existing
    entries) and from ``ExecutionManager.sync_closed_trades_rl`` (which owns the
    RL weight-update path for agent-consensus trades). This class never touches
    agent weights.
    """

    def __init__(
        self,
        *,
        journal_path: Optional[Path] = None,
        cursor_path: Optional[Path] = None,
        transactions_fetcher: Optional[
            Callable[[str], Tuple[List[Dict[str, Any]], Optional[str]]]
        ] = None,
        project_root: Optional[Path] = None,
        risk_state_path: Optional[Path] = None,
    ) -> None:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self._journal_path = (
            Path(journal_path)
            if journal_path
            else root / "trained_data" / "trade_journal_rl.json"
        )
        self._cursor_path = (
            Path(cursor_path)
            if cursor_path
            else root / ".claude" / "trend_journal_sync_cursor.json"
        )
        # Entry-anchored ATR stop distances the trend lane persists at entry
        # (src/equity/trend_risk_gates.py rule-5 bootstrap). Disk read only.
        self._risk_state_path = (
            Path(risk_state_path)
            if risk_state_path
            else root / "trained_data" / "oanda" / "risk_state.json"
        )
        self._fetch = transactions_fetcher or self._default_fetcher

    # --------------------------------------------------------------- public API

    def run_once(self) -> TrendSyncResult:
        """Catch-up pass. Never raises.

        1. Read cursor ``last_processed_tx_id`` (default 0)
        2. Paginate ``/transactions/sinceid``
        3. For each ORDER_FILL with ``tradeOpened``: create a journal entry
           (lane="trend", rl_eligible=False) if trade_id not already present.
        4. For each ORDER_FILL with ``tradesClosed``: patch outcome on the
           matching entry IF that entry is lane="trend" (never touches
           scanner-owned entries).
        5. Persist journal atomically + advance cursor.
        """
        try:
            last_id = self._read_cursor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrendJournalSync: cursor read failed (%s) — starting at 0", exc)
            last_id = "0"

        try:
            pages = self._collect_pages(last_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrendJournalSync: transaction fetch failed: %s", exc)
            return TrendSyncResult(last_tx_id=last_id, error=str(exc))

        transactions: List[Dict[str, Any]] = []
        for page_txs, _next in pages:
            transactions.extend(page_txs)

        result = TrendSyncResult(last_tx_id=last_id, pages=len(pages))
        if not transactions:
            return result

        new_cursor = self._max_tx_id(transactions, fallback=last_id)

        # Read-modify-write under the SAME lock file safe_json_write uses for
        # this journal (trade_journal_rl.json.lock), so this daemon's RMW
        # serializes against ExecutionManager.sync_closed_trades_rl writes in
        # the scanner process (two-writer collision is a known incident class
        # in this repo). NOTE: execution.py's own read side is not lock-held,
        # so its RMW can still lose our update — fixing that requires an
        # execution.py change (reported, out of this module's scope).
        try:
            with _FileLock(self._journal_path, exclusive=True):
                try:
                    journal = self._load_journal()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TrendJournalSync: journal load failed: %s", exc)
                    result.error = str(exc)
                    return result

                dirty = self._apply_transactions(journal, transactions, result)

                if dirty:
                    self._write_journal(journal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrendJournalSync: journal update failed: %s", exc)
            result.error = str(exc)
            return result

        result.last_tx_id = new_cursor

        try:
            self._write_cursor(new_cursor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrendJournalSync: cursor write failed: %s", exc)
            result.error = str(exc)

        if result.opened or result.closed or result.partials:
            logger.info(
                "TrendJournalSync: opened=%d closed=%d partials=%d cursor %s -> %s",
                result.opened, result.closed, result.partials, last_id, new_cursor,
            )
        return result

    # ----------------------------------------------------------------- internal

    def _apply_transactions(
        self,
        journal: List[Dict[str, Any]],
        transactions: List[Dict[str, Any]],
        result: TrendSyncResult,
    ) -> bool:
        """Apply opens, partial closes, then full closes. Returns dirty flag."""
        by_trade = self._index_journal(journal)
        opens = self._extract_opens(transactions)
        partials = self._extract_partial_closes(transactions)
        closes = self._extract_closures(transactions)
        risk_state = self._load_risk_state()

        dirty = False
        for op in opens:
            tid = op["trade_id"]
            if tid in by_trade:
                result.skipped_existing.append(tid)
                continue
            entry = self._build_open_entry(op, risk_state)
            journal.append(entry)
            by_trade[tid] = entry
            result.opened += 1
            dirty = True

        # Partials BEFORE closes so a same-batch final close totals correctly.
        for partial in partials:
            entry = by_trade.get(partial["trade_id"])
            if entry is None or entry.get("lane") != "trend":
                continue
            if self._apply_partial_close(entry, partial):
                result.partials += 1
                dirty = True

        for closure in closes:
            tid = closure["trade_id"]
            entry = by_trade.get(tid)
            if entry is None:
                result.unmatched_close_trade_ids.append(tid)
                continue
            if entry.get("lane") != "trend":
                # Not ours to patch — the scanner's own sync_closed_trades_rl /
                # OutcomeBackfill path owns non-trend entries.
                continue
            self._apply_closure(entry, closure, risk_state)
            result.closed += 1
            dirty = True
        return dirty

    def _collect_pages(
        self, last_id: str
    ) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        pages: List[Tuple[List[Dict[str, Any]], Optional[str]]] = []
        cursor = last_id
        for _ in range(100):  # safety cap against runaway pagination
            txs, next_cursor = self._fetch(cursor)
            pages.append((txs, next_cursor))
            if not next_cursor or not txs:
                break
            cursor = next_cursor
        return pages

    @staticmethod
    def _top_of_book(
        tx: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """(bid, ask, spread) from an ORDER_FILL's ``fullPrice`` top-of-book.

        Spread is ask-bid in RAW PRICE UNITS — deliberately not pip-converted
        (pip size varies across FX/JPY/metal/CFD instruments; a wrong
        conversion would be fabrication). All None when fullPrice is absent.
        """
        fp = tx.get("fullPrice") or {}

        def _first(side: str) -> Optional[float]:
            arr = fp.get(side) or []
            if not arr or not isinstance(arr[0], dict):
                return None
            return _f_or_none(arr[0].get("price"))

        bid = _first("bids")
        ask = _first("asks")
        spread = (ask - bid) if (bid is not None and ask is not None) else None
        return bid, ask, spread

    @staticmethod
    def _extract_opens(transactions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        opens: List[Dict[str, Any]] = []
        for tx in transactions:
            if tx.get("type") != "ORDER_FILL":
                continue
            opened = tx.get("tradeOpened")
            if not opened:
                continue
            tid = opened.get("tradeID") or opened.get("tradeId")
            if not tid:
                continue
            try:
                price = float(opened.get("price") or tx.get("price") or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            try:
                units = float(opened.get("units") or tx.get("units") or 0.0)
            except (TypeError, ValueError):
                units = 0.0
            bid, ask, spread = TrendJournalSync._top_of_book(tx)
            opens.append({
                "trade_id": str(tid),
                "instrument": tx.get("instrument", ""),
                "price": price,
                "units": units,
                "time": tx.get("time") or "",
                # Enrichment — all sourced from THIS open ORDER_FILL; None when
                # the field is absent from the transaction (never fabricated).
                "entry_bid": bid,
                "entry_ask": ask,
                "entry_spread": spread,
                "entry_half_spread_cost": _f_or_none(tx.get("halfSpreadCost")),
                "account_balance": _f_or_none(tx.get("accountBalance")),
                "requested_units": _f_or_none(tx.get("requestedUnits")),
            })
        return opens

    @staticmethod
    def _extract_partial_closes(
        transactions: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """``tradeReduced`` fills (partial position reductions).

        The trend lane rebalances position sizes, so partial reduces are
        routine; their realizedPL never appears in the final ``tradesClosed``
        fill and would otherwise be silently dropped from the journal.
        """
        out: List[Dict[str, Any]] = []
        for tx in transactions:
            if tx.get("type") != "ORDER_FILL":
                continue
            reduced = tx.get("tradeReduced")
            if not isinstance(reduced, dict):
                continue
            tid = reduced.get("tradeID") or reduced.get("tradeId")
            if not tid:
                continue
            out.append({
                "tx_id": str(tx.get("id") or ""),
                "trade_id": str(tid),
                "realized_pl": _f_or_none(reduced.get("realizedPL")) or 0.0,
                "time": tx.get("time") or "",
            })
        return out

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
            close_bid, close_ask, close_spread = TrendJournalSync._top_of_book(tx)
            for tc in closed:
                try:
                    realized_pl = float(tc.get("realizedPL") or 0.0)
                except (TypeError, ValueError):
                    realized_pl = 0.0
                tid = tc.get("tradeID") or tc.get("tradeId")
                if not tid:
                    continue
                closures.append({
                    "trade_id": str(tid),
                    "realized_pl": realized_pl,
                    "close_time": close_time,
                    "close_reason": close_reason,
                    "reason_raw": reason_raw,
                    # Enrichment — all sourced from THIS close ORDER_FILL;
                    # None when absent (never fabricated).
                    "exit_price": _f_or_none(tc.get("price")),
                    "close_half_spread_cost": _f_or_none(tc.get("halfSpreadCost")),
                    "financing": _f_or_none(tc.get("financing")),
                    "close_bid": close_bid,
                    "close_ask": close_ask,
                    "close_spread": close_spread,
                })
        return closures

    @staticmethod
    def _atr_stop_from_risk_state(
        risk_state: Dict[str, Any], trade_id: str,
    ) -> Optional[float]:
        """Entry-anchored ATR*mult stop distance the trend lane persisted at
        entry (trained_data/oanda/risk_state.json, keyed by trade_id).
        None when the file or trade is absent — never fabricated."""
        rec = risk_state.get(str(trade_id)) if isinstance(risk_state, dict) else None
        if not isinstance(rec, dict):
            return None
        return _f_or_none(rec.get("r_distance"))

    @staticmethod
    def _build_open_entry(
        op: Dict[str, Any], risk_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Journal entry for a trend-lane open. No agent votes exist — never
        fabricate them. rl_eligible=False is the explicit signal that
        sync_closed_trades_rl / update_weights_from_outcome must skip this trade.

        Field sources are documented in the module docstring; every enrichment
        field is None when its source data is unavailable."""
        direction = "LONG" if op["units"] >= 0 else "SHORT"
        return {
            "trade_id": op["trade_id"],
            "lane": "trend",
            "rl_eligible": False,
            "timestamp": op["time"] or datetime.now(timezone.utc).isoformat(),
            "pair": op["instrument"],
            "direction": direction,
            "entry_price": op["price"],
            "units": op["units"],
            "confidence": None,
            "gates": None,
            "agents": {"agent_votes": None, "note": "trend lane — no 15-agent consensus run"},
            # Dict-shaped (not bare None) so `.get("regime", {}).get(...)`-style
            # consumers don't AttributeError; the classification itself is
            # honestly null — the trend lane runs no regime classifier.
            "regime": {
                "volatility_regime": None,
                "note": "trend lane — no volatility-regime classification",
            },
            "outcome": None,
            "source": "trend_journal_sync",
            # Enrichment (source: the open ORDER_FILL transaction).
            "entry_bid": op.get("entry_bid"),
            "entry_ask": op.get("entry_ask"),
            "entry_spread": op.get("entry_spread"),
            "entry_half_spread_cost": op.get("entry_half_spread_cost"),
            "account_balance": op.get("account_balance"),
            "requested_units": op.get("requested_units"),
            # Enrichment (source: trained_data/oanda/risk_state.json — the
            # entry-anchored ATR*mult stop distance the trend lane persists).
            "atr_stop_distance": TrendJournalSync._atr_stop_from_risk_state(
                risk_state, op["trade_id"],
            ),
        }

    @staticmethod
    def _apply_partial_close(entry: Dict[str, Any], partial: Dict[str, Any]) -> bool:
        """Accumulate a tradeReduced fill's realizedPL. Idempotent by fill tx
        id (re-fetching the same transaction window never double-counts).
        Returns True if the entry changed."""
        tx_id = partial.get("tx_id") or ""
        seen = entry.setdefault("partial_close_tx_ids", [])
        if tx_id and tx_id in seen:
            return False
        if tx_id:
            seen.append(tx_id)
        prev = _f_or_none(entry.get("partial_realized_pl")) or 0.0
        entry["partial_realized_pl"] = round(prev + float(partial["realized_pl"]), 6)
        return True

    @staticmethod
    def _apply_closure(
        entry: Dict[str, Any],
        closure: Dict[str, Any],
        risk_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        realized_pl = float(closure["realized_pl"])
        # Total = accumulated partial-close P&L + the final fill's realizedPL.
        # `realized_pl` keeps the final fill's value (OutcomeBackfill schema
        # compat); `outcome` reflects the honest TOTAL.
        partial_pl = _f_or_none(entry.get("partial_realized_pl")) or 0.0
        total_pl = round(realized_pl + partial_pl, 6)
        entry["outcome"] = "win" if total_pl > 0 else "loss"
        entry["realized_pl"] = realized_pl
        entry["realized_pl_total"] = total_pl
        entry["close_time"] = closure["close_time"]
        entry["close_reason"] = closure["close_reason"]
        # Enrichment (source: the close ORDER_FILL transaction; None if absent).
        entry["close_reason_raw"] = closure.get("reason_raw") or None
        entry["exit_price"] = closure.get("exit_price")
        entry["close_bid"] = closure.get("close_bid")
        entry["close_ask"] = closure.get("close_ask")
        entry["close_spread"] = closure.get("close_spread")
        entry["close_half_spread_cost"] = closure.get("close_half_spread_cost")
        entry["financing"] = closure.get("financing")
        # Late join: entries opened before risk_state.json existed (or before
        # this enrichment shipped) get the ATR stop distance backfilled here.
        if entry.get("atr_stop_distance") is None and risk_state:
            entry["atr_stop_distance"] = TrendJournalSync._atr_stop_from_risk_state(
                risk_state, str(entry.get("trade_id") or ""),
            )
        entry["backfilled_at"] = datetime.now(timezone.utc).isoformat()
        entry["backfill_source"] = "trend_journal_sync"

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
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._cursor_path)

    def _load_journal(self) -> List[Dict[str, Any]]:
        if not self._journal_path.exists():
            return []
        with self._journal_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            # A parseable-but-wrong-shape journal must NOT be silently replaced
            # with only this run's trend entries — that would clobber a
            # financial artifact. Refuse loudly; run_once surfaces the error.
            raise ValueError(
                f"journal at {self._journal_path} is not a list "
                f"(got {type(data).__name__}) — refusing to rewrite"
            )
        return data

    def _load_risk_state(self) -> Dict[str, Any]:
        """trained_data/oanda/risk_state.json — trend lane's entry-anchored
        per-trade risk state. Missing/corrupt => {} (enrichment fields stay
        None; never blocks the sync)."""
        try:
            data = json.loads(self._risk_state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_journal(self, journal: List[Dict[str, Any]]) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        # .bak of the previous version (same convention as safe_json_write) so
        # a bad write of this financial artifact is recoverable.
        if self._journal_path.exists():
            try:
                bak = self._journal_path.with_suffix(self._journal_path.suffix + ".bak")
                bak.write_text(
                    self._journal_path.read_text(encoding="utf-8"), encoding="utf-8",
                )
            except OSError as exc:
                logger.debug("TrendJournalSync: journal .bak failed: %s", exc)
        tmp = self._journal_path.with_suffix(self._journal_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(journal, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._journal_path)

    # ----------------------------------------------------------- OANDA fetcher

    def _default_fetcher(
        self, since_id: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """GET /v3/accounts/{id}/transactions/sinceid?id=<since_id>.

        Returns ``(transactions, next_since_id_or_None)``. Mirrors
        ``OutcomeBackfill._default_fetcher`` exactly (same pagination contract).
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
