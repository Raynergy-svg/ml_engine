"""Robust OANDA v20 streaming + transaction-ledger layer (PRACTICE only).

Complements the REST ``OandaPracticeClient`` with the pieces a live demo needs:
streaming prices (persistent connection + auto-reconnect, not poll-storm), the
full transaction ledger (the audit trail: fills, financing, every state change)
surfaced to disk for the TUI, account/positions snapshots, and order-book /
position-book SENTIMENT as DATA-ONLY ingestion (flagged available; NOT wired into
the strategy — reserved for a future pre-registered contrarian test).

Clean separation: this is the DATA / LEDGER layer. Strategy lives in
``equity.oanda_trend``; execution is ``OandaPracticeClient.create_market_order``.

HARD LINE — PRACTICE ONLY. Both hosts are hard-pinned to the fxPractice domain
(``api-fxpractice`` / ``stream-fxpractice``); ``_assert_practice`` refuses any URL
containing ``fxtrade`` (the live/real-money host). No live path exists here.

Good-citizen discipline (recalling the earlier retry-storm): exponential backoff
with a cap, a minimum reconnect interval honoring OANDA's "max 2 new connections/
sec", and 429/5xx backoff. Heartbeats (every ~5s) are consumed and ignored.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Hard-pinned PRACTICE hosts (v20). No fxtrade/live constant exists in this module.
REST_PRACTICE_HOST = "https://api-fxpractice.oanda.com"
STREAM_PRACTICE_HOST = "https://stream-fxpractice.oanda.com"
_API = "/v3"

MIN_RECONNECT_S = 0.6        # OANDA allows <=2 new connections/sec -> >=0.5s apart
MAX_BACKOFF_S = 30.0
HEARTBEAT_TYPES = {"HEARTBEAT", "PRICE_HEARTBEAT", "TransactionHeartbeat", "PricingHeartbeat"}

LEDGER_DIR_DEFAULT = "trained_data/oanda"


class OandaV20Error(RuntimeError):
    pass


def _assert_practice(url: str) -> None:
    """Fail-closed: refuse anything that could reach the live/real-money host."""
    if "fxtrade" in url or "fxpractice" not in url:
        raise OandaV20Error(f"HARD LINE: non-practice OANDA URL refused: {url}")


# ---------------------------------------------------------------------------
# Stream consumption (pure, testable: parse chunked JSON lines -> typed dicts)
# ---------------------------------------------------------------------------
def consume_stream(
    lines: Iterable[bytes | str],
    *,
    on_message: Callable[[dict], None],
    drop_heartbeats: bool = True,
) -> int:
    """Parse a v20 chunked-line stream; dispatch non-heartbeat JSON objects.

    Each line is one JSON object (PRICE / TRANSACTION / *HEARTBEAT). Heartbeats are
    consumed and dropped (they only keep the socket warm). Returns the count of
    non-heartbeat messages dispatched. Blank lines and malformed JSON are skipped
    (never crash the consumer). PURE — no network — so it is unit-testable.
    """
    dispatched = 0
    for raw in lines:
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        if drop_heartbeats and obj.get("type") in HEARTBEAT_TYPES:
            continue
        on_message(obj)
        dispatched += 1
    return dispatched


def _backoff_seq(initial: float = MIN_RECONNECT_S) -> Iterator[float]:
    """Exponential backoff capped at MAX_BACKOFF_S, floored at MIN_RECONNECT_S."""
    d = max(initial, MIN_RECONNECT_S)
    while True:
        yield d
        d = min(d * 2.0, MAX_BACKOFF_S)


def run_stream(
    open_conn: Callable[[], Iterable[bytes | str]],
    on_message: Callable[[dict], None],
    *,
    stop: Optional[threading.Event] = None,
    on_reconnect: Optional[Callable[[float, Exception], None]] = None,
) -> None:
    """Persistent stream loop with auto-reconnect + capped exponential backoff.

    ``open_conn`` returns a fresh line-iterator (one real HTTP streaming response,
    or a synthetic iterator in tests). On any drop/error we back off and reopen,
    honoring the >=0.6s min-reconnect (good-citizen). A successful read resets the
    backoff. Runs until ``stop`` is set. ``open_conn`` is injectable so the whole
    reconnect machine is testable without a socket.
    """
    stop = stop or threading.Event()
    backoff = _backoff_seq()
    while not stop.is_set():
        try:
            lines = open_conn()
            consume_stream(_until_stop(lines, stop), on_message=on_message)
            backoff = _backoff_seq()  # clean end -> reset
        except Exception as exc:  # noqa: BLE001 - reconnect on ANY stream failure
            if stop.is_set():
                break
            delay = next(backoff)
            logger.warning("OANDA stream dropped (%s) — reconnect in %.1fs", exc, delay)
            if on_reconnect:
                on_reconnect(delay, exc)
            _interruptible_sleep(delay, stop)


def _until_stop(lines: Iterable[bytes | str], stop: threading.Event) -> Iterator[bytes | str]:
    for ln in lines:
        if stop.is_set():
            return
        yield ln


def _interruptible_sleep(seconds: float, stop: threading.Event) -> None:
    stop.wait(timeout=seconds)


# ---------------------------------------------------------------------------
# HTTP stream openers (real network) — hard-pinned to the practice stream host
# ---------------------------------------------------------------------------
def _open_http_stream(path: str, token: str, params: Optional[Dict] = None):
    import requests
    url = STREAM_PRACTICE_HOST + _API + path
    _assert_practice(url)
    resp = requests.get(
        url, headers={"Authorization": f"Bearer {token}"},
        params=params or {}, stream=True, timeout=(5, 30),
    )
    resp.raise_for_status()
    return resp.iter_lines(decode_unicode=False)


def price_stream_opener(token: str, account_id: str, instruments: List[str]):
    def _open():
        return _open_http_stream(
            f"/accounts/{account_id}/pricing/stream",
            token, {"instruments": ",".join(instruments)})
    return _open


def transaction_stream_opener(token: str, account_id: str):
    def _open():
        return _open_http_stream(f"/accounts/{account_id}/transactions/stream", token)
    return _open


# ---------------------------------------------------------------------------
# Transaction ledger (audit trail + realized P&L) — REST sinceid, persisted
# ---------------------------------------------------------------------------
@dataclass
class LedgerSummary:
    n_transactions: int
    n_fills: int
    realized_pl: float
    financing: float
    last_transaction_id: Optional[str]


class TransactionLedger:
    """Incremental transaction-ledger ingestion to a local JSONL audit file.

    Pulls new transactions via ``/transactions/sinceid`` (idempotent: tracks the
    last id), appends them to ``trained_data/oanda/transactions.jsonl`` (the TUI's
    trade-history source), and aggregates realized P&L + financing from ORDER_FILL
    transactions. Atomic writes. DATA layer only — never places an order.
    """

    def __init__(self, client, *, ledger_dir: Path = Path(LEDGER_DIR_DEFAULT),
                 backfill_count: int = 200):
        self._client = client
        self._dir = Path(ledger_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "transactions.jsonl"
        self._state = self._dir / "ledger_state.json"
        self._backfill = int(backfill_count)

    def _last_id(self) -> Optional[str]:
        try:
            return json.loads(self._state.read_text()).get("last_transaction_id")
        except (OSError, ValueError):
            return None

    def _save_last_id(self, tid: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump({"last_transaction_id": str(tid)}, fh)
        os.replace(tmp, self._state)

    def _append(self, txns: List[dict]) -> None:
        with open(self._path, "a", encoding="utf-8") as fh:
            for t in txns:
                fh.write(json.dumps(t, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def sync(self) -> LedgerSummary:
        """Fetch transactions newer than the last seen id; append + summarize."""
        last = self._last_id()
        if last is None:
            acct = self._client.get_account_summary().get("account", {})
            # First run: backfill the most-recent `backfill_count` transactions so the
            # TUI shows recent trade history (incl. the demo fills), without pulling
            # years of history. start = lastTransactionID - backfill_count (>= 1).
            try:
                last = str(max(1, int(acct.get("lastTransactionID", "1")) - self._backfill))
            except (ValueError, TypeError):
                last = "1"
        resp = self._client.get_transactions_since(last) if hasattr(
            self._client, "get_transactions_since") else self._client._request(
            "GET", f"/accounts/{self._client._config.account_id}/transactions/sinceid",
            params={"id": last})
        txns = (resp or {}).get("transactions", []) or []
        if txns:
            self._append(txns)
            self._save_last_id((resp or {}).get("lastTransactionID") or txns[-1].get("id"))
        return self.summarize()

    def summarize(self) -> LedgerSummary:
        """Aggregate realized P&L + financing across the full local ledger."""
        n = n_fill = 0
        pl = fin = 0.0
        last_id = self._last_id()
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                n += 1
                if t.get("type") == "ORDER_FILL":
                    n_fill += 1
                    pl += float(t.get("pl", 0) or 0)
                    fin += float(t.get("financing", 0) or 0)
        return LedgerSummary(n, n_fill, round(pl, 4), round(fin, 4), last_id)


def _position_brackets(client) -> Dict[str, Dict[str, Optional[float]]]:
    """Map instrument -> {stop_loss, take_profit} from OPEN trades' broker-side orders.

    A netted position can have several trade lots; we report the brackets of the
    LARGEST lot (the representative risk level). Best-effort: a fetch failure yields
    an empty map (positions then render SL/TP=None, never a fabricated level)."""
    try:
        trades = (client.get_trades(state="OPEN") or {}).get("trades", []) or []
    except Exception as exc:  # noqa: BLE001 - display data, never blocks the snapshot
        logger.warning("get_trades for brackets failed (non-blocking): %s", exc)
        return {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    best_units: Dict[str, float] = {}

    def _price(order: Optional[dict]) -> Optional[float]:
        if not order:
            return None
        try:
            return float(order.get("price"))
        except (TypeError, ValueError):
            return None

    for t in trades:
        inst = t.get("instrument")
        if not inst:
            continue
        try:
            units = abs(float(t.get("currentUnits", 0) or 0))
        except (TypeError, ValueError):
            units = 0.0
        if inst not in out or units > best_units.get(inst, -1.0):
            best_units[inst] = units
            out[inst] = {"stop_loss": _price(t.get("stopLossOrder")),
                         "take_profit": _price(t.get("takeProfitOrder"))}
    return out


def snapshot_account_state(client, *, out_dir: Path = Path(LEDGER_DIR_DEFAULT)) -> dict:
    """Write account summary + open positions to disk for the TUI; return a digest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = client.get_account_summary() or {}
    positions = client.get_open_positions() or {}
    brackets = _position_brackets(client)
    acct = summary.get("account", {})
    digest = {
        "nav": float(acct.get("NAV", 0) or 0),
        "unrealized_pl": float(acct.get("unrealizedPL", 0) or 0),
        "realized_pl": float(acct.get("pl", 0) or 0),
        "margin_used": float(acct.get("marginUsed", 0) or 0),
        "margin_available": float(acct.get("marginAvailable", 0) or 0),
        "open_trade_count": int(acct.get("openTradeCount", 0) or 0),
        "currency": acct.get("currency"),
        "account_id": acct.get("id"),
        "positions": [
            {"instrument": p.get("instrument"),
             "net_units": float((p.get("long") or {}).get("units", 0) or 0)
             + float((p.get("short") or {}).get("units", 0) or 0),
             "unrealized_pl": float(p.get("unrealizedPL", 0) or 0),
             # SL/TP for the position's representative (largest) open trade lot; None
             # when no broker-side bracket is attached. Read-only display for AXIOM.
             "stop_loss": (brackets.get(p.get("instrument")) or {}).get("stop_loss"),
             "take_profit": (brackets.get(p.get("instrument")) or {}).get("take_profit")}
            for p in positions.get("positions", []) or []
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(digest, fh, indent=2, sort_keys=True)
    os.replace(tmp, out_dir / "account_state.json")
    return digest


def fetch_sentiment(client, instruments: List[str], *, out_dir: Path = Path(LEDGER_DIR_DEFAULT)) -> dict:
    """DATA-ONLY: snapshot order-book / position-book sentiment (NOT a strategy input).

    Flagged available for a FUTURE pre-registered contrarian test (retail crowding
    -> fade). Persists the raw book percentages; nothing here feeds the trend signal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = {}
    for inst in instruments:
        try:
            ob = client.get_position_book(inst) or {}
            book[inst] = (ob.get("positionBook") or {}).get("buckets", [])[:5]
        except Exception as exc:  # noqa: BLE001 - sentiment is best-effort, never blocks
            logger.warning("sentiment fetch failed for %s: %s", inst, exc)
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump({"note": "DATA-ONLY; not wired into strategy", "books": book}, fh)
    os.replace(tmp, out_dir / "sentiment_snapshot.json")
    return {"instruments": list(book.keys())}
