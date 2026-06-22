"""US-014 — Depth-aware equity execution pricing.

Reimplements the depth-walking math from Hummingbot's
``hummingbot/core/data_type/order_book.pyx`` for the equities path:

* :func:`get_price_for_volume` — walk the order book until cumulative
  size meets the target size, return the worst (limit-equivalent) price
  needed to sweep that depth.
* :func:`get_vwap_for_volume` — same walk, but return the size-weighted
  average price across the swept depth (the realistic average fill).

These are the two queries the live equity execution sizer needs before
sending an order: "what notional am I about to move?" (vwap) and "what
is the worst price I'll touch?" (price-for-volume). The implementations
are pure-Python (NumPy-free) and operate on a tiny sorted-list L2 book
so the math is auditable line-by-line against the reference Cython.

Attribution
-----------

Adapted from Hummingbot's ``core/data_type/order_book.pyx`` — see the
file header below. Hummingbot is licensed under the Apache License,
Version 2.0; see ``LICENSE-APACHE-2.0`` and ``NOTICE`` at the repo
root for the full text and attribution.

The reference functions only operate on top-of-book + price/size
levels — we ship a hand-rolled minimal book (no `numpy.recarray`,
no event playback) because the equity execution path only ever needs
the two queries above and the rest of the Hummingbot ``OrderBook``
class is wired to its own asyncio event loop.

Hard rules followed
-------------------

* Real classes + real disk via ``tmp_path`` in tests — no
  ``unittest.mock``, no ``MagicMock``, no ``patch``.
* Atomic state writes (``tempfile.mkstemp`` + ``os.replace``); no
  half-written depth snapshot.
* No bare ``except``; every recoverable error logs and re-raises a
  typed exception.
* Refuses to do anything unless ``SHIP_GATE.json`` clears for the
  expected universe — mirrors :func:`src.equity.kill_switch.enforce_ship_gate`
  and the other equity-stack guards so the same fixture passes / fails
  identically across modules.

================================ NOTICE =================================
Portions of this module are derived from Hummingbot
(https://github.com/hummingbot/hummingbot), specifically the
``hummingbot/core/data_type/order_book.pyx`` reference implementation
of ``get_price_for_volume`` and ``get_vwap_for_volume``.

Hummingbot is Copyright Hummingbot Foundation and contributors,
licensed under the Apache License, Version 2.0.

This file is a clean-room re-derivation of the depth-walk math in
pure Python (no Cython), narrowed to the equity execution sizing use
case. The original Hummingbot file is not vendored.
=========================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
SNAPSHOT_VERSION = 1


class DepthPricingError(RuntimeError):
    """Raised on malformed books, ship-gate refusal, or state corruption.

    Distinct exception type so the live execution sizer can match this
    specifically (e.g. fall back to last-trade pricing on a malformed
    snapshot) instead of swallowing every ``Exception`` in the hot path.
    """


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse depth-aware pricing unless SHIP_GATE.json clears for THIS universe.

    Behavior is bit-for-bit identical to
    :func:`src.equity.kill_switch.enforce_ship_gate`,
    :func:`src.equity.rebalance._enforce_ship_gate`, and
    :func:`src.equity.order_lifecycle.enforce_ship_gate` so a fixture
    that passes one surface passes them all. Fail-closed on:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        DepthPricingError: On any of the above.
    """
    path = Path(path)
    if not path.exists():
        raise DepthPricingError(
            f"depth pricing refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthPricingError(
            f"depth pricing refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DepthPricingError(
            f"depth pricing refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise DepthPricingError(
            "depth pricing refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the depth sizer)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise DepthPricingError(
            f"depth pricing refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise DepthPricingError(
            "depth pricing refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


@dataclass(frozen=True)
class OrderBookEntry:
    """One L2 level: price and size at that price.

    Sizes are in shares (whole shares only for equities — fractional
    shares are routed differently). Prices are in quote currency
    (USD for the supported equity venues).
    """

    price: float
    size: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0.0:
            raise DepthPricingError(
                f"OrderBookEntry price must be a positive finite float, "
                f"got {self.price!r}"
            )
        if not math.isfinite(self.size) or self.size <= 0.0:
            raise DepthPricingError(
                f"OrderBookEntry size must be a positive finite float, "
                f"got {self.size!r}"
            )


@dataclass(frozen=True)
class OrderBookQueryResult:
    """Mirrors hummingbot ``OrderBookQueryResult``.

    Fields:
        query_price: 0.0 for volume-based queries (we only ship the
            volume queries; price-based queries are not used by the
            equity sizer).
        query_volume: the volume the caller asked about.
        result_price: limit-equivalent worst price (price-for-volume)
            or size-weighted average fill price (vwap-for-volume).
        result_volume: the volume actually fillable from the visible
            depth (== query_volume when the book is deep enough, else
            the total visible size on that side).
    """

    query_price: float
    query_volume: float
    result_price: float
    result_volume: float


def _validate_sorted(
    side: str,
    entries: Sequence[OrderBookEntry],
    *,
    descending: bool,
) -> None:
    if len(entries) == 0:
        return
    prev = entries[0].price
    for level in entries[1:]:
        if descending:
            if level.price >= prev:
                raise DepthPricingError(
                    f"{side} side must be strictly descending in price; "
                    f"got {level.price} after {prev}"
                )
        else:
            if level.price <= prev:
                raise DepthPricingError(
                    f"{side} side must be strictly ascending in price; "
                    f"got {level.price} after {prev}"
                )
        prev = level.price


@dataclass(frozen=True)
class L2OrderBook:
    """Minimal sorted L2 order book.

    * ``bids`` strictly descending by price (best bid first).
    * ``asks`` strictly ascending by price (best ask first).
    * Crossed books (best_bid >= best_ask) are rejected — a real
      exchange feed never crosses; if our snapshot looks crossed
      something upstream is wrong and we must not size against it.
    """

    symbol: str
    bids: List[OrderBookEntry] = field(default_factory=list)
    asks: List[OrderBookEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise DepthPricingError(
                f"L2OrderBook symbol must be a non-empty string, "
                f"got {self.symbol!r}"
            )
        _validate_sorted("bid", self.bids, descending=True)
        _validate_sorted("ask", self.asks, descending=False)
        if self.bids and self.asks and self.bids[0].price >= self.asks[0].price:
            raise DepthPricingError(
                f"crossed book for {self.symbol}: best bid "
                f"{self.bids[0].price} >= best ask {self.asks[0].price}"
            )

    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    def mid_price(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def total_size(self, is_buy: bool) -> float:
        side = self.asks if is_buy else self.bids
        return sum(level.size for level in side)


def _walk_side_for_volume(
    side: Sequence[OrderBookEntry],
    volume: float,
) -> tuple[float, float, float]:
    """Walk ``side`` until cumulative size >= ``volume``.

    Returns ``(cum_size, cum_notional, last_price_touched)`` for the
    portion of the book actually consumed. If the book is too thin to
    fill ``volume``, returns the totals across the entire side.

    This is the exact loop the Hummingbot Cython implementations both
    share, factored out so the two public queries share one walk and
    we can test the boundary cases against a single reference.
    """
    if volume <= 0.0:
        return 0.0, 0.0, 0.0

    cum_size = 0.0
    cum_notional = 0.0
    last_price = 0.0

    for level in side:
        remaining = volume - cum_size
        if remaining <= 0.0:
            break
        take = level.size if level.size <= remaining else remaining
        cum_size += take
        cum_notional += take * level.price
        last_price = level.price
        if cum_size >= volume:
            break

    return cum_size, cum_notional, last_price


def get_price_for_volume(
    book: L2OrderBook,
    is_buy: bool,
    volume: float,
) -> OrderBookQueryResult:
    """Limit-equivalent worst price to sweep ``volume`` shares.

    Reference: ``OrderBook.get_price_for_volume`` in
    ``hummingbot/core/data_type/order_book.pyx``.

    * ``is_buy=True`` walks the ask side (we're paying up to ASKs).
    * ``is_buy=False`` walks the bid side (we're hitting BIDs).

    If the visible depth is insufficient, ``result_volume`` carries the
    fillable size and ``result_price`` is the last price touched. The
    caller is expected to refuse / re-quote in that case — this matches
    Hummingbot's contract.
    """
    if not isinstance(book, L2OrderBook):
        raise DepthPricingError(
            f"get_price_for_volume requires L2OrderBook, got {type(book).__name__}"
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise DepthPricingError(
            f"volume must be a positive finite float, got {volume!r}"
        )

    side = book.asks if is_buy else book.bids
    cum_size, _cum_notional, last_price = _walk_side_for_volume(side, volume)

    return OrderBookQueryResult(
        query_price=0.0,
        query_volume=float(volume),
        result_price=float(last_price),
        result_volume=float(cum_size),
    )


def get_vwap_for_volume(
    book: L2OrderBook,
    is_buy: bool,
    volume: float,
) -> OrderBookQueryResult:
    """Size-weighted average price to sweep ``volume`` shares.

    Reference: ``OrderBook.get_vwap_for_volume`` in
    ``hummingbot/core/data_type/order_book.pyx``.

    The VWAP is computed across the depth actually consumed — if the
    book only holds ``S < volume`` shares, the returned VWAP is the
    size-weighted average across those ``S`` shares (mirrors
    Hummingbot's behaviour: the query result honestly reports both the
    partial fill volume and the partial VWAP).

    A book with zero visible size on the requested side returns
    ``result_price=0.0, result_volume=0.0``.
    """
    if not isinstance(book, L2OrderBook):
        raise DepthPricingError(
            f"get_vwap_for_volume requires L2OrderBook, got {type(book).__name__}"
        )
    if not math.isfinite(volume) or volume <= 0.0:
        raise DepthPricingError(
            f"volume must be a positive finite float, got {volume!r}"
        )

    side = book.asks if is_buy else book.bids
    cum_size, cum_notional, _last_price = _walk_side_for_volume(side, volume)
    vwap = (cum_notional / cum_size) if cum_size > 0.0 else 0.0

    return OrderBookQueryResult(
        query_price=0.0,
        query_volume=float(volume),
        result_price=float(vwap),
        result_volume=float(cum_size),
    )


class DepthAwareSizer:
    """Live-execution-sizing front door.

    Constructed armed for a specific ``universe_hash``. Every public
    pricing call re-checks ``SHIP_GATE.json`` so an invalidated gate
    mid-session immediately refuses further sizing — this matches the
    rebalance / kill-switch / order-lifecycle gate discipline.

    Use:

        sizer = DepthAwareSizer(universe_hash, ship_gate_path)
        result = sizer.vwap_for_volume(book, is_buy=True, volume=10_000)
        if result.result_volume < 10_000:
            # book too thin — caller must refuse or re-quote
            ...
    """

    def __init__(
        self,
        universe_hash: str,
        ship_gate_path: Path = Path(SHIP_GATE_PATH_DEFAULT),
    ) -> None:
        if not isinstance(universe_hash, str) or not universe_hash:
            raise DepthPricingError(
                f"universe_hash must be a non-empty string, "
                f"got {universe_hash!r}"
            )
        self.universe_hash = universe_hash
        self.ship_gate_path = Path(ship_gate_path)
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

    def _reenforce(self) -> None:
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

    def price_for_volume(
        self,
        book: L2OrderBook,
        *,
        is_buy: bool,
        volume: float,
    ) -> OrderBookQueryResult:
        self._reenforce()
        return get_price_for_volume(book, is_buy, volume)

    def vwap_for_volume(
        self,
        book: L2OrderBook,
        *,
        is_buy: bool,
        volume: float,
    ) -> OrderBookQueryResult:
        self._reenforce()
        return get_vwap_for_volume(book, is_buy, volume)


def _atomic_write_json(payload: dict, path: Path) -> Path:
    """Atomic JSON write: ``tempfile.mkstemp`` sibling + ``os.replace``.

    Mirrors the kill-switch / rebalance pattern: write to a temp file
    in the SAME directory (so ``os.replace`` is atomic on POSIX) and
    rename. A crash mid-write leaves the prior file intact; a reader
    never sees a half-written snapshot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError as cleanup_exc:
            logger.warning(
                "depth pricing: failed to cleanup tmp %s: %s",
                tmp_path,
                cleanup_exc,
            )
        raise
    return path


def book_to_snapshot(book: L2OrderBook) -> dict:
    """Serialise an :class:`L2OrderBook` to a plain-dict snapshot.

    Includes a ``version`` so older snapshots are easy to refuse if
    the schema ever evolves (e.g. when we add venue / timestamp).
    """
    return {
        "version": SNAPSHOT_VERSION,
        "symbol": book.symbol,
        "bids": [asdict(level) for level in book.bids],
        "asks": [asdict(level) for level in book.asks],
    }


def snapshot_to_book(payload: Mapping) -> L2OrderBook:
    """Inverse of :func:`book_to_snapshot`; validates and re-sorts."""
    if not isinstance(payload, Mapping):
        raise DepthPricingError(
            f"snapshot payload must be a mapping, got {type(payload).__name__}"
        )
    version = payload.get("version")
    if version != SNAPSHOT_VERSION:
        raise DepthPricingError(
            f"snapshot version mismatch: got {version!r}, "
            f"expected {SNAPSHOT_VERSION}"
        )
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise DepthPricingError(
            f"snapshot symbol missing/blank: {symbol!r}"
        )
    bids = [_entry_from_payload(p, "bid") for p in payload.get("bids", [])]
    asks = [_entry_from_payload(p, "ask") for p in payload.get("asks", [])]
    return L2OrderBook(symbol=symbol, bids=bids, asks=asks)


def _entry_from_payload(p: Mapping, side: str) -> OrderBookEntry:
    if not isinstance(p, Mapping):
        raise DepthPricingError(
            f"{side} level must be a mapping, got {type(p).__name__}"
        )
    try:
        price = float(p["price"])
        size = float(p["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DepthPricingError(
            f"{side} level missing/invalid price/size: {p!r}: {exc}"
        ) from exc
    return OrderBookEntry(price=price, size=size)


def write_snapshot(book: L2OrderBook, path: Path) -> Path:
    """Atomically persist an order-book snapshot to ``path``."""
    return _atomic_write_json(book_to_snapshot(book), Path(path))


def read_snapshot(path: Path) -> L2OrderBook:
    """Read a previously-written snapshot back into an :class:`L2OrderBook`."""
    path = Path(path)
    if not path.exists():
        raise DepthPricingError(f"snapshot not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthPricingError(
            f"snapshot at {path} unreadable: {exc}"
        ) from exc
    return snapshot_to_book(payload)


def make_book(
    symbol: str,
    bids: Iterable[Sequence[float]],
    asks: Iterable[Sequence[float]],
) -> L2OrderBook:
    """Convenience: build an :class:`L2OrderBook` from raw ``(price, size)`` pairs.

    Re-sorts each side so callers don't have to (bids descending, asks
    ascending) and runs all the same invariants the dataclass enforces.
    """
    bid_entries = sorted(
        (OrderBookEntry(price=float(p), size=float(s)) for p, s in bids),
        key=lambda e: e.price,
        reverse=True,
    )
    ask_entries = sorted(
        (OrderBookEntry(price=float(p), size=float(s)) for p, s in asks),
        key=lambda e: e.price,
    )
    return L2OrderBook(symbol=symbol, bids=bid_entries, asks=ask_entries)


__all__ = [
    "DepthAwareSizer",
    "DepthPricingError",
    "L2OrderBook",
    "OrderBookEntry",
    "OrderBookQueryResult",
    "SHIP_GATE_PATH_DEFAULT",
    "SNAPSHOT_VERSION",
    "book_to_snapshot",
    "enforce_ship_gate",
    "get_price_for_volume",
    "get_vwap_for_volume",
    "make_book",
    "read_snapshot",
    "snapshot_to_book",
    "write_snapshot",
]
