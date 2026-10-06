"""Offline order-event replay through the pinned Nautilus order implementation.

Raw reports are durable observations. The Nautilus order snapshot is a derived
view rebuilt from APPLIED raw reports after restart; it is not an authority,
broker receipt, or capital authorization.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


class NautilusOrderReplayUnavailable(RuntimeError):
    """Raised when the pinned Nautilus model package is not installed."""


class NautilusOrderReplayCorruption(RuntimeError):
    """Raised when durable raw reports cannot rebuild the derived order."""


@dataclass(frozen=True, slots=True)
class OrderSeed:
    trader_id: str = "TRADER-AXIOM2-REPLAY"
    strategy_id: str = "S-AXIOM2-REPLAY"
    instrument_id: str = "AAPL.SIM"
    client_order_id: str = "O-AXIOM2-REPLAY"
    account_id: str = "SIM-AXIOM2"
    venue_order_id: str = "VENUE-AXIOM2-1"
    quantity: int = 100
    price: str = "10.00"
    init_id: str = "00000000-0000-4000-8000-000000000001"
    ts_init: int = 0


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    status: str
    quantity_raw: int
    filled_quantity_raw: int
    leaves_quantity_raw: int
    event_count: int
    is_closed: bool
    is_canceled: bool
    last_event_type: str


@dataclass(frozen=True, slots=True)
class OrderIngestResult:
    observation_id: str
    disposition: str
    raw_row_id: int
    reason: str | None
    snapshot: OrderSnapshot


def _json_default(value: Any) -> str | int:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "to_formatted_str"):
        return value.to_formatted_str()
    return str(value)


def _payload_json(event: Any) -> str:
    return json.dumps(
        event.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _event_id(event: Any) -> str:
    value = event.event_id
    return str(getattr(value, "value", value))


class NautilusOrderReplay:
    """Persist raw order reports and derive state with Nautilus LimitOrder.

    This class only constructs the public model object and applies public order
    events. It creates no TradingNode, execution client, broker, credentials,
    submission gateway, or capital authority.
    """

    def __init__(self, path: str | Path, *, seed: OrderSeed | None = None) -> None:
        try:
            from nautilus_trader import model
        except ModuleNotFoundError as exc:
            raise NautilusOrderReplayUnavailable(
                "the pinned Nautilus package is required for order replay"
            ) from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._model = model
        self.seed = seed or OrderSeed()
        self._create_schema()
        self._order = self._new_order()
        self._rebuild()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_order_observations (
                raw_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                received_at_ns INTEGER NOT NULL,
                disposition TEXT NOT NULL,
                reason TEXT
            );

            CREATE INDEX IF NOT EXISTS raw_order_observations_by_observation
                ON raw_order_observations(observation_id);

            CREATE TABLE IF NOT EXISTS derived_order_states (
                raw_row_id INTEGER PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                FOREIGN KEY(raw_row_id) REFERENCES raw_order_observations(raw_row_id)
            );
            """
        )

    @contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _new_order(self) -> Any:
        from nautilus_trader.core import UUID4
        from nautilus_trader.model import ClientOrderId
        from nautilus_trader.model import InstrumentId
        from nautilus_trader.model import LimitOrder
        from nautilus_trader.model import OrderSide
        from nautilus_trader.model import Price
        from nautilus_trader.model import Quantity
        from nautilus_trader.model import StrategyId
        from nautilus_trader.model import TimeInForce
        from nautilus_trader.model import TraderId

        return LimitOrder(
            trader_id=TraderId(self.seed.trader_id),
            strategy_id=StrategyId(self.seed.strategy_id),
            instrument_id=InstrumentId.from_str(self.seed.instrument_id),
            client_order_id=ClientOrderId(self.seed.client_order_id),
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(self.seed.quantity),
            price=Price.from_str(self.seed.price),
            time_in_force=TimeInForce.GTC,
            post_only=False,
            reduce_only=False,
            quote_quantity=False,
            init_id=UUID4.from_str(self.seed.init_id),
            ts_init=self.seed.ts_init,
        )

    def _event_from_row(self, row: sqlite3.Row) -> Any:
        event_type = row["event_type"]
        event_class = getattr(self._model, event_type, None)
        if event_class is None or not hasattr(event_class, "from_dict"):
            raise NautilusOrderReplayCorruption(
                f"unsupported durable Nautilus event: {event_type}"
            )
        try:
            return event_class.from_dict(json.loads(row["payload_json"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise NautilusOrderReplayCorruption(
                f"invalid durable Nautilus event: {event_type}"
            ) from exc

    def _rebuild(self) -> None:
        self._order = self._new_order()
        rows = self.connection.execute(
            """
            SELECT * FROM raw_order_observations
            WHERE disposition = 'APPLIED'
            ORDER BY raw_row_id
            """
        ).fetchall()
        for row in rows:
            event = self._event_from_row(row)
            try:
                self._order.apply(event)
            except Exception as exc:
                raise NautilusOrderReplayCorruption(
                    f"accepted raw report no longer replays: {row['raw_row_id']}"
                ) from exc

    def snapshot(self) -> OrderSnapshot:
        status = self._order.status
        return OrderSnapshot(
            status=getattr(status, "name", str(status)),
            quantity_raw=int(self._order.quantity.raw),
            filled_quantity_raw=int(self._order.filled_qty.raw),
            leaves_quantity_raw=int(self._order.leaves_qty.raw),
            event_count=int(self._order.event_count),
            is_closed=bool(self._order.is_closed),
            is_canceled=bool(self._order.is_canceled),
            last_event_type=type(self._order.last_event).__name__,
        )

    def _insert_raw(
        self,
        *,
        observation_id: str,
        event: Any,
        payload_json: str,
        payload_digest: str,
        received_at_ns: int,
        disposition: str,
        reason: str | None,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO raw_order_observations (
                observation_id, event_id, event_type, payload_json,
                payload_digest, received_at_ns, disposition, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                _event_id(event),
                type(event).__name__,
                payload_json,
                payload_digest,
                received_at_ns,
                disposition,
                reason,
            ),
        )
        return int(cursor.lastrowid)

    def ingest(
        self,
        observation_id: str,
        event: Any,
        *,
        received_at_ns: int,
    ) -> OrderIngestResult:
        if type(observation_id) is not str or not observation_id.strip():
            raise ValueError("observation_id must be non-empty")
        if type(received_at_ns) is not int or received_at_ns < 0:
            raise ValueError("received_at_ns must be non-negative")

        payload_json = _payload_json(event)
        payload_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        existing = self.connection.execute(
            """
            SELECT payload_digest FROM raw_order_observations
            WHERE observation_id = ?
            ORDER BY raw_row_id
            LIMIT 1
            """,
            (observation_id,),
        ).fetchone()

        if existing is not None:
            disposition = (
                "DUPLICATE"
                if existing["payload_digest"] == payload_digest
                else "CONFLICT"
            )
            reason = None if disposition == "DUPLICATE" else "observation identity reused with different bytes"
            with self._transaction():
                raw_row_id = self._insert_raw(
                    observation_id=observation_id,
                    event=event,
                    payload_json=payload_json,
                    payload_digest=payload_digest,
                    received_at_ns=received_at_ns,
                    disposition=disposition,
                    reason=reason,
                )
            return OrderIngestResult(
                observation_id=observation_id,
                disposition=disposition,
                raw_row_id=raw_row_id,
                reason=reason,
                snapshot=self.snapshot(),
            )

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            try:
                self._order.apply(event)
            except Exception as exc:
                self.connection.execute("ROLLBACK")
                self._rebuild()
                reason = f"{type(exc).__name__}: {exc}"
                with self._transaction():
                    raw_row_id = self._insert_raw(
                        observation_id=observation_id,
                        event=event,
                        payload_json=payload_json,
                        payload_digest=payload_digest,
                        received_at_ns=received_at_ns,
                        disposition="REJECTED",
                        reason=reason,
                    )
                return OrderIngestResult(
                    observation_id=observation_id,
                    disposition="REJECTED",
                    raw_row_id=raw_row_id,
                    reason=reason,
                    snapshot=self.snapshot(),
                )

            raw_row_id = self._insert_raw(
                observation_id=observation_id,
                event=event,
                payload_json=payload_json,
                payload_digest=payload_digest,
                received_at_ns=received_at_ns,
                disposition="APPLIED",
                reason=None,
            )
            self.connection.execute(
                """
                INSERT INTO derived_order_states(raw_row_id, snapshot_json)
                VALUES (?, ?)
                """,
                (
                    raw_row_id,
                    json.dumps(asdict(self.snapshot()), sort_keys=True),
                ),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            self._rebuild()
            raise

        return OrderIngestResult(
            observation_id=observation_id,
            disposition="APPLIED",
            raw_row_id=raw_row_id,
            reason=None,
            snapshot=self.snapshot(),
        )

    def raw_observations(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM raw_order_observations
            ORDER BY raw_row_id
            """
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def applied_events(self) -> tuple[Any, ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM raw_order_observations
            WHERE disposition = 'APPLIED'
            ORDER BY raw_row_id
            """
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "NautilusOrderReplay":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
