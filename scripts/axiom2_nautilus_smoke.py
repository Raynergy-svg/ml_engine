#!/usr/bin/env python3
"""Run a no-credentials, replay-only public event/clock/actor smoke test.

The script intentionally uses only public Python APIs. It does not create a
client, connect a transport, submit an order, or mutate Axiom state.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import socket
import sys
from typing import Any


NAUTILUS_SOURCE_REVISION = os.environ.get("NAUTILUS_SOURCE_REVISION", "unknown")
NAUTILUS_ENABLED_FEATURES = os.environ.get(
    "NAUTILUS_ENABLED_FEATURES", "not-recorded"
)
CREDENTIAL_ENV_NAMES = (
    "ROBINHOOD_API_KEY",
    "ROBINHOOD_API_SECRET",
    "BROKER_API_KEY",
    "BROKER_API_SECRET",
    "NAUTILUS_API_KEY",
    "NAUTILUS_API_SECRET",
    "EXECUTION_API_KEY",
    "EXECUTION_API_SECRET",
)


def _install_network_guard() -> tuple[list[dict[str, str]], Any, Any]:
    attempts: list[dict[str, str]] = []
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def denied_connect(sock: socket.socket, address: Any) -> None:
        attempts.append({"operation": "socket.connect", "address": repr(address)})
        raise AssertionError(
            f"external network access attempted by the smoke harness: {address!r}"
        )

    def denied_create_connection(*args: Any, **kwargs: Any) -> Any:
        attempts.append(
            {"operation": "socket.create_connection", "address": repr(args[:1])}
        )
        raise AssertionError("external network access attempted by the smoke harness")

    socket.socket.connect = denied_connect  # type: ignore[method-assign]
    socket.create_connection = denied_create_connection  # type: ignore[assignment]
    return attempts, original_connect, original_create_connection


def _restore_network_guard(original_connect: Any, original_create_connection: Any) -> None:
    socket.socket.connect = original_connect  # type: ignore[method-assign]
    socket.create_connection = original_create_connection  # type: ignore[assignment]


def _artifact_identity() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("nautilus-trader")
    package = __import__("nautilus_trader")
    extension_spec = importlib.util.find_spec("nautilus_trader._libnautilus")
    return {
        "distribution": distribution.metadata["Name"],
        "installed_version": distribution.version,
        "package_file": getattr(package, "__file__", None),
        "extension_file": getattr(extension_spec, "origin", None),
        "package_version_attribute": getattr(package, "__version__", None),
        "build_mode": os.environ.get(
            "NAUTILUS_BUILD_MODE", "unspecified; expected source/make build"
        ),
    }


def _exercise_event_delivery_and_clock() -> dict[str, Any]:
    from nautilus_trader.common import Clock, MessageBus
    from nautilus_trader.model import TraderId

    clock = Clock.new_test()
    trader_id = TraderId("TRADER-AXIOM2-SMOKE")
    bus = MessageBus(trader_id, clock=clock)

    received: list[Any] = []
    topic = "axiom2.nautilus.smoke.observation"
    payload = {"event_id": "smoke-observation-1", "kind": "admitted_observation"}
    bus.subscribe(topic, received.append)
    bus.publish(topic, payload, external_pub=False)
    assert received == [payload], received

    timer_events: list[dict[str, Any]] = []

    def on_timer(event: Any) -> None:
        timer_events.append(
            {
                "name": event.name,
                "ts_event": event.ts_event,
                "event_type": type(event).__name__,
            }
        )

    start_ns = clock.timestamp_ns()
    interval_ns = 1_000_000_000
    timer_start_ns = start_ns + interval_ns
    expected_first_timer_ns = timer_start_ns + interval_ns
    clock.set_timer_ns(
        "axiom2.nautilus.smoke.timer",
        interval_ns=interval_ns,
        start_time_ns=timer_start_ns,
        callback=on_timer,
        allow_past=False,
        fire_immediately=False,
    )
    assert clock.timer_count() == 1
    assert (
        clock.next_time_ns("axiom2.nautilus.smoke.timer")
        == expected_first_timer_ns
    )

    # Event delivery is exercised above. The timer check below is deliberately
    # limited to scheduling and timestamp state: at this pinned revision,
    # fire_immediately=False schedules the first event at start_time_ns plus
    # one interval. The supported Python Clock surface has no public
    # advance_time or callback-dispatch method, so this is not timer delivery
    # evidence.
    clock.set_time(timer_start_ns)
    assert clock.timestamp_ns() == timer_start_ns

    bus.dispose()
    return {
        "event_delivery": {"topic": topic, "received": received},
        "deterministic_clock": {
            "start_ns": start_ns,
            "advanced_to_ns": timer_start_ns,
            "timestamp_advance_verified": True,
            "timer_scheduling": {
                "registered": True,
                "timer_start_ns": timer_start_ns,
                "first_event_ns": expected_first_timer_ns,
                "timer_count": clock.timer_count(),
                "next_timer_ns": clock.next_time_ns("axiom2.nautilus.smoke.timer"),
            },
            "callback_dispatch": {
                "supported_by_public_python_clock": False,
                "observed_events": timer_events,
                "note": "Clock.set_time advances the virtual timestamp only; this revision exposes no public advance_time or callback-dispatch API.",
            },
        },
    }


def _exercise_lifecycle() -> dict[str, Any]:
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import DataActor, DataActorConfig

    lifecycle_events: list[str] = []

    class SmokeActor(DataActor):
        def on_start(self) -> None:
            lifecycle_events.append("start")

        def on_stop(self) -> None:
            lifecycle_events.append("stop")

    engine = BacktestEngine(
        BacktestEngineConfig(bypass_logging=True, run_analysis=False)
    )
    actor = SmokeActor(DataActorConfig(log_events=False, log_commands=False))
    engine.add_actor(actor)
    try:
        engine.run()
    finally:
        engine.dispose()

    assert lifecycle_events == ["start", "stop"], lifecycle_events
    assert actor.is_disposed(), actor.state()
    return {
        "component": "DataActor",
        "lifecycle_driver": "BacktestEngine",
        "hook_events": lifecycle_events,
        "final_state": actor.state().name,
        "dispose_called": True,
    }


def _exercise_order_event_constructors() -> dict[str, Any]:
    from nautilus_trader.core import UUID4
    from nautilus_trader.model import (
        AccountId,
        ClientOrderId,
        Currency,
        CurrencyType,
        InstrumentId,
        LiquiditySide,
        OrderAccepted,
        OrderCancelRejected,
        OrderCanceled,
        OrderFilled,
        OrderRejected,
        OrderSide,
        OrderSubmitted,
        OrderType,
        Price,
        Quantity,
        StrategyId,
        Symbol,
        TradeId,
        TraderId,
        Venue,
        VenueOrderId,
    )

    trader_id = TraderId("TRADER-AXIOM2-SMOKE")
    strategy_id = StrategyId("STRATEGY-AXIOM2-SMOKE")
    instrument_id = InstrumentId(Symbol("AAPL"), Venue("XNAS"))
    client_order_id = ClientOrderId("O-AXIOM2-SMOKE")
    venue_order_id = VenueOrderId("V-AXIOM2-SMOKE")
    account_id = AccountId("ACC-AXIOM2-SMOKE")
    ts_ns = 1_700_000_000_000_000_000
    common = {
        "trader_id": trader_id,
        "strategy_id": strategy_id,
        "instrument_id": instrument_id,
        "client_order_id": client_order_id,
        "venue_order_id": venue_order_id,
        "account_id": account_id,
        "ts_event": ts_ns,
        "ts_init": ts_ns,
    }

    events: dict[str, Any] = {}
    events["OrderSubmitted"] = OrderSubmitted(
        event_id=UUID4(),
        **{key: value for key, value in common.items() if key != "venue_order_id"},
    )
    events["OrderAccepted"] = OrderAccepted(event_id=UUID4(), **common)
    events["OrderRejected"] = OrderRejected(
        event_id=UUID4(),
        trader_id=trader_id,
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        client_order_id=client_order_id,
        account_id=account_id,
        reason="synthetic rejection",
        ts_event=ts_ns,
        ts_init=ts_ns,
        reconciliation=False,
    )
    events["OrderCanceled"] = OrderCanceled(
        event_id=UUID4(),
        trader_id=trader_id,
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        client_order_id=client_order_id,
        ts_event=ts_ns,
        ts_init=ts_ns,
        reconciliation=False,
        venue_order_id=venue_order_id,
        account_id=account_id,
    )
    events["OrderCancelRejected"] = OrderCancelRejected(
        event_id=UUID4(),
        trader_id=trader_id,
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        client_order_id=client_order_id,
        reason="synthetic cancel rejection",
        ts_event=ts_ns,
        ts_init=ts_ns,
        reconciliation=False,
        venue_order_id=venue_order_id,
        account_id=account_id,
    )
    events["OrderFilled"] = OrderFilled(
        event_id=UUID4(),
        trader_id=trader_id,
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        client_order_id=client_order_id,
        venue_order_id=venue_order_id,
        account_id=account_id,
        trade_id=TradeId("T-AXIOM2-SMOKE"),
        order_side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        last_qty=Quantity(1.0, 0),
        last_px=Price(100.0, 2),
        currency=Currency("USD", 2, 840, "US Dollar", CurrencyType.FIAT),
        liquidity_side=LiquiditySide.TAKER,
        ts_event=ts_ns,
        ts_init=ts_ns,
        reconciliation=False,
    )

    return {
        "public_constructors_callable": sorted(events),
        "engine_integration_claim": False,
        "note": "Constructor calls prove model API availability only; no live execution client or broker was created.",
    }


def main() -> int:
    present_credentials = [
        name for name in CREDENTIAL_ENV_NAMES if os.environ.get(name)
    ]
    assert not present_credentials, (
        "credential environment variables are present: "
        + ", ".join(present_credentials)
    )

    attempts, original_connect, original_create_connection = _install_network_guard()
    try:
        identity = _artifact_identity()
        runtime = _exercise_event_delivery_and_clock()
        lifecycle = _exercise_lifecycle()
        order_events = _exercise_order_event_constructors()
    finally:
        _restore_network_guard(original_connect, original_create_connection)

    assert not attempts, attempts
    report = {
        "source_revision": NAUTILUS_SOURCE_REVISION,
        "enabled_features": NAUTILUS_ENABLED_FEATURES,
        "license": "LGPL-3.0-only",
        "artifact": identity,
        "runtime": runtime,
        "lifecycle": lifecycle,
        "order_events": order_events,
        "safety": {
            "broker_credentials_present": False,
            "execution_clients_created": False,
            "backtest_engine_created": True,
            "external_network_attempts": attempts,
            "order_submitted": False,
            "capital_authorized": False,
            "live_feed": False,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
