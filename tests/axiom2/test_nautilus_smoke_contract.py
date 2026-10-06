import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "axiom2" / "nautilus_runtime.json"
SMOKE_PATH = ROOT / "scripts" / "axiom2_nautilus_smoke.py"

UPSTREAM_REVISION = "4f021bafc2e99c5490cee204b0fc2bd2c83baab4"


def test_upstream_pin_and_execution_boundary_are_explicit():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["upstream"]["repository"] == "nautechsystems/nautilus_trader"
    assert config["upstream"]["revision"] == UPSTREAM_REVISION
    assert config["upstream"]["license"] == "LGPL-3.0-only"
    assert config["runtime_boundary"] == {
        "replay_only": True,
        "live_feed": False,
        "external_network": False,
        "broker_credentials": False,
        "execution_clients": False,
        "order_submission": False,
        "capital_authorization": False,
    }


def test_smoke_is_bound_to_public_runtime_and_order_event_apis():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")

    for required in (
        "Clock.new_test",
        "set_timer_ns",
        "set_time",
        "MessageBus",
        "subscribe",
        "publish",
        "DataActor",
        "OrderFilled",
        "OrderCancelRejected",
        '"timer_scheduling"',
        '"timestamp_advance_verified"',
        '"supported_by_public_python_clock"',
    ):
        assert required in smoke

    assert "timer_callback_dispatch" not in smoke
    assert "ExecutionClient(" not in smoke
    assert "TradingNode" not in smoke
