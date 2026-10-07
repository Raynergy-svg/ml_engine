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
        "BacktestEngine",
        "add_actor",
        "engine.run",
        "on_start",
        "on_stop",
        "OrderFilled",
        "OrderCancelRejected",
        '"timer_scheduling"',
        '"timer_start_ns"',
        '"first_event_ns"',
        "expected_first_timer_ns",
        '"timestamp_advance_verified"',
        '"supported_by_public_python_clock"',
    ):
        assert required in smoke

    assert "timer_callback_dispatch" not in smoke
    assert 'next_time_ns("axiom2.nautilus.smoke.timer") == start_ns + interval_ns' not in smoke
    assert "start_time_ns=timer_start_ns" in smoke
    assert "actor.start()" not in smoke
    assert "actor.stop()" not in smoke
    assert "ExecutionClient(" not in smoke
    assert "TradingNode" not in smoke


def test_smoke_serializes_component_state_with_supported_string_conversion():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")

    assert '"final_state": str(actor.state())' in smoke
    assert "actor.state().name" not in smoke
    assert "actor.state().value" not in smoke
    
def test_smoke_matches_pinned_order_accepted_constructor_requirements():
    smoke = SMOKE_PATH.read_text(encoding="utf-8")
    accepted_call = smoke.split(
        'events["OrderAccepted"] = OrderAccepted(', 1
    )[1].split(
        'events["OrderRejected"] = OrderRejected(', 1
    )[0]

    assert "reconciliation=False" in accepted_call

def test_runtime_workflow_preserves_exact_build_and_runs_all_gates():
    workflow = (
        ROOT / ".github" / "workflows" / "axiom2-nautilus-runtime.yml"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "config" / "axiom2" / "nautilus_build_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["upstream"]["revision"] == UPSTREAM_REVISION
    assert manifest["build"]["enabled_features"] == (
        "arrow,ffi,python,high-precision,streaming,defi"
    )
    assert manifest["build"]["uv_version"] == "0.12.23"
    assert "actions/cache/restore@" in workflow
    assert "actions/cache/save@" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "nautilus/.axiom2-nautilus-build-manifest.json" in workflow
    assert "PYTHONPATH=\"$GITHUB_WORKSPACE/nautilus/python:$GITHUB_WORKSPACE/axiom/src:$GITHUB_WORKSPACE/axiom\"" in workflow
    assert "test_runtime_recovery.py" in workflow
    assert "test_order_lifecycle.py" in workflow
    assert workflow.count("if: ${{ always() }}") == 4
