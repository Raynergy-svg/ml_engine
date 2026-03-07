from __future__ import annotations

import json
import zipfile

from src.agent.planner_runtime import PlannerRuntime
from src.agent.tool_registry import ToolRegistry
from src.agent.tool_schema import ToolResult, ToolSpec


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="use_market_source",
            description="use source",
            args=(),
        ),
        lambda args, state: ToolResult(ok=True, tool_name="use_market_source", args=args, result={"active_source": "oanda:EUR_USD:M5:300"}, human_summary="loaded oanda:EUR_USD:M5:300."),
    )
    registry.register(
        ToolSpec(name="run_prediction", description="predict"),
        lambda args, state: ToolResult(ok=True, tool_name="run_prediction", args=args, result={"prediction": 1.1}, human_summary="Prediction ready."),
    )
    registry.register(
        ToolSpec(name="summarize_last_prediction", description="summary"),
        lambda args, state: ToolResult(ok=True, tool_name="summarize_last_prediction", args=args, result={"has_prediction": True}, human_summary="Summary ready."),
    )
    registry.register(
        ToolSpec(name="run_runtime_command", description="runtime"),
        lambda args, state: ToolResult(ok=True, tool_name="run_runtime_command", args=args, result={"execute_mode": "dry-run"}, human_summary="Runtime updated."),
    )
    registry.register(
        ToolSpec(name="override_guardrail", description="override guardrail"),
        lambda args, state: ToolResult(
            ok=True,
            tool_name="override_guardrail",
            args=args,
            result={"applied": True, "override": args},
            human_summary=f"Guardrail override applied: {args.get('action')} ({args.get('scope', 'this_scan')}).",
        ),
    )
    registry.register(
        ToolSpec(name="run_oanda_command", description="oanda"),
        lambda args, state: ToolResult(ok=True, tool_name="run_oanda_command", args=args, result={}, human_summary="OANDA command ran."),
    )
    registry.register(
        ToolSpec(name="run_trade_command", description="trade"),
        lambda args, state: ToolResult(ok=True, tool_name="run_trade_command", args=args, result={}, human_summary="Trade command ran."),
    )
    registry.register(
        ToolSpec(name="get_status", description="status"),
        lambda args, state: ToolResult(ok=True, tool_name="get_status", args=args, result={"mode": "dry-run"}, human_summary="Status: dry-run."),
    )
    registry.register(
        ToolSpec(name="get_model_status", description="models"),
        lambda args, state: ToolResult(ok=True, tool_name="get_model_status", args=args, result={"count": 4}, human_summary="Model status loaded."),
    )
    registry.register(
        ToolSpec(name="get_decisions", description="decisions"),
        lambda args, state: ToolResult(ok=True, tool_name="get_decisions", args=args, result={"last": "event risk"}, human_summary="Loaded last decisions."),
    )
    registry.register(
        ToolSpec(name="get_journal_summary", description="journal"),
        lambda args, state: ToolResult(ok=True, tool_name="get_journal_summary", args=args, result={"days": args.get("days", 30)}, human_summary="Journal summary loaded."),
    )
    registry.register(
        ToolSpec(name="sync_journal", description="journal sync"),
        lambda args, state: ToolResult(ok=True, tool_name="sync_journal", args=args, result={}, human_summary="Journal synced."),
    )
    registry.register(
        ToolSpec(name="get_account_summary", description="account"),
        lambda args, state: ToolResult(ok=True, tool_name="get_account_summary", args=args, result={"balance": 1000}, human_summary="Account summary loaded."),
    )
    registry.register(
        ToolSpec(name="scan_market", description="scan"),
        lambda args, state: ToolResult(ok=True, tool_name="scan_market", args=args, result={"approved_count": 1}, human_summary="Scan finished on H1: 1 approved."),
    )
    registry.register(
        ToolSpec(name="compare_scan_results", description="compare"),
        lambda args, state: ToolResult(ok=True, tool_name="compare_scan_results", args=args, result={"winner": "EUR_USD"}, human_summary="EUR_USD is stronger because it passed the gates."),
    )
    registry.register(
        ToolSpec(name="configure_pair_sentiment_gate", description="config sentiment"),
        lambda args, state: ToolResult(ok=True, tool_name="configure_pair_sentiment_gate", args=args, result=args, human_summary="Pair sentiment gate updated."),
    )
    registry.register(
        ToolSpec(name="configure_correlation_sentiment_gate", description="config correlation sentiment"),
        lambda args, state: ToolResult(ok=True, tool_name="configure_correlation_sentiment_gate", args=args, result=args, human_summary="Correlation sentiment gate updated."),
    )
    registry.register(
        ToolSpec(name="retrain_meta_labelers", description="retrain meta"),
        lambda args, state: ToolResult(ok=True, tool_name="retrain_meta_labelers", args=args, result={"trained_count": 3}, human_summary="Meta-labeler retraining finished."),
    )
    registry.register(
        ToolSpec(name="run_basket_backtest", description="basket backtest"),
        lambda args, state: ToolResult(ok=True, tool_name="run_basket_backtest", args=args, result={"basket_total_return": 0.12}, human_summary="Basket backtest returned +12.0% with a clean equity curve."),
    )
    registry.register(
        ToolSpec(name="configure_aggressive_regime_sizing", description="regime sizing"),
        lambda args, state: ToolResult(ok=True, tool_name="configure_aggressive_regime_sizing", args=args, result=args, human_summary="Aggressive volatility-regime sizing enabled."),
    )
    registry.register(
        ToolSpec(name="configure_scan_profile", description="scan profile"),
        lambda args, state: ToolResult(ok=True, tool_name="configure_scan_profile", args=args, result=args, human_summary=f"Scanner profile set to {args.get('profile')}."),
    )
    registry.register(
        ToolSpec(name="answer_buddy_self_question", description="self"),
        lambda args, state: ToolResult(ok=True, tool_name="answer_buddy_self_question", args=args, result={"answer": "Buddy is grounded."}, human_summary="Buddy is grounded."),
    )
    registry.register(
        ToolSpec(name="import_journal_trades", description="journal import"),
        lambda args, state: ToolResult(ok=True, tool_name="import_journal_trades", args=args, result={"imported_trades": 2}, human_summary="Journal import finished: 2 untracked trades imported."),
    )
    return registry


def test_planner_runtime_deterministic_status_plan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    plan = runtime.plan_request("show status --decisions", runtime_context={"granularity": "M5"})
    assert [step.tool for step in plan.steps] == ["get_status", "get_decisions"]


def test_planner_runtime_deterministic_status_handles_loose_phrase():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    plan = runtime.plan_request("give me status", runtime_context={"granularity": "M5"})
    assert [step.tool for step in plan.steps] == ["get_status"]


def test_planner_runtime_routes_plain_english_status_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("where are we at right now?", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["get_status"]
    assert "status" in response.answer.lower()


def test_planner_runtime_routes_plain_english_account_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("how's the account looking?", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["get_account_summary"]
    assert "account summary" in response.answer.lower()


def test_planner_runtime_routes_plain_english_journal_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("how did we do over the last 7 days?", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["get_journal_summary"]
    assert response.plan.steps[0].args["days"] == 7
    assert "journal summary" in response.answer.lower()


def test_planner_runtime_deterministic_predict_command_routes_to_tool():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("predict", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_deterministic_trade_command_routes_to_tool():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("trade", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_handles_scan_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("scan h1 and auto-execute top 2", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert "1 approved" in response.answer.lower()


def test_planner_runtime_scan_without_timeframe_defaults_to_h1():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("scan top 3", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert response.plan.steps[0].args["top_n"] == 3
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_emits_progress_events_for_tool_flow():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    events: list[tuple[str, dict]] = []

    response = runtime.handle_request(
        "scan h1",
        runtime_context={"granularity": "M5"},
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    event_names = [event for event, _payload in events]
    assert event_names[0] == "thinking_start"
    assert "plan_ready" in event_names
    assert "tool_start" in event_names
    assert "tool_done" in event_names
    assert "reasoning_start" in event_names
    assert event_names[-1] == "done"
    tool_start_payload = next(payload for event, payload in events if event == "tool_start")
    assert tool_start_payload["tool"] == "scan_market"


def test_planner_runtime_routes_run_agents_to_agent_confirmation_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "run agents",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 2,
                "approved_count": 0,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.62, "agent_total": 0.0, "error": None},
                    {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.58, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["run_agent_confirmation"] is True
    assert response.plan.steps[0].args["pairs"] == "EUR_USD,GBP_USD"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_run_agents_prefers_last_scan_granularity():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "run agents",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "granularity": "H1",
                "count": 1,
                "tradable_count": 1,
                "approved_count": 0,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.62, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert response.plan.steps[0].args["run_agent_confirmation"] is True
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_run_agents_defaults_to_h1_when_scan_granularity_missing():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "run agents",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "count": 1,
                "tradable_count": 1,
                "approved_count": 0,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.62, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert response.plan.steps[0].args["run_agent_confirmation"] is True
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_execute_trades_followup_to_scan_context():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "execute trades",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 1,
                "tradable_count": 1,
                "approved_count": 1,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.72, "agent_total": 1.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["pairs"] == "EUR_USD"
    assert response.plan.steps[0].args["auto_execute"] is True
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_execute_trades_without_context_to_scan_execute():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("execute trades", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["auto_execute"] is True
    assert response.plan.steps[0].args["granularity"] == "M5"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_execute_with_prediction_context_to_trade():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("execute", runtime_context={"granularity": "H1", "has_prediction": True})
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert response.plan.steps[0].args["command"] == "trade"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_routes_execute_after_scan_to_scan_auto_execute_top_pair():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "execute",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 4,
                "tradable_count": 4,
                "approved_count": 2,
                "granularity": "H1",
                "session_filter_effective": True,
                "all_rows": [
                    {"pair": "USD_JPY", "direction": "LONG", "gates_passed": True, "confidence": 0.91, "agent_total": 3.0, "error": None},
                    {"pair": "NZD_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.88, "agent_total": 2.0, "error": None},
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.77, "agent_total": 1.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["pairs"] == "USD_JPY"
    assert response.plan.steps[0].args["auto_execute"] is True
    assert response.plan.steps[0].args["top_n"] == 1
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_execute_named_pair_after_scan_to_scan_auto_execute():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "execute nzd/usd",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 4,
                "tradable_count": 4,
                "approved_count": 2,
                "granularity": "H1",
                "session_filter_effective": False,
                "all_rows": [
                    {"pair": "USD_JPY", "direction": "LONG", "gates_passed": True, "confidence": 0.91, "agent_total": 3.0, "error": None},
                    {"pair": "NZD_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.88, "agent_total": 2.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["pairs"] == "NZD_USD"
    assert response.plan.steps[0].args["auto_execute"] is True
    assert response.plan.steps[0].args["force"] is True
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_scan_execute_trades_phrase_sets_auto_execute():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("scan h1 execute trades top 2", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["auto_execute"] is True
    assert response.plan.steps[0].args["top_n"] == 2
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_ignore_session_filter_to_override_guardrail():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("ignore session filter", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["guardrail_name"] == "session"
    assert response.plan.steps[0].args["action"] == "disable"
    assert response.plan.steps[0].args["scope"] == "this_scan"
    assert "guardrail override applied" in response.answer.lower()


def test_planner_runtime_bypass_session_filter_skips_local_voice_adapter(monkeypatch):
    runtime = PlannerRuntime(
        config_path="config.yaml",
        registry=_registry(),
        provider=None,
        adapter_path="trained_data/planner/checkpoints",
    )

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("bypass session filter", runtime_context={"granularity": "M5"})

    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert "guardrail override applied" in response.answer.lower()
    assert called["adapter"] is False


def test_planner_runtime_routes_override_plus_scan_to_two_steps():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("ignore session filter and scan h1 top 2", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["override_guardrail", "scan_market"]
    assert response.plan.steps[0].args["action"] == "disable"
    assert response.plan.steps[1].args["granularity"] == "H1"
    assert response.plan.steps[1].args["top_n"] == 2


def test_planner_runtime_routes_override_with_this_session_scope():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("ignore session filter for this session", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["scope"] == "this_session"


def test_planner_runtime_routes_allow_trades_at_2am_to_set_window():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("allow trades at 2am utc", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["action"] == "set_window_utc"
    start = int(response.plan.steps[0].args["window_start_utc"])
    end = int(response.plan.steps[0].args["window_end_utc"])
    assert start <= 2 < end


def test_planner_runtime_routes_permanent_override_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("disable session filter permanently", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["scope"] == "permanent"


def test_planner_runtime_routes_confirm_pending_override_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "confirm override",
        runtime_context={
            "granularity": "M5",
            "pending_guardrail_override": {
                "guardrail_name": "session",
                "action": "disable",
                "scope": "permanent",
                "confirmation_token": "abc123ef",
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["confirm_token"] == "abc123ef"


def test_planner_runtime_routes_cancel_pending_override_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "cancel override",
        runtime_context={
            "granularity": "M5",
            "pending_guardrail_override": {
                "guardrail_name": "session",
                "action": "disable",
                "scope": "permanent",
                "confirmation_token": "abc123ef",
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["override_guardrail"]
    assert response.plan.steps[0].args["confirm_token"] == "cancel"


def test_planner_runtime_routes_near_miss_scan_without_adapter(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("scaan m5", runtime_context={"granularity": "M5"})

    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "M5"
    assert "scan finished" in response.answer.lower()
    assert called["adapter"] is False


def test_planner_runtime_turns_opportunity_request_into_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("look for an opportunity", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_preserves_force_on_opportunity_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("force find an opportunity", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["force"] is True
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_guardrail_override_plus_opportunity_request_to_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "ignore session filter and find an opportunity",
        runtime_context={"granularity": "M5"},
    )
    assert [step.tool for step in response.plan.steps] == ["override_guardrail", "scan_market"]
    assert response.plan.steps[1].args["force"] is False
    assert response.plan.steps[1].args["granularity"] == "H1"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_handles_pair_comparison():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("why eur/usd and not gbp/usd?", runtime_context={})
    assert [step.tool for step in response.plan.steps] == ["compare_scan_results"]
    assert "eur_usd is stronger" in response.answer.lower()


def test_planner_runtime_routes_self_knowledge_to_grounded_tool():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("self knowledge", runtime_context={})
    assert [step.tool for step in response.plan.steps] == ["answer_buddy_self_question"]
    assert "buddy is grounded" in response.answer.lower()


def test_planner_runtime_defaults_ambiguous_request_to_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("do the thing", runtime_context={})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_plain_english_best_pair_request_to_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("find best pair to trade tomorrow", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_defaults_go_ahead_to_scan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("go ahead", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_plain_english_prediction_request_with_pair():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("what do you think about eur/usd?", runtime_context={"granularity": "M15"})
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert response.plan.steps[0].args["value"] == "EUR_USD"
    assert response.plan.steps[0].args["granularity"] == "M15"
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_routes_alias_prediction_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("what do you think about cable?", runtime_context={"granularity": "M15"})
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert response.plan.steps[0].args["value"] == "GBP_USD"
    assert response.plan.steps[0].args["granularity"] == "M15"
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_routes_sentiment_config_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "add sentiment confirmation to AUD/NZD shorts only",
        runtime_context={"granularity": "H1"},
    )
    assert [step.tool for step in response.plan.steps] == ["configure_pair_sentiment_gate"]
    assert response.plan.steps[0].args["pair"] == "AUD_NZD"
    assert response.plan.steps[0].args["directions"] == ["short"]
    assert response.plan.steps[0].args["enabled"] is True
    assert "sentiment gate updated" in response.answer.lower()


def test_planner_runtime_routes_meta_retrain_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "retrain the meta-labeler across all masters",
        runtime_context={"granularity": "H1"},
    )
    assert [step.tool for step in response.plan.steps] == ["retrain_meta_labelers"]
    assert response.plan.steps[0].args["data_source"] == "journal"
    assert "meta-labeler retraining finished" in response.answer.lower()


def test_planner_runtime_routes_regime_sizing_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "enable aggressive volatility-regime sizing on the basket",
        runtime_context={"granularity": "H1"},
    )
    assert [step.tool for step in response.plan.steps] == ["configure_aggressive_regime_sizing"]
    assert response.plan.steps[0].args["enabled"] is True
    assert "aggressive volatility-regime sizing enabled" in response.answer.lower()


def test_planner_runtime_routes_loosen_gates_request_to_aggressive_profile_and_rescan():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "loosen gates for trades to pass",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 5,
                "tradable_count": 5,
                "approved_count": 0,
                "granularity": "H1",
                "session_filter_effective": True,
                "all_rows": [
                    {"pair": "USD_JPY", "direction": "LONG", "gates_passed": False, "confidence": 0.76, "agent_total": 0.0, "error": None},
                    {"pair": "NZD_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.73, "agent_total": 0.0, "error": None},
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": False, "confidence": 0.70, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["configure_scan_profile", "scan_market"]
    assert response.plan.steps[0].args["profile"] == "aggressive"
    assert response.plan.steps[1].args["granularity"] == "H1"
    assert response.plan.steps[1].args["pairs"] == "USD_JPY,NZD_USD,EUR_USD"
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_routes_basket_backtest_with_sentiment():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "run full backtest on euro and aussie basket with sentiment on AUD/NZD shorts and report equity curve",
        runtime_context={"granularity": "H1"},
    )
    assert [step.tool for step in response.plan.steps] == ["configure_pair_sentiment_gate", "run_basket_backtest"]
    assert response.plan.steps[0].args["pair"] == "AUD_NZD"
    assert response.plan.steps[0].args["directions"] == ["short"]
    assert response.plan.steps[1].args["baskets"] == ["euro", "aussie"]
    assert response.plan.steps[1].args["pairs"] is None
    assert "basket backtest returned" in response.answer.lower()


def test_planner_runtime_routes_desk_alias_prediction_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("what do you think about uj?", runtime_context={"granularity": "M15"})
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert response.plan.steps[0].args["value"] == "USD_JPY"
    assert response.plan.steps[0].args["granularity"] == "M15"
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_routes_gold_alias_prediction_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("what do you think about gold?", runtime_context={"granularity": "H1"})
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert response.plan.steps[0].args["value"] == "XAU_USD"
    assert response.plan.steps[0].args["granularity"] == "H1"
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_clarifies_ambiguous_prediction_alias():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("what do you think about euro?", runtime_context={"granularity": "M5"})
    assert response.plan.final_answer is not None
    assert "ambiguous" in response.answer.lower()
    assert "eur_usd" in response.answer.lower()
    assert "eur_jpy" in response.answer.lower()


def test_planner_runtime_routes_short_alias_to_pair_specific_trade():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("short cable", runtime_context={"granularity": "H1"})
    assert [step.tool for step in response.plan.steps] == ["use_market_source", "run_trade_command"]
    assert response.plan.steps[0].args["value"] == "GBP_USD"
    assert response.plan.steps[1].args["command"] == "sell"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_clarifies_ambiguous_buy_alias():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("buy yen", runtime_context={"granularity": "H1"})
    assert response.plan.final_answer is not None
    assert "ambiguous" in response.answer.lower()
    assert "usd_jpy" in response.answer.lower()
    assert "eur_jpy" in response.answer.lower()


def test_planner_runtime_routes_flatten_alias_to_pair_specific_close():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("flatten cable", runtime_context={"granularity": "H1"})
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert response.plan.steps[0].args["command"] == "trade close GBP_USD"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_clarifies_ambiguous_fade_alias():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("fade euro", runtime_context={"granularity": "H1"})
    assert response.plan.final_answer is not None
    assert "ambiguous" in response.answer.lower()
    assert "eur_usd" in response.answer.lower()
    assert "eur_jpy" in response.answer.lower()


def test_planner_runtime_clarifies_ambiguous_short_alias():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("short euro", runtime_context={"granularity": "H1"})
    assert response.plan.final_answer is not None
    assert "ambiguous" in response.answer.lower()
    assert "eur_usd" in response.answer.lower()
    assert "eur_jpy" in response.answer.lower()


def test_planner_runtime_routes_prediction_followup_go_ahead_to_trade():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("go ahead", runtime_context={"granularity": "M5", "has_prediction": True})
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert response.plan.steps[0].args["command"] == "trade"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_routes_scan_followup_go_ahead_to_prediction():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "go ahead",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 2,
                "approved_count": 1,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.82, "agent_total": 2.0, "error": None},
                    {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.61, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["run_prediction"]
    assert response.plan.steps[0].args["value"] == "EUR_USD"
    assert "prediction ready" in response.answer.lower()


def test_planner_runtime_routes_scan_followup_why_to_comparison():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "why",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 2,
                "approved_count": 1,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.82, "agent_total": 2.0, "error": None},
                    {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.61, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert [step.tool for step in response.plan.steps] == ["compare_scan_results"]
    assert response.plan.steps[0].args == {"left_pair": "EUR_USD", "right_pair": "GBP_USD"}
    assert "stronger" in response.answer.lower()


def test_planner_runtime_routes_scan_followup_that_one_to_grounded_answer():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "that one",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 2,
                "approved_count": 1,
                "all_rows": [
                    {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.82, "agent_total": 2.0, "error": None},
                    {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.61, "agent_total": 0.0, "error": None},
                ],
            },
        },
    )
    assert response.plan.final_answer is not None
    assert "eur_usd" in response.answer.lower()
    assert "top approved setup" in response.answer.lower()


def test_planner_runtime_routes_close_that_trade_to_active_instrument():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "close that trade",
        runtime_context={"granularity": "M5", "active_instrument": "EUR_USD", "has_prediction": True},
    )
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert response.plan.steps[0].args["command"] == "trade close EUR_USD"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_routes_flatten_that_trade_to_active_instrument():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request(
        "flatten that trade",
        runtime_context={"granularity": "M5", "active_instrument": "GBP_JPY", "has_prediction": True},
    )
    assert [step.tool for step in response.plan.steps] == ["run_trade_command"]
    assert response.plan.steps[0].args["command"] == "trade close GBP_JPY"
    assert "trade command ran" in response.answer.lower()


def test_planner_runtime_uses_local_adapter_plan(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")

    monkeypatch.setattr(runtime, "_load_local_adapter", lambda: {"fake": True})
    monkeypatch.setattr(
        runtime,
        "_generate_local_adapter_text",
        lambda *_args, **_kwargs: json.dumps(
            {
                "target_plan": {
                    "needs_clarification": False,
                    "clarification_question": None,
                    "final_answer": None,
                    "steps": [{"tool": "scan_market", "args": {"granularity": "H1", "top_n": 2}}],
                },
                "target_response": "I scanned H1.",
            }
        ),
    )

    response = runtime.handle_request("scan top 2", runtime_context={"granularity": "M5"})
    assert [step.tool for step in response.plan.steps] == ["scan_market"]
    assert response.plan.steps[0].args["top_n"] == 2
    assert "scan finished" in response.answer.lower()


def test_planner_runtime_uses_target_response_for_direct_answer():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    plan = runtime._coerce_plan(
        json.dumps(
            {
                "target_plan": {
                    "needs_clarification": False,
                    "clarification_question": None,
                    "final_answer": None,
                    "steps": [],
                },
                "target_response": "Buddy is grounded.",
            }
        )
    )

    assert plan is not None
    assert plan.final_answer == "Buddy is grounded."


def test_planner_runtime_hybrid_routing_skips_adapter_for_known_request(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    plan = runtime.plan_request("show status --decisions", runtime_context={"granularity": "M5"})

    assert [step.tool for step in plan.steps] == ["get_status", "get_decisions"]
    assert called["adapter"] is False


def test_planner_runtime_answers_capability_question_without_adapter(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("what can you do?", runtime_context={"granularity": "M5"})

    assert response.plan.final_answer is not None
    assert "scan for setups" in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_answers_capability_question_with_spaced_punctuation(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("what can you do ?", runtime_context={"granularity": "M5"})

    assert response.plan.final_answer is not None
    assert "scan for setups" in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_explains_session_gated_scan_errors_without_adapter(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request(
        "whats with the error ?",
        runtime_context={
            "granularity": "M5",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 0,
                "approved_count": 0,
                "all_rows": [
                    {
                        "pair": "EUR_USD",
                        "direction": "HOLD",
                        "gates_passed": False,
                        "confidence": 0.0,
                        "agent_total": 0,
                        "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)",
                    },
                    {
                        "pair": "GBP_USD",
                        "direction": "HOLD",
                        "gates_passed": False,
                        "confidence": 0.0,
                        "agent_total": 0,
                        "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)",
                    },
                ],
            },
        },
    )

    assert response.plan.final_answer is not None
    assert "session filter" in response.answer.lower()
    assert "use force" in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_explains_market_closed_scan_errors_without_suggesting_force(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request(
        "whats with the error ?",
        runtime_context={
            "granularity": "H1",
            "last_scan_summary": {
                "count": 2,
                "tradable_count": 0,
                "approved_count": 0,
                "all_rows": [
                    {
                        "pair": "EUR_USD",
                        "direction": "HOLD",
                        "gates_passed": False,
                        "confidence": 0.0,
                        "agent_total": 0,
                        "error": "FX market closed (weekend: Saturday 17:00 UTC; reopens Sunday 22:00 UTC)",
                    },
                    {
                        "pair": "GBP_USD",
                        "direction": "HOLD",
                        "gates_passed": False,
                        "confidence": 0.0,
                        "agent_total": 0,
                        "error": "FX market closed (weekend: Saturday 17:00 UTC; reopens Sunday 22:00 UTC)",
                    },
                ],
            },
        },
    )

    assert response.plan.final_answer is not None
    assert "market is closed" in response.answer.lower()
    assert "session filter" not in response.answer.lower()
    assert "use force" not in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_explains_why_dry_run_without_adapter(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("why dry run?", runtime_context={"granularity": "M5", "execute_mode": "dry-run"})

    assert response.plan.final_answer is not None
    assert "defaults to safe execution" in response.answer.lower()
    assert "execute on" in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_uses_local_voice_for_direct_answer(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")
    monkeypatch.setattr(runtime, "_load_local_adapter", lambda: {"fake": True})
    monkeypatch.setattr(runtime, "_generate_local_adapter_text", lambda *_args, **_kwargs: "I can scan, explain the current mode, and work from the last grounded context.")

    response = runtime.handle_request("what can you do?", runtime_context={"granularity": "M5"})

    assert response.answer == "I can scan, explain the current mode, and work from the last grounded context."


def test_planner_runtime_uses_local_voice_for_tool_answer(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setattr(runtime, "_load_local_adapter", lambda: {"fake": True})
    monkeypatch.setattr(runtime, "_generate_local_adapter_text", lambda *_args, **_kwargs: "I'm on dry-run right now on M5, with risk set from config.")

    response = runtime.handle_request("status", runtime_context={"granularity": "M5"})

    assert response.answer == "I'm on dry-run right now on M5, with risk set from config."


def test_planner_runtime_answers_plain_no_without_clarification(monkeypatch):
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path="trained_data/planner/checkpoints")
    monkeypatch.setenv("BUDDY_PLANNER_HYBRID_ROUTING", "1")

    called = {"adapter": False}

    def _fake_load():
        called["adapter"] = True
        return None

    monkeypatch.setattr(runtime, "_load_local_adapter", _fake_load)
    response = runtime.handle_request("no", runtime_context={"granularity": "M5"})

    assert response.plan.final_answer is not None
    assert "tell me the task directly" in response.answer.lower()
    assert called["adapter"] is True


def test_planner_runtime_accepts_zip_adapter_bundle_and_extracts_all_files(tmp_path):
    bundle_path = tmp_path / "checkpoint-900.zip"
    bundle_files = {
        "adapter_config.json": json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}),
        "adapter_model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "README.md": "planner bundle",
        "training_args.bin": "args",
        "optimizer.pt": "optim",
    }
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        for name, contents in bundle_files.items():
            bundle.writestr(name, contents)

    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path=str(bundle_path))
    resolved = runtime._resolved_adapter_path()

    assert resolved is not None
    assert resolved.is_dir()
    assert (resolved / "adapter_config.json").exists()
    assert (resolved / "adapter_model.safetensors").exists()
    assert (resolved / "tokenizer.json").exists()
    assert (resolved / "tokenizer_config.json").exists()
    assert (resolved / "README.md").exists()
    assert (resolved / "training_args.bin").exists()
    assert (resolved / "optimizer.pt").exists()


def test_planner_runtime_finds_adapter_root_inside_nested_zip_bundle(tmp_path):
    bundle_path = tmp_path / "nested-checkpoint-900.zip"
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        bundle.writestr("checkpoint_900/adapter_config.json", json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}))
        bundle.writestr("checkpoint_900/adapter_model.safetensors", "weights")
        bundle.writestr("checkpoint_900/tokenizer.json", "{}")
        bundle.writestr("checkpoint_900/tokenizer_config.json", "{}")
        bundle.writestr("checkpoint_900/chat_template.jinja", "{{ bos_token }}")

    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None, adapter_path=str(bundle_path))
    resolved = runtime._resolved_adapter_path()

    assert resolved is not None
    assert resolved.name == "checkpoint_900"
    assert (resolved / "chat_template.jinja").exists()


def test_planner_runtime_create_default_prefers_checkpoint_zip(monkeypatch):
    monkeypatch.delenv("BUDDY_PLANNER_ADAPTER_PATH", raising=False)
    runtime = PlannerRuntime.create_default("config.yaml")
    assert runtime.adapter_path is not None
    assert runtime.adapter_path.endswith("trained_data/checkpoint-900.zip")
