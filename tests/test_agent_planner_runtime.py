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
        ToolSpec(name="get_decisions", description="decisions"),
        lambda args, state: ToolResult(ok=True, tool_name="get_decisions", args=args, result={"last": "event risk"}, human_summary="Loaded last decisions."),
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
    assert response.plan.steps[0].args["granularity"] == "M5"
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


def test_planner_runtime_clarifies_ambiguous_request():
    runtime = PlannerRuntime(config_path="config.yaml", registry=_registry(), provider=None)
    response = runtime.handle_request("do the thing", runtime_context={})
    assert response.plan.needs_clarification is True
    assert "grounded action" in response.answer.lower()
    assert "show status" in response.answer.lower()


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
