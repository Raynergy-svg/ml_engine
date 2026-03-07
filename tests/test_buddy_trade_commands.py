import pytest
from types import SimpleNamespace
import builtins
import sys

pytest.importorskip("unified_talk")

import unified_talk


def _dummy_fx_df():
    import pandas as pd

    base = 1.1000
    rows = []
    for i in range(20):
        c = base + (i * 0.00001)
        rows.append(
            {
                "open": c,
                "high": c + 0.00005,
                "low": c - 0.00005,
                "close": c,
                "volume": 100.0,
                "bid_close": c,
                "ask_close": c + 0.00002,  # 0.2 pips on non-JPY pairs
            }
        )
    return pd.DataFrame(rows)


class _StubOandaClient:
    def __init__(self):
        self.orders = []
        self.closes = []

    def create_market_order(self, *, instrument: str, units: int, **kwargs):
        self.orders.append((instrument, int(units)))
        # mimic OANDA response shape
        return {"orderCreateTransaction": {"id": "T-1"}}

    def close_position(self, *, instrument: str, **kwargs):
        self.closes.append(str(instrument))
        return {"closePositionTransaction": {"id": "C-1"}}

    def get_account_summary(self):
        return {
            "account": {
                "NAV": "10123.45",
                "balance": "10000.00",
                "marginAvailable": "8450.00",
                "currency": "USD",
            }
        }


def _ctx():
    # We avoid loading checkpoints by constructing a minimal TalkContext.
    return unified_talk.TalkContext(
        config_path="config_tuned.yaml",
        checkpoint_path="/dev/null",
        feature_columns=["open", "high", "low", "close", "volume"],
        sequence_length=60,
        target_shift=1,
        engine=None,  # type: ignore[arg-type]
        reasoning=None,  # type: ignore[arg-type]
        oanda_client=_StubOandaClient(),
        oanda_instrument="EUR_USD",
        oanda_execute=False,
        assistant_name="Buddy",
        verbose=False,
    )


@pytest.fixture(autouse=True)
def _stable_chat_dependencies(monkeypatch):
    monkeypatch.setattr(
        unified_talk,
        "_fx_trade_gate",
        lambda *_args, **_kwargs: (True, "confidence 0.90 band=high", 0.90),
    )
    monkeypatch.setattr(unified_talk, "_load_df_from_oanda", lambda *args, **kwargs: _dummy_fx_df())
    monkeypatch.setattr(
        unified_talk,
        "load_config",
        lambda *_args, **_kwargs: {"fx": {"risk": {"risk_per_trade_pct": 0.02}}},
    )
    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: kwargs["fallback"],
    )


def _prime_fx_context(ctx):
    # Ensure Tier-1 session gating is deterministic for tests.
    # 14:00 UTC on Dec 16 == 09:00 America/New_York.
    from datetime import datetime, timezone

    ctx.fx_now = datetime(2025, 12, 16, 14, 0, tzinfo=timezone.utc)
    ctx.active_df = _dummy_fx_df()
    ctx.active_source = "oanda:EUR_USD"
    return ctx


def test_trade_auto_dry_run_uses_last_prediction(capsys):
    ctx = _prime_fx_context(_ctx())
    # Provide state_probs + risk so confidence gating can evaluate.
    ctx.last_result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "buy" in out.lower()
    assert ctx.oanda_client.orders == []


def test_trade_auto_execute_places_order(capsys):
    ctx = _prime_fx_context(_ctx())
    ctx.oanda_execute = True
    ctx.last_result = {
        "prediction": 1.0000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "auto-trade executed" in out.lower()
    expected_units = -unified_talk._default_trade_units(ctx, "EUR_USD")
    assert ctx.oanda_client.orders == [("EUR_USD", expected_units)]


def test_manual_trade_buy_dry_run(capsys):
    ctx = _prime_fx_context(_ctx())

    # Manual trades also require a last prediction (for confidence checks).
    ctx.last_result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade buy 2000 EUR_USD", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "market order" in out.lower()
    assert ctx.oanda_client.orders == []


def test_manual_trade_close_dry_run(capsys):
    ctx = _prime_fx_context(_ctx())

    assert unified_talk._handle_talk_command(ctx, "trade close EUR_USD", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "close" in out.lower()
    assert ctx.oanda_client.closes == []


def test_agentic_request_runs_predict_then_trade(monkeypatch, capsys):
    ctx = _prime_fx_context(_ctx())

    monkeypatch.setattr(
        unified_talk,
        "_predict_one_locked",
        lambda *_args, **_kwargs: {
            "prediction": 1.2000,
            "last_close": 1.1000,
            "state_probs": [0.05, 0.05, 0.90],
            "risk": 0.10,
        },
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "analyze this setup and take the trade if it's strong",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "dry-run" in out
    assert "buy" in out
    assert ctx.last_result["prediction"] == 1.2000


def test_agentic_request_switches_pair_and_answers_risk(monkeypatch, capsys):
    ctx = _ctx()

    monkeypatch.setattr(unified_talk, "_load_df_from_oanda", lambda *args, **kwargs: _dummy_fx_df())
    monkeypatch.setattr(
        unified_talk,
        "_predict_one_locked",
        lambda *_args, **_kwargs: {
            "prediction": 1.0500,
            "last_close": 1.0490,
            "state_probs": [0.20, 0.20, 0.60],
            "risk": 0.82,
        },
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "check GBP/USD and tell me how risky it is",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert ctx.oanda_instrument == "GBP_USD"
    assert "risk is high" in out


def test_agentic_request_chains_summary_then_sl_tp(monkeypatch, capsys):
    ctx = _ctx()

    monkeypatch.setattr(unified_talk, "_load_df_from_oanda", lambda *args, **kwargs: _dummy_fx_df())
    monkeypatch.setattr(
        unified_talk,
        "_predict_one_locked",
        lambda *_args, **_kwargs: {
            "prediction": 1.1200,
            "last_close": 1.1000,
            "state_probs": [0.05, 0.10, 0.85],
            "risk": 0.20,
            "trend": 0.03,
        },
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "switch to GBP/USD, summarize it, then set stop loss to 25 and take profit to 50",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert ctx.oanda_instrument == "GBP_USD"
    assert ctx.stop_loss_pips == 25.0
    assert ctx.take_profit_pips == 50.0
    assert "here's what i'm seeing" in out
    assert "stop loss set to 25.0 pips" in out
    assert "take profit set to 50.0 pips" in out


def test_buddy_self_query_uses_grounded_knowledge(monkeypatch, capsys):
    ctx = _ctx()

    monkeypatch.setattr(
        unified_talk,
        "answer_buddy_question",
        lambda *args, **kwargs: "Based on my current workspace: grounded Buddy self-knowledge is active.",
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "how do you work?",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "grounded buddy self-knowledge is active" in out


def test_knowledge_only_smalltalk_does_not_try_prediction(monkeypatch, capsys):
    ctx = _ctx()
    ctx.active_df = _dummy_fx_df()
    ctx.active_source = "oanda:EUR_USD:M5:300"

    def _boom(*_args, **_kwargs):
        raise AssertionError("prediction should not run in knowledge-only smalltalk")

    monkeypatch.setattr(unified_talk, "_predict_one_locked", _boom)

    assert unified_talk._handle_talk_command(
        ctx,
        "hello",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "knowledge-first mode" in out or "i'm here in knowledge-first mode" in out


def test_knowledge_only_buddy_prompt_does_not_try_prediction(monkeypatch, capsys):
    ctx = _ctx()
    ctx.active_df = _dummy_fx_df()
    ctx.active_source = "oanda:EUR_USD:M5:300"

    def _boom(*_args, **_kwargs):
        raise AssertionError("prediction should not run for knowledge-only buddy prompt")

    monkeypatch.setattr(unified_talk, "_predict_one_locked", _boom)

    assert unified_talk._handle_talk_command(
        ctx,
        "buddy",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "knowledge-first mode" in out or "grounded" in out


def test_planner_mode_help_shows_natural_language_examples(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")

    assert unified_talk._handle_talk_command(
        ctx,
        "help",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "planner mode" in out
    assert "look for an opportunity" in out
    assert "scan rather than bounce you into a menu" in out


def test_planner_mode_banner_shows_buddy_planner_3b(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")

    unified_talk._print_talk_banner(ctx)

    out = capsys.readouterr().out.lower()
    assert "buddy planner 3b:" in out
    assert "buddy planner 3b" in out
    assert "prompt: scan-first grounded operator" in out
    assert "planner mode" in out
    assert "\x1b" not in out


def test_non_tty_eof_does_not_spin_forever(monkeypatch, capsys):
    ctx = _ctx()
    prompts = []
    inputs = iter(["scan for setups"])
    handled = []

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    def _fake_input(prompt):
        prompts.append(prompt)
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", _fake_input)
    monkeypatch.setattr(
        unified_talk,
        "_handle_talk_command",
        lambda _ctx, q, **_kwargs: handled.append(q) or True,
    )

    unified_talk._run_repl(ctx, period="5d", interval="1h")

    out = capsys.readouterr().out
    assert handled == ["scan for setups"]
    assert prompts == ["You: ", "You: "]
    assert out.strip().endswith("Bye.")


def test_repl_only_shows_idle_prompt_once_until_user_responds(monkeypatch, capsys):
    ctx = _ctx()
    responses = iter([None, None, "status"])
    handled = []

    monkeypatch.setattr(
        unified_talk,
        "_read_input_with_timeout",
        lambda _prompt, timeout: next(responses),
    )
    monkeypatch.setattr(
        unified_talk,
        "_handle_talk_command",
        lambda _ctx, q, **_kwargs: handled.append(q) or False,
    )

    unified_talk._run_repl(ctx, period="5d", interval="1h")

    out = capsys.readouterr().out
    assert handled == ["status"]
    assert out.count("Still here? Type something or 'exit' to quit.") == 1


def test_knowledge_only_startup_skips_initial_oanda_warmup(monkeypatch):
    ctx = _ctx()
    called = {"warmup": False}

    monkeypatch.setattr(
        unified_talk,
        "_start_oanda_warmup",
        lambda *_args, **_kwargs: called.__setitem__("warmup", True),
    )

    unified_talk._maybe_run_initial_prediction(
        ctx,
        csv_path=None,
        ticker=None,
        period="5d",
        interval="1h",
        oanda=True,
    )

    assert called["warmup"] is False


def test_chat_status_query_reports_last_decisions(monkeypatch, capsys):
    ctx = _ctx()

    monkeypatch.setattr(
        unified_talk,
        "load_config",
        lambda *_args, **_kwargs: {"fx": {"risk": {"risk_per_trade_pct": 0.01}}},
    )
    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: kwargs["fallback"],
    )

    import memory_client

    monkeypatch.setattr(
        memory_client.MLEngineMemory,
        "get_runtime_state",
        lambda self, namespace: {
            "last_trade_execution_decision": {
                "decision_type": "dry_run_signal",
                "instrument": "EUR_USD",
                "direction": "long",
                "reason": "all model gates passed",
            },
            "last_no_trade_decision": {
                "decision_type": "intelligent_reject",
                "instrument": "GBP_USD",
                "reason": "event risk",
            },
        } if namespace == "buddy_runtime" else {},
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "show status with last decisions",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "current status" in out
    assert "all model gates passed" in out
    assert "event risk" in out


def test_chat_status_decisions_phrase_routes_to_runtime_decisions(monkeypatch, capsys):
    ctx = _ctx()

    monkeypatch.setattr(
        unified_talk,
        "_status_fact_lines",
        lambda _ctx: [
            "active_instrument=EUR_USD",
            "last_trade_execution_decision: dry_run_signal | instrument=EUR_USD | direction=long | reason=all model gates passed",
            "last_no_trade_decision: intelligent_reject | instrument=GBP_USD | reason=event risk",
        ],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "status --decisions",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "all model gates passed" in out
    assert "event risk" in out


def test_basic_exit_bypasses_planner(monkeypatch, capsys):
    ctx = _ctx()
    called = {"planner": False}

    monkeypatch.setattr(unified_talk, "_planner_chat_enabled", lambda: True)

    def _fail_if_called(*_args, **_kwargs):
        called["planner"] = True
        raise AssertionError("planner should not handle exit")

    monkeypatch.setattr(unified_talk, "_handle_planner_chat", _fail_if_called)

    assert unified_talk._handle_talk_command(ctx, "exit", period="5d", interval="1h") is False
    out = capsys.readouterr().out.lower()
    assert "bye." in out
    assert called["planner"] is False


def test_bare_status_routes_to_planner_in_planner_mode(monkeypatch, capsys):
    ctx = _ctx()

    class _FakePlanner:
        def handle_request(self, *args, **kwargs):
            from src.agent.planner_runtime import PlannerRuntimeResponse
            from src.agent.planner_types import PlannerPlan

            return PlannerRuntimeResponse(plan=PlannerPlan(final_answer="planner status"), answer="planner status", metadata={"state": {}})

    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")
    ctx.planner_runtime = _FakePlanner()
    monkeypatch.setattr(
        unified_talk,
        "_handle_status_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy status handler should not run for bare status")),
    )

    assert unified_talk._handle_talk_command(ctx, "status", period="5d", interval="1h") is True

    out = capsys.readouterr().out.lower()
    assert "planner status" in out


def test_predict_routes_through_planner_in_planner_mode(monkeypatch, capsys):
    ctx = _prime_fx_context(_ctx())
    ctx.last_result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")
    monkeypatch.setattr(
        unified_talk,
        "_handle_predict_commands",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy predict handler should not run before planner")),
    )

    class _FakePlanner:
        def handle_request(self, *args, **kwargs):
            from src.agent.planner_runtime import PlannerRuntimeResponse
            from src.agent.planner_types import PlannerPlan

            return PlannerRuntimeResponse(plan=PlannerPlan(final_answer="planner predict"), answer="planner predict", metadata={"state": {}})

    ctx.planner_runtime = _FakePlanner()

    assert unified_talk._handle_talk_command(ctx, "predict", period="5d", interval="1h") is True

    out = capsys.readouterr().out.lower()
    assert "planner predict" in out


def test_chat_scan_query_runs_scan_and_summarizes(monkeypatch, capsys):
    ctx = _ctx()
    calls = {}

    def _fake_scan(**kwargs):
        calls.update(kwargs)
        result = SimpleNamespace(
            pair="EUR_USD",
            direction="LONG",
            gates_passed=True,
            ridge_confidence=0.81,
            agent_total=2.0,
            error=None,
        )
        return [(result, True)]

    import cli.buddy_scanning

    monkeypatch.setattr(cli.buddy_scanning, "buddy_scan", _fake_scan)
    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: kwargs["fallback"],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "scan all pairs and explain the best setups",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert calls["clean_output"] is True
    assert calls["auto_execute"] is False
    assert "scan finished" in out
    assert "eur_usd long" in out


def test_chat_scan_query_can_auto_execute(monkeypatch, capsys):
    ctx = _ctx()
    calls = {}

    def _fake_scan(**kwargs):
        calls.update(kwargs)
        return []

    import cli.buddy_scanning

    monkeypatch.setattr(cli.buddy_scanning, "buddy_scan", _fake_scan)
    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: kwargs["fallback"],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "scan and auto-execute if approved",
        period="5d",
        interval="1h",
    ) is True

    capsys.readouterr()
    assert calls["auto_execute"] is True
    assert calls["no_execute"] is False


def test_chat_scan_query_supports_top_n_and_granularity(monkeypatch, capsys):
    ctx = _ctx()
    calls = {}

    def _fake_scan(**kwargs):
        calls.update(kwargs)
        return []

    import cli.buddy_scanning

    monkeypatch.setattr(cli.buddy_scanning, "buddy_scan", _fake_scan)

    assert unified_talk._handle_talk_command(
        ctx,
        "scan H1 and auto-execute top 2",
        period="5d",
        interval="1h",
    ) is True

    capsys.readouterr()
    assert calls["granularity"] == "H1"
    assert calls["top_n"] == 2
    assert calls["auto_execute"] is True


def test_chat_scan_followup_filters_approved_setups(capsys):
    ctx = _ctx()
    ctx.last_scan_summary = {
        "count": 3,
        "tradable_count": 2,
        "approved_count": 1,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.81, "agent_total": 2.0, "error": None},
            {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.64, "agent_total": 0.0, "error": None},
            {"pair": "USD_JPY", "direction": "NONE", "gates_passed": False, "confidence": 0.40, "agent_total": 0.0, "error": "no edge"},
        ],
    }

    assert unified_talk._handle_talk_command(
        ctx,
        "only show approved setups",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "approved setups from the last scan" in out
    assert "eur_usd long" in out
    assert "gbp_usd" not in out


def test_chat_model_status_query_reports_inventory(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setattr(
        unified_talk,
        "_model_status_fact_lines",
        lambda _ctx: [
            "pair_model_count=7",
            "validated_pair_model_count=5",
            "legacy_modular_ensemble_present=True",
            "pair_model=EUR_USD validated=True",
            "pair_model=GBP_USD validated=True",
        ],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "show model status",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "model status: 7 pair models found" in out
    assert "validated pair models: 5" in out


def test_chat_journal_update_query_reports_sync(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setattr(
        unified_talk,
        "_journal_update_fact_lines",
        lambda: [
            "closed_updated=2",
            "open_updated=1",
            "tracked_live_trades=3",
            "untracked_live_trades=0",
        ],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "update the journal from oanda",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "journal sync finished" in out
    assert "closed updated: 2" in out
    assert "open updated: 1" in out


def test_chat_journal_import_query_reports_import(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setattr(
        unified_talk,
        "_journal_import_fact_lines",
        lambda: [
            "imported_trades=3",
        ],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "import untracked trades into the journal",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "journal import finished" in out
    assert "imported 3 untracked trades" in out


def test_chat_scan_comparison_explains_why_one_setup_beats_another(capsys):
    ctx = _ctx()
    ctx.last_scan_summary = {
        "count": 2,
        "tradable_count": 2,
        "approved_count": 1,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "LONG", "gates_passed": True, "confidence": 0.81, "agent_total": 2.0, "error": None},
            {"pair": "GBP_USD", "direction": "SHORT", "gates_passed": False, "confidence": 0.64, "agent_total": 0.0, "error": None},
        ],
    }

    assert unified_talk._handle_talk_command(
        ctx,
        "why eur/usd and not gbp/usd?",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "comparing eur_usd vs gbp_usd" in out
    assert "eur_usd is stronger because it passed the trading gates" in out


def test_agentic_request_can_enable_execute_then_trade(monkeypatch, capsys):
    ctx = _prime_fx_context(_ctx())

    monkeypatch.setattr(unified_talk, "_load_df_from_oanda", lambda *args, **kwargs: _dummy_fx_df())
    monkeypatch.setattr(
        unified_talk,
        "_predict_one_locked",
        lambda *_args, **_kwargs: {
            "prediction": 1.2000,
            "last_close": 1.1000,
            "state_probs": [0.05, 0.05, 0.90],
            "risk": 0.10,
        },
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "switch to GBP/USD, turn execution on, then take the trade if it's strong",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert ctx.oanda_execute is True
    assert ctx.oanda_instrument == "GBP_USD"
    assert "execute mode on" in out
    assert "auto-trade executed" in out
    assert ctx.oanda_client.orders


def test_chat_account_query_uses_account_summary(monkeypatch, capsys):
    ctx = _ctx()
    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: kwargs["fallback"],
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "what's my account status right now?",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "account status" in out
    assert "nav=10123.45" in out


def test_chat_default_market_answer_uses_grounded_synthesis(monkeypatch, capsys):
    ctx = _prime_fx_context(_ctx())
    result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
        "trend": 0.03,
    }
    ctx.last_result = result

    monkeypatch.setattr(
        unified_talk,
        "_predict_one_locked",
        lambda *_args, **_kwargs: result,
    )

    monkeypatch.setattr(
        unified_talk,
        "_grounded_chat_reply",
        lambda _ctx, **kwargs: "Synthesized grounded answer.",
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "should i buy here?",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "synthesized grounded answer" in out


def test_planner_chat_mode_routes_through_planner_runtime(monkeypatch, capsys):
    ctx = _ctx()

    class _FakePlanner:
        def handle_request(self, *args, **kwargs):
            from src.agent.planner_runtime import PlannerRuntimeResponse
            from src.agent.planner_types import PlannerPlan
            return PlannerRuntimeResponse(plan=PlannerPlan(final_answer="planner answer"), answer="planner answer", metadata={"state": {}})

    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")
    ctx.planner_runtime = _FakePlanner()

    assert unified_talk._handle_talk_command(
        ctx,
        "show me status",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "planner answer" in out


def test_planner_chat_mode_runs_before_legacy_status_handler(monkeypatch, capsys):
    ctx = _ctx()

    class _FakePlanner:
        def handle_request(self, *args, **kwargs):
            from src.agent.planner_runtime import PlannerRuntimeResponse
            from src.agent.planner_types import PlannerPlan
            return PlannerRuntimeResponse(plan=PlannerPlan(final_answer="planner first"), answer="planner first", metadata={"state": {}})

    monkeypatch.setenv("BUDDY_CHAT_MODE", "planner")
    ctx.planner_runtime = _FakePlanner()
    monkeypatch.setattr(
        unified_talk,
        "_handle_status_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy status handler should not run first")),
    )

    assert unified_talk._handle_talk_command(
        ctx,
        "show me status",
        period="5d",
        interval="1h",
    ) is True

    out = capsys.readouterr().out.lower()
    assert "planner first" in out
# — Raynergy-svg —
