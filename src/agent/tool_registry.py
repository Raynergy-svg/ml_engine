"""Tool registry for Buddy's planner runtime."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import logging
from pathlib import Path
import re
from typing import Any

from memory_client import MLEngineMemory
from src.utils import load_config
from src.utils.buddy_knowledge import answer_buddy_question

from .tool_schema import ToolArgSpec, ToolHandler, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of structured Buddy tools."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def available_tools(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def execute(self, tool_name: str, args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                args=dict(args),
                error=f"Unknown tool: {tool_name}",
                human_summary=f"I don't have a registered tool named {tool_name}.",
            )
        try:
            return handler(dict(args), state)
        except Exception as e:  # pragma: no cover - defensive runtime path
            logger.debug("Tool %s failed: %s", tool_name, e)
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                args=dict(args),
                error=str(e),
                human_summary=f"{tool_name} failed: {e}",
            )


def _normalize_pairs_arg(value: Any) -> str | None:
    if value in (None, "", [], ()):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).upper().replace("/", "_") for item in value if item)
    return str(value)


def _load_unified_talk_module() -> Any:
    for module_name in ("src.utils.unified_talk", "unified_talk"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    raise RuntimeError("Buddy planner tools require unified_talk to be importable.")


def _require_talk_context(state: dict[str, Any]) -> Any:
    ctx = state.get("_talk_context")
    if ctx is None:
        raise ValueError("Planner chat tool requires a live TalkContext.")
    return ctx


def _sync_state_from_talk_context(state: dict[str, Any], ctx: Any) -> None:
    state["active_instrument"] = getattr(ctx, "oanda_instrument", None)
    state["granularity"] = getattr(ctx, "oanda_granularity", None)
    state["execute_mode"] = "live" if getattr(ctx, "oanda_execute", False) else "dry-run"
    state["stop_loss_pips"] = getattr(ctx, "stop_loss_pips", None) if getattr(ctx, "use_stop_loss", False) else "off"
    state["take_profit_pips"] = getattr(ctx, "take_profit_pips", None) if getattr(ctx, "use_take_profit", False) else "off"
    state["knowledge_only"] = getattr(ctx, "engine", None) is None
    state["active_source"] = getattr(ctx, "active_source", None)
    state["has_prediction"] = getattr(ctx, "last_result", None) is not None
    if getattr(ctx, "last_scan_summary", None) is not None:
        state["last_scan_summary"] = ctx.last_scan_summary


def _capture_talk_output(callback: Any) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        callback()
    text = buf.getvalue().strip()
    if not text:
        return text
    return "\n".join(re.sub(r"^[A-Za-z][A-Za-z0-9_-]*:\s*", "", line) for line in text.splitlines()).strip()


def _tool_use_market_source() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()

        source_kind = str(args.get("source_kind") or "").strip().lower()
        value = str(args.get("value") or "").strip()
        period = str(args.get("period") or state.get("_period") or "5d")
        interval = str(args.get("interval") or state.get("_interval") or "1h")

        if source_kind == "csv" and value:
            command = f"use csv {value}"
        elif source_kind == "ticker" and value:
            command = f"use ticker {value}"
        elif source_kind == "oanda":
            instrument = value or str(args.get("instrument") or state.get("active_instrument") or getattr(ctx, "oanda_instrument", "EUR_USD"))
            granularity = str(args.get("granularity") or state.get("granularity") or getattr(ctx, "oanda_granularity", "M5"))
            candles = int(args.get("candles") or getattr(ctx, "oanda_candles", 300))
            command = f"use oanda {instrument} {granularity} {candles}"
        else:
            raise ValueError("use_market_source requires source_kind plus a valid value or instrument.")

        output = _capture_talk_output(
            lambda: unified_talk._handle_use_commands(ctx, command, period=period, interval=interval)
        )
        _sync_state_from_talk_context(state, ctx)
        return ToolResult(
            ok=True,
            tool_name="use_market_source",
            args=args,
            result={"active_source": ctx.active_source, "active_instrument": ctx.oanda_instrument, "granularity": ctx.oanda_granularity},
            human_summary=output or f"Loaded {ctx.active_source}.",
        )

    return handler


def _tool_run_prediction() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()
        period = str(args.get("period") or state.get("_period") or "5d")
        interval = str(args.get("interval") or state.get("_interval") or "1h")
        source_kind = str(args.get("source_kind") or "").strip().lower()
        value = str(args.get("value") or "").strip()

        setup_output = ""
        if source_kind:
            if source_kind == "csv" and value:
                setup_command = f"use csv {value}"
            elif source_kind == "ticker" and value:
                setup_command = f"use ticker {value}"
            elif source_kind == "oanda":
                instrument = value or str(args.get("instrument") or state.get("active_instrument") or getattr(ctx, "oanda_instrument", "EUR_USD"))
                granularity = str(args.get("granularity") or state.get("granularity") or getattr(ctx, "oanda_granularity", "M5"))
                candles = int(args.get("candles") or getattr(ctx, "oanda_candles", 300))
                setup_command = f"use oanda {instrument} {granularity} {candles}"
            else:
                raise ValueError("run_prediction received an unsupported source_kind/value combination.")
            setup_output = _capture_talk_output(
                lambda: unified_talk._handle_use_commands(ctx, setup_command, period=period, interval=interval)
            )

        command = str(args.get("command") or "").strip()
        if not command:
            if source_kind in {"csv", "ticker"} and value:
                command = f"predict {source_kind} {value}"
            else:
                command = "predict"

        output = _capture_talk_output(
            lambda: unified_talk._handle_predict_commands(ctx, command, command.lower(), period=period, interval=interval)
        )
        _sync_state_from_talk_context(state, ctx)
        summary = "\n".join(part for part in (setup_output, output) if part).strip()
        return ToolResult(
            ok=True,
            tool_name="run_prediction",
            args=args,
            result={"active_source": ctx.active_source, "has_prediction": ctx.last_result is not None, "prediction": ctx.last_result or {}},
            human_summary=summary or "Prediction run completed.",
        )

    return handler


def _tool_summarize_last_prediction() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()
        query = str(args.get("query") or "").strip()

        if query and query.lower() not in {"summary", "overview"}:
            answer = unified_talk._answer_from_last(ctx, query)
        else:
            answer = _capture_talk_output(lambda: unified_talk._handle_summary_command(ctx))

        _sync_state_from_talk_context(state, ctx)
        return ToolResult(
            ok=True,
            tool_name="summarize_last_prediction",
            args=args,
            result={"has_prediction": ctx.last_result is not None},
            human_summary=answer or "I don't have a recent prediction yet.",
        )

    return handler


def _tool_run_runtime_command() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()
        command = str(args.get("command") or "").strip()
        ql = command.lower()

        def _run() -> None:
            basic_out = unified_talk._handle_basic_commands(ctx, ql)
            if basic_out is not None:
                return
            sl_out = unified_talk._handle_sl_command(ctx, ql)
            if sl_out is not None:
                return
            tp_out = unified_talk._handle_tp_command(ctx, ql)
            if tp_out is not None:
                return
            raise ValueError(f"Unsupported runtime command: {command}")

        output = _capture_talk_output(_run)
        _sync_state_from_talk_context(state, ctx)
        return ToolResult(
            ok=True,
            tool_name="run_runtime_command",
            args=args,
            result={"execute_mode": state.get("execute_mode"), "stop_loss_pips": state.get("stop_loss_pips"), "take_profit_pips": state.get("take_profit_pips")},
            human_summary=output or f"Applied runtime command: {command}",
        )

    return handler


def _tool_run_oanda_command() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()
        command = str(args.get("command") or "").strip()
        ql = command.lower()
        output = _capture_talk_output(
            lambda: unified_talk._handle_oanda_info_commands(ctx, command, ql)
        )
        _sync_state_from_talk_context(state, ctx)
        return ToolResult(ok=True, tool_name="run_oanda_command", args=args, result={}, human_summary=output or f"Ran {command}.")

    return handler


def _tool_run_trade_command() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        ctx = _require_talk_context(state)
        unified_talk = _load_unified_talk_module()
        command = str(args.get("command") or "").strip()
        ql = command.lower()
        output = _capture_talk_output(
            lambda: unified_talk._handle_trade_commands(ctx, command, ql)
        )
        _sync_state_from_talk_context(state, ctx)
        return ToolResult(ok=True, tool_name="run_trade_command", args=args, result={"execute_mode": state.get("execute_mode")}, human_summary=output or f"Ran trade command: {command}")

    return handler


def _tool_answer_buddy_self_question(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        answer = answer_buddy_question(query, config_path=config_path, use_llm=False)
        return ToolResult(
            ok=True,
            tool_name="answer_buddy_self_question",
            args=args,
            result={"answer": answer},
            human_summary=answer,
        )

    return handler


def _tool_get_status(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        cfg = load_config(config_path)
        memory = MLEngineMemory()
        runtime = memory.get_runtime_state("buddy_runtime")
        status = {
            "active_instrument": state.get("active_instrument"),
            "granularity": state.get("granularity"),
            "execute_mode": state.get("execute_mode"),
            "stop_loss_pips": state.get("stop_loss_pips"),
            "take_profit_pips": state.get("take_profit_pips"),
            "configured_risk_per_trade_pct": (cfg.get("fx", {}) or {}).get("risk", {}).get("risk_per_trade_pct"),
            "last_trade_execution_decision": runtime.get("last_trade_execution_decision") if isinstance(runtime, dict) else None,
            "last_no_trade_decision": runtime.get("last_no_trade_decision") if isinstance(runtime, dict) else None,
        }
        summary = (
            f"You're on {status['active_instrument']} {status['granularity']}. "
            f"Execution is {status['execute_mode']} and configured risk is {status['configured_risk_per_trade_pct']} per trade."
        )
        return ToolResult(ok=True, tool_name="get_status", args=args, result=status, human_summary=summary)

    return handler


def _tool_get_decisions() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        memory = MLEngineMemory()
        runtime = memory.get_runtime_state("buddy_runtime")
        result = {
            "last_trade_execution_decision": runtime.get("last_trade_execution_decision") if isinstance(runtime, dict) else None,
            "last_no_trade_decision": runtime.get("last_no_trade_decision") if isinstance(runtime, dict) else None,
        }
        summary = "Loaded persisted Buddy runtime decision records."
        return ToolResult(ok=True, tool_name="get_decisions", args=args, result=result, human_summary=summary)

    return handler


def _tool_get_model_status(root_dir: Path) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        models_dir = root_dir / "trained_data" / "models"
        rows: list[dict[str, Any]] = []
        pair_count = 0
        validated_count = 0
        if models_dir.exists():
            for child in sorted(models_dir.iterdir()):
                if not child.is_dir():
                    continue
                keras_path = child / "transformer_direction.keras"
                meta_path = child / "transformer_direction.meta.pkl"
                if keras_path.exists() and meta_path.exists():
                    pair_count += 1
                    validated = (child / ".validation_result.json").exists()
                    if validated:
                        validated_count += 1
                    rows.append({"pair": child.name, "validated": validated})
        result = {
            "pair_model_count": pair_count,
            "validated_pair_model_count": validated_count,
            "rows": rows[:12],
        }
        summary = f"Model inventory: {pair_count} pair models, {validated_count} validated."
        return ToolResult(ok=True, tool_name="get_model_status", args=args, result=result, human_summary=summary)

    return handler


def _tool_get_journal_summary(root_dir: Path) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from src.utils.trade_journal import TradeJournal

        days = int(args.get("days") or 30)
        journal = TradeJournal(path=root_dir / "trained_data" / "trade_journal.json")
        stats = journal.get_statistics(days=days)
        recent = journal.get_recent_trades(n=3)
        result = {
            "days": days,
            "stats": stats,
            "recent_trades": [trade.__dict__ for trade in recent],
        }
        summary = f"Journal summary for {days}d: {stats.get('total_trades', 0)} trades, win rate {stats.get('win_rate', 0.0):.1%}."
        return ToolResult(ok=True, tool_name="get_journal_summary", args=args, result=result, human_summary=summary)

    return handler


def _tool_sync_journal(root_dir: Path) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from src.utils.trade_journal import TradeJournal
        from src.utils.oanda_practice import OandaPracticeClient

        journal = TradeJournal(path=root_dir / "trained_data" / "trade_journal.json")
        client = OandaPracticeClient.from_env()
        sync_results = journal.sync_open_trades(client)
        result = dict(sync_results)
        summary = (
            f"Journal sync finished: {sync_results.get('closed_updated', 0)} closed updated, "
            f"{sync_results.get('open_updated', 0)} open updated."
        )
        return ToolResult(ok=True, tool_name="sync_journal", args=args, result=result, human_summary=summary)

    return handler


def _tool_import_journal_trades(root_dir: Path) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from src.utils.trade_journal import TradeJournal
        from src.utils.oanda_practice import OandaPracticeClient

        journal = TradeJournal(path=root_dir / "trained_data" / "trade_journal.json")
        client = OandaPracticeClient.from_env()
        imported = int(journal.import_untracked_trades(client))
        result = {"imported_trades": imported}
        summary = f"Journal import finished: {imported} untracked trades imported."
        return ToolResult(ok=True, tool_name="import_journal_trades", args=args, result=result, human_summary=summary)

    return handler


def _tool_get_account_summary() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from src.utils.oanda_practice import OandaPracticeClient

        client = OandaPracticeClient.from_env()
        summary = client.get_account_summary()
        account = (summary or {}).get("account") or {}
        result = {
            "NAV": account.get("NAV") or account.get("nav"),
            "balance": account.get("balance"),
            "marginAvailable": account.get("marginAvailable"),
            "currency": account.get("currency"),
        }
        human = f"Account NAV={result['NAV']} balance={result['balance']} marginAvailable={result['marginAvailable']}."
        return ToolResult(ok=True, tool_name="get_account_summary", args=args, result=result, human_summary=human)

    return handler


def _summarize_scan_results(raw_results: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in raw_results or []:
        result = item[0] if isinstance(item, tuple) else item
        row = {
            "pair": str(getattr(result, "pair", None) or getattr(result, "instrument", None) or "unknown"),
            "direction": str(getattr(result, "direction", None) or "NONE"),
            "gates_passed": bool(getattr(result, "gates_passed", False)),
            "confidence": getattr(result, "ridge_confidence", None),
            "agent_total": getattr(result, "agent_total", None),
            "error": getattr(result, "error", None),
        }
        rows.append(row)
    tradable = [row for row in rows if row["direction"] not in {"NONE", "FLAT"} and row["error"] is None]
    approved = [row for row in tradable if row["gates_passed"]]
    return {
        "count": len(rows),
        "tradable_count": len(tradable),
        "approved_count": len(approved),
        "rows": rows[:8],
        "all_rows": rows,
    }


def _render_planner_scan_summary(summary: dict[str, Any], *, granularity: str) -> str:
    rows = list(summary.get("all_rows") or [])
    count = int(summary.get("count") or 0)
    tradable_count = int(summary.get("tradable_count") or 0)
    approved_count = int(summary.get("approved_count") or 0)
    session_blocked = [
        row for row in rows
        if str(row.get("error") or "").lower().startswith("outside trading session")
    ]

    if count and len(session_blocked) == count:
        first_reason = str(session_blocked[0].get("error") or "").strip()
        return (
            f"I checked {count} pairs on {granularity}. Nothing is tradeable right now because the session filter blocked every setup. "
            f"{first_reason}."
        )
    if tradable_count == 0:
        return f"I checked {count} pairs on {granularity}. Nothing is tradeable right now."
    if approved_count == 0:
        return (
            f"I checked {count} pairs on {granularity}. "
            f"I found {tradable_count} tradeable setups, but none cleared the full approval gates yet."
        )

    top_rows = [row for row in rows if row.get("error") is None][:2]
    if top_rows:
        pair_bits = ", ".join(
            f"{str(row.get('pair', 'unknown')).replace('_', '/')} {row.get('direction', 'HOLD')}"
            for row in top_rows
        )
        return (
            f"I checked {count} pairs on {granularity}. "
            f"I found {approved_count} approved setups and {tradable_count} tradeable ones. "
            f"Best read: {pair_bits}."
        )
    return (
        f"I checked {count} pairs on {granularity}. "
        f"I found {approved_count} approved setups and {tradable_count} tradeable ones."
    )


def _tool_scan_market(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from cli.buddy_scanning import buddy_scan

        pairs = _normalize_pairs_arg(args.get("pairs"))
        granularity = str(args.get("granularity") or state.get("granularity") or "H1")
        top_n = int(args.get("top_n") or 5)
        auto_execute = bool(args.get("auto_execute", False))
        force = bool(args.get("force", False))
        diversified = bool(args.get("diversified", False))
        raw_results = buddy_scan(
            config_path=config_path,
            pairs=pairs,
            granularity=granularity,
            top_n=top_n,
            force=force,
            diversified=diversified,
            auto_execute=auto_execute,
            no_execute=not auto_execute,
            clean_output=True,
        )
        summary = _summarize_scan_results(raw_results)
        state["last_scan_summary"] = summary
        human = _render_planner_scan_summary(summary, granularity=granularity)
        return ToolResult(ok=True, tool_name="scan_market", args=args, result=summary, human_summary=human)

    return handler


def _tool_compare_scan_results() -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        summary = state.get("last_scan_summary") or {}
        rows = summary.get("all_rows") or []
        left_pair = str(args.get("left_pair") or "").upper().replace("/", "_")
        right_pair = str(args.get("right_pair") or "").upper().replace("/", "_")
        left = next((row for row in rows if str(row.get("pair", "")).upper() == left_pair), None)
        right = next((row for row in rows if str(row.get("pair", "")).upper() == right_pair), None)
        if not left or not right:
            return ToolResult(
                ok=False,
                tool_name="compare_scan_results",
                args=args,
                error="Requested pairs not found in last scan summary.",
                human_summary="I couldn't find both requested pairs in the last scan results.",
            )
        comparison = {
            "left": left,
            "right": right,
            "winner": None,
        }
        if left.get("gates_passed") and not right.get("gates_passed"):
            comparison["winner"] = left["pair"]
            human = f"{left['pair']} is stronger because it passed the gates and {right['pair']} did not."
        elif right.get("gates_passed") and not left.get("gates_passed"):
            comparison["winner"] = right["pair"]
            human = f"{right['pair']} is stronger because it passed the gates and {left['pair']} did not."
        else:
            left_conf = float(left.get("confidence") or 0.0)
            right_conf = float(right.get("confidence") or 0.0)
            winner = left["pair"] if left_conf >= right_conf else right["pair"]
            comparison["winner"] = winner
            human = f"{winner} looks stronger on the stored scan metrics."
        return ToolResult(ok=True, tool_name="compare_scan_results", args=args, result=comparison, human_summary=human)

    return handler


def build_default_tool_registry(config_path: str, *, root_dir: str | Path | None = None) -> ToolRegistry:
    """Create the default Buddy planner tool registry."""
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="answer_buddy_self_question",
            description="Answer Buddy-about-Buddy questions from grounded repo/runtime state.",
            args=(ToolArgSpec("query", "string", "The Buddy self-knowledge question.", required=True),),
        ),
        _tool_answer_buddy_self_question(config_path),
    )
    registry.register(
        ToolSpec(
            name="use_market_source",
            description="Load CSV, ticker, or OANDA data as Buddy's active source.",
            args=(
                ToolArgSpec("source_kind", "string", "One of csv, ticker, or oanda.", required=True),
                ToolArgSpec("value", "string", "Path, ticker, or instrument for the source.", required=False),
                ToolArgSpec("granularity", "string", "Timeframe like M5 or H1 for OANDA.", required=False),
                ToolArgSpec("candles", "integer", "How many candles to load for OANDA.", required=False),
            ),
        ),
        _tool_use_market_source(),
    )
    registry.register(
        ToolSpec(
            name="run_prediction",
            description="Run Buddy prediction on the active source or on a specified source.",
            args=(
                ToolArgSpec("command", "string", "Optional raw predict command like 'predict' or 'predict ticker SPY'.", required=False),
                ToolArgSpec("source_kind", "string", "Optional source kind to load before predicting.", required=False),
                ToolArgSpec("value", "string", "Optional source value to load before predicting.", required=False),
                ToolArgSpec("granularity", "string", "Optional OANDA timeframe.", required=False),
                ToolArgSpec("candles", "integer", "Optional OANDA candle count.", required=False),
            ),
        ),
        _tool_run_prediction(),
    )
    registry.register(
        ToolSpec(
            name="summarize_last_prediction",
            description="Summarize or answer a question from Buddy's latest prediction context.",
            args=(ToolArgSpec("query", "string", "Optional summary request or follow-up question.", required=False),),
        ),
        _tool_summarize_last_prediction(),
    )
    registry.register(
        ToolSpec(
            name="run_runtime_command",
            description="Apply chat runtime settings like execute mode or SL/TP commands.",
            args=(ToolArgSpec("command", "string", "Raw runtime command such as 'execute on' or 'sl 20'.", required=True),),
        ),
        _tool_run_runtime_command(),
    )
    registry.register(
        ToolSpec(
            name="run_oanda_command",
            description="Run OANDA info commands such as account summary or live quote lookup.",
            args=(ToolArgSpec("command", "string", "Raw OANDA command like 'oanda account' or 'oanda quote EUR_USD'.", required=True),),
        ),
        _tool_run_oanda_command(),
    )
    registry.register(
        ToolSpec(
            name="run_trade_command",
            description="Run Buddy trade commands such as trade, buy, sell, or close.",
            args=(ToolArgSpec("command", "string", "Raw trade command to execute.", required=True),),
        ),
        _tool_run_trade_command(),
    )
    registry.register(ToolSpec(name="get_status", description="Read Buddy runtime status and settings."), _tool_get_status(config_path))
    registry.register(ToolSpec(name="get_decisions", description="Read persisted last trade and no-trade decisions."), _tool_get_decisions())
    registry.register(ToolSpec(name="get_model_status", description="Inspect model inventory and validation state."), _tool_get_model_status(root))
    registry.register(
        ToolSpec(
            name="get_journal_summary",
            description="Read trade journal statistics and recent trades.",
            args=(ToolArgSpec("days", "integer", "Number of days to summarize.", required=False),),
        ),
        _tool_get_journal_summary(root),
    )
    registry.register(ToolSpec(name="sync_journal", description="Sync the trade journal with OANDA."), _tool_sync_journal(root))
    registry.register(ToolSpec(name="import_journal_trades", description="Import untracked OANDA trades into the journal."), _tool_import_journal_trades(root))
    registry.register(ToolSpec(name="get_account_summary", description="Read the OANDA account summary."), _tool_get_account_summary())
    registry.register(
        ToolSpec(
            name="scan_market",
            description="Run a multi-pair scan with optional auto-execution.",
            args=(
                ToolArgSpec("pairs", "string|null", "Comma-separated pairs or null for defaults.", required=False),
                ToolArgSpec("granularity", "string", "Timeframe like M5 or H1.", required=False),
                ToolArgSpec("top_n", "integer", "How many top results to keep.", required=False),
                ToolArgSpec("auto_execute", "boolean", "Whether to auto-execute approved setups.", required=False),
                ToolArgSpec("force", "boolean", "Bypass session-hour filter.", required=False),
                ToolArgSpec("diversified", "boolean", "Filter correlated pairs.", required=False),
            ),
        ),
        _tool_scan_market(config_path),
    )
    registry.register(
        ToolSpec(
            name="compare_scan_results",
            description="Compare two pairs from the most recent scan result set.",
            args=(
                ToolArgSpec("left_pair", "string", "First pair to compare.", required=True),
                ToolArgSpec("right_pair", "string", "Second pair to compare.", required=True),
            ),
        ),
        _tool_compare_scan_results(),
    )
    return registry
