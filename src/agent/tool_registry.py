"""Tool registry for Buddy's planner runtime."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
import logging
from pathlib import Path
import re
import secrets
from typing import Any
import zlib

from memory_client import MLEngineMemory
from src.utils import load_config
from src.utils.buddy_knowledge import answer_buddy_question

from .tool_schema import ToolArgSpec, ToolHandler, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

_DEFAULT_MASTER_SIMULATION_PROFILES: dict[str, dict[str, float]] = {
    # Derived from the current AUD/NZD master profile:
    # 85.25% overall directional accuracy with effectively perfect shorts.
    "AUD_NZD": {
        "directional_accuracy": 0.8525,
        "short_accuracy": 1.0,
        "short_bias": 0.55,
    },
}


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


def _normalize_instrument(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text.replace("/", "_").replace("-", "_").replace(" ", "_")


def _scan_guardrail_defaults(config_path: str) -> tuple[bool, tuple[int, int]]:
    cfg = load_config(config_path) or {}
    scan_cfg = cfg.get("scan") if isinstance(cfg, dict) else {}
    if not isinstance(scan_cfg, dict):
        scan_cfg = {}

    enabled = bool(scan_cfg.get("enable_session_filter", True))
    start = int(scan_cfg.get("session_start_utc", 8))
    end = int(scan_cfg.get("session_end_utc", 21))
    if not (0 <= start < end <= 24):
        start, end = 8, 21
    return enabled, (start, end)


def _session_override_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    enabled_raw = payload.get("session_filter_enabled")
    enabled = bool(enabled_raw) if isinstance(enabled_raw, bool) else None

    window = None
    window_raw = payload.get("session_window_utc")
    if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
        try:
            start = int(window_raw[0])
            end = int(window_raw[1])
            if 0 <= start < end <= 24:
                window = [start, end]
        except Exception:
            window = None

    return {
        "guardrail_name": "session",
        "action": str(payload.get("action") or "").strip().lower(),
        "scope": str(payload.get("scope") or "").strip().lower(),
        "reason": str(payload.get("reason") or "").strip() or None,
        "session_filter_enabled": enabled,
        "session_window_utc": window,
    }


def _session_effective_state_from_context(
    state: dict[str, Any],
    *,
    config_path: str,
) -> tuple[bool, list[int]]:
    enabled, window = _scan_guardrail_defaults(config_path)
    session_store = state.get("guardrail_overrides_session")
    if isinstance(session_store, dict):
        session_override = _session_override_from_payload(session_store.get("session"))
        if session_override is not None:
            if isinstance(session_override.get("session_filter_enabled"), bool):
                enabled = bool(session_override["session_filter_enabled"])
            if isinstance(session_override.get("session_window_utc"), list):
                window = (int(session_override["session_window_utc"][0]), int(session_override["session_window_utc"][1]))
    return enabled, [int(window[0]), int(window[1])]


def _coerce_mapping_arg(value: Any, *, arg_name: str) -> dict[str, Any]:
    if value in (None, "", {}, [], ()):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        candidate = Path(text)
        if candidate.exists():
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        else:
            loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{arg_name} must resolve to a JSON object.")
        return dict(loaded)
    raise ValueError(f"{arg_name} must be an object, JSON string, or JSON file path.")


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_simulation_profile(
    normalized_pair: str,
    *,
    args: dict[str, Any],
    simulation_profiles: dict[str, Any],
) -> dict[str, float]:
    profile: dict[str, Any] = dict(_DEFAULT_MASTER_SIMULATION_PROFILES.get(normalized_pair, {}))
    pair_profile = simulation_profiles.get(normalized_pair)
    if pair_profile is None:
        pair_profile = simulation_profiles.get(normalized_pair.replace("_", "/"))
    if pair_profile is not None:
        if not isinstance(pair_profile, dict):
            raise ValueError(f"simulation_profiles[{normalized_pair!r}] must be an object.")
        profile.update(pair_profile)

    overall = _clamp_probability(profile.get("directional_accuracy", args.get("directional_accuracy") or 0.708))
    short_bias = _clamp_probability(profile.get("short_bias", args.get("short_bias") or 0.55))
    short_accuracy_raw = profile.get("short_accuracy", args.get("short_accuracy"))
    long_accuracy_raw = profile.get("long_accuracy", args.get("long_accuracy"))

    short_accuracy = None if short_accuracy_raw is None else _clamp_probability(short_accuracy_raw)
    long_accuracy = None if long_accuracy_raw is None else _clamp_probability(long_accuracy_raw)

    if short_accuracy is None and long_accuracy is None:
        short_accuracy = 0.726
        long_accuracy = 0.593
    elif short_accuracy is None and long_accuracy is not None:
        denom = max(short_bias, 1e-6)
        short_accuracy = _clamp_probability((overall - (1.0 - short_bias) * long_accuracy) / denom)
    elif long_accuracy is None and short_accuracy is not None:
        denom = max(1.0 - short_bias, 1e-6)
        long_accuracy = _clamp_probability((overall - short_bias * short_accuracy) / denom)

    return {
        "directional_accuracy": overall,
        "short_accuracy": float(short_accuracy),
        "long_accuracy": float(long_accuracy),
        "short_bias": short_bias,
    }


def _resolve_backtest_path(
    normalized_pair: str,
    *,
    root_dir: Path,
    default_path: Path,
    backtest_paths: dict[str, Any],
    backtest_dir: str | None,
) -> Path:
    configured = backtest_paths.get(normalized_pair)
    if configured is None:
        configured = backtest_paths.get(normalized_pair.replace("_", "/"))
    if configured is not None:
        return Path(str(configured))

    candidate_dirs = []
    if backtest_dir:
        candidate_dirs.append(Path(str(backtest_dir)))
    candidate_dirs.append(root_dir / "trained_data" / "backtests")

    candidate_names = (
        f"{normalized_pair}.json",
        f"{normalized_pair.lower()}.json",
        f"{normalized_pair}_backtest_results.json",
        f"{normalized_pair.lower()}_backtest_results.json",
    )

    for directory in candidate_dirs:
        for name in candidate_names:
            candidate = directory / name
            if candidate.exists():
                return candidate
        for nested in ("results.json", "backtest_results.json"):
            candidate = directory / normalized_pair / nested
            if candidate.exists():
                return candidate

    return default_path


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
        session_enabled_effective, session_window_effective = _session_effective_state_from_context(
            state,
            config_path=config_path,
        )
        session_overrides = state.get("guardrail_overrides_session")
        if not isinstance(session_overrides, dict):
            session_overrides = {}
        one_shot_overrides = state.get("guardrail_overrides_once")
        if not isinstance(one_shot_overrides, dict):
            one_shot_overrides = {}
        pending_override = state.get("pending_guardrail_override")
        if not isinstance(pending_override, dict):
            pending_override = None

        status = {
            "active_instrument": state.get("active_instrument"),
            "granularity": state.get("granularity"),
            "execute_mode": state.get("execute_mode"),
            "stop_loss_pips": state.get("stop_loss_pips"),
            "take_profit_pips": state.get("take_profit_pips"),
            "configured_risk_per_trade_pct": (cfg.get("fx", {}) or {}).get("risk", {}).get("risk_per_trade_pct"),
            "last_trade_execution_decision": runtime.get("last_trade_execution_decision") if isinstance(runtime, dict) else None,
            "last_no_trade_decision": runtime.get("last_no_trade_decision") if isinstance(runtime, dict) else None,
            "guardrail_overrides_session": session_overrides,
            "guardrail_overrides_once": one_shot_overrides,
            "pending_guardrail_override": pending_override,
            "session_filter_effective": session_enabled_effective,
            "session_window_effective": session_window_effective,
        }
        override_bits: list[str] = []
        if isinstance(session_overrides.get("session"), dict):
            override_bits.append("session override active")
        if isinstance(one_shot_overrides.get("session"), dict):
            override_bits.append("one-shot session override pending")
        if isinstance(pending_override, dict):
            override_bits.append("permanent session override pending confirmation")
        override_suffix = f" Guardrails: {', '.join(override_bits)}." if override_bits else ""
        summary = (
            f"You're on {status['active_instrument']} {status['granularity']}. "
            f"Execution is {status['execute_mode']} and configured risk is {status['configured_risk_per_trade_pct']} per trade."
            + override_suffix
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


def _is_market_closed_error(error: Any) -> bool:
    return str(error or "").strip().lower().startswith("fx market closed")


def _is_session_block_error(error: Any) -> bool:
    return str(error or "").strip().lower().startswith("outside trading session")


def _render_planner_scan_summary(summary: dict[str, Any], *, granularity: str) -> str:
    rows = list(summary.get("all_rows") or [])
    count = int(summary.get("count") or 0)
    tradable_count = int(summary.get("tradable_count") or 0)
    approved_count = int(summary.get("approved_count") or 0)
    market_closed = [
        row for row in rows
        if _is_market_closed_error(row.get("error"))
    ]
    session_blocked = [
        row for row in rows
        if _is_session_block_error(row.get("error"))
    ]

    if count and len(market_closed) == count:
        first_reason = str(market_closed[0].get("error") or "").strip()
        return (
            f"I checked {count} pairs on {granularity}. Nothing is tradeable right now because the FX market is closed. "
            f"{first_reason}."
        )

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
            f"I found {tradable_count} directional setups, but none cleared the full approval gates yet."
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


def _tool_override_guardrail(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import yaml

        guardrail_name = str(args.get("guardrail_name") or "").strip().lower()
        action = str(args.get("action") or "").strip().lower()
        scope = str(args.get("scope") or "this_scan").strip().lower()
        reason = str(args.get("reason") or "").strip() or None
        confirm_token = str(args.get("confirm_token") or "").strip()

        if guardrail_name != "session":
            raise ValueError(
                "override_guardrail currently supports guardrail_name='session' only."
            )

        if confirm_token:
            pending = state.get("pending_guardrail_override")
            if not isinstance(pending, dict):
                raise ValueError("There is no pending permanent guardrail override to confirm or cancel.")

            if confirm_token.lower() in {"cancel", "abort", "decline", "no"}:
                state.pop("pending_guardrail_override", None)
                return ToolResult(
                    ok=True,
                    tool_name="override_guardrail",
                    args=args,
                    result={"cancelled": True},
                    human_summary="Cancelled the pending permanent session override.",
                )

            expected_token = str(pending.get("confirmation_token") or "")
            if confirm_token != expected_token:
                raise ValueError(
                    "Confirmation token mismatch. Use the exact token from the pending override."
                )

            config_file = Path(config_path)
            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"Invalid YAML config at {config_path}: expected a mapping root.")
            scan_cfg = cfg.setdefault("scan", {})
            if not isinstance(scan_cfg, dict):
                raise ValueError(f"Invalid scan section in {config_path}: expected a mapping.")

            enabled = pending.get("session_filter_enabled")
            if isinstance(enabled, bool):
                scan_cfg["enable_session_filter"] = enabled
            window = pending.get("session_window_utc")
            if isinstance(window, (list, tuple)) and len(window) == 2:
                start = int(window[0])
                end = int(window[1])
                if not (0 <= start < end <= 24):
                    raise ValueError("Pending session window is invalid; expected 0 <= start < end <= 24.")
                scan_cfg["session_start_utc"] = start
                scan_cfg["session_end_utc"] = end

            config_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            state.pop("pending_guardrail_override", None)
            result = {
                "applied_permanent": True,
                "guardrail_name": "session",
                "session_filter_enabled": scan_cfg.get("enable_session_filter"),
                "session_window_utc": [
                    int(scan_cfg.get("session_start_utc", 8)),
                    int(scan_cfg.get("session_end_utc", 21)),
                ],
                "config_path": str(config_file),
            }
            return ToolResult(
                ok=True,
                tool_name="override_guardrail",
                args=args,
                result=result,
                human_summary=(
                    "Permanent session override applied to config: "
                    f"enabled={result['session_filter_enabled']}, "
                    f"window={result['session_window_utc'][0]}-{result['session_window_utc'][1]} UTC."
                ),
            )

        if action not in {"disable", "enable", "toggle", "set_window_utc"}:
            raise ValueError(
                "override_guardrail action must be one of: disable, enable, toggle, set_window_utc."
            )
        if scope not in {"this_scan", "this_session", "permanent"}:
            raise ValueError(
                "override_guardrail scope must be one of: this_scan, this_session, permanent."
            )

        effective_enabled, effective_window = _session_effective_state_from_context(
            state,
            config_path=config_path,
        )
        target_enabled: bool | None = None
        target_window: list[int] | None = None

        if action == "disable":
            target_enabled = False
        elif action == "enable":
            target_enabled = True
        elif action == "toggle":
            target_enabled = not effective_enabled
        elif action == "set_window_utc":
            if args.get("window_start_utc") is None or args.get("window_end_utc") is None:
                raise ValueError(
                    "set_window_utc requires both window_start_utc and window_end_utc."
                )
            start = int(args.get("window_start_utc"))
            end = int(args.get("window_end_utc"))
            if not (0 <= start < end <= 24):
                raise ValueError("Invalid UTC session window; expected 0 <= start < end <= 24.")
            target_enabled = effective_enabled
            target_window = [start, end]

        payload = {
            "guardrail_name": "session",
            "action": action,
            "scope": scope,
            "reason": reason,
            "session_filter_enabled": target_enabled,
            "session_window_utc": target_window,
        }

        if scope == "this_scan":
            one_shot = state.get("guardrail_overrides_once")
            if not isinstance(one_shot, dict):
                one_shot = {}
            one_shot["session"] = payload
            state["guardrail_overrides_once"] = one_shot
            return ToolResult(
                ok=True,
                tool_name="override_guardrail",
                args=args,
                result={"applied_scope": "this_scan", "override": payload},
                human_summary=(
                    "Applied one-shot session override for the next scan: "
                    f"action={action}."
                ),
            )

        if scope == "this_session":
            session_store = state.get("guardrail_overrides_session")
            if not isinstance(session_store, dict):
                session_store = {}
            session_store["session"] = payload
            state["guardrail_overrides_session"] = session_store
            return ToolResult(
                ok=True,
                tool_name="override_guardrail",
                args=args,
                result={"applied_scope": "this_session", "override": payload},
                human_summary=(
                    "Applied session-scoped guardrail override: "
                    f"action={action}."
                ),
            )

        token = secrets.token_hex(4)
        pending = dict(payload)
        pending["confirmation_token"] = token
        state["pending_guardrail_override"] = pending
        return ToolResult(
            ok=True,
            tool_name="override_guardrail",
            args=args,
            result={
                "pending_confirmation": True,
                "confirmation_token": token,
                "pending_override": pending,
            },
            human_summary=(
                "Permanent override requires confirmation. "
                f"Run a confirmation request with token '{token}', or cancel."
            ),
        )

    return handler


def _tool_scan_market(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from cli.buddy_scanning import buddy_scan

        pairs = _normalize_pairs_arg(args.get("pairs"))
        granularity = str(args.get("granularity") or "H1")
        top_n = int(args.get("top_n") or 5)
        auto_execute = bool(args.get("auto_execute", False))
        force_raw = args.get("force", None)
        force = force_raw is True or (
            isinstance(force_raw, str)
            and force_raw.strip().lower() in {"1", "true", "yes", "on"}
        )
        diversified = bool(args.get("diversified", False))
        run_agent_confirmation = bool(args.get("run_agent_confirmation", False))

        one_shot_store = state.get("guardrail_overrides_once")
        if not isinstance(one_shot_store, dict):
            one_shot_store = {}
        session_store = state.get("guardrail_overrides_session")
        if not isinstance(session_store, dict):
            session_store = {}

        one_shot_session = _session_override_from_payload(one_shot_store.get("session"))
        session_override = _session_override_from_payload(session_store.get("session"))

        session_filter_enabled_override: bool | None = None
        session_window_utc_override: tuple[int, int] | None = None
        applied_guardrail_overrides: list[dict[str, Any]] = []

        if force:
            session_filter_enabled_override = False
            applied_guardrail_overrides.append(
                {
                    "source": "scan_arg_force",
                    "guardrail_name": "session",
                    "session_filter_enabled": False,
                }
            )
        else:
            selected_override = one_shot_session or session_override
            if selected_override is not None:
                enabled_value = selected_override.get("session_filter_enabled")
                if isinstance(enabled_value, bool):
                    session_filter_enabled_override = enabled_value
                window_value = selected_override.get("session_window_utc")
                if isinstance(window_value, list):
                    session_window_utc_override = (int(window_value[0]), int(window_value[1]))
                applied_guardrail_overrides.append(
                    {
                        "source": "this_scan" if selected_override is one_shot_session else "this_session",
                        "guardrail_name": "session",
                        "action": selected_override.get("action"),
                        "session_filter_enabled": session_filter_enabled_override,
                        "session_window_utc": list(session_window_utc_override) if session_window_utc_override else None,
                    }
                )

        force_effective = force or (session_filter_enabled_override is False)

        try:
            raw_results = buddy_scan(
                config_path=config_path,
                pairs=pairs,
                granularity=granularity,
                top_n=top_n,
                force=force_effective,
                diversified=diversified,
                auto_execute=auto_execute,
                no_execute=not auto_execute,
                clean_output=True,
                run_agent_confirmation=run_agent_confirmation,
                session_filter_enabled_override=session_filter_enabled_override,
                session_window_utc_override=session_window_utc_override,
            )
        finally:
            if one_shot_session is not None:
                refreshed_once = state.get("guardrail_overrides_once")
                if isinstance(refreshed_once, dict):
                    refreshed_once.pop("session", None)
                    if refreshed_once:
                        state["guardrail_overrides_once"] = refreshed_once
                    else:
                        state.pop("guardrail_overrides_once", None)

        summary = _summarize_scan_results(raw_results)
        summary["granularity"] = granularity
        config_session_enabled, config_session_window = _scan_guardrail_defaults(config_path)
        effective_session_enabled = config_session_enabled
        effective_window = [int(config_session_window[0]), int(config_session_window[1])]
        if session_window_utc_override is not None:
            effective_window = [int(session_window_utc_override[0]), int(session_window_utc_override[1])]
        if session_filter_enabled_override is False or force_effective:
            effective_session_enabled = False
        elif session_filter_enabled_override is True:
            effective_session_enabled = True

        summary["applied_guardrail_overrides"] = applied_guardrail_overrides
        summary["session_filter_effective"] = effective_session_enabled
        summary["session_window_effective"] = effective_window
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


def _tool_configure_pair_sentiment_gate(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import yaml

        pair = _normalize_instrument(args.get("pair"))
        if not pair:
            raise ValueError("configure_pair_sentiment_gate requires a pair.")

        enabled = bool(args.get("enabled", True))
        directions_arg = args.get("directions")
        if isinstance(directions_arg, str):
            raw_directions = [part for part in re.split(r"[\s,]+", directions_arg) if part]
        elif isinstance(directions_arg, (list, tuple)):
            raw_directions = [str(part) for part in directions_arg if str(part).strip()]
        else:
            raw_directions = []

        normalized_directions: list[str] = []
        for raw in raw_directions:
            key = str(raw).strip().lower()
            if key in {"long", "buy"} and "long" not in normalized_directions:
                normalized_directions.append("long")
            elif key in {"short", "sell"} and "short" not in normalized_directions:
                normalized_directions.append("short")

        path = Path(config_path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config in {config_path}: expected a mapping at the root.")

        inference = cfg.setdefault("inference", {})
        if not isinstance(inference, dict):
            raise ValueError(f"Invalid inference section in {config_path}: expected a mapping.")

        by_pair = inference.setdefault("by_pair", {})
        if not isinstance(by_pair, dict):
            raise ValueError(f"Invalid inference.by_pair section in {config_path}: expected a mapping.")

        pair_cfg = by_pair.get(pair)
        if pair_cfg is None:
            pair_cfg = {}
        if not isinstance(pair_cfg, dict):
            raise ValueError(f"Invalid inference.by_pair.{pair} section in {config_path}: expected a mapping.")

        pair_cfg["sentiment_block_enabled"] = enabled
        if normalized_directions:
            pair_cfg["sentiment_block_directions"] = normalized_directions
        elif enabled:
            pair_cfg.pop("sentiment_block_directions", None)

        if "threshold" in args and args.get("threshold") is not None:
            pair_cfg["sentiment_block_threshold"] = float(args["threshold"])
        if "min_headlines" in args and args.get("min_headlines") is not None:
            pair_cfg["sentiment_min_headlines"] = int(args["min_headlines"])

        by_pair[pair] = pair_cfg
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        state["last_config_change"] = {
            "tool": "configure_pair_sentiment_gate",
            "pair": pair,
            "enabled": enabled,
            "directions": normalized_directions,
            "config_path": str(path),
        }
        scope = "/".join(normalized_directions) if normalized_directions else "all directions"
        action = "enabled" if enabled else "disabled"
        summary = f"Sentiment confirmation {action} for {pair.replace('_', '/')} on {scope} in {path.name}."
        return ToolResult(
            ok=True,
            tool_name="configure_pair_sentiment_gate",
            args=args,
            result={
                "pair": pair,
                "enabled": enabled,
                "directions": normalized_directions,
                "config_path": str(path),
                "pair_config": pair_cfg,
            },
            human_summary=summary,
        )

    return handler


def _tool_configure_correlation_sentiment_gate(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import yaml

        from src.training.correlation_group_config import get_correlated_direction_map

        master_pair = _normalize_instrument(args.get("master_pair"))
        direction = str(args.get("direction") or "").strip().lower()
        if not master_pair:
            raise ValueError("configure_correlation_sentiment_gate requires a master_pair.")
        if direction not in {"long", "short"}:
            raise ValueError("configure_correlation_sentiment_gate requires direction='long' or 'short'.")

        enabled = bool(args.get("enabled", True))
        correlated_map = get_correlated_direction_map(master_pair, direction)
        path = Path(config_path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config in {config_path}: expected a mapping at the root.")

        inference = cfg.setdefault("inference", {})
        if not isinstance(inference, dict):
            raise ValueError(f"Invalid inference section in {config_path}: expected a mapping.")

        by_pair = inference.setdefault("by_pair", {})
        if not isinstance(by_pair, dict):
            raise ValueError(f"Invalid inference.by_pair section in {config_path}: expected a mapping.")

        updated_pairs: list[dict[str, Any]] = []
        for pair, pair_direction in correlated_map.items():
            pair_cfg = by_pair.get(pair) or {}
            if not isinstance(pair_cfg, dict):
                raise ValueError(f"Invalid inference.by_pair.{pair} section in {config_path}: expected a mapping.")
            pair_cfg["sentiment_block_enabled"] = enabled
            pair_cfg["sentiment_block_directions"] = [pair_direction]
            if "threshold" in args and args.get("threshold") is not None:
                pair_cfg["sentiment_block_threshold"] = float(args["threshold"])
            if "min_headlines" in args and args.get("min_headlines") is not None:
                pair_cfg["sentiment_min_headlines"] = int(args["min_headlines"])
            by_pair[pair] = pair_cfg
            updated_pairs.append({"pair": pair, "direction": pair_direction})

        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        state["last_config_change"] = {
            "tool": "configure_correlation_sentiment_gate",
            "master_pair": master_pair,
            "direction": direction,
            "updated_pairs": updated_pairs,
            "config_path": str(path),
        }
        pair_summary = ", ".join(f"{item['pair'].replace('_', '/')} {item['direction']}" for item in updated_pairs[:6])
        summary = (
            f"Correlation sentiment confirmation updated from {master_pair.replace('_', '/')} {direction}: "
            f"{pair_summary}."
        )
        return ToolResult(
            ok=True,
            tool_name="configure_correlation_sentiment_gate",
            args=args,
            result={
                "master_pair": master_pair,
                "direction": direction,
                "updated_pairs": updated_pairs,
                "config_path": str(path),
            },
            human_summary=summary,
        )

    return handler


def _tool_retrain_meta_labelers(root_dir: Path) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        from src.training.correlation_group_config import get_unique_master_pairs
        from src.training.meta_labeler_retrainer import MetaLabelerRetrainer, MetaLabelerTrainingConfig

        data_source = str(args.get("data_source") or "journal").strip().lower()
        master_pairs_arg = args.get("master_pairs")
        if isinstance(master_pairs_arg, str):
            requested_pairs = [_normalize_instrument(part) for part in re.split(r"[\s,]+", master_pairs_arg) if part.strip()]
        elif isinstance(master_pairs_arg, (list, tuple)):
            requested_pairs = [_normalize_instrument(part) for part in master_pairs_arg if str(part).strip()]
        else:
            requested_pairs = []
        master_pairs = requested_pairs or get_unique_master_pairs()

        min_samples = int(args.get("min_samples") or 100)
        simulate_trades = int(args.get("simulate_trades") or 1000)
        simulation_profiles = _coerce_mapping_arg(args.get("simulation_profiles"), arg_name="simulation_profiles")
        backtest_paths = _coerce_mapping_arg(args.get("backtest_paths"), arg_name="backtest_paths")
        backtest_dir = str(args.get("backtest_dir")).strip() if args.get("backtest_dir") else None
        journal_path = Path(args.get("journal_path") or (root_dir / "trained_data" / "trade_journal.json"))
        backtest_path = Path(args.get("backtest_path") or (root_dir / "trained_data" / "basket_backtest_results.json"))

        pair_results: list[dict[str, Any]] = []
        for pair in master_pairs:
            normalized_pair = pair.upper().replace("/", "_")
            output_dir = root_dir / "trained_data" / "models" / pair
            config = MetaLabelerTrainingConfig(min_samples=min_samples)
            retrainer = MetaLabelerRetrainer(config=config, output_dir=output_dir)

            if data_source == "journal":
                samples = retrainer.load_trade_journal(journal_path)
            elif data_source == "backtest":
                pair_backtest_path = _resolve_backtest_path(
                    normalized_pair,
                    root_dir=root_dir,
                    default_path=backtest_path,
                    backtest_paths=backtest_paths,
                    backtest_dir=backtest_dir,
                )
                samples = retrainer.load_backtest_results(pair_backtest_path)
            elif data_source == "simulation":
                base_seed = args.get("seed")
                if base_seed is None:
                    simulation_seed = zlib.crc32(normalized_pair.encode("utf-8")) & 0xFFFFFFFF
                else:
                    simulation_seed = int(base_seed) + len(pair_results)
                simulation_profile = _resolve_simulation_profile(
                    normalized_pair,
                    args=args,
                    simulation_profiles=simulation_profiles,
                )
                samples = retrainer.simulate_trades_from_model(
                    n_trades=simulate_trades,
                    directional_accuracy=simulation_profile["directional_accuracy"],
                    short_accuracy=simulation_profile["short_accuracy"],
                    long_accuracy=simulation_profile["long_accuracy"],
                    short_bias=simulation_profile["short_bias"],
                    seed=simulation_seed,
                )
                for sample in samples:
                    sample.instrument = pair
            else:
                raise ValueError("Unsupported data_source. Use journal, backtest, or simulation.")
            pair_samples = [
                sample for sample in samples
                if _normalize_instrument(getattr(sample, "instrument", "")) == normalized_pair
            ]
            if len(pair_samples) < min_samples:
                pair_results.append(
                    {
                        "pair": normalized_pair,
                        "status": "skipped",
                        "sample_count": len(pair_samples),
                        "reason": f"insufficient_samples<{min_samples}",
                    }
                )
                continue

            metrics = retrainer.fit(pair_samples, verbose=False)
            model_path = output_dir / "meta_labeler.pkl"
            retrainer.save(model_path)
            pair_results.append(
                {
                    "pair": normalized_pair,
                    "status": "trained",
                    "sample_count": len(pair_samples),
                    "model_path": str(model_path),
                    "val_accuracy": metrics.get("ensemble_val_accuracy"),
                    "val_auc": metrics.get("ensemble_val_auc"),
                    "data_path": str(pair_backtest_path) if data_source == "backtest" else None,
                    "simulation_profile": simulation_profile if data_source == "simulation" else None,
                }
            )

        trained = [row for row in pair_results if row["status"] == "trained"]
        skipped = [row for row in pair_results if row["status"] != "trained"]
        state["last_meta_retrain"] = {
            "data_source": data_source,
            "results": pair_results,
        }
        summary = (
            f"Meta-labeler retraining finished for {len(trained)} masters"
            f" ({', '.join(row['pair'] for row in trained[:5]) or 'none'})."
        )
        if skipped:
            summary += f" Skipped {len(skipped)} due to insufficient data or config."
        return ToolResult(
            ok=True,
            tool_name="retrain_meta_labelers",
            args=args,
            result={
                "data_source": data_source,
                "trained_count": len(trained),
                "skipped_count": len(skipped),
                "pairs": pair_results,
            },
            human_summary=summary,
        )

    return handler


def _compute_quick_backtest_series(df: Any, direction: str, window: int) -> Any:
    import pandas as pd

    if "close" not in df.columns:
        raise ValueError("Backtest data is missing a close column.")
    if len(df) < window + 10:
        raise ValueError(f"Insufficient data ({len(df)} < {window + 10})")

    df_backtest = df.tail(window + 10).copy()
    df_backtest["returns"] = df_backtest["close"].pct_change()
    if str(direction).upper() == "LONG":
        df_backtest["signal_returns"] = df_backtest["returns"]
    else:
        df_backtest["signal_returns"] = -df_backtest["returns"]
    signal_returns = df_backtest["signal_returns"].dropna().tail(window)
    if len(signal_returns) == 0:
        raise ValueError("No backtest returns available.")
    return pd.Series(signal_returns.to_numpy(dtype=float), index=range(len(signal_returns)))


def _tool_run_basket_backtest(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import numpy as np

        from src.scanner.config import ScannerConfig
        from src.scanner.engine import Scanner
        from src.training.correlation_group_config import get_pairs_for_baskets

        basket_arg = args.get("baskets")
        if isinstance(basket_arg, str):
            basket_names = [part.strip() for part in re.split(r"[,;]+", basket_arg) if part.strip()]
        elif isinstance(basket_arg, (list, tuple)):
            basket_names = [str(part).strip() for part in basket_arg if str(part).strip()]
        else:
            basket_names = []

        pairs_arg = args.get("pairs")
        if isinstance(pairs_arg, str):
            explicit_pairs = [_normalize_instrument(part) for part in re.split(r"[\s,]+", pairs_arg) if part.strip()]
        elif isinstance(pairs_arg, (list, tuple)):
            explicit_pairs = [_normalize_instrument(part) for part in pairs_arg if str(part).strip()]
        else:
            explicit_pairs = []

        basket_pairs = get_pairs_for_baskets(basket_names)
        pair_list = explicit_pairs or basket_pairs
        if not pair_list:
            raise ValueError("run_basket_backtest requires pairs or basket names.")

        granularity = str(args.get("granularity") or state.get("granularity") or "H1")
        profile = str(args.get("profile") or "balanced")
        approved_only = bool(args.get("approved_only", True))
        force = bool(args.get("force", True))
        window = int(args.get("window") or 50)

        scan_config = ScannerConfig.from_cli_args(
            config_path=config_path,
            pairs=pair_list,
            granularity=granularity,
            profile=profile,
            non_interactive=True,
            force=force,
        )
        scanner = Scanner(config=scan_config)
        scan_result = scanner.scan_with_analysis(
            pairs=pair_list,
            run_backtest=True,
            run_correlation=False,
            run_drift_check=False,
            apply_diversification=False,
        )

        eligible = [
            analysis for analysis in scan_result.analyses
            if analysis.error is None
            and analysis.direction in {"LONG", "SHORT"}
            and (analysis.gates_passed if approved_only else True)
        ]
        if not eligible:
            summary = "Basket backtest found no eligible directional setups from the current scan."
            return ToolResult(
                ok=True,
                tool_name="run_basket_backtest",
                args=args,
                result={
                    "granularity": granularity,
                    "pairs_requested": pair_list,
                    "eligible_pairs": [],
                    "equity_curve": [],
                },
                human_summary=summary,
            )

        pair_backtests: list[dict[str, Any]] = []
        curve_inputs: list[Any] = []
        for analysis in eligible:
            df = scanner._fetch_pair_data(analysis.pair, scan_config.lookback_candles + 200)
            if df is None:
                continue
            series = _compute_quick_backtest_series(df, analysis.direction, window)
            bt_result = scanner._backtester.run(df, analysis.direction, window=window) if scanner._backtester else None
            curve_inputs.append(series)
            pair_backtests.append(
                {
                    "pair": analysis.pair,
                    "direction": analysis.direction,
                    "confidence": analysis.confidence,
                    "gates_passed": analysis.gates_passed,
                    "total_pnl": getattr(bt_result, "total_pnl", None),
                    "win_rate": getattr(bt_result, "win_rate", None),
                    "sharpe": getattr(bt_result, "sharpe", None),
                }
            )

        if not curve_inputs:
            raise ValueError("Basket backtest could not build return series for the selected pairs.")

        min_len = min(len(series) for series in curve_inputs)
        curve_matrix = np.column_stack([series.tail(min_len).to_numpy(dtype=float) for series in curve_inputs])
        basket_returns = curve_matrix.mean(axis=1)
        equity_curve = np.cumprod(1.0 + basket_returns)
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) else 0.0
        total_return = float(equity_curve[-1] - 1.0) if len(equity_curve) else 0.0
        win_rate = float(np.mean(basket_returns > 0)) if len(basket_returns) else 0.0

        result = {
            "granularity": granularity,
            "profile": profile,
            "baskets": basket_names,
            "pairs_requested": pair_list,
            "eligible_pairs": [row["pair"] for row in pair_backtests],
            "pair_backtests": pair_backtests,
            "bars": int(min_len),
            "basket_total_return": total_return,
            "basket_win_rate": win_rate,
            "basket_max_drawdown": max_drawdown,
            "equity_curve": [float(value) for value in equity_curve.tolist()],
        }
        state["last_backtest_summary"] = result
        summary = (
            f"Basket backtest on {len(pair_backtests)} pairs over {min_len} bars returned "
            f"{total_return:+.2%} with max drawdown {max_drawdown:.2%}. "
            f"Included: {', '.join(row['pair'].replace('_', '/') for row in pair_backtests[:6])}."
        )
        return ToolResult(
            ok=True,
            tool_name="run_basket_backtest",
            args=args,
            result=result,
            human_summary=summary,
        )

    return handler


def _tool_configure_aggressive_regime_sizing(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import yaml

        enabled = bool(args.get("enabled", True))
        high_scale = float(args.get("high_vol_scale") or 1.5)
        extreme_scale = float(args.get("extreme_vol_scale") or 1.75)
        min_meta_confidence = float(args.get("min_meta_confidence") or 0.52)

        path = Path(config_path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config in {config_path}: expected a mapping at the root.")

        scan_cfg = cfg.setdefault("scan", {})
        if not isinstance(scan_cfg, dict):
            raise ValueError(f"Invalid scan section in {config_path}: expected a mapping.")

        scan_cfg["position_sizing_enabled"] = True
        scan_cfg["aggressive_mode"] = enabled
        scan_cfg["regime_scaling_enabled"] = enabled
        scan_cfg["aggressive_scale_high_vol"] = high_scale
        scan_cfg["aggressive_scale_extreme_vol"] = extreme_scale
        scan_cfg["aggressive_min_meta_confidence"] = min_meta_confidence

        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        state["last_config_change"] = {
            "tool": "configure_aggressive_regime_sizing",
            "enabled": enabled,
            "high_vol_scale": high_scale,
            "extreme_vol_scale": extreme_scale,
            "min_meta_confidence": min_meta_confidence,
            "config_path": str(path),
        }
        status = "enabled" if enabled else "disabled"
        summary = (
            f"Aggressive volatility-regime sizing {status} "
            f"(HIGH={high_scale:.2f}x, EXTREME={extreme_scale:.2f}x, meta>={min_meta_confidence:.2f})."
        )
        return ToolResult(
            ok=True,
            tool_name="configure_aggressive_regime_sizing",
            args=args,
            result={
                "enabled": enabled,
                "high_vol_scale": high_scale,
                "extreme_vol_scale": extreme_scale,
                "min_meta_confidence": min_meta_confidence,
                "config_path": str(path),
            },
            human_summary=summary,
        )

    return handler


def _tool_configure_scan_profile(config_path: str) -> ToolHandler:
    def handler(args: dict[str, Any], state: dict[str, Any]) -> ToolResult:
        import yaml

        from src.scanner.config import VALID_SCAN_PROFILES

        profile = str(args.get("profile") or "balanced").strip().lower()
        if profile not in VALID_SCAN_PROFILES:
            valid = ", ".join(VALID_SCAN_PROFILES)
            raise ValueError(f"Unknown scan profile '{profile}'. Valid profiles: {valid}")

        path = Path(config_path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config in {config_path}: expected a mapping at the root.")

        scan_cfg = cfg.setdefault("scan", {})
        if not isinstance(scan_cfg, dict):
            raise ValueError(f"Invalid scan section in {config_path}: expected a mapping.")

        scan_cfg["profile"] = profile

        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        state["profile"] = profile
        state["last_config_change"] = {
            "tool": "configure_scan_profile",
            "profile": profile,
            "config_path": str(path),
        }

        if profile == "aggressive":
            summary = (
                "Scanner profile set to aggressive. "
                "That loosens the approval gates to surface more directional setups."
            )
        elif profile == "conservative":
            summary = (
                "Scanner profile set to conservative. "
                "That tightens the approval gates for fewer, higher-quality setups."
            )
        else:
            summary = (
                "Scanner profile reset to balanced. "
                "That restores the default approval gate thresholds."
            )

        return ToolResult(
            ok=True,
            tool_name="configure_scan_profile",
            args=args,
            result={"profile": profile, "config_path": str(path)},
            human_summary=summary,
        )

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
            name="override_guardrail",
            description=(
                "Temporarily or permanently override a planner guardrail. "
                "V1 supports guardrail_name='session'. "
                "Use this single tool for all session-guardrail natural-language requests "
                "such as 'ignore session filter', 'restore session filter', and "
                "'set session window 9-17 utc'."
            ),
            args=(
                ToolArgSpec("guardrail_name", "string", "Guardrail to override (session in v1).", required=True),
                ToolArgSpec("action", "string", "disable, enable, toggle, or set_window_utc.", required=True),
                ToolArgSpec("scope", "string", "this_scan, this_session, or permanent.", required=False),
                ToolArgSpec("reason", "string", "Optional reason for auditability.", required=False),
                ToolArgSpec("window_start_utc", "integer", "Required when action=set_window_utc.", required=False),
                ToolArgSpec("window_end_utc", "integer", "Required when action=set_window_utc.", required=False),
                ToolArgSpec("confirm_token", "string", "Confirmation token for pending permanent changes.", required=False),
            ),
        ),
        _tool_override_guardrail(config_path),
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
                ToolArgSpec("run_agent_confirmation", "boolean", "Run sub-inference confirmation agents on directional setups.", required=False),
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
    registry.register(
        ToolSpec(
            name="configure_pair_sentiment_gate",
            description="Update YAML inference config for pair-specific sentiment gating.",
            args=(
                ToolArgSpec("pair", "string", "FX pair like AUD_NZD or AUD/NZD.", required=True),
                ToolArgSpec("enabled", "boolean", "Whether the pair-specific sentiment gate should be enabled.", required=False),
                ToolArgSpec("directions", "string|array", "Optional directions such as short,long.", required=False),
                ToolArgSpec("threshold", "number", "Optional contrary-sentiment threshold override.", required=False),
                ToolArgSpec("min_headlines", "integer", "Optional minimum headline count override.", required=False),
            ),
        ),
        _tool_configure_pair_sentiment_gate(config_path),
    )
    registry.register(
        ToolSpec(
            name="configure_correlation_sentiment_gate",
            description="Propagate a master-pair sentiment direction across its correlation group.",
            args=(
                ToolArgSpec("master_pair", "string", "Master pair like AUD_NZD.", required=True),
                ToolArgSpec("direction", "string", "Master direction: long or short.", required=True),
                ToolArgSpec("enabled", "boolean", "Whether to enable the propagated sentiment gate.", required=False),
                ToolArgSpec("threshold", "number", "Optional contrary-sentiment threshold override.", required=False),
                ToolArgSpec("min_headlines", "integer", "Optional minimum headline count override.", required=False),
            ),
        ),
        _tool_configure_correlation_sentiment_gate(config_path),
    )
    registry.register(
        ToolSpec(
            name="retrain_meta_labelers",
            description="Retrain meta-labelers for the master pair set.",
            args=(
                ToolArgSpec("data_source", "string", "journal, backtest, or simulation.", required=False),
                ToolArgSpec("master_pairs", "string|array", "Optional master pair subset.", required=False),
                ToolArgSpec("min_samples", "integer", "Minimum samples required per pair.", required=False),
                ToolArgSpec("simulate_trades", "integer", "Simulation size when data_source=simulation.", required=False),
                ToolArgSpec("directional_accuracy", "number", "Simulation directional accuracy.", required=False),
                ToolArgSpec("short_accuracy", "number", "Simulation short-direction accuracy.", required=False),
                ToolArgSpec("long_accuracy", "number", "Simulation long-direction accuracy.", required=False),
                ToolArgSpec("short_bias", "number", "Probability of generating short trades in simulation.", required=False),
                ToolArgSpec("simulation_profiles", "object|string", "Optional per-pair simulation profile overrides.", required=False),
                ToolArgSpec("journal_path", "string", "Optional trade journal path.", required=False),
                ToolArgSpec("backtest_path", "string", "Optional backtest results path.", required=False),
                ToolArgSpec("backtest_paths", "object|string", "Optional per-pair backtest result paths.", required=False),
                ToolArgSpec("backtest_dir", "string", "Optional directory to auto-discover per-pair backtest JSON.", required=False),
            ),
        ),
        _tool_retrain_meta_labelers(root),
    )
    registry.register(
        ToolSpec(
            name="run_basket_backtest",
            description="Run a grounded basket backtest and return an equity-curve summary.",
            args=(
                ToolArgSpec("baskets", "string|array", "Basket aliases like euro,aussie.", required=False),
                ToolArgSpec("pairs", "string|array", "Explicit pair list.", required=False),
                ToolArgSpec("granularity", "string", "Timeframe like H1 or M15.", required=False),
                ToolArgSpec("profile", "string", "Inference profile.", required=False),
                ToolArgSpec("approved_only", "boolean", "Use only approved setups.", required=False),
                ToolArgSpec("window", "integer", "Quick-backtest window length.", required=False),
                ToolArgSpec("force", "boolean", "Bypass session filter during the scan.", required=False),
            ),
        ),
        _tool_run_basket_backtest(config_path),
    )
    registry.register(
        ToolSpec(
            name="configure_aggressive_regime_sizing",
            description="Enable or tune aggressive volatility-regime sizing in scanner config.",
            args=(
                ToolArgSpec("enabled", "boolean", "Whether aggressive regime sizing should be enabled.", required=False),
                ToolArgSpec("high_vol_scale", "number", "Scale factor for HIGH volatility.", required=False),
                ToolArgSpec("extreme_vol_scale", "number", "Scale factor for EXTREME volatility.", required=False),
                ToolArgSpec("min_meta_confidence", "number", "Meta confidence threshold for aggressive scaling.", required=False),
            ),
        ),
        _tool_configure_aggressive_regime_sizing(config_path),
    )
    registry.register(
        ToolSpec(
            name="configure_scan_profile",
            description="Set the scanner profile to conservative, balanced, or aggressive.",
            args=(
                ToolArgSpec("profile", "string", "Scanner profile: conservative, balanced, or aggressive.", required=True),
            ),
        ),
        _tool_configure_scan_profile(config_path),
    )
    return registry
