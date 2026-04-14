"""Planner runtime for Buddy's tool-grounded conversational agent."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_providers import llm_call, select_buddy_provider_name

from .planner_types import PlannerPlan, PlannerRuntimeResponse
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry, build_default_tool_registry
from .tool_schema import ToolCall, ToolResult

logger = logging.getLogger(__name__)

_FX_CURRENCIES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}
_FX_PRECISE_PAIR_ALIASES = {
    "eu": "EUR_USD",
    "fiber": "EUR_USD",
    "gu": "GBP_USD",
    "cable": "GBP_USD",
    "uj": "USD_JPY",
    "usdjpy": "USD_JPY",
    "guppy": "GBP_JPY",
    "gj": "GBP_JPY",
    "dragon": "GBP_JPY",
    "ej": "EUR_JPY",
    "eurjpy": "EUR_JPY",
    "swissy": "USD_CHF",
    "uc": "USD_CAD",
    "au": "AUD_USD",
    "nu": "NZD_USD",
    "gold": "XAU_USD",
    "xau": "XAU_USD",
}
_AMBIGUOUS_FX_ALIAS_OPTIONS = {
    "euro": ("EUR_USD", "EUR_JPY", "EUR_GBP"),
    "yen": ("USD_JPY", "EUR_JPY", "GBP_JPY"),
    "sterling": ("GBP_USD", "GBP_JPY", "EUR_GBP"),
    "aussie": ("AUD_USD", "AUD_JPY", "EUR_AUD"),
    "kiwi": ("NZD_USD", "NZD_JPY", "EUR_NZD"),
    "loonie": ("USD_CAD", "EUR_CAD", "GBP_CAD"),
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_SCAN_GRANULARITY = "H1"
_DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are Buddy Planner 3B, the grounded planning and response layer for Buddy. "
    "Be decisive, concise, and action-oriented. "
    "When a user asks for an opportunity, setup, trade idea, or what looks good, default to scanning the market on H1 unless they explicitly ask for another timeframe. "
    "Prefer taking the next grounded step over asking meta clarifications. "
    "Clarify only when the request cannot be satisfied safely from the available tools and context. "
    "Never invent trading facts. "
    "Route all session-guardrail changes through one tool: override_guardrail. "
    "Examples: 'ignore session filter', 'disable session guardrail', 're-enable session filter', "
    "'set session window 9 to 17 utc', and 'allow trades at 18 utc'."
)
_REMOTE_PLANNER_SYSTEM_PROMPT = (
    "You are Buddy Planner 3B, the grounded planning and response layer for Buddy. "
    "Be decisive and action-oriented. "
    "If the user asks for an opportunity, setup, trade idea, or what looks good, prefer a market scan on H1 unless they explicitly request another timeframe. "
    "Avoid meta clarifications unless the request cannot be grounded safely. "
    "Return JSON only. "
    "You must not invent trading facts. "
    "Either return a plan with tool steps or a direct clarification/final answer. "
    "Valid JSON shape: "
    '{"needs_clarification": bool, "clarification_question": str|null, "final_answer": str|null, '
    '"steps": [{"tool": str, "args": {...}}]}.'
)
_LOCAL_RESPONSE_SYSTEM_PROMPT = (
    "You are Buddy Planner 3B speaking directly to the user. "
    "Use the grounded answer and tool outputs exactly as your factual boundary. "
    "Rewrite them into a concise, natural Buddy reply in first person when appropriate. "
    "Do not add facts, permissions, market claims, or hidden state. "
    "Do not output JSON."
)
_DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parents[2] / "trained_data" / "checkpoint-900.zip"
_LOCAL_ADAPTER_CACHE: dict[str, Any] = {}


@dataclass
class PlannerRuntime:
    config_path: str
    registry: ToolRegistry
    provider: str | None = None
    adapter_path: str | None = None
    _local_adapter: Any = field(default=None, init=False, repr=False)
    _local_adapter_failed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create_default(cls, config_path: str, *, provider: str | None = None) -> "PlannerRuntime":
        adapter_path = os.getenv("BUDDY_PLANNER_ADAPTER_PATH")
        if not adapter_path:
            adapter_path = str(_DEFAULT_ADAPTER_PATH)
        return cls(
            config_path=config_path,
            registry=build_default_tool_registry(config_path),
            provider=provider,
            adapter_path=adapter_path,
        )

    def available_tools_payload(self) -> list[dict[str, Any]]:
        return [tool.prompt_schema() for tool in self.registry.available_tools()]

    def plan_request(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> PlannerPlan:
        if self._prefer_deterministic_first():
            deterministic = self._deterministic_plan(
                user_message,
                chat_history=chat_history,
                runtime_context=runtime_context,
            )
            if not deterministic.needs_clarification:
                return deterministic
        llm_plan = self._plan_with_llm(user_message, chat_history=chat_history, runtime_context=runtime_context)
        if llm_plan is not None:
            return llm_plan
        return self._deterministic_plan(user_message, chat_history=chat_history, runtime_context=runtime_context)

    def handle_request(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
        runtime_context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PlannerRuntimeResponse:
        def emit(event: str, payload: dict[str, Any] | None = None) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event, payload or {})
            except Exception as e:
                logger.debug("Planner progress callback failed (%s): %s", event, e)

        emit("thinking_start", {"message": user_message})
        runtime_state = dict(state or {})
        if runtime_context:
            runtime_state.update(runtime_context)
        prompt_context = self._planner_runtime_context(runtime_context=runtime_context, state=runtime_state)
        plan = self.plan_request(user_message, chat_history=chat_history, runtime_context=prompt_context)
        emit(
            "plan_ready",
            {
                "needs_clarification": bool(plan.needs_clarification),
                "step_count": len(plan.steps),
                "tools": [step.tool for step in plan.steps],
            },
        )
        if plan.needs_clarification:
            emit("reasoning_start", {"source": "clarification"})
            answer = plan.clarification_question or "I need a little more detail before I can act."
            answer = self._render_local_voice_answer(
                user_message,
                grounded_answer=answer,
                chat_history=chat_history,
                runtime_context=prompt_context,
                tool_results=[],
            )
            emit("done", {"tool_count": 0})
            return PlannerRuntimeResponse(plan=plan, tool_results=[], answer=answer)
        if plan.final_answer and not plan.steps:
            emit("reasoning_start", {"source": "final_answer"})
            answer = self._render_local_voice_answer(
                user_message,
                grounded_answer=plan.final_answer,
                chat_history=chat_history,
                runtime_context=prompt_context,
                tool_results=[],
            )
            emit("done", {"tool_count": 0})
            return PlannerRuntimeResponse(plan=plan, tool_results=[], answer=answer)

        executor = ToolExecutor(self.registry, state=runtime_state)
        results = executor.execute_plan(plan.steps, progress_callback=emit)
        emit("reasoning_start", {"source": "tool_results", "tool_count": len(results)})
        answer = self._synthesize_answer(
            user_message,
            results,
            chat_history=chat_history,
            runtime_context=prompt_context,
        )
        emit("done", {"tool_count": len(results)})
        return PlannerRuntimeResponse(plan=plan, tool_results=results, answer=answer, metadata={"state": runtime_state})

    @staticmethod
    def _planner_runtime_context(*, runtime_context: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
        prompt_context: dict[str, Any] = {}
        if runtime_context:
            prompt_context.update(runtime_context)
        for key, value in state.items():
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except Exception:
                continue
            prompt_context[key] = value
        return prompt_context

    def _plan_with_llm(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
    ) -> PlannerPlan | None:
        local_plan = self._plan_with_local_adapter(
            user_message,
            chat_history=chat_history,
            runtime_context=runtime_context,
        )
        if local_plan is not None:
            return local_plan

        if not (self.provider or os.getenv("BUDDY_PLANNER_USE_LLM", "").strip().lower() in _TRUE_VALUES):
            return None
        provider = self.provider or select_buddy_provider_name(None)
        if not provider:
            return None

        payload = self._planner_payload(
            user_message,
            chat_history=chat_history,
            runtime_context=runtime_context,
        )
        try:
            response = llm_call(
                json.dumps(payload, indent=2),
                system_prompt=_REMOTE_PLANNER_SYSTEM_PROMPT,
                provider=provider,
                temperature=0.0,
                max_tokens=600,
            )
            if not response:
                return None
            return self._coerce_plan(response)
        except Exception as e:
            logger.debug("Planner LLM path failed: %s", e)
            return None

    def _planner_payload(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        available_tools = [
            {
                "name": tool["name"],
                "args_schema": {key: value.get("type") for key, value in (tool.get("args_schema") or {}).items()},
            }
            for tool in self.available_tools_payload()
        ]
        return {
            "system": _DEFAULT_PLANNER_SYSTEM_PROMPT,
            "user_message": user_message,
            "chat_history": (chat_history or [])[-4:],
            "runtime_context": runtime_context or {},
            "available_tools": available_tools,
            "guardrail_training_examples": self._guardrail_training_examples(),
        }

    def _plan_with_local_adapter(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
    ) -> PlannerPlan | None:
        if not self._local_adapter_enabled():
            return None
        adapter = self._load_local_adapter()
        if adapter is None:
            return None
        prompt = json.dumps(
            self._planner_payload(
                user_message,
                chat_history=chat_history,
                runtime_context=runtime_context,
            ),
            ensure_ascii=True,
        )
        try:
            text = self._generate_local_adapter_text(adapter, prompt)
            if not text:
                return None
            return self._coerce_plan(text)
        except Exception as e:
            logger.debug("Planner local adapter path failed: %s", e)
            return None

    def _local_adapter_enabled(self) -> bool:
        return os.getenv("BUDDY_PLANNER_USE_ADAPTER", "1").strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _allow_remote_adapter_base() -> bool:
        return os.getenv("BUDDY_PLANNER_ALLOW_REMOTE_ADAPTER_BASE", "0").strip().lower() in _TRUE_VALUES

    def _load_local_adapter(self) -> Any:
        if self._local_adapter is not None:
            return self._local_adapter
        if self._local_adapter_failed:
            return None
        adapter_dir = self._resolved_adapter_path()
        if adapter_dir is None or not (adapter_dir / "adapter_config.json").exists():
            return None
        cache_key = str(adapter_dir)
        cached = _LOCAL_ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            self._local_adapter = cached
            return cached

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils import logging as hf_logging
        except Exception as e:
            logger.debug("Planner adapter dependencies unavailable: %s", e)
            self._local_adapter_failed = True
            return None

        try:
            hf_logging.set_verbosity_error()
            disable_progress_bar = getattr(hf_logging, "disable_progress_bar", None)
            if callable(disable_progress_bar):
                disable_progress_bar()

            adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_name = str(adapter_cfg["base_model_name_or_path"])
            local_files_only = not self._allow_remote_adapter_base()
            tokenizer = self._load_adapter_tokenizer(
                adapter_dir=adapter_dir,
                base_model_name=base_model_name,
                auto_tokenizer_cls=AutoTokenizer,
                local_files_only=local_files_only,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = None
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                dtype = torch.float16
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            model_load_kwargs: dict[str, Any] = {}
            if dtype is not None:
                model_load_kwargs["dtype"] = dtype
            if local_files_only:
                model_load_kwargs["local_files_only"] = True
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_load_kwargs)
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            if self._merge_adapter_for_inference() and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
            model.to(device)
            model.eval()
            if hasattr(model, "generation_config"):
                model.generation_config.do_sample = False
                model.generation_config.use_cache = True
                for attr in ("temperature", "top_p", "top_k"):
                    if hasattr(model.generation_config, attr):
                        setattr(model.generation_config, attr, None)
            self._local_adapter = {"model": model, "tokenizer": tokenizer, "device": device}
            _LOCAL_ADAPTER_CACHE[cache_key] = self._local_adapter
            logger.info("Loaded Buddy planner adapter from %s", adapter_dir)
            return self._local_adapter
        except Exception as e:
            logger.debug("Planner adapter load failed: %s", e)
            self._local_adapter_failed = True
            return None

    @staticmethod
    def _load_adapter_tokenizer(
        *,
        adapter_dir: Path,
        base_model_name: str,
        auto_tokenizer_cls: Any,
        local_files_only: bool,
    ) -> Any:
        try:
            return auto_tokenizer_cls.from_pretrained(str(adapter_dir))
        except Exception as e:
            logger.debug("Adapter-local tokenizer load failed, falling back to base tokenizer: %s", e)
            return auto_tokenizer_cls.from_pretrained(base_model_name, local_files_only=local_files_only)

    @staticmethod
    def _adapter_required_files() -> tuple[str, ...]:
        return (
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )

    @classmethod
    def _is_adapter_dir(cls, path: Path) -> bool:
        return path.is_dir() and all((path / name).exists() for name in cls._adapter_required_files())

    @classmethod
    def _find_adapter_root(cls, root: Path) -> Path | None:
        if cls._is_adapter_dir(root):
            return root

        try:
            candidates = sorted(
                path for path in root.rglob("adapter_config.json")
                if path.is_file()
            )
        except Exception:
            return None

        for config_path in candidates:
            candidate = config_path.parent
            if cls._is_adapter_dir(candidate):
                return candidate
        return None

    @staticmethod
    def _zip_cache_key(archive_path: Path) -> str:
        stat = archive_path.stat()
        payload = f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _extract_adapter_archive(cls, archive_path: Path) -> Path | None:
        cache_root = archive_path.parent / ".planner_adapter_cache"
        extracted_root = cache_root / f"{archive_path.stem}-{cls._zip_cache_key(archive_path)}"
        marker = extracted_root / ".extracted.ok"

        if marker.exists():
            adapter_root = cls._find_adapter_root(extracted_root)
            if adapter_root is not None:
                return adapter_root

        extracted_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path) as bundle:
                for member in bundle.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"Unsafe planner adapter bundle member: {member.filename}")
                    target = extracted_root / member_path
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
        except Exception as e:
            logger.debug("Planner adapter archive extraction failed: %s", e)
            return None

        adapter_root = cls._find_adapter_root(extracted_root)
        if adapter_root is None:
            logger.debug("Planner adapter archive %s does not contain a valid adapter bundle", archive_path)
            return None

        extracted_names: set[str] = set()
        try:
            extracted_names = {
                str(path.relative_to(extracted_root))
                for path in extracted_root.rglob("*")
                if path.is_file()
            }
        except Exception:
            extracted_names = set()

        try:
            marker.write_text(json.dumps({"files": sorted(extracted_names)}, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to persist planner adapter extraction marker for %s", archive_path)

        return adapter_root

    def _resolved_adapter_path(self) -> Path | None:
        if not self.adapter_path:
            return None
        adapter_path = Path(self.adapter_path).expanduser().resolve()
        if adapter_path.is_file() and adapter_path.suffix.lower() == ".zip":
            return self._extract_adapter_archive(adapter_path)
        if self._is_adapter_dir(adapter_path):
            return adapter_path
        return self._find_adapter_root(adapter_path)

    def _prefer_deterministic_first(self) -> bool:
        return os.getenv("BUDDY_PLANNER_HYBRID_ROUTING", "1").strip().lower() in _TRUE_VALUES

    def _merge_adapter_for_inference(self) -> bool:
        default = "1"
        return os.getenv("BUDDY_PLANNER_MERGE_ADAPTER", default).strip().lower() in _TRUE_VALUES

    def _local_max_new_tokens(self) -> int:
        try:
            value = int(os.getenv("BUDDY_PLANNER_MAX_NEW_TOKENS", "96"))
        except ValueError:
            value = 96
        return max(32, value)

    def _generate_local_adapter_text(
        self,
        adapter: dict[str, Any],
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str | None:
        import torch

        tokenizer = adapter["tokenizer"]
        model = adapter["model"]
        device = adapter["device"]
        inputs = tokenizer(prompt + "\n", return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens or self._local_max_new_tokens(),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        output_ids = generated[0][prompt_len:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return text or None

    @staticmethod
    def _should_render_local_voice(user_message: str) -> bool:
        ql = user_message.lower().strip()
        ql_compact = re.sub(r"[^a-z0-9]+", " ", ql).strip()
        direct_toolish_prefixes = (
            "scan",
            "predict",
            "trade",
            "buy",
            "sell",
            "use csv",
            "use ticker",
            "use oanda",
            "oanda ",
            "sl ",
            "tp ",
            "stop loss",
            "take profit",
            "execute on",
            "execute off",
            "force buy",
            "force sell",
        )
        if ql.startswith(direct_toolish_prefixes):
            return False
        if any(phrase in ql_compact for phrase in ("session filter", "session window", "guardrail")):
            return False
        if ql.startswith(("bypass ", "ignore ", "disable ", "enable ", "re enable ", "re-enable ")):
            return False
        if PlannerRuntime._looks_like_scan_request(user_message, ql, ql_compact):
            return False
        return True

    @staticmethod
    def _tool_results_prefer_grounded_answer(tool_results: list[ToolResult]) -> bool:
        if not tool_results:
            return False
        direct_tools = {
            "get_status",
            "get_decisions",
            "get_model_status",
            "get_account_summary",
            "get_journal_summary",
            "override_guardrail",
            "run_runtime_command",
            "configure_pair_sentiment_gate",
            "configure_correlation_sentiment_gate",
            "configure_aggressive_regime_sizing",
            "configure_scan_profile",
        }
        return all(result.tool_name in direct_tools for result in tool_results)

    def _render_local_voice_answer(
        self,
        user_message: str,
        *,
        grounded_answer: str,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
        tool_results: list[ToolResult],
    ) -> str:
        if not grounded_answer or not self._should_render_local_voice(user_message):
            return grounded_answer
        if self._tool_results_prefer_grounded_answer(tool_results):
            return grounded_answer

        adapter = self._load_local_adapter()
        if adapter is None:
            return grounded_answer

        tool_payload = [
            {
                "tool_name": result.tool_name,
                "ok": result.ok,
                "result": result.result,
                "human_summary": result.human_summary,
            }
            for result in tool_results
        ]
        prompt = json.dumps(
            {
                "system": _LOCAL_RESPONSE_SYSTEM_PROMPT,
                "user_message": user_message,
                "grounded_answer": grounded_answer,
                "runtime_context": runtime_context or {},
                "chat_history": (chat_history or [])[-4:],
                "tool_outputs": tool_payload,
            },
            ensure_ascii=True,
        )
        try:
            rewritten = self._generate_local_adapter_text(
                adapter,
                prompt,
                max_new_tokens=max(64, min(192, self._local_max_new_tokens() * 2)),
            )
        except Exception as e:
            logger.debug("Planner local voice rendering failed: %s", e)
            return grounded_answer
        if not rewritten:
            return grounded_answer
        return rewritten.strip()

    def _coerce_plan(self, response: str) -> PlannerPlan | None:
        data = self._extract_json_object(response)
        if not isinstance(data, dict):
            return None

        plan_data = data.get("target_plan") if isinstance(data.get("target_plan"), dict) else data
        if not isinstance(plan_data, dict):
            return None

        steps = [
            ToolCall(tool=str(step["tool"]), args=dict(step.get("args") or {}))
            for step in plan_data.get("steps") or []
            if isinstance(step, dict) and step.get("tool")
        ]
        needs_clarification = bool(plan_data.get("needs_clarification", False))
        target_response = data.get("target_response")
        clarification_question = plan_data.get("clarification_question")
        final_answer = plan_data.get("final_answer")

        if needs_clarification and not clarification_question and isinstance(target_response, str):
            clarification_question = target_response
        if not needs_clarification and not steps and not final_answer and isinstance(target_response, str):
            final_answer = target_response

        return PlannerPlan(
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            steps=steps,
            final_answer=final_answer,
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _help_answer() -> str:
        return (
            "I can scan for setups, compare the last scan, show runtime or model status, manage journal/account actions, "
            "change execute or SL/TP settings, and answer grounded Buddy self-knowledge questions. "
            "Try prompts like 'look for an opportunity', 'scan M5', 'show status', 'buy fiber', 'short cable', "
            "'fade uj', 'run agents', 'execute trades', 'ignore session filter', 'restore session filter', "
            "'set session window 9 to 17 utc', or 'compare EUR/USD and GBP/USD'."
        )

    @staticmethod
    def _guardrail_training_examples() -> list[dict[str, Any]]:
        return [
            {
                "user_phrase": "ignore session filter",
                "tool": "override_guardrail",
                "args": {
                    "guardrail_name": "session",
                    "action": "disable",
                    "scope": "this_scan",
                },
            },
            {
                "user_phrase": "disable session filter for this session",
                "tool": "override_guardrail",
                "args": {
                    "guardrail_name": "session",
                    "action": "disable",
                    "scope": "this_session",
                },
            },
            {
                "user_phrase": "turn on session filter again",
                "tool": "override_guardrail",
                "args": {
                    "guardrail_name": "session",
                    "action": "enable",
                    "scope": "this_scan",
                },
            },
            {
                "user_phrase": "re-enable session filter permanently",
                "tool": "override_guardrail",
                "args": {
                    "guardrail_name": "session",
                    "action": "enable",
                    "scope": "permanent",
                },
            },
            {
                "user_phrase": "set session window 9 to 17 utc",
                "tool": "override_guardrail",
                "args": {
                    "guardrail_name": "session",
                    "action": "set_window_utc",
                    "scope": "this_session",
                    "window_start_utc": 9,
                    "window_end_utc": 17,
                },
            },
        ]

    @staticmethod
    def _scan_error_followup_answer(runtime_context: dict[str, Any]) -> str | None:
        summary = runtime_context.get("last_scan_summary")
        if not isinstance(summary, dict):
            return None

        rows = summary.get("all_rows") or []
        if not isinstance(rows, list) or not rows:
            return None

        errors = [str(row.get("error") or "").strip() for row in rows if isinstance(row, dict) and row.get("error")]
        if not errors:
            return None

        market_closed_errors = [error for error in errors if error.lower().startswith("fx market closed")]
        if market_closed_errors:
            first = market_closed_errors[0]
            if len(market_closed_errors) == len(rows):
                return (
                    "That isn't a runtime failure. The FX market is closed right now: "
                    f"{first}. The market has to reopen before those setups can become actionable."
                )
            return (
                f"Some scan rows were blocked because the FX market is closed: {first}. "
                "Those rows will only become actionable after the market reopens."
            )

        session_errors = [error for error in errors if error.lower().startswith("outside trading session")]
        if not session_errors:
            return None

        first = session_errors[0]
        if len(session_errors) == len(rows):
            return (
                "That isn't a runtime failure. The scan was blocked by the session filter: "
                f"{first}. Run it during the configured session window or use force to bypass the session filter."
            )
        return (
            f"Some scan rows were blocked by the session filter: {first}. "
            "Those rows were held back because the scan ran outside the configured session window."
        )

    @staticmethod
    def _execute_mode_followup_answer(ql_compact: str, runtime_context: dict[str, Any]) -> str | None:
        phrases = (
            "why dry run",
            "why dry run mode",
            "why are you in dry run",
            "why are you in dry run mode",
            "why dry",
            "why not live",
            "why aren t you live",
            "why are you not live",
            "why not executing",
            "why aren t you executing",
        )
        if not any(phrase in ql_compact for phrase in phrases):
            return None

        execute_mode = str(runtime_context.get("execute_mode") or "").strip().lower()
        if execute_mode == "dry-run":
            return (
                "I'm in dry-run because Buddy chat defaults to safe execution. "
                "I will scan, explain, and stage trade actions without sending orders until you say 'execute on' or launch chat with --execute."
            )
        if execute_mode == "live":
            return "Execution is already live right now, so approved trade actions can place real practice orders."
        return "I don't have the current execution mode pinned down yet. Ask for status and I'll refresh it."

    @staticmethod
    def _looks_like_opportunity_request(ql_compact: str) -> bool:
        phrases = (
            "look for an opportunity",
            "look for opportunity",
            "find an opportunity",
            "find opportunity",
            "look for a setup",
            "find a setup",
            "look for a trade",
            "find a trade",
            "anything to trade",
            "anything tradable",
            "what looks good",
            "what looks tradable",
            "best opportunity",
            "best setup",
            "best pair to trade",
            "find best pair",
            "find the best pair",
            "look for the best pair",
            "what should i trade",
            "which pair should i trade",
            "what pair should i trade",
            "find something to trade",
            "pick a pair",
            "trade idea",
            "look for something",
        )
        if any(phrase in ql_compact for phrase in phrases):
            return True
        if "opportunity" in ql_compact:
            return True
        if "best pair" in ql_compact and "trade" in ql_compact:
            return True
        if "pair" in ql_compact and "trade" in ql_compact and any(word in ql_compact for word in ("best", "find", "which", "what")):
            return True
        return False

    @staticmethod
    def _looks_like_scan_request(q: str, ql: str, ql_compact: str) -> bool:
        if "scan" in ql:
            return True

        tokens = ql_compact.split()
        if not tokens:
            return False

        first = tokens[0]
        if difflib.SequenceMatcher(None, first, "scan").ratio() >= 0.75:
            return True

        granularity_hint = re.search(r"\b(M1|M5|M15|M30|H1|H4|D|D1)\b", q, flags=re.IGNORECASE)
        if granularity_hint and any(difflib.SequenceMatcher(None, token, "scan").ratio() >= 0.75 for token in tokens[:2]):
            return True

        return False

    @staticmethod
    def _looks_like_safe_default_scan_request(ql_compact: str) -> bool:
        exact_phrases = {
            "do the thing",
            "handle it",
            "take care of that",
            "run it",
            "make it happen",
            "what should we do",
            "do something useful",
            "work on this",
            "figure it out",
            "go ahead",
            "use your judgment",
            "take the next step",
            "do whatever makes sense",
            "help somehow",
            "help me out",
        }
        if ql_compact in exact_phrases:
            return True
        if "tomorrow" in ql_compact and "trade" in ql_compact and "pair" in ql_compact:
            return True
        return False

    @staticmethod
    def _default_scan_tool_call(
        q: str,
        ql: str,
        ql_compact: str,
        runtime_context: dict[str, Any],
        *,
        pairs: list[str] | None = None,
    ) -> ToolCall:
        return ToolCall(
            "scan_market",
            PlannerRuntime._scan_args_from_request(
                q,
                ql,
                ql_compact,
                runtime_context,
                pairs=pairs,
            ),
        )

    @staticmethod
    def _scan_args_from_request(
        q: str,
        ql: str,
        ql_compact: str,
        runtime_context: dict[str, Any],
        *,
        pairs: list[str] | None = None,
    ) -> dict[str, Any]:
        granularity_match = re.search(r"\b(M1|M5|M15|M30|H1|H4|D|D1)\b", q, flags=re.IGNORECASE)
        top_match = re.search(r"\btop\s+(\d+)\b", ql)
        return {
            "pairs": ",".join(pairs) if pairs else None,
            "granularity": granularity_match.group(1).upper() if granularity_match else _DEFAULT_SCAN_GRANULARITY,
            "top_n": int(top_match.group(1)) if top_match else 5,
            "auto_execute": any(
                phrase in ql
                for phrase in (
                    "auto-execute",
                    "auto execute",
                    "execute if approved",
                    "execute approved",
                    "execute trades",
                    "execute trade",
                )
            ),
            "force": "force" in ql,
            "diversified": "diversified" in ql,
            "run_agent_confirmation": PlannerRuntime._looks_like_agent_confirmation_request(ql_compact),
        }

    @staticmethod
    def _pairs_from_last_scan(runtime_context: dict[str, Any]) -> list[str]:
        summary = runtime_context.get("last_scan_summary")
        if not isinstance(summary, dict):
            return []
        raw_rows = summary.get("all_rows") or summary.get("rows") or []
        if not isinstance(raw_rows, list):
            return []
        pairs: list[str] = []
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            pair = str(row.get("pair") or "").upper().replace("/", "_")
            if pair and pair not in pairs:
                pairs.append(pair)
        return pairs

    @staticmethod
    def _scan_granularity_from_context(runtime_context: dict[str, Any]) -> str:
        summary = runtime_context.get("last_scan_summary")
        if isinstance(summary, dict):
            candidate = str(summary.get("granularity") or "").upper().strip()
            if candidate in {"M1", "M5", "M15", "M30", "H1", "H4", "D", "D1"}:
                return candidate
            return _DEFAULT_SCAN_GRANULARITY

        fallback = str(runtime_context.get("granularity") or "").upper().strip()
        if fallback in {"M1", "M5", "M15", "M30", "H1", "H4", "D", "D1"}:
            return fallback
        return _DEFAULT_SCAN_GRANULARITY

    @staticmethod
    def _scan_window_from_context(runtime_context: dict[str, Any]) -> tuple[int, int]:
        window = runtime_context.get("session_window_effective")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            try:
                start = int(window[0])
                end = int(window[1])
                if 0 <= start < end <= 24:
                    return start, end
            except Exception:
                pass

        summary = runtime_context.get("last_scan_summary")
        if isinstance(summary, dict):
            scan_window = summary.get("session_window_effective")
            if isinstance(scan_window, (list, tuple)) and len(scan_window) == 2:
                try:
                    start = int(scan_window[0])
                    end = int(scan_window[1])
                    if 0 <= start < end <= 24:
                        return start, end
                except Exception:
                    pass

        return 8, 21

    @staticmethod
    def _override_scope_from_phrase(ql_compact: str) -> str:
        if any(
            phrase in ql_compact
            for phrase in ("permanent", "permanently", "persist", "save default", "keep this forever", "from now on")
        ):
            return "permanent"
        if any(
            phrase in ql_compact
            for phrase in ("for now", "this session", "for this session", "until i change it")
        ):
            return "this_session"
        if any(
            phrase in ql_compact
            for phrase in ("this scan", "next scan", "just once", "one scan")
        ):
            return "this_scan"
        return "this_scan"

    @staticmethod
    def _extract_explicit_session_window_utc(text: str) -> tuple[int, int] | None:
        match = re.search(
            r"\b(?:session\s+window|window|trade\s+hours?)\s*(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})(?:\s*utc)?\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        start = int(match.group(1))
        end = int(match.group(2))
        if not (0 <= start < end <= 24):
            return None
        return start, end

    @staticmethod
    def _extract_allow_hour_utc(text: str) -> int | None:
        am_pm = re.search(r"\bat\s*(\d{1,2})\s*(am|pm)\s*utc\b", text, flags=re.IGNORECASE)
        if am_pm:
            hour = int(am_pm.group(1)) % 12
            suffix = am_pm.group(2).lower()
            if suffix == "pm":
                hour += 12
            return hour

        raw = re.search(r"\bat\s*(\d{1,2})(?::\d{2})?\s*utc\b", text, flags=re.IGNORECASE)
        if raw:
            hour = int(raw.group(1))
            if 0 <= hour <= 23:
                return hour
        return None

    @staticmethod
    def _infer_non_wrapping_window_for_hour(
        hour_utc: int,
        *,
        current_window: tuple[int, int],
    ) -> tuple[int, int]:
        start, end = current_window
        if start <= hour_utc < end:
            return start, end
        if hour_utc < start:
            return hour_utc, end
        return start, min(24, hour_utc + 1)

    @staticmethod
    def _pending_guardrail_override(runtime_context: dict[str, Any]) -> dict[str, Any] | None:
        pending = runtime_context.get("pending_guardrail_override")
        if isinstance(pending, dict):
            return pending
        return None

    @staticmethod
    def _looks_like_guardrail_confirm_request(ql_compact: str) -> bool:
        if any(
            phrase in ql_compact
            for phrase in (
                "confirm override",
                "confirm permanent change",
                "confirm permanent override",
                "confirm the override",
                "confirm it",
                "approve override",
            )
        ):
            return True
        return ql_compact in {"confirm", "yes confirm"}

    @staticmethod
    def _looks_like_guardrail_cancel_request(ql_compact: str) -> bool:
        if any(
            phrase in ql_compact
            for phrase in (
                "cancel override",
                "cancel permanent change",
                "cancel that override",
                "abort override",
                "don t persist",
                "dont persist",
            )
        ):
            return True
        return ql_compact in {"cancel", "abort", "never mind"}

    @staticmethod
    def _confirmation_token_from_text(text: str) -> str | None:
        token_match = re.search(r"\b([a-f0-9]{8})\b", text, flags=re.IGNORECASE)
        if token_match:
            return token_match.group(1).lower()
        return None

    def _guardrail_override_args_from_phrase(
        self,
        q: str,
        *,
        ql_compact: str,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        disable_phrases = (
            "ignore session filter",
            "disable session filter",
            "turn off session filter",
            "bypass session filter",
            "bypass session check",
            "disable guardrails",
            "disable the guardrails",
            "disable session guardrail",
            "turn off session guardrail",
            "turn off session guardrails",
            "outside trading hours",
            "outside hours",
            "bypass trading hours",
            "outside market hours",
            "ignore trading hours",
            "scan outside hours",
            "allow outside market hours",
            "ignore market hours",
        )
        enable_phrases = (
            "enable session filter",
            "turn session filter back on",
            "turn on session filter",
            "respect trading hours again",
            "restore session filter",
            "re-enable session filter",
            "re enable session filter",
            "reenable session filter",
            "turn trading hours on",
            "enable trading hours",
            "reinstate session filter",
            "restore session guardrail",
        )

        action: str | None = None
        args: dict[str, Any] = {"guardrail_name": "session"}
        if any(phrase in ql_compact for phrase in disable_phrases):
            action = "disable"
        elif any(phrase in ql_compact for phrase in enable_phrases):
            action = "enable"
        elif "toggle session filter" in ql_compact:
            action = "toggle"

        explicit_window = self._extract_explicit_session_window_utc(q)
        if explicit_window is not None:
            action = "set_window_utc"
            args["window_start_utc"] = explicit_window[0]
            args["window_end_utc"] = explicit_window[1]
            if action not in {"disable", "enable", "toggle", "set_window_utc"}:
                action = "set_window_utc"
        elif "allow trades at" in ql_compact and "utc" in ql_compact:
            hour = self._extract_allow_hour_utc(q)
            if hour is not None:
                start, end = self._infer_non_wrapping_window_for_hour(
                    hour,
                    current_window=self._scan_window_from_context(runtime_context),
                )
                action = "set_window_utc"
                args["window_start_utc"] = start
                args["window_end_utc"] = end

        if action is None:
            return None

        args["action"] = action
        args["scope"] = self._override_scope_from_phrase(ql_compact)
        return args

    @staticmethod
    def _looks_like_agent_confirmation_request(ql_compact: str) -> bool:
        phrases = (
            "run agents",
            "run agent",
            "rerun agents",
            "rerun agent",
            "execute agents",
            "agent confirmation",
            "confirm setups",
            "confirm setup",
            "run confirmations",
            "run confirmation",
        )
        return any(phrase in ql_compact for phrase in phrases)

    @staticmethod
    def _scan_rows_from_context(runtime_context: dict[str, Any], *, approved_only: bool = False) -> list[dict[str, Any]]:
        summary = runtime_context.get("last_scan_summary")
        if not isinstance(summary, dict):
            return []

        raw_rows = summary.get("all_rows") or summary.get("rows") or []
        if not isinstance(raw_rows, list):
            return []

        rows = [row for row in raw_rows if isinstance(row, dict)]
        if approved_only:
            filtered = [row for row in rows if bool(row.get("gates_passed"))]
            if filtered:
                rows = filtered
        else:
            tradable = [
                row for row in rows
                if str(row.get("direction") or "").upper() not in {"NONE", "FLAT", "HOLD"}
                and not row.get("error")
            ]
            if tradable:
                rows = tradable

        def _score(row: dict[str, Any]) -> tuple[int, float, float]:
            return (
                1 if row.get("gates_passed") else 0,
                float(row.get("confidence") or 0.0),
                float(row.get("agent_total") or 0.0),
            )

        return sorted(rows, key=_score, reverse=True)

    @classmethod
    def _top_scan_pair(cls, runtime_context: dict[str, Any], *, approved_only: bool = True) -> str | None:
        rows = cls._scan_rows_from_context(runtime_context, approved_only=approved_only)
        if not rows and approved_only:
            rows = cls._scan_rows_from_context(runtime_context, approved_only=False)
        if not rows:
            return None
        pair = str(rows[0].get("pair") or "").upper().replace("/", "_")
        return pair or None

    @classmethod
    def _executable_scan_pairs(cls, runtime_context: dict[str, Any]) -> list[str]:
        pairs: list[str] = []
        for row in cls._scan_rows_from_context(runtime_context, approved_only=False):
            pair = str(row.get("pair") or "").upper().replace("/", "_")
            direction = str(row.get("direction") or "").upper()
            if not pair or direction in {"", "NONE", "FLAT", "HOLD"}:
                continue
            if not bool(row.get("gates_passed")) or row.get("error"):
                continue
            if pair not in pairs:
                pairs.append(pair)
        return pairs

    @classmethod
    def _scan_selection_answer(cls, runtime_context: dict[str, Any]) -> str | None:
        rows = cls._scan_rows_from_context(runtime_context, approved_only=True)
        approved_only = True
        if not rows:
            rows = cls._scan_rows_from_context(runtime_context, approved_only=False)
            approved_only = False
        if not rows:
            return None

        row = rows[0]
        pair = str(row.get("pair") or "unknown pair")
        direction = str(row.get("direction") or "unknown direction")
        confidence = float(row.get("confidence") or 0.0)
        agent_total = float(row.get("agent_total") or 0.0)
        if approved_only:
            return (
                f"The top approved setup from the last scan is {pair} {direction} "
                f"with confidence {confidence:.2f} and agent score {agent_total:.2f}."
            )
        error = str(row.get("error") or "").strip()
        error_suffix = f" It was held back because {error}." if error else ""
        return (
            f"The top setup I still have in the last scan context is {pair} {direction} "
            f"with confidence {confidence:.2f}.{error_suffix}"
        ).strip()

    @classmethod
    def _scan_reason_answer(cls, runtime_context: dict[str, Any]) -> str | None:
        rows = cls._scan_rows_from_context(runtime_context, approved_only=True)
        if not rows:
            rows = cls._scan_rows_from_context(runtime_context, approved_only=False)
        if not rows:
            return None

        row = rows[0]
        pair = str(row.get("pair") or "unknown pair")
        direction = str(row.get("direction") or "unknown direction")
        confidence = float(row.get("confidence") or 0.0)
        agent_total = float(row.get("agent_total") or 0.0)
        if row.get("gates_passed"):
            return (
                f"{pair} was the strongest setup in the last scan because it passed the gates, "
                f"leaned {direction}, and carried confidence {confidence:.2f} with agent score {agent_total:.2f}."
            )
        error = str(row.get("error") or "").strip()
        if error:
            return f"{pair} was not approved in the last scan because {error}."
        return (
            f"{pair} was the top stored setup from the last scan on confidence {confidence:.2f} "
            f"and agent score {agent_total:.2f}, but it did not fully pass the gates."
        )

    @staticmethod
    def _looks_like_status_request(ql_compact: str) -> bool:
        phrases = (
            "show status",
            "give me status",
            "give me the status",
            "what s the status",
            "what is the status",
            "where are we at",
            "where do things stand",
            "how are we looking",
            "how s buddy set up",
            "how are you set up",
            "current mode",
            "current setup",
            "current settings",
            "risk settings",
            "execution mode",
            "are we live",
            "are we in dry run",
            "what mode are you in",
        )
        return any(phrase in ql_compact for phrase in phrases)

    @staticmethod
    def _looks_like_account_request(ql_compact: str) -> bool:
        phrases = (
            "account status",
            "account update",
            "how s the account",
            "how is the account",
            "what s my balance",
            "what is my balance",
            "how much margin",
            "buying power",
            "margin available",
            "nav",
            "account balance",
        )
        return any(phrase in ql_compact for phrase in phrases)

    @staticmethod
    def _looks_like_journal_import_request(ql_compact: str) -> bool:
        return "journal" in ql_compact and any(phrase in ql_compact for phrase in ("import", "untracked"))

    @staticmethod
    def _looks_like_journal_sync_request(ql_compact: str) -> bool:
        return "journal" in ql_compact and any(phrase in ql_compact for phrase in ("update", "sync", "refresh"))

    @staticmethod
    def _looks_like_journal_request(ql_compact: str) -> bool:
        phrases = (
            "journal",
            "recent trades",
            "recent performance",
            "how did we do",
            "how have we been doing",
            "how s performance",
            "how is performance",
            "win rate",
            "pnl",
            "performance lately",
            "trade history",
        )
        return any(phrase in ql_compact for phrase in phrases)

    @staticmethod
    def _looks_like_prediction_request(ql_compact: str, *, has_pair: bool) -> bool:
        if not has_pair:
            return False
        phrases = (
            "analyze",
            "check",
            "look at",
            "what do you think about",
            "thoughts on",
            "read on",
            "view on",
        )
        return any(phrase in ql_compact for phrase in phrases)

    @staticmethod
    def _looks_like_sentiment_config_request(ql_compact: str) -> bool:
        if "sentiment" not in ql_compact:
            return False
        phrases = (
            "sentiment confirmation",
            "sentiment confirm",
            "sentiment gate",
            "sentiment filter",
            "sentiment block",
            "sentiment on",
        )
        if any(phrase in ql_compact for phrase in phrases):
            return True
        return "headline" in ql_compact and any(word in ql_compact for word in ("long", "short", "buy", "sell"))

    def _sentiment_config_plan_from_phrase(
        self,
        *,
        ql_compact: str,
        pairs: list[str],
    ) -> PlannerPlan | None:
        if not self._looks_like_sentiment_config_request(ql_compact):
            return None
        if not pairs:
            return PlannerPlan(final_answer="Tell me which pair to change and I can update the pair-specific sentiment gate.")

        enabled = not any(phrase in ql_compact for phrase in ("disable", "turn off", "remove", "stop using"))
        directions: list[str] = []
        if any(word in ql_compact for word in ("short only", "shorts only", "short only ", " shorts ", " short ", "sell only", "sells only")):
            directions = ["short"]
        elif any(word in ql_compact for word in ("long only", "longs only", " long ", "buy only", "buys only")):
            directions = ["long"]

        args: dict[str, Any] = {"pair": pairs[0], "enabled": enabled}
        if directions:
            args["directions"] = directions
        return PlannerPlan(steps=[ToolCall("configure_pair_sentiment_gate", args)])

    @staticmethod
    def _looks_like_correlation_sentiment_request(ql_compact: str) -> bool:
        return "sentiment" in ql_compact and "correlation" in ql_compact

    def _correlation_sentiment_plan_from_phrase(
        self,
        *,
        ql_compact: str,
        pairs: list[str],
    ) -> PlannerPlan | None:
        if not self._looks_like_correlation_sentiment_request(ql_compact):
            return None
        if not pairs:
            return PlannerPlan(final_answer="Tell me which master pair to propagate and I can wire the sentiment gate across its correlation group.")
        direction = "short" if "short" in ql_compact or "sell" in ql_compact else "long"
        return PlannerPlan(
            steps=[
                ToolCall(
                    "configure_correlation_sentiment_gate",
                    {"master_pair": pairs[0], "direction": direction, "enabled": True},
                )
            ]
        )

    @staticmethod
    def _looks_like_meta_retrain_request(ql_compact: str) -> bool:
        return ("meta labeler" in ql_compact or "meta-labeler" in ql_compact or "meta_labeler" in ql_compact) and "retrain" in ql_compact

    def _meta_retrain_plan_from_phrase(
        self,
        *,
        ql_compact: str,
        pairs: list[str],
    ) -> PlannerPlan | None:
        if not self._looks_like_meta_retrain_request(ql_compact):
            return None
        args: dict[str, Any] = {"data_source": "journal"}
        if "simulation" in ql_compact:
            args["data_source"] = "simulation"
        elif "backtest" in ql_compact:
            args["data_source"] = "backtest"
        if "all masters" not in ql_compact and pairs:
            args["master_pairs"] = pairs
        return PlannerPlan(steps=[ToolCall("retrain_meta_labelers", args)])

    @staticmethod
    def _looks_like_regime_sizing_request(ql_compact: str) -> bool:
        return (
            ("regime sizing" in ql_compact or "volatility regime sizing" in ql_compact or "aggressive sizing" in ql_compact)
            and any(word in ql_compact for word in ("enable", "turn on", "disable", "tune", "set"))
        )

    def _regime_sizing_plan_from_phrase(self, *, ql_compact: str) -> PlannerPlan | None:
        if not self._looks_like_regime_sizing_request(ql_compact):
            return None
        enabled = not any(phrase in ql_compact for phrase in ("disable", "turn off"))
        args: dict[str, Any] = {"enabled": enabled}
        return PlannerPlan(steps=[ToolCall("configure_aggressive_regime_sizing", args)])

    @staticmethod
    def _scan_profile_from_phrase(ql_compact: str) -> str | None:
        aggressive_hints = (
            "loosen gates",
            "loosen the gates",
            "relax gates",
            "relax the gates",
            "looser gates",
            "less strict gates",
            "make more trades pass",
            "let more trades through",
            "let trades through",
            "more setups pass",
            "lower the thresholds",
            "lower thresholds",
            "aggressive profile",
            "scan aggressive",
        )
        conservative_hints = (
            "tighten gates",
            "tighten the gates",
            "stricter gates",
            "more strict gates",
            "make fewer trades pass",
            "raise the thresholds",
            "raise thresholds",
            "conservative profile",
            "scan conservative",
        )
        balanced_hints = (
            "balanced profile",
            "reset scan profile",
            "reset the scan profile",
            "restore balanced profile",
        )

        if any(hint in ql_compact for hint in aggressive_hints):
            return "aggressive"
        if any(hint in ql_compact for hint in conservative_hints):
            return "conservative"
        if any(hint in ql_compact for hint in balanced_hints):
            return "balanced"
        return None

    def _scan_profile_plan_from_phrase(
        self,
        *,
        ql_compact: str,
        runtime_context: dict[str, Any],
    ) -> PlannerPlan | None:
        profile = self._scan_profile_from_phrase(ql_compact)
        if profile is None:
            return None

        steps: list[ToolCall] = [ToolCall("configure_scan_profile", {"profile": profile})]
        if isinstance(runtime_context.get("last_scan_summary"), dict):
            pairs = self._pairs_from_last_scan(runtime_context)
            granularity = self._scan_granularity_from_context(runtime_context)
            force = runtime_context.get("last_scan_summary", {}).get("session_filter_effective") is False
            steps.append(
                ToolCall(
                    "scan_market",
                    {
                        "pairs": ",".join(pairs) if pairs else None,
                        "granularity": granularity,
                        "top_n": max(5, len(pairs)) if pairs else 5,
                        "auto_execute": False,
                        "force": force,
                        "diversified": False,
                        "run_agent_confirmation": False,
                    },
                )
            )
        return PlannerPlan(steps=steps)

    @staticmethod
    def _looks_like_backtest_request(ql_compact: str) -> bool:
        return "backtest" in ql_compact or "equity curve" in ql_compact

    @staticmethod
    def _extract_basket_aliases(ql_compact: str) -> list[str]:
        hits: list[tuple[int, str]] = []
        known = ("euro", "aussie", "sterling", "yen", "carry", "usd", "kiwi", "commodity", "crosses")
        for alias in known:
            for match in re.finditer(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", ql_compact):
                hits.append((match.start(), alias))
        if "basket" not in ql_compact:
            return []
        hits.sort(key=lambda item: item[0])
        aliases: list[str] = []
        for _idx, alias in hits:
            if alias not in aliases:
                aliases.append(alias)
        return aliases

    def _backtest_plan_from_phrase(
        self,
        *,
        ql_compact: str,
        pairs: list[str],
        runtime_context: dict[str, Any],
    ) -> PlannerPlan | None:
        if not self._looks_like_backtest_request(ql_compact):
            return None

        baskets = self._extract_basket_aliases(ql_compact)
        if not baskets and not pairs:
            return PlannerPlan(final_answer="Tell me which basket or pairs to backtest and I can run it.")

        steps: list[ToolCall] = []
        if self._looks_like_correlation_sentiment_request(ql_compact) and pairs:
            direction = "short" if "short" in ql_compact or "sell" in ql_compact else "long"
            steps.append(
                ToolCall(
                    "configure_correlation_sentiment_gate",
                    {"master_pair": pairs[0], "direction": direction, "enabled": True},
                )
            )
        elif self._looks_like_sentiment_config_request(ql_compact) and pairs:
            direction_list = ["short"] if "short" in ql_compact or "sell" in ql_compact else ["long"] if "long" in ql_compact or "buy" in ql_compact else []
            sentiment_args: dict[str, Any] = {"pair": pairs[0], "enabled": True}
            if direction_list:
                sentiment_args["directions"] = direction_list
            steps.append(ToolCall("configure_pair_sentiment_gate", sentiment_args))

        profile = runtime_context.get("profile", "balanced")
        if "aggressive" in ql_compact:
            profile = "aggressive"
        elif "conservative" in ql_compact:
            profile = "conservative"

        steps.append(
            ToolCall(
                "run_basket_backtest",
                {
                    "baskets": baskets or None,
                    "pairs": None if baskets else ",".join(pairs) if pairs else None,
                    "granularity": runtime_context.get("granularity", "H1"),
                    "profile": profile,
                    "approved_only": True,
                    "force": True,
                },
            )
        )
        return PlannerPlan(steps=steps)

    def _prediction_plan_from_phrase(
        self,
        *,
        original_text: str,
        ql_compact: str,
        runtime_context: dict[str, Any],
        pairs: list[str],
    ) -> PlannerPlan | None:
        if not self._looks_like_prediction_request(ql_compact, has_pair=bool(pairs)):
            ambiguous_aliases = self._find_ambiguous_aliases(original_text)
            if ambiguous_aliases and any(
                phrase in ql_compact for phrase in ("analyze", "check", "look at", "what do you think about", "thoughts on", "read on", "view on")
            ):
                return PlannerPlan(final_answer=self._ambiguous_alias_answer(ambiguous_aliases[0], action="analyze", runtime_context=runtime_context))
            return None

        return PlannerPlan(
            steps=[
                ToolCall(
                    "run_prediction",
                    {
                        "source_kind": "oanda",
                        "value": pairs[0],
                        "granularity": runtime_context.get("granularity", "H1"),
                        "candles": 300,
                    },
                )
            ]
        )

    @staticmethod
    def _looks_like_trade_request(ql_compact: str) -> bool:
        trade_phrases = (
            "buy ",
            "sell ",
            "short ",
            "fade ",
            "flatten ",
            "force buy",
            "force sell",
            "go long",
            "long it",
            "go short",
            "short it",
            "close the trade",
            "close the position",
            "close position",
            "close trade",
        )
        return any(phrase in ql_compact for phrase in trade_phrases)

    @staticmethod
    def _trade_action_from_phrase(ql_compact: str) -> str | None:
        if ql_compact.startswith("force buy"):
            return "force buy"
        if ql_compact.startswith("force sell"):
            return "force sell"
        if ql_compact.startswith("flatten ") or any(phrase in ql_compact for phrase in ("flatten that trade", "flatten that position", "flatten the trade", "flatten the position")):
            return "close"
        if ql_compact.startswith("fade "):
            return "sell"
        if ql_compact.startswith("buy ") or " go long " in f" {ql_compact} " or ql_compact.startswith("go long "):
            return "buy"
        if ql_compact.startswith("sell ") or ql_compact.startswith("short ") or " go short " in f" {ql_compact} ":
            return "sell"
        if ql_compact.startswith("close ") or any(phrase in ql_compact for phrase in ("close the trade", "close that trade", "close the position", "close that position")):
            return "close"
        return None

    @staticmethod
    def _find_ambiguous_aliases(text: str) -> list[str]:
        lower_text = text.lower()
        aliases: list[str] = []
        for alias in _AMBIGUOUS_FX_ALIAS_OPTIONS:
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower_text):
                aliases.append(alias)
        return aliases

    @classmethod
    def _ambiguous_alias_answer(
        cls,
        alias: str,
        *,
        action: str | None = None,
        runtime_context: dict[str, Any],
    ) -> str:
        scan_pairs = []
        for row in cls._scan_rows_from_context(runtime_context, approved_only=False):
            pair = str(row.get("pair") or "").upper().replace("/", "_")
            if pair in _AMBIGUOUS_FX_ALIAS_OPTIONS.get(alias, ()):
                scan_pairs.append(pair)
        options = tuple(dict.fromkeys(scan_pairs + list(_AMBIGUOUS_FX_ALIAS_OPTIONS.get(alias, ()))))
        verb = action or "trade"
        examples = ", ".join(f"'{verb} {pair}'" for pair in options[:3])
        return f"'{alias}' is ambiguous here. Do you mean {', '.join(options[:3])}? Try {examples}."

    @classmethod
    def _resolve_trade_pair(cls, runtime_context: dict[str, Any], pairs: list[str]) -> str | None:
        if pairs:
            return pairs[0]
        active_instrument = str(runtime_context.get("active_instrument") or "").strip().upper().replace("/", "_")
        if active_instrument and active_instrument.count("_") == 1:
            return active_instrument
        return cls._top_scan_pair(runtime_context, approved_only=True)

    def _trade_plan_from_phrase(
        self,
        *,
        original_text: str,
        ql_compact: str,
        runtime_context: dict[str, Any],
        pairs: list[str],
    ) -> PlannerPlan | None:
        action = self._trade_action_from_phrase(ql_compact)
        if action is None:
            return None

        ambiguous_aliases = self._find_ambiguous_aliases(original_text)
        if ambiguous_aliases:
            return PlannerPlan(final_answer=self._ambiguous_alias_answer(ambiguous_aliases[0], action=action, runtime_context=runtime_context))

        pair = self._resolve_trade_pair(runtime_context, pairs)
        if action == "close":
            if pair:
                return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": f"trade close {pair}"})])
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "trade close"})])

        if pair:
            granularity = runtime_context.get("granularity", "H1")
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "use_market_source",
                        {
                            "source_kind": "oanda",
                            "value": pair,
                            "granularity": granularity,
                            "candles": 300,
                        },
                    ),
                    ToolCall("run_trade_command", {"command": action}),
                ]
            )

        if action in {"buy", "sell", "force buy", "force sell"}:
            return PlannerPlan(final_answer="Tell me which pair to trade, or scan first so I can ground the action on a setup.")
        return None

    @staticmethod
    def _looks_like_followup_execute_request(ql_compact: str) -> bool:
        if ql_compact in {
            "do it",
            "do that",
            "go ahead",
            "go ahead then",
            "take it",
            "take that",
            "execute",
            "execute it",
            "execute trade",
            "execute trades",
            "place it",
            "send it",
            "run trade",
            "run trades",
        }:
            return True
        return any(
            ql_compact.startswith(prefix)
            for prefix in (
                "execute ",
                "execute it ",
                "execute trade ",
                "execute trades ",
                "run trade ",
                "run trades ",
                "place it ",
                "send it ",
            )
        )

    @staticmethod
    def _looks_like_followup_reason_request(ql_compact: str) -> bool:
        return ql_compact in {"why", "why that one", "why this one", "why that", "why this"}

    @staticmethod
    def _looks_like_followup_selection_request(ql_compact: str) -> bool:
        return ql_compact in {
            "that one",
            "this one",
            "the top one",
            "the best one",
            "which one",
            "what passed",
            "what passed the scan",
            "top setup",
            "best setup",
        }

    def _contextual_followup_plan(
        self,
        user_message: str,
        *,
        ql: str,
        ql_compact: str,
        runtime_context: dict[str, Any],
        pairs: list[str],
    ) -> PlannerPlan | None:
        has_prediction = bool(runtime_context.get("has_prediction"))
        has_scan = isinstance(runtime_context.get("last_scan_summary"), dict)

        if self._looks_like_followup_execute_request(ql_compact):
            if has_prediction:
                return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "trade"})])
            if has_scan:
                preview_only_followups = {
                    "do it",
                    "do that",
                    "go ahead",
                    "go ahead then",
                    "take it",
                    "take that",
                    "place it",
                    "send it",
                }
                if ql_compact in preview_only_followups:
                    top_pair = pairs[0] if pairs else self._top_scan_pair(runtime_context, approved_only=True)
                    if top_pair:
                        granularity = self._scan_granularity_from_context(runtime_context)
                        return PlannerPlan(
                            steps=[
                                ToolCall(
                                    "run_prediction",
                                    {
                                        "source_kind": "oanda",
                                        "value": top_pair,
                                        "granularity": granularity,
                                        "candles": 300,
                                    },
                                )
                            ]
                        )
                    return PlannerPlan(final_answer="The last scan did not leave me with an approved setup to inspect.")
                if ql_compact in {"execute trades", "run trades"}:
                    requested_pairs = pairs or self._pairs_from_last_scan(runtime_context)
                    granularity = self._scan_granularity_from_context(runtime_context)
                    return PlannerPlan(
                        steps=[
                            ToolCall(
                                "scan_market",
                                {
                                    "pairs": ",".join(requested_pairs) if requested_pairs else None,
                                    "granularity": granularity,
                                    "top_n": max(5, len(requested_pairs)) if requested_pairs else 5,
                                    "auto_execute": True,
                                    "force": False,
                                    "diversified": False,
                                    "run_agent_confirmation": False,
                                },
                            )
                        ]
                    )
                executable_pairs = self._executable_scan_pairs(runtime_context)
                if pairs:
                    requested_pair = pairs[0]
                    if requested_pair not in executable_pairs:
                        return PlannerPlan(
                            final_answer=(
                                f"{requested_pair} was not an approved executable setup in the last scan. "
                                "Ask me to show approved setups or run a fresh scan."
                            )
                        )
                    top_pair = requested_pair
                else:
                    top_pair = executable_pairs[0] if executable_pairs else None
                if top_pair:
                    granularity = self._scan_granularity_from_context(runtime_context)
                    force = runtime_context.get("last_scan_summary", {}).get("session_filter_effective") is False
                    return PlannerPlan(
                        steps=[
                            ToolCall(
                                "scan_market",
                                {
                                    "pairs": top_pair,
                                    "granularity": granularity,
                                    "top_n": 1,
                                    "auto_execute": True,
                                    "force": force,
                                    "diversified": False,
                                    "run_agent_confirmation": False,
                                },
                            )
                        ]
                    )
                return PlannerPlan(final_answer="The last scan did not leave me with an approved setup to act on.")

        if self._looks_like_followup_reason_request(ql_compact):
            if has_prediction:
                return PlannerPlan(steps=[ToolCall("summarize_last_prediction", {"query": user_message})])
            if has_scan:
                if len(pairs) >= 2:
                    return PlannerPlan(steps=[ToolCall("compare_scan_results", {"left_pair": pairs[0], "right_pair": pairs[1]})])
                rows = self._scan_rows_from_context(runtime_context, approved_only=False)
                if len(rows) >= 2:
                    return PlannerPlan(
                        steps=[
                            ToolCall(
                                "compare_scan_results",
                                {
                                    "left_pair": str(rows[0].get("pair") or ""),
                                    "right_pair": str(rows[1].get("pair") or ""),
                                },
                            )
                        ]
                    )
                answer = self._scan_reason_answer(runtime_context)
                if answer:
                    return PlannerPlan(final_answer=answer)

        if self._looks_like_followup_selection_request(ql_compact):
            if has_prediction:
                return PlannerPlan(steps=[ToolCall("summarize_last_prediction", {"query": "summary"})])
            if has_scan:
                answer = self._scan_selection_answer(runtime_context)
                if answer:
                    return PlannerPlan(final_answer=answer)

        if ql_compact in {"close it", "close that", "close that trade", "close that position", "flatten it", "flatten that", "flatten that trade", "flatten that position"}:
            pair = self._resolve_trade_pair(runtime_context, pairs)
            if pair:
                return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": f"trade close {pair}"})])
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "trade close"})])

        return None

    def _deterministic_plan(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
    ) -> PlannerPlan:
        q = user_message.strip()
        ql = q.lower()
        ql_compact = re.sub(r"[^a-z0-9]+", " ", ql).strip()
        runtime_context = runtime_context or {}
        pairs = self._extract_pairs(q)

        if ql in {"hi", "hello", "hey", "buddy", "thanks", "thank you"}:
            return PlannerPlan(final_answer="I'm here. Ask me for status, scans, journal/account actions, or Buddy self-knowledge.")
        if ql_compact in {"self knowledge", "self knowledge please", "buddy self knowledge", "self knowledge mode"}:
            return PlannerPlan(steps=[ToolCall("answer_buddy_self_question", {"query": q})])
        if ql_compact in {"what can you do", "what do you do", "something else", "anything else"}:
            return PlannerPlan(final_answer=self._help_answer())
        if ql_compact == "no":
            return PlannerPlan(final_answer="Then tell me the task directly, or ask for status, a scan, journal/account actions, or Buddy self-knowledge.")
        if any(phrase in ql_compact for phrase in ("what can buddy do", "what can i ask", "what are your commands", "what commands do you have")):
            return PlannerPlan(final_answer=self._help_answer())

        if any(
            phrase in ql_compact
            for phrase in (
                "what s with the error",
                "whats with the error",
                "why the error",
                "why the errors",
                "outside trading session",
                "market closed",
            )
        ):
            scan_error_answer = self._scan_error_followup_answer(runtime_context)
            if scan_error_answer:
                return PlannerPlan(final_answer=scan_error_answer)
        execute_mode_answer = self._execute_mode_followup_answer(ql_compact, runtime_context)
        if execute_mode_answer:
            return PlannerPlan(final_answer=execute_mode_answer)
        contextual_followup = self._contextual_followup_plan(
            q,
            ql=ql,
            ql_compact=ql_compact,
            runtime_context=runtime_context,
            pairs=pairs,
        )
        if contextual_followup is not None:
            return contextual_followup

        pending_override = self._pending_guardrail_override(runtime_context)
        if pending_override is not None and self._looks_like_guardrail_cancel_request(ql_compact):
            cancel_action = str(pending_override.get("action") or "disable")
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "override_guardrail",
                        {
                            "guardrail_name": "session",
                            "action": cancel_action,
                            "scope": "permanent",
                            "confirm_token": "cancel",
                        },
                    )
                ]
            )
        if pending_override is not None and self._looks_like_guardrail_confirm_request(ql_compact):
            token = self._confirmation_token_from_text(q) or str(pending_override.get("confirmation_token") or "")
            confirm_args = {
                "guardrail_name": str(pending_override.get("guardrail_name") or "session"),
                "action": str(pending_override.get("action") or "disable"),
                "scope": "permanent",
                "confirm_token": token,
            }
            pending_window = pending_override.get("session_window_utc")
            if isinstance(pending_window, (list, tuple)) and len(pending_window) == 2:
                confirm_args["window_start_utc"] = int(pending_window[0])
                confirm_args["window_end_utc"] = int(pending_window[1])
            return PlannerPlan(steps=[ToolCall("override_guardrail", confirm_args)])

        override_args = self._guardrail_override_args_from_phrase(
            q,
            ql_compact=ql_compact,
            runtime_context=runtime_context,
        )
        if override_args is not None:
            if (
                self._looks_like_scan_request(q, ql, ql_compact)
                or self._looks_like_opportunity_request(ql_compact)
                or self._looks_like_safe_default_scan_request(ql_compact)
            ):
                return PlannerPlan(
                    steps=[
                        ToolCall("override_guardrail", override_args),
                        ToolCall(
                            "scan_market",
                            self._scan_args_from_request(q, ql, ql_compact, runtime_context, pairs=pairs),
                        ),
                    ]
                )
            return PlannerPlan(steps=[ToolCall("override_guardrail", override_args)])

        if self._looks_like_agent_confirmation_request(ql_compact):
            requested_pairs = pairs or self._pairs_from_last_scan(runtime_context)
            granularity = self._scan_granularity_from_context(runtime_context)
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "scan_market",
                        {
                            "pairs": ",".join(requested_pairs) if requested_pairs else None,
                            "granularity": granularity,
                            "top_n": max(5, len(requested_pairs)) if requested_pairs else 5,
                            "auto_execute": False,
                            "force": False,
                            "diversified": False,
                            "run_agent_confirmation": True,
                        },
                    )
                ]
            )
        if self._looks_like_opportunity_request(ql_compact):
            return PlannerPlan(
                steps=[self._default_scan_tool_call(q, ql, ql_compact, runtime_context, pairs=pairs)]
            )
        if self._looks_like_safe_default_scan_request(ql_compact):
            return PlannerPlan(
                steps=[self._default_scan_tool_call(q, ql, ql_compact, runtime_context, pairs=pairs)]
            )

        if ql in {"summary", "overview"}:
            return PlannerPlan(steps=[ToolCall("summarize_last_prediction", {"query": q})])
        if any(phrase in ql for phrase in ("status --decisions", "show decisions", "last decisions")):
            return PlannerPlan(steps=[ToolCall("get_status"), ToolCall("get_decisions")])
        if any(phrase in ql for phrase in ("model status", "show models", "which models are loaded")):
            return PlannerPlan(steps=[ToolCall("get_model_status")])
        if self._looks_like_account_request(ql_compact) or any(phrase in ql for phrase in ("account", "balance", "margin available", "buying power")):
            return PlannerPlan(steps=[ToolCall("get_account_summary")])
        if self._looks_like_journal_import_request(ql_compact):
            return PlannerPlan(steps=[ToolCall("import_journal_trades")])
        if self._looks_like_journal_sync_request(ql_compact):
            return PlannerPlan(steps=[ToolCall("sync_journal")])
        if self._looks_like_journal_request(ql_compact):
            days_match = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", ql)
            days = int(days_match.group(1)) if days_match else 30
            return PlannerPlan(steps=[ToolCall("get_journal_summary", {"days": days})])
        if self._looks_like_status_request(ql_compact) or "status" in ql or any(phrase in ql for phrase in ("execute mode", "risk per trade", "what provider")):
            return PlannerPlan(steps=[ToolCall("get_status")])
        if ql.startswith("use csv "):
            return PlannerPlan(steps=[ToolCall("use_market_source", {"source_kind": "csv", "value": q[8:].strip()})])
        if ql.startswith("use ticker "):
            return PlannerPlan(steps=[ToolCall("use_market_source", {"source_kind": "ticker", "value": q[11:].strip()})])
        if ql.startswith("use oanda "):
            parts = q.split()
            args: dict[str, Any] = {"source_kind": "oanda"}
            if len(parts) >= 3:
                args["value"] = parts[2]
            if len(parts) >= 4:
                args["granularity"] = parts[3]
            if len(parts) >= 5:
                try:
                    args["candles"] = int(parts[4])
                except ValueError:
                    pass
            return PlannerPlan(steps=[ToolCall("use_market_source", args)])
        if ql.startswith("predict csv "):
            return PlannerPlan(steps=[ToolCall("run_prediction", {"source_kind": "csv", "value": q[12:].strip()})])
        if ql.startswith("predict ticker "):
            return PlannerPlan(steps=[ToolCall("run_prediction", {"source_kind": "ticker", "value": q[15:].strip()})])
        if ql == "predict":
            return PlannerPlan(steps=[ToolCall("run_prediction", {"command": "predict"})])
        prediction_phrase_plan = self._prediction_plan_from_phrase(
            original_text=q,
            ql_compact=ql_compact,
            runtime_context=runtime_context,
            pairs=pairs,
        )
        if prediction_phrase_plan is not None:
            return prediction_phrase_plan
        correlation_sentiment_plan = self._correlation_sentiment_plan_from_phrase(
            ql_compact=ql_compact,
            pairs=pairs,
        )
        if correlation_sentiment_plan is not None:
            return correlation_sentiment_plan
        meta_retrain_plan = self._meta_retrain_plan_from_phrase(
            ql_compact=ql_compact,
            pairs=pairs,
        )
        if meta_retrain_plan is not None:
            return meta_retrain_plan
        scan_profile_plan = self._scan_profile_plan_from_phrase(
            ql_compact=ql_compact,
            runtime_context=runtime_context,
        )
        if scan_profile_plan is not None:
            return scan_profile_plan
        regime_sizing_plan = self._regime_sizing_plan_from_phrase(ql_compact=ql_compact)
        if regime_sizing_plan is not None:
            return regime_sizing_plan
        backtest_plan = self._backtest_plan_from_phrase(
            ql_compact=ql_compact,
            pairs=pairs,
            runtime_context=runtime_context,
        )
        if backtest_plan is not None:
            return backtest_plan
        sentiment_config_plan = self._sentiment_config_plan_from_phrase(
            ql_compact=ql_compact,
            pairs=pairs,
        )
        if sentiment_config_plan is not None:
            return sentiment_config_plan
        if ql.startswith("oanda "):
            return PlannerPlan(steps=[ToolCall("run_oanda_command", {"command": q})])
        trade_phrase_plan = self._trade_plan_from_phrase(
            original_text=q,
            ql_compact=ql_compact,
            runtime_context=runtime_context,
            pairs=pairs,
        )
        if trade_phrase_plan is not None:
            return trade_phrase_plan
        if "close" in ql and any(phrase in ql for phrase in ("position", "trade")):
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "trade close"})])
        if self._looks_like_trade_request(ql_compact) and any(phrase in ql for phrase in ("go long", "long it")):
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "buy"})])
        if self._looks_like_trade_request(ql_compact) and any(phrase in ql for phrase in ("go short", "short it")):
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": "sell"})])
        if ql in {"buy", "sell", "force buy", "force sell"} or ql.startswith("trade"):
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": q})])
        if ql in {"execute on", "execute true", "execute off", "execute false", "risk", "risk settings", "sl/tp", "sltp"} or ql.startswith(("sl ", "tp ", "stoploss ", "stop loss ", "takeprofit ", "take profit ")):
            return PlannerPlan(steps=[ToolCall("run_runtime_command", {"command": q})])

        if ql_compact in {"execute trades", "run trades", "execute approved", "execute setups"}:
            requested_pairs = pairs or self._pairs_from_last_scan(runtime_context)
            granularity = self._scan_granularity_from_context(runtime_context)
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "scan_market",
                        {
                            "pairs": ",".join(requested_pairs) if requested_pairs else None,
                            "granularity": granularity,
                            "top_n": max(5, len(requested_pairs)) if requested_pairs else 5,
                            "auto_execute": True,
                            "force": False,
                            "diversified": False,
                            "run_agent_confirmation": False,
                        },
                    )
                ]
            )

        if ql_compact == "execute":
            return PlannerPlan(final_answer="I can execute after a scan or prediction context. Ask me to scan first, then execute.")

        if self._looks_like_scan_request(q, ql, ql_compact):
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "scan_market",
                        self._scan_args_from_request(q, ql, ql_compact, runtime_context, pairs=pairs),
                    )
                ]
            )
        if len(pairs) >= 2 and ("why" in ql or "compare" in ql or "vs" in ql or "versus" in ql):
            return PlannerPlan(steps=[ToolCall("compare_scan_results", {"left_pair": pairs[0], "right_pair": pairs[1]})])
        if any(phrase in ql for phrase in ("how do you work", "what do you remember", "who are you", "intelligent mode", "what provider")):
            return PlannerPlan(steps=[ToolCall("answer_buddy_self_question", {"query": q})])
        if "approved setups" in ql:
            return PlannerPlan(
                steps=[ToolCall("answer_buddy_self_question", {"query": "Explain that approved scan follow-ups should be answered from the last scan context."})],
                final_answer="Ask me to run a scan first, then I can summarize the approved setups from that last scan.",
            )
        if runtime_context.get("has_prediction"):
            return PlannerPlan(steps=[ToolCall("summarize_last_prediction", {"query": q})])

        return PlannerPlan(
            needs_clarification=True,
            clarification_question=(
                "I didn't get a grounded action from that. Ask me to scan, show status, explain the current mode, "
                "handle journal/account work, or answer a Buddy self-knowledge question."
            ),
        )

    def _synthesize_answer(
        self,
        user_message: str,
        results: list[ToolResult],
        *,
        chat_history: list[dict[str, str]] | None,
        runtime_context: dict[str, Any] | None,
    ) -> str:
        if not results:
            return "I couldn't produce any grounded tool results for that request."
        fallback = "\n".join(result.human_summary for result in results if result.human_summary).strip()
        if not fallback:
            return fallback or "I completed the request."
        local_voice = self._render_local_voice_answer(
            user_message,
            grounded_answer=fallback,
            chat_history=chat_history,
            runtime_context=runtime_context,
            tool_results=results,
        )
        if local_voice and local_voice != fallback:
            return local_voice
        if not (self.provider or os.getenv("BUDDY_PLANNER_USE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}):
            return fallback
        provider = self.provider or select_buddy_provider_name(None)
        if not provider:
            return fallback

        tool_payload = [
            {
                "tool_name": result.tool_name,
                "ok": result.ok,
                "args": result.args,
                "result": result.result,
                "error": result.error,
                "human_summary": result.human_summary,
            }
            for result in results
        ]
        system_prompt = (
            "You are Buddy's grounded response synthesizer. "
            "Answer only from tool outputs and runtime context. "
            "Do not invent values or hidden state."
        )
        prompt = json.dumps(
            {
                "user_message": user_message,
                "runtime_context": runtime_context or {},
                "chat_history": chat_history or [],
                "tool_outputs": tool_payload,
            },
            indent=2,
        )
        try:
            response = llm_call(
                prompt,
                system_prompt=system_prompt,
                provider=provider,
                temperature=0.1,
                max_tokens=400,
            )
            if response and response.strip():
                return response.strip()
        except Exception as e:
            logger.debug("Planner synthesis LLM path failed: %s", e)
        return fallback or "I completed the request."

    @staticmethod
    def _extract_pairs(text: str) -> list[str]:
        pairs: list[str] = []
        seen: set[str] = set()
        patterns = (
            r"(?<![A-Za-z])([A-Za-z]{3})/([A-Za-z]{3})(?![A-Za-z])",
            r"(?<![A-Za-z])([A-Za-z]{3})_([A-Za-z]{3})(?![A-Za-z])",
            r"(?<![A-Za-z])([A-Za-z]{3})-([A-Za-z]{3})(?![A-Za-z])",
            r"(?<![A-Za-z])([A-Za-z]{3})\s+([A-Za-z]{3})(?![A-Za-z])",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                base = match.group(1).upper()
                quote = match.group(2).upper()
                if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
                    pair = f"{base}_{quote}"
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
        if not pairs:
            for match in re.finditer(r"([A-Za-z]{3})/([A-Za-z]{3})", text):
                base = match.group(1).upper()
                quote = match.group(2).upper()
                if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
                    pair = f"{base}_{quote}"
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
        lower_text = text.lower()
        for alias, pair in _FX_PRECISE_PAIR_ALIASES.items():
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower_text):
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs
