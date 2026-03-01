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
from typing import Any

from llm_providers import llm_call, select_buddy_provider_name

from .planner_types import PlannerPlan, PlannerRuntimeResponse
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry, build_default_tool_registry
from .tool_schema import ToolCall, ToolResult

logger = logging.getLogger(__name__)

_FX_CURRENCIES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are Buddy Planner 3B, the grounded planning and response layer for Buddy. "
    "Be decisive, concise, and action-oriented. "
    "When a user asks for an opportunity, setup, trade idea, or what looks good, default to scanning the market on the current granularity. "
    "Prefer taking the next grounded step over asking meta clarifications. "
    "Clarify only when the request cannot be satisfied safely from the available tools and context. "
    "Never invent trading facts."
)
_REMOTE_PLANNER_SYSTEM_PROMPT = (
    "You are Buddy Planner 3B, the grounded planning and response layer for Buddy. "
    "Be decisive and action-oriented. "
    "If the user asks for an opportunity, setup, trade idea, or what looks good, prefer a market scan on the current granularity. "
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
_DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parents[2] / "trained_data" / "planner" / "checkpoints"
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
    ) -> PlannerRuntimeResponse:
        runtime_state = dict(state or {})
        if runtime_context:
            runtime_state.update(runtime_context)
        prompt_context = self._planner_runtime_context(runtime_context=runtime_context, state=runtime_state)
        plan = self.plan_request(user_message, chat_history=chat_history, runtime_context=prompt_context)
        if plan.needs_clarification:
            answer = plan.clarification_question or "I need a little more detail before I can act."
            answer = self._render_local_voice_answer(
                user_message,
                grounded_answer=answer,
                chat_history=chat_history,
                runtime_context=prompt_context,
                tool_results=[],
            )
            return PlannerRuntimeResponse(plan=plan, tool_results=[], answer=answer)
        if plan.final_answer and not plan.steps:
            answer = self._render_local_voice_answer(
                user_message,
                grounded_answer=plan.final_answer,
                chat_history=chat_history,
                runtime_context=prompt_context,
                tool_results=[],
            )
            return PlannerRuntimeResponse(plan=plan, tool_results=[], answer=answer)

        executor = ToolExecutor(self.registry, state=runtime_state)
        results = executor.execute_plan(plan.steps)
        answer = self._synthesize_answer(
            user_message,
            results,
            chat_history=chat_history,
            runtime_context=prompt_context,
        )
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
        except Exception as e:
            logger.debug("Planner adapter dependencies unavailable: %s", e)
            self._local_adapter_failed = True
            return None

        try:
            adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_name = str(adapter_cfg["base_model_name_or_path"])
            tokenizer = self._load_adapter_tokenizer(adapter_dir=adapter_dir, base_model_name=base_model_name, auto_tokenizer_cls=AutoTokenizer)
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
    def _load_adapter_tokenizer(*, adapter_dir: Path, base_model_name: str, auto_tokenizer_cls: Any) -> Any:
        try:
            return auto_tokenizer_cls.from_pretrained(str(adapter_dir))
        except Exception as e:
            logger.debug("Adapter-local tokenizer load failed, falling back to base tokenizer: %s", e)
            return auto_tokenizer_cls.from_pretrained(base_model_name)

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
        if PlannerRuntime._looks_like_scan_request(user_message, ql, ql_compact):
            return False
        return True

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
            "Try prompts like 'look for an opportunity', 'scan M5', 'show status', or 'compare EUR/USD and GBP/USD'."
        )

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
            "trade idea",
            "look for something",
        )
        if any(phrase in ql_compact for phrase in phrases):
            return True
        if "opportunity" in ql_compact:
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

        if any(phrase in ql_compact for phrase in ("what s with the error", "whats with the error", "why the error", "why the errors", "outside trading session")):
            scan_error_answer = self._scan_error_followup_answer(runtime_context)
            if scan_error_answer:
                return PlannerPlan(final_answer=scan_error_answer)
        execute_mode_answer = self._execute_mode_followup_answer(ql_compact, runtime_context)
        if execute_mode_answer:
            return PlannerPlan(final_answer=execute_mode_answer)
        if self._looks_like_opportunity_request(ql_compact):
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "scan_market",
                        {
                            "pairs": ",".join(pairs) if pairs else None,
                            "granularity": runtime_context.get("granularity", "H1"),
                            "top_n": 5,
                            "auto_execute": False,
                            "force": False,
                            "diversified": False,
                        },
                    )
                ]
            )

        if ql in {"summary", "overview"}:
            return PlannerPlan(steps=[ToolCall("summarize_last_prediction", {"query": q})])
        if any(phrase in ql for phrase in ("status --decisions", "show decisions", "last decisions")):
            return PlannerPlan(steps=[ToolCall("get_status"), ToolCall("get_decisions")])
        if any(phrase in ql for phrase in ("model status", "show models", "which models are loaded")):
            return PlannerPlan(steps=[ToolCall("get_model_status")])
        if "status" in ql or any(phrase in ql for phrase in ("execute mode", "risk per trade", "what provider")):
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
        if ql.startswith("oanda "):
            return PlannerPlan(steps=[ToolCall("run_oanda_command", {"command": q})])
        if ql in {"buy", "sell", "force buy", "force sell"} or ql.startswith("trade"):
            return PlannerPlan(steps=[ToolCall("run_trade_command", {"command": q})])
        if ql in {"execute on", "execute true", "execute off", "execute false", "risk", "risk settings", "sl/tp", "sltp"} or ql.startswith(("sl ", "tp ", "stoploss ", "stop loss ", "takeprofit ", "take profit ")):
            return PlannerPlan(steps=[ToolCall("run_runtime_command", {"command": q})])
        if "journal" in ql and any(phrase in ql for phrase in ("import", "untracked")):
            return PlannerPlan(steps=[ToolCall("import_journal_trades")])
        if "journal" in ql and any(phrase in ql for phrase in ("update", "sync", "refresh")):
            return PlannerPlan(steps=[ToolCall("sync_journal")])
        if any(phrase in ql for phrase in ("journal", "recent trades", "performance", "win rate", "pnl")):
            days_match = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", ql)
            days = int(days_match.group(1)) if days_match else 30
            return PlannerPlan(steps=[ToolCall("get_journal_summary", {"days": days})])
        if any(phrase in ql for phrase in ("account", "nav", "balance", "margin available", "margin")):
            return PlannerPlan(steps=[ToolCall("get_account_summary")])
        if self._looks_like_scan_request(q, ql, ql_compact):
            granularity_match = re.search(r"\b(M1|M5|M15|M30|H1|H4|D|D1)\b", q, flags=re.IGNORECASE)
            top_match = re.search(r"\btop\s+(\d+)\b", ql)
            return PlannerPlan(
                steps=[
                    ToolCall(
                        "scan_market",
                        {
                            "pairs": ",".join(pairs) if pairs else None,
                            "granularity": granularity_match.group(1).upper() if granularity_match else runtime_context.get("granularity", "H1"),
                            "top_n": int(top_match.group(1)) if top_match else 5,
                            "auto_execute": any(phrase in ql for phrase in ("auto-execute", "auto execute", "execute if approved", "execute approved")),
                            "force": "force" in ql,
                            "diversified": "diversified" in ql,
                        },
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
        return pairs
