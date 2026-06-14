"""
LLM Macro Agent — runtime specialist agent for macro reasoning.

US-001: Emits structured AgentVerdicts after querying macro data sources
(FRED policy rates, BIS REER) and reasoning with an LLM.

Shadow mode: when enabled, votes are logged but do NOT affect the ensemble score.
Caching: responses are cached per (prompt_id, date) to avoid API costs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

import pandas as pd

from src.scanner.agents._team import AgentDecisionContext, AgentVerdict, _clip01

logger = logging.getLogger(__name__)

# Optional Pydantic — gracefully degrade if unavailable
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]

# ------------------------------------------------------------------
# Structured output schema
# ------------------------------------------------------------------

class _MacroReasoningSchema:
    """JSON-schema description for macro reasoning output.

    Used when Pydantic is unavailable; the LLM prompt embeds this schema.
    """
    SCHEMA = {
        "type": "object",
        "properties": {
            "regime_shift_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "direction_bias": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string", "minLength": 10},
            "key_drivers": {"type": "array", "items": {"type": "string"}},
            "data_sources_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "regime_shift_probability", "direction_bias", "confidence",
            "reasoning", "key_drivers", "data_sources_used",
        ],
    }


if PYDANTIC_AVAILABLE:
    class MacroReasoningOutput(BaseModel):  # type: ignore[no-redef]
        """Structured output from the LLM macro reasoning step."""

        regime_shift_probability: float = Field(
            ge=0.0, le=1.0,
            description="Probability that a macro regime shift is imminent (0–1).",
        )
        direction_bias: str = Field(
            pattern=r"^(LONG|SHORT|NEUTRAL)$",
            description="Directional bias implied by macro reasoning.",
        )
        confidence: float = Field(
            ge=0.0, le=1.0,
            description="Confidence in the macro reasoning (0–1).",
        )
        reasoning: str = Field(
            min_length=10,
            description="Concise macro reasoning (1–3 sentences).",
        )
        key_drivers: List[str] = Field(
            default_factory=list,
            description="List of key macro drivers cited.",
        )
        data_sources_used: List[str] = Field(
            default_factory=list,
            description="Data sources consulted.",
        )
else:
    class MacroReasoningOutput:  # type: ignore[no-redef]
        """Dict-backed fallback when Pydantic is unavailable."""

        def __init__(self, **kwargs: Any):
            self.regime_shift_probability = float(kwargs.get("regime_shift_probability", 0.0))
            self.direction_bias = str(kwargs.get("direction_bias", "NEUTRAL"))
            self.confidence = float(kwargs.get("confidence", 0.5))
            self.reasoning = str(kwargs.get("reasoning", ""))
            self.key_drivers = list(kwargs.get("key_drivers", []))
            self.data_sources_used = list(kwargs.get("data_sources_used", []))

        @classmethod
        def model_validate(cls, obj: Dict[str, Any]) -> "MacroReasoningOutput":
            return cls(**obj)

        @classmethod
        def model_validate_json(cls, text: str) -> "MacroReasoningOutput":
            return cls(**json.loads(text))

        def model_dump(self) -> Dict[str, Any]:
            return {
                "regime_shift_probability": self.regime_shift_probability,
                "direction_bias": self.direction_bias,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
                "key_drivers": self.key_drivers,
                "data_sources_used": self.data_sources_used,
            }


# ------------------------------------------------------------------
# FRED / BIS helpers (reusing existing factor-layer patterns)
# ------------------------------------------------------------------

_FRED_RATE_IDS: Dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
}


def _fetch_fred_latest(ccy: str) -> Optional[float]:
    """Fetch the latest OECD 3-month interbank rate for a currency from FRED."""
    if requests is None:
        return None
    fid = _FRED_RATE_IDS.get(ccy)
    if fid is None:
        return None
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.text) > 100:
            from io import StringIO

            df = pd.read_csv(StringIO(r.text))
            df.columns = ["date", "rate"]
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
            s = df.dropna().set_index("date")["rate"].sort_index()
            if s.empty:
                return None
            return float(s.iloc[-1])
    except Exception as exc:
        logger.warning("FRED fetch failed for %s (%s): %s", ccy, fid, exc)
    return None


def _fetch_fred_all() -> Dict[str, float]:
    """Fetch latest rates for all tracked currencies."""
    results: Dict[str, float] = {}
    for ccy in _FRED_RATE_IDS:
        val = _fetch_fred_latest(ccy)
        if val is not None:
            results[ccy] = val
    return results


# ------------------------------------------------------------------
# Cache layer
# ------------------------------------------------------------------

class ResponseCache:
    """Simple JSON file cache for LLM macro responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt_id: str, date_str: str) -> str:
        raw = f"{prompt_id}:{date_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prompt_id: str, date_str: str) -> Optional[Dict[str, Any]]:
        path = self._path(self._key(prompt_id, date_str))
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # TTL: invalidate if older than 24h
            cached_at = datetime.fromisoformat(data.get("cached_at", "1970-01-01"))
            if (datetime.now(timezone.utc) - cached_at).total_seconds() > 86400:
                return None
            return data.get("payload")
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", prompt_id, exc)
            return None

    def set(self, prompt_id: str, date_str: str, payload: Dict[str, Any]) -> None:
        path = self._path(self._key(prompt_id, date_str))
        try:
            data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", prompt_id, exc)


# ------------------------------------------------------------------
# LLM Macro Agent
# ------------------------------------------------------------------

@dataclass
class LLMMacroAgentConfig:
    """Configuration for the LLM Macro Agent."""

    base_weight: float = 0.95
    score_threshold: float = 0.55
    block_threshold: float = 0.30
    # LLM settings
    model_name: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.2
    # Cache
    cache_dir: Path = field(default_factory=lambda: Path(".buddy_cache/llm_macro"))
    # Shadow mode
    shadow_mode: bool = True
    # Data
    enable_fred: bool = True
    enable_bis_reer: bool = False  # staged — requires BIS data ingestion


class LLMMacroAgent:
    """Runtime specialist agent that queries macro data and reasons with an LLM.

    Emits an AgentVerdict that can participate in the ScannerAgentTeam ensemble.
    When ``shadow_mode=True``, the verdict is logged but does not affect the vote.
    """

    name = "llm_macro"

    def __init__(self, config: Optional[LLMMacroAgentConfig] = None):
        self.cfg = config or LLMMacroAgentConfig()
        self._cache = ResponseCache(self.cfg.cache_dir)
        self._llm_available = self._check_llm()

    # ------------------------------------------------------------------
    # LLM availability
    # ------------------------------------------------------------------
    def _check_llm(self) -> bool:
        """Return True if an LLM client can be instantiated."""
        # Prefer Anthropic Claude
        if os.getenv(self.cfg.api_key_env):
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic()
                logger.info("LLMMacroAgent: using Anthropic Claude")
                return True
            except Exception as exc:
                logger.debug("Anthropic init failed: %s", exc)
        # Fallback: OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai  # type: ignore

                self._client = openai.OpenAI()
                logger.info("LLMMacroAgent: using OpenAI")
                return True
            except Exception as exc:
                logger.debug("OpenAI init failed: %s", exc)
        logger.warning(
            "LLMMacroAgent: no LLM credentials found (%s or OPENAI_API_KEY). "
            "Running in degraded mode (heuristic macro reasoning).",
            self.cfg.api_key_env,
        )
        self._client = None
        return False

    # ------------------------------------------------------------------
    # Public API — matches ScannerAgentTeam agent pattern
    # ------------------------------------------------------------------
    def evaluate(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Emit an AgentVerdict using macro reasoning."""
        pair = getattr(ctx.analysis, "pair", "UNKNOWN")
        direction = getattr(ctx.analysis, "direction", "HOLD")

        # Build macro context
        macro_context = self._build_macro_context(pair)

        # Try cache first
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"{pair}:{direction}:{hash(str(macro_context))}"
        cached = self._cache.get(cache_key, date_str)
        if cached:
            logger.debug("LLMMacroAgent: cache hit for %s", pair)
            reasoning = MacroReasoningOutput(**cached)
        else:
            reasoning = self._query_llm(macro_context, pair, direction)
            if reasoning:
                self._cache.set(cache_key, date_str, reasoning.model_dump())

        if reasoning is None:
            return self._fallback_verdict(pair, "llm_unavailable")

        score = _clip01(reasoning.confidence)
        passed = score >= self.cfg.score_threshold
        block_trade = score < self.cfg.block_threshold
        confidence_delta = (score - 0.5) * 0.10

        # Shadow mode: always log, but neutralize the vote
        if self.cfg.shadow_mode:
            effective_passed = False
            effective_block = False
            reason = (
                f"[SHADOW] {reasoning.reasoning} — "
                f"regime_shift={reasoning.regime_shift_probability:.2f}, "
                f"bias={reasoning.direction_bias}"
            )
            reason_code = "llm_macro_shadow"
        else:
            effective_passed = passed and reasoning.direction_bias == direction
            effective_block = block_trade
            reason = (
                f"{reasoning.reasoning} — "
                f"regime_shift={reasoning.regime_shift_probability:.2f}, "
                f"bias={reasoning.direction_bias}"
            )
            reason_code = "llm_macro_support" if effective_passed else "llm_macro_neutral"

        metadata: Dict[str, Any] = {
            "regime_shift_probability": reasoning.regime_shift_probability,
            "direction_bias": reasoning.direction_bias,
            "key_drivers": reasoning.key_drivers,
            "data_sources_used": reasoning.data_sources_used,
            "shadow_mode": self.cfg.shadow_mode,
            "llm_available": self._llm_available,
        }

        return AgentVerdict(
            name=self.name,
            score=score,
            passed=effective_passed,
            weight=self.cfg.base_weight,
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=effective_block,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_macro_context(self, pair: str) -> Dict[str, Any]:
        """Gather macro data for the LLM prompt."""
        ctx: Dict[str, Any] = {
            "pair": pair,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Extract base and quote currencies
        if "_" in pair:
            base, quote = pair.split("_", 1)
        else:
            base = quote = "UNKNOWN"

        ctx["base_currency"] = base
        ctx["quote_currency"] = quote

        # FRED rates
        if self.cfg.enable_fred:
            rates = _fetch_fred_all()
            ctx["fred_rates"] = rates
            if base in rates and quote in rates:
                ctx["rate_differential_bps"] = (rates[base] - rates[quote]) * 100

        return ctx

    def _query_llm(
        self,
        macro_context: Dict[str, Any],
        pair: str,
        direction: str,
    ) -> Optional[MacroReasoningOutput]:
        """Query the LLM with macro context and parse structured output."""
        if not self._llm_available or self._client is None:
            return None

        system_prompt = (
            "You are a senior macro strategist at a systematic FX hedge fund. "
            "Given macroeconomic data (policy rates, carry differentials), "
            "reason about the directional bias for a currency pair and the "
            "probability of a near-term regime shift."
            "\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "regime_shift_probability": float [0–1],\n'
            '  "direction_bias": "LONG" | "SHORT" | "NEUTRAL",\n'
            '  "confidence": float [0–1],\n'
            '  "reasoning": string (1–3 sentences),\n'
            '  "key_drivers": [string, ...],\n'
            '  "data_sources_used": [string, ...]\n'
            "}"
        )

        user_prompt = (
            f"Currency pair: {pair}\n"
            f"Proposed direction: {direction}\n"
            f"Macro context:\n{json.dumps(macro_context, indent=2, default=str)}\n\n"
            f"Provide your structured macro assessment."
        )

        try:
            if hasattr(self._client, "messages"):
                # Anthropic
                resp = self._client.messages.create(
                    model=self.cfg.model_name,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw = resp.content[0].text if resp.content else ""
            elif hasattr(self._client, "chat"):
                # OpenAI
                resp = self._client.chat.completions.create(
                    model=self.cfg.model_name.replace("claude-", "gpt-4"),
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = resp.choices[0].message.content or ""
            else:
                raw = ""

            # Extract JSON from response
            raw_json = self._extract_json(raw)
            if raw_json:
                return MacroReasoningOutput.model_validate_json(raw_json)
        except Exception as exc:
            logger.warning("LLM query failed for %s: %s", pair, exc)

        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from an LLM response that may contain markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Find first line after opening fence
            start = 1 if lines[0].startswith("```json") else 0
            # Find closing fence
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()
        return text

    def _fallback_verdict(self, pair: str, reason_code: str) -> AgentVerdict:
        return AgentVerdict(
            name=self.name,
            score=0.5,
            passed=False,
            weight=self.cfg.base_weight,
            reason=f"LLM Macro Agent fallback — {reason_code}",
            reason_code=reason_code,
            confidence_delta=0.0,
            block_trade=False,
            metadata={"error": reason_code, "shadow_mode": self.cfg.shadow_mode},
        )
