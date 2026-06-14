"""Tests for LLM Macro Agent (US-001).

No external API calls in CI — all LLM-dependent paths are mocked or
 exercised with the degraded heuristic fallback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scanner.agents.llm_macro_agent import (
    LLMMacroAgent,
    LLMMacroAgentConfig,
    MacroReasoningOutput,
    PYDANTIC_AVAILABLE,
    ResponseCache,
    _fetch_fred_latest,
)
from src.scanner.agents._team import AgentDecisionContext, AgentVerdict


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def agent_config(tmp_path: Path) -> LLMMacroAgentConfig:
    return LLMMacroAgentConfig(
        cache_dir=tmp_path / "cache",
        shadow_mode=True,
    )


@pytest.fixture
def agent(agent_config: LLMMacroAgentConfig) -> LLMMacroAgent:
    return LLMMacroAgent(agent_config)


@pytest.fixture
def mock_ctx() -> AgentDecisionContext:
    analysis = MagicMock()
    analysis.pair = "EUR_USD"
    analysis.direction = "LONG"
    analysis.current_price = 1.0850
    analysis.volatility_regime = "NORMAL"

    df_raw = pd.DataFrame({
        "open": [1.08, 1.082, 1.084],
        "high": [1.09, 1.092, 1.094],
        "low": [1.07, 1.072, 1.074],
        "close": [1.085, 1.087, 1.089],
        "volume": [1000, 1200, 1100],
    })
    df_feat = pd.DataFrame({
        "sma_20": [1.084],
        "sma_50": [1.082],
        "adx": [25.0],
    })

    return AgentDecisionContext(
        analysis=analysis,
        df_raw=df_raw,
        df_feat=df_feat,
        gate_details={},
        config=MagicMock(),
    )


# ------------------------------------------------------------------
# MacroReasoningOutput schema
# ------------------------------------------------------------------

class TestMacroReasoningOutput:

    def test_valid_parsing(self):
        raw = {
            "regime_shift_probability": 0.3,
            "direction_bias": "LONG",
            "confidence": 0.75,
            "reasoning": "Fed hawkish pause supports USD strength.",
            "key_drivers": ["Fed policy"],
            "data_sources_used": ["FRED"],
        }
        obj = MacroReasoningOutput.model_validate(raw)
        assert obj.regime_shift_probability == 0.3
        assert obj.direction_bias == "LONG"

    def test_model_dump_roundtrip(self):
        raw = {
            "regime_shift_probability": 0.3,
            "direction_bias": "LONG",
            "confidence": 0.75,
            "reasoning": "Fed hawkish pause supports USD strength.",
            "key_drivers": ["Fed policy"],
            "data_sources_used": ["FRED"],
        }
        obj = MacroReasoningOutput.model_validate(raw)
        dumped = obj.model_dump()
        assert dumped["direction_bias"] == "LONG"
        assert dumped["confidence"] == 0.75

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
    def test_invalid_direction_rejected(self):
        with pytest.raises(Exception):
            MacroReasoningOutput.model_validate({
                "regime_shift_probability": 0.5,
                "direction_bias": "BUY",
                "confidence": 0.5,
                "reasoning": "test",
            })

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not installed")
    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(Exception):
            MacroReasoningOutput.model_validate({
                "regime_shift_probability": 0.5,
                "direction_bias": "LONG",
                "confidence": 1.5,
                "reasoning": "test",
            })


# ------------------------------------------------------------------
# ResponseCache
# ------------------------------------------------------------------

class TestResponseCache:

    def test_round_trip(self, tmp_path: Path):
        cache = ResponseCache(tmp_path)
        cache.set("prompt_a", "2026-06-14", {"foo": "bar"})
        got = cache.get("prompt_a", "2026-06-14")
        assert got == {"foo": "bar"}

    def test_miss_returns_none(self, tmp_path: Path):
        cache = ResponseCache(tmp_path)
        assert cache.get("missing", "2026-06-14") is None

    def test_ttl_expiry(self, tmp_path: Path):
        cache = ResponseCache(tmp_path)
        # Manually write an old cache entry
        key = cache._key("old", "2026-06-14")
        path = cache._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        old = {"cached_at": "2020-01-01T00:00:00+00:00", "payload": {"x": 1}}
        path.write_text(json.dumps(old))
        assert cache.get("old", "2026-06-14") is None


# ------------------------------------------------------------------
# LLMMacroAgent — degraded mode (no API keys)
# ------------------------------------------------------------------

class TestLLMMacroAgentDegraded:

    def test_init_no_api_keys(self, agent: LLMMacroAgent):
        assert agent._llm_available is False
        assert agent._client is None

    @patch("src.scanner.agents.llm_macro_agent._fetch_fred_all")
    def test_evaluate_returns_verdict(self, mock_fred, agent: LLMMacroAgent, mock_ctx: AgentDecisionContext):
        mock_fred.return_value = {"EUR": 3.5, "USD": 5.25}
        verdict = agent.evaluate(mock_ctx)
        assert isinstance(verdict, AgentVerdict)
        assert verdict.name == "llm_macro"
        # In degraded mode without LLM, we get fallback — shadow_mode neutralizes
        assert verdict.passed is False
        assert verdict.metadata["shadow_mode"] is True

    @patch("src.scanner.agents.llm_macro_agent._fetch_fred_all")
    def test_evaluate_fallback_reason_code(self, mock_fred, agent: LLMMacroAgent, mock_ctx: AgentDecisionContext):
        mock_fred.return_value = {"EUR": 3.5, "USD": 5.25}
        verdict = agent.evaluate(mock_ctx)
        # When LLM is unavailable and shadow_mode=True, reason_code is llm_unavailable
        # (the fallback) — shadow_mode flag tells consumers this was a shadow vote
        assert verdict.reason_code in ("llm_unavailable", "llm_macro_shadow")
        assert verdict.metadata["shadow_mode"] is True

    def test_build_macro_context_extracts_currencies(self, agent: LLMMacroAgent):
        ctx = agent._build_macro_context("GBP_JPY")
        assert ctx["base_currency"] == "GBP"
        assert ctx["quote_currency"] == "JPY"

    def test_extract_json_strips_markdown(self, agent: LLMMacroAgent):
        text = '```json\n{"a": 1}\n```'
        assert agent._extract_json(text) == '{"a": 1}'

    def test_extract_json_plain_json(self, agent: LLMMacroAgent):
        text = '{"a": 1}'
        assert agent._extract_json(text) == '{"a": 1}'


# ------------------------------------------------------------------
# FRED helpers (network mocked)
# ------------------------------------------------------------------

class TestFREDHelpers:

    @patch("requests.get")
    def test_fetch_fred_latest_success(self, mock_get: MagicMock):
        # Build a CSV with >100 chars to pass the len(r.text) > 100 guard
        rows = ["DATE,IR3TIB01USM156N"]
        for i in range(30):
            rows.append(f"2026-05-{i+1:02d},{4.0 + i * 0.05:.2f}")
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "\n".join(rows)
        val = _fetch_fred_latest("USD")
        assert val is not None
        assert val > 0

    @patch("requests.get")
    def test_fetch_fred_latest_failure(self, mock_get: MagicMock):
        mock_get.return_value.status_code = 404
        val = _fetch_fred_latest("USD")
        assert val is None
