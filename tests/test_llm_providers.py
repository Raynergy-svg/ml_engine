from __future__ import annotations

from types import SimpleNamespace

import llm_providers


def test_select_buddy_provider_name_prefers_local_ollama(monkeypatch):
    def fake_get_provider(name: str):
        availability = {
            "ollama": True,
            "claude": False,
            "openai": False,
        }
        return SimpleNamespace(name=name, is_available=availability.get(name, False))

    monkeypatch.delenv("BUDDY_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BUDDY_PREFER_LOCAL_LLM", "1")
    monkeypatch.setattr(llm_providers, "get_provider", fake_get_provider)

    assert llm_providers.select_buddy_provider_name(None) == "ollama"


def test_select_buddy_provider_name_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("BUDDY_LLM_PROVIDER", "openai")
    assert llm_providers.select_buddy_provider_name(None) == "openai"
