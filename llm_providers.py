"""
LLM Providers - Unified interface for multiple LLM backends.

Supports:
- Claude (API, default) - claude-3-sonnet, claude-3-haiku via Anthropic API
- OpenAI (API) - gpt-4, gpt-3.5-turbo
- Ollama (local fallback) - llama3.1:7b, mistral:7b, etc.

Provider selection:
1. If ANTHROPIC_API_KEY env var is set → Claude (recommended)
2. Else if OPENAI_API_KEY env var is set → OpenAI
3. Otherwise → Ollama (local, may be slow)
4. CLI override: --llm-provider claude|openai|ollama

Usage:
    from llm_providers import get_llm_provider, llm_call
    
    # Auto-select based on env vars
    provider = get_llm_provider()
    response = provider.call("What is 2+2?", system_prompt="Be concise")
    
    # Or use unified call interface
    response = llm_call("What is 2+2?")
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Load .env file to get API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

logger = logging.getLogger(__name__)


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:7b"
    ollama_fallback_models: tuple = ("mistral:7b", "llama2:7b", "gemma:7b")
    ollama_timeout: int = 120
    
    # Claude settings - use latest model names (2025)
    claude_default_model: str = "claude-sonnet-4-20250514"  # Best quality-speed balance
    claude_fast_model: str = "claude-haiku-4-20250514"  # Faster, cheaper
    claude_max_tokens: int = 4096
    
    # General settings
    default_temperature: float = 0.2
    max_retries: int = 3
    retry_delay: float = 1.0


# Global config instance
_config = LLMConfig()


def configure_llm(
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    claude_model: str | None = None,
    temperature: float | None = None,
) -> None:
    """Configure LLM provider settings."""
    global _config
    if ollama_base_url:
        _config.ollama_base_url = ollama_base_url
    if ollama_model:
        _config.ollama_default_model = ollama_model
    if claude_model:
        _config.claude_default_model = claude_model
    if temperature is not None:
        _config.default_temperature = temperature


# =============================================================================
# BASE PROVIDER CLASS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (API key set, server running, etc.)."""
        pass
    
    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Make an LLM call.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Model override (optional)
            temperature: Temperature override (optional)
            max_tokens: Max tokens override (optional)
            
        Returns:
            Response text or None on failure
        """
        pass
    
    def call_with_retry(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
    ) -> str | None:
        """Call with automatic retry on failure."""
        retries = max_retries or _config.max_retries
        
        for attempt in range(retries):
            try:
                result = self.call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"{self.name} call failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(_config.retry_delay * (attempt + 1))
        
        return None


# =============================================================================
# OLLAMA PROVIDER (Local, Default)
# =============================================================================

class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference.
    
    Supports llama3.1:7b, mistral:7b, and other Ollama models.
    Requires Ollama running locally (ollama serve).
    """
    
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._base_url = base_url or _config.ollama_base_url
        self._model = model or _config.ollama_default_model
        self._available: bool | None = None
        self._detected_model: str | None = None
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama server is running and has models."""
        if self._available is not None:
            return self._available
        
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Try to find best available model
                for candidate in [self._model] + list(_config.ollama_fallback_models):
                    # Check exact match or prefix match (e.g., "llama3.1:7b" matches "llama3.1:7b-instruct-q4_0")
                    for m in models:
                        if m == candidate or m.startswith(candidate.split(":")[0]):
                            self._detected_model = m
                            self._available = True
                            logger.info(f"Ollama available with model: {self._detected_model}")
                            return True
                
                # No matching model found
                if models:
                    logger.warning(f"Ollama running but no compatible model. Available: {models}")
                    # Use first available model as fallback
                    self._detected_model = models[0]
                    self._available = True
                    return True
                else:
                    logger.warning("Ollama running but no models installed. Run: ollama pull llama3.1:7b")
                    self._available = False
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
        
        return self._available or False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            import requests
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Call Ollama API."""
        if not self.is_available:
            logger.warning("Ollama not available")
            return None
        
        try:
            import requests
            
            # Use detected model or specified model
            use_model = model or self._detected_model or self._model
            use_temp = temperature if temperature is not None else _config.default_temperature
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Ollama chat API
            payload = {
                "model": use_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": use_temp,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=_config.ollama_timeout,
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return content.strip() if content else None
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None


# =============================================================================
# CLAUDE PROVIDER (Anthropic API)
# =============================================================================

class ClaudeProvider(LLMProvider):
    """Claude provider via Anthropic API.
    
    Requires ANTHROPIC_API_KEY environment variable.
    Uses claude-3-5-sonnet by default (best quality-speed balance).
    """
    
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model or _config.claude_default_model
        self._client: Any = None
    
    @property
    def name(self) -> str:
        return "claude"
    
    @property
    def is_available(self) -> bool:
        """Check if Anthropic API key is set."""
        return bool(self._api_key)
    
    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                return None
        return self._client
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Call Claude API."""
        if not self.is_available:
            logger.warning("Claude not available (ANTHROPIC_API_KEY not set)")
            return None
        
        client = self._get_client()
        if not client:
            return None
        
        try:
            use_model = model or self._model
            use_temp = temperature if temperature is not None else _config.default_temperature
            use_max_tokens = max_tokens or _config.claude_max_tokens
            
            # Build message
            messages = [{"role": "user", "content": prompt}]
            
            # Claude API call
            kwargs = {
                "model": use_model,
                "max_tokens": use_max_tokens,
                "messages": messages,
                "temperature": use_temp,
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = client.messages.create(**kwargs)
            
            # Extract text from response
            if response.content:
                content = response.content[0]
                if hasattr(content, "text"):
                    return content.text.strip()
            
            return None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Claude call failed: {e}")
            
            # Detect billing/credit issues - mark provider as unavailable
            if "credit balance" in error_msg.lower() or "billing" in error_msg.lower():
                logger.error("Claude API billing issue - please add credits at https://console.anthropic.com/settings/plans")
                self._billing_error = True
            
            return None
    
    @property
    def has_billing_error(self) -> bool:
        """Check if provider has billing issues."""
        return getattr(self, "_billing_error", False)


# =============================================================================
# OPENAI PROVIDER (Fallback for existing code)
# =============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4, etc.) - for backward compatibility."""
    
    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._client: Any = None
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                logger.error("openai package not installed")
                return None
        return self._client
    
    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        if not self.is_available:
            return None
        
        client = self._get_client()
        if not client:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                temperature=temperature or _config.default_temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return None


# =============================================================================
# PROVIDER FACTORY & GLOBAL INTERFACE
# =============================================================================

# Cache for provider instances
_providers: Dict[str, LLMProvider] = {}
_active_provider: LLMProvider | None = None


def get_provider(name: str) -> LLMProvider:
    """Get a specific provider by name."""
    if name not in _providers:
        if name == "ollama":
            _providers[name] = OllamaProvider()
        elif name == "claude":
            _providers[name] = ClaudeProvider()
        elif name == "openai":
            _providers[name] = OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")
    return _providers[name]


def get_llm_provider(prefer: str | None = None) -> LLMProvider:
    """Get the best available LLM provider.
    
    Selection order:
    1. If prefer is specified, use that provider
    2. If ANTHROPIC_API_KEY is set → Claude
    3. Otherwise → Ollama (local)
    4. Fallback → OpenAI (if OPENAI_API_KEY set)
    
    Args:
        prefer: Preferred provider name ("ollama", "claude", "openai")
        
    Returns:
        Best available LLM provider
    """
    global _active_provider
    
    # If prefer specified, try that first
    if prefer:
        provider = get_provider(prefer)
        if provider.is_available:
            _active_provider = provider
            logger.info(f"Using preferred LLM provider: {provider.name}")
            return provider
        logger.warning(f"Preferred provider '{prefer}' not available, trying alternatives")
    
    # Auto-selection logic: Claude first (fast cloud API), then OpenAI, then Ollama
    # Claude is the default when ANTHROPIC_API_KEY is set
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = get_provider("claude")
        if claude.is_available:
            _active_provider = claude
            logger.info("Auto-selected Claude (ANTHROPIC_API_KEY detected)")
            return claude
    
    # Fallback to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        openai = get_provider("openai")
        if openai.is_available:
            _active_provider = openai
            logger.info("Fallback to OpenAI")
            return openai
    
    # Last resort: Ollama (local, but can be slow)
    ollama = get_provider("ollama")
    if ollama.is_available:
        _active_provider = ollama
        logger.info("Fallback to Ollama (local) - may be slow")
        return ollama
    
    # No provider available - return Claude anyway (will fail gracefully with clear message)
    logger.warning("No LLM provider available. Set ANTHROPIC_API_KEY for best experience")
    return get_provider("claude")


def set_active_provider(provider: str | LLMProvider) -> None:
    """Set the active LLM provider."""
    global _active_provider
    if isinstance(provider, str):
        _active_provider = get_provider(provider)
    else:
        _active_provider = provider
    logger.info(f"Active LLM provider set to: {_active_provider.name}")


def get_active_provider() -> LLMProvider:
    """Get the currently active provider."""
    global _active_provider
    if _active_provider is None:
        _active_provider = get_llm_provider()
    return _active_provider


# =============================================================================
# UNIFIED CALL INTERFACE
# =============================================================================

def llm_call(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
) -> str | None:
    """Unified LLM call interface.
    
    Uses the active provider (auto-selected or explicitly set).
    
    Args:
        prompt: User prompt
        system_prompt: System prompt (optional)
        model: Model override (optional, provider-specific)
        temperature: Temperature override (optional)
        max_tokens: Max tokens override (optional)
        provider: Provider override ("ollama", "claude", "openai")
        
    Returns:
        Response text or None on failure
    """
    if provider:
        p = get_provider(provider)
    else:
        p = get_active_provider()
    
    return p.call_with_retry(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# =============================================================================
# BUDDY INTELLIGENT MODE INTEGRATION
# =============================================================================

def create_buddy_llm_call() -> Callable[..., str | None]:
    """Create an LLM call function compatible with buddy_intelligent_mode.
    
    This function creates a wrapper that can be passed to
    buddy_intelligent_mode.set_llm_call_function().
    
    Returns:
        Function compatible with buddy_intelligent_mode's LLM call signature
    """
    def buddy_compatible_call(
        prompt: str,
        system_prompt: str | None = None,
        model: str = "gpt-4",  # Ignored, uses active provider's model
        temperature: float = 0.2,
    ) -> str | None:
        """LLM call compatible with buddy_intelligent_mode signature."""
        return llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    
    return buddy_compatible_call


def initialize_buddy_llm(provider: str | None = None) -> LLMProvider:
    """Initialize LLM for buddy intelligent mode.
    
    Sets up the provider and registers it with buddy_intelligent_mode.
    
    Args:
        provider: Preferred provider ("ollama", "claude", "openai")
        
    Returns:
        The initialized provider
    """
    # Get provider
    p = get_llm_provider(prefer=provider)
    
    # Register with buddy_intelligent_mode
    try:
        from buddy_intelligent_mode import set_llm_call_function
        set_llm_call_function(create_buddy_llm_call())
        logger.info(f"Buddy intelligent mode LLM initialized with {p.name}")
    except ImportError:
        logger.warning("buddy_intelligent_mode not available")
    
    return p


# =============================================================================
# CLI HELPER
# =============================================================================

def check_provider_status() -> Dict[str, Any]:
    """Check status of all LLM providers.
    
    Returns:
        Dict with status of each provider
    """
    status = {}
    
    # Check Ollama
    ollama = get_provider("ollama")
    status["ollama"] = {
        "available": ollama.is_available,
        "models": ollama.get_available_models() if isinstance(ollama, OllamaProvider) else [],
    }
    
    # Check Claude
    claude = get_provider("claude")
    status["claude"] = {
        "available": claude.is_available,
        "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    
    # Check OpenAI
    openai = get_provider("openai")
    status["openai"] = {
        "available": openai.is_available,
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
    }
    
    # Current selection
    active = get_active_provider()
    status["active"] = active.name if active else None
    
    return status


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Checking LLM provider status...")
    status = check_provider_status()
    
    for name, info in status.items():
        if name == "active":
            print(f"\nActive provider: {info}")
        else:
            available = "✓" if info.get("available") else "✗"
            print(f"  {name}: {available}")
            if name == "ollama" and info.get("models"):
                print(f"    Models: {', '.join(info['models'][:5])}")
    
    # Test call if any provider available
    provider = get_llm_provider()
    if provider.is_available:
        print(f"\nTesting {provider.name}...")
        response = llm_call("What is 2+2? Answer with just the number.")
        print(f"Response: {response}")
    else:
        print("\nNo LLM provider available!")
        print("Options:")
        print("  1. Install Ollama: brew install ollama && ollama pull llama3.1:7b")
        print("  2. Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)
