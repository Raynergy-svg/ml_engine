"""Cloud Fallback Synthesizer — LLM-assisted pattern analysis.

PRD v2.2 §7.1 Phase 3: "Cloud fallback for complex pattern synthesis"

When local statistical methods (T1 frequency, T2 correlation) detect patterns
but can't explain them or have low confidence, this module optionally delegates
to an LLM for deeper narrative synthesis.

Design principles:
1. FULLY OPTIONAL — system works without any LLM provider configured
2. GRACEFUL DEGRADATION — returns local-only results when LLM unavailable
3. RATE-LIMITED — max N calls per day to control costs
4. EVIDENCE-BASED — LLM receives structured evidence, not raw data
5. AUDITABLE — every synthesis is logged with source evidence and model used

The synthesizer does NOT replace local analysis. It adds a narrative layer:
- "Why might stress correlate with lower win rates?"
- "What behavioral arc connects these 3 patterns?"
- "Given these override patterns, what specific trigger is likely?"

Usage:
    synth = CloudPatternSynthesizer()
    if synth.is_available():
        result = synth.synthesize_pattern_explanation(pattern, evidence)
    else:
        # Continue with local-only analysis
        pass
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rate limiting defaults
DEFAULT_MAX_CALLS_PER_DAY = 20
DEFAULT_MAX_TOKENS_PER_CALL = 500
DEFAULT_COOLDOWN_SECONDS = 60  # Min time between calls


@dataclass
class SynthesisResult:
    """Result from a cloud synthesis attempt."""

    success: bool
    narrative: str = ""                     # The LLM-generated explanation
    suggested_actions: List[str] = field(default_factory=list)
    confidence_boost: float = 0.0           # How much to increase pattern confidence
    source_model: str = ""                  # Which model was used
    tokens_used: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False             # True if local fallback was used
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "narrative": self.narrative,
            "suggested_actions": self.suggested_actions,
            "confidence_boost": round(self.confidence_boost, 3),
            "source_model": self.source_model,
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 1),
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


@dataclass
class SynthesisLog:
    """Audit log entry for a synthesis request."""

    timestamp: str
    pattern_id: str
    model_used: str
    tokens_used: int
    latency_ms: float
    success: bool
    error: str = ""


class CloudPatternSynthesizer:
    """Optional LLM-powered pattern synthesis layer.

    Reads LLM configuration from environment variables:
      - AURA_LLM_PROVIDER: "openai" | "anthropic" | "local" | unset
      - AURA_LLM_API_KEY: API key for the provider
      - AURA_LLM_MODEL: Model name (default varies by provider)
      - AURA_LLM_MAX_DAILY_CALLS: Rate limit (default 20)

    When no provider is configured, all methods gracefully return
    local-only results with fallback_used=True.

    Args:
        log_dir: Directory for synthesis audit logs
        max_daily_calls: Maximum LLM calls per day
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_daily_calls: Optional[int] = None,
    ):
        self.log_dir = log_dir or Path(".aura/synthesis_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Read config from environment
        self._provider = os.environ.get("AURA_LLM_PROVIDER", "").lower().strip()
        self._api_key = os.environ.get("AURA_LLM_API_KEY", "").strip()
        self._model = os.environ.get("AURA_LLM_MODEL", "").strip()
        self._max_daily_calls = max_daily_calls or int(
            os.environ.get("AURA_LLM_MAX_DAILY_CALLS", str(DEFAULT_MAX_CALLS_PER_DAY))
        )

        # Set model defaults per provider
        if self._provider == "openai" and not self._model:
            self._model = "gpt-4o-mini"
        elif self._provider == "anthropic" and not self._model:
            self._model = "claude-3-haiku-20240307"

        # Rate limiting state
        self._daily_call_count = 0
        self._daily_reset_date = ""
        self._last_call_time = 0.0

        # Synthesis history for audit
        self._log_path = self.log_dir / "synthesis_log.jsonl"

        self._load_daily_count()

    # --- Public API ---

    def is_available(self) -> bool:
        """Check if cloud synthesis is configured and within rate limits."""
        if not self._provider or not self._api_key:
            return False
        self._check_daily_reset()
        return self._daily_call_count < self._max_daily_calls

    def get_status(self) -> Dict[str, Any]:
        """Get synthesizer status for CLI/orchestrator display."""
        self._check_daily_reset()
        return {
            "configured": bool(self._provider and self._api_key),
            "provider": self._provider or "none",
            "model": self._model or "none",
            "available": self.is_available(),
            "daily_calls_used": self._daily_call_count,
            "daily_calls_max": self._max_daily_calls,
            "daily_calls_remaining": max(
                0, self._max_daily_calls - self._daily_call_count
            ),
        }

    def synthesize_pattern_explanation(
        self,
        pattern_description: str,
        evidence_items: List[Dict[str, Any]],
        domain_context: str = "forex trading",
    ) -> SynthesisResult:
        """Ask LLM to explain why a detected pattern exists.

        This is the primary synthesis method. Given a pattern description
        and its supporting evidence, the LLM generates:
        1. A narrative explanation of WHY the pattern exists
        2. Suggested concrete actions
        3. A confidence assessment

        Args:
            pattern_description: Human-readable pattern description
            evidence_items: List of evidence dicts from DetectedPattern
            domain_context: What domain we're operating in

        Returns:
            SynthesisResult with narrative, actions, and metadata
        """
        if not self.is_available():
            return self._local_fallback_explanation(
                pattern_description, evidence_items
            )

        prompt = self._build_explanation_prompt(
            pattern_description, evidence_items, domain_context
        )

        return self._call_llm(
            prompt=prompt,
            pattern_id=f"explain_{hash(pattern_description) % 10000:04d}",
            parse_fn=self._parse_explanation_response,
        )

    def synthesize_pattern_connections(
        self,
        patterns: List[Dict[str, Any]],
        domain_context: str = "forex trading",
    ) -> SynthesisResult:
        """Ask LLM to find connections between multiple detected patterns.

        When T1/T2 detect several patterns independently, this method
        asks the LLM to synthesize them into a coherent behavioral narrative.

        Args:
            patterns: List of pattern dicts (from DetectedPattern.to_dict())
            domain_context: Operating domain

        Returns:
            SynthesisResult with cross-pattern narrative
        """
        if not self.is_available():
            return self._local_fallback_connections(patterns)

        prompt = self._build_connections_prompt(patterns, domain_context)

        return self._call_llm(
            prompt=prompt,
            pattern_id=f"connect_{len(patterns)}patterns",
            parse_fn=self._parse_connections_response,
        )

    def synthesize_override_risk_narrative(
        self,
        override_context: Dict[str, Any],
        recent_patterns: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """Generate a real-time risk narrative for an active override situation.

        Called when the trader is about to override Buddy's recommendation.
        Uses pattern history to provide personalized risk context.

        Args:
            override_context: Current override details (pair, confidence, emotion, etc.)
            recent_patterns: Recent relevant patterns for context

        Returns:
            SynthesisResult with risk narrative and specific warnings
        """
        if not self.is_available():
            return self._local_fallback_override_risk(
                override_context, recent_patterns
            )

        prompt = self._build_override_risk_prompt(
            override_context, recent_patterns
        )

        return self._call_llm(
            prompt=prompt,
            pattern_id="override_risk",
            parse_fn=self._parse_override_risk_response,
        )

    # --- Prompt Builders ---

    def _build_explanation_prompt(
        self,
        pattern_description: str,
        evidence_items: List[Dict[str, Any]],
        domain_context: str,
    ) -> str:
        """Build a structured prompt for pattern explanation."""
        evidence_text = "\n".join(
            f"  - [{e.get('source_type', 'unknown')}] {e.get('summary', '')}"
            for e in evidence_items[:10]  # Cap at 10 to limit tokens
        )

        return f"""You are a behavioral analyst specializing in {domain_context} psychology.

A pattern detection system has identified the following pattern:

PATTERN: {pattern_description}

SUPPORTING EVIDENCE:
{evidence_text}

Please provide:
1. EXPLANATION: A concise psychological/behavioral explanation for WHY this pattern exists (2-3 sentences)
2. ACTIONS: 2-3 specific, actionable steps the trader should take
3. CONFIDENCE: How confident are you in this explanation? (low/medium/high)

Format your response as JSON:
{{"explanation": "...", "actions": ["...", "..."], "confidence": "medium"}}"""

    def _build_connections_prompt(
        self,
        patterns: List[Dict[str, Any]],
        domain_context: str,
    ) -> str:
        """Build prompt for cross-pattern connection synthesis."""
        pattern_text = "\n".join(
            f"  {i+1}. {p.get('description', 'Unknown pattern')} "
            f"(confidence: {p.get('confidence', 0):.0%})"
            for i, p in enumerate(patterns[:8])  # Cap at 8
        )

        return f"""You are a behavioral analyst specializing in {domain_context} psychology.

Multiple independent patterns have been detected in a trader's behavior:

PATTERNS:
{pattern_text}

Please identify:
1. NARRATIVE: A unified behavioral narrative that connects these patterns (3-4 sentences)
2. ROOT_CAUSE: The most likely underlying root cause or behavioral driver
3. PRIORITY_ACTION: The single most impactful action to address these patterns

Format your response as JSON:
{{"narrative": "...", "root_cause": "...", "priority_action": "..."}}"""

    def _build_override_risk_prompt(
        self,
        override_context: Dict[str, Any],
        recent_patterns: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for real-time override risk assessment."""
        pattern_text = "\n".join(
            f"  - {p.get('description', '')}"
            for p in recent_patterns[:5]
        )

        return f"""You are a trading psychology advisor providing real-time guidance.

A trader is about to OVERRIDE their trading bot's recommendation:

OVERRIDE CONTEXT:
  - Pair: {override_context.get('pair', 'unknown')}
  - Bot confidence: {override_context.get('buddy_confidence', 0):.0%}
  - Trader's emotional state: {override_context.get('emotional_state', 'unknown')}
  - Cognitive load: {override_context.get('cognitive_load', 'unknown')}
  - Override type: {override_context.get('override_type', 'unknown')}

RELEVANT BEHAVIORAL PATTERNS:
{pattern_text if pattern_text.strip() else '  (No recent patterns detected)'}

Provide a brief (2-3 sentences), empathetic but honest risk assessment.
Focus on whether this specific override situation matches known risky patterns.

Format as JSON:
{{"risk_narrative": "...", "risk_level": "low|moderate|high|critical", "proceed_recommendation": "proceed|caution|reconsider"}}"""

    # --- LLM Call Infrastructure ---

    def _call_llm(
        self,
        prompt: str,
        pattern_id: str,
        parse_fn,
    ) -> SynthesisResult:
        """Execute an LLM call with rate limiting and error handling.

        This is the single point where all LLM calls go through.
        Handles rate limiting, timing, error wrapping, and audit logging.
        """
        # Cooldown check
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < DEFAULT_COOLDOWN_SECONDS:
            return SynthesisResult(
                success=False,
                fallback_used=True,
                error=f"Cooldown: {DEFAULT_COOLDOWN_SECONDS - elapsed:.0f}s remaining",
            )

        start_time = time.time()
        try:
            response_text, tokens = self._dispatch_to_provider(prompt)
            latency = (time.time() - start_time) * 1000

            self._last_call_time = time.time()
            self._daily_call_count += 1
            self._save_daily_count()

            # Parse provider response
            result = parse_fn(response_text)
            result.source_model = f"{self._provider}/{self._model}"
            result.tokens_used = tokens
            result.latency_ms = latency
            result.success = True

            # Log for audit
            self._log_synthesis(SynthesisLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                pattern_id=pattern_id,
                model_used=f"{self._provider}/{self._model}",
                tokens_used=tokens,
                latency_ms=latency,
                success=True,
            ))

            logger.info(
                "Cloud synthesis: %s/%s, %d tokens, %.0fms",
                self._provider, self._model, tokens, latency,
            )
            return result

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._log_synthesis(SynthesisLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                pattern_id=pattern_id,
                model_used=f"{self._provider}/{self._model}",
                tokens_used=0,
                latency_ms=latency,
                success=False,
                error=str(e),
            ))
            logger.warning("Cloud synthesis failed: %s", e)
            return SynthesisResult(
                success=False,
                fallback_used=True,
                error=str(e),
            )

    def _dispatch_to_provider(self, prompt: str) -> tuple:
        """Route the prompt to the configured LLM provider.

        Returns:
            (response_text, tokens_used) tuple

        Raises:
            RuntimeError: If provider not supported or API call fails
        """
        if self._provider == "openai":
            return self._call_openai(prompt)
        elif self._provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self._provider == "local":
            return self._call_local(prompt)
        else:
            raise RuntimeError(f"Unsupported LLM provider: {self._provider}")

    def _call_openai(self, prompt: str) -> tuple:
        """Call OpenAI API. Lazy import to keep zero-dependency default."""
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DEFAULT_MAX_TOKENS_PER_CALL,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    def _call_anthropic(self, prompt: str) -> tuple:
        """Call Anthropic API. Lazy import to keep zero-dependency default."""
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=DEFAULT_MAX_TOKENS_PER_CALL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
        return text, tokens

    def _call_local(self, prompt: str) -> tuple:
        """Call a local LLM server (OpenAI-compatible API).

        Expects AURA_LLM_API_KEY to be the local server URL
        (e.g., "http://localhost:11434/v1" for Ollama).
        """
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        client = openai.OpenAI(
            base_url=self._api_key,  # For local, "api_key" holds the base URL
            api_key="local",
        )
        response = client.chat.completions.create(
            model=self._model or "llama3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DEFAULT_MAX_TOKENS_PER_CALL,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

    # --- Response Parsers ---

    def _parse_explanation_response(self, text: str) -> SynthesisResult:
        """Parse LLM response for pattern explanation."""
        data = self._extract_json(text)
        confidence_map = {"low": 0.05, "medium": 0.10, "high": 0.15}

        return SynthesisResult(
            success=True,
            narrative=data.get("explanation", text[:500]),
            suggested_actions=data.get("actions", []),
            confidence_boost=confidence_map.get(
                data.get("confidence", "medium"), 0.10
            ),
        )

    def _parse_connections_response(self, text: str) -> SynthesisResult:
        """Parse LLM response for cross-pattern connections."""
        data = self._extract_json(text)

        narrative = data.get("narrative", "")
        root_cause = data.get("root_cause", "")
        if root_cause:
            narrative += f" Root cause: {root_cause}"

        return SynthesisResult(
            success=True,
            narrative=narrative or text[:500],
            suggested_actions=[data.get("priority_action", "")] if data.get("priority_action") else [],
            confidence_boost=0.10,
        )

    def _parse_override_risk_response(self, text: str) -> SynthesisResult:
        """Parse LLM response for override risk assessment."""
        data = self._extract_json(text)

        risk_map = {"low": 0.0, "moderate": 0.05, "high": 0.10, "critical": 0.15}
        risk_level = data.get("risk_level", "moderate")

        return SynthesisResult(
            success=True,
            narrative=data.get("risk_narrative", text[:500]),
            suggested_actions=[data.get("proceed_recommendation", "caution")],
            confidence_boost=risk_map.get(risk_level, 0.05),
        )

    # --- Local Fallbacks ---

    def _local_fallback_explanation(
        self,
        pattern_description: str,
        evidence_items: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """Generate a local-only explanation when LLM is unavailable.

        Uses template-based heuristics — not as good as LLM but better
        than nothing. Matches known pattern types to canned explanations.
        """
        narrative = self._template_match_explanation(pattern_description)
        evidence_count = len(evidence_items)

        return SynthesisResult(
            success=True,
            narrative=narrative,
            suggested_actions=self._template_match_actions(pattern_description),
            confidence_boost=min(0.05, evidence_count * 0.01),
            source_model="local/template",
            fallback_used=True,
        )

    def _local_fallback_connections(
        self,
        patterns: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """Generate local-only cross-pattern narrative."""
        if len(patterns) < 2:
            return SynthesisResult(
                success=True,
                narrative="Insufficient patterns for cross-pattern analysis.",
                fallback_used=True,
                source_model="local/template",
            )

        # Simple heuristic: look for shared themes
        descriptions = " ".join(
            p.get("description", "") for p in patterns
        ).lower()

        themes = []
        if "stress" in descriptions:
            themes.append("stress-related")
        if "override" in descriptions:
            themes.append("override-related")
        if "readiness" in descriptions:
            themes.append("readiness-related")
        if "emotion" in descriptions:
            themes.append("emotional")
        if "loss" in descriptions or "win rate" in descriptions:
            themes.append("performance-related")

        if themes:
            narrative = (
                f"Cross-pattern analysis detected {len(patterns)} patterns with "
                f"shared themes: {', '.join(themes)}. These may indicate a "
                f"compound behavioral cycle — consider reviewing these patterns "
                f"together for deeper insight."
            )
        else:
            narrative = (
                f"Detected {len(patterns)} independent patterns. "
                f"Cloud synthesis unavailable for deeper connection analysis."
            )

        return SynthesisResult(
            success=True,
            narrative=narrative,
            fallback_used=True,
            source_model="local/template",
            confidence_boost=0.03,
        )

    def _local_fallback_override_risk(
        self,
        override_context: Dict[str, Any],
        recent_patterns: List[Dict[str, Any]],
    ) -> SynthesisResult:
        """Generate local-only override risk narrative."""
        risk_factors = []
        emotional_state = override_context.get("emotional_state", "").lower()
        confidence = override_context.get("buddy_confidence", 0)

        high_risk_emotions = {"revenge", "frustrated", "anxious", "fomo", "tilted"}
        if emotional_state in high_risk_emotions:
            risk_factors.append(f"emotional state '{emotional_state}' is high-risk")

        if confidence > 0.7:
            risk_factors.append(
                f"bot confidence is high ({confidence:.0%}) — overriding strong signals"
            )

        override_patterns = [
            p for p in recent_patterns
            if "override" in p.get("description", "").lower()
        ]
        if override_patterns:
            risk_factors.append(
                f"{len(override_patterns)} recent override-related patterns detected"
            )

        if risk_factors:
            narrative = (
                f"Risk factors for this override: {'; '.join(risk_factors)}. "
                f"Historical data suggests caution."
            )
            risk_level = "high" if len(risk_factors) >= 2 else "moderate"
        else:
            narrative = "No specific risk factors detected for this override."
            risk_level = "low"

        return SynthesisResult(
            success=True,
            narrative=narrative,
            suggested_actions=[risk_level],
            fallback_used=True,
            source_model="local/heuristic",
            confidence_boost=0.02,
        )

    # --- Template Matching ---

    def _template_match_explanation(self, description: str) -> str:
        """Match pattern description to a template explanation."""
        desc_lower = description.lower()

        if "stress" in desc_lower and "win rate" in desc_lower:
            return (
                "Stress narrows attention and increases impulsive decision-making, "
                "which reduces trade quality. During stressful periods, cognitive "
                "bandwidth is diverted from market analysis to stressor processing."
            )
        elif "readiness" in desc_lower and "pnl" in desc_lower:
            return (
                "Readiness scores capture your holistic state — emotional, cognitive, "
                "and discipline factors. When readiness is low, systematic errors "
                "compound: looser risk management, emotional entries, and reduced "
                "pattern recognition."
            )
        elif "override" in desc_lower and "loss" in desc_lower:
            return (
                "Overrides during adverse conditions often reflect emotional rather "
                "than analytical decision-making. The bot's signal aggregation works "
                "best precisely when human intuition is most compromised."
            )
        elif "emotion" in desc_lower:
            return (
                "Emotional states directly influence risk perception and decision "
                "quality. Positive emotional states correlate with more patient, "
                "disciplined trading."
            )
        else:
            return (
                f"Pattern detected with local analysis. Cloud synthesis unavailable "
                f"for deeper behavioral explanation."
            )

    def _template_match_actions(self, description: str) -> List[str]:
        """Match pattern to suggested actions."""
        desc_lower = description.lower()

        if "stress" in desc_lower:
            return [
                "Reduce position sizes during high-stress weeks",
                "Consider a trading pause when stress score exceeds threshold",
            ]
        elif "override" in desc_lower:
            return [
                "Review override decision log before next override",
                "Implement a 5-minute cooling period before override execution",
            ]
        elif "readiness" in desc_lower:
            return [
                "Use readiness score as a position-sizing multiplier",
                "Set a minimum readiness threshold for trade entry",
            ]
        else:
            return ["Review pattern evidence for actionable insights"]

    # --- Utility ---

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text, handling markdown code fences."""
        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from code fence
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except (json.JSONDecodeError, TypeError):
                    continue

        # Try finding JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return {}

    # --- Rate Limiting Persistence ---

    def _check_daily_reset(self) -> None:
        """Reset daily call count if the date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_call_count = 0
            self._daily_reset_date = today
            self._save_daily_count()

    def _save_daily_count(self) -> None:
        """Persist daily call count."""
        count_path = self.log_dir / "daily_count.json"
        try:
            count_path.write_text(json.dumps({
                "date": self._daily_reset_date,
                "count": self._daily_call_count,
            }))
        except Exception:
            pass

    def _load_daily_count(self) -> None:
        """Load daily call count from disk."""
        count_path = self.log_dir / "daily_count.json"
        if not count_path.exists():
            return
        try:
            data = json.loads(count_path.read_text())
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if data.get("date") == today:
                self._daily_call_count = data.get("count", 0)
                self._daily_reset_date = today
        except Exception:
            pass

    def _log_synthesis(self, log_entry: SynthesisLog) -> None:
        """Append to the audit log."""
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": log_entry.timestamp,
                    "pattern_id": log_entry.pattern_id,
                    "model_used": log_entry.model_used,
                    "tokens_used": log_entry.tokens_used,
                    "latency_ms": log_entry.latency_ms,
                    "success": log_entry.success,
                    "error": log_entry.error,
                }) + "\n")
        except Exception as e:
            logger.warning("Failed to write synthesis log: %s", e)
