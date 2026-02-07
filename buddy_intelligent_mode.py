"""
Buddy Intelligent Mode - LLM Wrapper Enhancements for Static Quant Model

This module implements a hybrid wrapper approach where a lightweight LLM
adds "true AI" layers on top of Buddy's static quant model predictions.

The core prediction remains untouched (Buddy's profit-generating edge),
while the LLM provides:
1. Self-Improvement via Self-Refine (autonomous learning from mistakes)
2. Reasoning (transparent explanation of trade rationale)
3. Multi-Modal Fusion (incorporating news/sentiment/calendar)
4. Online Learning (memory-based adaptation from past trades)
5. Meta-Learning (architecture/hyperparam improvement suggestions)

This hybrid is:
- Low-latency (call LLM only post-prediction)
- Hallucination-resistant (anchored to Buddy's numbers)
- Deployable as an optional "intelligent mode"

References:
- Self-Refine: Madaan et al., 2023
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# LLM CALL INFRASTRUCTURE
# =============================================================================

def _default_llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Optional[str]:
    """Default LLM call using OpenAI API.
    
    Falls back gracefully if OpenAI is not configured.
    """
    try:
        import openai
        
        api_key = openai.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not set, LLM call skipped")
            return None
        
        # Create client with explicit API key instead of modifying global state
        client = openai.OpenAI(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


# Global LLM call function (can be replaced for testing or alternative models)
_llm_call_fn: Callable[..., Optional[str]] = _default_llm_call


def set_llm_call_function(fn: Callable[..., Optional[str]]) -> None:
    """Set a custom LLM call function (e.g., for local models or testing)."""
    global _llm_call_fn
    _llm_call_fn = fn
    logger.info("Custom LLM call function registered")


def llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Optional[str]:
    """Call the configured LLM."""
    return _llm_call_fn(prompt, system_prompt, model, temperature)


# =============================================================================
# 1. SELF-IMPROVEMENT (via existing self_refine.py)
# =============================================================================
# This functionality already exists in self_refine.py with:
# - buddy_self_improve() function
# - get_feedback_only() function
# - refine_response() function
# 
# Re-export for convenience:
try:
    from self_refine import (
        buddy_self_improve as self_improve_interpretation,
        get_feedback_only,
        refine_response,
    )
except ImportError:
    logger.debug("self_refine module not available (optional)")
    
    def self_improve_interpretation(*args, **kwargs):
        """Fallback if self_refine not available."""
        return {"response": None, "error": "self_refine module not available"}
    
    def get_feedback_only(*args, **kwargs):
        return None
    
    def refine_response(*args, **kwargs):
        return None


# =============================================================================
# 2. REASONING (Explain Why the Trade)
# =============================================================================

BUDDY_INTERPRETER_SYSTEM = """You are Buddy Interpreter: Translate static quant model outputs into clear reasoning.

Always structure EXACTLY:
**Market Context** - Price action, trend, volatility.

**Buddy Signal Breakdown**
- Raw: Confidence {score}%, Prob {prob}%
- Why this score: Interpret driving patterns (e.g., "High confidence from momentum alignment").

**Key Drivers**
- Bullish/Bearish factors from features/data.
- Explicitly explain WHY the model leans this way.

**Risks & Alternatives**
- Downsides, confounders.

**Reasoning Summary**
- Direct causal chain: "Score high because X > Y, leading to BUY despite Z risk."

**Final Call**
- Trade: BUY/SELL/HOLD/NO TRADE
- Translated Confidence: High/Med/Low
- Sizing/Targets if applicable.

Never just repeat score — always explain why."""


@dataclass
class BuddyRawOutput:
    """Structured representation of Buddy's raw model output."""
    confidence: float  # 0-1 or 0-100
    prediction: float
    probability: Optional[float] = None
    direction: Optional[str] = None  # 'long' or 'short'
    last_price: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    gate_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection."""
        result = {
            "confidence": self.confidence,
            "prediction": self.prediction,
        }
        if self.probability is not None:
            result["probability"] = self.probability
        if self.direction is not None:
            result["direction"] = self.direction
        if self.last_price is not None:
            result["last_price"] = self.last_price
            result["delta"] = self.prediction - self.last_price
        if self.features is not None:
            result["key_features"] = self.features
        if self.gate_results is not None:
            result["gate_results"] = self.gate_results
        return result


def generate_trade_reasoning(
    ticker: str,
    timeframe: str,
    buddy_raw: BuddyRawOutput,
    price_data: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Generate transparent reasoning for a trade based on Buddy's output.
    
    Forces Chain-of-Thought (CoT) linking Buddy's score to evidence.
    
    Args:
        ticker: Trading instrument (e.g., "USD_JPY")
        timeframe: Prediction timeframe (e.g., "H1", "M5")
        buddy_raw: Buddy's raw model output
        price_data: Optional technical/price data for context
        model: LLM model to use
        temperature: Generation temperature
        
    Returns:
        Dict with:
        - reasoning: Full structured reasoning text
        - trade_call: Extracted trade decision (BUY/SELL/HOLD/NO_TRADE)
        - confidence_level: Translated confidence (High/Med/Low)
        - parsed_sections: Dict of parsed sections if successful
    """
    # Format buddy raw for prompt
    buddy_dict = buddy_raw.to_dict()
    
    # Build prompt
    prompt = f"""Buddy Raw: {json.dumps(buddy_dict, indent=2)}
Ticker: {ticker}
Timeframe: {timeframe}
"""
    
    if price_data:
        prompt += f"\nProvided Data: {json.dumps(price_data, indent=2)}\n"
    
    prompt += "\nProvide full structured reasoning:"
    
    response = llm_call(
        prompt=prompt,
        system_prompt=BUDDY_INTERPRETER_SYSTEM,
        model=model,
        temperature=temperature,
    )
    
    if response is None:
        return {
            "reasoning": None,
            "trade_call": "NO_TRADE",
            "confidence_level": "Low",
            "error": "LLM call failed",
            "parsed_sections": {},
        }
    
    # Parse response to extract key fields using more robust pattern matching
    trade_call = _extract_trade_call(response)
    confidence_level = _extract_confidence_level(response)
    
    return {
        "reasoning": response,
        "trade_call": trade_call,
        "confidence_level": confidence_level,
        "parsed_sections": _parse_reasoning_sections(response),
    }


# =============================================================================
# TRADE VALIDATION - LLM actually influences decisions
# =============================================================================

LLM_TRADE_VALIDATOR_SYSTEM = """You are a risk-aware FX trade validator. Your job is to review a proposed trade and decide if it should proceed.

You will receive:
1. The ML model's recommendation (direction, confidence, gate statuses)
2. Current market context (price, ATR, RSI, ADX, trend)
3. Any available news/sentiment data

Your response MUST be valid JSON with these exact fields:
{
  "approve": true/false,
  "size_multiplier": 0.5 to 1.5 (adjust position size: 0.5=half, 1.0=normal, 1.5=increase),
  "reason": "Brief explanation",
  "risk_flags": ["list", "of", "concerns"] or []
}

APPROVAL GUIDELINES:
- APPROVE if model confidence is reasonable and no major red flags
- REDUCE SIZE (0.5-0.8) if mixed signals or moderate risk
- REJECT if:
  * Extreme RSI (<5 or >95) going against the extreme
  * Major news event imminent for this currency
  * Model direction contradicts strong trend (ADX>50)
  * Multiple conflicting signals

Be decisive. Don't reject good setups. Only reject clear problems."""


@dataclass 
class LLMTradeValidation:
    """Result of LLM trade validation."""
    approve: bool
    size_multiplier: float  # 0.5 to 1.5
    reason: str
    risk_flags: List[str]
    
    @classmethod
    def default_approve(cls) -> "LLMTradeValidation":
        """Return default approval (used when LLM unavailable)."""
        return cls(approve=True, size_multiplier=1.0, reason="LLM unavailable", risk_flags=[])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMTradeValidation":
        """Create from dict (parsed JSON)."""
        return cls(
            approve=d.get("approve", True),
            size_multiplier=max(0.5, min(1.5, float(d.get("size_multiplier", 1.0)))),
            reason=d.get("reason", ""),
            risk_flags=d.get("risk_flags", []),
        )


def validate_trade_with_llm(
    ticker: str,
    direction: str,
    buddy_raw: BuddyRawOutput,
    price_data: Optional[Dict[str, Any]] = None,
    news_sentiment: Optional[Dict[str, Any]] = None,
) -> LLMTradeValidation:
    """Use LLM to validate a trade before execution.
    
    This is the key integration point where LLM actually influences trading.
    
    Args:
        ticker: Instrument (e.g., "USD_JPY")
        direction: Proposed direction ("long" or "short")
        buddy_raw: Full model output
        price_data: Technical indicators
        news_sentiment: Any news/sentiment data
        
    Returns:
        LLMTradeValidation with approve/reject and size adjustment
    """
    # Build concise prompt
    prompt = f"""PROPOSED TRADE: {direction.upper()} {ticker}

MODEL OUTPUT:
- Direction: {direction}
- Confidence: {buddy_raw.confidence:.1f}%
- Probability: {buddy_raw.probability:.1%}
"""
    
    if buddy_raw.gate_results:
        prompt += f"- Gates: {json.dumps(buddy_raw.gate_results, indent=2)}\n"
    
    if price_data:
        prompt += f"""
MARKET CONTEXT:
- Price: {price_data.get('last_close', 'N/A')}
- ATR: {price_data.get('atr', 'N/A')}
- RSI: {price_data.get('rsi', 'N/A')}
- ADX: {price_data.get('adx', 'N/A')}
- Trend: {price_data.get('trend', 'N/A')}
"""
    
    if news_sentiment:
        prompt += f"""
NEWS/SENTIMENT:
- Sentiment: {news_sentiment.get('sentiment', 'N/A')} ({news_sentiment.get('score', 0):+.2f})
- Headlines: {news_sentiment.get('headline_count', 0)}
"""
    
    prompt += "\nRespond with JSON only:"
    
    response = llm_call(
        prompt=prompt,
        system_prompt=LLM_TRADE_VALIDATOR_SYSTEM,
        temperature=0.1,  # Low temp for consistent decisions
    )
    
    if response is None:
        logger.warning("LLM validation unavailable, defaulting to approve")
        return LLMTradeValidation.default_approve()
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return LLMTradeValidation.from_dict(result)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM validation response: {e}")
    
    # Fallback: try to infer from text
    response_lower = response.lower()
    if "reject" in response_lower or "do not" in response_lower or '"approve": false' in response_lower:
        return LLMTradeValidation(
            approve=False,
            size_multiplier=0.0,
            reason="LLM recommended rejection (parsing failed)",
            risk_flags=["parse_error"],
        )
    
    return LLMTradeValidation.default_approve()


def validate_trade_with_risk_assessment(
    ticker: str,
    direction: str,
    buddy_raw: BuddyRawOutput,
    price_data: Optional[Dict[str, Any]] = None,
    news_sentiment: Optional[Dict[str, Any]] = None,
    market_intel: Optional[Any] = None,
) -> LLMTradeValidation:
    """Enhanced trade validation with Phase 5 pre-trade risk assessment.
    
    This combines:
    1. Event risk assessment (economic calendar + LLM)
    2. Multi-factor risk score
    3. LLM trade validation
    
    Args:
        ticker: Instrument (e.g., "USD_JPY")
        direction: Proposed direction ("long" or "short")
        buddy_raw: Full model output
        price_data: Technical indicators
        news_sentiment: Any news/sentiment data
        market_intel: Optional MarketIntelligence instance
        
    Returns:
        LLMTradeValidation with comprehensive risk-aware decision
    """
    risk_flags = []
    size_multiplier = 1.0
    
    # 1. Run pre-trade risk assessment if market_intel available
    if market_intel is not None:
        try:
            # Build context for risk assessment
            risk_context = {}
            if price_data:
                risk_context.update({
                    'volatility': price_data.get('atr', 0.0),
                    'atr_pct': price_data.get('atr_pct', price_data.get('atr', 0.0) / price_data.get('last_close', 1.0)),
                    'rsi': price_data.get('rsi', 50.0),
                    'adx': price_data.get('adx', 20.0),
                })
            if news_sentiment:
                risk_context['sentiment'] = news_sentiment.get('score', 0.0)
            
            can_trade, risk_score, event_risk = market_intel.assess_pre_trade_risk(
                instrument=ticker,
                market_context=risk_context,
                proposed_direction=direction,
                llm_call_fn=llm_call,
            )
            
            # Check event risk
            if event_risk and event_risk.avoid_trade:
                return LLMTradeValidation(
                    approve=False,
                    size_multiplier=0.0,
                    reason=event_risk.reason,
                    risk_flags=["economic_event"],
                )
            
            # Apply risk score to size multiplier
            size_multiplier = risk_score.size_multiplier
            risk_flags.extend(risk_score.factors)
            
            # Block if risk score too high
            if risk_score.recommendation == "avoid":
                return LLMTradeValidation(
                    approve=False,
                    size_multiplier=0.0,
                    reason=f"Risk score too high ({risk_score.score:.0f}/100): {', '.join(risk_flags)}",
                    risk_flags=risk_flags,
                )
                
        except Exception as e:
            logger.debug(f"Risk assessment error: {e}")
    
    # 2. Run standard LLM validation
    validation = validate_trade_with_llm(
        ticker=ticker,
        direction=direction,
        buddy_raw=buddy_raw,
        price_data=price_data,
        news_sentiment=news_sentiment,
    )
    
    # 3. Combine size multipliers
    final_size = validation.size_multiplier * size_multiplier
    final_flags = list(set(validation.risk_flags + risk_flags))
    
    return LLMTradeValidation(
        approve=validation.approve,
        size_multiplier=max(0.25, min(1.5, final_size)),  # Clamp to valid range
        reason=validation.reason,
        risk_flags=final_flags,
    )


def _extract_trade_call(response: str) -> str:
    """Extract trade call from response with explicit pattern matching.
    
    Looks for explicit trade declarations like "Trade: BUY" or "Final Call: SELL".
    """
    response_lower = response.lower()
    
    # Look for explicit final call patterns first
    import re
    
    # Pattern: "Trade: BUY" or "Final Call: SELL"
    explicit_pattern = r"(?:trade|final call)[:\s]*(buy|sell|hold|no[_\s]?trade)"
    match = re.search(explicit_pattern, response_lower)
    if match:
        call = match.group(1).replace(" ", "_").replace("-", "_").upper()
        return call if call in ("BUY", "SELL", "HOLD", "NO_TRADE") else "NO_TRADE"
    
    # Fallback: Count occurrences to determine dominant signal
    buy_count = response_lower.count("buy")
    sell_count = response_lower.count("sell")
    hold_count = response_lower.count("hold")
    
    if buy_count > sell_count and buy_count > hold_count:
        return "BUY"
    elif sell_count > buy_count and sell_count > hold_count:
        return "SELL"
    elif hold_count > 0:
        return "HOLD"
    
    return "NO_TRADE"


def _extract_confidence_level(response: str) -> str:
    """Extract confidence level from response."""
    response_lower = response.lower()
    
    # Look for explicit confidence patterns
    import re
    
    pattern = r"(?:translated )?confidence[:\s]*(high|medium|med|low)"
    match = re.search(pattern, response_lower)
    if match:
        level = match.group(1)
        if level in ("medium", "med"):
            return "Med"
        return level.capitalize()
    
    # Fallback to keyword detection
    if "high" in response_lower:
        return "High"
    elif "medium" in response_lower or "med" in response_lower:
        return "Med"
    
    return "Low"


def _parse_reasoning_sections(text: str) -> Dict[str, str]:
    """Parse reasoning text into sections."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in text.split("\n"):
        if line.startswith("**") and line.endswith("**"):
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            # Start new section
            current_section = line.strip("*").strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


# =============================================================================
# 3. MULTI-MODAL FUSION (News/Sentiment/Calendar)
# =============================================================================

MULTI_MODAL_SYSTEM = """You are Multi-Modal Buddy: Enhance static price-based predictions with news, sentiment, macro.

Structure:
**Multi-Modal Fusion**
- How news/sentiment aligns or contradicts Buddy's signal (e.g., "Positive earnings override weak technicals").
- Event risks (e.g., "NFP tomorrow — reduce confidence").

Then provide full reasoning structure:
**Market Context** - Price action, trend, volatility.
**Buddy Signal Breakdown** - Raw score and interpretation.
**Key Drivers** - Bullish/Bearish factors.
**Risks & Alternatives** - Downsides, confounders.
**Reasoning Summary** - Direct causal chain.
**Final Call** - Trade decision with translated confidence.

Rules:
- If strong non-price signal contradicts Buddy → Downgrade trade or NO TRADE.
- Explicitly weight modalities: "Buddy 70% weight, sentiment boosts to 80% effective."

Output enhanced recommendation."""


@dataclass
class MultiModalContext:
    """Multi-modal context for fusion."""
    news_summaries: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to +1
    upcoming_events: List[str] = field(default_factory=list)
    event_impact_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection."""
        return {
            "news_summaries": self.news_summaries,
            "aggregate_sentiment": self.sentiment_score,
            "upcoming_events": self.upcoming_events,
            "impact_flags": self.event_impact_flags,
        }


def multi_modal_fusion(
    ticker: str,
    timeframe: str,
    buddy_raw: BuddyRawOutput,
    context: MultiModalContext,
    price_data: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Generate multi-modal enhanced recommendation.
    
    Fuses Buddy's price-only signal with news, sentiment, and macro data.
    
    Args:
        ticker: Trading instrument
        timeframe: Prediction timeframe
        buddy_raw: Buddy's raw model output
        context: Multi-modal context (news, sentiment, events)
        price_data: Optional technical/price data
        model: LLM model to use
        temperature: Generation temperature
        
    Returns:
        Dict with:
        - enhanced_reasoning: Full multi-modal reasoning
        - trade_call: Final trade decision
        - confidence_level: Translated confidence
        - modality_weights: Explicit weighting applied
        - event_risk: Identified event risks
    """
    buddy_dict = buddy_raw.to_dict()
    context_dict = context.to_dict()
    
    prompt = f"""Provided:
- Buddy Raw (price-only): {json.dumps(buddy_dict, indent=2)}
- News/Sentiment: {json.dumps(context_dict.get('news_summaries', []))}
- Aggregate Sentiment: {context_dict.get('aggregate_sentiment', 0.0)} (-1 to +1)
- Economic Calendar: {json.dumps(context_dict.get('upcoming_events', []))}
- Impact flags: {json.dumps(context_dict.get('impact_flags', []))}

Ticker: {ticker}
Timeframe: {timeframe}
"""
    
    if price_data:
        prompt += f"\nPrice/Technical Data: {json.dumps(price_data, indent=2)}\n"
    
    prompt += "\nProvide multi-modal fusion analysis and enhanced recommendation:"
    
    response = llm_call(
        prompt=prompt,
        system_prompt=MULTI_MODAL_SYSTEM,
        model=model,
        temperature=temperature,
    )
    
    if response is None:
        return {
            "enhanced_reasoning": None,
            "trade_call": "NO_TRADE",
            "confidence_level": "Low",
            "modality_weights": {"buddy": 1.0, "sentiment": 0.0, "events": 0.0},
            "event_risk": None,
            "error": "LLM call failed",
        }
    
    # Parse response
    response_lower = response.lower()
    
    trade_call = "NO_TRADE"
    if "buy" in response_lower and "sell" not in response_lower:
        trade_call = "BUY"
    elif "sell" in response_lower and "buy" not in response_lower:
        trade_call = "SELL"
    elif "hold" in response_lower:
        trade_call = "HOLD"
    
    confidence_level = "Low"
    if "high" in response_lower:
        confidence_level = "High"
    elif "medium" in response_lower or "med" in response_lower:
        confidence_level = "Med"
    
    # Check for event risk mentions
    event_risk = None
    if any(keyword in response_lower for keyword in ["nfp", "fomc", "ecb", "boe", "event"]):
        event_risk = "High-impact event detected"
    
    return {
        "enhanced_reasoning": response,
        "trade_call": trade_call,
        "confidence_level": confidence_level,
        "modality_weights": _extract_modality_weights(response),
        "event_risk": event_risk,
    }


def _extract_modality_weights(text: str) -> Dict[str, float]:
    """Extract modality weights from response text."""
    weights = {"buddy": 0.7, "sentiment": 0.15, "events": 0.15}  # Defaults
    
    # Simple pattern matching for explicit weights
    import re
    
    buddy_match = re.search(r"buddy[^\d]*(\d+)%", text.lower())
    if buddy_match:
        weights["buddy"] = float(buddy_match.group(1)) / 100
    
    sentiment_match = re.search(r"sentiment[^\d]*(\d+)%", text.lower())
    if sentiment_match:
        weights["sentiment"] = float(sentiment_match.group(1)) / 100
    
    return weights


# =============================================================================
# 3.5 DEEP LLM INTEGRATION - Under the Hood Enhancements
# =============================================================================

LLM_SENTIMENT_AGGREGATOR_SYSTEM = """You are a financial news analyst. Your job is to read multiple headlines about a currency pair and provide a single, coherent sentiment assessment.

Do NOT just average scores. Actually READ the headlines and understand the narrative.

Consider:
1. Are headlines consistent or contradictory?
2. Is this event-driven (specific catalyst) or general flow?
3. Which headlines are most impactful?
4. Is there a dominant theme (e.g., "central bank hawkish", "risk-off", "data beat")?

Respond with JSON only:
{
  "direction": "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish",
  "confidence": 0-100,
  "theme": "Brief description of the dominant narrative",
  "contradictions": true/false,
  "event_driven": true/false,
  "key_headline": "The most important headline"
}"""


LLM_THRESHOLD_ADVISOR_SYSTEM = """You are a trading risk advisor. Given current market conditions, suggest whether standard gate thresholds should be adjusted.

Standard thresholds:
- RSI extreme: <10 (oversold) or >90 (overbought)
- ADX strong trend: >35
- TCN probability minimum: 0.55

You can suggest:
- TIGHTER thresholds (more conservative) if conditions are risky
- LOOSER thresholds (more permissive) if conditions are favorable
- NO CHANGE if standard is appropriate

Respond with JSON only:
{
  "adjust_thresholds": true/false,
  "rsi_extreme_low": 10,
  "rsi_extreme_high": 90,
  "adx_strong_trend": 35,
  "tcn_min_probability": 0.55,
  "reason": "Brief explanation"
}"""


@dataclass
class LLMSentimentAnalysis:
    """Result of LLM-powered sentiment aggregation."""
    direction: str  # strong_bullish, bullish, neutral, bearish, strong_bearish
    confidence: int  # 0-100
    theme: str
    contradictions: bool
    event_driven: bool
    key_headline: Optional[str] = None
    raw_score: float = 0.0  # -1 to +1 for compatibility
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMSentimentAnalysis":
        """Create from parsed JSON."""
        direction = d.get("direction", "neutral")
        # Convert direction to raw_score
        score_map = {
            "strong_bullish": 0.8,
            "bullish": 0.4,
            "neutral": 0.0,
            "bearish": -0.4,
            "strong_bearish": -0.8,
        }
        return cls(
            direction=direction,
            confidence=int(d.get("confidence", 50)),
            theme=d.get("theme", ""),
            contradictions=bool(d.get("contradictions", False)),
            event_driven=bool(d.get("event_driven", False)),
            key_headline=d.get("key_headline"),
            raw_score=score_map.get(direction, 0.0),
        )
    
    @classmethod
    def neutral(cls) -> "LLMSentimentAnalysis":
        """Return neutral sentiment (fallback)."""
        return cls(
            direction="neutral",
            confidence=50,
            theme="No analysis available",
            contradictions=False,
            event_driven=False,
            raw_score=0.0,
        )


def aggregate_sentiment_with_llm(
    headlines: List[str],
    instrument: str,
    existing_scores: Optional[List[float]] = None,
) -> LLMSentimentAnalysis:
    """Use LLM to aggregate multiple headlines into coherent sentiment.
    
    This is smarter than averaging per-headline scores because:
    1. LLM understands contradictions ("bulls vs bears")
    2. LLM identifies dominant themes
    3. LLM weights by importance
    
    Args:
        headlines: List of news headlines
        instrument: Currency pair (e.g., "USD_JPY")
        existing_scores: Optional per-headline scores for reference
        
    Returns:
        LLMSentimentAnalysis with holistic sentiment assessment
    """
    if not headlines:
        return LLMSentimentAnalysis.neutral()
    
    # Build prompt with headlines
    headline_text = "\n".join(f"- {h}" for h in headlines[:15])  # Limit to 15
    
    prompt = f"""Analyze these headlines for {instrument}:

{headline_text}

"""
    if existing_scores:
        avg_score = sum(existing_scores) / len(existing_scores)
        prompt += f"(Per-headline sentiment avg: {avg_score:+.2f})\n"
    
    prompt += "\nProvide your holistic sentiment assessment as JSON:"
    
    response = llm_call(
        prompt=prompt,
        system_prompt=LLM_SENTIMENT_AGGREGATOR_SYSTEM,
        temperature=0.1,
    )
    
    if response is None:
        logger.debug("LLM sentiment aggregation unavailable")
        return LLMSentimentAnalysis.neutral()
    
    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return LLMSentimentAnalysis.from_dict(result)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM sentiment response: {e}")
    
    return LLMSentimentAnalysis.neutral()


@dataclass
class DynamicThresholds:
    """Dynamic gate thresholds suggested by LLM."""
    adjust_thresholds: bool
    rsi_extreme_low: float = 10.0
    rsi_extreme_high: float = 90.0
    adx_strong_trend: float = 35.0
    tcn_min_probability: float = 0.55
    reason: str = ""
    
    @classmethod
    def default(cls) -> "DynamicThresholds":
        """Return standard thresholds (no adjustment)."""
        return cls(adjust_thresholds=False, reason="Using standard thresholds")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicThresholds":
        """Create from parsed JSON."""
        return cls(
            adjust_thresholds=bool(d.get("adjust_thresholds", False)),
            rsi_extreme_low=float(d.get("rsi_extreme_low", 10.0)),
            rsi_extreme_high=float(d.get("rsi_extreme_high", 90.0)),
            adx_strong_trend=float(d.get("adx_strong_trend", 35.0)),
            tcn_min_probability=float(d.get("tcn_min_probability", 0.55)),
            reason=d.get("reason", ""),
        )


def get_dynamic_thresholds(
    instrument: str,
    market_context: Dict[str, Any],
    only_for_edge_cases: bool = True,
) -> DynamicThresholds:
    """Ask LLM whether gate thresholds should be adjusted.
    
    This is called ONLY for edge cases to minimize LLM calls:
    - RSI in 8-15 or 85-92 range (near extreme thresholds)
    - ADX in 30-40 range (near trend threshold)
    
    Args:
        instrument: Currency pair
        market_context: Dict with rsi, adx, atr_pct, news_summary
        only_for_edge_cases: If True, only call LLM for borderline cases
        
    Returns:
        DynamicThresholds with possibly adjusted values
    """
    rsi = market_context.get('rsi', 50.0)
    adx = market_context.get('adx', 20.0)
    
    # Only call LLM for edge cases to save cost/latency
    if only_for_edge_cases:
        rsi_edge = (8 <= rsi <= 15) or (85 <= rsi <= 92)
        adx_edge = 30 <= adx <= 40
        
        if not (rsi_edge or adx_edge):
            return DynamicThresholds.default()
    
    prompt = f"""Current market conditions for {instrument}:

- RSI: {rsi:.1f}
- ADX: {adx:.1f}
- Volatility (ATR%): {market_context.get('atr_pct', 'N/A')}
- Recent News: {market_context.get('news_summary', 'None')}
- Trend: {market_context.get('trend', 'Unknown')}

Should standard gate thresholds be adjusted?

Note: This is an edge case because {"RSI is near threshold" if 8 <= rsi <= 15 or 85 <= rsi <= 92 else "ADX is near threshold"}.

Respond with JSON:"""

    response = llm_call(
        prompt=prompt,
        system_prompt=LLM_THRESHOLD_ADVISOR_SYSTEM,
        temperature=0.1,
    )
    
    if response is None:
        return DynamicThresholds.default()
    
    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            thresholds = DynamicThresholds.from_dict(result)
            
            # Sanity checks - don't allow crazy values
            thresholds.rsi_extreme_low = max(5, min(20, thresholds.rsi_extreme_low))
            thresholds.rsi_extreme_high = max(80, min(95, thresholds.rsi_extreme_high))
            thresholds.adx_strong_trend = max(25, min(50, thresholds.adx_strong_trend))
            thresholds.tcn_min_probability = max(0.50, min(0.70, thresholds.tcn_min_probability))
            
            return thresholds
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM threshold response: {e}")
    
    return DynamicThresholds.default()


def enrich_trade_context(
    instrument: str,
    direction: str,
    price_data: Dict[str, Any],
    online_memory: Optional["OnlineLearningMemory"] = None,
) -> Dict[str, Any]:
    """Enrich trade context with LLM-generated insights.
    
    This runs ONCE before trade validation to add:
    1. Historical pattern recognition
    2. Time-of-day insights
    3. Memory-based adjustments
    
    Args:
        instrument: Currency pair
        direction: Proposed direction
        price_data: Technical indicators
        online_memory: Past trade lessons
        
    Returns:
        Enriched context dict
    """
    context = {
        "time_insight": _get_time_insight(instrument),
        "pattern_note": None,
        "memory_adjustment": None,
    }
    
    # Add memory-based insight if available
    if online_memory and online_memory.trades:
        recent_similar = [
            t for t in online_memory.trades[-20:]
            if t.get("instrument") == instrument and t.get("direction") == direction
        ]
        if recent_similar:
            wins = sum(1 for t in recent_similar if t.get("outcome") == "win")
            total = len(recent_similar)
            win_rate = wins / total if total > 0 else 0.5
            
            if win_rate < 0.35:
                context["memory_adjustment"] = f"Recent {direction} on {instrument}: {wins}/{total} wins. Consider size reduction."
            elif win_rate > 0.65:
                context["memory_adjustment"] = f"Recent {direction} on {instrument}: {wins}/{total} wins. Setup looks favorable."
    
    return context


def _get_time_insight(instrument: str) -> Optional[str]:
    """Get time-based trading insight."""
    from datetime import datetime
    
    now = datetime.utcnow()
    hour = now.hour
    
    # Session insights
    if 0 <= hour < 7:  # Asian session
        if "JPY" in instrument or "AUD" in instrument or "NZD" in instrument:
            return "Asian session - primary session for this pair"
        return "Asian session - lower liquidity for majors"
    elif 7 <= hour < 12:  # London morning
        return "London session - peak EUR/GBP liquidity"
    elif 12 <= hour < 17:  # US/London overlap
        return "US/London overlap - highest liquidity"
    elif 17 <= hour < 21:  # US afternoon
        return "US session - watch for late-day reversals"
    else:  # Late US/early Asia
        return "Session transition - wider spreads possible"


# =============================================================================
# 4. ONLINE LEARNING (Memory-Based Adaptation)
# =============================================================================

ONLINE_REFLECTION_SYSTEM = """You have online memory from past trades:
{past_lessons_summary}

Before reasoning:
- Reflect: How does this apply to current setup?
- Adapt: Adjust confidence, sizing, or call based on lessons (e.g., "Reducing size due to past news whipsaws").

Add section:
**Adaptation from Experience**
- Relevant past patterns.
- Changes made this time.

Then proceed with full reasoning."""


@dataclass
class TradeLessonMemory:
    """Memory of lessons learned from past trades."""
    lessons: List[str] = field(default_factory=list)
    win_rate: float = 0.5
    recent_drawdown: float = 0.0
    regime_notes: Optional[str] = None
    
    def summarize(self) -> str:
        """Generate summary for prompt injection."""
        if not self.lessons:
            return "No significant past lessons recorded."
        
        summary_parts = []
        summary_parts.append(f"Recent win rate: {self.win_rate:.1%}")
        
        if self.recent_drawdown > 0:
            summary_parts.append(f"Recent drawdown: {self.recent_drawdown:.1%}")
        
        if self.regime_notes:
            summary_parts.append(f"Regime: {self.regime_notes}")
        
        summary_parts.append("Key lessons:")
        for lesson in self.lessons[-5:]:  # Last 5 lessons
            summary_parts.append(f"- {lesson}")
        
        return "\n".join(summary_parts)


class OnlineLearningMemory:
    """Manages online learning memory from past trades.
    
    Stores trade outcomes and uses LLM to summarize lessons periodically.
    """
    
    def __init__(
        self,
        memory_path: Optional[Path] = None,
        max_lessons: int = 100,
    ):
        """Initialize online learning memory.
        
        Args:
            memory_path: Path to persist memory (optional)
            max_lessons: Maximum lessons to retain
        """
        self.memory_path = memory_path
        self.max_lessons = max_lessons
        
        # In-memory storage
        self._outcomes: List[Dict[str, Any]] = []
        self._lessons: List[str] = []
        self._summary_cache: Optional[str] = None
        
        # Load persisted memory if available
        if memory_path and memory_path.exists():
            self._load_memory()
    
    def record_outcome(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        confidence: float,
        pnl: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a trade outcome for learning.
        
        Args:
            trade_id: Unique trade identifier
            instrument: Trading instrument
            direction: Trade direction (long/short)
            confidence: Model confidence at entry
            pnl: Profit/loss result
            context: Optional context (news, events, etc.)
        """
        outcome = {
            "trade_id": trade_id,
            "instrument": instrument,
            "direction": direction,
            "confidence": confidence,
            "pnl": pnl,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self._outcomes.append(outcome)
        self._summary_cache = None  # Invalidate cache
        
        # Trim if needed
        if len(self._outcomes) > self.max_lessons * 2:
            self._outcomes = self._outcomes[-self.max_lessons:]
        
        # Persist if configured
        if self.memory_path:
            self._save_memory()
    
    def extract_lessons(
        self,
        model: str = "gpt-4",
        force_refresh: bool = False,
    ) -> List[str]:
        """Extract lessons from recent outcomes using LLM.
        
        Args:
            model: LLM model to use
            force_refresh: Force re-extraction even if cached
            
        Returns:
            List of lesson strings
        """
        if not force_refresh and self._lessons:
            return self._lessons
        
        if len(self._outcomes) < 5:
            return ["Insufficient data for lesson extraction."]
        
        # Prepare recent outcomes summary
        recent = self._outcomes[-20:]  # Last 20 trades
        
        wins = [o for o in recent if o["pnl"] > 0]
        losses = [o for o in recent if o["pnl"] < 0]
        
        prompt = f"""Analyze these recent trade outcomes and extract 3-5 key lessons:

Recent trades (last {len(recent)}):
- Win rate: {len(wins)/len(recent):.1%}
- Average win PnL: {sum(o['pnl'] for o in wins)/max(len(wins),1):.2f}
- Average loss PnL: {sum(o['pnl'] for o in losses)/max(len(losses),1):.2f}

Sample outcomes:
{json.dumps(recent[-5:], indent=2)}

Extract actionable lessons like:
- "In high-vol regimes, overconfidence led to -15% drawdown"
- "News sentiment overrides improved wins by 12%"
- "Low confidence trades (<60%) had 30% win rate - avoid"

Return JSON array of lesson strings only:"""
        
        response = llm_call(
            prompt=prompt,
            model=model,
            temperature=0.3,
        )
        
        if response:
            try:
                # Try to parse as JSON array
                lessons = json.loads(response)
                if isinstance(lessons, list):
                    self._lessons = lessons
                    return lessons
            except json.JSONDecodeError:
                pass
            
            # Fallback: split by newlines
            lines = [l.strip("- ").strip() for l in response.split("\n") if l.strip()]
            self._lessons = lines[:5]
            return self._lessons
        
        return ["Unable to extract lessons from outcomes."]
    
    def get_memory_context(self) -> TradeLessonMemory:
        """Get memory context for injection into prompts."""
        if not self._outcomes:
            return TradeLessonMemory()
        
        recent = self._outcomes[-20:]
        wins = [o for o in recent if o["pnl"] > 0]
        
        return TradeLessonMemory(
            lessons=self._lessons or self.extract_lessons(),
            win_rate=len(wins) / len(recent) if recent else 0.5,
            recent_drawdown=self._calculate_recent_drawdown(),
        )
    
    def _calculate_recent_drawdown(self) -> float:
        """Calculate recent drawdown from outcomes."""
        if len(self._outcomes) < 2:
            return 0.0
        
        pnls = [o["pnl"] for o in self._outcomes[-20:]]
        
        # Use itertools.accumulate for O(n) cumulative sum
        from itertools import accumulate
        cumsum = list(accumulate(pnls))
        
        peak = cumsum[0]
        max_dd = 0.0
        
        for val in cumsum:
            if val > peak:
                peak = val
            dd = (peak - val) / max(abs(peak), 1.0) if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _save_memory(self) -> None:
        """Save memory to disk."""
        if self.memory_path:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "outcomes": self._outcomes,
                "lessons": self._lessons,
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            self.memory_path.write_text(json.dumps(data, indent=2))
    
    def _load_memory(self) -> None:
        """Load memory from disk."""
        if self.memory_path and self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                self._outcomes = data.get("outcomes", [])
                self._lessons = data.get("lessons", [])
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")


def generate_adaptive_reasoning(
    ticker: str,
    timeframe: str,
    buddy_raw: BuddyRawOutput,
    memory: OnlineLearningMemory,
    price_data: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Generate reasoning with online learning adaptation.
    
    Injects past trade lessons for LLM reflection and adaptation.
    
    Args:
        ticker: Trading instrument
        timeframe: Prediction timeframe
        buddy_raw: Buddy's raw model output
        memory: Online learning memory instance
        price_data: Optional technical/price data
        model: LLM model to use
        temperature: Generation temperature
        
    Returns:
        Dict with adapted reasoning and adaptation notes
    """
    # Get memory context
    memory_context = memory.get_memory_context()
    past_lessons_summary = memory_context.summarize()
    
    # Build system prompt with memory
    system_prompt = ONLINE_REFLECTION_SYSTEM.format(
        past_lessons_summary=past_lessons_summary
    )
    
    buddy_dict = buddy_raw.to_dict()
    
    prompt = f"""Buddy Raw: {json.dumps(buddy_dict, indent=2)}
Ticker: {ticker}
Timeframe: {timeframe}
"""
    
    if price_data:
        prompt += f"\nCurrent Context: {json.dumps(price_data, indent=2)}\n"
    
    prompt += "\nProvide reasoning with adaptation from experience:"
    
    response = llm_call(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
    )
    
    if response is None:
        return {
            "adaptive_reasoning": None,
            "trade_call": "NO_TRADE",
            "adaptations_made": [],
            "error": "LLM call failed",
        }
    
    # Parse response
    response_lower = response.lower()
    
    trade_call = "NO_TRADE"
    if "buy" in response_lower and "sell" not in response_lower:
        trade_call = "BUY"
    elif "sell" in response_lower and "buy" not in response_lower:
        trade_call = "SELL"
    elif "hold" in response_lower:
        trade_call = "HOLD"
    
    # Extract adaptations mentioned
    adaptations = []
    adaptation_keywords = ["reducing", "increasing", "adjusting", "avoiding", "due to past"]
    for keyword in adaptation_keywords:
        if keyword in response_lower:
            adaptations.append(keyword)
    
    return {
        "adaptive_reasoning": response,
        "trade_call": trade_call,
        "adaptations_made": adaptations,
        "memory_summary": past_lessons_summary,
    }


# =============================================================================
# 5. META-LEARNING (Architecture/Hyperparam Suggestions)
# =============================================================================

META_LEARNING_SYSTEM = """You are Buddy Evolutor: Suggest autonomous upgrades to the static quant model.

For each suggestion:
- Expected impact
- Implementation ease
- Test plan

Rank by risk-adjusted potential.
Output structured list only."""


META_LEARNING_PROMPT = """Current Architecture: {architecture_description}
Recent Performance: {performance_stats}
Known Weaknesses: {weaknesses}

Propose 3-5 concrete, prioritized improvements:
1. Hyperparams (e.g., increase lookback, adjust thresholds)
2. Features (add sentiment embeddings)
3. Architecture (try online adapters, causal layers)
4. Training (switch to incremental updates)

Return JSON with structure:
{{
    "improvements": [
        {{
            "category": "hyperparam|feature|architecture|training",
            "suggestion": "description",
            "expected_impact": "description",
            "implementation_ease": "easy|medium|hard",
            "test_plan": "how to validate"
        }}
    ]
}}"""


@dataclass
class BuddyArchitectureDescription:
    """Description of Buddy's current architecture for meta-learning."""
    model_type: str = "Ensemble (TCN, XGBoost, Ridge, RandomForest)"
    feature_set: str = "Technical indicators, price patterns"
    lookback_window: int = 60
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    ensemble_weights: Optional[Dict[str, float]] = None
    
    def to_description(self) -> str:
        """Generate text description."""
        parts = [
            f"Model: {self.model_type}",
            f"Features: {self.feature_set}",
            f"Lookback: {self.lookback_window} bars",
            f"Hidden: {self.hidden_size}, Layers: {self.num_layers}",
            f"Dropout: {self.dropout}",
        ]
        if self.ensemble_weights:
            parts.append(f"Weights: {self.ensemble_weights}")
        return "; ".join(parts)


@dataclass
class PerformanceStats:
    """Performance statistics for meta-learning analysis."""
    win_rate: float = 0.50
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_pnl: float = 0.0
    total_trades: int = 0
    recent_streak: str = "mixed"
    
    def to_summary(self) -> str:
        """Generate summary for prompt."""
        return (
            f"Win rate: {self.win_rate:.1%}, "
            f"Profit factor: {self.profit_factor:.2f}, "
            f"Sharpe: {self.sharpe_ratio:.2f}, "
            f"Max DD: {self.max_drawdown:.1%}, "
            f"Total trades: {self.total_trades}, "
            f"Recent: {self.recent_streak}"
        )


@dataclass
class ModelImprovement:
    """A suggested model improvement."""
    category: str  # hyperparam, feature, architecture, training
    suggestion: str
    expected_impact: str
    implementation_ease: str  # easy, medium, hard
    test_plan: str
    confidence: float = 0.5  # 0-1 confidence in the suggestion
    sharpe_delta: float = 0.0  # Expected Sharpe ratio improvement


@dataclass
class GatedApprovalResult:
    """Result of gated auto-approval for suggestions."""
    approved: List[ModelImprovement]
    pending_review: List[ModelImprovement]
    rejected: List[ModelImprovement]
    applied_changes: List[Dict[str, Any]]
    changelog: List[str]


def filter_hyperparam_suggestions(
    improvements: List[ModelImprovement],
) -> tuple[List[ModelImprovement], List[ModelImprovement]]:
    """Filter improvements into hyperparams (auto-approvable) and others (manual review).
    
    Args:
        improvements: List of model improvements
        
    Returns:
        Tuple of (hyperparam suggestions, other suggestions)
    """
    hyperparams = []
    others = []
    
    for imp in improvements:
        if imp.category.lower() in ("hyperparam", "hyperparameter", "parameter"):
            hyperparams.append(imp)
        else:
            others.append(imp)
    
    return hyperparams, others


def check_approval_gates(
    improvement: ModelImprovement,
    min_confidence: float = 0.7,
    min_sharpe_delta: float = 0.05,  # 5% improvement
) -> tuple[bool, str]:
    """Check if a hyperparam improvement passes approval gates.
    
    Gates:
    1. confidence > 0.7
    2. sharpe_delta > 5%
    3. implementation_ease == "easy"
    
    Args:
        improvement: The improvement to check
        min_confidence: Minimum confidence threshold (default 0.7)
        min_sharpe_delta: Minimum Sharpe improvement (default 5%)
        
    Returns:
        Tuple of (passed, reason)
    """
    # Gate 1: Confidence check
    if improvement.confidence < min_confidence:
        return False, f"Confidence {improvement.confidence:.1%} < {min_confidence:.1%}"
    
    # Gate 2: Expected Sharpe improvement
    if improvement.sharpe_delta < min_sharpe_delta:
        return False, f"Sharpe delta {improvement.sharpe_delta:.1%} < {min_sharpe_delta:.1%}"
    
    # Gate 3: Implementation ease
    if improvement.implementation_ease.lower() != "easy":
        return False, f"Implementation ease '{improvement.implementation_ease}' != 'easy'"
    
    return True, "All gates passed"


def apply_hyperparam_to_config(
    suggestion: str,
    config_path: Path,
) -> tuple[bool, str, Dict[str, Any]]:
    """Parse and apply a hyperparam suggestion to config file.
    
    Args:
        suggestion: The suggestion text (e.g., "Increase learning rate to 0.001")
        config_path: Path to the YAML config file
        
    Returns:
        Tuple of (success, message, changes_dict)
    """
    import re
    import yaml
    
    # Common hyperparam patterns
    patterns = {
        r"(?:learning[_\s]?rate|lr)[:\s]+(?:to[:\s]+)?(\d+\.?\d*(?:e[+-]?\d+)?)": ("lr", float),
        r"(?:batch[_\s]?size)[:\s]+(?:to[:\s]+)?(\d+)": ("batch_size", int),
        r"(?:epochs?)[:\s]+(?:to[:\s]+)?(\d+)": ("epochs", int),
        r"(?:seq(?:uence)?[_\s]?len(?:gth)?)[:\s]+(?:to[:\s]+)?(\d+)": ("seq_len", int),
        r"(?:patience)[:\s]+(?:to[:\s]+)?(\d+)": ("patience", int),
        r"(?:dropout)[:\s]+(?:to[:\s]+)?(\d+\.?\d*)": ("dropout", float),
        r"(?:hidden[_\s]?size)[:\s]+(?:to[:\s]+)?(\d+)": ("hidden_size", int),
    }
    
    changes = {}
    suggestion_lower = suggestion.lower()
    
    for pattern, (param_name, param_type) in patterns.items():
        match = re.search(pattern, suggestion_lower)
        if match:
            try:
                value = param_type(match.group(1))
                changes[param_name] = value
            except (ValueError, IndexError):
                continue
    
    if not changes:
        return False, "Could not parse hyperparam from suggestion", {}
    
    # Read and update config
    try:
        if not config_path.exists():
            return False, f"Config file not found: {config_path}", {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Navigate to buddy.train_defaults section
        if 'buddy' not in config:
            config['buddy'] = {}
        if 'train_defaults' not in config['buddy']:
            config['buddy']['train_defaults'] = {}
        
        # Apply changes
        applied = {}
        for param, value in changes.items():
            old_value = config['buddy']['train_defaults'].get(param)
            config['buddy']['train_defaults'][param] = value
            applied[param] = {'old': old_value, 'new': value}
        
        # Write back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True, f"Applied {len(applied)} changes", applied
        
    except Exception as e:
        return False, f"Failed to update config: {e}", {}


def gated_auto_approve_suggestions(
    improvements: List[ModelImprovement],
    config_path: Optional[Path] = None,
    auto_apply: bool = False,
    min_confidence: float = 0.7,
    min_sharpe_delta: float = 0.05,
) -> GatedApprovalResult:
    """Process suggestions through gated auto-approval.
    
    Only hyperparam suggestions that pass all gates are auto-approved.
    Architecture, feature, and training suggestions require manual review.
    
    Args:
        improvements: List of model improvements
        config_path: Path to config file (for auto-apply)
        auto_apply: Whether to actually apply approved changes
        min_confidence: Minimum confidence threshold
        min_sharpe_delta: Minimum Sharpe improvement threshold
        
    Returns:
        GatedApprovalResult with approved, pending, rejected lists
    """
    # Separate hyperparams from others
    hyperparams, others = filter_hyperparam_suggestions(improvements)
    
    approved = []
    rejected = []
    applied_changes = []
    changelog = []
    
    # Process hyperparams through gates
    for imp in hyperparams:
        passed, reason = check_approval_gates(
            imp,
            min_confidence=min_confidence,
            min_sharpe_delta=min_sharpe_delta,
        )
        
        if passed:
            approved.append(imp)
            changelog.append(f"✓ APPROVED: {imp.suggestion} ({reason})")
            
            # Auto-apply if requested and config path provided
            if auto_apply and config_path:
                success, msg, changes = apply_hyperparam_to_config(
                    imp.suggestion,
                    config_path,
                )
                if success:
                    applied_changes.append({
                        'suggestion': imp.suggestion,
                        'changes': changes,
                    })
                    changelog.append(f"  → Applied: {changes}")
                else:
                    changelog.append(f"  → Failed to apply: {msg}")
        else:
            rejected.append(imp)
            changelog.append(f"✗ REJECTED: {imp.suggestion} - {reason}")
    
    # Others require manual review
    for imp in others:
        changelog.append(f"⏸ PENDING REVIEW ({imp.category}): {imp.suggestion}")
    
    return GatedApprovalResult(
        approved=approved,
        pending_review=others,
        rejected=rejected,
        applied_changes=applied_changes,
        changelog=changelog,
    )


def suggest_model_improvements(
    architecture: BuddyArchitectureDescription,
    performance: PerformanceStats,
    known_weaknesses: List[str],
    model: str = "gpt-4",
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Generate meta-learning suggestions for model improvement.
    
    Analyzes current architecture and performance to suggest upgrades.
    
    Args:
        architecture: Description of current model architecture
        performance: Recent performance statistics
        known_weaknesses: List of known model weaknesses
        model: LLM model to use
        temperature: Generation temperature
        
    Returns:
        Dict with:
        - improvements: List of ModelImprovement dataclasses
        - priority_ranking: Ordered list of improvement indices
        - raw_response: Original LLM response
    """
    prompt = META_LEARNING_PROMPT.format(
        architecture_description=architecture.to_description(),
        performance_stats=performance.to_summary(),
        weaknesses=", ".join(known_weaknesses) if known_weaknesses else "None documented",
    )
    
    response = llm_call(
        prompt=prompt,
        system_prompt=META_LEARNING_SYSTEM,
        model=model,
        temperature=temperature,
    )
    
    if response is None:
        return {
            "improvements": [],
            "priority_ranking": [],
            "raw_response": None,
            "error": "LLM call failed",
        }
    
    # Parse JSON response
    improvements = []
    try:
        # Clean response if needed
        clean_response = response
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0]
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[1].split("```")[0]
        
        data = json.loads(clean_response)
        
        for item in data.get("improvements", []):
            # Extract confidence and sharpe_delta from expected_impact if possible
            confidence = item.get("confidence", 0.5)
            sharpe_delta = item.get("sharpe_delta", 0.0)
            
            # Try to parse sharpe improvement from expected_impact text
            expected_impact = item.get("expected_impact", "")
            if sharpe_delta == 0.0 and expected_impact:
                import re
                # Look for patterns like "5% improvement", "+10% sharpe", etc.
                sharpe_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:improvement|increase|gain|sharpe)", expected_impact.lower())
                if sharpe_match:
                    sharpe_delta = float(sharpe_match.group(1)) / 100
            
            improvements.append(ModelImprovement(
                category=item.get("category", "unknown"),
                suggestion=item.get("suggestion", ""),
                expected_impact=expected_impact,
                implementation_ease=item.get("implementation_ease", "medium"),
                test_plan=item.get("test_plan", ""),
                confidence=confidence,
                sharpe_delta=sharpe_delta,
            ))
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse meta-learning response: {e}")
        # Return raw response for manual parsing
        return {
            "improvements": [],
            "priority_ranking": [],
            "raw_response": response,
            "parse_error": str(e),
        }
    
    # Generate priority ranking based on ease and impact
    priority_ranking = list(range(len(improvements)))
    
    def priority_score(idx: int) -> float:
        imp = improvements[idx]
        ease_score = {"easy": 3, "medium": 2, "hard": 1}.get(imp.implementation_ease, 2)
        return ease_score
    
    priority_ranking.sort(key=priority_score, reverse=True)
    
    return {
        "improvements": improvements,
        "priority_ranking": priority_ranking,
        "raw_response": response,
    }


# =============================================================================
# INTELLIGENT MODE ORCHESTRATOR
# =============================================================================

class BuddyIntelligentMode:
    """Orchestrator for Buddy's intelligent mode features.
    
    Provides a unified interface to all LLM-enhanced capabilities while
    keeping Buddy's core predictions untouched.
    """
    
    def __init__(
        self,
        memory_path: Optional[Path] = None,
        default_model: str = "gpt-4",
        default_temperature: float = 0.2,
    ):
        """Initialize intelligent mode.
        
        Args:
            memory_path: Path for persisting online learning memory
            default_model: Default LLM model to use
            default_temperature: Default generation temperature
        """
        self.memory = OnlineLearningMemory(memory_path=memory_path)
        self.default_model = default_model
        self.default_temperature = default_temperature
        
        logger.info("BuddyIntelligentMode initialized")
    
    def get_reasoning(
        self,
        ticker: str,
        timeframe: str,
        buddy_raw: BuddyRawOutput,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get transparent reasoning for a trade signal.
        
        Basic reasoning without multi-modal or memory adaptation.
        """
        return generate_trade_reasoning(
            ticker=ticker,
            timeframe=timeframe,
            buddy_raw=buddy_raw,
            price_data=price_data,
            model=self.default_model,
            temperature=self.default_temperature,
        )
    
    def get_multi_modal_reasoning(
        self,
        ticker: str,
        timeframe: str,
        buddy_raw: BuddyRawOutput,
        context: MultiModalContext,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get multi-modal enhanced reasoning with news/sentiment fusion."""
        return multi_modal_fusion(
            ticker=ticker,
            timeframe=timeframe,
            buddy_raw=buddy_raw,
            context=context,
            price_data=price_data,
            model=self.default_model,
            temperature=self.default_temperature,
        )
    
    def get_adaptive_reasoning(
        self,
        ticker: str,
        timeframe: str,
        buddy_raw: BuddyRawOutput,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get reasoning with online learning adaptation."""
        return generate_adaptive_reasoning(
            ticker=ticker,
            timeframe=timeframe,
            buddy_raw=buddy_raw,
            memory=self.memory,
            price_data=price_data,
            model=self.default_model,
            temperature=self.default_temperature,
        )
    
    def get_full_intelligent_analysis(
        self,
        ticker: str,
        timeframe: str,
        buddy_raw: BuddyRawOutput,
        context: Optional[MultiModalContext] = None,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get complete intelligent analysis combining all features.
        
        This chains:
        1. Multi-modal fusion (if context provided)
        2. Adaptive reasoning with memory
        3. Self-improvement refinement
        """
        # Start with base reasoning
        if context:
            initial = self.get_multi_modal_reasoning(
                ticker=ticker,
                timeframe=timeframe,
                buddy_raw=buddy_raw,
                context=context,
                price_data=price_data,
            )
        else:
            initial = self.get_reasoning(
                ticker=ticker,
                timeframe=timeframe,
                buddy_raw=buddy_raw,
                price_data=price_data,
            )
        
        if initial.get("error"):
            return initial
        
        # Apply self-improvement
        initial_response = initial.get("reasoning") or initial.get("enhanced_reasoning")
        
        if initial_response:
            refined = self_improve_interpretation(
                query=f"Interpret Buddy's prediction for {ticker} on {timeframe}",
                max_iterations=2,
            )
            
            if refined.get("response"):
                initial["refined_reasoning"] = refined["response"]
                initial["refinement_iterations"] = refined.get("iterations", 0)
        
        return initial
    
    def record_trade_outcome(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        confidence: float,
        pnl: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a trade outcome for online learning."""
        self.memory.record_outcome(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            confidence=confidence,
            pnl=pnl,
            context=context,
        )
    
    def get_improvement_suggestions(
        self,
        architecture: BuddyArchitectureDescription,
        performance: PerformanceStats,
        known_weaknesses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get meta-learning suggestions for model improvement."""
        return suggest_model_improvements(
            architecture=architecture,
            performance=performance,
            known_weaknesses=known_weaknesses or [],
            model=self.default_model,
            temperature=0.3,
        )


# Singleton instance
_intelligent_mode: Optional[BuddyIntelligentMode] = None


def get_intelligent_mode(
    memory_path: Optional[Path] = None,
) -> BuddyIntelligentMode:
    """Get or create the global BuddyIntelligentMode instance."""
    global _intelligent_mode
    if _intelligent_mode is None:
        _intelligent_mode = BuddyIntelligentMode(memory_path=memory_path)
    return _intelligent_mode
