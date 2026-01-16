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
        
        if not openai.api_key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.debug("OPENAI_API_KEY not set, LLM call skipped")
                return None
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
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
    logger.warning("self_refine module not available")
    
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
    
    # Parse response to extract key fields
    trade_call = "NO_TRADE"
    confidence_level = "Low"
    
    response_lower = response.lower()
    
    # Extract trade call
    if "buy" in response_lower and "sell" not in response_lower:
        trade_call = "BUY"
    elif "sell" in response_lower and "buy" not in response_lower:
        trade_call = "SELL"
    elif "hold" in response_lower:
        trade_call = "HOLD"
    
    # Extract confidence level
    if "high" in response_lower:
        confidence_level = "High"
    elif "medium" in response_lower or "med" in response_lower:
        confidence_level = "Med"
    
    return {
        "reasoning": response,
        "trade_call": trade_call,
        "confidence_level": confidence_level,
        "parsed_sections": _parse_reasoning_sections(response),
    }


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
        cumsum = [sum(pnls[:i+1]) for i in range(len(pnls))]
        
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
            improvements.append(ModelImprovement(
                category=item.get("category", "unknown"),
                suggestion=item.get("suggestion", ""),
                expected_impact=item.get("expected_impact", ""),
                implementation_ease=item.get("implementation_ease", "medium"),
                test_plan=item.get("test_plan", ""),
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
# — Raynergy-svg —
