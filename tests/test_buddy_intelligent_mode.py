"""Tests for Buddy Intelligent Mode - LLM Wrapper Enhancements."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from buddy_intelligent_mode import (
    BuddyArchitectureDescription,
    BuddyIntelligentMode,
    BuddyRawOutput,
    ModelImprovement,
    MultiModalContext,
    OnlineLearningMemory,
    PerformanceStats,
    TradeLessonMemory,
    _extract_modality_weights,
    _parse_reasoning_sections,
    generate_adaptive_reasoning,
    generate_trade_reasoning,
    get_intelligent_mode,
    llm_call,
    multi_modal_fusion,
    set_llm_call_function,
    suggest_model_improvements,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def mock_llm_call_success(
    prompt: str,
    system_prompt: str = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> str:
    """Mock LLM that returns successful responses.
    
    This mock dispatches based on distinctive keywords in each prompt type.
    The order of checks matters:
    - Meta-learning prompts are checked first because they contain "sentiment" 
      as an example feature, which would otherwise match the multi-modal check.
    - Each prompt type is matched by its most distinctive identifying keywords.
    """
    prompt_lower = prompt.lower()
    
    # Check for meta-learning/improvements FIRST (before sentiment check)
    # because meta-learning prompt contains the word "sentiment" in examples
    if "current architecture" in prompt_lower or "propose 3-5" in prompt_lower:
        return json.dumps({
            "improvements": [
                {
                    "category": "hyperparam",
                    "suggestion": "Increase lookback window to 90 bars",
                    "expected_impact": "Better trend capture in longer cycles",
                    "implementation_ease": "easy",
                    "test_plan": "A/B test on 3-month backtest"
                },
                {
                    "category": "feature",
                    "suggestion": "Add sentiment embeddings from news",
                    "expected_impact": "Capture market sentiment shifts",
                    "implementation_ease": "medium",
                    "test_plan": "Compare with/without on recent data"
                }
            ]
        })
    
    if "reasoning" in prompt_lower or "interpret" in prompt_lower:
        return """**Market Context** - EUR/USD showing upward trend with moderate volatility.

**Buddy Signal Breakdown**
- Raw: Confidence 75%, Prob 68%
- Why this score: High confidence from momentum alignment with trend.

**Key Drivers**
- Bullish: RSI divergence, MACD crossover
- Support at 1.0800 holding

**Risks & Alternatives**
- Risk: NFP data tomorrow could cause reversal
- Alternative: Wait for confirmation

**Reasoning Summary**
- Score high because momentum > mean-reversion signals, leading to BUY despite event risk.

**Final Call**
- Trade: BUY
- Translated Confidence: High
- Target: 1.0920"""
    
    if "multi-modal" in prompt_lower or "news/sentiment" in prompt_lower:
        return """**Multi-Modal Fusion**
- News sentiment (+0.3) aligns with Buddy's bullish signal.
- Event risk: NFP tomorrow — reduce confidence.
- Buddy 70% weight, sentiment boosts to 80% effective.

**Market Context** - Upward trend continues.

**Final Call**
- Trade: BUY
- Translated Confidence: Medium"""
    
    if "lessons" in prompt_lower or "outcomes" in prompt_lower:
        return json.dumps([
            "High volatility regimes require reduced position size",
            "Sentiment overrides improved win rate by 15%",
            "Low confidence trades (<60%) underperformed"
        ])
    
    if "adaptation" in prompt_lower or "memory" in prompt_lower:
        return """**Adaptation from Experience**
- Past high-vol trades had 30% win rate - reducing size by 50%
- Sentiment override improved recent wins - maintaining position

**Market Context** - Similar to past setups where caution paid off.

**Final Call**
- Trade: BUY
- Translated Confidence: Medium"""
    
    return "Default LLM response"


def mock_llm_call_failure(
    prompt: str,
    system_prompt: str = None,
    model: str = "gpt-4",
    temperature: float = 0.2,
) -> None:
    """Mock LLM that always fails."""
    return None


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestBuddyRawOutput(unittest.TestCase):
    """Test BuddyRawOutput dataclass."""

    def test_basic_creation(self):
        """Can create basic output."""
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        self.assertEqual(raw.confidence, 0.75)
        self.assertEqual(raw.prediction, 1.0850)

    def test_to_dict_basic(self):
        """to_dict includes required fields."""
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        d = raw.to_dict()
        self.assertEqual(d["confidence"], 0.75)
        self.assertEqual(d["prediction"], 1.0850)

    def test_to_dict_with_optional(self):
        """to_dict includes optional fields when set."""
        raw = BuddyRawOutput(
            confidence=0.75,
            prediction=1.0850,
            last_price=1.0800,
            direction="long",
            features={"rsi": 65.0},
        )
        d = raw.to_dict()
        self.assertEqual(d["last_price"], 1.0800)
        self.assertAlmostEqual(d["delta"], 0.0050, places=4)
        self.assertEqual(d["direction"], "long")
        self.assertEqual(d["key_features"]["rsi"], 65.0)


class TestMultiModalContext(unittest.TestCase):
    """Test MultiModalContext dataclass."""

    def test_creation(self):
        """Can create context."""
        ctx = MultiModalContext(
            news_summaries=["EUR rises on ECB decision"],
            sentiment_score=0.5,
            upcoming_events=["NFP Friday"],
        )
        self.assertEqual(ctx.sentiment_score, 0.5)
        self.assertEqual(len(ctx.news_summaries), 1)

    def test_to_dict(self):
        """to_dict returns expected structure."""
        ctx = MultiModalContext(
            news_summaries=["News 1"],
            sentiment_score=0.3,
        )
        d = ctx.to_dict()
        self.assertIn("news_summaries", d)
        self.assertIn("aggregate_sentiment", d)
        self.assertEqual(d["aggregate_sentiment"], 0.3)


class TestTradeLessonMemory(unittest.TestCase):
    """Test TradeLessonMemory dataclass."""

    def test_empty_summarize(self):
        """Empty memory returns default message."""
        mem = TradeLessonMemory()
        summary = mem.summarize()
        self.assertIn("No significant past lessons", summary)

    def test_summarize_with_lessons(self):
        """Summarize includes lessons and stats."""
        mem = TradeLessonMemory(
            lessons=["Lesson 1", "Lesson 2"],
            win_rate=0.60,
            recent_drawdown=0.10,
            regime_notes="High volatility",
        )
        summary = mem.summarize()
        self.assertIn("60", summary)
        self.assertIn("Lesson 1", summary)
        self.assertIn("High volatility", summary)


# =============================================================================
# REASONING TESTS
# =============================================================================

class TestGenerateTradeReasoning(unittest.TestCase):
    """Test generate_trade_reasoning function."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def tearDown(self):
        """Reset mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_successful_reasoning(self):
        """Generates reasoning successfully."""
        raw = BuddyRawOutput(
            confidence=0.75,
            prediction=1.0850,
            last_price=1.0800,
        )
        
        result = generate_trade_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
        )
        
        self.assertIsNotNone(result["reasoning"])
        self.assertEqual(result["trade_call"], "BUY")
        self.assertEqual(result["confidence_level"], "High")

    def test_failed_llm_call(self):
        """Handles LLM failure gracefully."""
        set_llm_call_function(mock_llm_call_failure)
        
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        
        result = generate_trade_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
        )
        
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["trade_call"], "NO_TRADE")
        self.assertIn("error", result)

    def test_with_price_data(self):
        """Includes price data in prompt."""
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        price_data = {"trend": "up", "volatility": "medium"}
        
        result = generate_trade_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
            price_data=price_data,
        )
        
        self.assertIsNotNone(result["reasoning"])


class TestParseReasoningSections(unittest.TestCase):
    """Test _parse_reasoning_sections function."""

    def test_parses_sections(self):
        """Parses sections correctly."""
        text = """**Market Context**
Upward trend with moderate volatility.

**Key Drivers**
Bullish momentum.

**Final Call**
BUY"""
        
        sections = _parse_reasoning_sections(text)
        
        self.assertIn("Market Context", sections)
        self.assertIn("Key Drivers", sections)
        self.assertIn("Final Call", sections)
        self.assertIn("Upward trend", sections["Market Context"])

    def test_empty_text(self):
        """Handles empty text."""
        sections = _parse_reasoning_sections("")
        self.assertEqual(sections, {})


# =============================================================================
# MULTI-MODAL FUSION TESTS
# =============================================================================

class TestMultiModalFusion(unittest.TestCase):
    """Test multi_modal_fusion function."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_successful_fusion(self):
        """Generates multi-modal fusion successfully."""
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        context = MultiModalContext(
            news_summaries=["EUR rises on ECB decision"],
            sentiment_score=0.3,
            upcoming_events=["NFP Friday"],
        )
        
        result = multi_modal_fusion(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
            context=context,
        )
        
        self.assertIsNotNone(result["enhanced_reasoning"])
        self.assertIn("trade_call", result)
        self.assertIn("modality_weights", result)

    def test_event_risk_detection(self):
        """Detects event risk in response."""
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        context = MultiModalContext(
            upcoming_events=["NFP Release"],
            event_impact_flags=["high"],
        )
        
        result = multi_modal_fusion(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
            context=context,
        )
        
        # Response mentions NFP, so event_risk should be detected
        self.assertIsNotNone(result.get("event_risk"))


class TestExtractModalityWeights(unittest.TestCase):
    """Test _extract_modality_weights function."""

    def test_extracts_buddy_weight(self):
        """Extracts Buddy weight from text."""
        text = "Buddy 70% weight, sentiment 20%"
        weights = _extract_modality_weights(text)
        self.assertEqual(weights["buddy"], 0.70)

    def test_extracts_sentiment_weight(self):
        """Extracts sentiment weight from text."""
        text = "Sentiment 25% contribution"
        weights = _extract_modality_weights(text)
        self.assertEqual(weights["sentiment"], 0.25)

    def test_defaults_when_not_found(self):
        """Returns defaults when weights not found."""
        text = "No explicit weights mentioned"
        weights = _extract_modality_weights(text)
        self.assertEqual(weights["buddy"], 0.7)
        self.assertEqual(weights["sentiment"], 0.15)


# =============================================================================
# ONLINE LEARNING MEMORY TESTS
# =============================================================================

class TestOnlineLearningMemory(unittest.TestCase):
    """Test OnlineLearningMemory class."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_record_outcome(self):
        """Records outcome correctly."""
        memory = OnlineLearningMemory()
        memory.record_outcome(
            trade_id="T001",
            instrument="EUR_USD",
            direction="long",
            confidence=0.75,
            pnl=100.0,
        )
        
        self.assertEqual(len(memory._outcomes), 1)
        self.assertEqual(memory._outcomes[0]["pnl"], 100.0)

    def test_extract_lessons_insufficient_data(self):
        """Returns message when insufficient data."""
        memory = OnlineLearningMemory()
        lessons = memory.extract_lessons()
        self.assertIn("Insufficient data", lessons[0])

    def test_extract_lessons_with_data(self):
        """Extracts lessons when data available."""
        memory = OnlineLearningMemory()
        
        # Add enough outcomes
        for i in range(10):
            memory.record_outcome(
                trade_id=f"T{i:03d}",
                instrument="EUR_USD",
                direction="long",
                confidence=0.70,
                pnl=100.0 if i % 2 == 0 else -50.0,
            )
        
        lessons = memory.extract_lessons()
        self.assertIsInstance(lessons, list)
        self.assertGreater(len(lessons), 0)

    def test_get_memory_context(self):
        """Returns memory context."""
        memory = OnlineLearningMemory()
        memory.record_outcome("T001", "EUR_USD", "long", 0.75, 100.0)
        memory.record_outcome("T002", "EUR_USD", "short", 0.60, -50.0)
        
        context = memory.get_memory_context()
        self.assertIsInstance(context, TradeLessonMemory)
        self.assertEqual(context.win_rate, 0.5)

    def test_persistence(self):
        """Memory persists to disk."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            
            # Create and populate memory
            memory1 = OnlineLearningMemory(memory_path=path)
            memory1.record_outcome("T001", "EUR_USD", "long", 0.75, 100.0)
            
            # Load in new instance
            memory2 = OnlineLearningMemory(memory_path=path)
            self.assertEqual(len(memory2._outcomes), 1)


class TestGenerateAdaptiveReasoning(unittest.TestCase):
    """Test generate_adaptive_reasoning function."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_generates_adaptive_reasoning(self):
        """Generates reasoning with adaptation."""
        memory = OnlineLearningMemory()
        memory._lessons = ["Past lesson 1", "Past lesson 2"]
        
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        
        result = generate_adaptive_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
            memory=memory,
        )
        
        self.assertIsNotNone(result["adaptive_reasoning"])
        self.assertIn("memory_summary", result)


# =============================================================================
# META-LEARNING TESTS
# =============================================================================

class TestSuggestModelImprovements(unittest.TestCase):
    """Test suggest_model_improvements function."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_generates_suggestions(self):
        """Generates improvement suggestions."""
        architecture = BuddyArchitectureDescription(
            model_type="Ensemble",
            lookback_window=60,
        )
        performance = PerformanceStats(
            win_rate=0.52,
            profit_factor=1.1,
        )
        
        result = suggest_model_improvements(
            architecture=architecture,
            performance=performance,
            known_weaknesses=["Regime shifts", "News blindness"],
        )
        
        self.assertIsInstance(result["improvements"], list)
        self.assertGreater(len(result["improvements"]), 0)

    def test_improvement_structure(self):
        """Improvement has expected structure."""
        architecture = BuddyArchitectureDescription()
        performance = PerformanceStats()
        
        result = suggest_model_improvements(
            architecture=architecture,
            performance=performance,
            known_weaknesses=[],
        )
        
        if result["improvements"]:
            imp = result["improvements"][0]
            self.assertIsInstance(imp, ModelImprovement)
            self.assertTrue(hasattr(imp, "category"))
            self.assertTrue(hasattr(imp, "suggestion"))


class TestBuddyArchitectureDescription(unittest.TestCase):
    """Test BuddyArchitectureDescription dataclass."""

    def test_to_description(self):
        """Generates description string."""
        arch = BuddyArchitectureDescription(
            model_type="TCN Ensemble",
            lookback_window=90,
            hidden_size=256,
        )
        desc = arch.to_description()
        
        self.assertIn("TCN Ensemble", desc)
        self.assertIn("90", desc)
        self.assertIn("256", desc)


class TestPerformanceStats(unittest.TestCase):
    """Test PerformanceStats dataclass."""

    def test_to_summary(self):
        """Generates summary string."""
        stats = PerformanceStats(
            win_rate=0.55,
            profit_factor=1.3,
            sharpe_ratio=1.5,
        )
        summary = stats.to_summary()
        
        self.assertIn("55", summary)
        self.assertIn("1.3", summary)
        self.assertIn("1.5", summary)


# =============================================================================
# INTELLIGENT MODE ORCHESTRATOR TESTS
# =============================================================================

class TestBuddyIntelligentMode(unittest.TestCase):
    """Test BuddyIntelligentMode orchestrator."""

    def setUp(self):
        """Set up mock LLM."""
        set_llm_call_function(mock_llm_call_success)

    def test_initialization(self):
        """Initializes correctly."""
        mode = BuddyIntelligentMode()
        self.assertIsNotNone(mode.memory)
        self.assertEqual(mode.default_model, "gpt-4")

    def test_get_reasoning(self):
        """get_reasoning returns result."""
        mode = BuddyIntelligentMode()
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        
        result = mode.get_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
        )
        
        self.assertIsNotNone(result["reasoning"])

    def test_get_multi_modal_reasoning(self):
        """get_multi_modal_reasoning returns result."""
        mode = BuddyIntelligentMode()
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        context = MultiModalContext(sentiment_score=0.3)
        
        result = mode.get_multi_modal_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
            context=context,
        )
        
        self.assertIsNotNone(result["enhanced_reasoning"])

    def test_get_adaptive_reasoning(self):
        """get_adaptive_reasoning returns result."""
        mode = BuddyIntelligentMode()
        raw = BuddyRawOutput(confidence=0.75, prediction=1.0850)
        
        result = mode.get_adaptive_reasoning(
            ticker="EUR_USD",
            timeframe="H1",
            buddy_raw=raw,
        )
        
        self.assertIn("adaptive_reasoning", result)

    def test_record_trade_outcome(self):
        """Records trade outcome to memory."""
        mode = BuddyIntelligentMode()
        mode.record_trade_outcome(
            trade_id="T001",
            instrument="EUR_USD",
            direction="long",
            confidence=0.75,
            pnl=100.0,
        )
        
        self.assertEqual(len(mode.memory._outcomes), 1)

    def test_get_improvement_suggestions(self):
        """Gets improvement suggestions."""
        mode = BuddyIntelligentMode()
        arch = BuddyArchitectureDescription()
        perf = PerformanceStats()
        
        result = mode.get_improvement_suggestions(
            architecture=arch,
            performance=perf,
        )
        
        self.assertIn("improvements", result)

    def test_with_persistence(self):
        """Works with memory persistence."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            
            mode = BuddyIntelligentMode(memory_path=path)
            mode.record_trade_outcome("T001", "EUR_USD", "long", 0.75, 100.0)
            
            self.assertTrue(path.exists())


class TestGetIntelligentMode(unittest.TestCase):
    """Test get_intelligent_mode singleton."""

    def test_returns_instance(self):
        """Returns BuddyIntelligentMode instance."""
        import buddy_intelligent_mode
        buddy_intelligent_mode._intelligent_mode = None  # Reset
        
        mode = get_intelligent_mode()
        self.assertIsInstance(mode, BuddyIntelligentMode)

    def test_returns_same_instance(self):
        """Returns same instance on repeated calls."""
        import buddy_intelligent_mode
        buddy_intelligent_mode._intelligent_mode = None  # Reset
        
        mode1 = get_intelligent_mode()
        mode2 = get_intelligent_mode()
        self.assertIs(mode1, mode2)


# =============================================================================
# LLM CALL INFRASTRUCTURE TESTS
# =============================================================================

class TestSetLlmCallFunction(unittest.TestCase):
    """Test LLM call function configuration."""

    def tearDown(self):
        """Reset to success mock."""
        set_llm_call_function(mock_llm_call_success)

    def test_custom_function_used(self):
        """Custom LLM function is used."""
        call_log = []
        
        def custom_llm(prompt, system_prompt=None, model="gpt-4", temperature=0.2):
            call_log.append(prompt)
            return "Custom response"
        
        set_llm_call_function(custom_llm)
        
        result = llm_call("Test prompt")
        
        self.assertEqual(result, "Custom response")
        self.assertEqual(len(call_log), 1)
        self.assertIn("Test prompt", call_log[0])


if __name__ == "__main__":
    unittest.main()
