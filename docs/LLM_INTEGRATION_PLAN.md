# LLM Deep Integration Plan for Buddy

## Overview
Integrate LLM reasoning "under the hood" to make Buddy smarter without changing the core quant model. The LLM wraps and enhances - never replaces - the statistical edge.

---

## Current State (Implemented ✓)

| Feature | File | Status | Flag |
|---------|------|--------|------|
| Trade Validation | `buddy_intelligent_mode.py` | ✓ LLM can APPROVE/REJECT trades | `--intelligent` (default) |
| Size Adjustment | `main.py` | ✓ LLM adjusts position size 0.5x-1.5x | Automatic with validation |
| Explain Reasoning | `buddy_intelligent_mode.py` | ✓ `--explain` for verbose reasoning | `--explain` |
| Online Learning Memory | `buddy_intelligent_mode.py` | ✓ Learns from past trades | Automatic |
| Multi-Modal Fusion | `buddy_intelligent_mode.py` | ✓ News + Calendar + Price | Automatic |
| Meta-Learning Suggestions | `buddy_intelligent_mode.py` | ✓ `suggest-improvements` | `buddy suggest-improvements -i PAIR` |
| **Dynamic Thresholds** | `modular_inference.py` | ✓ LLM adjusts gates for edge cases | `--llm-enhance` |
| **Smart Sentiment** | `buddy_intelligent_mode.py` | ✓ LLM aggregates headlines | `--llm-enhance` |
| RSI Extreme Gate | `modular_inference.py` | ✓ Blocks trades at RSI <10/>90 | Automatic |
| Trend Contradiction Gate | `modular_inference.py` | ✓ Blocks counter-trend at ADX >35 | Automatic |

---

## Usage

### Basic (LLM validates trades)
```bash
buddy -i USD_JPY                    # LLM validates, can reject/approve
buddy -i USD_JPY --explain          # Also show verbose reasoning
```

### With Deep LLM Integration
```bash
buddy -i USD_JPY --llm-enhance      # Dynamic thresholds + smart sentiment
```

### Without LLM (pure quant)
```bash
buddy -i USD_JPY --no-intelligent   # Disable all LLM features
```

---

## Phase 1: Smarter Gate Decisions (High Value, Low Risk)

### 1.1 Dynamic Gate Thresholds
**Where:** `modular_inference.py` → `predict()` method

**Current:** Static thresholds (RSI <10, ADX >35, etc.)
**Enhanced:** LLM adjusts thresholds based on context

```python
# Before running gates, ask LLM for context-aware thresholds
def get_dynamic_thresholds(instrument: str, market_context: dict) -> dict:
    """LLM suggests threshold adjustments based on context."""
    prompt = f"""
    Instrument: {instrument}
    RSI: {market_context['rsi']:.1f}
    ADX: {market_context['adx']:.1f}
    Volatility: {market_context['atr_pct']:.2%}
    Recent News: {market_context.get('news_summary', 'None')}
    
    Current thresholds: RSI extreme <10/>90, ADX trend >35
    
    Should thresholds be adjusted? Respond JSON:
    {{"rsi_extreme_low": 10, "rsi_extreme_high": 90, "adx_trend": 35, "reason": "..."}}
    """
    # Only call for edge cases (RSI 8-15 or 85-92 range)
```

**Files to modify:**
- `modular_inference.py` - Add threshold adjustment call
- `buddy_intelligent_mode.py` - Add `get_dynamic_thresholds()` function

### 1.2 Regime-Aware Gate Bypass
**Where:** `modular_inference.py` → gate evaluation

**Idea:** In certain regimes (news-driven, range-bound), some gates should be weighted differently.

```python
# LLM can suggest gate weight adjustments
{
    "regime": "news_driven",
    "gate_adjustments": {
        "momentum_gate": 0.5,  # Less important during news
        "risk_gate": 1.5,      # More important
        "sentiment_gate": 2.0  # Much more important
    }
}
```

---

## Phase 2: Enhanced Sentiment Analysis (Medium Value)

### 2.1 LLM News Interpretation
**Where:** `market_intelligence.py` → `NewsSentimentAnalyzer`

**Current:** VADER/FinBERT scores headlines independently
**Enhanced:** LLM reads ALL headlines together, understands context

```python
def llm_aggregate_sentiment(headlines: List[str], instrument: str) -> dict:
    """LLM interprets news holistically, not just per-headline."""
    prompt = f"""
    Headlines for {instrument}:
    {chr(10).join(f'- {h}' for h in headlines[:10])}
    
    1. What's the NET sentiment direction? (strong_bullish/bullish/neutral/bearish/strong_bearish)
    2. Are any headlines contradictory?
    3. Is this event-driven or general flow?
    4. Confidence (0-100)?
    
    JSON: {{"direction": "...", "contradictions": false, "event_driven": false, "confidence": 70}}
    """
```

**Files to modify:**
- `market_intelligence.py` - Add `llm_aggregate_sentiment()`
- `modular_inference.py` - Use aggregated sentiment for gate

### 2.2 Cross-Pair Correlation Detection
**Where:** New function in `market_intelligence.py`

**Idea:** LLM notices "USD weakness across multiple pairs"

```python
def detect_cross_pair_themes(pair_sentiments: Dict[str, float]) -> dict:
    """LLM identifies themes across pairs."""
    # EUR_USD bullish + GBP_USD bullish + USD_JPY bearish = USD weakness
    prompt = f"""
    Current sentiment by pair:
    {json.dumps(pair_sentiments, indent=2)}
    
    Identify any cross-pair themes:
    - Currency strength/weakness (e.g., "USD weak across board")
    - Risk-on/risk-off signals
    - Divergences to watch
    
    JSON: {{"theme": "...", "affected_pairs": [...], "confidence": 0.7}}
    """
```

---

## Phase 3: Position Sizing Intelligence (High Value)

### 3.1 Confidence-Weighted Sizing
**Where:** `position_sizing.py` or `main.py` execution flow

**Current:** Fixed position size based on account risk
**Enhanced:** LLM adjusts size based on conviction

```python
def get_llm_size_adjustment(signal: TradingSignal, context: dict) -> float:
    """LLM suggests position size multiplier (0.25x to 1.5x)."""
    prompt = f"""
    Signal: {signal.direction} {signal.instrument}
    TCN Probability: {signal.tcn_probability:.2f}
    Gate Confidence: {context['gate_confidence']}
    Recent Win Rate: {context['recent_win_rate']:.1%}
    News: {context.get('news_summary', 'Neutral')}
    
    Based on signal quality and context, suggest size multiplier:
    - 0.25x: Very uncertain, test position
    - 0.5x: Below average conviction
    - 1.0x: Standard conviction (default)
    - 1.25x: High conviction setup
    - 1.5x: Exceptional setup (rare)
    
    JSON: {{"size_multiplier": 1.0, "reason": "..."}}
    """
```

**Already partially implemented** in `validate_trade_with_llm()` - just need to use the `size_multiplier` return value.

### 3.2 Drawdown-Aware Sizing
**Where:** `risk_management.py` or position sizing flow

**Idea:** LLM considers recent drawdown when sizing

```python
# If recent drawdown > 3%, LLM might suggest:
{"size_multiplier": 0.5, "reason": "Reducing size during drawdown recovery"}
```

---

## Phase 4: Trade Journal & Pattern Learning (Medium-Long Term)

### 4.1 Automatic Trade Journaling
**Where:** New `trade_journal.py` module

**Idea:** After each trade closes, LLM writes a journal entry analyzing what happened.

```python
def journal_closed_trade(trade: ClosedTrade) -> str:
    """LLM analyzes why trade won/lost."""
    prompt = f"""
    Trade: {trade.direction} {trade.instrument}
    Entry: {trade.entry_price} at {trade.entry_time}
    Exit: {trade.exit_price} at {trade.exit_time}
    P/L: {trade.pnl_pips} pips ({trade.pnl_pct:.2%})
    
    Context at entry:
    - RSI: {trade.entry_context['rsi']}
    - News sentiment: {trade.entry_context['sentiment']}
    - Gates passed: {trade.entry_context['gates']}
    
    What happened after entry:
    - Max favorable: {trade.max_favorable_pips} pips
    - Max adverse: {trade.max_adverse_pips} pips
    
    Write a brief trade journal entry:
    1. What worked or didn't work?
    2. Was the entry timing good?
    3. What lesson to remember?
    """
```

### 4.2 Pattern Recognition Memory
**Where:** `buddy_intelligent_mode.py` → `OnlineLearningMemory`

**Idea:** LLM identifies recurring patterns (e.g., "USD_JPY reversals during Tokyo open")

```python
def identify_patterns(trade_history: List[Trade]) -> List[str]:
    """LLM finds patterns across trade history."""
    # Group trades by time, pair, outcome
    # Ask LLM to find commonalities
```

---

## Phase 5: Pre-Trade Risk Assessment (High Value) ✓ IMPLEMENTED

### 5.1 Economic Calendar Intelligence ✓
**Where:** `market_intelligence.py` → `assess_event_risk()`

**Current:** Shows events, blocks trades near high-impact
**Enhanced:** LLM interprets expected vs actual, impact direction

```python
def assess_event_risk(events: List[EconomicEvent], instrument: str) -> EventRiskAssessment:
    """LLM assesses upcoming event risk."""
    # Returns EventRiskAssessment with:
    # - avoid_trade: bool
    # - volatility: 'low'|'medium'|'high'|'extreme'
    # - bias: 'bullish'|'bearish'|'none'
    # - surprise_direction: 'beat'|'miss'|'inline' (for past events)
```

### 5.2 Multi-Factor Risk Score ✓
**Where:** `market_intelligence.py` → `compute_llm_risk_score()`

```python
def compute_llm_risk_score(context: dict) -> MultiFactorRiskScore:
    """LLM computes 0-100 risk score considering ALL factors."""
    # Returns MultiFactorRiskScore with:
    # - score: 0-100 overall risk
    # - volatility_risk, event_risk, drawdown_risk, sentiment_risk, time_of_day_risk
    # - recommendation: 'trade'|'reduce_size'|'avoid'
    # - size_multiplier: 0.25-1.0
```

### 5.3 Integration ✓
**Where:** `buddy_intelligent_mode.py` → `validate_trade_with_risk_assessment()`

Combines event risk, multi-factor risk score, and LLM validation into a single function.

---

## Implementation Priority

### Immediate (This Week)
1. **Use existing `size_multiplier`** from `validate_trade_with_llm()` - already returns it, just need to apply
2. **Add `--auto-size` flag** to use LLM sizing suggestions

### Short Term (Next 2 Weeks)
3. **LLM aggregate sentiment** - Better than per-headline scores
4. **Dynamic gate thresholds** - Only for edge cases (fast, low cost)

### Medium Term (1 Month)
5. **Trade journaling** - Automatic post-trade analysis
6. **Pattern memory** - Learn from history

### Long Term (Ongoing)
7. **Cross-pair correlation** - USD strength detection
8. **Regime-aware weights** - Adjust strategy by regime

---

## Cost/Performance Considerations

| Feature | LLM Calls/Trade | Latency | Cost Est |
|---------|----------------|---------|----------|
| Trade validation | 1 | ~500ms | $0.002 |
| Dynamic thresholds | 0-1 (edge cases) | ~300ms | $0.001 |
| Aggregate sentiment | 1 | ~400ms | $0.002 |
| Size adjustment | 0 (included in validation) | 0 | $0 |
| Trade journal | 1 (post-close) | ~800ms | $0.003 |

**Total per trade: ~$0.005-0.008** (Claude Sonnet pricing)

---

## File Changes Summary

| File | Changes |
|------|---------|
| `modular_inference.py` | Add dynamic threshold hook, use LLM risk score |
| `buddy_intelligent_mode.py` | Add `get_dynamic_thresholds()`, improve `validate_trade_with_llm()` |
| `market_intelligence.py` | Add `llm_aggregate_sentiment()`, cross-pair themes |
| `main.py` | Apply `size_multiplier` from LLM, add `--auto-size` |
| `trade_journal.py` (new) | Trade journaling and pattern recognition |

---

## Quick Win: Apply Size Multiplier Now

The `validate_trade_with_llm()` already returns `size_multiplier` but it's not being used. Let's wire it up:

```python
# In main.py, after LLM validation:
if llm_validation.approve:
    final_size = base_position_size * llm_validation.size_multiplier
    # Use final_size for execution
```

This is a 5-line change for immediate value.
