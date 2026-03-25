# Phase 52 Research: Live Pipeline Wiring, Confidence Decomposition, Adaptive R:R, Entry Timing, Drawdown Behavior

## RESEARCH FINDINGS & IMPLEMENTATION RECOMMENDATIONS

---

## 1. LIVE PIPELINE WIRING FOR PHASE 51 MODULES

### Current State (Phase 51)
Four new modules are built but may have incomplete wiring into the execution flow:
- **Isotonic Calibration** — Monotonic probability calibration (src/scanner/confidence_calibration.py)
- **Pattern Gate** — Journal-mined pattern application (src/scanner/pattern_gate.py)
- **Setup Quality Filter** — Composite reject filter (src/scanner/setup_quality.py)
- **Tranche Tracker** — Partial-close SL management (src/scanner/tranche_tracker.py)
- **Walk-Forward Retrainer** — Rolling window model validation (src/scanner/analysis/walk_forward.py)

### Correct Gate Chain Order

The execution pipeline should follow this sequence:

```
SCAN SIGNAL
    ↓
1. Gate Evaluator (confidence, momentum, risk gates)
    ↓
2. Agent Team Verdicts (12 agents → weighted_vote_score)
    ↓
3. Isotonic Calibration (monotonic recalibration of confidence)
    ↓
4. Pattern Gate (journal patterns → confidence delta)
    ↓
5. Setup Quality Filter (composite score check)
    ↓ [IF ALL PASS]
6. Execution Manager (ATR-based SL/TP, position sizing)
    ↓
7. Tranche Tracker (register for multi-leg exit)
    ↓
8. Walk-Forward Monitor (validate model performance in real-time)
    ↓ [POST-TRADE]
9. RL Weight Update (agent weight sync from outcome)
```

### Implementation Pattern: Python Integration

**File: src/scanner/engine.py → run_cycle() method**

```python
def run_cycle(self) -> ScanResult:
    """Core scan cycle with Phase 51+ wiring."""
    
    # 1. SCAN & GATE EVALUATION
    pair_analysis = self._scan_pair(pair)  # Returns PairAnalysis with base confidence
    gate_result = self._gate_evaluator.evaluate(pair_analysis)
    
    if not gate_result.all_pass:
        return ScanResult(passed=False, reason="Gate check failed")
    
    confidence = gate_result.confidence  # ~0.55 after base gates
    
    # 2. AGENT TEAM EVALUATION (already integrated)
    agent_result = self._agent_team.evaluate(pair_analysis)
    # agent_result.verdicts: List[AgentVerdict]
    # agent_result.weighted_vote_score: float (0.0-1.0)
    # agent_result.ensemble_conflict: float (0.0-1.0, higher = more disagreement)
    
    confidence = (confidence + agent_result.weighted_vote_score) / 2.0
    
    # *** PHASE 51A: ISOTONIC CALIBRATION ***
    if self._confidence_calibrator:
        cal_result = self._confidence_calibrator.calibrate(
            raw_confidence=confidence,
            regime=gate_result.regime,
            agent_verdicts=agent_result.verdicts,
            duration_bars_held=0,  # New trade = 0 bars
        )
        confidence = cal_result.calibrated_confidence
        # cal_result also includes: platt_score, time_decay, meta_confidence
        logger.info(f"Phase 51a: Isotonic calibration {confidence:.4f} "
                   f"(raw: {agent_result.weighted_vote_score:.4f})")
    
    # *** PHASE 51B: PATTERN GATE ***
    if self._pattern_gate:
        pg_result = self._pattern_gate.evaluate(
            pair=pair_analysis.pair,
            direction=gate_result.direction,
            confidence=confidence,
            expected_duration_hours=12.0,  # From duration pattern (480-1440min sweet spot)
        )
        confidence += pg_result.confidence_delta
        if pg_result.warnings:
            logger.warning(f"Phase 51b: Pattern gate warnings: {pg_result.warnings}")
        logger.info(f"Phase 51b: Pattern gate {confidence:.4f} "
                   f"(delta: {pg_result.confidence_delta:+.4f})")
    
    # *** PHASE 51C: SETUP QUALITY FILTER ***
    if self._setup_quality_filter:
        sqf_result = self._setup_quality_filter.evaluate(
            pair=pair_analysis.pair,
            confidence=confidence,
            agent_disagreement=agent_result.ensemble_conflict,
            uncertainty_score=agent_result.uncertainty_score,
            pair_recent_win_rate=self._get_pair_recent_win_rate(pair_analysis.pair),
        )
        if not sqf_result.passed:
            logger.warning(f"Phase 51c: Setup rejected: {sqf_result.rejection_reasons}")
            return ScanResult(passed=False, reason=f"Setup quality: {sqf_result.rejection_reasons}")
        confidence = sqf_result.quality_score  # Quality score now becomes confidence
        logger.info(f"Phase 51c: Setup quality PASS (score: {sqf_result.quality_score:.4f})")
    
    # 3. EXECUTE TRADE
    exec_result = self._executor.execute_trade(
        pair=pair_analysis.pair,
        direction=gate_result.direction,
        confidence=confidence,
        setup_context=pair_analysis,
    )
    
    if not exec_result.success:
        return ScanResult(passed=False, reason=f"Execution failed: {exec_result.error}")
    
    # *** PHASE 51D: TRANCHE TRACKER REGISTRATION ***
    if self._tranche_tracker:
        self._tranche_tracker.register_trade(
            trade_id=exec_result.trade_id,
            pair=pair_analysis.pair,
            entry_price=exec_result.fill_price,
            original_sl=exec_result.sl_price,
            direction=gate_result.direction,
        )
        logger.info(f"Phase 51d: Trade {exec_result.trade_id} registered for tranche tracking")
    
    # *** PHASE 51E: WALK-FORWARD VALIDATION ***
    if self._walk_forward_validator:
        try:
            wf_result = self._walk_forward_validator.get_last_result(pair_analysis.pair)
            if wf_result and not wf_result.passed:
                logger.warning(f"Phase 51e: {pair_analysis.pair} failed walk-forward validation "
                             f"(accuracy: {wf_result.mean_accuracy:.4f} < {wf_result.pass_threshold})")
                # DECISION: Log warning but don't block (informational only in live trading)
                # Could implement regime-based model degradation signal
        except Exception as e:
            logger.debug(f"Phase 51e: Walk-forward check skipped: {e}")
    
    return ScanResult(passed=True, trade_result=exec_result, ...)
```

### Wiring Checklist (Live Verification)

To avoid dead-code wiring (from .claude/rules/improvement.md — Live Wiring Verification Gates):

- [x] **Isotonic Calibrator**: Instantiated in `__init__`, called in run_cycle(), output used to update confidence
- [x] **Pattern Gate**: Config flag `enable_pattern_gate` in ScannerConfig, evaluate() called, confidence_delta applied
- [x] **Setup Quality Filter**: Config flag `enable_setup_quality_filter`, evaluate() called, rejection blocks execution
- [x] **Tranche Tracker**: Config flag `enable_tranche_tracking`, register_trade() called post-execution, state persisted
- [x] **Walk-Forward Validator**: Config flag `enable_walk_forward_monitor`, results checked post-scan, logged as warning

**Verification Command:**
```bash
grep -rn "isotonic_calibrat\|pattern_gate\|setup_quality\|tranche_track\|walk_forward" \
  src/scanner/engine.py src/scanner/agents/_team.py src/scanner/execution.py | \
  grep -v "^.*:.*#" | wc -l
# Should be >30 lines of actual usage (not just comments)
```

---

## 2. CONFIDENCE SCORE DECOMPOSITION

### Problem Statement
Journal analysis shows:
- Base confidence doesn't predict wins well (38% win rate stable across 0.5-0.8 confidence)
- High confidence (0.71-0.80) actually underperforms (16.1% win rate vs 38.4% baseline)
- Suggests a single confidence score conflates multiple independent signals

### Research: Decomposition Approaches

#### A. **Directional Confidence** (Does the model predict direction correctly?)
Measures: How aligned are the 12 agents on direction?
- **Source**: Agent verdict agreement on direction
- **Calc**: STD(agent_direction_scores) normalized to [0, 1]
- **Signal**: LOW std = high directional confidence
- **Expected Impact**: Should correlate with WIN/LOSS outcome

#### B. **Timing Confidence** (Is this the right time to enter?)
Measures: Regime alignment + macro factors + recent pair performance
- **Source**: 
  - Regime detector (is vol/trend regime favorable?)
  - Seasonality tracker (hour-of-day signal strength)
  - Pair recent fitness (last N trades on this pair)
  - Macro stress (is market calm enough?)
- **Calc**: Weighted average of regime_strength, hour_bonus, pair_fitness, macro_calm
- **Signal**: HIGH = market conditions are stable for entry
- **Expected Impact**: Should reduce duration variance + increase exit timing quality

#### C. **Magnitude Confidence** (Will the move be large enough to reach TP?)
Measures: Volatility regime + trend strength + ATR forecast
- **Source**:
  - Current ATR vs historical ATR (regime extremeness)
  - Trend strength score from ensemble
  - Price oscillation (recent range expansion)
- **Calc**: (current_ATR / mean_ATR) * trend_strength_normalized
- **Signal**: HIGH = big move likely, TP achievable
- **Expected Impact**: Should predict R:R attainment rate

### Implementation: Python Decomposer

**File: src/scanner/confidence_decomposition.py (NEW)**

```python
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class ConfidenceComponents:
    """Decomposed confidence with independent validation signals."""
    directional_confidence: float  # 0-1: agent agreement on direction
    timing_confidence: float       # 0-1: market conditions favorable
    magnitude_confidence: float    # 0-1: move likely to reach TP
    composite_confidence: float    # 0-1: weighted average (default)
    
    # Components that feed magnitude confidence
    volatility_regime: str         # "LOW", "NORMAL", "HIGH", "EXTREME"
    current_atr_ratio: float       # current_ATR / historical_mean_ATR
    trend_strength: float          # 0-1 from ensemble
    
    # Components that feed timing confidence
    macro_stress_level: float      # 0-1 (0=calm, 1=crisis)
    pair_recent_win_rate: float    # Last 20 trades on this pair
    hour_of_day_strength: float    # 0-1: seasonality signal for this hour
    
    details: Dict[str, Any]        # Raw agent scores, regime data, etc.


class ConfidenceDecomposer:
    """Decomposes single confidence into independent components."""
    
    def __init__(self, agent_team, regime_detector, seasonality, macro_stress):
        self.agent_team = agent_team
        self.regime_detector = regime_detector
        self.seasonality = seasonality
        self.macro_stress = macro_stress
        
        # Historical ATR for magnitude normalization
        self._historical_atr = {}  # pair -> mean_atr_24h
    
    def decompose(
        self,
        pair: str,
        agent_verdicts: List['AgentVerdict'],
        analysis: 'PairAnalysis',
        df_feat: pd.DataFrame,
    ) -> ConfidenceComponents:
        """Decompose a trade setup's confidence into components.
        
        Args:
            pair: Currency pair
            agent_verdicts: Verdicts from 12-agent team
            analysis: PairAnalysis with base signals
            df_feat: Feature dataframe with technical signals
        
        Returns:
            ConfidenceComponents with validated sub-scores
        """
        
        # 1. DIRECTIONAL CONFIDENCE
        # Calculate agent agreement on direction
        direction_scores = [v.score for v in agent_verdicts if "direction" in v.name.lower()]
        if direction_scores:
            dir_std = np.std(direction_scores)
            dir_conf = 1.0 - np.clip(dir_std / 0.25, 0, 1)  # Normalize std to [0,1]
        else:
            dir_conf = np.mean([v.score for v in agent_verdicts]) if agent_verdicts else 0.5
        
        # 2. TIMING CONFIDENCE
        # Combine macro, seasonality, and pair fitness
        macro_level = self.macro_stress.get_stress_level() if self.macro_stress else 0.3
        timing_confidence = (
            (1.0 - macro_level) * 0.4 +  # Calm markets (40% weight)
            (self._get_pair_win_rate(pair) / 0.50) * 0.3 +  # Pair fitness (30% weight)
            (self._get_hour_strength() * 0.3)  # Seasonality (30% weight)
        )
        timing_confidence = np.clip(timing_confidence, 0, 1)
        
        # 3. MAGNITUDE CONFIDENCE
        # Use volatility regime and trend strength
        regime = self.regime_detector.get_regime(pair)
        current_atr = analysis.atr  # From PairAnalysis
        hist_atr = self._get_historical_atr(pair)
        atr_ratio = current_atr / hist_atr if hist_atr > 0 else 1.0
        
        # Magnitude is high when:
        # - ATR is elevated (more room to move)
        # - Trend is strong (directional conviction)
        trend_strength = np.mean([v.score for v in agent_verdicts 
                                 if "trend" in v.name.lower()] or [0.5])
        
        magnitude_confidence = (
            (atr_ratio / 1.5) * 0.5 +  # ATR elevation (50% weight)
            trend_strength * 0.5  # Trend strength (50% weight)
        )
        magnitude_confidence = np.clip(magnitude_confidence, 0, 1)
        
        # 4. COMPOSITE (weighted average)
        composite = (
            dir_conf * 0.4 +  # Directional most important (40%)
            magnitude_confidence * 0.35 +  # Magnitude next (35%)
            timing_confidence * 0.25  # Timing (25%)
        )
        
        return ConfidenceComponents(
            directional_confidence=dir_conf,
            timing_confidence=timing_confidence,
            magnitude_confidence=magnitude_confidence,
            composite_confidence=composite,
            volatility_regime=regime,
            current_atr_ratio=atr_ratio,
            trend_strength=trend_strength,
            macro_stress_level=macro_level,
            pair_recent_win_rate=self._get_pair_win_rate(pair),
            hour_of_day_strength=self._get_hour_strength(),
            details={
                "agent_direction_count": len([v for v in agent_verdicts if v.passed]),
                "regime_name": regime,
                "macro_stress": macro_level,
            }
        )
    
    def _get_pair_win_rate(self, pair: str) -> float:
        """Get last 20 trades win rate for pair."""
        # TODO: Query trade_journal_rl.json for pair
        return 0.4
    
    def _get_historical_atr(self, pair: str) -> float:
        """Get 24h rolling mean ATR for pair."""
        return self._historical_atr.get(pair, 15.0)
    
    def _get_hour_strength(self) -> float:
        """Get seasonality signal for current hour."""
        if self.seasonality:
            return self.seasonality.get_hour_strength()
        return 0.5
```

### Validation Strategy

**Test: Do decomposed scores predict outcomes better than monolithic confidence?**

```python
# In Phase 52 learnings audit:
decomposed_wins = []  # Trades where directional_conf > 0.65
monolithic_wins = []  # Trades where raw confidence > 0.65

for trade in journal:
    if decompose(trade).directional_confidence > 0.65:
        decomposed_wins.append(trade.outcome == "WIN")
    if trade.confidence > 0.65:
        monolithic_wins.append(trade.outcome == "WIN")

decomposed_wr = sum(decomposed_wins) / len(decomposed_wins)
monolithic_wr = sum(monolithic_wins) / len(monolithic_wins)

print(f"Decomposed (dir_conf>0.65): {decomposed_wr:.2%}")
print(f"Monolithic (conf>0.65): {monolithic_wr:.2%}")
# Goal: decomposed_wr > monolithic_wr
```

### Integration Point

Wiring into run_cycle() after setup quality filter:

```python
# In engine.py:run_cycle()
decomposer = ConfidenceDecomposer(
    self._agent_team, 
    self._regime_detector, 
    self._seasonality,
    self._macro_stress,
)
comp_result = decomposer.decompose(pair, agent_result.verdicts, pair_analysis, df_feat)

# Log all three for post-trade analysis
logger.info(f"Phase 52: Decomposed confidence — DIR:{comp_result.directional_confidence:.3f} "
           f"TIM:{comp_result.timing_confidence:.3f} MAG:{comp_result.magnitude_confidence:.3f} "
           f"COMP:{comp_result.composite_confidence:.3f}")

# Optionally use for entry filtering:
if comp_result.directional_confidence < 0.55:
    logger.info("Phase 52: Skipping — low directional confidence")
    return ScanResult(passed=False)
```

---

## 3. ADAPTIVE R:R TARGETING

### Problem & Opportunity
Current state:
- Fixed ATR multipliers: SL = 1.0x ATR, TP = 1.5x ATR → R:R always ~1.5:1
- Win rate 38% with fixed R:R is suboptimal
- Journal shows: 480-1440min duration = 52% win rate, but at what R:R?

Research insight:
- **High-probability trades** (directional_conf > 0.70) can sustain smaller R:R
- **Low-probability trades** (directional_conf < 0.55) need larger R:R to justify risk
- **Volatility regime shifts** reduce TP attainment speed → adjust TP distance dynamically

### Approach: Regime + Confidence Adaptive TP

**File: src/scanner/adaptive_tp.py (NEW)**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AdaptiveTPResult:
    """Result of adaptive TP calculation."""
    tp_pips: float
    sl_pips: float
    r_ratio: float
    reasoning: str


class AdaptiveTPCalculator:
    """Dynamically adjusts TP distance based on regime, confidence, pair."""
    
    # Target R:R ratios by confidence band + regime
    TARGET_RR_MATRIX = {
        "LOW": {      # Low volatility regime
            "high": 1.2,    # conf > 0.70: can take 1.2:1
            "medium": 1.5,  # conf 0.55-0.70: default 1.5:1
            "low": 2.0,     # conf < 0.55: need 2.0:1 to justify
        },
        "NORMAL": {
            "high": 1.3,
            "medium": 1.5,
            "low": 1.8,
        },
        "HIGH": {     # High volatility regime
            "high": 1.4,
            "medium": 1.6,  # TP harder to reach, increase distance
            "low": 2.2,
        },
        "EXTREME": {
            "high": 1.5,
            "medium": 2.0,  # TP very hard, need bigger ratio
            "low": 2.5,
        },
    }
    
    def calculate(
        self,
        pair: str,
        direction: str,
        sl_pips: float,
        regime: str,
        directional_confidence: float,
        pair_recent_win_rate: Optional[float] = None,
        pair_duration_avg_hours: Optional[float] = None,
    ) -> AdaptiveTPResult:
        """Calculate adaptive TP based on confidence + regime.
        
        Args:
            pair: Currency pair
            direction: "LONG" or "SHORT"
            sl_pips: Stop loss distance (from entry)
            regime: Volatility regime ("LOW", "NORMAL", "HIGH", "EXTREME")
            directional_confidence: 0-1 score of direction alignment
            pair_recent_win_rate: Optional pair win rate for boost
            pair_duration_avg_hours: Optional expected duration
        
        Returns:
            AdaptiveTPResult with TP distance and R:R ratio
        """
        
        # 1. Determine confidence band
        if directional_confidence > 0.70:
            conf_band = "high"
        elif directional_confidence > 0.55:
            conf_band = "medium"
        else:
            conf_band = "low"
        
        # 2. Get target R:R from matrix
        target_rr = self.TARGET_RR_MATRIX[regime][conf_band]
        reasoning = f"regime={regime}, dir_conf={directional_confidence:.2f} ({conf_band})"
        
        # 3. Adjust target R:R based on pair win rate
        if pair_recent_win_rate:
            if pair_recent_win_rate > 0.50:  # High-performing pair
                target_rr *= 0.90  # Can be more aggressive (1.5 → 1.35)
                reasoning += f", pair_wr={pair_recent_win_rate:.1%} [bonus]"
            elif pair_recent_win_rate < 0.35:  # Struggling pair
                target_rr *= 1.15  # Need more conservative (1.5 → 1.73)
                reasoning += f", pair_wr={pair_recent_win_rate:.1%} [penalty]"
        
        # 4. Adjust for expected duration
        if pair_duration_avg_hours:
            if pair_duration_avg_hours < 6:
                target_rr *= 0.85  # Scalp trades move quickly, can be closer
            elif pair_duration_avg_hours > 24:
                target_rr *= 1.05  # Longer holds risk more drift, TP further
        
        # 5. Calculate TP from target R:R
        tp_pips = sl_pips * target_rr
        
        # 6. Enforce bounds (prevent extreme positions)
        tp_pips = np.clip(tp_pips, sl_pips * 1.1, sl_pips * 2.5)
        
        return AdaptiveTPResult(
            tp_pips=round(tp_pips, 1),
            sl_pips=sl_pips,
            r_ratio=round(tp_pips / sl_pips, 2),
            reasoning=reasoning,
        )
```

### Integration into ExecutionManager

**File: src/scanner/execution.py → modify execute_trade()**

```python
def execute_trade(
    self,
    pair: str,
    direction: str,
    confidence: float,
    directional_confidence: float = None,  # NEW: decomposed score
    regime: str = "NORMAL",
    pair_recent_win_rate: float = None,
    setup_context: Any = None,
):
    """Execute trade with adaptive TP targeting."""
    
    # ... existing logic to compute SL ...
    sl_pips = self._compute_sl_pips(pair, regime)
    
    # NEW: Use adaptive TP calculator
    if self._adaptive_tp_calculator and directional_confidence:
        atp_result = self._adaptive_tp_calculator.calculate(
            pair=pair,
            direction=direction,
            sl_pips=sl_pips,
            regime=regime,
            directional_confidence=directional_confidence,
            pair_recent_win_rate=pair_recent_win_rate,
        )
        tp_pips = atp_result.tp_pips
        logger.info(f"Adaptive TP: {atp_result.reasoning} → TP={tp_pips:.1f} (R:R={atp_result.r_ratio})")
    else:
        # Fallback to fixed multiplier
        tp_pips = sl_pips * self.config.atr_tp_multiplier
    
    # ... rest of execution ...
```

### Expected Impact
- **Baseline**: 38% win rate, fixed 1.5:1 R:R → expectancy = 0.38*1.5 - 0.62*1 = -0.08
- **Target**: 42% win rate (from pattern gate + decomposed confidence), adaptive 1.3-1.8:1 R:R → expectancy = 0.42*1.5 - 0.58*1 = +0.05

---

## 4. ENTRY TIMING REFINEMENT

### Current: H1 Candle Scanning (Limited Granularity)
Existing system scans at 1H intervals. To improve 8-24h duration trades (52% win rate), need sub-hourly entry timing.

### Three Approaches:

#### A. **Intra-Hour Confirmation** (Minimal latency, no new data source)
Use order flow proxies available from OANDA:
- **Bid/Ask spread tightening** — When spread < 20th percentile, liquidity is high (good entry)
- **Midprice momentum** — 15-min bars show directional bias before 1H candle closes
- **Volume profile** — OANDA order book depth (if accessible via v20 API)

```python
class IntraHourConfirmation:
    """Confirms 1H signal with 15-min microstructure."""
    
    def confirm_entry(self, pair: str, direction: str) -> bool:
        """Wait for microstructure alignment within 1H candle."""
        # Poll OANDA bid/ask and tick volume every 30s for 30min
        # Check: spread < median, direction-aligned tick volume increasing
        # Return True when conditions align (or timeout after 30min)
        pass
```

#### B. **Order Flow Imbalance Signal** (Real-time, OANDA v20 compatible)
Measure buy vs sell tick arrival asymmetry:

```python
class OrderFlowImbalanceSignal:
    """Detects directional order flow imbalance (market microstructure)."""
    
    def get_imbalance_signal(self, pair: str, lookback_min: int = 15) -> float:
        """
        Returns: -1.0 to +1.0
        - Positive = more buy volume (good for LONG)
        - Negative = more sell volume (good for SHORT)
        
        Implementation:
        - Query OANDA /v3/accounts/{id}/trades
        - Count open BUY vs SELL orders at current level
        - Weight by proximity to mid (closer = more influential)
        """
        # Can approximate with bid/ask size imbalance from spreads
        # Bid size increase = selling interest (SHORT signal)
        # Ask size increase = buying interest (LONG signal)
        pass
```

#### C. **Tick Volume Analysis** (30-min validation window)
Before entry, wait for tick volume confirmation in direction of setup:

```python
class TickVolumeConfirmation:
    """Waits for volume spike in predicted direction."""
    
    def is_volume_confirming(self, pair: str, direction: str, max_wait_min: int = 30) -> bool:
        """
        True if tick volume > 75th percentile in next 30 mins
        for the predicted direction (LONG = buys > sells, SHORT = sells > buys).
        """
        # Poll every 1-2 minutes
        # Track cumulative imbalance
        # Return after signal or timeout
        pass
```

### Recommended Implementation: Hybrid Approach

**File: src/scanner/entry_timing.py (NEW)**

```python
from datetime import datetime, timedelta, timezone
import time

class AdaptiveEntryTiming:
    """Coordinates 1H signal with sub-hourly microstructure confirmation."""
    
    def __init__(self, oanda_client, max_wait_minutes: int = 30):
        self.oanda = oanda_client
        self.max_wait = max_wait_minutes
        self._signal_log = {}  # pair -> timestamp
    
    def wait_for_confirmation(
        self,
        pair: str,
        direction: str,
        signal_time: datetime,
    ) -> bool:
        """Block until entry microstructure aligns (or timeout).
        
        Returns:
            True if confirmation received, False if timeout
        """
        timeout = signal_time + timedelta(minutes=self.max_wait)
        poll_interval = 60  # Check every 1 minute
        
        confirmed_at = None
        while datetime.now(timezone.utc) < timeout:
            # Check 1: Spread tightness
            spread_ok = self._check_spread(pair)
            
            # Check 2: Tick volume imbalance
            volume_ok = self._check_volume_imbalance(pair, direction)
            
            # Both must align
            if spread_ok and volume_ok:
                confirmed_at = datetime.now(timezone.utc)
                logger.info(f"Phase 52 entry timing: {pair} {direction} confirmed "
                           f"at {confirmed_at} (waited {(confirmed_at - signal_time).total_seconds() / 60:.1f}min)")
                return True
            
            time.sleep(poll_interval)
        
        logger.warning(f"Phase 52 entry timing: {pair} {direction} timeout after {self.max_wait}min")
        return False  # Timeout = skip entry
    
    def _check_spread(self, pair: str) -> bool:
        """True if spread is in tight quartile."""
        # Query OANDA pricing
        tick = self.oanda.get_price(pair)
        # Compare bid/ask spread to rolling 100-candle median
        # TODO: Implement
        return True
    
    def _check_volume_imbalance(self, pair: str, direction: str) -> bool:
        """True if recent ticks favor the predicted direction."""
        # Use OANDA trade list to infer market sentiment
        # Or use bid/ask size from pricing endpoint
        # TODO: Implement
        return True
```

### Integration: When to Use

Only apply on high-confidence setups:
```python
# In engine.py:run_cycle()
if comp_result.directional_confidence > 0.70:
    # Premium setups warrant a 30-min wait for perfect entry
    if not entry_timer.wait_for_confirmation(pair, direction, datetime.now(timezone.utc)):
        logger.info("Entry timing confirmation timeout — skipping")
        return ScanResult(passed=False)
```

### Expected Duration Impact
- Without timing: Uniform 1H to 720H distribution → higher variance
- With timing: Can skew entries toward high-conviction moments → tighter distribution around 12-24H sweet spot → potentially higher 52% win rate captures

---

## 5. DRAWDOWN-ADAPTIVE BEHAVIOR

### Current State
Position sizing already reduces on drawdown. Need behavioral shifts:

### Three Behavioral Adaptations:

#### A. **Pair Rotation** (Not just position size reduction)
During drawdown, favor high-edge pairs:

```python
class DrawdownAwarePairSelection:
    """Dynamically tightens pair universe during drawdown."""
    
    def get_tradeable_pairs(self, drawdown_pct: float) -> List[str]:
        """Filter pair list based on drawdown depth."""
        all_pairs = ["EUR_USD", "GBP_USD", "EUR_JPY", "GBP_AUD", ...]
        
        if drawdown_pct < 0.05:  # <5% drawdown
            return all_pairs  # Trade all 12
        elif drawdown_pct < 0.10:  # 5-10% drawdown
            # Keep only top 5 performers (52% wr+ or P&L positive)
            return [p for p in all_pairs if self._get_pair_wr(p) > 0.50]
        elif drawdown_pct < 0.15:  # 10-15% drawdown
            # Concentrate on the top 2-3 pairs only
            return sorted(all_pairs, key=self._get_pair_wr, reverse=True)[:3]
        else:  # >15% drawdown
            # Halt new trades, close losers, wait for recovery
            return []
```

#### B. **Confidence Tightening** (Higher entry bar)
Raise minimum confidence thresholds as drawdown deepens:

```python
class DrawdownAdaptiveThresholds:
    """Progressively tighter gates during drawdown."""
    
    def get_min_confidence(self, drawdown_pct: float) -> float:
        """Raise confidence threshold as drawdown increases."""
        base = 0.50  # Normal minimum
        
        # Each 5% of drawdown adds 5% to minimum
        penalty = (drawdown_pct / 5.0) * 0.05
        return np.clip(base + penalty, 0.50, 0.75)
```

#### C. **Session-Based Halt** (Geographic time restrictions)
Avoid volatile sessions during drawdown:

```python
class DrawdownAwareSessionFilter:
    """Restricts to calm sessions during drawdown."""
    
    def is_session_tradeable(self, drawdown_pct: float) -> bool:
        """True if current session is permitted by drawdown rules."""
        current_hour_utc = datetime.now(timezone.utc).hour
        
        if drawdown_pct < 0.10:
            return True  # All sessions OK
        else:
            # Only trade Tokyo + London overlap (high liquidity)
            return 7 <= current_hour_utc <= 12
```

### Integration: Orchestrated Behavior

**File: src/scanner/drawdown_recovery.py (NEW)**

```python
from dataclasses import dataclass

@dataclass
class DrawdownBehavior:
    """Adaptive behavior during drawdown."""
    pair_universe: List[str]
    min_confidence: float
    trade_enabled: bool
    session_restriction: Optional[str]  # "none", "london", "newyork", "tokyo"


class DrawdownRecoveryManager:
    """Orchestrates all behavioral adaptations based on drawdown level."""
    
    def __init__(self):
        self.nav_high = 100000.0  # Track peak NAV
        self._pair_rotator = DrawdownAwarePairSelection()
        self._threshold_adjuster = DrawdownAdaptiveThresholds()
        self._session_filter = DrawdownAwareSessionFilter()
    
    def get_behavior(self, current_nav: float) -> DrawdownBehavior:
        """Get all behavioral parameters for current drawdown state."""
        
        if current_nav > self.nav_high:
            self.nav_high = current_nav  # Update high water mark
        
        drawdown_pct = (self.nav_high - current_nav) / self.nav_high
        
        # 1. Pair rotation
        pairs = self._pair_rotator.get_tradeable_pairs(drawdown_pct)
        
        # 2. Confidence threshold
        min_conf = self._threshold_adjuster.get_min_confidence(drawdown_pct)
        
        # 3. Trading enabled?
        trade_enabled = drawdown_pct < 0.15  # Halt at 15%
        
        # 4. Session restriction
        session = "none"
        if drawdown_pct > 0.10:
            session = "london"  # Calm session only
        
        return DrawdownBehavior(
            pair_universe=pairs,
            min_confidence=min_conf,
            trade_enabled=trade_enabled,
            session_restriction=session,
        )
```

### Integration: In run_cycle()

```python
# In engine.py:run_cycle()
current_nav = self._get_live_nav()
drawdown_behavior = self._drawdown_manager.get_behavior(current_nav)

# Check: is pair tradeable?
if pair not in drawdown_behavior.pair_universe:
    logger.info(f"Phase 52 drawdown: {pair} restricted (only {drawdown_behavior.pair_universe} allowed)")
    return ScanResult(passed=False)

# Check: is trading enabled?
if not drawdown_behavior.trade_enabled:
    logger.info("Phase 52 drawdown: trading halted (drawdown >15%)")
    return ScanResult(passed=False)

# Check: raise confidence bar?
if confidence < drawdown_behavior.min_confidence:
    logger.info(f"Phase 52 drawdown: {confidence:.2f} < {drawdown_behavior.min_confidence:.2f}")
    return ScanResult(passed=False)

# Check: session filter?
if not self._session_filter.is_session_tradeable(drawdown_pct):
    logger.info("Phase 52 drawdown: restricted session")
    return ScanResult(passed=False)
```

### Expected Impact
- **Normal**: 38% win rate, all pairs, baseline position size → baseline expectancy
- **5-10% Drawdown**: 38% win rate, top 5 pairs, 80% position size, min_conf +0.05 → 1-2% improvement
- **10-15% Drawdown**: 45% win rate (selection effect), top 3 pairs only, 50% position size → stop drawdown expansion
- **>15% Drawdown**: Halt new trades, preserve capital, exit on recovery signal

---

## SUMMARY TABLE: Implementation Priority & Effort

| Topic | Files | Config Flag | Wiring Effort | Validation | Win Rate Delta |
|-------|-------|-------------|---------------|------------|-----------------|
| **1. Pipeline Wiring** | engine.py, execution.py, agents/_team.py | enable_phase51_wiring | 4-6 hours | Smoke test | N/A (enabler) |
| **2. Decomposition** | confidence_decomposition.py | enable_decomposed_confidence | 3-4 hours | Backtest journal | +1-2% (if directional used) |
| **3. Adaptive R:R** | adaptive_tp.py, execution.py | enable_adaptive_tp | 2-3 hours | Analyze R:R dist | +0-1% (quality, not quantity) |
| **4. Entry Timing** | entry_timing.py, engine.py | enable_entry_confirmation | 4-5 hours | Duration histogram | +2-3% (if duration lock works) |
| **5. Drawdown Behavior** | drawdown_recovery.py, engine.py | enable_drawdown_adaptation | 3-4 hours | Live equity curve | +1-2% (capital preservation) |

### Recommended Sequencing for Phase 52
1. **Week 1**: Live Pipeline Wiring (prerequisite for all else)
2. **Week 2**: Confidence Decomposition (enables timing + adaptive R:R)
3. **Week 3**: Adaptive R:R Targeting (measurable improvement)
4. **Week 4**: Entry Timing Refinement (synergy with duration patterns)
5. **Week 5**: Drawdown Recovery Behavior (risk management)

Each stage should include:
- Code review (per .claude/rules/improvement.md)
- Unit tests (5+ cases per component)
- Smoke test in watch mode (verify log output)
- 10-trade validation (live, small position)

