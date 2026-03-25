# ML Engine Exit Strategy Analysis
**Date:** 2026-03-24
**Status:** Comprehensive Review of Current Implementation & Weakness Analysis

---

## EXECUTIVE SUMMARY

The ML Engine has a **sophisticated, modular exit strategy system** with five concurrent evaluation strategies (Chandelier, Time, Confidence Decay, Volatility-Adjusted Trail, Partial Profits). However, the system has **significant integration gaps**:

1. **Adaptive exit manager is instantiated but underutilized** — only called in `evaluate_exits()` which is rarely invoked
2. **Trailing stops rely only on simple ATR-based logic** — no regime adaptation, no confidence consideration
3. **No agent-driven exit guidance** — agents inform entry but don't guide exits
4. **Partial profit taking is configured but never executed** — infrastructure exists but no orchestration
5. **No feedback loop to RL system** — exits don't update agent weights based on exit quality
6. **Missing: adaptive regimes for exit parameters** — use same multipliers regardless of market conditions

---

## CURRENT IMPLEMENTATION STRENGTHS

### 1. **Adaptive Exit Manager (adaptive_exits.py)**
**Status:** Well-architected, stateless design
**Lines:** 1-738

Implements **five exit strategies with priority-based merging:**

| Strategy | Mechanism | Config | Status |
|----------|-----------|--------|--------|
| **Confidence Decay** | Exit if confidence < 0.35 after exponential decay | `confidence_decay_rates` (0.92-0.99/bar) | Defined, not invoked |
| **Time Exit** | Backstop: exit if bars_held >= regime max | `time_exit_bars` (6-40 bars by regime) | Defined, not invoked |
| **Chandelier** | ATR-based trail from peak (22-bar max high) | `chandelier_atr_multiplier=3.0` | Defined, not invoked |
| **Vol-Adjusted Trail** | Regime-aware trailing stop (1.5-4.0x ATR) | `vol_trail_multipliers` | Defined, not invoked |
| **Partial Profit** | Scale-out at R-multiples (3 tranches configured) | `partial_tranches` with ratio/r_target/type | Defined, not invoked |

**Key Classes:**
- `ExitStrategyConfig` (lines 28-115) — validation with dataclass defaults
- `TradeContext` (lines 118-206) — immutable state passed to evaluators
- `ExitAction` (lines 210-228) — decision representation (action/reason/strategy/params)
- `AdaptiveExitManager` (lines 236-695) — orchestrator with 5 evaluation methods + priority merge

**Priority Merge (lines 659-695):**
```
1. EXIT_FULL + critical → 2. EXIT_FULL + high → 3. TAKE_PARTIAL →
4. EXIT_FULL + normal → 5. TIGHTEN_SL → 6. HOLD
```

---

### 2. **Trailing Stop Implementation (execution.py lines 2483-2594)**
**Status:** Basic ATR-only, no regime/confidence adaptation

**Current Logic:**
```python
def update_trailing_stops(self, atr_by_pair: Optional[Dict[str, float]] = None) -> List[str]:
    # Line 2507: trail_mult = 1.5 (hardcoded config value)
    # Line 2533: trail_distance = atr * trail_mult
    # Lines 2542-2551: For LONG: new_sl = mid - trail_distance (only tighten, never widen)
    # Lines 2554-2593: Batch modify via OANDA API with retry logic
```

**Weaknesses:**
- No consideration of entry confidence
- No regime-dependent scaling
- No position size/risk adjustment
- Runs every scan cycle (overhead) with no adaptive frequency
- No partial take profit execution

---

### 3. **Exit Evaluation Hook (execution.py lines 1994-2120)**
**Status:** Exists but rarely called

```python
def evaluate_exits(
    self,
    trade_statuses: List[Dict[str, Any]],
    regime_name: str = "NORMAL",
    current_confidence: float = 0.5,
    atr_values: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
```

**What It Does:**
- Takes trade statuses from `monitor_open_trades()`
- Builds synthetic `TradeContext` with estimated bar counts from `time_in_minutes // 15`
- Calls `self._adaptive_exit_manager.evaluate_exit(ctx)` for each trade
- Returns list of exit actions (trade_id, action, reason, details)

**Problem:** Only called in `continuous.py` around line 800+ (position management phase), and only if:
- `enable_position_management=True` (not everywhere)
- `_scan_count % position_management_interval == 0` (defaults to every 3 cycles)
- There are open trades

**Impact:** Exit signals generated but **never acted on** (no close_position call).

---

### 4. **Learning from Exit Patterns (learning_engine.py lines 689-822)**
**Status:** Extraction framework exists

**Patterns Extracted:**
- Pattern 3 (lines 779-793): Trailing stop outperforms fixed TP
- Pattern 4 (lines 795-806): High timeout rate → entry timing issue
- Pattern 5 (lines 808-820): Breakeven stop dominance → tighten trailing params

**Captures:**
```json
{
  "exit_reason": ["tp_hit", "sl_hit", "trailing_stop", "breakeven_stop", "timeout"],
  "exit_accountability": "agent_voted_yes → won_via_trailing_stop"
}
```

**Problem:** Patterns extracted but **not promoted to active rules** (no feedback to `_adaptive_exit_manager` config).

---

## CRITICAL GAPS & WEAKNESSES

### Gap 1: **Adaptive Exit Manager is Instantiated but Disconnected**

**Location:** `execution.py` line 190
```python
self._adaptive_exit_manager = create_default_exit_manager()
```

**Problem:**
- Instantiated once at init
- `evaluate_exits()` called only occasionally
- Exit actions never translated to `close_position()` or `modify_sl()` calls
- No real-time orchestration

**Impact:** Five well-designed strategies do nothing.

---

### Gap 2: **Trailing Stops Ignore Confidence & Regime Context**

**Location:** `execution.py` lines 2507-2533
```python
trail_mult = float(getattr(self.config, "trailing_atr_multiplier", 1.5))
trail_distance = atr * trail_mult  # Same multiplier for all pairs, all regimes
```

**Missing Features:**
- No scaling based on current_confidence (should widen trail when confident, tighten when uncertain)
- No regime adaptation (HIGH vol should use 3.0x, LOW vol should use 1.5x)
- No position size weighting (large positions should have tighter trail)
- No partial profit execution at all

**Improvement Points:**
```python
# Should be:
regime_mult = vol_trail_multipliers.get(regime_name, 2.5)
confidence_scale = 0.8 + (current_confidence * 0.4)  # 0.8-1.2x
trail_mult = regime_mult * confidence_scale
```

---

### Gap 3: **Partial Profit Taking is Configured but Never Executed**

**Location:** `adaptive_exits.py` lines 583-657
```python
def _evaluate_partial_profit(self, ctx: TradeContext) -> ExitAction:
    # Returns TAKE_PARTIAL with partial_close_ratio (e.g., 0.5)
    # But no one calls this or acts on the result
```

**Config Defined (config.py):**
```python
partial_tranches = [
    {"ratio": 0.50, "r_target": 1.0, "type": "fixed"},
    {"ratio": 0.25, "r_target": 1.5, "type": "trailing"},
    {"ratio": 0.25, "r_target": 2.0, "type": "fixed_or_time"},
]
```

**Missing Implementation:**
- No `close_partial_position()` method in ExecutionManager
- No orchestration to close fraction of trade at R-target
- No tranche state tracking across cycles

---

### Gap 4: **No Agent-Driven Exit Guidance**

**Location:** All agent evaluation (agents/_team.py) is entry-only

**Missing:**
- Agents don't evaluate exit readiness or confidence in trade thesis
- No "WIN_PROB" or "UNCERTAINTY_ELEVATED" post-entry signals
- Exit accountability learning (lines 193-247 in learning_engine.py) analyzes but doesn't feed back

**Would Enable:**
- Risk sentinel could evaluate trade thesis degradation
- Momentum agent could signal trend reversal
- Uncertainty agent could exit on overconfidence collapse

---

### Gap 5: **No Feedback Loop to RL Agent Weights**

**Location:** `execution.py` line 2842-2853 (minimal RL sync)
```python
# Only SL/TP hit, trailing_stop, and breakeven_stop are differentiated
# No agent credit/blame for exit quality
```

**Missing:**
- Exit reason doesn't update agent weights (unlike entry, which has RL sync at line 2852)
- Agents that voted YES on a trade exited via trailing stop should get +weight boost
- Agents that voted NO on a trade but it hit TP should get -weight penalty

---

### Gap 6: **Confidence Decay & Time Exit Thresholds Not Calibrated**

**Location:** `adaptive_exits.py` lines 66-76, 56-63
```python
confidence_exit_threshold: float = 0.35  # Too loose? Never triggers?
time_exit_bars = {"LOW": 40, "NORMAL": 25, "HIGH": 15, "EXTREME": 8}
```

**Questions:**
- With 95-99% decay rates and 15-40 bar holding times, does confidence ever hit 0.35?
- Are time exits realistic for 4H/1D timeframes? (25 bars ≈ 100 hours)
- What entry confidence distribution? (needed to tune decay rate)

**No Calibration Data:**
- No journal records of confidence_decay exits
- No histogram of trade duration by pair
- No learning trigger when trades stay open "too long"

---

### Gap 7: **Synthetic TradeContext Lacks Realistic Price Data**

**Location:** `execution.py` lines 2070-2078
```python
prices_close = np.full(price_array_size, current_price, dtype=np.float64)
prices_high = np.full(price_array_size, max(highest_price, current_price), ...)
# Synthetic data — Chandelier exit will always use current price as max high
```

**Problem:**
- Chandelier exit (22-bar lookback) evaluates on fabricated data
- Peak price tracking is unrealistic (assumes all past bars were current price)
- Time exit accuracy depends on `time_in_minutes // 15` which loses granularity

**Ideal:** Populate from actual OHLC history (not available in monitor_open_trades).

---

### Gap 8: **No Adaptive Exit Parameter Tuning**

**Location:** `config.py` (exit parameters are static)

**Missing Dynamic Adjustment:**
- Vol-trail multipliers should scale with market regime volatility measured over last N bars
- Confidence decay rate should adapt to how fast/slow market moves relative to SL
- Time exit bars should scale based on current pair's mean reversion speed
- Partial profit R-targets should adjust based on recent pair profitability

**Learning Capture:** `learning_engine.py` extracts patterns but doesn't update `ExitStrategyConfig`.

---

## SPECIFIC CODE LOCATIONS FOR IMPROVEMENT

### **File 1: `/sessions/clever-peaceful-knuth/mnt/ml_engine/src/scanner/adaptive_exits.py`**

| Line(s) | Function | Issue | Priority |
|---------|----------|-------|----------|
| 28-115 | `ExitStrategyConfig` | Static config; needs version/timestamp and update hooks | HIGH |
| 263-317 | `evaluate_exit()` | Good entry point; needs to propagate from continuous.py | HIGH |
| 319-368 | `_evaluate_confidence_decay()` | Logic sound; but threshold/decay rates need calibration | MEDIUM |
| 370-414 | `_evaluate_time_exit()` | Missing: time-based TP (exit at X hours if no profit) | MEDIUM |
| 416-500 | `_evaluate_chandelier()` | Missing: synthetic data issue (see Gap 7) | MEDIUM |
| 502-581 | `_evaluate_vol_adjusted_trail()` | Missing: confidence scaling, position size weighting | HIGH |
| 583-657 | `_evaluate_partial_profit()` | Missing: execution path (close_partial_position call) | HIGH |
| 659-695 | `_priority_merge()` | Good; but no logging of conflicts or override frequency | MEDIUM |

---

### **File 2: `/sessions/clever-peaceful-knuth/mnt/ml_engine/src/scanner/execution.py`**

| Line(s) | Function | Issue | Priority |
|---------|----------|-------|----------|
| 35-101 | `ExecutionConfig` | Missing: exit-related params (adaptive_trail_confidence_scale, etc.) | HIGH |
| 1994-2120 | `evaluate_exits()` | Exists but never acted upon; no close_position call | HIGH |
| 2483-2594 | `update_trailing_stops()` | No regime/confidence scaling; hardcoded multiplier | HIGH |
| 2627-2670 | `_determine_exit_reason()` | Good; captures exit reason, but doesn't feed back to agents | MEDIUM |
| New | `close_partial_position()` | **MISSING** — needed for TAKE_PARTIAL actions | CRITICAL |
| New | `apply_exit_action()` | **MISSING** — orchestrator to act on ExitAction | CRITICAL |

---

### **File 3: `/sessions/clever-peaceful-knuth/mnt/ml_engine/src/scanner/automation/continuous.py`**

| Line(s) | Issue | Priority |
|---------|-------|----------|
| ~800 | Position management only runs every 3 cycles (slow) | MEDIUM |
| 792 | Trailing stop updates run every cycle but ignore confidence | HIGH |
| Missing | No orchestration of evaluate_exits() → close_position() | CRITICAL |
| Missing | No check of ExitAction.action == "TAKE_PARTIAL" → close_partial | CRITICAL |

---

### **File 4: `/sessions/clever-peaceful-knuth/mnt/ml_engine/src/scanner/automation/learning_engine.py`**

| Line(s) | Issue | Priority |
|---------|-------|----------|
| 689-822 | `extract_exit_reason_patterns()` | Extracts but doesn't update ExitStrategyConfig | MEDIUM |
| 193-247 | Exit accountability learning | Good; but not used to adjust agent weights (only observed) | MEDIUM |
| Missing | No promotion rule for "trailing_stop consistently beats tp_hit" | MEDIUM |

---

### **File 5: `/sessions/clever-peaceful-knuth/mnt/ml_engine/src/scanner/agents/_team.py`**

| Issue | Priority |
|-------|----------|
| Agents don't generate post-entry exit signals | HIGH |
| No "exit_readiness" or "trade_thesis_confidence_decay" agent | MEDIUM |
| No feedback from exit outcomes to agent weights | HIGH |

---

## MISSING FEATURES (NOT YET IMPLEMENTED)

### 1. **Confidence-Scaled Trailing Stop**
**Where:** `execution.py` update_trailing_stops() (line 2507-2533)

Should compute:
```python
regime_mult = vol_trail_multipliers[regime_name]  # 1.5-4.0
confidence_scale = 0.7 + (current_confidence * 0.6)  # 0.7-1.3
final_mult = regime_mult * confidence_scale
trail_distance = atr * final_mult
```

---

### 2. **Partial Position Closing (Scale-Out)**
**Where:** New method needed in ExecutionManager

```python
def close_partial_position(
    self,
    trade_id: str,
    close_ratio: float,  # 0.5 = close 50%
    reason: str
) -> Dict[str, Any]:
    """Close fraction of position, keeping remainder open with original SL/TP."""
    # 1. Calculate new lot size = current_lots * (1 - close_ratio)
    # 2. Send OANDA market order for remaining lots
    # 3. Log partial close with reason
    # 4. Return remaining trade_id and new position details
```

---

### 3. **Exit Action Orchestrator**
**Where:** New method in continuous.py or ExecutionManager

```python
def apply_exit_action(
    self,
    exit_action: ExitAction,
    trade_id: str,
    trade_status: Dict[str, Any],
) -> bool:
    """Execute exit decision from adaptive exit manager."""
    if exit_action.action == "EXIT_FULL":
        return self.close_position(trade_id, reason=exit_action.reason)
    elif exit_action.action == "TAKE_PARTIAL":
        return self.close_partial_position(
            trade_id,
            close_ratio=exit_action.partial_close_ratio,
            reason=exit_action.reason
        )
    elif exit_action.action == "TIGHTEN_SL":
        return self.modify_stop_loss(
            trade_id,
            new_sl_price=exit_action.new_sl_price,
            reason=exit_action.reason
        )
    # HOLD: do nothing
    return True
```

---

### 4. **Agent Exit Readiness Signals**
**Where:** New agent in agents/_team.py (or new phase)

```python
class ExitReadinessAgent(ScannerAgent):
    """Evaluates if entry thesis is still valid post-entry."""

    def evaluate(self, trade_context) -> Verdict:
        # Check:
        # 1. Confidence degradation > 25% → vote for exit
        # 2. Regime shift contradicts entry signal → vote for exit
        # 3. P/L exceeded target but still running → vote for partial take
        # 4. SL-to-TP ratio has inverted (SL widened) → vote for caution
```

---

### 5. **Adaptive Exit Parameter Tuning**
**Where:** New automation in learning_engine.py + config.py

```python
def tune_exit_parameters(
    self,
    journal_data: List[Dict]
) -> ExitStrategyConfig:
    """Adjust exit thresholds based on historical trade outcomes."""

    # If trailing_stop avg pnl > tp_hit avg pnl by 50%:
    #   → increase vol_trail_multipliers by 0.2
    # If time_exit happens > 20% of trades:
    #   → decrease time_exit_bars
    # If confidence_decay_exit rate is 0:
    #   → increase confidence_decay_threshold (currently never triggers)
```

---

## INTERACTION MAP: EXIT SYSTEMS

```
┌─────────────────────────────────────────────────────────────┐
│                  continuous.py (main loop)                   │
├─────────────────────────────────────────────────────────────┤
│  1. monitor_open_trades() → trade_statuses                   │
│  2. update_trailing_stops(atr_by_pair)      ← Line 792       │
│     └─ ExecutionManager.update_trailing_stops()              │
│        └─ Hardcoded ATR*1.5, no regime/confidence            │
│  3. evaluate_exits(trade_statuses, regime, confidence)       │
│     └─ ExecutionManager.evaluate_exits()     ← Line 1994     │
│        └─ AdaptiveExitManager.evaluate_exit(TradeContext)    │
│           ├─ _evaluate_confidence_decay()    [Config-driven] │
│           ├─ _evaluate_time_exit()           [Config-driven] │
│           ├─ _evaluate_chandelier()          [Synthetic data] │
│           ├─ _evaluate_vol_adjusted_trail()  [No confidence]  │
│           ├─ _evaluate_partial_profit()      [Never executed] │
│           └─ _priority_merge(actions)        [Returns choice] │
│        └─ Returns: List[Dict] exit_actions   [NEVER ACTED ON] │
│  4. learning_engine.extract_exit_reason_patterns()           │
│     └─ Extracts: trailing_stop_outperforms_tp               │
│        └─ [Never fed back to ExitStrategyConfig]             │
│                                                              │
│  ❌ MISSING LINK:                                            │
│     apply_exit_action(exit_action) → close_position()        │
│  ❌ MISSING: close_partial_position() for TAKE_PARTIAL       │
│  ❌ MISSING: Agent exit signals / trade_thesis_confidence    │
│  ❌ MISSING: RL feedback loop for exit quality to agents     │
└─────────────────────────────────────────────────────────────┘
```

---

## RECOMMENDED IMPLEMENTATION ROADMAP

### **Phase 1: Wiring (1-2 days)** — HIGH PRIORITY
1. ✅ Add `apply_exit_action()` orchestrator
2. ✅ Integrate `evaluate_exits()` output → `apply_exit_action()` calls
3. ✅ Add `close_partial_position()` method
4. ✅ Wire trailing stop to use regime & confidence scaling

### **Phase 2: Enrichment (2-3 days)** — HIGH PRIORITY
1. ✅ Implement `ExitReadinessAgent` in agent team
2. ✅ Add exit-focused verdict aggregation in scan cycle
3. ✅ Update RL sync to credit agents for exit quality
4. ✅ Add time-based TP (exit after X hours if no profit)

### **Phase 3: Adaptation (2-3 days)** — MEDIUM PRIORITY
1. ✅ Calibrate confidence decay rates from historical data
2. ✅ Implement adaptive exit parameter tuning
3. ✅ Feed learning patterns back to ExitStrategyConfig
4. ✅ Use real OHLC data for Chandelier exit (not synthetic)

### **Phase 4: Validation (1 day)** — MEDIUM PRIORITY
1. ✅ Unit tests for `apply_exit_action()` paths
2. ✅ Integration tests: evaluate_exits + apply = actual closes
3. ✅ Backtesting: verify exit quality vs baseline
4. ✅ Edge case: can't close (pending order, market closed, etc.)

---

## SUMMARY TABLE: WHAT'S READY vs. MISSING

| Component | Built | Wired | Tested | Notes |
|-----------|-------|-------|--------|-------|
| Confidence Decay Exit | ✅ | ❌ | ❓ | Threshold may never trigger |
| Time Exit Backstop | ✅ | ❌ | ❓ | Good; not called |
| Chandelier Exit | ✅ | ❌ | ❓ | Synthetic data issue |
| Vol-Adjusted Trail | ✅ | ❌ | ❓ | Missing confidence/regime |
| Partial Profit Taking | ✅ | ❌ | ❌ | No execute method |
| Trailing Stops (ATR) | ✅ | ✅ | ✓ | Basic; no adaptation |
| Exit Learning | ✅ | ❌ | ✓ | Extracts; doesn't feed back |
| Exit Orchestration | ❌ | ❌ | ❌ | **CRITICAL GAP** |
| Agent Exit Signals | ❌ | ❌ | ❌ | **NEW AGENT NEEDED** |
| RL Exit Feedback | ❌ | ❌ | ❌ | **NOT IMPLEMENTED** |
| Adaptive Tuning | ❌ | ❌ | ❌ | **FUTURE** |

---

## QUICK WINS (1-2 hour implementations)

1. **Enable evaluate_exits → apply_exit_action loop** in continuous.py
   Impact: Unlock 5 exit strategies immediately

2. **Add regime/confidence scaling to trailing stop multiplier**
   ```python
   trail_mult *= (0.7 + (current_confidence * 0.6))  # Already in rules
   ```
   Impact: Confident trades run longer, uncertain trades stop tight

3. **Calibrate confidence_decay_threshold from journal data**
   Impact: Time exit and confidence decay start triggering appropriately

4. **Add exit action logging/stats to journal**
   Impact: Visibility into which strategies are winning

5. **Wire learning patterns → ExitStrategyConfig updates**
   Impact: Automatic adaptation to market conditions

---

## CONCLUSION

The ML Engine has a **world-class exit strategy architecture** that is 80% implemented but only 20% operationalized. The adaptive exit manager is ready to use; it just needs:

1. **Integration:** Wire exit actions to actual position closures
2. **Adaptation:** Feed confidence and regime into trailing logic
3. **Feedback:** Close the RL loop for exit quality
4. **Completion:** Implement missing orchestration methods

The three critical missing pieces are:
- `apply_exit_action()` — translate strategy decisions to OANDA orders
- `close_partial_position()` — execute scale-out trades
- Exit signals from agent team — thesis validation post-entry

With these wired, Buddy will move from "enter on signal, exit on SL/TP" to "continuous adaptive exit monitoring with multi-strategy consensus."
