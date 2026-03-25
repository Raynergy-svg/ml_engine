# Phase 52 Research Summary — 5 Implementable Topics

**Status**: Research Complete | Win Rate Target: 38% → 45% | Timeline: 5 weeks

---

## 1. LIVE PIPELINE WIRING FOR PHASE 51 MODULES

**Problem**: Phase 51 modules (isotonic calibration, pattern gate, setup quality, tranche tracker, walk-forward) built but may have dead-code wiring.

**Solution**: Enforce correct gate chain order in `src/scanner/engine.py::run_cycle()`:

```
SCAN → GATE → AGENT TEAM → ISOTONIC CAL → PATTERN GATE → SETUP QUALITY → EXEC → TRANCHE TRACKER → WALK-FORWARD
```

**Key Integration Points**:
- Line ~450 in engine.py: Add isotonic calibration after agent_result
- Line ~460: Add pattern gate evaluation and confidence delta
- Line ~470: Add setup quality filter (reject if doesn't pass)
- Line ~500: Register trade with tranche tracker post-execution
- Line ~510: Log walk-forward validation result (warning only, don't block)

**Verification**: Run grep for all 5 module names in production files (not just tests). Each should appear 3+ times in call sites outside its own class.

**Status**: CRITICAL FIRST STEP — Everything else depends on this wiring.

---

## 2. CONFIDENCE SCORE DECOMPOSITION

**Problem**: Single confidence score (38% predictive) conflates direction, timing, and magnitude signals independently.

**Solution**: Replace monolithic confidence with 3-component score in `src/scanner/confidence_decomposition.py`:

- **Directional Confidence** (40% weight): Agent agreement on direction (STD of agent scores)
- **Timing Confidence** (25% weight): Macro stress + pair fitness + hour-of-day seasonality
- **Magnitude Confidence** (35% weight): ATR elevation + trend strength

**Expected Validation**:
```python
# Trades where directional_conf > 0.65 should have higher win rate than 
# trades where monolithic confidence > 0.65
# Target: +3-5% win rate improvement on high-directional-conf subset
```

**Integration**:
- Post-setup-quality-filter: decompose confidence
- Log all 3 components to trade journal
- Use directional_conf for entry timing (if > 0.70, wait for microstructure)

**Files Created**: `src/scanner/confidence_decomposition.py` (150 lines)

---

## 3. ADAPTIVE R:R TARGETING

**Problem**: Fixed 1.5:1 R:R suboptimal. High-confidence trades can accept 1.2:1, low-confidence need 2.0:1.

**Solution**: Build `src/scanner/adaptive_tp.py` with regime + confidence matrix:

```
         LOW Vol    NORMAL Vol    HIGH Vol    EXTREME Vol
High Conf  1.2:1      1.3:1        1.4:1        1.5:1
Med Conf   1.5:1      1.5:1        1.6:1        2.0:1
Low Conf   2.0:1      1.8:1        2.2:1        2.5:1
```

Adjustments:
- Pair recent win rate > 50%: reduce target (1.5 → 1.35, more aggressive)
- Pair recent win rate < 35%: increase target (1.5 → 1.73, more conservative)
- Expected duration < 6h: reduce target (scalp moves fast)
- Expected duration > 24h: increase target (drift risk)

**Integration**:
- In `execution.py::execute_trade()`: Call adaptive calculator instead of fixed multiplier
- Log reasoning (regime, dir_conf, pair_wr) to trade journal
- Enforce min 1.1:1, max 2.5:1 bounds

**Expected Impact**: +0-1% win rate (quality adjustment, not quantity)

**Files Created**: `src/scanner/adaptive_tp.py` (120 lines)

---

## 4. ENTRY TIMING REFINEMENT

**Problem**: H1 candle scanning provides coarse entry timing. 8-24h trades (52% win rate) would benefit from sub-hourly confirmation.

**Solution**: Hybrid microstructure confirmation in `src/scanner/entry_timing.py`:

On high-directional-confidence setups (>0.70), wait up to 30 minutes for:
1. **Spread tightening** — Bid/ask spread < 20th percentile (liquidity high)
2. **Volume imbalance** — Tick volume favors predicted direction
3. Either both align OR timeout after 30 min → enter anyway

```python
if decomposed.directional_confidence > 0.70:
    confirmed = entry_timer.wait_for_confirmation(pair, direction, now())
    if not confirmed:
        logger.info("Timeout — enter anyway")
```

**No new data source required** — Uses OANDA pricing endpoint (bid/ask) + trade list.

**Expected Impact**: +2-3% win rate (tighter entry clustering around high-conviction moments)

**Files Created**: `src/scanner/entry_timing.py` (180 lines)

---

## 5. DRAWDOWN-ADAPTIVE BEHAVIOR

**Problem**: Only position sizing reduces on drawdown. Behavior should adapt (pair rotation, confidence tightening, session restrictions).

**Solution**: `src/scanner/drawdown_recovery.py` with 3 adaptations:

**A. Pair Rotation** (High-edge only):
```
<5% DD:   Trade all 12 pairs
5-10% DD: Top 5 pairs only (>50% win rate)
10-15% DD: Top 3 pairs only
>15% DD:   Halt new trades
```

**B. Confidence Tightening**:
```
Base minimum: 0.50
+5% per 5% drawdown
Cap at 0.75
```

**C. Session Restriction**:
```
<10% DD: All sessions
10-15% DD: London session only (8-12 UTC, calm)
```

**Integration**:
- Early in run_cycle(): Get DrawdownBehavior from manager
- Check: pair in allowed universe?
- Check: confidence > min_threshold?
- Check: current hour in allowed session?
- Check: trading_enabled flag?

**Expected Impact**: +1-2% win rate (prevents whipsaws during recovery, preserves capital)

**Files Created**: `src/scanner/drawdown_recovery.py` (150 lines)

---

## IMPLEMENTATION TIMELINE

| Week | Component | Effort | Validation |
|------|-----------|--------|-----------|
| 1 | Pipeline Wiring (1) | 4-6h | Smoke test |
| 2 | Decomposition (2) | 3-4h | Backtest validation |
| 3 | Adaptive R:R (3) | 2-3h | R:R distribution |
| 4 | Entry Timing (4) | 4-5h | Duration histogram |
| 5 | Drawdown Behavior (5) | 3-4h | Equity curve |

**All phases require**:
- Code review (per .claude/rules/improvement.md — Code Quality Gates)
- Unit tests (5+ per module)
- Live smoke test (verify log output in watch mode)
- 10-trade validation (small size)

---

## KEY FILES & INTEGRATION POINTS

| File | Method | Line# | Integration |
|------|--------|-------|-------------|
| engine.py | run_cycle() | 450 | Add isotonic calibration |
| engine.py | run_cycle() | 460 | Add pattern gate |
| engine.py | run_cycle() | 470 | Add setup quality check |
| engine.py | run_cycle() | 500 | Register tranche tracker |
| execution.py | execute_trade() | 280 | Swap fixed TP for adaptive |
| engine.py | run_cycle() | 420 | Add entry timing wait |
| engine.py | run_cycle() | 400 | Add drawdown behavior checks |
| agents/_team.py | evaluate() | 320 | Return ensemble_conflict (for decomposition) |

---

## CONFIDENCE GATES (from .claude/rules/improvement.md)

**Live Wiring Verification**:
- [ ] Each new module appears in production files (not just tests)
- [ ] Each module has a config flag in ScannerConfig
- [ ] Each module has a logger.info() call in the hot path
- [ ] Unit tests mock dependencies and pass independently
- [ ] Grep confirms method is actually called (not just defined)

**Example Check**:
```bash
grep -n "isotonic_calibrat\|_confidence_calibrator\|calibrate(" \
  src/scanner/engine.py src/scanner/agents/_team.py | wc -l
# Should be >5 lines (not counting comments or docstrings)
```

---

## RESEARCH SOURCES & REFERENCES

**Journal Patterns** (from trained_data/models/journal_patterns.json):
- LONG direction: 49.5% win rate (+11.1% vs baseline)
- Duration 480-1440min: 52.0% win rate (+13.6%)
- GBP_AUD pair: 62.5% win rate (+24.1%)
- EUR_JPY pair: 55.0% win rate (+16.6%)
- Confidence 0.71-0.80: 16.1% win rate (-22.3%) ← INVERSE correlation

**Agent Weights** (trained_data/models/agent_weights.json):
- 12 agents with learned weights (RL-tuned)
- Ensemble disagreement correlates with losses
- Uncertainty score > 0.45 signals caution

**Gate Chain Evidence**:
- Current: GateEvaluator → Agent Team → Execution
- Missing: Isotonic Cal, Pattern Gate, Setup Quality sitting unused
- Tranche Tracker only partially wired in execution
- Walk-Forward validator not live-checked

---

## NEXT STEPS

1. **Confirm wiring status**: Audit Phase 51 modules in engine.py (are they called?)
2. **Schedule sprint**: Allocate 5 weeks for sequenced implementation
3. **Create test suite**: Write unit tests for all 5 new components
4. **Live validation**: Deploy 1-2 modules, collect 50 trades, measure win rate delta

Full technical specifications: `.claude/phase52_research.md` (893 lines, code samples included)

