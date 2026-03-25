# Phase 52 Research Index

## Documents

### 1. PHASE52_SUMMARY.md (233 lines)
Quick reference guide with:
- Problem statement for each of 5 topics
- Implementation strategy
- Expected win rate impact (+7% target)
- 5-week timeline
- Integration checkpoints

**Read this first** for executive overview.

---

### 2. phase52_research.md (893 lines)
Deep technical specification with:
- Detailed research on each topic
- Full Python code samples (copy-paste ready)
- Integration patterns for engine.py, execution.py, agents/_team.py
- Validation strategy per component
- Backtest evidence from 522-trade journal

**Read this second** for implementation details.

---

## Quick Navigation

**Topic 1: Live Pipeline Wiring**
- Summary: PHASE52_SUMMARY.md line 10
- Technical: phase52_research.md line 15
- Evidence: Grep for module names in engine.py

**Topic 2: Confidence Decomposition**
- Summary: PHASE52_SUMMARY.md line 52
- Technical: phase52_research.md line 140
- Code: ConfidenceDecomposer class (copy-paste)

**Topic 3: Adaptive R:R Targeting**
- Summary: PHASE52_SUMMARY.md line 100
- Technical: phase52_research.md line 320
- Code: AdaptiveTPCalculator with regime matrix

**Topic 4: Entry Timing Refinement**
- Summary: PHASE52_SUMMARY.md line 145
- Technical: phase52_research.md line 490
- Code: AdaptiveEntryTiming (30-min microstructure wait)

**Topic 5: Drawdown-Adaptive Behavior**
- Summary: PHASE52_SUMMARY.md line 190
- Technical: phase52_research.md line 610
- Code: DrawdownRecoveryManager (3-lever adaptation)

---

## Implementation Checklist

- [ ] Week 1: Audit Phase 51 wiring status (is code being called?)
- [ ] Week 1-2: Implement pipeline wiring (prerequisite)
- [ ] Week 2: Implement decomposition module
- [ ] Week 3: Implement adaptive R:R module
- [ ] Week 4: Implement entry timing module
- [ ] Week 5: Implement drawdown behavior module

Each module requires:
- [ ] Unit tests (5+ cases)
- [ ] Config flag in ScannerConfig
- [ ] Logger.info() in hot path (for smoke test verification)
- [ ] Grep verification (method appears in production files)
- [ ] 10-trade live validation

---

## Key Evidence from Journal

**Baseline**: 38% win rate, 522 trades

**Patterns that Work**:
- LONG direction: 49.5% ↑11.1%
- Duration 480-1440min: 52.0% ↑13.6%
- GBP_AUD pair: 62.5% ↑24.1%
- Confidence 0.53-0.62: 42.3% ↑3.9%

**Patterns that Fail**:
- Confidence 0.71-0.80: 16.1% ↓22.3% ← IMPORTANT
- SHORT direction: 25.4% ↓13.0%
- Duration 120-240min: 26.1% ↓12.3%

**Insight**: High confidence can hurt! Decomposition needed to separate signal.

---

## Files to Create (Phase 52)

```
src/scanner/
├── confidence_decomposition.py   (150 lines)
├── adaptive_tp.py                (120 lines)
├── entry_timing.py               (180 lines)
└── drawdown_recovery.py          (150 lines)

tests/
├── test_confidence_decomposition.py
├── test_adaptive_tp.py
├── test_entry_timing.py
└── test_drawdown_recovery.py
```

---

## Integration Points (Lines Approximate)

| File | Method | Integration |
|------|--------|-------------|
| engine.py | run_cycle | ~400: Add drawdown checks |
| engine.py | run_cycle | ~420: Add entry timing wait |
| engine.py | run_cycle | ~450: Add isotonic calibration |
| engine.py | run_cycle | ~460: Add pattern gate |
| engine.py | run_cycle | ~470: Add setup quality filter |
| engine.py | run_cycle | ~500: Register tranche tracker |
| execution.py | execute_trade | ~280: Swap fixed TP for adaptive |
| config.py | ScannerConfig | Add 5 feature flags |

---

## Validation Commands

```bash
# Check Phase 51 wiring
grep -n "isotonic\|pattern_gate\|setup_quality\|tranche\|walk_forward" \
  src/scanner/engine.py | grep -v "^.*:.*#"

# Run all Phase 52 tests
pytest tests/test_confidence_decomposition.py \
       tests/test_adaptive_tp.py \
       tests/test_entry_timing.py \
       tests/test_drawdown_recovery.py -v

# Live smoke test (30 min watch cycle)
python -m src.scanner.engine --watch --pairs EUR_USD GBP_AUD EUR_JPY
# Look for log lines:
# "Phase 52: Decomposed confidence"
# "Adaptive TP:"
# "Phase 52 entry timing:"
# "Phase 52 drawdown:"
```

---

## Success Metrics

**End of Week 1**: All Phase 51 modules confirmed wired and logging
**End of Week 2**: Decomposition module deployed, 20 trades logged
**End of Week 3**: Adaptive R:R live, distribution analysis complete
**End of Week 4**: Entry timing live, duration histogram shows clustering
**End of Week 5**: Drawdown behavior active, equity curve shows smoothing

**Final Target**: 38% → 45% win rate by end of Phase 52 (7% absolute improvement)

---

