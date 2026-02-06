# ML Engine - Project Breakdown & Phase Tracking

## Executive Summary

This document tracks the implementation phases for the FX Trading Bot ML Engine, documenting what has been completed and what remains to be done.

---

## Phase Status Overview

| Phase | Description | Status | PR |
|-------|-------------|--------|-----|
| Phase 1 | Quick Wins - M1 Metal Optimizations | COMPLETE | Merged |
| Phase 2 | Model Convergence Improvements | COMPLETE | Merged |
| Phase 3 | Data Quality & Multi-Instrument | COMPLETE | Merged |
| Phase 4 | Production Hardening | COMPLETE | #17 (Merged) |
| Phase 5 | Codebase Modularization | IN PROGRESS | - |

---

## Phase 1: Quick Wins - M1 Metal Optimizations

**Status:** COMPLETE

### Completed Items
- [x] Disable RecurrentDropout for Metal GPU compatibility
- [x] Increase batch size from 32 to 64-128 for M1 optimization
- [x] Use native `tf.keras.optimizers.AdamW`
- [x] Add proper `AUTOTUNE` + `cache()` to tf.data pipeline
- [x] Create `config_m1_optimized.yaml` configuration

### Files Modified
- `src/models/tensorflow_models.py` - Set `recurrent_dropout=0.0`
- `config/config_m1_optimized.yaml` - New optimized config
- `src/training/tensorflow_data_pipeline.py` - Prefetching optimization

### Evidence
- PRs #1-5 merged implementing core ML improvements
- Intel Mac support added (PR #13)

---

## Phase 2: Model Convergence Improvements

**Status:** COMPLETE

### Completed Items
- [x] Rebalance loss weights (direction from 20.0 to 5.0)
- [x] Add learning rate warmup with CosineDecay
- [x] Implement walk-forward validation
- [x] Add focal loss for imbalanced direction labels
- [x] Reduce model complexity for dataset size

### Files Modified
- `config/config_improved_H1.yaml` - Updated loss weights
- `src/training/walkforward_validation.py` - Walk-forward CV implementation
- `src/models/tensorflow_engine.py` - LR scheduling

### Evidence
- PRs #3, #5 merged with model improvements
- Walk-forward validation fully operational

---

## Phase 3: Data Quality & Multi-Instrument

**Status:** COMPLETE

### Completed Items
- [x] Fetch more historical data (15,000+ H1 candles)
- [x] Multi-instrument data loading
- [x] News sentiment features via FinBERT
- [x] Economic calendar integration
- [x] Post-trade incremental training

### Files Modified
- `src/core/modular_data_loaders.py` - Multi-instrument support
- `market_intelligence.py` - News sentiment via FinBERT
- `news_features.py` - Economic calendar features
- `online_retrainer.py` - Incremental training

### Evidence
- PR #10 - Sentiment analysis features
- PR #14 - Data leakage fixes
- PR #15 - LLM wrapper enhancements

---

## Phase 4: Production Hardening

**Status:** COMPLETE (PR #17 Draft)

### Completed Items
- [x] Trading Guardrails (`src/risk/fx_guardrails.py`)
  - [x] Session time management (08:00-11:30 EST)
  - [x] Force-flat cutoff (11:55)
  - [x] Daily loss stop (5% default)
  - [x] Maximum drawdown (10% default)
  - [x] Entry limits (1 per day)
  - [x] Spread filtering per instrument
  - [x] Confidence-based profit stops

- [x] Advanced Backtesting (`src/training/walkforward_validation.py`)
  - [x] Monte Carlo simulation (1000+ runs)
  - [x] Transaction cost modeling (spread, slippage, commission)
  - [x] Walk-forward + Monte Carlo combined
  - [x] Confidence intervals for Sharpe ratio
  - [x] P(profitable) probability estimates

- [x] Monitoring & Alerting (`src/utils/monitoring.py`)
  - [x] Alert system (INFO, WARNING, CRITICAL)
  - [x] Model drift detection
  - [x] Performance tracking with baselines
  - [x] CLI dashboard (`buddy monitor`)
  - [x] JSON report export

- [x] TensorBoard Integration (already existed)

### Files Created/Modified
- `src/utils/monitoring.py` (590 lines) - NEW
- `scripts/monitoring_dashboard.py` (257 lines) - NEW
- `tests/test_monitoring.py` (241 lines) - NEW
- `src/training/walkforward_validation.py` (+298 lines)
- `main.py` (+216 lines for monitor command)
- `docs/PHASE_4_COMPLETE.md` - Comprehensive documentation
- `docs/PHASE_4_SUMMARY.md` - Summary documentation

### CLI Commands Added
```bash
buddy monitor                    # Dashboard
buddy monitor --monitor-alerts   # Detailed alerts
buddy monitor --monitor-drift    # Model drift history
buddy monitor --monitor-report   # Full JSON report
```

### Evidence
- PR #17 - 2,558 additions implementing all Phase 4 features
- Full documentation in `docs/PHASE_4_COMPLETE.md`

---

## Phase 5: Codebase Modularization

**Status:** IN PROGRESS

### Problem Statement
`main.py` has grown to **12,987 lines** (567KB), making it difficult for:
- Agents to extract and work on specific sections
- Developers to navigate and maintain
- Testing individual components
- Code review efficiency

### Proposed Module Split

Based on analysis, main.py should be split into **9 focused modules**:

#### 1. `cli/config.py` (Lines 1-140)
- Data classes: `OandaFetchOptions`, `BuddyTrainingAdvancedOptions`, `BuddyTrainingOptions`
- Configuration models

#### 2. `cli/calibration.py` (Lines 141-277)
- Tier-2 calibration utilities (`_tier2_*` functions)
- TP-before-SL simulation

#### 3. `cli/io_utils.py` (Lines 280-640)
- OANDA data fetching (`_oanda_fetch_to_csv`)
- Model loading/checkpointing (`_load_buddy_checkpoint`, `_migrate_keras2_to_keras3`)
- Instrument normalization

#### 4. `cli/wizard.py` (Lines 749-1086)
- Interactive wizard (`_buddy_interactive_wizard`)
- TensorFlow configuration helpers
- CLI input parsing

#### 5. `cli/models.py` (Lines 1087-1340)
- Model architectures
- `_build_buddy_model()`, `_build_buddy_model_tcn()`, `_build_xgboost_model()`

#### 6. `cli/training.py` (Lines 1343-5260)
- `train_buddy()` - Main training API
- `_train_buddy_impl()` - Core training logic (~3800 lines)
- Enterprise features, continual learning

#### 7. `cli/fx_trading.py` (Lines 5262-6155)
- Paper trading and FX execution
- `_FxPaperTradePlan`, risk management
- Policy enforcement, signal processing

#### 8. `cli/commands.py` (Lines 5797-12059)
- High-level command implementations
- `buddy()`, `buddy_scan()`, `buddy_monitor()`
- Model management, validation

#### 9. `cli/__init__.py` + `main.py` (Lines 12060-12986)
- `main()` function with argparse CLI
- Command routing and dispatch
- Minimal entry point

### Implementation Tasks

- [ ] Create `cli/` package directory
- [ ] Extract `cli/config.py` - Data classes
- [ ] Extract `cli/calibration.py` - Tier-2 utilities
- [ ] Extract `cli/io_utils.py` - I/O functions
- [ ] Extract `cli/wizard.py` - Interactive wizard
- [ ] Extract `cli/models.py` - Model builders
- [ ] Extract `cli/training.py` - Training logic
- [ ] Extract `cli/fx_trading.py` - FX execution
- [ ] Extract `cli/commands.py` - Command implementations
- [ ] Refactor `main.py` to minimal dispatcher
- [ ] Update all imports across codebase
- [ ] Run full test suite
- [ ] Verify all CLI commands work

### Priority Order
1. **HIGH**: Extract training.py (3800 lines) - most complex
2. **HIGH**: Extract commands.py (6000+ lines) - reduces main.py significantly
3. **MEDIUM**: Extract fx_trading.py, models.py
4. **LOW**: Extract config.py, calibration.py, io_utils.py, wizard.py

---

## Existing Merged PRs Reference

| PR | Title | Status |
|----|-------|--------|
| #17 | Phase 4: Production Hardening - Monte Carlo Backtesting and Monitoring | DRAFT |
| #16 | Refactor: Clean up repository structure and remove legacy code | MERGED |
| #15 | Add Buddy Intelligent Mode: LLM wrapper enhancements | MERGED |
| #14 | Fix data leakage in XGBoost and Ridge data loaders | MERGED |
| #13 | Intel mac optimized | MERGED |
| #12 | Add quant critic module for improving Buddy ML trading rationale | MERGED |
| #11 | Add autonomous self-improvement to Buddy AI using Self-Refine | MERGED |
| #10 | Add news sentiment features, economic calendar, post-trade training | MERGED |
| #9 | Add Raynergy-svg watermark to all Python modules | MERGED |
| #8 | Add risk disclaimer and streamline README | MERGED |
| #7 | Update issue templates | MERGED |
| #6 | Rewrite README as comprehensive Buddy CLI documentation | OPEN |
| #5 | Enhance ML model predictions | MERGED |
| #4 | Add checkpoint resume capability | MERGED |
| #3 | Refactor ML Engine: Fix syntax errors, add validation | MERGED |
| #2 | Harden ML trading engine with security patches | MERGED |
| #1 | Complete truncated Python files, improve ML engine | MERGED |

---

## Architecture Overview

```
ml_engine/
├── bin/
│   └── Buddy                          # Shell wrapper for CLI
├── cli/                               # NEW - Modularized CLI (Phase 5)
│   ├── __init__.py
│   ├── config.py                      # Data classes
│   ├── calibration.py                 # Tier-2 calibration
│   ├── io_utils.py                    # I/O utilities
│   ├── wizard.py                      # Interactive wizard
│   ├── models.py                      # Model builders
│   ├── training.py                    # Training logic
│   ├── fx_trading.py                  # FX execution
│   └── commands.py                    # Command implementations
├── config/
│   ├── config_improved_H1.yaml        # H1 timeframe config (DEFAULT)
│   └── config_m1_optimized.yaml       # Apple Silicon optimized
├── src/
│   ├── core/
│   │   ├── modular_inference.py       # Gated ensemble inference
│   │   └── modular_data_loaders.py    # Feature preparation
│   ├── models/
│   │   ├── tensorflow_models.py       # Transformer, TCN, TFT
│   │   ├── tensorflow_engine.py       # Training pipeline
│   │   └── ensemble_model.py          # Ensemble stacking
│   ├── training/
│   │   ├── modular_trainers.py        # Model trainers
│   │   ├── buddy_training_helpers.py  # Training orchestration
│   │   └── walkforward_validation.py  # Walk-forward CV + Monte Carlo
│   ├── risk/
│   │   ├── fx_guardrails.py           # Trading guardrails
│   │   ├── position_sizing.py         # Kelly-based sizing
│   │   └── triple_barrier.py          # Trade labeling
│   └── utils/
│       ├── oanda_practice.py          # OANDA API client
│       ├── trade_journal.py           # Trade logging
│       └── monitoring.py              # Monitoring & alerting
├── scripts/
│   └── monitoring_dashboard.py        # Monitoring CLI
├── tests/
│   ├── test_fx_guardrails.py
│   ├── test_monitoring.py
│   └── ...
├── main.py                            # CLI entry point (to be minimized)
├── plans/
│   └── Breakdown.md                   # This file
└── docs/
    ├── PHASE_4_COMPLETE.md
    └── PHASE_4_SUMMARY.md
```

---

## Next Steps

1. **Merge PR #17** - Phase 4 Production Hardening (currently draft)
2. **Create new branch** for Phase 5 modularization
3. **Split main.py** into the 9 proposed modules
4. **Update tests** to work with new module structure
5. **Verify all CLI commands** still work correctly

---

## Agent Assignment for Phase 5

For efficient parallel work, agents should be assigned to independent modules:

| Agent | Module | Lines | Dependencies |
|-------|--------|-------|--------------|
| Agent 1 | cli/config.py | 1-140 | None (leaf module) |
| Agent 2 | cli/calibration.py | 141-277 | config.py |
| Agent 3 | cli/io_utils.py | 280-640 | config.py |
| Agent 4 | cli/wizard.py | 749-1086 | config.py |
| Agent 5 | cli/models.py | 1087-1340 | config.py |
| Agent 6 | cli/training.py | 1343-5260 | All above |
| Agent 7 | cli/fx_trading.py | 5262-6155 | config.py, io_utils.py |
| Agent 8 | cli/commands.py | 5797-12059 | All above |
| Coordinator | main.py refactor | 12060-12986 | All CLI modules |

**Recommended Order:**
1. Agents 1-5 in parallel (leaf modules)
2. Agent 6 (training) - depends on 1-5
3. Agents 7-8 in parallel - depend on subsets
4. Coordinator finalizes main.py

---

## Success Criteria

- [ ] All 9 modules extracted successfully
- [ ] `main.py` reduced to < 500 lines
- [ ] All 16 CLI commands functional
- [ ] Full test suite passes
- [ ] No circular imports
- [ ] Documentation updated
- [ ] PR created and merged

---

*Last updated: 2026-02-05*
