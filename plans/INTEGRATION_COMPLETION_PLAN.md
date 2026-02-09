# Integration Completion Plan

**Created**: 2026-02-08
**Status**: ACTIVE
**Scope**: Complete all unfinished integrations in ml_engine

---

## Executive Summary

After thorough codebase analysis, **9 workstreams** have been identified covering critical broken imports, incomplete decomposition, placeholder implementations, missing tests, and CI/CD gaps. Each workstream is designed for parallel agent execution where possible.

---

## Workstream 1: Fix Broken Engine Head Imports (CRITICAL - P0)

### Problem
5 custom Keras layers (`MLEngineHead`, `MREngineHead`, `MSEngineHead`, `MTEngineHead`, `MXEngineHead`) are quarantined in `legacy_quarantine/python/orphans/` but **actively imported by 8+ production files**. Any inference/model-loading call will crash with `ModuleNotFoundError`.

### Files Affected (imports that will fail)
- `cli/commands.py` (lines 1081-1085, 1772-1776, 3008-3012)
- `cli/io_utils.py` (lines 463-467)
- `cli/models.py` (lines 19-23)
- `src/models/model_builders.py` (lines 279-283)
- `src/training/checkpoint_manager.py` (lines 181-185)
- `scripts/buddy_audit.py` (lines 42-46)
- `scripts/replay_buddy_checkpoint.py` (lines 32-36)

### Source Files (in quarantine)
- `legacy_quarantine/python/orphans/ml_head_engine.py`
- `legacy_quarantine/python/orphans/mr_engine.py`
- `legacy_quarantine/python/orphans/ms_head_engine.py`
- `legacy_quarantine/python/orphans/mt_engine.py`
- `legacy_quarantine/python/orphans/mx_head_engine.py`

### Action
1. Move engine head files to `src/models/heads/` (proper location in modular architecture)
2. Create `src/models/heads/__init__.py` that re-exports all 5 classes
3. Create backward-compatible shim modules at project root (like existing pattern: `modular_inference.py`, `feature_engineering.py` etc.)
4. Update all 8+ importing files to use `from src.models.heads import ...`
5. Add unit tests for each engine head (instantiation, serialization roundtrip)

### Validation
- `python -c "from src.models.heads import MLEngineHead, MREngineHead, MSEngineHead, MTEngineHead, MXEngineHead"`
- All existing tests pass
- Model loading works end-to-end

---

## Workstream 2: Complete main.py Decomposition (Phase 4-5)

### Problem
`main.py` is 13,847+ lines. Phases 1-3 extracted 4,899 lines into proper modules, but **Phases 4-5 are at 0% completion**. The extracted modules are NOT wired back into main.py.

### Current State (from plans/Breakdown.md)
| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1-3 | COMPLETE | 4,899 lines extracted to 8 modules |
| Phase 4 | 0% PENDING | CLI command modularization |
| Phase 5 | 0% PENDING | Final consolidation (target: 400-600 lines) |

### Extracted Modules (ready to wire in)
- `src/training/training_config.py` (455 lines)
- `src/training/checkpoint_manager.py` (535 lines)
- `src/training/data_preparation.py` (1,037 lines)
- `src/training/retrain_gates.py` (248 lines)
- `src/models/model_builders.py` (591 lines)
- `src/utils/instrument_validation.py` (346 lines)
- `src/trading/execution.py` (880 lines)
- `src/trading/paper_trade.py` (807 lines)

### Action
1. Replace inline code blocks in main.py with imports from extracted modules
2. Create CLI command modules in `src/cli/commands/` for each command group
3. Wire `_dispatch_train_buddy`, `buddy()`, `buddy_loop()`, scan commands to CLI modules
4. Reduce main.py to CLI-only routing (~400-600 lines)
5. Verify all CLI commands still work via `./bin/Buddy` smoke tests
6. Add backward-compatibility imports at old locations

### Validation
- Every CLI command listed in copilot-instructions.md works
- No duplicate code between main.py and extracted modules
- Import cycle detection: `python -c "import main"`

---

## Workstream 3: Fix train_buddy Forward Declaration

### Problem
`cli/commands.py:34-36` has a forward declaration that raises `NotImplementedError`. The real implementation exists in `cli/training.py:40`, but the forward declaration creates confusion and may break code paths that import from `cli.commands`.

### Action
1. Replace the `NotImplementedError` stub with a proper re-export: `from cli.training import train_buddy`
2. Verify `cli/__init__.py` exports are consistent
3. Ensure `cli/legacy_commands.py` delegation paths work
4. Verify `cli/candle_optimizer.py:574` import path still resolves

### Validation
- `python -c "from cli.commands import train_buddy; print(train_buddy)"`
- `python -c "from cli import train_buddy; print(train_buddy)"`

---

## Workstream 4: Complete News Features / Sentiment Integration

### Problem
`news_features.py` has placeholder implementations for news data fetching. The module says "Override with actual implementation by setting NEWS_FETCHER callback" but no callback system is wired. `market_intelligence.py:2304` uses "Placeholder values (no news API)".

### Action
1. Implement configurable news fetcher callback system in `news_features.py`
2. Add at least one concrete news source adapter (FinnHub free API or RSS feeds)
3. Wire `NEWS_FETCHER` configuration into `config/config_improved_H1.yaml`
4. Connect news sentiment to Gate 6 (sentiment_block_threshold) in `src/core/modular_inference.py`
5. Add fallback behavior documentation (when API is unavailable, gate passes by default)
6. Add integration test with mocked news API

### Key Files
- `news_features.py` (placeholder implementations)
- `market_intelligence.py` (placeholder values)
- `src/core/modular_inference.py` (Gate 6 evaluation)
- `config/config_improved_H1.yaml` (news config section)

### Validation
- Gate 6 correctly blocks trades when strong contrary sentiment detected
- Graceful fallback when no news API configured
- Unit test with mocked sentiment data

---

## Workstream 5: Wire Online Retrainer into Main Pipeline

### Problem
`online_retrainer.py` is a standalone module with full implementation but is NOT connected to the main inference/trading pipeline. The `create_retrain_callback()` helper exists but nothing calls it.

### Action
1. Integrate `OnlineRetrainer` into the inference pipeline in `src/core/modular_inference.py`
2. Connect drift detection callback via `create_retrain_callback()` in the buddy loop (`cli/commands.py:buddy_loop()`)
3. Add retrainer status to `./bin/Buddy status` output
4. Add config section to `config/config_improved_H1.yaml`:
   ```yaml
   online_retraining:
     enabled: true
     cooldown_minutes: 60
     max_retrains_per_day: 3
     min_samples_for_retrain: 50
   ```
5. Add logging for retrain events to trade journal
6. Add integration test verifying drift -> retrain -> model update flow

### Key Files
- `online_retrainer.py` (standalone, needs integration)
- `market_intelligence.py` (DriftDetectionManager, OnlineLearner)
- `cli/commands.py` (buddy_loop for continuous trading)
- `src/core/modular_inference.py` (model reload after retrain)

### Validation
- Buddy loop detects drift and triggers retrain automatically
- Cooldown limits respected (max 3/day)
- Retrain results logged to trade journal

---

## Workstream 6: Complete RL Position Sizer Placeholders

### Problem
RL position sizer has multiple placeholder values in core modules:
- `src/rl/utils.py:224` - `recent_drawdown = 0.0` (placeholder)
- `src/rl/curriculum.py:277` - `returns_2 placeholder` (zeros)
- `src/rl/curriculum.py:282` - `bb_position placeholder` (zeros)
- `src/scanner/engine.py:649` - `drawdown=0.02` (placeholder)

### Action
1. Replace `recent_drawdown` placeholder in `src/rl/utils.py` with actual drawdown calculation from trade history
2. Replace `returns_2` and `bb_position` placeholders in `src/rl/curriculum.py` with actual multi-timeframe returns and Bollinger Band position
3. Replace scanner engine drawdown placeholder with actual portfolio drawdown tracking
4. Add unit tests for each replaced calculation
5. Verify RL training still converges with real feature values

### Key Files
- `src/rl/utils.py` (feature computation)
- `src/rl/curriculum.py` (curriculum learning)
- `src/scanner/engine.py` (scanner integration)
- `rl_position_sizing.py` (main RL module)

### Validation
- `python main.py train-rl-sizer --timesteps 10000` runs without errors
- RL agent reward improves over random baseline
- No NaN or infinity values in computed features

---

## Workstream 7: Expand CI/CD Pipeline

### Problem
`.github/workflows/code-quality.yml` only has flake8 linting and Python syntax checks. No test execution, no model validation, no integration tests, no type checking.

### Action
1. Add `pytest` job to CI pipeline:
   ```yaml
   test:
     name: Unit Tests
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v4
       - uses: actions/setup-python@v5
         with: { python-version: '3.11' }
       - run: pip install -r requirements.txt
       - run: pytest tests/ -v --tb=short -x
   ```
2. Add import validation job (verify all modules import cleanly)
3. Add type checking with mypy or pyright (incremental, src/ only)
4. Add coverage reporting (threshold: 60% minimum)
5. Ensure tests don't require OANDA credentials (mock external APIs)

### Key Files
- `.github/workflows/code-quality.yml` (existing, extend)
- `pytest.ini` (test configuration)
- `requirements.txt` (CI dependencies)

### Validation
- CI pipeline passes on clean branch
- Tests run without OANDA API keys
- Coverage baseline established

---

## Workstream 8: Add Missing Integration Tests

### Problem
Multiple critical subsystems lack test coverage:
- No test for engine head loading/serialization
- No test for end-to-end inference pipeline
- No test for online retrainer integration
- No test for news sentiment gate
- Incomplete RL training tests
- Missing calibration integration tests

### Action
1. Create `tests/test_engine_heads.py` - Engine head instantiation and Keras serialization
2. Create `tests/test_inference_pipeline.py` - End-to-end mock inference (all 8 gates)
3. Create `tests/test_online_retrainer.py` - Retrain trigger, cooldown, model update
4. Create `tests/test_news_sentiment.py` - Sentiment scoring with mocked data
5. Expand `tests/test_rl_training.py` - RL feature computation, curriculum progression
6. Create `tests/test_cli_commands.py` - CLI command dispatch, argument parsing
7. Ensure all tests can run WITHOUT external APIs (OANDA, news, etc.)

### Validation
- `pytest tests/ -v` passes with 0 failures
- Coverage > 60% for `src/` modules
- No tests require network access

---

## Workstream 9: Clean Up Deprecated / Legacy Code

### Problem
- `buddy_scanner_old.py` - Entire module deprecated (use `src.scanner`)
- `cli/legacy_commands.py` - 12+ deprecated functions with `@_deprecated` decorator
- `modular_trainers.py` (root) - Deprecated wrapper
- `src/core/modular_data_loaders.py:2058` - `load_tcn_data_legacy()` deprecated
- Root-level backward-compat shims (7 files) - Maintained but should have deprecation warnings
- Multiple `.bak_*` backup files of main.py

### Action
1. Delete `buddy_scanner_old.py` (replaced by `src/scanner/`)
2. Add deprecation warnings to root-level shim modules (log once per session)
3. Remove `load_tcn_data_legacy()` from `modular_data_loaders.py` (update callers)
4. Audit `cli/legacy_commands.py` - remove functions with zero callers, keep those still referenced
5. Delete `main.py.12k_backup` and `main.py.bak_13k` backup files
6. Add `__all__` exports to key modules to clarify public API

### Validation
- No import warnings in standard usage
- `grep -r "buddy_scanner_old" --include="*.py"` returns 0 results
- All existing tests still pass

---

## Dependency Graph (Execution Order)

```
Workstream 1 (Engine Heads) ──────────────┐
Workstream 3 (train_buddy fix) ───────────┤
Workstream 9 (Legacy cleanup) ────────────┤
                                          ├──▶ Workstream 2 (main.py decomposition)
Workstream 4 (News features) ─────────────┤        │
Workstream 5 (Online retrainer) ──────────┤        │
Workstream 6 (RL placeholders) ───────────┘        │
                                                   ▼
                                    Workstream 8 (Integration tests)
                                           │
                                           ▼
                                    Workstream 7 (CI/CD pipeline)
```

**Parallel-safe workstreams**: 1, 3, 4, 5, 6, 9 can all run in parallel.
**Sequential dependencies**: 2 depends on 1 and 3; 8 depends on all feature work; 7 depends on 8.

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Engine head restoration breaks model loading | HIGH | Test with existing .keras files before/after |
| main.py decomposition introduces regressions | HIGH | Run full CLI smoke test after each phase |
| RL placeholder changes affect training convergence | MEDIUM | Compare reward curves before/after |
| News API integration adds external dependency | LOW | Make fully optional with graceful fallback |
| CI/CD tests fail due to TF/Metal dependency | MEDIUM | Use CPU-only TF in CI, mock Metal calls |
