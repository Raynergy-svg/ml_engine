# main.py Extraction Map

**File:** [main.py](../main.py)  
**Total Lines:** 13,848  
**Generated:** 2025-01-XX  

---

## Overview

This document provides exact line ranges for extracting code from `main.py` into modular components. Each section identifies:
- **Line Range:** Exact start and end lines
- **Functions/Classes:** All identifiers found
- **Dependencies:** Global variables, imports, and cross-references
- **Target File:** Where to extract this code

---

## 1. Imports & Setup (Lines 1-31)

### Line Range: 1-31
### Target: Keep in main.py (reduced set) + distribute to target modules

#### Contents:
- Standard library imports
- TensorFlow logging suppression
- Rich console initialization
- Type hints

#### Key Lines:
```
Lines 1-10:   Standard imports (sys, os, logging, time, json, argparse, etc.)
Lines 11-20:  TensorFlow/Keras imports with warning suppression
Lines 21-31:  Rich console, typing, Path imports
```

---

## 2. Dataclasses & Configuration (Lines 33-139)

### Target: `src/training/training_config.py`

#### Functions/Classes:
| Line | Name | Type | Description |
|------|------|------|-------------|
| 33-41 | `OandaFetchOptions` | @dataclass | OANDA fetch parameters (instrument, granularity, candles, price, save_csv) |
| 44-68 | `BuddyTrainingAdvancedOptions` | @dataclass | Tier-2 calibration, smoothing, filtering options |
| 71-139 | `BuddyTrainingOptions` | @dataclass | Main training config (epochs, batch_size, lr, M1 optimizations, RL sizer, enterprise options) |

#### Dependencies:
- `from dataclasses import dataclass, field`
- `from typing import Optional, Any`
- `from pathlib import Path`

---

## 3. Tier-2 Calibration Functions (Lines 142-289) ⚠️ DUPLICATE

### Target: `src/cli/calibration.py` (ALREADY EXISTS)
### Action: REMOVE from main.py after verification

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 142-148 | `_tier2_get_calibration_dict()` | Get calibration dict from meta |
| 151-168 | `_tier2_points_from_bins()` | Convert bins to calibration points |
| 171-199 | `_tier2_interpolate_points()` | Interpolate calibration curve |
| 202-210 | `_tier2_clip_prob()` | Clip probability to [eps, 1-eps] |
| 213-217 | `_tier2_logit()` | Probability to logit transform |
| 220-227 | `_tier2_sigmoid()` | Logit to probability transform |
| 230-238 | `_tier2_temperature_scale_prob()` | Apply temperature scaling |
| 241-247 | `_tier2_nll()` | Negative log-likelihood loss |
| 250-263 | `_tier2_spearman()` | Spearman correlation |
| 266-289 | `_tier2_apply_calibration()` | Apply full calibration pipeline |

#### Dependencies:
- `numpy` for math operations
- `scipy.stats` for Spearman correlation

---

## 4. OANDA Data Fetch (Lines 292-395)

### Target: `src/data/oanda_fetcher.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 292-395 | `_oanda_fetch_to_csv()` | Fetch candles from OANDA with pagination, save to CSV |

#### Dependencies:
- `OandaPracticeClient` from `src.utils.oanda_practice`
- `OandaFetchOptions` dataclass
- Console for logging

---

## 5. Keras Migration Helper (Lines 398-459)

### Target: Keep in main.py (internal utility) or `src/utils/keras_compat.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 398-459 | `_migrate_keras2_to_keras3()` | Rebuild model architecture for Keras version compatibility |

#### Dependencies:
- TensorFlow/Keras
- Custom layer classes

---

## 6. Checkpoint Management (Lines 462-518)

### Target: `src/training/checkpoint_manager.py`

#### Functions/Constants:
| Line | Name | Type | Description |
|------|------|------|-------------|
| 462-505 | `_load_buddy_checkpoint()` | Function | Load model with custom objects |
| 508 | `BUDDY_META_FILENAME` | Constant | "buddy_tf.meta.json" |
| 511-518 | `_meta_path_for_checkpoint()` | Function | Find metadata file for checkpoint |

#### Dependencies:
- TensorFlow/Keras
- Custom layer classes (MLEngineHead, MREngineHead, etc.)
- Path operations

---

## 7. Constants & Console Helpers (Lines 520-610)

### Target: Distribute to respective modules

#### Constants (Lines 520-530):
| Line | Name | Value |
|------|------|-------|
| 520 | `DEFAULT_CONFIG_PATH` | "config/config_improved_H1.yaml" |
| 521 | `TIMESTAMP_FORMAT` | "%Y%m%d_%H%M%S" |
| 522 | `DEFAULT_CURRICULUM_KS` | "20,50,100,0" |
| 523 | `UTC_OFFSET_SUFFIX` | "+00:00" |

#### Console & Logger (Lines 532-545):
- `console = Console()`
- `logger = setup_logging()`

#### Helper Functions (Lines 550-610):
| Line | Name | Description |
|------|------|-------------|
| 550-560 | `_parse_bool_answer()` | Parse y/n user input |
| 563-575 | `_parse_float_answer()` | Parse float user input |
| 578-590 | `_parse_int_answer()` | Parse int user input |

---

## 8. Instrument Validation (Lines 613-747)

### Target: `src/utils/instrument_validation.py`

#### Functions/Constants:
| Line | Name | Type | Description |
|------|------|------|-------------|
| 613-629 | `VALID_OANDA_INSTRUMENTS` | Set | 60+ valid FX pairs |
| 632-637 | `_normalize_instrument()` | Function | Normalize pair string (EUR/USD → EUR_USD) |
| 640-680 | `_extract_instrument_from_csv_path()` | Function | Extract pair from filename via regex |
| 683-719 | `_get_pair_model_paths()` | Function | Generate pair-specific model paths |
| 722-747 | `_validate_instrument()` | Function | Validate with typo suggestions |

#### Dependencies:
- `re` for regex
- `difflib.get_close_matches` for suggestions

---

## 9. Interactive Wizard Functions (Lines 750-948)

### Target: `src/cli/wizard.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 750-780 | `launch_buddy_repl_from_wizard()` | Launch interactive REPL |
| 783-820 | `_buddy_wizard_ask()` | Get user input with prompt |
| 823-850 | `_buddy_wizard_menu()` | Display menu options |
| 853-948 | `_buddy_interactive_wizard()` | Main CLI wizard for mode selection |

#### Dependencies:
- Rich console
- sys.stdin for input

---

## 10. TensorFlow Configuration (Lines 950-1054)

### Target: `src/utils/tf_config.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 950-965 | `_configure_predict_output()` | Suppress TF warnings |
| 968-990 | `_tf_get_metal_device()` | Get Metal GPU device |
| 993-1010 | `_tf_set_memory_growth()` | Enable memory growth |
| 1013-1054 | `_configure_tf_metal()` | Full Metal GPU configuration |

#### Dependencies:
- TensorFlow
- Platform detection (Apple Silicon)

---

## 11. Model Builders (Lines 1057-1330)

### Target: `src/models/model_builders.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 1057-1100 | `_build_buddy_model()` | 5-head LSTM architecture |
| 1103-1143 | `_build_buddy_model_shared_encoder()` | Shared LSTM encoder variant |
| 1146-1249 | `_build_buddy_model_tcn()` | TCN with dilated causal convolutions (M1 optimized) |
| 1252-1270 | `_build_xgboost_model()` | XGBoost wrapper |
| 1273-1330 | `_build_buddy_model_for_type()` | Factory function for model selection |

#### Dependencies:
- TensorFlow/Keras layers
- Custom TCN implementation
- XGBoost

---

## 12. Training Entry Points (Lines 1332-1423)

### Target: `src/training/training_pipeline.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 1332-1351 | `train_buddy()` | Public training entry point (wrapper) |
| 1354-1423 | `_train_rl_position_sizer_if_ready()` | RL sizer training helper |

#### Dependencies:
- BuddyTrainingOptions
- Config loading

---

## 13. `_train_buddy_impl()` - MASSIVE FUNCTION (Lines 1430-5554)

### Total: 4,124 lines - REQUIRES DECOMPOSITION

### Target: Multiple files

#### Subsections:

| Line Range | Section | Target File |
|------------|---------|-------------|
| 1430-1600 | Config loading, environment setup | `src/training/training_pipeline.py` |
| 1600-1800 | Multi-pair mode, CSV loading, OOS split | `src/training/data_preparation.py` |
| 1800-2100 | Feature engineering, standardization, target creation | `src/training/data_preparation.py` |
| 2100-2400 | Dataset creation with tf.data pipelines | `src/training/data_preparation.py` |
| 2400-2600 | Feature curriculum callback | `src/training/callbacks.py` |
| 2600-3200 | Ensemble training (Transformer, XGBoost, RF, Ridge) | `src/training/modular_trainers.py` (exists) |
| 3200-3600 | Enterprise validation (MLflow, Bootstrap CI, Walk-Forward CV) | `src/training/enterprise_training.py` |
| 3600-4000 | XGBoost standalone path, RL sizer | `src/training/training_pipeline.py` |
| 4000-4400 | Combined objective callback, early stopping | `src/training/callbacks.py` |
| 4400-4700 | OOS replay evaluation | `src/training/evaluation.py` |
| 4700-5200 | Tier-2 calibration (temperature scaling, bins) | `src/training/tier2_calibration.py` |
| 5200-5450 | Meta-labeling training | `src/training/meta_labeling.py` |
| 5450-5554 | Metadata JSON creation, model saving | `src/training/training_pipeline.py` |

#### Key Internal Functions:
- Dataset windowing functions
- Calibration simulation
- Bootstrap confidence intervals
- Walk-forward cross-validation

---

## 14. FX Trading Helper Functions (Lines 5556-6100)

### Target: `src/trading/execution.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 5556-5568 | `_buddy_live_enabled_from_meta()` | Check if live trading enabled |
| 5571-5582 | `_fx_confirm()` | Confirm trading action |
| 5585-5596 | `_fx_open_position_instruments()` | Get open position instruments |
| 5599-5620 | `_fx_enforce_fx_policy()` | Enforce trading policy |
| 5623-5660 | `_fx_refresh_fx_state()` | Refresh FX state from broker |
| 5663-5680 | `_fx_maybe_force_flat()` | Force close positions if needed |
| 5683-5688 | `_fx_gate_fx_entry()` | Gate check for entry |
| 5691-5710 | `_fx_load_fx_df()` | Load FX dataframe |
| 5713-5722 | `_fx_spread_and_slippage()` | Calculate spread and slippage |
| 5725-5732 | `_fx_require_account_metrics()` | Require account metrics |
| 5735-5758 | `_fx_get_signal_context()` | Get signal context |
| 5761-5786 | `_fx_build_risk_rules()` | Build risk rules |
| 5789-5804 | `_fx_compute_confidence_and_band()` | Compute confidence band |
| 5807-5838 | `_fx_apply_daily_stops()` | Apply daily stop limits |
| 5841-5874 | `_fx_build_order_units_and_prices()` | Build order parameters |
| 5877-5915 | `_fx_execution_guard_price_bound()` | Price boundary guard |
| 5918-5956 | `_schedule_auto_close()` | Schedule auto-close for scalping |

---

## 15. Paper Trading & Dashboard (Lines 5958-6600)

### Target: `src/trading/paper_trading.py` and `src/cli/dashboard.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 5958-6100 | Various `_fx_*` continuation | FX helper functions |
| 6101-6200 | `fx_paper_trade()` | Paper trading entry |
| 6201-6330 | Dashboard generation helpers | Dashboard layout |
| 6336-6366 | `generate_dashboard()` | Rich dashboard generation |
| 6369-6440 | More dashboard functions | Dashboard components |
| 6445-6595 | `train_model()` | Legacy training function |
| 6596-6600 | `evaluate_model()` | Legacy evaluation function |

---

## 16. CLI Commands (Lines 6601-6876)

### Target: Keep in main.py or `src/cli/commands.py`

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 6633-6640 | `visualize_dashboard()` | Dashboard visualization |
| 6643-6700 | `openai_tune()` | OpenAI-based auto-tuning |
| 6703-6740 | `predict_price()` | Price prediction |
| 6743-6760 | `realtime_loop()` | Real-time inference loop |
| 6763-6780 | `tune_model()` | Hyperparameter tuning |
| 6783-6800 | `profile_pipeline()` | Pipeline profiling |
| 6803-6810 | `run_ai_assistant()` | RETIRED |
| 6813-6830 | `train_unified()` | Legacy alias |
| 6833-6860 | `train_oanda_unified()` | Legacy alias |
| 6863-6876 | `chat_unified()` | Interactive chat |

---

## 17. `buddy()` Function (Lines 6877-8418)

### Total: 1,541 lines
### Target: `src/inference/buddy_inference.py`

#### Key Subsections:
| Line Range | Section | Description |
|------------|---------|-------------|
| 6877-6950 | Function signature & docstring | All parameters documented |
| 6950-7050 | Intelligent mode initialization | LLM provider setup |
| 7050-7200 | Journal sync & online learning | Trade journal integration |
| 7200-7350 | Account balance fetch | Live NAV from OANDA |
| 7350-7450 | Aggressive scaling engine | $100K→$1M strategy |
| 7450-7650 | Modular ensemble inference path | 4-gate system |
| 7650-7900 | Trade execution logic | Order placement |
| 7900-8200 | Fallback to standard model | Legacy TF model path |
| 8200-8418 | Tier-1/Tier-2 gating | Confidence thresholds |

#### Dependencies:
- ModularEnsembleInference
- OandaPracticeClient
- FeatureEngineering
- TradeJournal
- AggressiveScalingEngine (optional)
- LLM providers (optional)

---

## 18. `buddy_loop()` Function (Lines 8421-8977)

### Total: 556 lines
### Target: `src/inference/buddy_loop.py`

#### Key Sections:
| Line Range | Section | Description |
|------------|---------|-------------|
| 8421-8480 | Function signature | Loop parameters |
| 8480-8550 | Candle streaming setup | Deque buffer |
| 8550-8700 | Candle boundary calculation | Sleep until next candle |
| 8700-8850 | Inference loop | Per-candle prediction |
| 8850-8977 | Order execution | Trade placement on signal |

#### Dependencies:
- Same as `buddy()`
- Time boundary calculations
- Streaming candle buffer

---

## 19. Dispatch Helpers (Lines 8980-9100)

### Target: Keep in main.py

#### Functions:
| Line | Name | Description |
|------|------|-------------|
| 8980-8995 | `_normalize_command_args()` | Normalize CLI args |
| 8998-9020 | `_maybe_run_buddy_interactive_wizard()` | Check for wizard mode |
| 9023-9035 | `_maybe_launch_buddy_repl()` | Check for REPL mode |
| 9038-9055 | `_compute_force_units()` | Compute force units |
| 9058-9100 | `_dispatch_buddy()` | Dispatch buddy command |

---

## 20. `_dispatch_train_buddy()` (Lines 9103-9350)

### Target: Keep in main.py or `src/cli/dispatch.py`

#### Line Range: 9103-9350
- Reads config
- Sets training parameters
- Calls `train_buddy()`

---

## 21. `buddy_scan()` Function (Lines 9353-9550)

### Target: `src/scanner/scanner_cli.py`

#### Key Sections:
| Line Range | Section |
|------------|---------|
| 9353-9400 | Function signature & docstring |
| 9400-9450 | Scanner initialization |
| 9450-9500 | Pair scanning loop |
| 9500-9550 | Results display |

---

## 22. `_display_scan_results_v3()` (Lines 9553-9600)

### Target: `src/scanner/scanner_cli.py`

---

## 23. `buddy_predict_78()` (Lines 9603-9780)

### Target: `src/scanner/predict.py`

---

## 24. `_buddy_scan_legacy()` (Lines 9783-9960)

### Target: Can be removed (legacy fallback)

---

## 25. `train_rl_sizer()` (Lines 9964-10195)

### Target: `src/rl/training.py`

#### Key Sections:
| Line Range | Section |
|------------|---------|
| 9964-10000 | Function signature |
| 10000-10100 | Data generation |
| 10100-10150 | Subprocess PPO training |
| 10150-10195 | Finalization |

---

## 26. `retrain_gates()` (Lines 10198-10600)

### Target: `src/training/gate_trainer.py`

#### Key Sections:
| Line Range | Section |
|------------|---------|
| 10198-10250 | Function signature |
| 10250-10400 | OANDA data fetch |
| 10400-10500 | Feature computation |
| 10500-10550 | XGBoost training |
| 10550-10580 | RandomForest training |
| 10580-10600 | Ridge training & metadata |

---

## 27. `train_confidence_model()` (Lines 10603-10800)

### Target: `src/training/confidence_trainer.py`

---

## 28. `recalibrate_scanner()` (Lines 10803-10845)

### Target: `src/risk/confidence_calibration.py` (already exists)

---

## 29. `train_tcn_regime()` (Lines 10848-11280)

### Target: `src/training/tcn_regime_trainer.py`

#### Key Sections:
| Line Range | Section |
|------------|---------|
| 10848-10900 | Function signature |
| 10900-11000 | Data fetch from OANDA |
| 11000-11100 | Forward-looking label creation |
| 11100-11200 | TCN model training |
| 11200-11280 | Model saving & summary |

---

## 30. `suggest_improvements()` (Lines 11283-11475)

### Target: `src/cli/meta_learning.py`

---

## 31. `buddy_journal()` (Lines 11478-11785)

### Target: `src/cli/journal_cli.py`

---

## 32. `buddy_analyze()` (Lines 11788-12095)

### Target: `src/cli/analyze_cli.py`

---

## 33. `_buddy_test_modular_ensemble()` (Lines 12098-12390)

### Target: `src/validation/hindcast.py`

---

## 34. `buddy_validate()` (Lines 12393-12430)

### Target: `src/validation/validate_cli.py`

---

## 35. `buddy_test()` (Lines 12433-12600)

### Target: DEPRECATED - redirects to validate

---

## 36. `model_status()` (Lines 12770-12870)

### Target: `src/cli/status.py`

---

## 37. `promote_model()` (Lines 12873-12980)

### Target: `src/cli/model_management.py`

---

## 38. `main()` - CLI Entry Point (Lines 12983-13848)

### Target: Keep in main.py (reduced)

#### Key Sections:
| Line Range | Section |
|------------|---------|
| 12983-13000 | Function signature & description |
| 13000-13200 | ArgumentParser setup |
| 13200-13500 | Training arguments |
| 13500-13650 | Inference arguments |
| 13650-13750 | Enterprise/MLOps arguments |
| 13750-13848 | Command dispatch & execution |

---

## Global Dependencies Summary

### Shared Constants (extract to `src/constants.py`):
- `DEFAULT_CONFIG_PATH`
- `TIMESTAMP_FORMAT`
- `DEFAULT_CURRICULUM_KS`
- `UTC_OFFSET_SUFFIX`
- `BUDDY_META_FILENAME`
- `VALID_OANDA_INSTRUMENTS`

### Shared Utilities (already exist or create):
- Console instance → `src/utils/logging.py`
- Config loading → `src/utils/config.py` (exists)
- TF configuration → `src/utils/tf_config.py`
- Instrument validation → `src/utils/instrument_validation.py`

---

## Extraction Priority

### Phase 1 - Quick Wins (Low Risk):
1. ✅ Tier-2 calibration functions (DUPLICATE - just remove)
2. Instrument validation (self-contained)
3. TF configuration (self-contained)
4. Model builders (self-contained)
5. Checkpoint management (self-contained)

### Phase 2 - Medium Effort:
1. FX trading helpers
2. RL training
3. Gate retraining
4. TCN regime training

### Phase 3 - Major Refactoring:
1. `_train_buddy_impl()` decomposition (4,124 lines)
2. `buddy()` extraction (1,541 lines)
3. `buddy_loop()` extraction (556 lines)

### Phase 4 - CLI Cleanup:
1. Reduce main() to pure dispatch
2. Move command implementations to `src/cli/`
