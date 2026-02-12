# Buddy Train Terminal Output Implementation

## Overview
This document describes the implementation of structured terminal output for the `buddy train` command as specified in the problem statement.

## Implementation Status

### ✅ Completed Phases

#### Phase 1: Configuration Panel
**Location:** `cli/training.py` (lines ~327-367)

Displays at the start of training with:
- Instrument name (e.g., EUR_USD)
- Granularity (e.g., H1)
- Data source (OANDA live fetch or CSV file)
- Model type, epochs, batch size, and learning rate

**Example Output:**
```
⚙️  Configuration
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Training Configuration                   ┃
┃                                          ┃
┃ Instrument: EUR_USD                      ┃
┃ Granularity: H1                          ┃
┃ Data Source: OANDA live fetch (15,000...)┃
┃ Model Type: ensemble                     ┃
┃ Epochs: 200  Batch Size: 64  LR: 0.0003 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

#### Phase 2: Data Fetching Progress
**Location:** `cli/io_utils.py` `_oanda_fetch_to_csv()` function

Features:
- Progress spinner while downloading candles from OANDA
- Shows current page/total pages for multi-page downloads
- Displays candle count progress (e.g., "5,000/15,000 candles")
- Transient output (disappears when complete)

**Example Output:**
```
⠋ Downloading EUR_USD H1: page 2/3 (10,000/15,000 candles)...
```

#### Phase 3: Feature Engineering
**Location:** `cli/training.py` (lines ~475-508)

Enhanced panel showing:
- Smoothing status
- Median window setting (or "None" if not used)
- **Exact DataFrame shape**: "12,847 rows × 127 columns" (highlighted in green)
- Elapsed time in seconds

**Example Output:**
```
⚙️  Feature Engineering
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Feature Engineering Complete            ┃
┃                                          ┃
┃ Smoothing: True  Median Window: None    ┃
┃ Output: 12,847 rows × 127 columns       ┃
┃ Elapsed Time: 3.45s                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

#### Phase 4: Direction/Regime Labels
**Location:** `cli/training.py` (lines ~604-688)

**Already implemented** - shows:
- Class balance statistics: "train=48.2% up / val=51.8% up"
- Tier-2 TP/SL simulation stats when enabled:
  ```
  ✓ Tier-2 Labels: USD_JPY | SL=15 TP=30 | horizon=288 stride=5 | 2,400 samples in 3.2s
  ```

#### Phase 5: Model Training (4 Steps)
**Location:** `cli/training.py` (lines ~1468-1661)

**Already implemented** - Four distinct panels:

**Step 1/4:** Transformer/TCN Direction Predictor
- Panel header with model type
- Feature descriptions
- Keras training progress bar (handled by Keras/TF)
- Completion message with accuracy and EarlyStopping info

**Step 2/4:** XGBoost Momentum Analyzer
- Panel showing momentum analysis task
- Progress output (handled by XGBoost trainer)
- Completion with momentum_mae and acceleration accuracy

**Step 3/4:** Random Forest Risk Assessor
- Panel showing risk assessment task
- Forest construction progress
- Completion with drawdown MAE and streak probability MAE

**Step 4/4:** Ridge/ElasticNet Confidence Scorer
- Panel showing confidence scoring task
- Fast completion (usually instant)
- Shows R², MAE, alpha, L1 ratio, and sparsity

#### Phase 6: Validation & Calibration
**Location:** `cli/training.py` (lines ~1788-2004)

**Already implemented** - includes:
- Performance metrics table with accuracy, F1 scores
- Bootstrap confidence intervals (if enabled)
- Walk-forward cross-validation results
- Tier-2 calibration output (temperature scaling)
- Optional RL Position Sizer training panel

#### Phase 7: Training Complete Summary
**Location:** `cli/training.py` (lines ~2213-2258)

Comprehensive summary panel showing:
- List of saved model files:
  - transformer_direction.keras (or transformer_regime.keras)
  - xgb_momentum.pkl
  - rf_risk.pkl
  - ridge_confidence.pkl
  - modular_ensemble.meta.json
- Model save location
- Training instrument
- Total training time (formatted as seconds or minutes)
- Next steps with example commands

**Example Output:**
```
🎉 Training Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Training Complete ✓                      ┃
┃                                          ┃
┃ Saved Models:                            ┃
┃   • transformer_direction.keras          ┃
┃   • xgb_momentum.pkl                     ┃
┃   • rf_risk.pkl                          ┃
┃   • ridge_confidence.pkl                 ┃
┃   • modular_ensemble.meta.json           ┃
┃                                          ┃
┃ Location: trained_data/models            ┃
┃ Instrument: EUR_USD                      ┃
┃ Total Time: 12.5m                        ┃
┃                                          ┃
┃ Next Steps:                              ┃
┃   Run buddy EUR_USD to test inference    ┃
┃   Or buddy EUR_USD -x to execute trades  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Files Modified

### 1. `cli/io_utils.py`
- Added `Progress`, `SpinnerColumn`, `TextColumn` imports from Rich
- Modified `_oanda_fetch_to_csv()` to wrap OANDA API calls in a Progress context
- Added progress updates for multi-page downloads
- Progress is transient (disappears when complete)

### 2. `cli/training.py`
- Added Phase 1 Configuration Panel at start of training
- Enhanced Phase 3 Feature Engineering panel with better formatting
- Added Phase 7 Training Complete Summary panel
- Added section comments to delineate the 7 phases

## Testing Recommendations

To verify the implementation works correctly:

```bash
# Test with OANDA live fetch (recommended)
./bin/Buddy train -I EUR_USD --oanda-live

# Test with existing CSV
./bin/Buddy train -I EUR_USD --csv market_data/EUR_USD_H1.csv

# Test ensemble mode (default)
python main.py train-buddy --instrument EUR_USD --oanda-live

# Monitor output for:
# 1. Configuration panel at start
# 2. Progress spinner during data fetch
# 3. Feature engineering panel with exact row×column count
# 4. Direction label statistics
# 5. Four model training panels (Steps 1-4)
# 6. Validation metrics table
# 7. Training complete summary with all model files listed
```

## Notes

- Phases 4, 5, and 6 were already well-implemented in the modular trainers
- The Keras progress bar in Phase 5 Step 1 is handled by TensorFlow/Keras callbacks
- XGBoost, Random Forest, and Ridge trainers have their own output formatting
- The implementation maintains backward compatibility with existing training flows
- All changes are minimal and surgical, preserving existing functionality

## Example Complete Training Flow

When running `buddy train -I EUR_USD --oanda-live`, users will see:

1. ⚙️  Configuration panel
2. ⠋ OANDA download spinner
3. 📄 CSV Data panel (from existing code)
4. ⚙️  Feature Engineering panel with exact dimensions
5. ✓ Direction Labels stats
6. ✓ Tier-2 Labels simulation stats
7. 📊 Dataset Configuration panel
8. Step 1/4: Transformer training with Keras progress
9. Step 2/4: XGBoost training
10. Step 3/4: Random Forest training
11. Step 4/4: Ridge/ElasticNet training
12. 📊 Model Performance table
13. 💾 Saved Artifacts table
14. 🏢 Enterprise validation (if enabled)
15. 🎉 Training Summary panel

This creates a professional, structured output that clearly communicates progress through all training stages.
