# Model Improvements Plan - XGBoost, RF, Ridge Fixes

**Date:** 2026-01-07  
**Status:** Transformer ✅ Excellent (78.3% accuracy) | XGBoost ❌ Broken | RF/Ridge ⏳ Not Evaluated

## 🚨 Critical Issues Identified

### 1. XGBoost Momentum - Data Leakage (SEVERITY: CRITICAL)

**Problem:**
- Validation MAE: 0.000835 (almost perfect)
- Test MAE: 0.342410 (catastrophic - 400x worse!)
- Root cause: Target leakage in `modular_data_loaders.py` line 1096

**Data Leakage Location:**
```python
# CURRENT CODE (WRONG) - Line 1093-1101
for i in range(momentum_window, n):
    window_returns = np.abs(returns[i-momentum_window:i])  # ❌ Includes current bar!
    raw_momentum_all[i] = np.mean(window_returns)
```

**The Issue:**
- At bar `i`, the momentum calculation looks at returns from `i-10` to `i-1`
- But `returns[i]` is calculated as `(close[i] - close[i-1]) / close[i-1]`
- The slice `[i-momentum_window:i]` in Python is **exclusive** of `i`, so it's actually OK
- **REAL PROBLEM**: The features themselves likely contain lookahead bias!

**Investigation Needed:**
1. Check if `momentum_data['X_train']` contains features calculated with lookahead
2. Verify feature engineering doesn't use future bars
3. Check if normalization was done on entire dataset (train+val+test together)

**Immediate Fix:**
```python
# Add this diagnostic after loading momentum_data:
console.print("\n[bold red]🔍 XGBoost Data Leakage Diagnostic[/bold red]")
console.print(f"Features: {momentum_data['feature_names'][:10]}")
console.print(f"y_train range: [{momentum_data['y_train'][:, 0].min():.6f}, {momentum_data['y_train'][:, 0].max():.6f}]")
console.print(f"y_val range: [{momentum_data['y_val'][:, 0].min():.6f}, {momentum_data['y_val'][:, 0].max():.6f}]")
console.print(f"y_test range: [{momentum_data['y_test'][:, 0].min():.6f}, {momentum_data['y_test'][:, 0].max():.6f}]")

# Check if features were normalized on entire dataset
X_train_mean = momentum_data['X_train'].mean(axis=0)
X_test_mean = momentum_data['X_test'].mean(axis=0)
mean_diff = np.abs(X_train_mean - X_test_mean).mean()
console.print(f"Mean feature diff (train vs test): {mean_diff:.6f}")
if mean_diff < 0.01:
    console.print("[red]⚠️ LEAKAGE: Features too similar across splits![/red]")
```

### 2. Random Forest - Not Evaluated Yet

**Status:** Cell 6.6 not run yet  
**Expected Issues:**
- Similar data leakage risk as XGBoost
- Drawdown calculation may use future bars
- Needs validation on test set

### 3. Ridge/ElasticNet - Not Evaluated Yet

**Status:** Cell 6.7 not run yet  
**Expected Issues:**
- Confidence calculation may have lookahead bias
- Feature normalization across splits

## 📋 Action Plan

### Phase 1: Immediate Diagnostics (DO FIRST)

1. **Add XGBoost leakage diagnostic cell BEFORE training**
```python
# Insert new cell BEFORE cell 6.5 (XGBoost training)
console.print("\n[bold red]🔍 Pre-Training Data Validation[/bold red]")

# Check for global normalization leakage
X_all = np.vstack([momentum_data['X_train'], momentum_data['X_val'], momentum_data['X_test']])
X_train_std = momentum_data['X_train'].std(axis=0)
X_all_std = X_all.std(axis=0)
std_ratio = X_train_std / (X_all_std + 1e-8)

console.print(f"Train-only std / All-data std ratio: {std_ratio.mean():.4f}")
if abs(std_ratio.mean() - 1.0) < 0.05:
    console.print("[red]❌ LEAKAGE: Features normalized on entire dataset![/red]")
else:
    console.print("[green]✅ Features normalized separately per split[/green]")

# Check target distributions
console.print(f"\nTarget (momentum) statistics:")
console.print(f"  Train: mean={momentum_data['y_train'][:, 0].mean():.6f}, std={momentum_data['y_train'][:, 0].std():.6f}")
console.print(f"  Val:   mean={momentum_data['y_val'][:, 0].mean():.6f}, std={momentum_data['y_val'][:, 0].std():.6f}")
console.print(f"  Test:  mean={momentum_data['y_test'][:, 0].mean():.6f}, std={momentum_data['y_test'][:, 0].std():.6f}")
```

2. **Update Quality Thresholds in Cell 6.9**
```python
QUALITY_THRESHOLDS = {
    'transformer_min_accuracy': 0.70,
    'transformer_min_balanced': 0.68,
    'transformer_max_drift': 0.05,
    'xgb_max_momentum_mae': 0.01,  # ❌ WAS: 0.005 (too strict) → NOW: 0.01
    'xgb_max_test_drift': 0.05,    # ✅ NEW: Test MAE shouldn't be >5% higher than val
    'rf_max_drawdown_pct': 5.0,
    'ridge_min_r2': -0.5,
    'ridge_max_mae': 15.0,
}

# Add XGBoost test drift check
if 'test_mom_mae' in dir() and test_mom_mae is not None:
    val_mom_mae = xgb_result['momentum_mae']
    xgb_drift = (test_mom_mae - val_mom_mae) / max(val_mom_mae, 1e-6)
    
    console.print(f"\n[bold cyan]XGBoost Test Drift Check:[/bold cyan]")
    console.print(f"  Val MAE: {val_mom_mae:.6f}")
    console.print(f"  Test MAE: {test_mom_mae:.6f}")
    console.print(f"  Drift: {xgb_drift*100:.1f}%")
    
    if xgb_drift > 400:  # 400x worse
        console.print(f"[bold red]  ❌ CATASTROPHIC DRIFT - DATA LEAKAGE CONFIRMED![/bold red]")
        quality_passed = False
    elif xgb_drift > 1.0:  # >100% worse
        console.print(f"[bold red]  ❌ SEVERE DRIFT - Likely data leakage[/bold red]")
        quality_passed = False
    elif xgb_drift > 0.5:
        console.print(f"[yellow]  ⚠️ Significant drift - needs investigation[/yellow]")
```

### Phase 2: Fix Data Loaders (AFTER diagnostics confirm leakage)

1. **Update modular_data_loaders.py - XGBoost Section**
```python
# Fix lines 1047-1130 to ensure NO lookahead bias
# Ensure all features are calculated STRICTLY from past data only
# Verify scaler is fit ONLY on train set, then transform val/test separately
```

2. **Add Temporal Validation**
```python
def validate_no_leakage(X_train, X_val, X_test, y_train, y_val, y_test):
    """Validate no data leakage across splits"""
    # Features should have different distributions across splits
    train_mean = X_train.mean(axis=0)
    val_mean = X_val.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    
    train_val_corr = np.corrcoef(train_mean, val_mean)[0, 1]
    train_test_corr = np.corrcoef(train_mean, test_mean)[0, 1]
    
    if train_val_corr > 0.99 or train_test_corr > 0.99:
        raise ValueError("Leakage detected: Feature means too similar across splits")
    
    return True
```

### Phase 3: Enhanced Quality Validation

**Update Cell 6.10 (Holdout Test) to add leakage detection:**

```python
# After transformer test, add XGBoost leakage check
if test_mom_mae > 0.01 or (test_mom_mae / val_mom_mae) > 5.0:
    console.print("\n[bold red]🚨 XGBoost CRITICAL FAILURE DETECTED 🚨[/bold red]")
    console.print(f"Validation MAE: {val_mom_mae:.6f} (suspiciously low)")
    console.print(f"Test MAE: {test_mom_mae:.6f} (catastrophically high)")
    console.print(f"Ratio: {test_mom_mae/val_mom_mae:.1f}x worse on test set")
    console.print("\n[yellow]DIAGNOSIS: Data leakage in XGBoost training[/yellow]")
    console.print("ACTIONS REQUIRED:")
    console.print("  1. Check feature engineering for lookahead bias")
    console.print("  2. Verify scaler fit ONLY on train set")
    console.print("  3. Confirm momentum calculation uses past bars only")
    console.print("  4. Re-run with fixed data loader")
    
    holdout_results['xgboost_leakage_detected'] = True
    holdout_results['production_ready'] = False
```

## 🔧 Specific Fixes

### Fix 1: XGBoost Data Loader (modular_data_loaders.py)

**Problem:** Features may be normalized on entire dataset  
**Fix:** Ensure scaler is fit ONLY on train, transform val/test separately

```python
# BEFORE (WRONG):
X = df[features].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)  # ❌ All data together
train_idx, val_idx, test_idx = temporal_split(len(X), *split)

# AFTER (CORRECT):
X = df[features].values.astype(np.float32)
train_idx, val_idx, test_idx = temporal_split(len(X), *split)

# Fit scaler ONLY on train
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X[train_idx])  # ✅ Fit on train only
X_val = scaler.transform(X[val_idx])          # ✅ Transform val
X_test = scaler.transform(X[test_idx])        # ✅ Transform test

return {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'scaler': scaler,  # Save for inference
    ...
}
```

### Fix 2: Update Quality Thresholds

**Cell 6.9 updates:**

```python
QUALITY_THRESHOLDS = {
    'transformer_min_accuracy': 0.70,
    'transformer_min_balanced': 0.68,
    'transformer_max_drift': 0.05,
    'xgb_max_momentum_mae': 0.01,      # Realistic threshold
    'xgb_max_val_test_ratio': 5.0,    # NEW: Test shouldn't be >5x worse than val
    'rf_max_drawdown_pct': 5.0,
    'rf_max_test_drift': 0.5,          # NEW: Test shouldn't be >50% worse
    'ridge_min_r2': -0.5,
    'ridge_max_mae': 15.0,
    'ridge_max_test_drift': 0.3,       # NEW: Test shouldn't be >30% worse
}
```

## 📊 Expected Results After Fixes

### XGBoost (Fixed)
- Validation MAE: 0.001-0.003 (realistic for normalized momentum)
- Test MAE: 0.001-0.005 (similar to validation)
- Drift: <2x (test vs val)

### Random Forest
- Drawdown MAE: 1-3% (realistic for risk prediction)
- Test/Val ratio: <1.5x

### Ridge/ElasticNet
- Confidence MAE: 10-20 (realistic for 0-100 confidence scores)
- R²: 0.0-0.3 (anything >0 is useful)
- Test/Val ratio: <1.3x

## 🎯 Success Criteria

Models are production-ready when:
1. ✅ Transformer: 78.3% accuracy (already achieved)
2. ✅ XGBoost: Test MAE < 0.01 AND test/val ratio < 3x
3. ✅ RF: Drawdown MAE < 5% AND test/val ratio < 2x
4. ✅ Ridge: MAE < 15 AND test/val ratio < 1.5x
5. ✅ All models: No leakage detected in shuffle test

## 🚀 Next Steps

1. **Run diagnostics** (insert new cell before XGBoost training)
2. **Confirm leakage** (check feature normalization)
3. **Fix data loaders** (update modular_data_loaders.py)
4. **Retrain all models** (cells 6.5, 6.6, 6.7)
5. **Validate on holdout** (cell 6.10)
6. **Update quality thresholds** (cell 6.9)
7. **Deploy** (only if all checks pass)
