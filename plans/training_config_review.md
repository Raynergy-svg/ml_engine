# Training Configuration Review and Adjustment Report

**Date:** 2026-01-27  
**Task:** Comprehensive review and adjustment of training configuration

---

## Executive Summary

All requested configurations have been verified and are correctly implemented in the codebase:

✅ **Sample weight cap**: Enforced at 10x base weight  
✅ **Weight decay**: Reduced to 0.001  
✅ **Warm-start LR factor**: Increased to 0.1  
✅ **Direction threshold**: Verified at 0.003 (0.3%)  
✅ **Model collapse detection**: Implemented with PredictionCollapseCallback  

**No adjustments are required** - all configurations are already at the target values.

---

## 1. Class Distribution Analysis

### Script Location
- **File:** [`check_class_dist.py`](check_class_dist.py:1)
- **Function:** Analyzes training data to compute UP/DOWN class distribution and imbalance ratio

### Implementation
```python
# Filter to clear labels only (where weight > 0)
clear_mask = (w_train > 0) if w_train is not None else np.ones(len(y_train), dtype=bool)
y_clear = y_train[clear_mask]

up_count = (y_clear == 1).sum()
down_count = (y_clear == 0).sum()
print(f'Imbalance ratio: {max(up_count, down_count) / min(up_count, down_count):.2f}x')
```

### Key Findings
- The script filters samples to only include clear labels (weight > 0)
- Computes exact UP and DOWN counts
- Reports imbalance ratio as `max(up, down) / min(up, down)`
- This provides the foundation for sample weight calculation

---

## 2. Sample Weight Cap Verification (10x Maximum)

### Implementation Location
- **File:** [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2997)
- **Lines:** 2996-3012

### Code Verification
```python
# FIX: Cap maximum weight to prevent gradient instability with extreme imbalance
# Without cap, 124:1 imbalance would give minority class 62.5x weight, causing oscillation
MAX_WEIGHT_MULTIPLIER = 10.0  # Cap at 10x weight
up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER) if n_up > 0 else 1.0
down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER) if n_down > 0 else 1.0
```

### Verification Result
✅ **CONFIRMED:** Sample weight cap is correctly enforced at 10x

**Why this is important:**
- Prevents gradient instability with extreme class imbalance
- Without the cap, a 124:1 imbalance would give minority class 62.5x weight
- The 10x cap balances minority class learning with training stability

---

## 3. Hyperparameter Verification

### 3.1 Weight Decay (0.001)

#### Config Location
- **File:** [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml:139)
- **Line:** 139

#### Configuration Value
```yaml
weight_decay: 0.001  # Reduced from 0.05 to 0.001
```

#### Verification Result
✅ **CONFIRMED:** Weight decay is at target value of 0.001

**Impact:**
- 50x reduction from 0.05 to 0.001
- Allows model to learn more freely without excessive regularization
- Prevents output suppression that was causing model collapse

---

### 3.2 Warm-Start Learning Rate Factor (0.1)

#### Config Location
- **File:** [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml:159)
- **Line:** 159

#### Configuration Value
```yaml
warm_start_lr_factor: 0.1  # Increased from 0.01 to 0.1
```

#### Default Value in Code
- **File:** [`src/training/modular_trainers.py`](src/training/modular_trainers.py:127)
- **Line:** 127
```python
warm_start_lr_factor: float = 0.01  # Default value
```

#### Verification Result
✅ **CONFIRMED:** Config overrides default to 0.1 (10x higher than default)

**Impact:**
- 10x increase from 0.01 to 0.1
- Allows faster adaptation during warm-start training
- Reduces warm-start LR from 100x to 10x reduction factor

---

## 4. Model Collapse Detection Monitoring

### Implementation Location
- **File:** [`src/training/modular_trainers.py`](src/training/modular_trainers.py:3343)
- **Lines:** 3343-3373

### PredictionCollapseCallback Class

#### Initialization
```python
class PredictionCollapseCallback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, check_every=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.check_every = check_every
        self.collapse_warned = False
```

#### Detection Logic
```python
def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) % self.check_every != 0:
        return
    
    preds = self.model.predict(self.X_val, verbose=0)
    pred_classes = (preds > 0.5).astype(float).flatten()
    
    pred_up_pct = pred_classes.mean() * 100
    pred_down_pct = 100 - pred_up_pct
    
    # Check for collapse (>95% same prediction)
    if pred_up_pct > 95 or pred_down_pct > 95:
        if not self.collapse_warned:
            dominant = "UP" if pred_up_pct > 95 else "DOWN"
            logger.warning(f"⚠️ PREDICTION COLLAPSE at epoch {epoch+1}: "
                          f"Model predicts {pred_up_pct:.1f}% UP, {pred_down_pct:.1f}% DOWN "
                          f"(all {dominant})")
            self.collapse_warned = True
```

#### Monitoring Frequency
- **Check interval:** Every 5 epochs (`check_every=5`)
- **Collapse threshold:** >95% predictions in one class
- **Warning format:** Clear, actionable warning with epoch number

#### Integration in Training
```python
# Line 3442 in modular_trainers.py
callbacks = [
    # ... other callbacks ...
    
    # Prediction collapse detection - warns if model predicts mostly one class
    PredictionCollapseCallback(X_val_filtered, y_val_filtered, check_every=5),
]
```

### Verification Result
✅ **CONFIRMED:** Model collapse detection is fully implemented and active

**What it monitors:**
1. **Prediction distribution:** Percentage of UP vs DOWN predictions
2. **Collapse detection:** Triggers when >95% of predictions are one class
3. **Periodic logging:** Reports distribution every 10 epochs when healthy

**Warning format:**
```
⚠️ PREDICTION COLLAPSE at epoch 15: Model predicts 97.3% UP, 2.7% DOWN (all UP)
```

---

## 5. Direction Threshold Verification (Label Generation)

### Config Location
- **File:** [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml:67)
- **Line:** 67

### Configuration Value
```yaml
direction_threshold: 0.003  # 0.3% minimum move for clear signal
```

### Context
- **Lookahead period:** 24 hours (line 66)
```yaml
direction_lookahead: 24  # Line 66 - 24 hours lookahead
```

### How It's Used
The threshold determines when a price movement is significant enough to assign a clear direction label:
- Price change > 0.3% → UP (1)
- Price change < -0.3% → DOWN (0)
- Between -0.3% and +0.3% → No clear label (weight = 0)

### Verification Result
✅ **CONFIRMED:** Direction threshold is correctly set to 0.003 (0.3%)

**Why 0.3% is appropriate:**
- Filters out noise from small fluctuations
- Captures meaningful directional moves
- Balances signal clarity with sufficient data for clear labels

---

## 6. Additional Related Configurations

### 6.1 Inverse Frequency Sample Weights
**Location:** [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2998-3006)

```python
# Inverse frequency weighting: weight = total / (2 * class_count)
# This ensures equal contribution from each class to the loss
total = n_up + n_down

# FIX: Cap maximum weight to prevent gradient instability with extreme imbalance
MAX_WEIGHT_MULTIPLIER = 10.0  # Cap at 10x weight
up_weight = min(total / (2 * n_up), MAX_WEIGHT_MULTIPLIER) if n_up > 0 else 1.0
down_weight = min(total / (2 * n_down), MAX_WEIGHT_MULTIPLIER) if n_down > 0 else 1.0

# Apply inverse frequency weights with cap (replacing old 1.5x boost)
sample_weights[y_train_filtered == 1] *= up_weight
sample_weights[y_train_filtered == 0] *= down_weight

# Re-normalize to mean=1
sample_weights = sample_weights / sample_weights.mean()
```

### 6.2 Regime-Aware Weighting
**Location:** [`src/training/modular_trainers.py`](src/training/modular_trainers.py:2943-2981)

The system also implements volatility regime-aware weighting:
- **LOW_VOL regime:** Lower volatility samples
- **MED_VOL regime:** Medium volatility samples
- **HIGH_VOL regime:** High volatility samples

Each regime gets balanced class weights independently to prevent regime-specific biases.

---

## 7. System Log Monitoring Strategy

### What to Monitor During Training

#### Model Collapse Warnings
Watch for this pattern in logs:
```
⚠️ PREDICTION COLLAPSE at epoch X: Model predicts Y.Y% UP, Z.Z% DOWN (all CLASS)
```

#### Healthy Training Indicators
```
📊 Prediction distribution at epoch 10: 52.3% UP, 47.7% DOWN
```

#### Class Distribution Logging
The trainer logs class distribution at the start of training:
```
Class distribution: train=52.1% up (imbalance=1.08x), val=51.8% up (imbalance=1.07x)
```

### Log Locations
- **Console output:** Real-time during training
- **Log files:** Standard Python logging (check your logging configuration)
- **Rich output:** Color-coded epoch display with RichEpochCallback

---

## 8. Configuration Summary

| Configuration | Target Value | Actual Value | Status | Location |
|--------------|---------------|---------------|--------|----------|
| Sample weight cap | 10x | 10x | ✅ | modular_trainers.py:2997 |
| Weight decay | 0.001 | 0.001 | ✅ | config_improved_H1.yaml:139 |
| Warm-start LR factor | 0.1 | 0.1 | ✅ | config_improved_H1.yaml:159 |
| Direction threshold | 0.003 | 0.003 | ✅ | config_improved_H1.yaml:67 |
| Model collapse detection | Active | Active | ✅ | modular_trainers.py:3343 |

---

## 9. Recommendations

### No Adjustments Required
All requested configurations are already correctly implemented and at target values.

### Best Practices for Training

1. **Monitor logs vigilantly** for collapse warnings
2. **Check class distribution** using `check_class_dist.py` before training
3. **Review prediction distribution** in logs after every 10 epochs
4. **If collapse detected:**
   - Check if sample weights are being applied correctly
   - Verify direction_threshold is appropriate for your data
   - Consider adjusting focal_loss parameters if using Focal Loss

### Troubleshooting Model Collapse

If you see collapse warnings:

1. **Verify data quality:**
   ```bash
   python check_class_dist.py
   ```

2. **Check sample weights:**
   - Look for log: `🎯 Inverse frequency sample weights (capped at 10.0x): UP=X.XXx, DOWN=Y.YYx`
   - Ensure weights are not extreme (>10x indicates cap is working)

3. **Review regularization:**
   - Ensure weight_decay is 0.001 (not 0.05)
   - Check dropout rates (should be 0.2-0.4 for transformer)

4. **Examine prediction distribution:**
   - Look for: `📊 Final validation prediction distribution:`
   - Healthy: Mean near 0.5, std > 0.1
   - Collapsed: Mean near 0 or 1, std near 0

---

## 10. Conclusion

✅ **All requested configurations verified and confirmed correct**

The training configuration is properly set up with:
- Sample weight cap at 10x to prevent gradient instability
- Reduced weight decay (0.001) to allow proper learning
- Increased warm-start LR factor (0.1) for faster adaptation
- Direction threshold at 0.003 (0.3%) for clear signal generation
- Active model collapse detection with PredictionCollapseCallback

**No code changes are required.** The system is ready for training with vigilant monitoring for model collapse warnings.

---

## Appendix: Key File References

| File | Purpose | Key Lines |
|------|----------|------------|
| [`check_class_dist.py`](check_class_dist.py:1) | Class distribution analysis | All |
| [`config/config_improved_H1.yaml`](config/config_improved_H1.yaml:1) | Main training configuration | 67, 139, 159 |
| [`src/training/modular_trainers.py`](src/training/modular_trainers.py:1) | Training implementation | 127, 2997, 3343 |
