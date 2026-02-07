# Model Training Troubleshooting Guide

## Common Training Issues and Solutions

### Issue 1: Prediction Collapse

**Symptom**: Model predicts >90% one class (all UP or all DOWN)

**Example Log**:
```
🚨 SEVERE PREDICTION COLLAPSE at epoch 82: 92.0% UP, 8.0% DOWN (all UP)
🔧 COLLAPSE RECOVERY attempt 1/5
  → Strategy 1: Restoring best balanced weights (balance=0.850)
❌ STOPPING: Prediction collapse persists after 5 recovery attempts
```

**Root Causes**:
1. **Class Imbalance**: Training data heavily skewed (e.g., 80% UP, 20% DOWN)
2. **Poor Feature Quality**: Features don't discriminate between classes
3. **Learning Rate Too High**: Model converges too quickly to wrong solution
4. **Insufficient Regularization**: Model overfits to majority class

**Solutions**:

✅ **Check Class Distribution**:
```bash
# After training starts, check the logs for:
Direction labels: LONG=4523, SHORT=4477 (50.3% LONG, 49.7% SHORT)  ✓ Balanced
Direction labels: LONG=7234, SHORT=1766 (80.4% LONG, 19.6% SHORT)  ⚠️ Imbalanced
```

✅ **Adjust Direction Threshold** (config):
```yaml
# In config/config_improved_H1.yaml
direction_threshold: 0.003  # Lower = more labeled data, but noisier
direction_threshold: 0.005  # Higher = cleaner labels, but less data
```

✅ **Enable Sample Weighting** (already enabled):
The trainer automatically applies sample weights to balance class contributions.

✅ **Reduce Learning Rate**:
```yaml
# In config/config_improved_H1.yaml
optimizer:
  learning_rate: 0.0003  # Default
  learning_rate: 0.0001  # Try lower if collapsing
```

✅ **Increase Regularization**:
```yaml
# In config/config_improved_H1.yaml
transformer:
  dropout: 0.2   # Current
  dropout: 0.3   # Higher = more regularization
```

✅ **Use Stronger Focal Loss**:
```yaml
# In config/config_improved_H1.yaml
focal_gamma: 2.0  # Current
focal_gamma: 3.0  # Higher = more focus on hard examples
```

---

### Issue 2: Low Validation Accuracy (<55%)

**Symptom**: Model trains but achieves poor validation accuracy

**Example Log**:
```
📊 Final validation accuracy: 52.3%
   Balanced accuracy: 51.8% (up=53%, down=50%)
```

**Root Causes**:
1. **Noisy Labels**: Direction threshold too low, creates random labels
2. **Insufficient Training Data**: Not enough candles for pattern learning
3. **Model Too Simple**: Can't capture complex patterns
4. **Model Too Complex**: Overfitting to training noise

**Solutions**:

✅ **Increase Direction Threshold** (cleaner labels):
```yaml
direction_threshold: 0.003  # Current - allows small moves
direction_threshold: 0.005  # Higher - only clear directional moves
```

✅ **Fetch More Data**:
```bash
# Increase candle count from default
./bin/Buddy train -i EUR_USD -c 20000  # 20k candles instead of 15k
```

✅ **Adjust Model Capacity**:
```yaml
# Too simple? Increase:
transformer:
  d_model: 32      # Current
  d_model: 48      # Higher capacity
  num_layers: 2    # Current
  num_layers: 3    # Deeper model

# Too complex? Decrease:
transformer:
  d_model: 32      # Current
  d_model: 24      # Lower capacity
```

✅ **Check Sequence Length**:
```yaml
# H1 timeframe
sequence_length: 60  # 60 hours = 2.5 days
```

---

### Issue 3: Training Stops Early

**Symptom**: Training stops before reaching max epochs

**Example Log**:
```
Early stopping triggered: no improvement for 40 epochs
Final epoch: 89/200
```

**Root Causes**:
1. **Early Stopping Too Aggressive**: Patience too low
2. **Learning Rate Decay**: LR dropped too low, no more learning
3. **Model Converged**: Actually finished training (not an issue)

**Solutions**:

✅ **Increase Early Stopping Patience**:
```yaml
# In config/config_improved_H1.yaml
training:
  early_stopping_patience: 40  # Current
  early_stopping_patience: 60  # More patience
```

✅ **Adjust Minimum Epochs**:
```yaml
training:
  min_epochs: 60   # Current - early stopping disabled before epoch 60
  min_epochs: 100  # Higher - allow more exploration
```

✅ **Check if Legitimate**:
```
# Look for:
Best validation accuracy: 64.2% at epoch 87
```
If accuracy peaked and hasn't improved, early stopping is working correctly.

---

### Issue 4: Model Not Saving

**Symptom**: Training completes but no models found

**Expected Location**:
```
trained_data/models/EUR_USD/
  ├── transformer_direction.keras
  ├── transformer_direction.meta.pkl
  ├── xgb_momentum.pkl
  ├── ridge_confidence.pkl
  ├── rf_risk.pkl
  └── modular_ensemble.meta.json
```

**Root Causes**:
1. **Training Failed**: Errors during training prevented save
2. **Path Issues**: Incorrect save directory
3. **Disk Space**: Out of storage

**Solutions**:

✅ **Check Training Completion**:
```bash
# Look for success message:
✅ Transformer saved to trained_data/models/EUR_USD/transformer_direction.keras
```

✅ **Verify Directory Exists**:
```bash
ls -la trained_data/models/EUR_USD/
```

✅ **Check Disk Space**:
```bash
df -h
```

✅ **Manual Verification**:
```python
from pathlib import Path
import pickle

# Check if model exists
model_path = Path("trained_data/models/EUR_USD/transformer_direction.keras")
print(f"Model exists: {model_path.exists()}")

# Check metadata
meta_path = model_path.with_suffix('.meta.pkl')
if meta_path.exists():
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    print(f"Validation accuracy: {meta['metrics']['val_accuracy']:.4f}")
```

---

### Issue 5: Training Very Slow

**Symptom**: Each epoch takes >60 seconds on H1 data

**Root Causes**:
1. **Metal GPU Not Used**: On M1/M2/M3 Macs
2. **Batch Size Too Small**: More iterations needed
3. **Data Pipeline Slow**: I/O bottleneck

**Solutions**:

✅ **Verify GPU Usage** (M1/M2/M3):
```bash
# Check for Metal device:
# Should see in logs:
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

✅ **Increase Batch Size**:
```yaml
batch_size: 64   # Current
batch_size: 128  # Double - faster training, more memory
```

✅ **Disable Debugging**:
```yaml
training:
  run_eagerly: false  # Must be false for speed
  jit_compile: false  # Metal doesn't support JIT well
```

✅ **Check recurrent_dropout**:
```yaml
model:
  recurrent_dropout: 0.0  # MUST be 0 on Metal (10x slowdown if not)
```

---

### Issue 6: "sklearn version mismatch" Warnings

**Symptom**: Warnings about sklearn model compatibility

**Example Log**:
```
⚠️ Ridge gate BYPASSED (permissive mode): sklearn version mismatch
```

**Root Cause**: Models trained with different sklearn version

**Solutions**:

✅ **Retrain Gate Models**:
```bash
# Retrain just the sklearn models (Ridge, XGBoost, RF)
python main.py retrain-gates
```

✅ **Check Permissive Mode** (already enabled):
This allows graceful degradation - gates bypass if models can't load.

✅ **Upgrade sklearn** (if needed):
```bash
pip install --upgrade scikit-learn
```

---

### Issue 7: NaN or Inf in Training

**Symptom**: NaN values appear during training

**Example Log**:
```
WARNING: NaN detected in predictions
Loss: nan, Accuracy: 0.0
```

**Root Causes**:
1. **Exploding Gradients**: Learning rate too high
2. **Bad Data**: NaN/Inf in features
3. **Numerical Instability**: Division by zero

**Solutions**:

✅ **Check for Bad Data**:
```python
# Already handled in modular_trainers.py:
df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
```

✅ **Gradient Clipping** (already enabled):
```python
optimizer = keras.optimizers.Adam(clipnorm=1.0)
```

✅ **Reduce Learning Rate**:
```yaml
optimizer:
  learning_rate: 0.0001  # Lower if NaN appears
```

---

## Diagnostic Commands

### Check Training Status
```bash
# View recent training logs
tail -f trained_data/logs/buddy_training.log

# Check saved models
ls -lh trained_data/models/EUR_USD/
```

### Inspect Model Metadata
```python
from pathlib import Path
import pickle

meta_path = Path("trained_data/models/EUR_USD/transformer_direction.meta.pkl")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

print(f"Validation Accuracy: {meta['metrics']['val_accuracy']:.4f}")
print(f"Balanced Accuracy: {meta['metrics']['val_balanced_accuracy']:.4f}")
print(f"UP Accuracy: {meta['metrics']['val_up_accuracy']:.4f}")
print(f"DOWN Accuracy: {meta['metrics']['val_down_accuracy']:.4f}")
print(f"Epochs Trained: {meta['metrics']['epochs_trained']}")
```

### Test Model Loading
```python
from src.training.modular_trainers import TransformerDirectionTrainer

trainer = TransformerDirectionTrainer()
trainer.load("trained_data/models/EUR_USD/transformer_direction.keras", "EUR_USD")

print(f"Model loaded successfully: {trainer.is_trained}")
print(f"Features: {trainer.feature_names[:5]}...")
```

---

## Getting Help

1. **Check Logs**: Look for errors/warnings in terminal output
2. **Review Metrics**: Inspect `val_accuracy`, `balanced_accuracy`
3. **Compare Pairs**: Does EUR_USD train better than GBP_USD?
4. **Try Different Timeframe**: H1 vs M1 may have different characteristics
5. **Consult Documentation**: 
   - [Copilot Instructions](../.github/copilot-instructions.md)
   - [Prediction Collapse System](./PREDICTION_COLLAPSE_SYSTEM.md)

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Prediction Collapse | Reduce LR, increase dropout, check class balance |
| Low Accuracy (<55%) | Increase direction_threshold, fetch more data |
| Training Stops Early | Increase patience, check if legitimate |
| Slow Training | Verify Metal GPU, increase batch size |
| NaN/Inf | Reduce LR, check data quality |
| Model Not Saving | Check logs for errors, verify disk space |
| sklearn Warnings | Run `retrain-gates` command |
