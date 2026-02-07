# Prediction Collapse Detection & Recovery System

## Overview

The Prediction Collapse Detection & Recovery System is a critical component of the ML Engine's training pipeline that monitors and corrects model behavior when it begins predicting predominantly one class (e.g., always UP or always DOWN).

## Problem Statement

During training, neural networks can sometimes "collapse" to predicting a single class for all inputs. This occurs when:
- The model finds a local minimum where predicting one class minimizes loss
- Class imbalance leads the model to favor the majority class
- Gradient flow issues cause the model to converge to a degenerate solution

Example collapse scenario:
```
Epoch 50: 55% UP, 45% DOWN  ✓ Balanced
Epoch 60: 65% UP, 35% DOWN  ⚠️ Slight bias
Epoch 70: 82% UP, 18% DOWN  ⚡ Early warning
Epoch 80: 91% UP, 9% DOWN   🚨 COLLAPSE
```

## Graduated Detection Levels

The system uses three graduated warning levels to detect collapse early:

### 1. Early Warning (80-85%)
**Threshold**: When predictions exceed 80% but remain below 85% for one class

**Action**: Informational log only, no intervention
```
⚡ Early imbalance warning at epoch 72: 82.0% UP, 18.0% DOWN (trending toward UP)
```

**Purpose**: Alert developers that the model is trending toward collapse

### 2. Moderate Imbalance (85-90%)
**Threshold**: When predictions exceed 85% but remain below 90% for one class

**Action**: Warning log, monitoring intensifies
```
⚠️ MODERATE IMBALANCE at epoch 78: 87.0% UP, 13.0% DOWN (biased toward UP)
```

**Purpose**: Signal that intervention may be needed soon

### 3. Severe Collapse (>90%)
**Threshold**: When predictions exceed 90% for one class

**Action**: Immediate recovery intervention
```
🚨 SEVERE PREDICTION COLLAPSE at epoch 82: 92.0% UP, 8.0% DOWN (all UP)
🔧 COLLAPSE RECOVERY attempt 1/5
```

**Purpose**: Trigger automatic recovery to restore model balance

## Recovery Strategies

The system implements **5 progressive recovery strategies**, each more aggressive than the last:

### Strategy 1-2: Restore Best Balanced Weights
**Attempts**: 1-2  
**Method**: Restore model to previously saved checkpoint with best prediction balance  
**LR Adjustment**: 
- Attempt 1: LR × 0.5
- Attempt 2: LR × 0.3

```python
# Restore weights from best balanced state
model.set_weights(best_weights)
new_lr = initial_lr * 0.5  # or 0.3 for attempt 2
```

**Rationale**: If we've seen better balance before, return to that state with reduced learning rate to avoid repeating the collapse

### Strategy 3: Perturb Output Layer
**Attempt**: 3  
**Method**: Add stronger noise to output layer weights and reset bias  
**LR Adjustment**: LR × 0.4

```python
# Add noise to output weights
weights[-2] = weights[-2] + np.random.normal(0, 0.15, weights[-2].shape)

# Reset bias to log odds of class ratio
p_up = np.mean(y_train)
bias_init = np.log(p_up / (1.0 - p_up))
weights[-1] = np.array([bias_init])
```

**Rationale**: The output layer has likely converged to a degenerate solution. Perturb it while keeping earlier layers intact.

### Strategy 4: Perturb All Layers
**Attempt**: 4  
**Method**: Add noise to all trainable weights  
**LR Adjustment**: LR × 0.2 (most aggressive)

```python
# Perturb all layers except output
for i in range(len(weights) - 2):
    weights[i] = weights[i] + np.random.normal(0, 0.05, weights[i].shape)

# More aggressive perturbation for output layer
weights[-2] = weights[-2] + np.random.normal(0, 0.2, weights[-2].shape)
weights[-1] = np.random.normal(0, 0.1, weights[-1].shape)
```

**Rationale**: The entire network has converged poorly. Add noise throughout to escape the local minimum.

### Strategy 5: Complete Output Reinitialization
**Attempt**: 5 (Last Resort)  
**Method**: Completely reinitialize the output layer  
**LR Adjustment**: LR × 0.6 (higher to allow relearning)

```python
# Complete reinitialization
weights[-2] = np.random.normal(0, 0.3, weights[-2].shape)
weights[-1] = np.zeros_like(weights[-1])
```

**Rationale**: As a last resort, reset the output layer entirely and let it relearn from the feature representations in earlier layers.

## Balance Metric

The system tracks a **balance metric** to quantify prediction distribution:

```python
balance = min(pred_up_pct, pred_down_pct) / 50.0
```

**Scale**: 0.0 to 1.0
- **1.0**: Perfect balance (50% UP, 50% DOWN)
- **0.8**: Moderate imbalance (60/40 or 40/60)
- **0.4**: High imbalance (80/20 or 20/80)
- **0.1**: Severe collapse (95/5 or 5/95)

**Checkpoint Threshold**: Weights are saved when `balance > 0.25` and improving

## Prediction History Tracking

The system maintains a rolling history of the last **10 prediction checks**:

```python
prediction_history = [
    {'epoch': 70, 'up_pct': 82.0, 'down_pct': 18.0, 'balance': 0.36},
    {'epoch': 72, 'up_pct': 85.0, 'down_pct': 15.0, 'balance': 0.30},
    {'epoch': 74, 'up_pct': 88.0, 'down_pct': 12.0, 'balance': 0.24},
    # ... up to 10 entries
]
```

**Purpose**: 
- Detect collapse trends over time
- Provide debugging information if training stops
- Help identify when collapse began

**Logged on Failure**:
```
❌ STOPPING: Prediction collapse persists after 5 recovery attempts
   Final distribution: 92.0% UP, 8.0% DOWN
   Collapse history (last 10 checks):
     Epoch 70: 82.0% UP, 18.0% DOWN (balance=0.360)
     Epoch 72: 85.0% UP, 15.0% DOWN (balance=0.300)
     Epoch 74: 88.0% UP, 12.0% DOWN (balance=0.240)
     Epoch 76: 90.0% UP, 10.0% DOWN (balance=0.200)
     Epoch 78: 92.0% UP, 8.0% DOWN (balance=0.160)
```

## Configuration

### Key Parameters

```python
class PredictionCollapseCallback:
    def __init__(
        self,
        X_val,               # Validation data for prediction checks
        y_val,               # Validation labels
        y_train=None,        # Training labels for bias calculation
        check_every=2,       # Check every N epochs
        max_recovery_attempts=5  # Maximum recovery attempts
    ):
```

**check_every**: How often to check predictions (default: every 2 epochs)  
**max_recovery_attempts**: Number of recovery strategies to try (default: 5)

### Stopping Criteria

Training stops when:
1. Severe collapse (>90%) persists for **4 consecutive checks** (8 epochs)
2. All **5 recovery attempts** have been exhausted

This gives the model `4 × 2 = 8` epochs after the final recovery attempt to demonstrate improvement.

## Integration with Training

The callback is automatically integrated into the Transformer training pipeline:

```python
# In TransformerDirectionTrainer.train()
callbacks = [
    # ... other callbacks ...
    PredictionCollapseCallback(
        X_val=X_val,
        y_val=y_val,
        y_train=y_train,
        check_every=2,
        max_recovery_attempts=5
    ),
]

history = model.fit(X_train, y_train, callbacks=callbacks, ...)
```

## Interaction with Other Components

### Focal Loss
The **AntiCollapseFocalLoss** works in tandem with collapse detection:
- **Loss Function**: Penalizes the model for predicting the same class
- **Callback**: Monitors and intervenes if loss function fails to prevent collapse

### Early Stopping
Early stopping is disabled during recovery:
```python
# Recovery resets collapse_epochs counter
self.collapse_epochs = 0  # Gives model more time to improve
```

### Learning Rate Scheduling
Recovery overrides any LR scheduler temporarily:
```python
# Each recovery strategy sets its own LR factor
self.model.optimizer.learning_rate.assign(new_lr)
```

## Troubleshooting

### High Collapse Rate

**Symptom**: Model collapses frequently during training

**Possible Causes**:
1. **Class Imbalance**: Training data heavily skewed toward one class
2. **High Learning Rate**: Model learning too quickly, overshooting
3. **Insufficient Regularization**: Model overfitting to majority class

**Solutions**:
1. Use **sample weights** to balance class contribution
2. Reduce initial learning rate
3. Increase dropout or L2 regularization
4. Use focal loss with higher gamma (more focus on minority class)

### Recovery Failures

**Symptom**: All 5 recovery attempts fail, training stops

**Possible Causes**:
1. **Data Quality**: Poor signal in features, model has no information
2. **Architecture**: Model too simple or too complex for the task
3. **Hyperparameters**: Learning rate, regularization poorly tuned

**Solutions**:
1. **Inspect Data**: Verify features have predictive power
2. **Adjust Architecture**: Try different model sizes or types
3. **Tune Hyperparameters**: Use lower LR, stronger regularization
4. **Manual Intervention**: Use warm-start from a pre-trained model

### Oscillating Predictions

**Symptom**: Model oscillates between classes after recovery

**Example**:
```
Epoch 80: 92% UP → Recovery → Epoch 85: 91% DOWN → Recovery → ...
```

**Possible Causes**:
1. **LR Too High**: Model jumping between minima
2. **Noise Too Strong**: Weight perturbation too aggressive
3. **Unstable Training**: Batch size too small, high variance

**Solutions**:
1. Reduce initial learning rate
2. Use gentler weight perturbation (reduce noise scale)
3. Increase batch size for more stable gradients
4. Use gradient clipping (already enabled with `clipnorm=1.0`)

## Monitoring

### Logs to Watch

**Healthy Training**:
```
📊 Prediction distribution at epoch 50: 52.0% UP, 48.0% DOWN (balance=0.960)
📊 Prediction distribution at epoch 60: 54.0% UP, 46.0% DOWN (balance=0.920)
```

**Warning Signs**:
```
⚡ Early imbalance warning at epoch 70: 82.0% UP, 18.0% DOWN
⚠️ MODERATE IMBALANCE at epoch 75: 87.0% UP, 13.0% DOWN
```

**Recovery**:
```
🚨 SEVERE PREDICTION COLLAPSE at epoch 80: 92.0% UP, 8.0% DOWN
🔧 COLLAPSE RECOVERY attempt 1/5
  → Strategy 1: Restoring best balanced weights (balance=0.850)
  → Learning rate adjusted to 5.00e-05 (factor=0.5)
```

**Failure**:
```
❌ STOPPING: Prediction collapse persists after 5 recovery attempts
   Final distribution: 93.0% UP, 7.0% DOWN
```

## Best Practices

1. **Monitor Balance Early**: Watch for early warning signs (>80%)
2. **Save Best Weights**: Lower threshold (0.25) ensures checkpoints exist
3. **Use Focal Loss**: Helps prevent initial collapse
4. **Balanced Data**: Ensure training data isn't heavily skewed
5. **Conservative LR**: Start with lower learning rate if collapse is frequent
6. **Check Data Quality**: If all recoveries fail, inspect features

## Related Components

- **AntiCollapseFocalLoss** (`src/models/tensorflow_models.py`): Loss function that penalizes collapse
- **Output Calibration** (`src/training/modular_trainers.py`): Post-training threshold adjustment
- **EMA Weights** (`src/training/modular_trainers.py`): Exponential moving average for stable inference
- **Sample Weighting** (`src/training/modular_trainers.py`): Balance class contributions during training

## Performance Impact

**Overhead**: Minimal
- Prediction check every 2 epochs: ~2-5 seconds per check
- Recovery intervention: ~1-2 seconds for weight manipulation
- Total: <1% of training time

**Benefits**:
- Prevents wasted training time on collapsed models
- Higher success rate in achieving balanced predictions
- Better model quality through early intervention

## Version History

- **v1.0** (Original): Basic collapse detection at >90%, 3 recovery attempts
- **v2.0** (Current): Graduated detection (80%/85%/90%), 5 progressive strategies, prediction history tracking
