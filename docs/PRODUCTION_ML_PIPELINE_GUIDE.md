# Production-Grade ML Pipeline - Usage Guide

This guide covers the three main components of the production-grade ML pipeline:
1. **Global Seed Management** - Reproducibility across all random number generators
2. **Optuna Integration** - Automated hyperparameter optimization
3. **Model Registry & Versioning** - Centralized model artifact management

---

## 1. Global Seed Management

### Overview
The seed manager ensures complete reproducibility across Python, NumPy, TensorFlow, and scikit-learn by synchronizing all random number generators with a single global seed.

### Basic Usage

```python
from src.utils.seed_manager import set_global_seed, get_global_seed

# Set global seed for reproducibility
set_global_seed(42)

# Now all random operations are deterministic
import numpy as np
import random

# These will produce the same values every time
random_value = random.random()
numpy_array = np.random.rand(10)
```

### Configuration File

Enable in `config/config_improved_H1.yaml`:

```yaml
reproducibility:
  enabled: true
  global_seed: 42  # Set to null to disable seeding
```

### Environment Variable

Set via environment variable:

```bash
export RANDOM_SEED=42
python main.py train-buddy --instrument EUR_USD
```

### Temporary Seed Changes

Use context manager for temporary seed changes:

```python
from src.utils.seed_manager import set_global_seed, SeedContext

set_global_seed(42)

# Temporarily use different seed
with SeedContext(123):
    # Code here uses seed 123
    pass

# Seed 42 is restored here
```

### Thread-Safe Usage

Get thread-local RNG for concurrent operations:

```python
from src.utils.seed_manager import get_thread_rng

# Each thread gets its own RNG initialized with global seed
rng = get_thread_rng()
random_value = rng.random()
```

### scikit-learn Integration

```python
from src.utils.seed_manager import set_global_seed, get_sklearn_random_state
from sklearn.ensemble import RandomForestClassifier

set_global_seed(42)

# Use global seed for sklearn models
clf = RandomForestClassifier(random_state=get_sklearn_random_state())
```

---

## 2. Optuna Hyperparameter Optimization

### Overview
Optuna provides automated hyperparameter tuning with advanced features like pruning, multi-objective optimization, and persistent storage.

### Basic Usage

```python
from src.training.optuna_tuner import OptunaConfig, OptunaTuner

# Define objective function
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train model with these parameters
    # ... training code ...
    
    return val_loss  # Metric to minimize

# Configure optimization
config = OptunaConfig(
    study_name="model_optimization",
    n_trials=100,
    direction="minimize",
    storage="sqlite:///trained_data/optuna/studies.db",
)

# Run optimization
tuner = OptunaTuner(config)
results = tuner.optimize(objective)

print(f"Best params: {results['best_params']}")
print(f"Best value: {results['best_value']:.4f}")
```

### Configuration File

Enable in `config/config_improved_H1.yaml`:

```yaml
optuna:
  enabled: true
  study_name: fx_trading_bot_optimization
  n_trials: 100
  storage: "sqlite:///trained_data/optuna/studies.db"
  
  # Sampler (optimization algorithm)
  sampler: tpe  # 'tpe', 'cmaes', 'random'
  
  # Pruner (early stopping)
  pruner: median  # 'median', 'successive_halving', 'hyperband', null
  
  # Parameter search space
  param_space:
    d_model: [16, 32, 64, 128]
    num_heads: [2, 4, 8]
    dropout: [0.1, 0.5]
    learning_rate: [1e-5, 1e-2]
```

### Pruning (Early Stopping)

Use pruning to stop unpromising trials early:

```python
def objective(trial):
    # Configure model
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Train for multiple epochs
    for epoch in range(epochs):
        # ... training code ...
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss
```

### Multi-Objective Optimization

Optimize multiple metrics simultaneously:

```python
config = OptunaConfig(
    study_name="multi_objective",
    n_trials=100,
    directions=['minimize', 'maximize'],  # Two objectives
    metric_names=['val_loss', 'f1_score'],
)

def objective(trial):
    # ... configure and train model ...
    return [val_loss, f1_score]  # Return both metrics

tuner = OptunaTuner(config)
results = tuner.optimize(objective)

# Get Pareto-optimal solutions
best_trials = results['best_trials']
for trial in best_trials:
    print(f"Params: {trial.params}, Metrics: {trial.values}")
```

### Visualization

```python
# Plot optimization history
tuner.plot_optimization_history(save_path="history.html")

# Plot parameter importances
tuner.plot_param_importances(save_path="importances.html")
```

### Transformer-Specific Helper

```python
from src.training.optuna_tuner import create_transformer_objective

# Create objective for Transformer model
objective = create_transformer_objective(
    train_fn=train_model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    fixed_params={'epochs': 50},  # Don't optimize epochs
)

tuner = OptunaTuner(config)
results = tuner.optimize(objective)
```

---

## 3. Model Registry & Versioning

### Overview
The model registry provides centralized management of model artifacts with version control, metadata tracking, and model promotion workflow.

### Basic Usage

```python
from src.utils.model_registry import ModelRegistry, ModelMetadata

# Create registry
registry = ModelRegistry(registry_path="trained_data/models")

# Create model metadata
metadata = ModelMetadata(
    model_name="transformer_direction",
    version="1.0.0",
    hyperparameters={
        "d_model": 32,
        "num_heads": 4,
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    metrics={
        "val_loss": 0.45,
        "val_accuracy": 0.78,
        "f1_score": 0.76,
    },
    description="Initial production model",
)

# Register model
registry.register_model(
    metadata,
    model_path="trained_data/models/EUR_USD/transformer_direction.keras",
    dataset_path="market_data/EUR_USD_H1.csv",  # For hashing
    stage="dev",
)
```

### Automatic Metadata from Training

```python
from src.utils.model_registry import create_model_metadata_from_training
from src.training.modular_trainers import TrainerConfig

# Training config
config = TrainerConfig(
    d_model=32,
    num_heads=4,
    dropout=0.2,
)

# Training metrics
metrics = {
    "val_loss": 0.45,
    "val_accuracy": 0.78,
}

# Create metadata automatically
metadata = create_model_metadata_from_training(
    model_name="transformer_direction",
    version="1.0.0",
    trainer_config=config,
    metrics=metrics,
    description="Trained on EUR_USD H1 data",
)
```

### Version Management

```python
# List all versions
versions = registry.list_versions("transformer_direction")
for v in versions:
    print(f"v{v.version}: loss={v.metrics.get('val_loss', 'N/A')}")

# Get best model by metric
best_model = registry.get_best_model(
    "transformer_direction",
    metric="val_loss",
    higher_is_better=False,
)
print(f"Best model: v{best_model.version} with loss {best_model.metrics['val_loss']}")

# Get specific version
model_v1 = registry.get_metadata("transformer_direction", "1.0.0")
```

### Version Comparison

```python
# Compare two versions
comparison = registry.compare_versions(
    "transformer_direction",
    version1="1.0.0",
    version2="2.0.0",
)

print("Hyperparameter changes:")
for param, change in comparison["hyperparameter_changes"].items():
    print(f"  {param}: {change['v1']} → {change['v2']}")

print("\nMetric changes:")
for metric, change in comparison["metric_changes"].items():
    print(f"  {metric}: {change['v1']} → {change['v2']} ({change['pct_change']:+.1f}%)")
```

### Model Promotion Workflow

```python
# Promote from dev to staging
registry.promote_model(
    model_name="transformer_direction",
    version="1.0.0",
    from_stage="dev",
    to_stage="staging",
)

# Promote to production
registry.promote_model(
    model_name="transformer_direction",
    version="1.0.0",
    from_stage="staging",
    to_stage="production",
)

# Check current stage
metadata = registry.get_metadata("transformer_direction", "1.0.0")
print(f"Current stage: {metadata.tags.get('stage')}")
```

### MLflow Integration

```python
# Enable MLflow tracking
registry = ModelRegistry(
    registry_path="trained_data/models",
    enable_mlflow=True,
    mlflow_tracking_uri="http://localhost:5000",  # MLflow server
)

# Models will be automatically logged to MLflow
registry.register_model(metadata, model_path)
```

---

## Complete Training Workflow Example

Here's how all three components work together:

```python
from src.utils.seed_manager import set_global_seed
from src.training.optuna_tuner import OptunaConfig, OptunaTuner
from src.utils.model_registry import (
    ModelRegistry,
    create_model_metadata_from_training,
)

# 1. Set seed for reproducibility
set_global_seed(42)

# 2. Optimize hyperparameters with Optuna
def train_and_evaluate(params, X_train, y_train, X_val, y_val, trial=None):
    # Build model with params
    model = build_model(params)
    
    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Return validation loss
    return history.history['val_loss'][-1]

optuna_config = OptunaConfig(
    study_name="transformer_tuning",
    n_trials=50,
    storage="sqlite:///trained_data/optuna/studies.db",
)

tuner = OptunaTuner(optuna_config)
results = tuner.optimize(
    lambda trial: train_and_evaluate(
        params={
            'd_model': trial.suggest_categorical('d_model', [16, 32, 64]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        },
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        trial=trial,
    )
)

# 3. Train final model with best parameters
best_params = results['best_params']
final_model = build_model(best_params)
history = final_model.fit(X_train, y_train, validation_data=(X_val, y_val))

# 4. Save model with versioning
model_path = "trained_data/models/EUR_USD/transformer_direction.keras"
final_model.save(model_path)

# 5. Register in model registry
registry = ModelRegistry()
metadata = create_model_metadata_from_training(
    model_name="transformer_direction",
    version="1.0.0",
    trainer_config=best_params,
    metrics={
        "val_loss": history.history['val_loss'][-1],
        "val_accuracy": history.history['val_accuracy'][-1],
    },
    description="Optimized with Optuna, trained with seed 42",
)

registry.register_model(
    metadata,
    model_path=model_path,
    dataset_path="market_data/EUR_USD_H1.csv",
    stage="dev",
)

print("✅ Training complete with full reproducibility and versioning")
print(f"Best params: {best_params}")
print(f"Model registered: v{metadata.version}")
```

---

## Best Practices

### Reproducibility
1. Always set global seed at the start of training scripts
2. Use environment variables for seed configuration in CI/CD
3. Record seed in model metadata for later reproduction
4. Use thread-safe RNG for concurrent operations

### Hyperparameter Optimization
1. Start with small `n_trials` (10-20) to test setup
2. Use pruning to save compute on unpromising trials
3. Enable persistent storage for long-running optimizations
4. Visualize results to understand parameter importance
5. Use multi-objective optimization when trading off metrics

### Model Versioning
1. Use semantic versioning (MAJOR.MINOR.PATCH)
2. Always register models with complete metadata
3. Compare versions before promoting to production
4. Track git commits for code reproducibility
5. Use model promotion workflow (dev → staging → production)

### Integration
1. Set seed before Optuna optimization for reproducible trials
2. Register best model from Optuna in model registry
3. Include Optuna study name in model metadata
4. Use MLflow for experiment tracking across all components

---

## Troubleshooting

### Seed Not Taking Effect
- Check that seed is set before importing ML libraries
- Verify TensorFlow determinism is enabled
- Some operations may still be non-deterministic (e.g., GPU reductions)

### Optuna Trials Failing
- Check objective function for exceptions
- Verify pruning is not too aggressive
- Ensure storage path is writable
- Check parameter ranges are valid

### Model Registry Issues
- Ensure registry path has write permissions
- Check git is available for commit tracking
- Verify dataset path exists for hashing
- MLflow integration requires server setup

---

## API Reference

### Seed Manager
- `set_global_seed(seed: int)` - Set global seed
- `get_global_seed() -> Optional[int]` - Get current seed
- `get_thread_rng() -> np.random.Generator` - Get thread-local RNG
- `get_sklearn_random_state() -> int` - Get seed for sklearn
- `reset_seed()` - Reset seed to None
- `initialize_from_env() -> Optional[int]` - Load from environment
- `SeedContext(seed)` - Context manager for temporary seed

### Optuna Tuner
- `OptunaConfig` - Configuration dataclass
- `OptunaTuner(config)` - Main tuner class
- `optimize(objective)` - Run optimization
- `get_best_params()` - Get best parameters
- `plot_optimization_history()` - Visualize progress
- `plot_param_importances()` - Parameter importance

### Model Registry
- `ModelMetadata` - Metadata dataclass
- `ModelRegistry(path)` - Main registry class
- `register_model(metadata, path)` - Register new version
- `get_metadata(name, version)` - Get version metadata
- `list_versions(name)` - List all versions
- `get_best_model(name, metric)` - Get best version
- `compare_versions(name, v1, v2)` - Compare versions
- `promote_model(name, version, from, to)` - Promote stage
