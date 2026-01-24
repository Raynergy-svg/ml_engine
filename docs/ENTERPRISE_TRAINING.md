# Enterprise ML Training System

## Overview

The Enterprise Training System adds production-grade MLOps features to the existing `buddy train` command. Enable with `--enterprise` flag.

## Features

### 1. Experiment Tracking (MLflow Integration)
- **Full Experiment Logging**: All parameters, metrics, and artifacts logged automatically
- **Model Versioning**: Automatic model versioning with MLflow model registry
- **Hyperparameter Tracking**: Complete parameter search space recorded
- **Artifact Storage**: Models, configs, and reports stored with experiments

### 2. Statistical Validation
- **Bootstrap Confidence Intervals**: 95% CI for all key metrics
- **Walk-Forward Cross-Validation**: Time-series appropriate validation
- **Statistical Significance Tests**: T-tests for comparing models
- **Distribution Analysis**: Shapiro-Wilk normality tests

### 3. Reproducibility
- **Seed Management**: All random seeds captured and logged
- **Environment Fingerprinting**: Package versions, system info recorded
- **Deterministic Training**: Reproducible results across runs

### 4. Resource Monitoring
- **CPU/Memory Tracking**: Real-time resource utilization
- **GPU Monitoring**: Metal GPU metrics (M1/M2 optimized)
- **Training Time Estimation**: ETA and duration tracking

### 5. Data Validation
- **Schema Enforcement**: Automatic data type and range validation
- **Drift Detection**: Statistical tests for distribution changes
- **Missing Value Handling**: Configurable strategies for NaN handling

### 6. Professional Logging
- **Structured JSON Logs**: Machine-parseable log format
- **Severity Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Context Propagation**: Request IDs across training pipeline

## Installation

```bash
# Install new dependencies
pip install mlflow>=2.9.0 structlog>=23.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Quick Start

### Enterprise Training (DEFAULT for ensemble)
```bash
# All enterprise features enabled automatically!
buddy train --model-type ensemble --candles 12000 --granularity H1 --oanda-live --instrument EUR/USD
```

This automatically runs with:
- ✅ MLflow experiment tracking
- ✅ Walk-forward cross-validation (5 folds)
- ✅ Bootstrap 95% confidence intervals
- ✅ Professional training report generation

### Disable Enterprise Features (for faster training)
```bash
# Disable all enterprise features
buddy train --model-type ensemble --candles 12000 --oanda-live \
    --no-enterprise

# Or disable specific features
buddy train --model-type ensemble --candles 12000 --oanda-live \
    --no-bootstrap --cv-folds 0 --no-report
```

### CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--enterprise` | **True** | Enable enterprise features (MLflow, CV, bootstrap) |
| `--no-enterprise` | - | Disable all enterprise features |
| `--cv-folds` | **5** | Walk-forward cross-validation folds (0 to disable) |
| `--bootstrap` | **True** | Enable bootstrap 95% confidence intervals |
| `--no-bootstrap` | - | Disable bootstrap CI |
| `--bootstrap-samples` | 1000 | Number of bootstrap iterations |
| `--mlflow-experiment` | auto | MLflow experiment name |
| `--generate-report` | **True** | Generate markdown training report |
| `--no-report` | - | Skip generating the report |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────────┐  │
│  │ Data Source │──▶│ Validation   │──▶│  Feature Engineering         │  │
│  │ (OANDA/CSV) │   │ & Drift Det. │   │  (120+ financial features)   │  │
│  └─────────────┘   └──────────────┘   └──────────────────────────────┘  │
│                                                   │                      │
│                                                   ▼                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MODULAR DATA LOADERS                          │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │    │
│  │  │ Direction   │ │ Momentum    │ │ Risk        │ │ Confidence │ │    │
│  │  │ Sequences   │ │ Features    │ │ Features    │ │ Features   │ │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              ENTERPRISE ENSEMBLE TRAINER                         │    │
│  │                                                                   │    │
│  │  ┌────────────────────────────────────────────────────────────┐  │    │
│  │  │              Experiment Tracking (MLflow)                   │  │    │
│  │  │  • Parameters • Metrics • Artifacts • Model Registry       │  │    │
│  │  └────────────────────────────────────────────────────────────┘  │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │                    MODEL TRAINING                            │ │    │
│  │  │                                                               │ │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐   │ │    │
│  │  │  │ Transformer │  │  XGBoost    │  │  Random Forest     │   │ │    │
│  │  │  │ (Direction) │  │  (Momentum) │  │  (Risk/Volatility) │   │ │    │
│  │  │  │             │  │             │  │                     │   │ │    │
│  │  │  │ • EMA       │  │ • Gradient  │  │ • Bootstrap         │   │ │    │
│  │  │  │ • EWC       │  │   Boosted   │  │   Aggregating      │   │ │    │
│  │  │  │ • Replay    │  │ • Feature   │  │ • Feature           │   │ │    │
│  │  │  │   Buffer    │  │   Import.   │  │   Importance        │   │ │    │
│  │  │  └─────────────┘  └─────────────┘  └────────────────────┘   │ │    │
│  │  │                                                               │ │    │
│  │  │  ┌─────────────┐  ┌─────────────────────────────────────┐   │ │    │
│  │  │  │   Ridge     │  │   HistGradientBoosting              │   │ │    │
│  │  │  │(Confidence) │  │   (Hybrid Voting Ensemble)          │   │ │    │
│  │  │  └─────────────┘  └─────────────────────────────────────┘   │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐ │    │
│  │  │              VALIDATION & ANALYSIS                           │ │    │
│  │  │                                                               │ │    │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │ │    │
│  │  │  │ Walk-Forward  │  │   Bootstrap   │  │  Statistical    │  │ │    │
│  │  │  │     CV        │  │   CI (95%)    │  │   Tests         │  │ │    │
│  │  │  └───────────────┘  └───────────────┘  └─────────────────┘  │ │    │
│  │  └─────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                   │                                      │
│                                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       OUTPUT & REPORTING                          │    │
│  │                                                                    │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │
│  │  │  Models    │  │  Metrics   │  │  Reports   │  │  MLflow    │  │    │
│  │  │  (.keras)  │  │  (.json)   │  │  (.md)     │  │  Artifacts │  │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration Reference

### EnterpriseConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | required | Unique experiment identifier |
| `instrument` | str | "EUR_USD" | Trading instrument |
| `granularity` | str | "H1" | Candle timeframe |
| `enable_cross_validation` | bool | True | Enable walk-forward CV |
| `enable_statistical_validation` | bool | True | Enable bootstrap CI |
| `cv_n_splits` | int | 5 | Number of CV folds |
| `bootstrap_n_samples` | int | 1000 | Bootstrap iterations |
| `save_dir` | str | "trained_data/models" | Model save path |
| `log_dir` | str | "trained_data/logs" | Log directory |
| `warm_start` | bool | False | Continue from checkpoint |

### TrainerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 200 | Maximum training epochs |
| `batch_size` | int | 64 | Training batch size |
| `learning_rate` | float | 0.0003 | Initial learning rate |
| `patience` | int | 20 | Early stopping patience |

## Output Structure

```
trained_data/
├── models/
│   ├── transformer_direction.keras
│   ├── xgboost_momentum.json
│   ├── rf_risk.pkl
│   ├── ridge_confidence.pkl
│   └── hgb_voting.pkl
├── logs/
│   ├── experiments/
│   │   ├── EUR_USD_H1_20250120_results.json
│   │   └── EUR_USD_H1_20250120_report.md
│   ├── training.log
│   └── training.json  (structured logs)
└── mlruns/
    └── <experiment_id>/
        ├── meta.yaml
        └── <run_id>/
            ├── params/
            ├── metrics/
            └── artifacts/
```

## Statistical Validation Details

### Walk-Forward Cross-Validation

Walk-forward CV respects the temporal nature of financial data:

```
Fold 1: [Train: 0-2000] [Val: 2000-2400]
Fold 2: [Train: 0-2400] [Val: 2400-2800]
Fold 3: [Train: 0-2800] [Val: 2800-3200]
Fold 4: [Train: 0-3200] [Val: 3200-3600]
Fold 5: [Train: 0-3600] [Val: 3600-4000]
```

Each fold trains on all prior data and validates on future data, preventing look-ahead bias.

### Bootstrap Confidence Intervals

Bootstrap resampling provides non-parametric confidence intervals:

```python
# 1000 bootstrap samples
for i in range(1000):
    sample = resample(predictions, labels)
    metrics[i] = calculate_accuracy(sample)

# 95% CI from percentiles
ci_lower = percentile(metrics, 2.5)
ci_upper = percentile(metrics, 97.5)
```

### Statistical Significance Tests

- **T-Test**: Compares mean performance between models
- **Wilcoxon Signed-Rank**: Non-parametric alternative
- **Effect Size (Cohen's d)**: Practical significance measure

## MLflow Integration

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser: http://localhost:5000
```

### Querying Experiments

```python
import mlflow

# Get all runs
runs = mlflow.search_runs(
    experiment_names=["EUR_USD_H1_production"],
    filter_string="metrics.val_accuracy > 0.55"
)

# Get best run
best_run = runs.sort_values("metrics.val_accuracy", ascending=False).iloc[0]
print(f"Best accuracy: {best_run['metrics.val_accuracy']:.2%}")
```

## Best Practices

### 1. Data Preparation
- Always validate data before training
- Check for distribution drift from training data
- Use appropriate train/validation splits (temporal for time series)

### 2. Experiment Tracking
- Use meaningful experiment names: `{instrument}_{granularity}_{date}`
- Always enable cross-validation for production models
- Log all hyperparameters for reproducibility

### 3. Model Validation
- Never rely on single validation set accuracy
- Use bootstrap CI to understand performance variance
- Compare against baseline models

### 4. Production Deployment
- Only deploy models with statistically significant improvements
- Monitor for data drift in production
- Set up alerting for model degradation

## Troubleshooting

### Common Issues

**MLflow not tracking runs:**
```bash
# Ensure MLflow is installed
pip install mlflow>=2.9.0

# Check tracking URI
echo $MLFLOW_TRACKING_URI
```

**Bootstrap CI too wide:**
- Increase sample size (more data)
- Increase `bootstrap_n_samples` (more iterations)
- Check for data quality issues

**Cross-validation scores vary wildly:**
- Check for non-stationarity in data
- Consider longer training history
- Use more CV folds for stability

## API Reference

### enterprise_training.py

Core infrastructure classes:

```python
class ReproducibilityManager:
    """Manages random seeds and environment fingerprinting."""
    
class ResourceMonitor:
    """Tracks CPU, memory, and GPU utilization."""
    
class ExperimentTracker:
    """MLflow-based experiment tracking."""
    
class WalkForwardValidator:
    """Time-series cross-validation."""
    
class StatisticalValidator:
    """Bootstrap CI and statistical tests."""
    
class DataValidator:
    """Data quality and drift detection."""
```

### enterprise_integration.py

Integration layer:

```python
class EnterpriseEnsembleTrainer:
    """Main training orchestrator."""
    
    def train(self, data: Dict, config: TrainerConfig) -> EnsembleResults:
        """Run complete enterprise training pipeline."""
        
def generate_training_report(results: EnsembleResults, path: str) -> str:
    """Generate professional markdown report."""
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and formatting
- Testing requirements
- Pull request process

## License

Copyright 2025. See LICENSE for details.
