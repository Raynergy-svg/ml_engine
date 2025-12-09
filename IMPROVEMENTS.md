# ML Engine Trading Bot - Enhanced Version

A comprehensive, production-ready machine learning engine for stock market prediction and algorithmic trading. This enhanced version includes state-of-the-art models, advanced feature engineering, backtesting capabilities, and robust evaluation metrics.

## 🚀 New Features & Improvements

### Core Enhancements
- **Advanced Model Architectures**: LSTM, Attention-LSTM, GRU, Transformer, and TCN models
- **Comprehensive Feature Engineering**: 100+ technical indicators and statistical features
- **Robust Data Pipeline**: Enhanced data loading, validation, and preprocessing
- **Model Evaluation**: Extensive metrics, visualization, and performance analysis
- **Backtesting Framework**: Test strategies with realistic trading simulation
- **Memory Management**: Optimized for both CPU and GPU with intelligent caching
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)

### Technical Indicators Included
- Moving Averages (SMA, EMA)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- CCI (Commodity Channel Index)
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- Williams %R
- ADX (Average Directional Index)
- And many more...

### Statistical Features
- Returns and log returns
- Volatility measures
- Skewness and kurtosis
- Z-scores
- Momentum indicators
- Volume analysis
- Price ratios

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- yfinance (for data)
- Rich (for CLI)
- Matplotlib, Seaborn (for visualization)

## 🎯 Quick Start

### 1. Basic Training

Train a model with enhanced features:

```bash
python train_enhanced.py --data market_data/TSLA_data.csv --epochs 100
```

### 2. Using Different Model Architectures

Edit `config.yaml` to change the model:

```yaml
architecture: attention_lstm  # Options: lstm, attention_lstm, gru, transformer, tcn
hidden_size: 128
num_layers: 3
dropout: 0.2
learning_rate: 0.001
batch_size: 32
```

### 3. Custom Data

```python
from data_loader import MarketDataLoader

# Load your data
loader = MarketDataLoader(config)
df = loader.load_csv('path/to/your/data.csv')

# Preprocess with feature engineering
X_train, y_train, X_val, y_val, X_test, y_test = loader.preprocess(
    df,
    add_features=True,
    sequence_length=60
)
```

## 📊 Model Architectures

### 1. Attention-LSTM (Recommended)
Best for capturing long-term dependencies with attention mechanism.

```python
from models_enhanced import AttentiveLSTM

model = AttentiveLSTM(
    input_size=7,
    hidden_size=128,
    num_layers=3,
    num_heads=4,
    dropout=0.2
)
```

### 2. Transformer
State-of-the-art architecture for sequence modeling.

```python
from models_enhanced import TransformerPredictor

model = TransformerPredictor(
    input_size=7,
    hidden_size=128,
    num_layers=3,
    num_heads=8
)
```

### 3. GRU
Faster training with similar performance to LSTM.

```python
from models_enhanced import GRUPredictor

model = GRUPredictor(
    input_size=7,
    hidden_size=128,
    num_layers=3
)
```

## 🔬 Feature Engineering

The `FeatureEngineering` class provides comprehensive feature creation:

```python
from feature_engineering import FeatureEngineering

fe = FeatureEngineering()

# Add all features
df_enhanced = fe.create_features(df, include_all=True)

# Selective feature engineering
df = fe.add_technical_indicators(df)
df = fe.add_statistical_features(df)
df = fe.add_time_features(df)

# Feature selection
selected_df, top_features = fe.select_features(
    df,
    method='correlation',  # or 'f_test', 'mutual_info'
    top_k=50
)
```

## 📈 Evaluation & Backtesting

### Model Evaluation

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate predictions
metrics = evaluator.evaluate(y_true, y_pred)
evaluator.print_metrics(metrics)

# Plot results
evaluator.plot_predictions(
    y_true,
    y_pred,
    save_path='results.png'
)
```

Metrics include:
- MSE, RMSE, MAE
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Direction Accuracy
- Max/Median Error

### Backtesting

```python
from evaluation import Backtester, generate_simple_strategy_signals

# Initialize backtester
backtester = Backtester(
    initial_capital=10000.0,
    commission=0.001,
    slippage=0.001
)

# Generate trading signals
signals = generate_simple_strategy_signals(predictions, actual_prices)

# Run backtest
metrics = backtester.run_backtest(actual_prices, signals)

# Results include:
# - Total return %
# - Sharpe ratio
# - Maximum drawdown
# - Win rate
# - Number of trades
```

## 🎛️ Configuration

`config.yaml` provides comprehensive configuration:

```yaml
# Model Configuration
architecture: attention_lstm
hidden_size: 128
num_layers: 3
num_heads: 4
dropout: 0.3

# Training Configuration
learning_rate: 0.001
batch_size: 32
epochs: 200
early_stopping_patience: 20

# Data Configuration
data:
  sequence_length: 60
  test_size: 0.2
  validation_size: 0.1

# Hardware Configuration
device: cuda  # or 'cpu'
enable_mixed_precision: true
```

## 💾 Memory Management

The enhanced memory manager optimizes resource usage:

```python
from memory_manager_enhanced import MemoryManager, memory_efficient

# Initialize memory manager
memory_mgr = MemoryManager(
    device='cuda',
    threshold_mb=1000,
    proactive_cleanup=True
)

# Use decorator for memory-efficient functions
@memory_efficient
def train_model():
    # Your training code
    pass

# Monitor memory
stats = memory_mgr.monitor_memory()
print(f"Used: {stats['used_mb']:.2f}MB")
```

## 📊 Results Visualization

All visualizations are automatically saved to `trained_data/visualizations/`:

- `test_predictions.png`: Actual vs Predicted values
- Scatter plots
- Residuals analysis
- Distribution plots
- Backtest results

## 🧪 Testing

Run tests to ensure everything works:

```bash
pytest tests/
```

## 🔧 Advanced Usage

### Ensemble Models

```python
from models_enhanced import EnsemblePredictor, StockPredictor, AttentiveLSTM

# Create base models
models = [
    StockPredictor(input_size=7, hidden_size=128),
    AttentiveLSTM(input_size=7, hidden_size=128),
]

# Create ensemble
ensemble = EnsemblePredictor(
    models=models,
    ensemble_method='weighted'  # or 'average'
)
```

### Custom Feature Engineering

```python
fe = FeatureEngineering()

# Add custom indicators
df['custom_indicator'] = df['close'].rolling(30).mean() / df['close'].rolling(60).mean()

# Then add standard features
df = fe.create_features(df)
```

### Hyperparameter Tuning

```python
# Use Optuna or similar
import optuna

def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Train and evaluate model
    # Return validation loss
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## 📝 Best Practices

1. **Data Quality**: Always validate your data before training
2. **Feature Selection**: Use feature selection to reduce overfitting
3. **Cross-Validation**: Use time-series cross-validation
4. **Regularization**: Apply dropout and weight decay
5. **Early Stopping**: Prevent overfitting with patience
6. **Learning Rate Scheduling**: Use ReduceLROnPlateau
7. **Backtesting**: Always backtest before live trading
8. **Risk Management**: Implement proper position sizing

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in config.yaml
batch_size: 16

# Enable gradient accumulation
gradient_accumulation_steps: 4
```

### Poor Model Performance
1. Check data quality and preprocessing
2. Try different architectures
3. Adjust hyperparameters
4. Add more features
5. Increase training data

### Slow Training
1. Enable mixed precision training
2. Use GPU if available
3. Reduce sequence length
4. Optimize batch size
5. Use DataLoader with multiple workers

## 📚 Project Structure

```
ml_engine/
├── models_enhanced.py          # Enhanced model architectures
├── data_loader.py               # Data loading and preprocessing
├── feature_engineering.py       # Feature creation
├── evaluation.py                # Model evaluation and backtesting
├── memory_manager_enhanced.py   # Memory optimization
├── train_enhanced.py            # Training script
├── trading_env.py               # RL environment
├── config.yaml                  # Configuration
├── requirements.txt             # Dependencies
└── trained_data/
    ├── models/                  # Saved models
    ├── logs/                    # Training logs
    ├── visualizations/          # Plots
    └── checkpoints/             # Model checkpoints
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for the framework
- Stable-Baselines3 for RL implementations
- yfinance for market data
- The open-source ML community

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review the examples

## 🎓 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Time Series Forecasting](https://otexts.com/fpp3/)
- [Quantitative Trading](https://www.quantstart.com/)
- [Machine Learning for Trading](https://www.coursera.org/learn/machine-learning-trading)

---

**Note**: This is a research and educational tool. Always backtest thoroughly and use proper risk management before any live trading.
