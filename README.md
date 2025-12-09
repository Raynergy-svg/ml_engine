# Enhanced ML Engine Trading Bot

A comprehensive, production-ready machine learning engine for stock market prediction and algorithmic trading. This system includes state-of-the-art deep learning models, robust data processing, real-time monitoring, and extensive evaluation capabilities.

## ⚠️ Important Note

This is a research and educational tool. Always backtest thoroughly and use proper risk management before any live trading.

## Features

- **Modular Architecture:** Easily extend and customize model training, evaluation, and prediction.
- **Live Dashboard:** Real-time monitoring of training progression and evaluation metrics.
- **AI Assistant:** Interactive assistance for code explanations, improvements, and troubleshooting.
- **OpenAI Integration:** Automatic configuration tuning and recommendations using OpenAI.
- **Advanced Visualization:** Built-in support for rich dashboard displays and data visualizations.
- **Real-Time Inference:** Continuous model inference with dynamic updates.
- **Hyperparameter Tuning:** Advanced mechanisms to optimize model performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Raynergy-svg/ml_engine.git
   cd ml_engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the engine by editing `config.yaml`:
   ```bash
   cp config.yaml config.yaml.backup  # Backup default config
   # Edit config.yaml with your preferred settings
   ```

## Quick Start

```bash
# Train a model with default configuration
python main.py train-model --config config.yaml

# Evaluate the trained model
python main.py evaluate-model --config config.yaml

# Make predictions
python main.py predict-price --config config.yaml
```

## Usage

Run the CLI with:
```bash
python main.py <command> [--config path/to/config.yaml]
```

### Available Commands

- **train-model:** Train the ML model with live progress updates.
- **evaluate-model:** Evaluate model performance with detailed metrics.
- **predict-price:** Generate price predictions using the trained model.
- **realtime-loop:** Start continuous real-time inference.
- **tune-model:** Initiate hyperparameter tuning.
- **profile-pipeline:** Profile the ML pipeline to identify bottlenecks.
- **visualize:** Launch dashboard visualizations.
- **openai-tune:** Execute auto-tuning via OpenAI integration.
- **ai-assistant:** Engage the interactive AI Assistant for code and system inquiries.

## Configuration

Customize the engine settings in the `config.yaml` file. Adjust parameters such as learning rate, batch size, model architecture, hardware settings, and more.

## Contributing

Contributions are welcome! To get started:
1. Fork the repository.
2. Create a feature branch.
3. Commit and push your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License.

## Project Structure

```
ml_engine/
├── main.py                      # Main CLI entry point
├── ml_engine.py                 # Core ML engine implementation
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── models_enhanced.py           # Neural network model architectures
├── data_loader.py               # Data loading utilities
├── data_processing.py           # Data preprocessing
├── data_processing_optimized.py # Optimized data processing
├── feature_engineering.py       # Feature creation
├── evaluation.py                # Model evaluation and metrics
├── memory_manager_enhanced.py   # Memory optimization
├── train_enhanced.py            # Enhanced training script
├── trading_env.py               # Trading environment
├── visualizer.py                # Visualization utilities
├── utils.py                     # Utility functions
├── config_validator.py          # Configuration validation
│
├── tests/                       # Test suite
│   ├── test_integration.py
│   ├── test_mock_integration.py
│   └── test_optimized_model.py
│
├── trained_data/                # Generated during training
│   ├── models/                  # Saved model checkpoints
│   ├── logs/                    # Training logs
│   ├── visualizations/          # Generated plots
│   └── checkpoints/             # Model checkpoints
│
└── market_data/                 # Market data files (not in repo)
```

## Configuration Options

The `config.yaml` file controls all aspects of the ML engine. Key sections:

### Model Configuration
```yaml
model:
  architecture: attention_lstm  # lstm, attention_lstm, gru, transformer, tcn
  hidden_size: 128
  num_layers: 3
  dropout: 0.3
```

### Training Configuration
```yaml
training:
  epochs: 200
  early_stopping_patience: 15

batch_size: 512
learning_rate: 0.0005
```

### Hardware Configuration
```yaml
device: cpu  # or 'cuda' for GPU
hardware:
  num_workers: 16
  pin_memory: true
  torch_num_threads: 16
```

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError` when running scripts
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Problem**: Configuration validation fails
```bash
# Solution: Check your config.yaml for errors
python config_validator.py
```

**Problem**: Out of memory errors
```bash
# Solution: Reduce batch_size in config.yaml
batch_size: 32  # or smaller
```

**Problem**: Training is too slow
```bash
# Solution: Enable GPU if available
device: cuda
enable_mixed_precision: true
```

**Problem**: Poor model performance
- Check data quality and preprocessing
- Try different model architectures
- Adjust hyperparameters (learning rate, batch size)
- Add more training data
- Increase model capacity (hidden_size, num_layers)

## Development

### Running Tests
```bash
# Run all tests (requires pytest)
pytest tests/

# Run specific test file
python tests/test_integration.py
```

### Code Style
The project uses:
- Black for code formatting
- Ruff for linting
- Type hints for better code quality

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For support or inquiries:
- Open an issue on [GitHub Issues](https://github.com/Raynergy-svg/ml_engine/issues)
- Check existing documentation in the `IMPROVEMENTS.md` file
