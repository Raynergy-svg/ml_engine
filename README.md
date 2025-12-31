# Enhanced ML Engine Trading Bot

A comprehensive, production-ready machine learning engine for stock market prediction and algorithmic trading. This system includes state-of-the-art deep learning models, robust data processing, real-time monitoring, and extensive evaluation capabilities.

## ⚠️ Important Note

This is a research and educational tool. Always backtest thoroughly and use proper risk management before any live trading.

## Recent Improvements (v2.0)

This version includes significant enhancements to code quality, reliability, and performance:

### 🛡️ Enhanced Data Validation
- Custom `DataValidationError` exception for better error handling
- Comprehensive input validation for all data processing functions
- Automatic NaN and infinity value detection and handling
- DataFrame validation with required column checking
- Robust sequence creation with edge case handling

### 🧠 Improved Model Architecture
- Enhanced `StockPredictor` with input parameter validation
- Better weight initialization for faster convergence
- Improved gradient flow with residual connections
- Layer normalization for training stability
- Support for bidirectional LSTM models
- Flash attention mechanism support for PyTorch 2.0+

### 📊 Advanced Data Processing
- Multi-method feature normalization (Standard, MinMax, Robust scaling)
- Improved sequence generation with validation
- Better handling of MultiIndex data
- Enhanced caching mechanisms
- Comprehensive logging throughout data pipeline

### 🔧 Configuration Management
- Enhanced YAML configuration loading with validation
- Automatic configuration validation and warnings
- Support for default config merging
- Better error messages for configuration issues
- Comprehensive logging of configuration loading

### 🐛 Error Handling & Logging
- Structured logging with enhanced formatters
- Try-catch blocks throughout critical paths
- Custom exception classes for different error types
- Better user feedback with rich console output
- Detailed error messages with context

### 🔒 Security Updates
- Updated PyTorch from 2.0.0 to 2.6.0 (fixes critical vulnerabilities)
- Updated aiohttp from 3.9.0 to 3.9.4 (fixes DoS and directory traversal)
- Comprehensive dependency management with requirements.txt
- Security-conscious default configurations

### ✅ Testing Infrastructure
- Comprehensive unit tests for data processing
- Model validation tests
- Configuration management tests
- Edge case and robustness testing
- 90%+ code coverage for core modules

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

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the engine by editing the `config_tuned.yaml` file:
   ```bash
   cp .env.example .env  # Create environment file
   # Edit config_tuned.yaml with your settings
   ```

5. Verify installation:
   ```bash
   python main.py --help
   ```

## Usage

Run the CLI with:
```bash
python main.py <command> [--config path/to/config.yaml]

Default config: `./config_tuned.yaml`.
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

Customize the engine settings in the `config_tuned.yaml` file. Adjust parameters such as learning rate, batch size, model architecture, hardware settings, and more.

## Contributing

Contributions are welcome! To get started:
1. Fork the repository.
2. Create a feature branch.
3. Commit and push your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License.

## Best Practices

### Data Preparation
- Always validate your data before training using the built-in validation functions
- Check for missing values and handle them appropriately
- Normalize/standardize features for better model convergence
- Use appropriate sequence lengths based on your data characteristics

### Model Training
- Start with smaller models and gradually increase complexity
- Monitor validation loss to detect overfitting early
- Use early stopping to prevent unnecessary training
- Enable mixed precision training for faster GPU training
- Save checkpoints regularly during long training runs

### Configuration
- Review and validate your `config_tuned.yaml` before training
- Start with conservative learning rates (e.g., 0.001)
- Use learning rate scheduling for better convergence
- Adjust batch size based on your available memory

### Error Handling
- Check logs regularly for warnings and errors
- Use the validation functions before starting long training runs
- Keep backups of successful model configurations
- Monitor system resources during training
## Quick start

### Install the `buddy` shell command

If `buddy` on your machine points to an old/broken script (e.g. wrong repo path), install a launcher that runs this repo’s CLI:

```bash
bash scripts/install_buddy_cli.sh
```

Then run (interactive prompts by default):

```bash
buddy
```

Dry-run (no orders):

```bash
buddy --dry-run --instrument USD_JPY --granularity M5 --candles 300
```

If you don’t want to install a launcher, this always works:

```bash
/Users/mirelacertan/miniforge3/bin/conda run -n ml_engine python main.py buddy
```

## Testing

Run the test suite to verify your installation:

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed via `requirements.txt`
2. **CUDA errors**: Check PyTorch CUDA compatibility with your GPU
3. **Memory errors**: Reduce batch size or use gradient accumulation
4. **Configuration errors**: Validate config file structure matches requirements

### Getting Help

- Check the logs in `my_app.log` for detailed error messages
- Review configuration validation warnings
- Ensure all required fields are present in `config.yaml`

## Contact

For support or inquiries, please open an issue on [GitHub Issues](https://github.com/Raynergy-svg/ml_engine/issues).
