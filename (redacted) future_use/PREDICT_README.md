# Real-Time Stock Price Predictor

## Overview

The `predict.py` module provides a comprehensive real-time stock price prediction system with the following improvements:

## Key Features

### 1. **Enhanced Error Handling**
- Robust exception handling with detailed error messages
- Graceful fallback mechanisms for data fetching
- Proper validation of inputs and outputs

### 2. **Real Market Data Integration**
- Fetches live market data using yfinance
- Supports both real and simulated data modes
- Automatic data validation and preprocessing

### 3. **Rich Console Interface**
- Beautiful terminal UI with tables and progress indicators
- Color-coded output for better readability
- Trend indicators showing prediction changes
- Session summaries with statistics

### 4. **Validation Framework**
- Built-in model validation against historical data
- Calculates MAE, RMSE, and MAPE metrics
- Quick verification before production use

### 5. **Flexible Configuration**
- Command-line argument support
- Customizable prediction intervals
- Multiple ticker support
- Simulated vs real data modes

## Usage

### Basic Usage

```bash
# Use default settings (AAPL, 60s interval, real data)
python predict.py

# Specify a different ticker
python predict.py --ticker TSLA

# Use simulated data for testing
python predict.py --simulated

# Run validation before predictions
python predict.py --validate --ticker NVDA

# Custom interval (5 minutes)
python predict.py --ticker MSFT --interval 300
```

### Command-Line Arguments

- `--checkpoint PATH`: Path to model checkpoint (default: `./trained_data/models/model.pth`)
- `--ticker SYMBOL`: Stock ticker symbol (default: `AAPL`)
- `--interval SECONDS`: Prediction interval in seconds (default: `60`)
- `--simulated`: Use simulated data instead of real market data
- `--validate`: Run validation before starting predictions

## Improvements from Original

### Before
```python
# Simple loop with simulated data only
# No error handling
# No real data support
# No validation
# Plain text output
```

### After
```python
# Professional prediction system with:
✓ Real market data fetching
✓ Comprehensive error handling
✓ Model validation framework
✓ Rich terminal UI
✓ Trend analysis
✓ Session statistics
✓ CLI argument support
✓ Logging integration
```

## Functions

### `load_trained_engine(checkpoint_path, config)`
Loads a trained model with comprehensive error checking.

**Features:**
- File existence validation
- Detailed error messages
- Success confirmation

### `fetch_recent_data(ticker, days)`
Fetches recent stock market data for predictions.

**Features:**
- Automatic date range calculation
- Error handling for network issues
- Data validation

### `make_prediction(engine, data)`
Makes a prediction with proper preprocessing.

**Features:**
- Feature engineering integration
- Data validation
- Error recovery

### `real_time_prediction_loop(engine, ticker, interval, use_real_data)`
Main prediction loop with rich features.

**Features:**
- Real-time data fetching or simulation
- Progress indicators
- Prediction history tracking
- Trend analysis
- Beautiful table output
- Session summaries

### `validate_prediction(engine, ticker, days_back)`
Validates model accuracy against historical data.

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Example Output

```
══════════════════════════════════════
  Real-Time Stock Price Predictor  
══════════════════════════════════════

✓ Model loaded from ./trained_data/models/model.pth

Running quick test prediction...
✓ Model is working. Sample output: 185.23

Starting real-time prediction loop.
Ticker: AAPL | Interval: 60s | Mode: Real Data
Press Ctrl+C to exit.

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
┃ Time                ┃ Prediction ┃ Ticker ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
│ 2025-12-08 15:30:45 │ $187.45    │ AAPL   │
└─────────────────────┴────────────┴────────┘

↑ 1.21% from last prediction
```

## Error Handling

The module handles various error scenarios:

1. **Missing Checkpoint**: Clear error message with helpful tip
2. **Network Issues**: Graceful fallback and retry logic
3. **Data Validation Failures**: Informative warnings
4. **Prediction Errors**: Logged with context for debugging

## Logging

Comprehensive logging is implemented:
- INFO: Normal operations and status updates
- WARNING: Recoverable issues
- ERROR: Failed operations with details
- EXCEPTION: Full stack traces for debugging

## Integration

Works seamlessly with:
- `train.py`: Load models trained with the training module
- `preprocess.py`: Uses preprocessing pipeline
- Enhanced ML Engine: Compatible with all model architectures

## Best Practices

1. **Always validate first**: Use `--validate` before production
2. **Choose appropriate intervals**: Don't overload APIs
3. **Monitor predictions**: Watch for anomalies
4. **Keep models updated**: Retrain regularly with recent data

## Future Enhancements

Potential improvements:
- Multiple ticker simultaneous predictions
- Prediction confidence intervals
- Alert system for significant price movements
- Historical prediction tracking
- Web dashboard integration
- Automated trading signal generation

## Dependencies

- numpy
- pandas
- yfinance
- rich (for terminal UI)
- torch
- Custom modules: train, preprocess

## License

Part of the ML Engine project.
