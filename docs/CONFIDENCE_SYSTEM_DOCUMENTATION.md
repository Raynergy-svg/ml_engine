# Confidence-Based Trading System Documentation

This document describes the comprehensive confidence-based trading system improvements implemented to address the issues identified in the trade execution analysis.

## Overview

The trading bot now features a sophisticated confidence management system that:

1. **Recalibrates confidence scoring** to align with actual win probabilities
2. **Implements minimum win probability thresholds** to prevent low-confidence trades
3. **Provides dynamic position sizing** based on confidence levels
4. **Adjusts risk management** (SL/TP) based on confidence

## Key Components

### 1. Confidence Calibration (`confidence_calibration.py`)

**Purpose**: Aligns model confidence scores with actual win probabilities using historical data.

**Key Features**:
- Platt scaling (logistic regression) calibration
- Isotonic regression calibration
- Directional adjustment for neutral predictions
- Win probability adjustment
- Trading context adjustment

**Usage**:
```python
from confidence_calibration import create_default_calibrator, CalibrationResult

# Create calibrator
calibrator = create_default_calibrator()

# Fit with historical data
confidence_scores = [0.7, 0.8, 0.9, 0.6, 0.5]
actual_outcomes = [True, True, False, True, False]
calibrator.fit(confidence_scores, actual_outcomes)

# Calibrate new confidence
result: CalibrationResult = calibrator.calibrate_confidence(0.85)
print(f"Original: {result.original_confidence}")
print(f"Calibrated: {result.calibrated_confidence}")
print(f"Valid: {result.is_valid}")
```

**Configuration**:
```yaml
confidence_calibration:
  method: "platt"  # 'platt', 'isotonic', or 'none'
  min_confidence_threshold: 0.5
  max_confidence_threshold: 0.95
  apply_directional_adjustment: true
  neutral_threshold: 0.05
  directional_penalty: 0.1
  apply_win_probability_adjustment: true
  win_probability_threshold: 0.5
  apply_trading_context_adjustment: true
  trading_penalty: 0.15
```

### 2. Dynamic Position Sizing (`position_sizing.py`)

**Purpose**: Adjusts position size based on confidence levels and risk parameters.

**Key Features**:
- Risk-based position sizing with configurable risk per trade
- Confidence-based scaling factors
- Position size constraints (minimum/maximum)
- Integration with confidence calibration system

**Position Size Bands**:
- **Low Confidence** (50-65%): 50% of base position
- **Medium Confidence** (65-80%): 100% of base position  
- **High Confidence** (80%+): 200% of base position

**Usage**:
```python
from position_sizing import create_default_position_sizer, PositionSize

# Create position sizer
sizer = create_default_position_sizer()

# Calculate position size
position: PositionSize = sizer.calculate_position_size(
    account_equity=100000.0,
    stop_loss_pips=20.0,
    instrument="EUR_USD",
    raw_confidence=0.85
)

print(f"Units: {position.units}")
print(f"Confidence Level: {position.confidence_level}")
print(f"Position Multiplier: {position.position_multiplier}")
print(f"Risk Amount: ${position.risk_amount}")
```

**Configuration**:
```yaml
position_sizing:
  risk_per_trade_pct: 0.02  # 2% risk per trade
  min_confidence_threshold: 0.5
  max_position_multiplier: 3.0
  low_confidence_band: [0.5, 0.65]
  medium_confidence_band: [0.65, 0.8]
  high_confidence_band: [0.8, 1.0]
  low_confidence_multiplier: 0.5
  medium_confidence_multiplier: 1.0
  high_confidence_multiplier: 2.0
  max_position_pct: 0.10  # 10% max position
  min_position_size: 1000
```

### 3. Confidence-Based Risk Management (`risk_management.py`)

**Purpose**: Adjusts stop loss and take profit levels based on confidence levels.

**Key Features**:
- Dynamic risk-reward ratios based on confidence
- Confidence-based SL/TP adjustments
- Instrument-specific constraints
- Integration with position sizing

**Risk-Reward Ratios by Confidence**:
- **Low Confidence**: 1:1.5 R:R, wider stops, tighter targets
- **Medium Confidence**: 1:2.0 R:R, standard stops and targets
- **High Confidence**: 1:3.0 R:R, tighter stops, wider targets

**Usage**:
```python
from risk_management import create_default_risk_manager, RiskManagementResult

# Create risk manager
manager = create_default_risk_manager()

# Calculate risk levels
result: RiskManagementResult = manager.calculate_risk_levels(
    entry_price=1.2000,
    raw_confidence=0.85,
    base_stop_loss_pips=20.0,
    base_take_profit_pips=40.0,
    instrument="EUR_USD"
)

print(f"Stop Loss: {result.stop_loss_pips} pips")
print(f"Take Profit: {result.take_profit_pips} pips")
print(f"R:R Ratio: {result.risk_reward_ratio}")
print(f"Confidence Level: {result.confidence_level}")
```

**Configuration**:
```yaml
risk_management:
  low_confidence_rr: 1.5
  medium_confidence_rr: 2.0
  high_confidence_rr: 3.0
  low_confidence_threshold: 0.5
  medium_confidence_threshold: 0.7
  high_confidence_threshold: 0.85
  min_rr_ratio: 1.0
  max_rr_ratio: 5.0
  low_confidence_sl_multiplier: 1.2  # Wider stops
  medium_confidence_sl_multiplier: 1.0  # Standard stops
  high_confidence_sl_multiplier: 0.8  # Tighter stops
  low_confidence_tp_multiplier: 0.8  # Tighter targets
  medium_confidence_tp_multiplier: 1.0  # Standard targets
  high_confidence_tp_multiplier: 1.5  # Wider targets
  max_stop_loss_pips: 100.0
  min_take_profit_pips: 5.0
```

## Integration with Unified Talk

The confidence system is fully integrated with the unified_talk.py trading commands:

### Enhanced Trade Commands

**Auto Trade with Confidence**:
```bash
# System automatically uses confidence-based sizing and risk management
trade
trade auto
trade auto 10000 EUR_USD
```

**Manual Trade with Confidence**:
```bash
# System applies confidence-based SL/TP adjustments
buy
sell
```

**Confidence-Based Risk Management**:
```bash
# View current risk settings
risk

# Set stop loss and take profit (system will apply confidence adjustments)
sl 25
tp 75
```

### Trade Execution Flow

1. **Prediction**: Model generates prediction with confidence score
2. **Calibration**: Confidence is calibrated using historical data
3. **Validation**: Check if calibrated confidence meets minimum threshold
4. **Position Sizing**: Calculate position size based on confidence and risk
5. **Risk Management**: Determine SL/TP levels based on confidence
6. **Execution**: Execute trade with calculated parameters

## Configuration Updates

The main configuration file (`config.yaml`) now includes comprehensive settings for the confidence system:

```yaml
risk_management:
  # Base risk management settings
  base_position_fraction: 0.02
  min_confidence: 0.7
  max_confidence: 1.0
  
  # Confidence-based position sizing
  position_sizing:
    risk_per_trade_pct: 0.02
    min_confidence_threshold: 0.5
    max_position_multiplier: 3.0
    # ... position sizing configuration
  
  # Confidence-based risk management
  risk_management:
    low_confidence_rr: 1.5
    medium_confidence_rr: 2.0
    high_confidence_rr: 3.0
    # ... risk management configuration
  
  # Confidence calibration
  confidence_calibration:
    method: "platt"
    min_confidence_threshold: 0.5
    # ... calibration configuration
```

## Testing

Comprehensive test suites are provided:

### Unit Tests
- `tests/test_confidence_calibration.py`: Confidence calibration functionality
- `tests/test_position_sizing.py`: Position sizing logic
- `tests/test_risk_management.py`: Risk management features

### Integration Tests
- `tests/test_confidence_integration.py`: End-to-end confidence pipeline

### Running Tests
```bash
# Run all confidence system tests
pytest tests/test_confidence_*.py -v

# Run specific test modules
pytest tests/test_confidence_calibration.py -v
pytest tests/test_position_sizing.py -v
pytest tests/test_risk_management.py -v
pytest tests/test_confidence_integration.py -v
```

## Usage Examples

### Example 1: High Confidence Trade
```python
# Model prediction: 85% confidence
raw_confidence = 0.85

# After calibration and adjustments
calibrated_confidence = 0.78  # Adjusted down due to trading context

# Position sizing (account: $100,000, risk: 2% = $2,000)
position_size = 20,000 units  # 2x base position due to high confidence

# Risk management
stop_loss = 16 pips  # Tighter due to high confidence (20 * 0.8)
take_profit = 96 pips  # Wider due to high confidence (32 * 3.0)
# R:R = 6.0 (excellent risk-reward)

# Result: Large position with tight risk controls and excellent R:R
```

### Example 2: Medium Confidence Trade
```python
# Model prediction: 70% confidence
raw_confidence = 0.70

# After calibration and adjustments
calibrated_confidence = 0.65  # Medium confidence

# Position sizing
position_size = 10,000 units  # Standard position size

# Risk management
stop_loss = 20 pips  # Standard stops
take_profit = 40 pips  # Standard targets
# R:R = 2.0 (good risk-reward)

# Result: Standard position with balanced risk management
```

### Example 3: Low Confidence Trade (Rejected)
```python
# Model prediction: 45% confidence
raw_confidence = 0.45

# After calibration and adjustments
calibrated_confidence = 0.38  # Below minimum threshold

# Result: Trade rejected - no position taken
```

## Benefits

1. **Improved Accuracy**: Confidence scores better aligned with actual win probabilities
2. **Risk Management**: Dynamic position sizing prevents overexposure on low-confidence trades
3. **Better Risk-Reward**: Confidence-based SL/TP adjustments optimize risk-reward ratios
4. **Conservative Approach**: Minimum thresholds prevent poor-quality trades
5. **Adaptive**: System adapts to different market conditions and instrument characteristics

## Backward Compatibility

The confidence system is designed to be backward compatible:
- Existing configurations continue to work
- New features are opt-in through configuration
- Default settings provide conservative behavior
- Graceful degradation when confidence data is unavailable

## Monitoring and Tuning

### Key Metrics to Monitor
- Calibration accuracy (Brier score improvement)
- Win rate by confidence band
- Position sizing distribution
- Risk-reward ratio distribution
- Trade rejection rate

### Tuning Parameters
- Confidence thresholds based on historical performance
- Position sizing multipliers based on risk tolerance
- Risk-reward ratios based on market conditions
- Calibration frequency based on data availability

## Troubleshooting

### Common Issues

**Issue**: Trades being rejected too frequently
**Solution**: Lower minimum confidence threshold or improve model calibration

**Issue**: Position sizes too small
**Solution**: Increase risk per trade percentage or position size multipliers

**Issue**: Poor calibration accuracy
**Solution**: Collect more historical data or try different calibration methods

**Issue**: Inconsistent risk-reward ratios
**Solution**: Review confidence bands and adjust risk management parameters

### Debug Mode
Enable verbose logging to see confidence calculations:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed information about confidence adjustments, position sizing calculations, and risk management decisions.