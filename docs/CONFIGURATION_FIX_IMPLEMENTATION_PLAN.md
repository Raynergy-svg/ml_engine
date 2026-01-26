# Configuration Fix Implementation Plan

## Overview
This plan outlines the systematic approach to fix all configuration mismatches, parameter discrepancies, and structural inconsistencies between `config.yaml` and `main.py` to ensure proper training workflow integration with OANDA services.

## Implementation Strategy

### Phase 1: Configuration Structure Reorganization
**Objective**: Remove duplicates and organize sections to match main.py expectations

**Actions**:
1. Remove duplicate configuration entries
2. Reorganize sections in logical order
3. Ensure consistent naming conventions
4. Add missing required sections

**Target Sections**:
- `data:` - Data processing configuration
- `model:` - Model architecture configuration  
- `training:` - Training workflow configuration
- `buddy:` - Buddy-specific configuration
- `fx:` - FX trading configuration
- `paths:` - Directory paths

### Phase 2: Parameter Consistency Fix
**Objective**: Ensure all parameters match between config.yaml and main.py expectations

**Actions**:
1. Verify all training parameters match main.py defaults
2. Fix model architecture parameters
3. Ensure FX configuration parameters are consistent
4. Add missing parameters that main.py expects

**Key Parameters to Fix**:
- `data.sequence_length` (should be 120)
- `model.input_size` (should be dynamic, remove hardcoded value)
- `training.epochs` (should be 200)
- `training.batch_size` (should be 32)
- `training.learning_rate` (should be 0.001)

### Phase 3: FX Configuration Integration
**Objective**: Properly integrate FX configuration with Buddy training workflow

**Actions**:
1. Map `fx:` configuration to Buddy training parameters
2. Create proper instrument selection logic
3. Integrate risk management settings
4. Ensure OANDA API configuration is accessible

**Integration Points**:
- `fx.instrument` → Buddy instrument selection
- `fx.granularity` → Buddy granularity setting
- `fx.equity` → Buddy equity configuration
- `fx.risk_per_trade_pct` → Buddy risk management

### Phase 4: OANDA Services Integration
**Objective**: Ensure OANDA services configuration is properly used by main.py

**Actions**:
1. Add OANDA API key management
2. Implement proper instrument and granularity handling
3. Ensure FX configuration is accessible to Buddy commands
4. Add proper error handling for OANDA integration

### Phase 5: Training Workflow Integration
**Objective**: Ensure all training workflow components work together

**Actions**:
1. Add complete data processing configuration
2. Ensure model architecture configuration matches expectations
3. Add complete training workflow configuration
4. Test configuration loading and validation

## Detailed Implementation Steps

### Step 1: Remove Duplicates and Reorganize
```yaml
# Remove these duplicates:
sequence_length: 120
sequence_length: 120  # Remove this

# Reorganize sections in this order:
paths:
data:
model:
training:
buddy:
fx:
visualization:
```

### Step 2: Add Missing Required Sections
```yaml
# Add missing data section
data:
  data_dir: trained_data/data
  sequence_length: 120
  required_features:
    - open
    - high
    - low
    - close
    - volume
    # ... (complete feature list)

# Add missing training section
training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 20
  # ... (complete training config)
```

### Step 3: Fix Model Configuration
```yaml
# Remove hardcoded input_size (should be dynamic)
model:
  type: attention_lstm
  # input_size: 104  # Remove this line
  hidden_size: 48
  num_layers: 3
  dropout: 0.35
  # ... (other model params)
```

### Step 4: Integrate FX Configuration
```yaml
# Ensure FX config is properly structured
fx:
  enabled: true
  tier: 1
  timezone: America/New_York
  granularity: M5
  instruments:
    - EUR_USD
    - GBP_USD
    - USD_JPY
  # ... (complete FX config)

# Map FX settings to Buddy
buddy:
  instrument: USD_JPY  # Default from fx.instruments
  granularity: M5      # Default from fx.granularity
  equity: 10000.0      # Default from fx.equity
  risk_per_trade_pct: 0.005  # Default from fx.risk
```

### Step 5: Add OANDA Integration
```yaml
# Add OANDA API configuration
oanda:
  api_key: ${OANDA_API_KEY}  # Environment variable
  account_id: ${OANDA_ACCOUNT_ID}
  environment: practice  # or live
  # ... (other OANDA settings)
```

## Testing Strategy

### Test 1: Configuration Loading
- Verify `utils.load_config()` loads the fixed config without errors
- Check that all required sections are present
- Validate parameter types and ranges

### Test 2: Training Workflow
- Test `python main.py train-buddy` with the fixed configuration
- Verify all training parameters are correctly applied
- Check that model architecture matches expectations

### Test 3: FX Integration
- Test `python main.py buddy` with FX configuration
- Verify instrument and granularity settings work
- Check risk management integration

### Test 4: OANDA Services
- Test OANDA data fetching with configuration
- Verify API key and account settings work
- Check error handling for API issues

## Success Criteria

1. **Configuration Loading**: No errors when loading config.yaml
2. **Training Execution**: `train-buddy` command works with all parameters
3. **FX Integration**: Buddy commands properly use FX configuration
4. **OANDA Services**: API integration works with configuration settings
5. **No Duplicates**: No duplicate configuration entries
6. **Complete Coverage**: All main.py expected sections are present

## Risk Mitigation

1. **Backup Original**: Keep original config.yaml as backup
2. **Incremental Changes**: Apply fixes incrementally and test each step
3. **Validation**: Use configuration validation at each step
4. **Rollback Plan**: Document how to revert changes if issues occur

## Implementation Timeline

- **Phase 1**: 30 minutes - Remove duplicates and reorganize
- **Phase 2**: 45 minutes - Fix parameter consistency  
- **Phase 3**: 60 minutes - FX configuration integration
- **Phase 4**: 45 minutes - OANDA services integration
- **Phase 5**: 60 minutes - Training workflow integration
- **Testing**: 60 minutes - Comprehensive testing

**Total Estimated Time**: 4 hours

This plan ensures systematic resolution of all configuration issues while maintaining compatibility with the existing codebase and OANDA services integration.