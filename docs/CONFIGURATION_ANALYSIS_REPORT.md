# Configuration Consistency Analysis Report

## Overview
This report analyzes the configuration consistency between `config.yaml` and `main.py` for the ML Engine trading bot, focusing on training execution integration and OANDA services API usage.

## Key Findings

### 1. Configuration Loading System
- **main.py** uses `utils.load_config()` function to load YAML configuration
- **utils.py** provides caching and validation for configuration loading
- **main.py** defaults to `"./config.yaml"` as `DEFAULT_CONFIG_PATH`

### 2. Configuration Structure Analysis

#### Main Configuration Sections in config.yaml:
- `paths`: Directory paths for data, models, logs, etc.
- `data`: Data processing configuration
- `model`: Model architecture and training parameters
- `optimizer`: Training optimization settings
- `training`: Training workflow configuration
- `buddy`: Buddy-specific training and trading parameters
- `fx`: Forex trading configuration
- `visualization`: Dashboard and visualization settings

#### Critical Configuration Mismatches Found:

### 3. Parameter Discrepancies

#### A. Training Parameters
| Parameter | config.yaml Value | main.py Expected | Status |
|-----------|------------------|------------------|---------|
| `epochs` | 200 | 200 | ✅ Consistent |
| `batch_size` | 32 | 32 | ✅ Consistent |
| `learning_rate` | 0.001 | 0.001 | ✅ Consistent |
| `early_stopping_patience` | 20 | 20 | ✅ Consistent |
| `sequence_length` | 120 | 120 | ✅ Consistent |

#### B. Model Architecture Parameters
| Parameter | config.yaml Value | main.py Expected | Status |
|-----------|------------------|------------------|---------|
| `input_size` | 104 | Calculated from features | ⚠️ Dynamic |
| `hidden_size` | 48 | 48 | ✅ Consistent |
| `num_layers` | 3 | 3 | ✅ Consistent |
| `dropout` | 0.35 | 0.35 | ✅ Consistent |
| `num_heads` | 4 | 4 | ✅ Consistent |

#### C. OANDA/FX Configuration Issues

**Critical Issue Found:**
- **config.yaml** has extensive FX configuration under `fx:` section
- **main.py** Buddy commands expect FX configuration under `buddy:` section
- **Missing integration** between FX config and Buddy training

**Specific Problems:**
1. `fx.instrument` vs `buddy.instrument` - Inconsistent naming
2. `fx.granularity` vs `buddy.granularity` - Inconsistent naming  
3. `fx.equity` vs `buddy.equity` - Inconsistent naming
4. `fx.risk_per_trade_pct` vs `buddy.risk_per_trade_pct` - Inconsistent naming

### 4. Configuration Import Analysis

#### main.py Configuration Usage:
- **Line 675**: `cfg = load_config(config_path)` - Loads main config
- **Line 721**: `buddy_cfg = (cfg.get("buddy") or {})` - Extracts buddy section
- **Line 2679**: `cfg = load_config(config_path)` - For FX trading
- **Line 2682**: `policy = fxg.load_fx_policy(cfg)` - Loads FX policy

#### Missing Configuration Sections:
1. **No `data_loader` configuration** - main.py expects data loading config
2. **No `model` section** - main.py expects model architecture config
3. **No `training` section** - main.py expects training workflow config

### 5. Structural Inconsistencies

#### A. Duplicate Configuration Entries
The config.yaml contains multiple conflicting entries:
```yaml
# Multiple sequence_length definitions
sequence_length: 120
sequence_length: 120  # Duplicate

# Multiple data directory definitions
DATA_DIR: trained_data/data
data_dir: trained_data/data
```

#### B. Missing Required Sections
main.py expects these sections that are missing or incomplete:
- `data:` section for data processing
- `model:` section for model architecture
- `training:` section for training workflow

#### C. FX Configuration Structure
The FX configuration is present but not properly integrated:
- `fx:` section exists with comprehensive settings
- But Buddy commands look for FX settings in `buddy:` section
- No clear mapping between FX config and Buddy training

### 6. OANDA Services Integration Issues

#### A. API Configuration
- **config.yaml** has `fx:` section with OANDA settings
- **main.py** Buddy commands use hardcoded defaults or CLI args
- **Missing**: Proper integration of FX config into Buddy training

#### B. Instrument Configuration
```yaml
# config.yaml has:
fx:
  instruments:
  - EUR_USD
  - GBP_USD
  - USD_JPY

# But main.py expects:
buddy:
  instrument: "USD_JPY"  # Single instrument
```

#### C. Risk Management Configuration
- **config.yaml** has detailed risk management under `fx.risk:`
- **main.py** Buddy uses simplified risk configuration
- **Missing**: Integration of FX risk management into Buddy

### 7. Training Workflow Integration Problems

#### A. Data Processing Configuration
main.py expects data processing config but config.yaml lacks:
- `data.sequence_length`
- `data.data_dir` 
- `data.required_features`

#### B. Model Architecture Configuration
main.py expects model config but config.yaml has:
- Inconsistent `model.input_size` (should be dynamic)
- Missing some model parameters

#### C. Training Workflow Configuration
main.py expects training config but config.yaml lacks:
- `training.epochs`
- `training.batch_size`
- `training.learning_rate`

## Recommendations

### 1. Configuration Structure Fix
- Remove duplicate entries
- Reorganize sections to match main.py expectations
- Add missing required sections

### 2. FX Configuration Integration
- Map `fx:` configuration to Buddy training
- Create proper instrument selection logic
- Integrate risk management settings

### 3. OANDA Services Integration
- Properly integrate FX configuration into Buddy commands
- Add OANDA API key management
- Implement proper instrument and granularity handling

### 4. Training Workflow Integration
- Add complete data processing configuration
- Ensure model architecture configuration matches expectations
- Add complete training workflow configuration

## Conclusion

The main issues preventing proper training workflow integration are:

1. **Structural inconsistencies** in configuration organization
2. **Missing required configuration sections** that main.py expects
3. **FX configuration not properly integrated** with Buddy training
4. **OANDA services configuration** exists but isn't used by main.py
5. **Duplicate and conflicting configuration entries**

The project has comprehensive FX trading capabilities but the configuration system needs to be reorganized to properly support the Buddy training workflow and OANDA integration.