---
name: ML Engine Configuration Manager
version: 1.0.0
author: Kilo Code
description: Centralized configuration management with validation, versioning, and environment-specific overrides for the ML Engine FX trading system.
tags: [configuration, validation, yaml, environment, versioning]
category: infrastructure
dependencies: []
---

# ML Engine Configuration Manager

Centralized configuration management system for the ML Engine FX trading ecosystem with validation, environment-specific overrides, versioning, and automated migration.

## Overview

The Configuration Manager skill provides robust configuration handling for the ML Engine's complex multi-model ensemble system. It addresses configuration drift, validation gaps, and deployment complexity across different environments (development, testing, production).

### Key Capabilities

- **Schema Validation**: Pydantic-based validation with clear error messages
- **Environment Overrides**: Development, testing, production configurations
- **Version Management**: Configuration versioning with migration support
- **Automated Generation**: Create configs from templates and best practices
- **Diff & Merge**: Compare and merge configurations intelligently
- **Backup & Rollback**: Automatic backups with rollback capability

### Use Cases

1. **Validate Configuration**: Ensure config files conform to schema before training
2. **Generate Config**: Create new config from template with best practices
3. **Migrate Config**: Upgrade config between versions safely
4. **Override Config**: Apply environment-specific settings
5. **Compare Configs**: Visual diff between two configurations
6. **Rollback Config**: Revert to previous configuration version

## Quick Start

### Validate Existing Configuration

```bash
# Validate a configuration file
python -m kilocode.skills.ml_engine_config_manager validate config/config_improved_H1.yaml

# Validate with detailed output
python -m kilocode.skills.ml_engine_config_manager validate config/config_improved_H1.yaml --verbose

# Check for deprecated fields
python -m kilocode.skills.ml_engine_config_manager validate config/config_improved_H1.yaml --check-deprecated
```

### Generate New Configuration

```bash
# Generate from default template
python -m kilocode.skills.ml_engine_config_manager generate --output config/my_config.yaml

# Generate for specific pair
python -m kilocode.skills.ml_engine_config_manager generate --pair EUR_USD --output config/eur_usd_config.yaml

# Generate with custom settings
python -m kilocode.skills.ml_engine_config_manager generate \
  --pair GBP_JPY \
  --seq-len 120 \
  --epochs 200 \
  --output config/gbp_jpy_config.yaml
```

### Apply Environment Overrides

```bash
# Apply production overrides
python -m kilocode.skills.ml_engine_config_manager override \
  --base config/config_improved_H1.yaml \
  --env production \
  --output config/production_config.yaml

# Apply custom overrides
python -m kilocode.skills.ml_engine_config_manager override \
  --base config/config_improved_H1.yaml \
  --env custom \
  --overrides '{"training.epochs": 200, "training.batch_size": 128}' \
  --output config/custom_config.yaml
```

### Compare Configurations

```bash
# Visual diff between two configs
python -m kilocode.skills.ml_engine_config_manager diff \
  config/config_improved_H1.yaml \
  config/config_m1_optimized.yaml

# Diff with color output
python -m kilocode.skills.ml_engine_config_manager diff \
  config/config_improved_H1.yaml \
  config/config_m1_optimized.yaml \
  --color

# Diff with structural comparison (ignoring values)
python -m kilocode.skills.ml_engine_config_manager diff \
  config/config_improved_H1.yaml \
  config/config_m1_optimized.yaml \
  --structure-only
```

### Migrate Configuration

```bash
# Migrate to latest schema version
python -m kilocode.skills.ml_engine_config_manager migrate \
  config/legacy_config.yaml \
  --output config/migrated_config.yaml

# Migrate with dry-run (preview changes)
python -m kilocode.skills.ml_engine_config_manager migrate \
  config/legacy_config.yaml \
  --output config/migrated_config.yaml \
  --dry-run

# Migrate with backup
python -m kilocode.skills.ml_engine_config_manager migrate \
  config/legacy_config.yaml \
  --output config/migrated_config.yaml \
  --backup
```

## Configuration Schema

The ML Engine configuration follows a hierarchical structure with these main sections:

### buddy
CLI and command interface settings
```yaml
buddy:
  fetch_days: 365  # Days of historical data to fetch
  fetch_candles: 3000  # Number of candles for training
  instrument: "EUR_USD"  # Default trading pair
```

### training
Training pipeline configuration
```yaml
training:
  epochs: 100  # Training epochs
  batch_size: 64  # Batch size
  learning_rate: 0.001  # Learning rate
  patience: 20  # Early stopping patience
  seq_len: 60  # Sequence length for sliding window
  use_class_balanced_loss: true  # Use class-balanced loss
  use_anti_collapse_loss: true  # Use anti-collapse loss
```

### model
Model architecture settings
```yaml
model:
  transformer_d_model: 32  # Transformer model dimension
  transformer_num_heads: 4  # Number of attention heads
  transformer_num_layers: 2  # Number of encoder layers
  transformer_dff: 64  # Feedforward network dimension
  transformer_dropout: 0.2  # Dropout rate
```

### data
Data processing and feature engineering
```yaml
data:
  use_normalized_features: true  # Use normalized features
  feature_engineering:  # Feature engineering settings
    adx_period: 14
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
```

### fx
FX trading settings and guardrails
```yaml
fx:
  session_windows:  # Trading session windows
    london: ["08:00", "11:30"]
    new_york: ["13:00", "17:00"]
  daily_circuit_breaker:  # Daily loss limit
    enabled: true
    max_loss_pips: 100
  spread_filter:  # Spread filter
    enabled: true
    max_spread_pips: 3.0
```

## Environment-Specific Overrides

### Development
```yaml
# config/overrides/dev.yaml
training:
  epochs: 10  # Quick iteration
  batch_size: 32  # Smaller batches
model:
  transformer_d_model: 16  # Smaller model
fx:
  paper_trading: true  # Always paper trade in dev
```

### Testing
```yaml
# config/overrides/test.yaml
training:
  epochs: 20
  use_ema: false  # Disable for speed
  use_ewc: false  # Disable for speed
data:
  fetch_days: 90  # Smaller dataset
```

### Production
```yaml
# config/overrides/prod.yaml
training:
  epochs: 100
  patience: 20
  use_class_balanced_loss: true
  use_anti_collapse_loss: true
  use_ema: true
  use_ewc: true
fx:
  paper_trading: false
  session_windows:
    london: ["08:00", "11:30"]
    new_york: ["13:00", "17:00"]
  daily_circuit_breaker:
    enabled: true
    max_loss_pips: 100
```

## Advanced Usage

### Custom Validation Rules

Create custom validation rules in `references/custom_validators.py`:

```python
from pydantic import validator

class CustomConfig(BaseModel):
    # Custom validator for sequence length
    @validator('seq_len')
    def validate_seq_len(cls, v):
        if v < 30:
            raise ValueError('seq_len must be at least 30')
        if v > 200:
            raise ValueError('seq_len cannot exceed 200')
        if v % 10 != 0:
            raise ValueError('seq_len must be multiple of 10')
        return v
```

### Configuration Templates

Use templates in `references/config_templates/` for common scenarios:

```bash
# List available templates
python -m kilocode.skills.ml_engine_config_manager list-templates

# Generate from template
python -m kilocode.skills.ml_engine_config_manager generate \
  --template aggressive \
  --pair GBP_JPY \
  --output config/aggressive_gbp_jpy.yaml
```

### Configuration Diff Analysis

```bash
# Analyze differences between configs
python -m kilocode.skills.ml_engine_config_manager analyze-diff \
  config/config_improved_H1.yaml \
  config/config_m1_optimized.yaml \
  --output reports/config_diff_analysis.md

# Include impact analysis
python -m kilocode.skills.ml_engine_config_manager analyze-diff \
  config/config_improved_H1.yaml \
  config/config_m1_optimized.yaml \
  --impact \
  --output reports/config_impact_analysis.md
```

### Backup and Restore

```bash
# Create backup
python -m kilocode.skills.ml_engine_config_manager backup \
  config/config_improved_H1.yaml \
  --output backups/config_backup_20240129.yaml

# List backups
python -m kilocode.skills.ml_engine_config_manager list-backups

# Restore from backup
python -m kilocode.skills.ml_engine_config_manager restore \
  backups/config_backup_20240129.yaml \
  --output config/config_improved_H1.yaml
```

## Error Handling

The Configuration Manager provides clear error messages for common issues:

### Validation Errors

```
ValidationError: training.epochs must be at least 10
ValidationError: model.transformer_d_model must be multiple of 8
ValidationError: fx.session_windows.london must be in format HH:MM
```

### Migration Errors

```
MigrationError: Cannot migrate from version 1.0 to 3.0 (unsupported)
MigrationError: Field 'legacy_field' is deprecated and has no migration path
```

### Override Errors

```
OverrideError: Override path 'training.invalid_field' does not exist in base config
OverrideError: Override type mismatch: expected int, got str
```

## Best Practices

1. **Always validate before training**: Run validation before starting long training jobs
2. **Use version control**: Commit configuration changes with descriptive messages
3. **Environment separation**: Keep dev/test/prod configs separate
4. **Document changes**: Add comments explaining why values were changed
5. **Test migrations**: Dry-run migrations before applying to production
6. **Backup before changes**: Always create backup before modifying production config
7. **Use templates**: Start from templates rather than from scratch
8. **Review diffs**: Always review diff before applying changes

## Troubleshooting

### Issue: Validation fails with unclear error

**Solution**: Use `--verbose` flag to see detailed validation context
```bash
python -m kilocode.skills.ml_engine_config_manager validate config.yaml --verbose
```

### Issue: Migration changes unexpected values

**Solution**: Use `--dry-run` to preview migration changes
```bash
python -m kilocode.skills.ml_engine_config_manager migrate old.yaml --dry-run
```

### Issue: Override not applied

**Solution**: Check override path matches base config structure exactly
```bash
# Check structure first
python -m kilocode.skills.ml_engine_config_manager show-structure config.yaml
```

## Integration with ML Engine

The Configuration Manager integrates seamlessly with ML Engine:

```python
# In main.py
from kilocode.skills.ml_engine_config_manager import ConfigManager

# Load and validate config
config = ConfigManager.load('config/config_improved_H1.yaml')

# Access config with type hints
epochs = config.training.epochs
seq_len = config.data.seq_len

# Apply environment overrides
config = ConfigManager.apply_overrides(
    config,
    environment='production'
)

# Validate before training
errors = ConfigManager.validate(config)
if errors:
    print(f"Configuration errors: {errors}")
    sys.exit(1)
```

## Bundled Resources

### Scripts

- **scripts/config_validator.py**: Schema validation engine
- **scripts/config_migrator.py**: Version migration tool
- **scripts/config_generator.py**: Automated config generation
- **scripts/config_diff.py**: Diff and comparison tool
- **scripts/config_backup.py**: Backup and restore utilities

### References

- **references/config_schema.json**: Complete schema definition
- **references/config_templates/**: Common configuration templates
- **references/migration_guide.md**: Migration procedures
- **references/best_practices.md**: Configuration best practices

### Assets

- **assets/config_examples/**: Example configurations for various scenarios
- **assets/validation_rules/**: Custom validation rule examples

## Version History

- **1.0.0** (2024-01-29): Initial release with validation, versioning, and environment overrides
