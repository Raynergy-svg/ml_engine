#!/usr/bin/env python3
"""
Configuration Validator for ML Engine.

Validates YAML configuration files against Pydantic schema with detailed error reporting.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ValidationError, validator, Field

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC SCHEMA
# =============================================================================

class BuddyConfig(BaseModel):
    """Buddy CLI configuration."""
    fetch_days: int = Field(default=365, ge=1, le=1000)
    fetch_candles: int = Field(default=3000, ge=100, le=10000)
    instrument: str = Field(default="EUR_USD", regex=r"^[A-Z]{3}_[A-Z]{3}$")


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=0.001, ge=1e-6, le=1.0)
    patience: int = Field(default=20, ge=1, le=100)
    seq_len: int = Field(default=60, ge=10, le=200)
    
    @validator('seq_len')
    def validate_seq_len(cls, v):
        if v % 10 != 0:
            raise ValueError('seq_len must be a multiple of 10')
        return v
    
    use_class_balanced_loss: bool = Field(default=True)
    use_anti_collapse_loss: bool = Field(default=True)
    use_ema: bool = Field(default=True)
    use_ewc: bool = Field(default=True)
    warm_start_lr_factor: float = Field(default=0.1, ge=0.01, le=1.0)


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    transformer_d_model: int = Field(default=32, ge=8, le=512)
    transformer_num_heads: int = Field(default=4, ge=1, le=16)
    transformer_num_layers: int = Field(default=2, ge=1, le=8)
    transformer_dff: int = Field(default=64, ge=16, le=1024)
    transformer_dropout: float = Field(default=0.2, ge=0.0, le=0.8)
    
    @validator('transformer_d_model')
    def validate_d_model(cls, v):
        if v % 8 != 0:
            raise ValueError('transformer_d_model must be a multiple of 8')
        return v
    
    @validator('transformer_num_heads')
    def validate_num_heads(cls, v, values):
        d_model = values.get('transformer_d_model', 32)
        if d_model % v != 0:
            raise ValueError(f'transformer_num_heads ({v}) must divide transformer_d_model ({d_model}) evenly')
        return v


class DataConfig(BaseModel):
    """Data processing configuration."""
    use_normalized_features: bool = Field(default=True)
    
    feature_engineering: Dict[str, Any] = Field(default_factory=lambda: {
        'adx_period': 14,
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
    })


class SessionWindow(BaseModel):
    """Trading session window."""
    start: str = Field(..., regex=r"^\d{2}:\d{2}$")
    end: str = Field(..., regex=r"^\d{2}:\d{2}$")


class DailyCircuitBreaker(BaseModel):
    """Daily circuit breaker configuration."""
    enabled: bool = Field(default=True)
    max_loss_pips: float = Field(default=100.0, ge=0.0, le=1000.0)


class SpreadFilter(BaseModel):
    """Spread filter configuration."""
    enabled: bool = Field(default=True)
    max_spread_pips: float = Field(default=3.0, ge=0.0, le=20.0)


class FXConfig(BaseModel):
    """FX trading configuration."""
    session_windows: Dict[str, List[str]] = Field(default_factory=lambda: {
        'london': ['08:00', '11:30'],
        'new_york': ['13:00', '17:00'],
    })
    
    daily_circuit_breaker: DailyCircuitBreaker = Field(default_factory=DailyCircuitBreaker)
    spread_filter: SpreadFilter = Field(default_factory=SpreadFilter)
    paper_trading: bool = Field(default=True)


class MLEngineConfig(BaseModel):
    """Complete ML Engine configuration schema."""
    buddy: BuddyConfig = Field(default_factory=BuddyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    fx: FXConfig = Field(default_factory=FXConfig)


# =============================================================================
# VALIDATOR ENGINE
# =============================================================================

class ConfigValidator:
    """Configuration validation engine."""
    
    def __init__(self, schema_path: Optional[Path] = None):
        self.schema_path = schema_path
        self.schema = MLEngineConfig
        self.deprecated_fields = {
            'tcn_num_layers': 'Use model.transformer_num_layers instead',
            'tcn_hidden_size': 'Use model.transformer_d_model instead',
            'use_gpu': 'Removed - use environment variables instead',
        }
    
    def load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {config_path}: {e}")
            raise
    
    def validate(
        self,
        config_path: Path,
        check_deprecated: bool = False,
        verbose: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration file against schema.
        
        Args:
            config_path: Path to configuration file
            check_deprecated: Check for deprecated fields
            verbose: Show detailed validation context
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Load YAML
        try:
            config_dict = self.load_yaml(config_path)
        except Exception as e:
            return False, [f"Failed to load config: {e}"]
        
        # Check for deprecated fields
        if check_deprecated:
            deprecated_found = self._check_deprecated_fields(config_dict)
            if deprecated_found:
                errors.extend(deprecated_found)
        
        # Validate against Pydantic schema
        try:
            self.schema(**config_dict)
            if verbose:
                logger.info(f"✓ Configuration {config_path} is valid")
        except ValidationError as e:
            for error in e.errors():
                error_msg = self._format_validation_error(error)
                errors.append(error_msg)
                if verbose:
                    logger.error(f"✗ {error_msg}")
        
        return len(errors) == 0, errors
    
    def _check_deprecated_fields(self, config_dict: Dict[str, Any]) -> List[str]:
        """Check for deprecated configuration fields."""
        deprecated_errors = []
        
        def check_dict(d: Dict, path: str = ""):
            for key, value in d.items():
                full_path = f"{path}.{key}" if path else key
                if key in self.deprecated_fields:
                    deprecated_errors.append(
                        f"Deprecated field '{full_path}': {self.deprecated_fields[key]}"
                    )
                if isinstance(value, dict):
                    check_dict(value, full_path)
        
        check_dict(config_dict)
        return deprecated_errors
    
    def _format_validation_error(self, error: Dict[str, Any]) -> str:
        """Format Pydantic validation error for readability."""
        loc = " -> ".join(str(x) for x in error['loc'])
        msg = error['msg']
        typ = error['type']
        
        if typ == 'value_error.number.not_gt':
            return f"{loc}: {msg} (must be > {error['ctx']['limit_value']})"
        elif typ == 'value_error.number.not_lt':
            return f"{loc}: {msg} (must be < {error['ctx']['limit_value']})"
        elif typ == 'value_error.missing':
            return f"{loc}: Required field is missing"
        else:
            return f"{loc}: {msg}"
    
    def show_structure(self, config_path: Path) -> None:
        """Display configuration structure (keys only)."""
        config_dict = self.load_yaml(config_path)
        
        def print_structure(d: Dict, indent: int = 0):
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    print(f"{prefix}{key}/")
                    print_structure(value, indent + 1)
                else:
                    value_preview = str(value)[:50]
                    print(f"{prefix}{key}: {value_preview}")
        
        print(f"\nConfiguration structure for {config_path}:")
        print_structure(config_dict)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate ML Engine configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  %(prog)s validate config/config_improved_H1.yaml
  
  # Validate with detailed output
  %(prog)s validate config/config_improved_H1.yaml --verbose
  
  # Check for deprecated fields
  %(prog)s validate config/config_improved_H1.yaml --check-deprecated
  
  # Show configuration structure
  %(prog)s show-structure config/config_improved_H1.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config_path', type=Path, help='Path to configuration file')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    validate_parser.add_argument('--check-deprecated', '-d', action='store_true', help='Check for deprecated fields')
    
    # Show structure command
    structure_parser = subparsers.add_parser('show-structure', help='Show configuration structure')
    structure_parser.add_argument('config_path', type=Path, help='Path to configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize validator
    validator = ConfigValidator()
    
    try:
        if args.command == 'validate':
            is_valid, errors = validator.validate(
                args.config_path,
                check_deprecated=args.check_deprecated,
                verbose=args.verbose
            )
            
            if is_valid:
                print(f"✓ Configuration {args.config_path} is valid")
                sys.exit(0)
            else:
                print(f"✗ Configuration {args.config_path} has {len(errors)} error(s):")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
        
        elif args.command == 'show-structure':
            validator.show_structure(args.config_path)
    
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
