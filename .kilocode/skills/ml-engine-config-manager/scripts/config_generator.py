#!/usr/bin/env python3
"""
Configuration Generator for ML Engine.

Generate configuration files from templates with best practices.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION TEMPLATES
# =============================================================================

DEFAULT_CONFIG = {
    'buddy': {
        'fetch_days': 365,
        'fetch_candles': 3000,
        'instrument': 'EUR_USD',
    },
    'training': {
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'patience': 20,
        'seq_len': 60,
        'use_class_balanced_loss': True,
        'use_anti_collapse_loss': True,
        'use_ema': True,
        'use_ewc': True,
        'warm_start_lr_factor': 0.1,
    },
    'model': {
        'transformer_d_model': 32,
        'transformer_num_heads': 4,
        'transformer_num_layers': 2,
        'transformer_dff': 64,
        'transformer_dropout': 0.2,
    },
    'data': {
        'use_normalized_features': True,
        'feature_engineering': {
            'adx_period': 14,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
        },
    },
    'fx': {
        'session_windows': {
            'london': ['08:00', '11:30'],
            'new_york': ['13:00', '17:00'],
        },
        'daily_circuit_breaker': {
            'enabled': True,
            'max_loss_pips': 100.0,
        },
        'spread_filter': {
            'enabled': True,
            'max_spread_pips': 3.0,
        },
        'paper_trading': True,
    },
}

PAIR_SPECIFIC_CONFIGS = {
    'EUR_USD': {
        'training': {
            'epochs': 100,
            'batch_size': 64,
        },
        'fx': {
            'daily_circuit_breaker': {'max_loss_pips': 50.0},
            'spread_filter': {'max_spread_pips': 1.5},
        },
    },
    'GBP_USD': {
        'training': {
            'epochs': 120,
            'batch_size': 64,
        },
        'fx': {
            'daily_circuit_breaker': {'max_loss_pips': 80.0},
            'spread_filter': {'max_spread_pips': 2.0},
        },
    },
    'USD_JPY': {
        'training': {
            'epochs': 150,
            'batch_size': 32,
        },
        'fx': {
            'daily_circuit_breaker': {'max_loss_pips': 100.0},
            'spread_filter': {'max_spread_pips': 3.0},
        },
    },
    'GBP_JPY': {
        'training': {
            'epochs': 150,
            'batch_size': 32,
        },
        'fx': {
            'daily_circuit_breaker': {'max_loss_pips': 120.0},
            'spread_filter': {'max_spread_pips': 4.0},
        },
    },
}

VOLATILITY_TIER_CONFIGS = {
    'volatile': {
        'pairs': ['GBP_JPY', 'GBP_AUD', 'GBP_NZD', 'EUR_NZD', 'AUD_JPY', 'NZD_JPY', 'CAD_JPY'],
        'config': {
            'training': {
                'epochs': 150,
                'patience': 25,
            },
            'model': {
                'transformer_dropout': 0.3,
            },
        },
    },
    'stable': {
        'pairs': ['EUR_USD', 'USD_CHF', 'EUR_CHF', 'EUR_GBP', 'USD_SGD'],
        'config': {
            'training': {
                'epochs': 80,
                'patience': 15,
            },
            'model': {
                'transformer_dropout': 0.15,
            },
        },
    },
}

PLATFORM_CONFIGS = {
    'm1_metal': {
        'training': {
            'batch_size': 32,
            'use_mixed_precision': True,
        },
        'model': {
            'disable_recurrent_dropout': True,
            'use_tcn_instead_of_lstm': True,
        },
    },
    'gpu': {
        'training': {
            'batch_size': 128,
            'use_mixed_precision': True,
        },
    },
    'cpu': {
        'training': {
            'batch_size': 32,
            'use_mixed_precision': False,
        },
    },
}


# =============================================================================
# CONFIG GENERATOR
# =============================================================================

class ConfigGenerator:
    """Configuration file generator."""
    
    def __init__(self):
        self.templates = {
            'default': DEFAULT_CONFIG,
            'pair_specific': PAIR_SPECIFIC_CONFIGS,
            'volatility': VOLATILITY_TIER_CONFIGS,
            'platform': PLATFORM_CONFIGS,
        }
    
    def generate(
        self,
        template: str = 'default',
        pair: Optional[str] = None,
        volatility: Optional[str] = None,
        platform: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate configuration from template.
        
        Args:
            template: Template name ('default', 'pair_specific', etc.)
            pair: Currency pair (e.g., 'EUR_USD')
            volatility: Volatility tier ('volatile', 'stable')
            platform: Platform type ('m1_metal', 'gpu', 'cpu')
            overrides: Custom overrides to apply
            
        Returns:
            Generated configuration dictionary
        """
        # Start with default config
        config = self._deep_copy(DEFAULT_CONFIG)
        
        # Apply template-specific config
        if template == 'pair_specific' and pair:
            pair_config = PAIR_SPECIFIC_CONFIGS.get(pair, {})
            config = self._merge_configs(config, pair_config)
        elif template == 'volatility' and volatility:
            vol_config = VOLATILITY_TIER_CONFIGS.get(volatility, {})
            config = self._merge_configs(config, vol_config.get('config', {}))
        elif template == 'platform' and platform:
            plat_config = PLATFORM_CONFIGS.get(platform, {})
            config = self._merge_configs(config, plat_config)
        
        # Apply platform config if specified
        if platform:
            plat_config = PLATFORM_CONFIGS.get(platform, {})
            config = self._merge_configs(config, plat_config)
        
        # Apply volatility config if specified
        if volatility:
            vol_config = VOLATILITY_TIER_CONFIGS.get(volatility, {})
            config = self._merge_configs(config, vol_config.get('config', {}))
        
        # Apply pair-specific config if specified
        if pair:
            pair_config = PAIR_SPECIFIC_CONFIGS.get(pair, {})
            config = self._merge_configs(config, pair_config)
        
        # Apply custom overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        # Add metadata
        config['_metadata'] = {
            'generated_at': datetime.utcnow().isoformat(),
            'template': template,
            'pair': pair,
            'volatility': volatility,
            'platform': platform,
        }
        
        return config
    
    def _deep_copy(self, d: Dict) -> Dict:
        """Deep copy dictionary."""
        import copy
        return copy.deepcopy(d)
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two configuration dictionaries."""
        result = self._deep_copy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = self._deep_copy(value)
        
        return result
    
    def _apply_overrides(self, config: Dict, overrides: Dict[str, Any]) -> Dict:
        """Apply dot-notation overrides to config."""
        result = self._deep_copy(config)
        
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            current = result
            
            # Navigate to parent of target
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
                if not isinstance(current, dict):
                    current[key] = {}
                    current = current[key]
                elif not isinstance(current, dict):
                    # Can't override non-dict with nested path
                    logger.warning(f"Cannot override {key_path}: {keys[:-1]} is not a dict")
                    continue
            
            # Set final value
            final_key = keys[-1]
            current[final_key] = value
        
        return result
    
    def save(self, config: Dict, output_path: Path) -> None:
        """Save configuration to YAML file."""
        # Remove metadata before saving
        config_to_save = {k: v for k, v in config.items() if k != '_metadata'}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def list_templates(self) -> None:
        """List available templates."""
        print("\nAvailable Templates:")
        print("=" * 50)
        
        print("\n1. Default Template")
        print("   - Standard configuration for most use cases")
        
        print("\n2. Pair-Specific Templates")
        for pair in sorted(PAIR_SPECIFIC_CONFIGS.keys()):
            print(f"   - {pair}")
        
        print("\n3. Volatility Tier Templates")
        for tier in sorted(VOLATILITY_TIER_CONFIGS.keys()):
            pairs = VOLATILITY_TIER_CONFIGS[tier].get('pairs', [])
            print(f"   - {tier}: {', '.join(pairs[:3])}{'...' if len(pairs) > 3 else ''}")
        
        print("\n4. Platform Templates")
        for platform in sorted(PLATFORM_CONFIGS.keys()):
            print(f"   - {platform}")
        
        print("\n" + "=" * 50)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ML Engine configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default configuration
  %(prog)s generate --output config/my_config.yaml
  
  # Generate for specific pair
  %(prog)s generate --pair EUR_USD --output config/eur_usd.yaml
  
  # Generate with custom settings
  %(prog)s generate \\
    --pair GBP_JPY \\
    --platform m1_metal \\
    --overrides '{"training.epochs": 200, "training.batch_size": 128}' \\
    --output config/custom.yaml
  
  # List available templates
  %(prog)s list-templates
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate configuration file')
    generate_parser.add_argument('--template', '-t', default='default',
                             choices=['default', 'pair_specific', 'volatility', 'platform'],
                             help='Template to use')
    generate_parser.add_argument('--pair', '-p', help='Currency pair (e.g., EUR_USD)')
    generate_parser.add_argument('--volatility', '-v',
                             choices=['volatile', 'stable'],
                             help='Volatility tier')
    generate_parser.add_argument('--platform',
                             choices=['m1_metal', 'gpu', 'cpu'],
                             help='Platform type')
    generate_parser.add_argument('--overrides', '-o',
                             help='Custom overrides (JSON format)')
    generate_parser.add_argument('--seq-len', type=int, help='Sequence length')
    generate_parser.add_argument('--epochs', type=int, help='Training epochs')
    generate_parser.add_argument('--batch-size', type=int, help='Batch size')
    generate_parser.add_argument('--output', required=True, type=Path,
                             help='Output configuration file path')
    
    # List templates command
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize generator
    generator = ConfigGenerator()
    
    try:
        if args.command == 'generate':
            # Build overrides from args
            overrides = {}
            if args.overrides:
                import json
                overrides = json.loads(args.overrides)
            if args.seq_len:
                overrides['training.seq_len'] = args.seq_len
            if args.epochs:
                overrides['training.epochs'] = args.epochs
            if args.batch_size:
                overrides['training.batch_size'] = args.batch_size
            
            # Generate config
            config = generator.generate(
                template=args.template,
                pair=args.pair,
                volatility=args.volatility,
                platform=args.platform,
                overrides=overrides,
            )
            
            # Save config
            generator.save(config, args.output)
            print(f"✓ Configuration generated: {args.output}")
            
            # Show summary
            metadata = config.get('_metadata', {})
            print(f"  Template: {metadata.get('template', 'default')}")
            print(f"  Pair: {metadata.get('pair', 'N/A')}")
            print(f"  Platform: {metadata.get('platform', 'N/A')}")
        
        elif args.command == 'list-templates':
            generator.list_templates()
    
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        sys.exit(130)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in overrides: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
