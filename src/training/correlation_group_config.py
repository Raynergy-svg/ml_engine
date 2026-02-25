"""
Correlation Group Configuration for FX Pairs.

This module maps currency pairs to their correlation groups and provides
optimized parameters from the best-performing "master" pair in each group.

When training or loading a model for inference, the system uses the master
pair's optimized parameters, which transfer well to correlated pairs.

Usage:
    from src.training.correlation_group_config import (
        get_correlation_group,
        get_master_pair,
        get_best_params_for_pair,
        get_transfer_params_for_pair,
    )
    
    # Get best params for any pair (uses master pair's params)
    params = get_best_params_for_pair('EUR_JPY')
    
    # Check if a pair should use transfer learning
    master = get_master_pair('AUD_NZD')  # Returns 'AUD_NZD' (it's the master)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class CorrelationGroup:
    """Defines a correlation group of currency pairs."""
    name: str
    pairs: List[str]
    master_pair: str
    avg_correlation: str  # e.g., "0.75-0.90"
    sync_reason: str
    master_justification: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# CORRELATION GROUPS DEFINITION
# =============================================================================

# Best parameters from wandb sweep results (entity: tencylinder8310-smartdebtflow-com, project: ml_engine_fx)
# These are the optimal TCN parameters for each master pair

GBP_JPY_BEST_PARAMS = {
    # Architecture
    'batch_size': 128,
    'seq_len': 120,
    'tcn_nb_filters': 256,
    'tcn_kernel_size': 3,
    'tcn_dilations': [1, 2, 4, 8, 16, 32],
    
    # Regularization
    'dropout_rate': 0.236,
    'l2_reg': 0.000149,
    
    # Optimizer
    'lr': 0.000508,
    
    # Loss weights
    'direction_weight': 0.491,
    'confidence_weight': 0.276,
    'volatility_weight': 0.175,
    'combined_w_dir': 0.7,
    
    # Training
    'epochs': 200,
    'patience': 10,
    'es_monitor': 'direction',
    
    # Features
    'top_features': 150,
    'train_smoothing': False,
    'mixed_precision': True,
    
    # Model info
    'model_type': 'tcn',
    'granularity': 'H1',
}

EUR_GBP_BEST_PARAMS = {
    # Architecture
    'batch_size': 128,
    'seq_len': 90,
    'tcn_nb_filters': 256,
    'tcn_kernel_size': 5,
    'tcn_dilations': [1, 2, 4, 8, 16, 32],
    
    # Regularization
    'dropout_rate': 0.363,
    'l2_reg': 0.000098,
    
    # Optimizer
    'lr': 0.000415,
    
    # Loss weights
    'direction_weight': 0.493,
    'confidence_weight': 0.238,
    'volatility_weight': 0.297,
    'combined_w_dir': 0.7,
    
    # Training
    'epochs': 200,
    'patience': 25,
    'es_monitor': 'direction',
    
    # Features
    'top_features': 150,
    'train_smoothing': True,
    'mixed_precision': False,
    
    # Model info
    'model_type': 'tcn',
    'granularity': 'H1',
}

AUD_USD_BEST_PARAMS = {
    # Architecture
    'batch_size': 128,
    'seq_len': 90,
    'tcn_nb_filters': 64,
    'tcn_kernel_size': 5,
    'tcn_dilations': [1, 2, 4, 8, 16, 32],
    
    # Regularization
    'dropout_rate': 0.183,
    'l2_reg': 0.000405,
    
    # Optimizer
    'lr': 0.000630,
    
    # Loss weights
    'direction_weight': 0.530,
    'confidence_weight': 0.346,
    'volatility_weight': 0.145,
    'combined_w_dir': 0.7,
    
    # Training
    'epochs': 200,
    'patience': 20,
    'es_monitor': 'direction',
    
    # Features
    'top_features': None,  # Uses all features
    'train_smoothing': True,
    'mixed_precision': True,
    
    # Model info
    'model_type': 'tcn',
    'granularity': 'H1',
}

USD_CAD_BEST_PARAMS = {
    # Architecture
    'batch_size': 128,
    'seq_len': 120,
    'tcn_nb_filters': 128,
    'tcn_kernel_size': 5,
    'tcn_dilations': [1, 2, 4, 8, 16],
    
    # Regularization
    'dropout_rate': 0.468,
    'l2_reg': 0.000013,
    
    # Optimizer
    'lr': 0.000552,
    
    # Loss weights
    'direction_weight': 0.435,
    'confidence_weight': 0.325,
    'volatility_weight': 0.264,
    'combined_w_dir': 0.7,
    
    # Training
    'epochs': 200,
    'patience': 25,
    'es_monitor': 'direction',
    
    # Features
    'top_features': 150,
    'train_smoothing': True,
    'mixed_precision': False,
    
    # Model info
    'model_type': 'tcn',
    'granularity': 'H1',
}

AUD_NZD_BEST_PARAMS = {
    # Architecture
    'batch_size': 128,
    'seq_len': 90,
    'tcn_nb_filters': 256,
    'tcn_kernel_size': 7,
    'tcn_dilations': [1, 2, 4, 8, 16],
    
    # Regularization
    'dropout_rate': 0.420,
    'l2_reg': 0.000208,
    
    # Optimizer
    'lr': 0.000376,
    
    # Loss weights
    'direction_weight': 0.764,
    'confidence_weight': 0.117,
    'volatility_weight': 0.101,
    'combined_w_dir': 0.7,
    
    # Training
    'epochs': 200,
    'patience': 20,
    'es_monitor': 'direction',
    
    # Features
    'top_features': 100,
    'train_smoothing': True,
    'mixed_precision': True,
    
    # Model info
    'model_type': 'tcn',
    'granularity': 'H1',
}

# Default params for pairs not in sweeps (use EUR/GBP as base)
DEFAULT_PARAMS = EUR_GBP_BEST_PARAMS.copy()


# =============================================================================
# CORRELATION GROUPS REGISTRY
# =============================================================================

CORRELATION_GROUPS: Dict[str, CorrelationGroup] = {
    'yen_carry': CorrelationGroup(
        name='Yen Carry / Risk-On',
        pairs=['EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'NZD_JPY', 'USD_JPY'],
        master_pair='GBP_JPY',
        avg_correlation='0.75-0.90',
        sync_reason='Yen weakens on risk-on, strengthens on fear',
        master_justification='Best combined score in group (0.5656), 66.55% direction accuracy',
        best_params=GBP_JPY_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5656,
            'val_direction_accuracy': 0.6655,
            'val_direction_f1': 0.6724,
            'val_loss': 0.3222,
        }
    ),
    'euro_basket': CorrelationGroup(
        name='Euro Basket',
        pairs=['EUR_USD', 'EUR_JPY', 'EUR_GBP', 'EUR_AUD'],
        master_pair='EUR_GBP',  # Best available from sweeps
        avg_correlation='0.85-0.92',
        sync_reason='Euro strength/weakness drives them all',
        master_justification='Best combined score (0.5724), 66.58% direction accuracy',
        best_params=EUR_GBP_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5724,
            'val_direction_accuracy': 0.6658,
            'val_direction_f1': 0.6627,
            'val_loss': 0.4332,
        }
    ),
    'sterling_cluster': CorrelationGroup(
        name='Sterling Cluster',
        pairs=['GBP_USD', 'GBP_JPY', 'GBP_AUD', 'EUR_GBP'],
        master_pair='EUR_GBP',  # Best performer in group
        avg_correlation='0.78-0.88',
        sync_reason='UK data + Brexit noise, but tracks USD & yen',
        master_justification='Best combined score (0.5724), most stable',
        best_params=EUR_GBP_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5724,
            'val_direction_accuracy': 0.6658,
            'val_direction_f1': 0.6627,
            'val_loss': 0.4332,
        }
    ),
    'aussie_kiwi_commodity': CorrelationGroup(
        name='Aussie / Kiwi Commodity',
        pairs=['AUD_USD', 'NZD_USD', 'AUD_JPY', 'NZD_JPY'],
        master_pair='AUD_USD',
        avg_correlation='0.80-0.90',
        sync_reason='Commodity prices + China growth',
        master_justification='More volume than NZD, best commodity proxy',
        best_params=AUD_USD_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5683,
            'val_direction_accuracy': 0.6569,
            'val_direction_f1': 0.6642,
            'val_loss': 0.3365,
        }
    ),
    'usd_safe_haven': CorrelationGroup(
        name='USD Safe-Haven',
        pairs=['USD_JPY', 'USD_CHF', 'USD_CAD'],
        master_pair='USD_CAD',
        avg_correlation='0.70-0.85',
        sync_reason='USD strength vs risk-off (yen, franc)',
        master_justification='Best combined score (0.5725), 66.37% direction accuracy',
        best_params=USD_CAD_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5725,
            'val_direction_accuracy': 0.6637,
            'val_direction_f1': 0.6574,
            'val_loss': 0.3989,
        }
    ),
    'crosses': CorrelationGroup(
        name='Crosses (Less Common)',
        pairs=['GBP_AUD', 'EUR_AUD', 'AUD_NZD'],
        master_pair='AUD_NZD',
        avg_correlation='0.65-0.80',
        sync_reason='Commodity + risk-on, but noisier',
        master_justification='Highest combined score (0.5804) of all pairs',
        best_params=AUD_NZD_BEST_PARAMS,
        performance_metrics={
            'val_combined_score': 0.5804,
            'val_direction_accuracy': 0.6636,
            'val_direction_f1': 0.6445,
            'val_loss': 0.3273,
        }
    ),
}

# Build reverse lookup: pair -> group_name
PAIR_TO_GROUP: Dict[str, str] = {}
for group_name, group in CORRELATION_GROUPS.items():
    for pair in group.pairs:
        PAIR_TO_GROUP[pair] = group_name


# =============================================================================
# PUBLIC API
# =============================================================================

def get_correlation_group(pair: str) -> Optional[CorrelationGroup]:
    """Get the correlation group for a given pair.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        
    Returns:
        CorrelationGroup if found, None otherwise
    """
    group_name = PAIR_TO_GROUP.get(pair)
    if group_name:
        return CORRELATION_GROUPS[group_name]
    return None


def get_master_pair(pair: str) -> str:
    """Get the master pair for a given pair.
    
    If the pair is itself a master, returns itself.
    If the pair is not in any group, returns 'EUR_GBP' as default.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        
    Returns:
        Master pair name
    """
    group = get_correlation_group(pair)
    if group:
        return group.master_pair
    return 'EUR_GBP'  # Default master


def get_best_params_for_pair(pair: str) -> Dict[str, Any]:
    """Get the best parameters for training/inference of a pair.
    
    This returns the master pair's optimized parameters, which transfer
    well to all pairs in the correlation group.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        
    Returns:
        Dictionary of optimized parameters
    """
    group = get_correlation_group(pair)
    if group:
        params = group.best_params.copy()
        # Override instrument with the actual pair being trained
        params['instrument'] = pair
        return params
    
    # Default: use EUR/GBP params
    params = DEFAULT_PARAMS.copy()
    params['instrument'] = pair
    logger.info(f"Pair {pair} not in any correlation group, using default params")
    return params


def get_transfer_params_for_pair(pair: str, keep_instrument: bool = False) -> Dict[str, Any]:
    """Get parameters for transfer learning from master to target pair.
    
    Similar to get_best_params_for_pair but with option to keep original
    instrument (for loading master model weights).
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        keep_instrument: If True, keep master pair's instrument setting
        
    Returns:
        Dictionary of parameters for transfer learning
    """
    group = get_correlation_group(pair)
    if group:
        params = group.best_params.copy()
        if not keep_instrument:
            params['instrument'] = pair
        return params
    
    params = DEFAULT_PARAMS.copy()
    params['instrument'] = pair
    return params


def get_all_pairs_in_group(group_name: str) -> List[str]:
    """Get all pairs in a correlation group.
    
    Args:
        group_name: Name of the correlation group
        
    Returns:
        List of pairs in the group
    """
    group = CORRELATION_GROUPS.get(group_name)
    if group:
        return group.pairs.copy()
    return []


def get_performance_metrics(pair: str) -> Optional[Dict[str, float]]:
    """Get performance metrics for the master pair of a group.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        
    Returns:
        Dictionary with val_combined_score, val_direction_accuracy, etc.
    """
    group = get_correlation_group(pair)
    if group:
        return group.performance_metrics.copy()
    return None


def is_master_pair(pair: str) -> bool:
    """Check if a pair is the master for its group.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        
    Returns:
        True if this pair is the master for its group
    """
    group = get_correlation_group(pair)
    if group:
        return group.master_pair == pair
    return False


def get_correlation_summary() -> Dict[str, Dict[str, Any]]:
    """Get a summary of all correlation groups.
    
    Returns:
        Dictionary mapping group names to their summaries
    """
    summary = {}
    for group_name, group in CORRELATION_GROUPS.items():
        summary[group_name] = {
            'name': group.name,
            'pairs': group.pairs,
            'master_pair': group.master_pair,
            'avg_correlation': group.avg_correlation,
            'sync_reason': group.sync_reason,
            'performance': group.performance_metrics,
        }
    return summary


def print_correlation_summary() -> None:
    """Print a formatted summary of all correlation groups."""
    print("\n" + "=" * 80)
    print("FX CORRELATION GROUPS - BEST PARAMETERS")
    print("=" * 80)
    
    for group_name, group in CORRELATION_GROUPS.items():
        print(f"\n{group.name}")
        print(f"  Pairs: {', '.join(group.pairs)}")
        print(f"  Correlation: {group.avg_correlation}")
        print(f"  Master: {group.master_pair}")
        print(f"  Performance: Score={group.performance_metrics.get('val_combined_score', 0):.4f}, "
              f"Dir Acc={group.performance_metrics.get('val_direction_accuracy', 0):.2%}")
    
    print("\n" + "=" * 80)


# =============================================================================
# CONFIGURATION FILE GENERATION
# =============================================================================

def generate_pair_config(pair: str, output_format: str = 'dict') -> Any:
    """Generate a configuration for a specific pair.
    
    Args:
        pair: Currency pair in format 'XXX_YYY'
        output_format: 'dict', 'yaml', or 'json'
        
    Returns:
        Configuration in the requested format
    """
    params = get_best_params_for_pair(pair)
    metrics = get_performance_metrics(pair)
    group = get_correlation_group(pair)
    
    config = {
        'pair_metadata': {
            'name': pair,
            'correlation_group': group.name if group else 'unknown',
            'master_pair': get_master_pair(pair),
            'is_master': is_master_pair(pair),
            'performance_source': metrics,
        },
        'tcn': {
            'batch_size': params.get('batch_size', 128),
            'seq_len': params.get('seq_len', 90),
            'tcn_nb_filters': params.get('tcn_nb_filters', 64),
            'tcn_kernel_size': params.get('tcn_kernel_size', 3),
            'tcn_dilations': params.get('tcn_dilations', [1, 2, 4, 8, 16]),
            'dropout_rate': params.get('dropout_rate', 0.3),
            'l2_reg': params.get('l2_reg', 0.0001),
            'lr': params.get('lr', 0.0005),
            'epochs': params.get('epochs', 200),
            'patience': params.get('patience', 20),
            'direction_weight': params.get('direction_weight', 0.5),
            'confidence_weight': params.get('confidence_weight', 0.25),
            'volatility_weight': params.get('volatility_weight', 0.2),
            'top_features': params.get('top_features', 150),
            'train_smoothing': params.get('train_smoothing', False),
            'mixed_precision': params.get('mixed_precision', True),
            'model_type': params.get('model_type', 'tcn'),
            'granularity': params.get('granularity', 'H1'),
        }
    }
    
    if output_format == 'yaml':
        import yaml
        return yaml.dump(config, default_flow_style=False)
    elif output_format == 'json':
        import json
        return json.dumps(config, indent=2)
    else:
        return config


if __name__ == '__main__':
    # Demo usage
    print_correlation_summary()
    
    # Example: Get params for a pair
    print("\nExample: Best params for EUR_JPY")
    params = get_best_params_for_pair('EUR_JPY')
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
