"""
Shared constants for ML Engine configuration.

This module defines authoritative defaults that match config/config_improved_H1.yaml
to prevent configuration drift across the codebase.
"""

# =============================================================================
# DIRECTION LABELING DEFAULTS (H1 TIMEFRAME)
# =============================================================================
# These values match config/config_improved_H1.yaml:
#   direction_threshold: 0.003  # Min 0.3% move for clear signal
#   direction_lookahead: 24     # 24 hours lookahead
#
# Used by:
# - cli/training.py (training orchestration)
# - src/core/modular_data_loaders.py (data loading functions)
# - cli/commands.py (backtest evaluation)

DIRECTION_DEFAULTS = {
    'threshold': 0.003,  # 0.3% minimum price move for clear signal
    'lookahead': 24,     # 24 H1 bars = 24 hours lookahead
}


# =============================================================================
# INFERENCE GATE DEFAULTS
# =============================================================================
# These values match config/config_improved_H1.yaml inference section

INFERENCE_DEFAULTS = {
    'min_tcn_probability': 0.60,  # 60% minimum confidence (not coin-flip)
    'min_confidence': 50.0,       # ADX-based trend strength threshold
    'min_momentum': 0.20,         # Momentum percentile threshold
    'max_drawdown_pct': 0.025,    # 2.5% max expected drawdown
}
