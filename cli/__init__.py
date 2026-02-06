#!/usr/bin/env python3
"""CLI package for ML Engine Trading Bot."""
from cli.config import OandaFetchOptions, BuddyTrainingOptions, BuddyTrainingAdvancedOptions
from cli.io_utils import (
    console, logger, DEFAULT_CONFIG_PATH, VALID_OANDA_INSTRUMENTS,
    _validate_instrument, _normalize_instrument, BUDDY_META_FILENAME,
    DEFAULT_CURRICULUM_KS,
)
from cli.calibration import _tier2_apply_calibration
from cli.tf_config import _configure_tf_metal
from cli.training import train_buddy
from cli.commands import (
    buddy, buddy_loop, buddy_scan, buddy_monitor, buddy_journal,
    buddy_analyze, buddy_validate, buddy_test, model_status, promote_model,
    train_model, evaluate_model, train_rl_sizer, retrain_gates, suggest_improvements,
    _normalize_command_args, _maybe_run_buddy_interactive_wizard,
    _maybe_launch_buddy_repl, _dispatch_buddy, _dispatch_train_buddy,
)
from cli.fx_trading import fx_paper_trade, generate_dashboard
from cli.wizard import _buddy_interactive_wizard
from cli.candle_optimizer import find_optimal_candles

__all__ = [
    # Config
    'OandaFetchOptions', 'BuddyTrainingOptions', 'BuddyTrainingAdvancedOptions',
    # IO Utils
    'console', 'logger', 'DEFAULT_CONFIG_PATH', 'VALID_OANDA_INSTRUMENTS',
    'DEFAULT_CURRICULUM_KS',
    # Training
    'train_buddy',
    # Commands
    'buddy', 'buddy_loop', 'buddy_scan', 'buddy_monitor', 'buddy_journal',
    'buddy_analyze', 'buddy_validate', 'buddy_test', 'model_status', 'promote_model',
    'train_model', 'evaluate_model', 'train_rl_sizer', 'retrain_gates', 'suggest_improvements',
    '_normalize_command_args', '_maybe_run_buddy_interactive_wizard',
    '_maybe_launch_buddy_repl', '_dispatch_buddy', '_dispatch_train_buddy',
    # FX Trading
    'fx_paper_trade', 'generate_dashboard',
    # Wizard
    '_buddy_interactive_wizard',
    # Candle Optimizer
    'find_optimal_candles',
]
