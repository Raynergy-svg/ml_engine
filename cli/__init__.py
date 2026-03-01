#!/usr/bin/env python3
"""CLI package for ML Engine Trading Bot."""
from cli.config import OandaFetchOptions, BuddyTrainingOptions, BuddyTrainingAdvancedOptions
from cli.io_utils import (
    console, logger, DEFAULT_CONFIG_PATH, VALID_OANDA_INSTRUMENTS,
    DEFAULT_CURRICULUM_KS,
)
from cli.calibration import _tier2_apply_calibration  # noqa: F401
from cli.tf_config import _configure_tf_metal  # noqa: F401
from cli.training import train_buddy
from cli.commands import (
    buddy, buddy_loop,
    buddy_validate, buddy_test,
    _normalize_command_args, _maybe_run_buddy_interactive_wizard,
    _maybe_launch_buddy_repl, _maybe_launch_buddy_chat, _dispatch_buddy, _dispatch_train_buddy,
)
from cli.buddy_scanning import buddy_scan
from cli.analytics_commands import (
    buddy_monitor, buddy_journal, buddy_analyze, suggest_improvements,
)
from cli.model_management import model_status, promote_model
from cli.legacy_commands import train_model, evaluate_model
from cli.training_ops import (
    train_rl_sizer, retrain_gates, train_rl_gates, train_rl_exits, retrain_all,
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
    'train_model', 'evaluate_model', 'train_rl_sizer', 'retrain_gates',
    'train_rl_gates', 'train_rl_exits', 'retrain_all', 'suggest_improvements',
    '_normalize_command_args', '_maybe_run_buddy_interactive_wizard',
    '_maybe_launch_buddy_repl', '_maybe_launch_buddy_chat', '_dispatch_buddy', '_dispatch_train_buddy',
    # FX Trading
    'fx_paper_trade', 'generate_dashboard',
    # Wizard
    '_buddy_interactive_wizard',
    # Candle Optimizer
    'find_optimal_candles',
    # Re-exports
    '_tier2_apply_calibration',
    '_configure_tf_metal',
]
