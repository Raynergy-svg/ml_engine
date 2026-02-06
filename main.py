from __future__ import annotations

# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI

A command-line interface for the ML trading engine that handles:
- Command processing and orchestration
- Interactive dashboard visualization
- Configuration management
- Training/evaluation progress display
- Real-time monitoring

Improvements:
- Modular architecture with clear separation of concerns
- Custom exception handling with helpful error messages
- Lazy imports for performance optimization
- Full type hints for better IDE support
- argparse subparsers for better command organization
- Early validation to fail fast
- Proper resource cleanup

Author: ML Engine Team
Version: 2.0.0
"""

# Suppress TensorFlow CPU instruction warnings (irrelevant on Apple Silicon/Metal)
# This import must come before any TensorFlow imports
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # 0=all, 1=info, 2=warning, 3=error

# ============================================================================
# Standard Library Imports (lightweight, always needed)
# ============================================================================

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, TypeVar

# ============================================================================
# Third-Party Imports (lightweight, always needed)
# ============================================================================

from rich.console import Console

# ============================================================================
# Local Imports - Core utilities (lightweight, always needed)
# ============================================================================

from src.utils import load_config, setup_logging

# ============================================================================
# Constants
# ============================================================================

DEFAULT_CONFIG_PATH = "./config/config_improved_H1.yaml"
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"
DEFAULT_CURRICULUM_KS = "32,64,128,0"
UTC_OFFSET_SUFFIX = "+00:00"

# ============================================================================
# Custom Exceptions
# ============================================================================


class CLIError(Exception):
    """Base exception for CLI errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None) -> None:
        """Initialize CLI error.
        
        Args:
            message: Error message
            suggestion: Optional suggestion for fixing the error
        """
        super().__init__(message)
        self.suggestion = suggestion
        self.message = message


class ConfigError(CLIError):
    """Exception raised for configuration-related errors."""
    pass


class ValidationError(CLIError):
    """Exception raised for validation errors."""
    pass


class CommandError(CLIError):
    """Exception raised for command execution errors."""
    pass


class ArgumentError(CLIError):
    """Exception raised for argument validation errors."""
    pass


# ============================================================================
# Type Definitions
# ============================================================================


class CommandArgs(TypedDict, total=False):
    """Typed dictionary for command arguments."""
    command: str
    config: str
    csv: Optional[str]
    model_path: Optional[str]
    pca_components: Optional[int]
    seq_len: Optional[int]
    epochs: Optional[int]
    batch_size: Optional[int]
    lr: Optional[float]
    warm_start: bool
    init_from: Optional[str]
    patience: Optional[int]
    es_monitor: Optional[str]
    combined_w_dir: Optional[float]
    combined_w_conf: Optional[float]
    top_features: Optional[int]
    feature_curriculum: Optional[bool]
    curriculum_ks: Optional[str]
    ignore_input_mismatches: bool
    disable_tier2_on_mismatch: bool
    median_window: Optional[int]
    tier2_calibrate: bool
    tier2_calibration_stride: Optional[int]
    tier2_horizon_candles: Optional[int]
    vol_norm_window: Optional[int]
    min_volume: Optional[int]
    spread_filter: bool
    spread_pctl: float
    spread_mult: float
    no_train_smoothing: bool
    seed: int
    run_tag: Optional[str]
    verbose: bool
    instrument: str
    granularity: str
    candles: int
    min_confidence: float
    all_pairs: bool
    lookahead: int
    pairs: Optional[str]
    top: int
    no_train: bool
    skip_execute: bool
    watch: bool
    interval: int
    auto_execute: bool
    diversified: bool
    last: int
    export: Optional[str]
    save: bool
    update: bool
    days: int
    import_trades: bool
    timesteps: int
    rl_episodes: Optional[int]
    use_rl_sizer: bool
    skip_rl: bool
    oanda_live: bool
    oanda_price: str
    oanda_save_csv: Optional[str]
    equity: float
    risk: float
    force_units: Optional[int]
    execute: bool
    force_execute: bool
    dry_run: bool
    loop: bool
    max_trades: int
    all_features: bool
    repl: bool
    device: str
    mixed_precision: bool
    steps_per_execution: Optional[int]
    jit_compile: bool
    shuffle_buffer: Optional[int]
    prefetch: Optional[int]
    cache_val: bool
    combined_use_predict: bool
    shared_encoder: Optional[bool]
    model_type: Optional[str]
    timing: bool
    fit_verbose: Optional[int]
    max_hold_minutes: Optional[float]
    force_margin: Optional[float]
    aggressive_scaling: bool
    intelligent: bool
    explain: bool
    llm_provider: str
    llm_enhance: bool
    retrain_interval: float
    drift_threshold: float
    feature_drift_threshold: float
    enable_drift_check: bool
    ewc_lambda: float
    ewc_gamma: float
    replay_mix_ratio: float
    replay_capacity_ratio: float
    ema_alpha: float
    ema_update_freq: int
    disable_continual_learning: bool
    enterprise: bool
    cv_folds: int
    bootstrap: bool
    bootstrap_samples: int
    mlflow_experiment: Optional[str]
    generate_report: bool
    disable_ewc: bool
    monitor_alerts: bool
    monitor_drift: bool
    monitor_report: bool
    monitor_limit: int
    multi_pair: bool
    foundation_pairs: Optional[str]
    auto_apply: bool


CommandHandler = Callable[[CommandArgs], None]
T = TypeVar('T')


# ============================================================================
# Argument Parser Builder
# ============================================================================


class ArgumentParserBuilder:
    """Builder class for creating the argument parser with subparsers."""
    
    def __init__(self) -> None:
        """Initialize the argument parser builder."""
        self._parser: Optional[argparse.ArgumentParser] = None
        self._subparsers: Optional[argparse._SubParsersAction] = None
        self._common_parser: Optional[argparse.ArgumentParser] = None
    
    def build(self) -> argparse.ArgumentParser:
        """Build and return the argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        if self._parser is not None:
            return self._parser
        
        # Create common parent parser with shared arguments
        self._common_parser = argparse.ArgumentParser(add_help=False)
        self._common_parser.add_argument(
            "--config",
            "-c",
            default=DEFAULT_CONFIG_PATH,
            help="Path to configuration file",
        )
        self._common_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Show detailed logs (default: quiet output)",
        )
        
        self._parser = argparse.ArgumentParser(
            description="ML Engine Trading Bot CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog(),
        )
        
        # Add global arguments to main parser as well (for --help)
        self._add_global_arguments()
        
        # Add subparsers for commands
        self._subparsers = self._parser.add_subparsers(
            dest='command',
            title='Available Commands',
            description='Use "buddy <command> --help" for command-specific help',
        )
        
        # Add all command subparsers
        self._add_train_command()
        self._add_buddy_command()
        self._add_scan_command()
        self._add_validate_command()
        self._add_journal_command()
        self._add_monitor_command()
        self._add_analyze_command()
        self._add_model_management_commands()
        
        return self._parser
    
    def _get_epilog(self) -> str:
        """Get the help epilog text.
        
        Returns:
            Help epilog string
        """
        return """
COMMAND REFERENCE:
  train        Train model for specific pair
  buddy        Single-pair inference → execute one trade
  scan         Scan ALL pairs → shows tradeable + needs training
  validate     Validate trained models
  journal      View trade journal
  monitor      Show monitoring dashboard and alerts
  analyze      Analyze model performance
  
EXAMPLES:
  buddy scan                     # Scan all majors, show recommendations
  buddy buddy -I EUR_USD --execute     # Single EUR_USD trade
  buddy train -I AUD_USD         # Train model for AUD_USD
  buddy journal                  # View trade journal
  buddy monitor                  # Show monitoring dashboard
  buddy monitor --alerts         # Show detailed alerts
"""
    
    def _add_global_arguments(self) -> None:
        """Add global arguments that apply to all commands."""
        assert self._parser is not None
        
        self._parser.add_argument(
            "--config",
            "-c",
            default=DEFAULT_CONFIG_PATH,
            help="Path to configuration file",
        )
        self._parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Show detailed logs (default: quiet output)",
        )
    
    def _add_train_command(self) -> None:
        """Add the train command subparser."""
        assert self._subparsers is not None
        
        train_parser = self._subparsers.add_parser(
            "train",
            help="Train model for specific pair",
            aliases=["train-buddy"],
            parents=[self._common_parser],
        )
        
        # Input data arguments
        train_parser.add_argument(
            "--csv",
            "-f",
            default=None,
            help="Path to market CSV file",
        )
        train_parser.add_argument(
            "--instrument",
            "-I",
            default="EUR_USD",
            help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD",
        )
        train_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity, e.g. M5, M15, H1",
        )
        train_parser.add_argument(
            "--candles",
            "-n",
            type=int,
            default=5000,
            help="How many candles to fetch for training (default: 5000)",
        )
        
        # Model architecture arguments
        train_parser.add_argument(
            "--model-type",
            type=str,
            choices=["tcn", "lstm", "xgboost", "ensemble", "attention_lstm"],
            default=None,
            help="Model architecture (default: config model.type or tcn)",
        )
        train_parser.add_argument(
            "--shared-encoder",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Use a single shared LSTM encoder instead of 5 parallel heads",
        )
        
        # Training hyperparameters
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Epochs for training (default: config buddy.train_defaults.epochs or 300)",
        )
        train_parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Batch size (default: config buddy.train_defaults.batch_size or 32)",
        )
        train_parser.add_argument(
            "--lr",
            type=float,
            default=None,
            help="Learning rate (default: config buddy.train_defaults.lr or 0.001)",
        )
        train_parser.add_argument(
            "--seq-len",
            type=int,
            default=None,
            help="Sequence length (default: config buddy.train_defaults.seq_len or 50)",
        )
        train_parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="EarlyStopping patience (default: config buddy.train_defaults.patience or 10)",
        )
        train_parser.add_argument(
            "--es-monitor",
            default=None,
            choices=["direction", "combined", "val_loss", "loss"],
            help="EarlyStopping monitor metric",
        )
        
        # Feature engineering arguments
        train_parser.add_argument(
            "--pca-components",
            type=int,
            default=None,
            help="PCA components (e.g. 20-30)",
        )
        train_parser.add_argument(
            "--top-features",
            type=int,
            default=None,
            help="Keep only top-K most correlated numeric features",
        )
        train_parser.add_argument(
            "--feature-curriculum",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable feature curriculum",
        )
        train_parser.add_argument(
            "--curriculum-ks",
            default=None,
            help=f"Comma-separated K schedule (0 = all). Default: {DEFAULT_CURRICULUM_KS}",
        )
        train_parser.add_argument(
            "--all-features",
            "-A",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use all engineered features (default: enabled)",
        )
        
        # Warm start arguments
        train_parser.add_argument(
            "--warm-start",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Initialize weights from existing model (default: enabled)",
        )
        train_parser.add_argument(
            "--init-from",
            default=None,
            help="Path to a .keras checkpoint to warm-start from",
        )
        
        # Tier-2 calibration arguments
        train_parser.add_argument(
            "--tier2-calibrate",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable/disable Tier-2 TP/SL calibration (default enabled)",
        )
        train_parser.add_argument(
            "--tier2-calibration-stride",
            type=int,
            default=None,
            help="Stride for Tier-2 labeling/calibration simulation",
        )
        train_parser.add_argument(
            "--tier2-horizon-candles",
            type=int,
            default=None,
            help="Tier-2 TP/SL max horizon in candles",
        )
        
        # Data filtering arguments
        train_parser.add_argument(
            "--min-volume",
            type=int,
            default=None,
            help="Drop candles with volume below this threshold",
        )
        train_parser.add_argument(
            "--spread-filter",
            action="store_true",
            help="Drop candles with abnormally wide bid/ask spread",
        )
        train_parser.add_argument(
            "--spread-pctl",
            type=float,
            default=0.99,
            help="Spread percentile used as filter threshold (default 0.99)",
        )
        train_parser.add_argument(
            "--spread-mult",
            type=float,
            default=3.0,
            help="Also filter if spread > (median * mult) (default 3.0)",
        )
        train_parser.add_argument(
            "--median-window",
            type=int,
            default=None,
            help="Apply rolling median to close before EMA smoothing (e.g. 3,5)",
        )
        train_parser.add_argument(
            "--no-train-smoothing",
            action="store_true",
            help="Disable candle noise reduction (resample to M5 + EMA(14) close)",
        )
        
        # Training performance arguments
        train_parser.add_argument(
            "--device",
            default="auto",
            choices=["auto", "cpu", "gpu"],
            help="Train device preference for TensorFlow (default auto)",
        )
        train_parser.add_argument(
            "--mixed-precision",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable mixed precision for 1.5-2x speedup on M1 (default enabled)",
        )
        train_parser.add_argument(
            "--steps-per-execution",
            type=int,
            default=None,
            help="Keras steps_per_execution to reduce per-step overhead",
        )
        train_parser.add_argument(
            "--jit-compile",
            action="store_true",
            help="Enable Keras jit_compile (XLA) if supported",
        )
        train_parser.add_argument(
            "--shuffle-buffer",
            type=int,
            default=None,
            help="Override window shuffle buffer size",
        )
        train_parser.add_argument(
            "--prefetch",
            type=int,
            default=None,
            help="Override tf.data prefetch (0=AUTOTUNE)",
        )
        train_parser.add_argument(
            "--cache-val",
            action="store_true",
            help="Cache validation window dataset in memory",
        )
        train_parser.add_argument(
            "--timing",
            action="store_true",
            help="Print per-batch timing",
        )
        train_parser.add_argument(
            "--fit-verbose",
            type=int,
            choices=[0, 1, 2],
            default=None,
            help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line per epoch",
        )
        
        # Continual learning arguments
        train_parser.add_argument(
            "--retrain-interval",
            type=float,
            default=168.0,
            help="Recommended hours between retraining sessions (default: 168 = 1 week)",
        )
        train_parser.add_argument(
            "--drift-threshold",
            type=float,
            default=0.03,
            help="Performance drop threshold for drift detection (default: 0.03 = 3%%)",
        )
        train_parser.add_argument(
            "--feature-drift-threshold",
            type=float,
            default=0.10,
            help="Feature distribution shift threshold (default: 0.10 = 10%%)",
        )
        train_parser.add_argument(
            "--enable-drift-check",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable automatic drift detection (default: enabled)",
        )
        train_parser.add_argument(
            "--ewc-lambda",
            type=float,
            default=1000.0,
            help="EWC penalty strength (default: 1000.0)",
        )
        train_parser.add_argument(
            "--ewc-gamma",
            type=float,
            default=0.95,
            help="EWC Fisher decay factor (default: 0.95)",
        )
        train_parser.add_argument(
            "--replay-mix-ratio",
            type=float,
            default=0.20,
            help="Fraction of training batch from replay buffer (default: 0.20)",
        )
        train_parser.add_argument(
            "--replay-capacity-ratio",
            type=float,
            default=0.10,
            help="Replay buffer capacity as fraction of total samples (default: 0.10)",
        )
        train_parser.add_argument(
            "--ema-alpha",
            type=float,
            default=0.999,
            help="EMA smoothing factor for shadow weights (default: 0.999)",
        )
        train_parser.add_argument(
            "--ema-update-freq",
            type=int,
            default=16,
            help="Update EMA weights every N training steps (default: 16)",
        )
        train_parser.add_argument(
            "--disable-continual-learning",
            action="store_true",
            help="Disable all continual learning features",
        )
        
        # Enterprise training arguments
        train_parser.add_argument(
            "--enterprise",
            action="store_true",
            default=True,
            help="Enable enterprise training features (default: enabled)",
        )
        train_parser.add_argument(
            "--no-enterprise",
            action="store_true",
            help="Disable enterprise training features",
        )
        train_parser.add_argument(
            "--cv-folds",
            type=int,
            default=5,
            help="Number of walk-forward cross-validation folds (default: 5)",
        )
        train_parser.add_argument(
            "--bootstrap",
            action="store_true",
            default=True,
            help="Enable bootstrap confidence intervals (default: enabled)",
        )
        train_parser.add_argument(
            "--no-bootstrap",
            action="store_true",
            help="Disable bootstrap confidence intervals",
        )
        train_parser.add_argument(
            "--bootstrap-samples",
            type=int,
            default=1000,
            help="Number of bootstrap iterations (default: 1000)",
        )
        train_parser.add_argument(
            "--mlflow-experiment",
            type=str,
            default=None,
            help="MLflow experiment name (default: auto-generated)",
        )
        train_parser.add_argument(
            "--generate-report",
            action="store_true",
            default=True,
            help="Generate professional markdown training report (default: enabled)",
        )
        train_parser.add_argument(
            "--no-report",
            action="store_true",
            help="Skip generating the training report",
        )
        train_parser.add_argument(
            "--disable-ewc",
            action="store_true",
            help="Disable EWC (Elastic Weight Consolidation)",
        )
        
        # Multi-pair training
        train_parser.add_argument(
            "--multi-pair",
            action="store_true",
            help="Train foundation model on multiple pairs simultaneously",
        )
        train_parser.add_argument(
            "--foundation-pairs",
            type=str,
            default=None,
            help="Comma-separated pairs for multi-pair training",
        )
        
        # OANDA live training
        train_parser.add_argument(
            "--oanda-live",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Fetch fresh candles from OANDA PRACTICE (default: enabled)",
        )
        train_parser.add_argument(
            "--oanda-price",
            default="MBA",
            help="OANDA price component: M, B, A, or MBA (default: MBA)",
        )
        train_parser.add_argument(
            "--oanda-save-csv",
            default=None,
            help="Optional path to write fetched OANDA candles CSV",
        )
        
        # Other training arguments
        train_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for training (default 42)",
        )
        train_parser.add_argument(
            "--run-tag",
            default=None,
            help="Optional tag to version outputs",
        )
        train_parser.add_argument(
            "--skip-rl",
            action="store_true",
            help="Skip RL position sizer training after ensemble",
        )
        train_parser.add_argument(
            "--ignore-input-mismatches",
            action="store_true",
            help="Best-effort mode: downgrade input/PCA/curriculum mismatches to warnings",
        )
        train_parser.add_argument(
            "--disable-tier2-on-mismatch",
            action="store_true",
            help="Skip Tier-2 instead of best-effort on input mismatches",
        )
    
    def _add_buddy_command(self) -> None:
        """Add the buddy command subparser."""
        assert self._subparsers is not None
        
        buddy_parser = self._subparsers.add_parser(
            "buddy",
            help="Single-pair inference → execute one trade",
            aliases=["Buddy"],
            parents=[self._common_parser],
        )
        
        buddy_parser.add_argument(
            "--instrument",
            "-I",
            default="EUR_USD",
            help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD",
        )
        buddy_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity, e.g. M5, M15, H1",
        )
        buddy_parser.add_argument(
            "--candles",
            "-n",
            type=int,
            default=5000,
            help="How many candles to fetch (default: 5000)",
        )
        buddy_parser.add_argument(
            "--model-path",
            default=None,
            help="Optional Buddy model path override",
        )
        buddy_parser.add_argument(
            "--equity",
            type=float,
            default=10_000.0,
            help="Paper equity for sizing (default: 10,000)",
        )
        buddy_parser.add_argument(
            "--risk",
            "-r",
            type=float,
            default=0.005,
            help="Risk per trade fraction (e.g. 0.005 = 0.5%%) (default: 0.005)",
        )
        buddy_parser.add_argument(
            "--force-units",
            type=int,
            default=None,
            help="Override computed units and place exactly this many units (PRACTICE only)",
        )
        buddy_parser.add_argument(
            "--execute",
            "-x",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable live trading on OANDA practice account (default: enabled)",
        )
        buddy_parser.add_argument(
            "--force-execute",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Bypass the training accuracy live-trading gate (PRACTICE only) (default: enabled)",
        )
        buddy_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Disable live trading, only simulate orders (overrides --execute)",
        )
        buddy_parser.add_argument(
            "--loop",
            action="store_true",
            help="Keep Buddy warm and act on each new candle (default: off)",
        )
        buddy_parser.add_argument(
            "--max-trades",
            type=int,
            default=1,
            help="Stop after this many executed trades (0 = unlimited) (default 1)",
        )
        buddy_parser.add_argument(
            "--use-rl-sizer",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use RL position sizer instead of Kelly criterion (default: enabled if RL model exists)",
        )
        buddy_parser.add_argument(
            "--max-hold-minutes",
            type=float,
            default=None,
            help="Maximum holding time in minutes for opened trades",
        )
        buddy_parser.add_argument(
            "--force-margin",
            type=float,
            default=None,
            help="Force approximate margin in USD for the trade",
        )
        buddy_parser.add_argument(
            "--aggressive-scaling",
            action="store_true",
            help="Enable $100K→$1M aggressive scaling strategy",
        )
        
        # Intelligent mode arguments
        buddy_parser.add_argument(
            "--intelligent",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable LLM-enhanced intelligent mode (default: enabled)",
        )
        buddy_parser.add_argument(
            "--explain",
            action="store_true",
            help="Generate detailed causal reasoning explaining the trade recommendation",
        )
        buddy_parser.add_argument(
            "--llm-provider",
            type=str,
            choices=["ollama", "claude", "openai", "auto"],
            default="auto",
            help="LLM provider for intelligent mode (default: auto)",
        )
        buddy_parser.add_argument(
            "--llm-enhance",
            "--no-llm-enhance",
            dest="llm_enhance",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable deep LLM integration (default: enabled)",
        )
        
        # REPL argument
        buddy_parser.add_argument(
            "--repl",
            action="store_true",
            help="Launch the interactive Buddy REPL",
        )
    
    def _add_scan_command(self) -> None:
        """Add the scan command subparser."""
        assert self._subparsers is not None
        
        scan_parser = self._subparsers.add_parser(
            "scan",
            help="Scan ALL pairs → shows tradeable + needs training",
            parents=[self._common_parser],
        )
        
        scan_parser.add_argument(
            "--pairs",
            type=str,
            default=None,
            help="Comma-separated pairs to scan (e.g., EUR_USD,GBP_USD,USD_JPY)",
        )
        scan_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity (default: H1)",
        )
        scan_parser.add_argument(
            "--top",
            type=int,
            default=5,
            help="Number of top results to show (default: 5)",
        )
        scan_parser.add_argument(
            "--no-train",
            action="store_true",
            dest="no_train",
            help="Skip the train prompt after scanning (deprecated)",
        )
        scan_parser.add_argument(
            "--skip-execute",
            action="store_true",
            dest="skip_execute",
            help="Skip the trade execution prompt after scanning",
        )
        scan_parser.add_argument(
            "--watch",
            action="store_true",
            help="Enable continuous watch mode (scans repeatedly)",
        )
        scan_parser.add_argument(
            "--interval",
            type=int,
            default=5,
            help="Minutes between scans for --watch (default: 5)",
        )
        scan_parser.add_argument(
            "--auto-execute",
            action="store_true",
            help="For --watch: automatically execute passing trades",
        )
        scan_parser.add_argument(
            "--diversified",
            "-d",
            action="store_true",
            help="Auto-filter correlated pairs (only show best from each cluster)",
        )
        scan_parser.add_argument(
            "--use-rl-sizer",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use RL position sizer instead of Kelly criterion (default: enabled)",
        )
        scan_parser.add_argument(
            "--force",
            action="store_true",
            help="Force scan even outside optimal trading hours",
        )
    
    def _add_validate_command(self) -> None:
        """Add the validate command subparser."""
        assert self._subparsers is not None
        
        validate_parser = self._subparsers.add_parser(
            "validate",
            help="Validate trained models",
            parents=[self._common_parser],
        )
        
        validate_parser.add_argument(
            "--instrument",
            "-I",
            default="EUR_USD",
            help="OANDA instrument to validate",
        )
        validate_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity (default: H1)",
        )
        validate_parser.add_argument(
            "--candles",
            "-n",
            type=int,
            default=800,
            help="Number of candles for validation (default: 800)",
        )
        validate_parser.add_argument(
            "--lookahead",
            type=int,
            default=24,
            help="Hours ahead to measure actual outcome (default: 24)",
        )
        validate_parser.add_argument(
            "--all-pairs",
            action="store_true",
            help="Validate all trained pair models",
        )
    
    def _add_journal_command(self) -> None:
        """Add the journal command subparser."""
        assert self._subparsers is not None
        
        journal_parser = self._subparsers.add_parser(
            "journal",
            help="View trade journal",
            parents=[self._common_parser],
        )
        
        journal_parser.add_argument(
            "--update",
            action="store_true",
            help="Update trade results from OANDA",
        )
        journal_parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days of history to show (default: 30)",
        )
        journal_parser.add_argument(
            "--import-trades",
            action="store_true",
            help="Import untracked open trades from OANDA into journal",
        )
    
    def _add_monitor_command(self) -> None:
        """Add the monitor command subparser."""
        assert self._subparsers is not None
        
        monitor_parser = self._subparsers.add_parser(
            "monitor",
            help="Show monitoring dashboard and alerts",
            parents=[self._common_parser],
        )
        
        monitor_parser.add_argument(
            "--monitor-alerts",
            action="store_true",
            help="Show detailed alerts",
        )
        monitor_parser.add_argument(
            "--monitor-drift",
            action="store_true",
            help="Show model drift history",
        )
        monitor_parser.add_argument(
            "--monitor-report",
            action="store_true",
            help="Generate full monitoring report",
        )
        monitor_parser.add_argument(
            "--monitor-limit",
            type=int,
            default=20,
            help="Limit for drift history display (default: 20)",
        )
    
    def _add_analyze_command(self) -> None:
        """Add the analyze command subparser."""
        assert self._subparsers is not None
        
        analyze_parser = self._subparsers.add_parser(
            "analyze",
            help="Analyze model performance",
            parents=[self._common_parser],
        )
        
        analyze_parser.add_argument(
            "--top",
            type=int,
            default=30,
            help="Number of top results to show (default: 30)",
        )
        analyze_parser.add_argument(
            "--save",
            action="store_true",
            help="Save plots to trained_data/visualizations",
        )
    
    def _add_model_management_commands(self) -> None:
        """Add model management command subparsers."""
        assert self._subparsers is not None
        
        # Promote model command
        _ = self._subparsers.add_parser(
            "promote-model",
            help="Promote a model to production",
            parents=[self._common_parser],
        )
        
        # Model status command
        _ = self._subparsers.add_parser(
            "model-status",
            help="Show model status and information",
            parents=[self._common_parser],
        )
        
        # Retrain gates command
        retrain_gates_parser = self._subparsers.add_parser(
            "retrain-gates",
            help="Retrain sklearn gates only (XGBoost, RF, Ridge)",
            parents=[self._common_parser],
        )
        retrain_gates_parser.add_argument(
            "--pairs",
            type=str,
            default=None,
            help="Comma-separated pairs",
        )
        retrain_gates_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity (default: H1)",
        )
        retrain_gates_parser.add_argument(
            "--candles",
            "-n",
            type=int,
            default=5000,
            help="Number of candles (default: 5000)",
        )
        
        # Train RL sizer command
        train_rl_parser = self._subparsers.add_parser(
            "train-rl-sizer",
            help="Train RL position sizing agent",
            parents=[self._common_parser],
        )
        train_rl_parser.add_argument(
            "--timesteps",
            type=int,
            default=500_000,
            help="Total training timesteps (default: 500000)",
        )
        train_rl_parser.add_argument(
            "--rl-episodes",
            type=int,
            default=None,
            help="Training episodes (overrides --timesteps)",
        )
        train_rl_parser.add_argument(
            "--pairs",
            type=str,
            default=None,
            help="Comma-separated pairs",
        )
        train_rl_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity (default: H1)",
        )
        train_rl_parser.add_argument(
            "--candles",
            "-n",
            type=int,
            default=5000,
            help="Number of candles (default: 5000)",
        )
        
        # Suggest improvements command
        suggest_parser = self._subparsers.add_parser(
            "suggest-improvements",
            help="Suggest hyperparameter improvements",
            parents=[self._common_parser],
        )
        suggest_parser.add_argument(
            "--instrument",
            "-I",
            default=None,
            help="OANDA instrument",
        )
        suggest_parser.add_argument(
            "--auto-apply",
            action="store_true",
            help="Automatically apply approved hyperparam changes to config",
        )

        # Find optimal candles command
        find_candles_parser = self._subparsers.add_parser(
            "find-candles",
            help="Find optimal training candle count for an instrument",
            parents=[self._common_parser],
        )
        find_candles_parser.add_argument(
            "--instrument",
            "-I",
            required=True,
            help="OANDA instrument (e.g. EUR_USD)",
        )
        find_candles_parser.add_argument(
            "--granularity",
            "-g",
            default="H1",
            help="OANDA candle granularity (default: H1)",
        )
        find_candles_parser.add_argument(
            "--min-candles",
            type=int,
            default=3000,
            help="Minimum candle count to test (default: 3000)",
        )
        find_candles_parser.add_argument(
            "--max-candles",
            type=int,
            default=30000,
            help="Maximum candle count to test (default: 30000)",
        )
        find_candles_parser.add_argument(
            "--step",
            type=int,
            default=3000,
            help="Step size between candidate counts (default: 3000)",
        )
        find_candles_parser.add_argument(
            "--cv-folds",
            type=int,
            default=3,
            help="Walk-forward CV folds per candidate (default: 3)",
        )
        find_candles_parser.add_argument(
            "--no-auto-train",
            action="store_true",
            help="Show results only — do not auto-train with optimal count",
        )


# ============================================================================
# Configuration Validator
# ============================================================================


class ConfigValidator:
    """Validator for configuration files and arguments."""
    
    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the configuration validator.
        
        Args:
            logger: Logger instance for validation messages
        """
        self._logger = logger
    
    def validate_config_path(self, config_path: str) -> Path:
        """Validate that the configuration file exists and is readable.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Path object for the configuration file
            
        Raises:
            ConfigError: If configuration file is invalid
        """
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigError(
                f"Configuration file not found: {config_path}",
                suggestion=f"Create a configuration file at {config_path} or specify a different path with --config",
            )
        
        if not path.is_file():
            raise ConfigError(
                f"Configuration path is not a file: {config_path}",
                suggestion="Please provide a valid file path",
            )
        
        if not os.access(path, os.R_OK):
            raise ConfigError(
                f"Configuration file is not readable: {config_path}",
                suggestion="Check file permissions",
            )
        
        self._logger.debug(f"Configuration file validated: {path}")
        return path
    
    def validate_config_content(self, config: Dict[str, Any]) -> None:
        """Validate the content of the loaded configuration.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ConfigError: If configuration content is invalid
        """
        # 'fx' is the modern section name, 'oanda' is legacy - accept either
        required_sections = ['buddy', 'model']
        missing_sections = [s for s in required_sections if s not in config]
        
        # Either 'fx' or 'oanda' must be present
        if 'fx' not in config and 'oanda' not in config:
            missing_sections.append('fx (or oanda)')
        
        if missing_sections:
            raise ConfigError(
                f"Missing required configuration sections: {', '.join(missing_sections)}",
                suggestion="Ensure your configuration file has all required sections",
            )
        
        self._logger.debug("Configuration content validated")
    
    def validate_instrument(self, instrument: str) -> str:
        """Validate and normalize an OANDA instrument name.
        
        Args:
            instrument: Instrument name to validate
            
        Returns:
            Normalized instrument name
            
        Raises:
            ValidationError: If instrument is invalid
        """
        if not instrument:
            raise ValidationError(
                "Instrument name cannot be empty",
                suggestion="Provide a valid instrument name with --instrument",
            )
        
        # Normalize: replace / with _ and uppercase
        normalized = instrument.replace("/", "_").upper()
        
        # Basic format validation (should be like EUR_USD, USD_JPY, etc.)
        if "_" not in normalized:
            raise ValidationError(
                f"Invalid instrument format: {instrument}",
                suggestion="Use format like EUR_USD or USD_JPY",
            )
        
        self._logger.debug(f"Instrument validated: {instrument} -> {normalized}")
        return normalized
    
    def validate_granularity(self, granularity: str) -> str:
        """Validate an OANDA candle granularity.
        
        Args:
            granularity: Granularity to validate
            
        Returns:
            Validated granularity string
            
        Raises:
            ValidationError: If granularity is invalid
        """
        valid_granularities = {
            'M1', 'M2', 'M3', 'M4', 'M5', 'M10', 'M15', 'M30',
            'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12',
            'D', 'W', 'M'
        }
        
        if granularity.upper() not in valid_granularities:
            raise ValidationError(
                f"Invalid granularity: {granularity}",
                suggestion=f"Use one of: {', '.join(sorted(valid_granularities))}",
            )
        
        self._logger.debug(f"Granularity validated: {granularity}")
        return granularity.upper()


# ============================================================================
# Command Dispatcher
# ============================================================================


class CommandDispatcher:
    """Dispatcher for routing CLI commands to their handlers."""
    
    def __init__(self, console: Console, logger: logging.Logger) -> None:
        """Initialize the command dispatcher.
        
        Args:
            console: Rich console instance for output
            logger: Logger instance for logging
        """
        self._console = console
        self._logger = logger
        self._command_handlers: Dict[str, CommandHandler] = {}
        self._register_handlers()
    
    def _register_handlers(self) -> None:
        """Register all command handlers."""
        self._command_handlers = {
            "train": self._handle_train,
            "train-buddy": self._handle_train,
            "retrain-gates": self._handle_retrain_gates,
            "train-rl-sizer": self._handle_train_rl_sizer,
            "buddy": self._handle_buddy,
            "Buddy": self._handle_buddy,
            "promote-model": self._handle_promote_model,
            "model-status": self._handle_model_status,
            "test": self._handle_test,
            "validate": self._handle_validate,
            "scan": self._handle_scan,
            "analyze": self._handle_analyze,
            "journal": self._handle_journal,
            "monitor": self._handle_monitor,
            "suggest-improvements": self._handle_suggest_improvements,
            "find-candles": self._handle_find_candles,
        }
    
    def dispatch(self, args: CommandArgs) -> None:
        """Dispatch a command to its handler.
        
        Args:
            args: Command arguments
            
        Raises:
            CommandError: If command execution fails
        """
        command = args.get('command')
        
        if not command:
            raise CommandError(
                "No command specified",
                suggestion="Use 'buddy --help' to see available commands",
            )
        
        handler = self._command_handlers.get(command)
        
        if not handler:
            raise CommandError(
                f"Unknown command: {command}",
                suggestion="Use 'buddy --help' to see available commands",
            )
        
        self._logger.info(f"Executing command: {command}")
        
        try:
            handler(args)
            if command != 'scan':
                self._logger.info(f"Command completed successfully: {command}")
        except Exception as e:
            self._logger.error(f"Command failed: {command}", exc_info=True)
            raise CommandError(
                f"Command '{command}' failed: {str(e)}",
                suggestion="Check logs for more details",
            ) from e
    
    # Command handlers
    # =================
    
    def _handle_train(self, args: CommandArgs) -> None:
        """Handle the train command."""
        import argparse
        from cli import _dispatch_train_buddy
        from cli.training import train_buddy

        # _dispatch_train_buddy expects argparse.Namespace (attribute access)
        ns = argparse.Namespace(**args)
        # command_map must point to train_buddy, NOT _handle_train (avoids recursion)
        cmd_map = {"train-buddy": train_buddy}
        _dispatch_train_buddy(ns, cmd_map)
    
    def _handle_retrain_gates(self, args: CommandArgs) -> None:
        """Handle the retrain-gates command."""
        from cli import retrain_gates as _cli_retrain_gates
        
        _cli_retrain_gates(
            config_path=args['config'],
            pairs=args.get('pairs'),
            granularity=args.get('granularity', 'H1'),
            candles=args.get('candles', 5000),
            verbose=args.get('verbose', False),
        )
    
    def _handle_train_rl_sizer(self, args: CommandArgs) -> None:
        """Handle the train-rl-sizer command."""
        from cli import train_rl_sizer as _cli_train_rl_sizer
        
        _cli_train_rl_sizer(
            config_path=args['config'],
            timesteps=args.get('timesteps', 500_000),
            episodes=args.get('rl_episodes'),
            pairs=args.get('pairs'),
            granularity=args.get('granularity', 'H1'),
            candles=args.get('candles', 5000),
            verbose=args.get('verbose', False),
        )
    
    def _handle_buddy(self, args: CommandArgs) -> None:
        """Handle the buddy command."""
        from cli import _dispatch_buddy
        
        _dispatch_buddy(args, self._command_handlers)
    
    def _handle_promote_model(self, args: CommandArgs) -> None:
        """Handle the promote-model command."""
        from cli.model_management import promote_model as _cli_promote_model
        
        _cli_promote_model(args['config'])
    
    def _handle_model_status(self, args: CommandArgs) -> None:
        """Handle the model-status command."""
        from cli import model_status as _cli_model_status
        
        _cli_model_status(args['config'])
    
    def _handle_test(self, args: CommandArgs) -> None:
        """Handle the test command (legacy - redirects to validate)."""
        from cli import buddy_test as _cli_buddy_test
        
        _cli_buddy_test(
            args['config'],
            instrument=args.get('instrument', 'USD_JPY'),
            granularity=args.get('granularity', 'H1'),
            test_candles=args.get('candles', 50),
            min_confidence=args.get('min_confidence', 0.0),
            verbose=args.get('verbose', False),
        )
    
    def _handle_validate(self, args: CommandArgs) -> None:
        """Handle the validate command."""
        from cli import buddy_validate as _cli_buddy_validate
        
        _cli_buddy_validate(
            args['config'],
            instrument=args.get('instrument', 'EUR_USD'),
            granularity=args.get('granularity', 'H1'),
            candles=args.get('candles', 800),
            lookahead=args.get('lookahead', 24),
            all_pairs=args.get('all_pairs', False),
            verbose=args.get('verbose', False),
        )
    
    def _handle_scan(self, args: CommandArgs) -> None:
        """Handle the scan command."""
        import logging as _logging
        
        # Suppress all verbose logging during scan — only scanner output should show
        root_logger = _logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(_logging.CRITICAL)
        
        try:
            from cli import buddy_scan as _cli_buddy_scan
            
            watch_mode = args.get('watch', False)
            
            if watch_mode:
                root_logger.setLevel(prev_level)  # Restore for continuous mode
                from buddy_scanner import BuddyScanner
                
                scanner = BuddyScanner(
                    config_path=args['config'],
                    use_rl_sizer=args.get('use_rl_sizer', True),
                )
                
                pairs_str = args.get('pairs')
                pair_list = None
                if pairs_str:
                    pair_list = [p.strip().upper().replace("/", "_") for p in pairs_str.split(",")]
                
                scanner.scan_continuous(
                    pairs=pair_list,
                    granularity=args.get('granularity', 'H1'),
                    interval_minutes=args.get('interval', 5),
                    auto_execute=args.get('auto_execute', False),
                    top_n=args.get('top', 5),
                )
            else:
                _cli_buddy_scan(
                    args['config'],
                    pairs=args.get('pairs'),
                    granularity=args.get('granularity', 'H1'),
                    top_n=args.get('top', 5),
                    verbose=args.get('verbose', False),
                    prompt_train=not args.get('no_train', False),
                    no_execute=args.get('skip_execute', False),
                    use_rl_sizer=args.get('use_rl_sizer', True),
                    diversified=args.get('diversified', False),
                    force=args.get('force', False),
                )
        finally:
            root_logger.setLevel(prev_level)
    
    def _handle_analyze(self, args: CommandArgs) -> None:
        """Handle the analyze command."""
        from cli import buddy_analyze as _cli_buddy_analyze
        
        _cli_buddy_analyze(
            args['config'],
            top_n=args.get('top', 30),
            save_plots=args.get('save', False),
            verbose=args.get('verbose', False),
        )
    
    def _handle_journal(self, args: CommandArgs) -> None:
        """Handle the journal command."""
        from cli import buddy_journal as _cli_buddy_journal
        
        _cli_buddy_journal(
            args['config'],
            update=args.get('update', False),
            days=args.get('days', 30),
            verbose=args.get('verbose', False),
            import_trades=args.get('import_trades', False),
        )
    
    def _handle_monitor(self, args: CommandArgs) -> None:
        """Handle the monitor command."""
        from cli import buddy_monitor as _cli_buddy_monitor
        
        _cli_buddy_monitor(
            args['config'],
            show_alerts=args.get('monitor_alerts', False),
            show_drift=args.get('monitor_drift', False),
            generate_report=args.get('monitor_report', False),
            drift_limit=args.get('monitor_limit', 20),
        )
    
    def _handle_suggest_improvements(self, args: CommandArgs) -> None:
        """Handle the suggest-improvements command."""
        from cli import suggest_improvements as _cli_suggest_improvements
        
        _cli_suggest_improvements(
            args['config'],
            instrument=args.get('instrument'),
            auto_apply=args.get('auto_apply', False),
            verbose=args.get('verbose', False),
        )

    def _handle_find_candles(self, args: CommandArgs) -> None:
        """Handle the find-candles command."""
        from cli.candle_optimizer import find_optimal_candles

        find_optimal_candles(
            args['config'],
            instrument=args['instrument'],
            granularity=args.get('granularity', 'H1'),
            min_candles=args.get('min_candles', 3000),
            max_candles=args.get('max_candles', 30000),
            step=args.get('step', 3000),
            n_folds=args.get('cv_folds', 3),
            auto_train=not args.get('no_auto_train', False),
            verbose=args.get('verbose', False),
        )


# ============================================================================
# Main Application
# ============================================================================


class MLTradingCLI:
    """Main CLI application class."""
    
    def __init__(self) -> None:
        """Initialize the CLI application."""
        self._console = Console()
        self._logger = setup_logging(log_file="cli.log")
        self._parser_builder = ArgumentParserBuilder()
        self._config_validator = ConfigValidator(self._logger)
        self._dispatcher: Optional[CommandDispatcher] = None
    
    def run(self, argv: Optional[List[str]] = None) -> int:
        """Run the CLI application.
        
        Args:
            argv: Command line arguments (defaults to sys.argv[1:])
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Build and parse arguments
            parser = self._parser_builder.build()
            args = parser.parse_args(argv)
            
            # Convert argparse Namespace to CommandArgs dict
            args_dict: CommandArgs = vars(args)
            
            # Handle empty command (show help)
            if not args_dict.get('command'):
                parser.print_help()
                return 0
            
            # Check for REPL mode
            if args_dict.get('repl'):
                self._launch_repl()
                return 0
            
            # Suppress all logging for scan command — only scanner output should show
            _scan_suppressed = False
            if args_dict.get('command') == 'scan':
                import logging as _logging
                _root = _logging.getLogger()
                _prev = _root.level
                _root.setLevel(_logging.CRITICAL)
                _scan_suppressed = True
            
            # Validate configuration
            self._validate_configuration(args_dict)
            
            # Normalize arguments
            self._normalize_arguments(args_dict)
            
            # Dispatch command
            if self._dispatcher is None:
                self._dispatcher = CommandDispatcher(self._console, self._logger)
            
            self._dispatcher.dispatch(args_dict)
            
            # Restore logging if suppressed
            if _scan_suppressed:
                _root.setLevel(_prev)
            
            return 0
            
        except KeyboardInterrupt:
            self._console.print("\n[yellow]Operation interrupted by user[/yellow]")
            return 130  # Standard exit code for SIGINT
        
        except CLIError as e:
            self._console.print(f"[red]Error: {e.message}[/red]")
            if e.suggestion:
                self._console.print(f"[yellow]Suggestion: {e.suggestion}[/yellow]")
            self._logger.error(f"CLI error: {e.message}", exc_info=True)
            return 1
        
        except Exception as e:
            self._console.print(f"[red]Unexpected error: {str(e)}[/red]")
            self._logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1
    
    def _validate_configuration(self, args: CommandArgs) -> None:
        """Validate the configuration file and arguments.
        
        Args:
            args: Command arguments
            
        Raises:
            ConfigError: If configuration is invalid
            ValidationError: If arguments are invalid
        """
        # Validate config file
        config_path = args['config']
        self._config_validator.validate_config_path(config_path)
        
        # Load and validate config content
        config = load_config(config_path)
        self._config_validator.validate_config_content(config)
        
        # Validate instrument if provided
        if args.get('instrument'):
            args['instrument'] = self._config_validator.validate_instrument(args['instrument'])
        
        # Validate granularity if provided
        if args.get('granularity'):
            args['granularity'] = self._config_validator.validate_granularity(args['granularity'])
    
    def _normalize_arguments(self, args: CommandArgs) -> None:
        """Normalize and preprocess command arguments.
        
        Args:
            args: Command arguments to normalize (modified in place)
        """
        # Import normalization functions from cli package
        from cli import _normalize_command_args
        
        # Create a namespace-like object for compatibility
        class ArgsNamespace:
            def __init__(self, args_dict: CommandArgs):
                for key, value in args_dict.items():
                    setattr(self, key, value)
        
        args_namespace = ArgsNamespace(args)
        _normalize_command_args(args_namespace)
        
        # Update args dict with normalized values
        for key in args.keys():
            args[key] = getattr(args_namespace, key, args[key])
    
    def _launch_repl(self) -> None:
        """Launch the interactive REPL."""
        from cli import _maybe_launch_buddy_repl
        
        # Create a namespace-like object for compatibility
        class ArgsNamespace:
            def __init__(self):
                self.config = DEFAULT_CONFIG_PATH
                self.verbose = False
        
        args_namespace = ArgsNamespace()
        _maybe_launch_buddy_repl(args_namespace)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Check for interactive wizard mode
    from cli import _maybe_run_buddy_interactive_wizard
    
    if _maybe_run_buddy_interactive_wizard(default_config=DEFAULT_CONFIG_PATH):
        return 0
    
    # Run the CLI application
    cli = MLTradingCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
