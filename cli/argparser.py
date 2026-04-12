#!/usr/bin/env python3
"""CLI argument parser for ML Engine Trading Bot.

This module contains all argparse definitions, separated from main.py
for cleaner organization and easier maintenance.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from cli.io_utils import DEFAULT_CONFIG_PATH, DEFAULT_CURRICULUM_KS

if TYPE_CHECKING:
    from argparse import Namespace


# CLI commands available
COMMANDS = [
    "train",
    "train-buddy",
    "train-joint",
    "train-tcn-volatility",
    "retrain-gates",
    "retrain-all",
    "train-rl-sizer",
    "train-rl-gates",
    "train-rl-exits",
    "buddy",
    "buddy-chat",
    "Buddy",
    "promote-model",
    "model-status",
    "test",
    "validate",
    "scan",
    "analyze",
    "journal",
    "trade-analysis",
    "suggest-improvements",
    "monitor",
    "find-candles",
    "transfer",
    "learn",
    "feedback-status",
    "train-offline-sizer",
]

CLI_EPILOG = """
COMMAND REFERENCE:
  scan        Scan ALL pairs → shows tradeable + needs training
  buddy       Single-pair inference → execute one trade
  buddy-chat  Grounded Buddy conversational REPL
  train       Train model for specific pair
  monitor     Show monitoring dashboard and alerts

EXAMPLES:
  buddy scan                     # Scan all majors, show recommendations
  buddy -I EUR_USD --execute     # Single EUR_USD trade
  buddy buddy-chat -I EUR_USD    # Start Buddy chat REPL (dry-run by default)
  buddy train -I AUD_USD         # Train model for AUD_USD
  buddy journal                  # View trade journal
  buddy monitor                  # Show monitoring dashboard
  buddy monitor --alerts         # Show detailed alerts
"""


def _add_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Add core CLI arguments (command, config, csv, model-path)."""
    parser.add_argument(
        "command",
        nargs="?",
        default="buddy",
        choices=COMMANDS,
        help="Command: scan (multi-pair) | buddy (single-pair) | train | validate | journal | monitor | suggest-improvements",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--csv",
        "-f",
        default=None,
        help="Path to market CSV file (used by train-buddy)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional Buddy model path override (defaults to trained_data/models/buddy_tf.keras)",
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments (epochs, batch-size, lr, etc.)."""
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Optional PCA components (e.g. 20-30) for Buddy training",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length for Buddy training (default: config buddy.train_defaults.seq_len or 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs for Buddy training (default: config buddy.train_defaults.epochs or 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Buddy training (default: config buddy.train_defaults.batch_size or 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for Buddy training (default: config buddy.train_defaults.lr or 0.001)",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: initialize weights from existing model before training (preserves learned patterns). (default: enabled; use --no-warm-start for fresh training)",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="For train-buddy: path to a .keras checkpoint to warm-start from (overrides --warm-start)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="EarlyStopping patience for Buddy training (applies to --es-monitor) (default: config buddy.train_defaults.patience or 10)",
    )
    parser.add_argument(
        "--es-monitor",
        default=None,
        choices=["direction", "combined", "val_loss", "loss"],
        help="EarlyStopping monitor for train-buddy: direction|combined|val_loss|loss (default: config buddy.train_defaults.es_monitor or direction)",
    )
    parser.add_argument(
        "--combined-w-dir",
        type=float,
        default=None,
        help="Weight for direction accuracy in combined early-stopping objective (default: config buddy.train_defaults.combined_w_dir or 0.7)",
    )
    parser.add_argument(
        "--combined-w-conf",
        type=float,
        default=None,
        help="Weight for confidence score (1 - MAE) in combined early-stopping objective (default: config buddy.train_defaults.combined_w_conf or 0.3)",
    )


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    """Add feature engineering and curriculum arguments."""
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="For train-buddy: keep only top-K most correlated numeric features (disables warm-start)",
    )
    parser.add_argument(
        "--feature-curriculum",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For train-buddy: enable/disable feature curriculum (CLI overrides config). Cannot be used with --top-features.",
    )
    parser.add_argument(
        "--curriculum-ks",
        default=None,
        help=(
            "For train-buddy --feature-curriculum: comma-separated K schedule (0 = all). "
            f"If unset, uses config buddy.curriculum_ks or default {DEFAULT_CURRICULUM_KS}."
        ),
    )
    parser.add_argument(
        "--ignore-input-mismatches",
        action="store_true",
        help="Best-effort mode: downgrade some input/PCA/curriculum mismatches to warnings.",
    )
    parser.add_argument(
        "--disable-tier2-on-mismatch",
        action="store_true",
        help="If Tier-2 calibration encounters input mismatches, skip Tier-2 instead of best-effort.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=None,
        help="For train-buddy: apply rolling median to close before EMA smoothing (e.g. 3,5)",
    )
    parser.add_argument(
        "--all-features",
        "-A",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all engineered features (technical indicators, statistical, time, lag, rolling) for training. (default: enabled; use --no-all-features to disable)",
    )


def _add_tier2_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Tier-2 calibration control arguments."""
    parser.add_argument(
        "--tier2-calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: enable/disable Tier-2 TP/SL calibration + labeling (default enabled)",
    )
    parser.add_argument(
        "--tier2-calibration-stride",
        type=int,
        default=None,
        help="For train-buddy: stride for Tier-2 labeling/calibration simulation (higher = faster, default: config buddy.train_defaults.tier2_calibration_stride or 5)",
    )
    parser.add_argument(
        "--tier2-horizon-candles",
        type=int,
        default=None,
        help="For train-buddy: Tier-2 TP/SL max horizon in candles (lower = faster). Default uses config.",
    )


def _add_data_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """Add data quality filtering arguments."""
    parser.add_argument(
        "--vol-norm-window",
        type=int,
        default=None,
        help="For train-buddy: normalize returns by rolling volatility when ranking top features (legacy) (e.g. 50)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=None,
        help="For train-buddy: drop candles with volume below this threshold before training",
    )
    parser.add_argument(
        "--spread-filter",
        action="store_true",
        help="For train-buddy: drop candles with abnormally wide bid/ask spread (requires bid_close+ask_close)",
    )
    parser.add_argument(
        "--spread-pctl",
        type=float,
        default=0.99,
        help="For train-buddy: spread percentile used as filter threshold (default 0.99)",
    )
    parser.add_argument(
        "--spread-mult",
        type=float,
        default=3.0,
        help="For train-buddy: also filter if spread > (median * mult) (default 3.0)",
    )
    parser.add_argument(
        "--no-train-smoothing",
        action="store_true",
        help="For train-buddy: disable candle noise reduction (resample to M5 + EMA(14) close)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Buddy training (default 42)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional tag to version outputs (writes buddy_tf_<tag>.keras + meta)",
    )


def _add_oanda_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OANDA-related arguments."""
    parser.add_argument(
        "--instrument",
        "-I",
        default="EUR_USD",
        help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="H1",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=5000,
        help="How many candles to fetch for training (default: 5000)",
    )
    parser.add_argument(
        "--oanda-live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: fetch fresh candles from OANDA PRACTICE (default: enabled; use --no-oanda-live to use --csv file instead)",
    )
    parser.add_argument(
        "--oanda-price",
        default="MBA",
        help="OANDA price component for candles: M, B, A, or MBA (used by --oanda-live)",
    )
    parser.add_argument(
        "--oanda-save-csv",
        default=None,
        help="Optional path to write fetched OANDA candles CSV (used by --oanda-live)",
    )
    parser.add_argument(
        "--master-pair",
        type=str,
        default=None,
        help="For buddy: force model source pair for inference (e.g., GBP_JPY when trading EUR_JPY).",
    )
    parser.add_argument(
        "--interactive-master",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For buddy: interactively choose model source pair from correlation group (default: enabled).",
    )


def _add_broker_arguments(parser: argparse.ArgumentParser) -> None:
    """Add broker selection and configuration arguments (US-004)."""
    parser.add_argument(
        "--broker",
        type=str,
        choices=["oanda", "ibkr"],
        default="oanda",
        help="Broker to use for execution and data fetching: oanda (OANDA v20) | ibkr (Interactive Brokers) (default: oanda)",
    )
    parser.add_argument(
        "--oanda-account-id",
        type=str,
        default=None,
        help="OANDA account ID (e.g., 001-001-123456-001). If not provided, uses OANDA_ACCOUNT_ID env var.",
    )
    parser.add_argument(
        "--oanda-api-key",
        type=str,
        default=None,
        help="OANDA API key for authentication. If not provided, uses OANDA_API_KEY env var.",
    )
    parser.add_argument(
        "--oanda-environment",
        type=str,
        choices=["practice", "live"],
        default="practice",
        help="OANDA environment: practice (demo) | live (real money, use with caution) (default: practice)",
    )
    parser.add_argument(
        "--ibkr-host",
        type=str,
        default="127.0.0.1",
        help="Interactive Brokers TWS/IBGateway host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ibkr-port",
        type=int,
        default=7497,
        help="Interactive Brokers TWS port (7497) or IBGateway live (4002) (default: 7497)",
    )
    parser.add_argument(
        "--ibkr-client-id",
        type=int,
        default=1,
        help="Interactive Brokers unique client ID (default: 1)",
    )


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add validation and test arguments."""
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="For test: only count predictions with confidence >= this value (e.g., 0.6)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="For validate: validate all trained pair models",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=24,
        help="For validate: hours ahead to measure actual outcome (default: 24)",
    )


def _add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    """Add scan command arguments."""
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="For scan: comma-separated pairs to scan (e.g., EUR_USD,GBP_USD,USD_JPY)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="For scan/analyze: number of top results to show (0 = all)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        dest="no_train",
        help="For scan: skip the train prompt after scanning (deprecated)",
    )
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        dest="skip_execute",
        help="For scan: skip the trade execution prompt after scanning",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="For scan: enable continuous watch mode (scans repeatedly)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="For scan --watch: minutes between scans (default: 5)",
    )
    parser.add_argument(
        "--auto-execute",
        action="store_true",
        help="For scan: automatically execute passing/agent-promoted trades (also works with --watch)",
    )
    parser.add_argument(
        "--diversified",
        "-d",
        action="store_true",
        help="For scan: auto-filter correlated pairs (only show best from each correlation cluster)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For scan: bypass session-hour filter and scan/trade outside peak liquidity hours",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["conservative", "balanced", "aggressive", "smart", "futures_paper", "futures_live"],
        default="balanced",
        help="For buddy/scan: gate profile tuning (conservative|balanced|aggressive|smart|futures_paper|futures_live)",
    )
    parser.add_argument(
        "--futures",
        action="store_true",
        help="For scan: enable futures trading mode (shorthand for --broker ibkr --profile futures_paper)",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="For scan: comma-separated instruments to override profile pairs (e.g., ES,NQ,CL,GC)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="For scan: show compact Codex-style reasoning output instead of the full table",
    )


def _add_journal_arguments(parser: argparse.ArgumentParser) -> None:
    """Add journal and trade analysis arguments."""
    parser.add_argument(
        "--last",
        type=int,
        default=50,
        help="For trade-analysis: number of recent trades to analyze (default: 50)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="For trade-analysis: export report to JSON file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="For analyze: save plots to trained_data/visualizations",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="For journal: update trade results from OANDA",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="For journal: number of days of history to show (default: 30)",
    )
    parser.add_argument(
        "--import-trades",
        action="store_true",
        help="For journal: import untracked open trades from OANDA into journal",
    )


def _add_rl_arguments(parser: argparse.ArgumentParser) -> None:
    """Add RL position sizer arguments."""
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="For train-rl-* and train-buddy RL: timesteps override (default: use config)",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=None,
        help="For train-rl-sizer: training episodes (if set, overrides --timesteps)",
    )
    parser.add_argument(
        "--use-rl-sizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For buddy/scan: use RL position sizer instead of Kelly criterion. (default: enabled if RL model exists; use --no-use-rl-sizer to disable)",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="For train-buddy: skip RL position sizer training after ensemble",
    )


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Add trade execution arguments."""
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Paper equity for sizing (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=float,
        default=0.005,
        help="Risk per trade fraction (e.g. 0.005 = 0.5%%) (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--force-units",
        type=int,
        default=None,
        help="Override computed units and place exactly this many units (PRACTICE only).",
    )
    parser.add_argument(
        "--execute",
        "-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live trading on OANDA practice account (default: enabled; use --dry-run or --no-execute to simulate)",
    )
    parser.add_argument(
        "--force-execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bypass the training accuracy live-trading gate (PRACTICE only) (default: enabled; use --no-force-execute to enforce the gate).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable live trading, only simulate orders (overrides --execute)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="For buddy: keep Buddy warm and act on each new candle (default: off)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=1,
        help="For buddy --loop: stop after this many executed trades (0 = unlimited) (default 1)",
    )
    parser.add_argument(
        "--max-hold-minutes",
        type=float,
        default=None,
        help="Maximum holding time in minutes for opened trades (practice mode). After this time, open positions for the instrument will be closed automatically.",
    )
    parser.add_argument(
        "--force-margin",
        type=float,
        default=None,
        help="Force approximate margin in USD for the trade (converted to units using 1 unit = $0.05).",
    )
    parser.add_argument(
        "--aggressive-scaling",
        action="store_true",
        help="Enable $100K→$1M aggressive scaling strategy with slippage-adjusted Kelly sizing, smart order execution, and circuit breakers.",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="For buddy: launch the interactive Buddy REPL (interactive trading commands)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs (default: quiet output)",
    )


def _add_tf_performance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add TensorFlow performance tuning arguments."""
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Train device preference for TensorFlow: auto|cpu|gpu (default auto)",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mixed precision (mixed_float16) for Buddy training - 1.5-2x speedup on M1. (default: from config; use --no-mixed-precision to disable)",
    )
    parser.add_argument(
        "--steps-per-execution",
        type=int,
        default=None,
        help="Keras steps_per_execution to reduce per-step overhead (default: config buddy.train_defaults.steps_per_execution or 1)",
    )
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help="Enable Keras jit_compile (XLA) if supported by your TensorFlow build",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=None,
        help="Override window shuffle buffer size (else uses BUDDY_SHUFFLE_BUFFER or default 256)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Override tf.data prefetch: N>0 bounded, 0=AUTOTUNE (else uses BUDDY_PREFETCH or default 1)",
    )
    parser.add_argument(
        "--cache-val",
        action="store_true",
        help="Cache the validation window dataset in memory (can speed up validation; uses more RAM)",
    )
    parser.add_argument(
        "--combined-use-predict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --es-monitor combined: use an extra predict()+label pass per epoch to compute confidence MAE (best for calibration; disable for speed)",
    )
    parser.add_argument(
        "--shared-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a single shared LSTM encoder instead of 5 parallel heads (default: config buddy.train_defaults.shared_encoder or false)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tcn", "lstm", "xgboost", "ensemble", "attention_lstm"],
        default=None,
        help="Model architecture: ensemble (best accuracy), tcn (fast, M1 optimized), xgboost (gradient boosting), lstm (legacy) (default: config model.type or tcn)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-batch timing (helps distinguish TF tracing warmup from true slow steps)",
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Keras fit verbosity for train-buddy: 0=silent, 1=progress bar, 2=one line per epoch (default: config buddy.train_defaults.fit_verbose or 1)",
    )


def _add_intelligent_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """Add LLM-enhanced intelligent mode arguments."""
    parser.add_argument(
        "--intelligent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM-enhanced intelligent mode: adds reasoning, multi-modal fusion, and adaptive learning on top of Buddy's predictions. (default: enabled; use --no-intelligent to disable)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate detailed causal reasoning explaining why Buddy recommends this trade (Market Context → Signal → Key Drivers → Risks → Final Call).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "claude", "openai", "auto"],
        default="auto",
        help="LLM provider for intelligent mode: claude (Anthropic API, default), openai, ollama (local), or auto (detect from env vars).",
    )
    parser.add_argument(
        "--llm-enhance/--no-llm-enhance",
        dest="llm_enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deep LLM integration: dynamic gate thresholds (adjusted for edge cases), smarter sentiment aggregation, and context-aware trading. (default: enabled; use --no-llm-enhance to disable)",
    )


def _add_continual_learning_arguments(parser: argparse.ArgumentParser) -> None:
    """Add continual learning and scheduling arguments."""
    parser.add_argument(
        "--retrain-interval",
        type=float,
        default=168.0,
        help="Recommended hours between retraining sessions (default: 168 = 1 week). Used by drift detection.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.03,
        help="Performance drop threshold for drift detection (default: 0.03 = 3%% accuracy drop triggers drift alert).",
    )
    parser.add_argument(
        "--feature-drift-threshold",
        type=float,
        default=0.10,
        help="Feature distribution shift threshold (default: 0.10 = 10%% change in feature means triggers drift).",
    )
    parser.add_argument(
        "--enable-drift-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic drift detection after each training session. (default: enabled; use --no-enable-drift-check to disable)",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC penalty strength for preventing catastrophic forgetting (default: 1000.0). Higher = stronger constraint.",
    )
    parser.add_argument(
        "--ewc-gamma",
        type=float,
        default=0.95,
        help="EWC Fisher decay factor when adding new tasks (default: 0.95). Lower = faster forgetting of old tasks.",
    )
    parser.add_argument(
        "--replay-mix-ratio",
        type=float,
        default=0.20,
        help="Fraction of training batch from replay buffer (default: 0.20 = 20%% old data mixed in).",
    )
    parser.add_argument(
        "--replay-capacity-ratio",
        type=float,
        default=0.10,
        help="Replay buffer capacity as fraction of total samples seen (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.999,
        help="EMA smoothing factor for shadow weights (default: 0.999). Higher = more stable, slower adaptation.",
    )
    parser.add_argument(
        "--ema-update-freq",
        type=int,
        default=16,
        help="Update EMA weights every N training steps (default: 16).",
    )
    parser.add_argument(
        "--disable-continual-learning",
        action="store_true",
        help="Disable all continual learning features (EMA, EWC, replay buffer) for this training session.",
    )
    parser.add_argument(
        "--disable-ewc",
        action="store_true",
        help="Disable EWC (Elastic Weight Consolidation) for continual learning. EWC is enabled by default.",
    )


def _add_enterprise_arguments(parser: argparse.ArgumentParser) -> None:
    """Add enterprise training and MLOps arguments."""
    parser.add_argument(
        "--enterprise",
        action="store_true",
        default=True,
        help="Enable enterprise training features: MLflow tracking, walk-forward CV, bootstrap CI, statistical validation. (default: enabled)",
    )
    parser.add_argument(
        "--no-enterprise",
        action="store_true",
        help="Disable enterprise training features for faster training.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of walk-forward cross-validation folds (default: 5). Set to 0 to disable CV.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=True,
        help="Enable bootstrap confidence intervals for metrics (95%% CI). (default: enabled)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: auto-generated from instrument/granularity).",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate professional markdown training report with statistical analysis. (default: enabled)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the training report.",
    )


def _add_monitoring_arguments(parser: argparse.ArgumentParser) -> None:
    """Add monitoring command arguments."""
    parser.add_argument(
        "--decisions",
        action="store_true",
        help="For model-status: show persisted last trade and last no-trade decisions from Buddy runtime state.",
    )
    parser.add_argument(
        "--monitor-alerts",
        action="store_true",
        help="For monitor command: show detailed alerts",
    )
    parser.add_argument(
        "--monitor-drift",
        action="store_true",
        help="For monitor command: show model drift history",
    )
    parser.add_argument(
        "--monitor-report",
        action="store_true",
        help="For monitor command: generate full monitoring report",
    )
    parser.add_argument(
        "--monitor-limit",
        type=int,
        default=20,
        help="For monitor command: limit for drift history display (default: 20)",
    )
    # Training visualization options
    parser.add_argument(
        "--training",
        action="store_true",
        help="For monitor command: show training visualization mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="For monitor command: show model architecture (TCN, TRANSFORMER)",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="For monitor command: replay saved training history file",
    )


def _add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Weights & Biases experiment tracking arguments."""
    parser.add_argument(
        "--wandb",
        action="store_true",
        dest="wandb_enabled",
        help="Enable Weights & Biases experiment tracking for training runs.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ml_engine_fx",
        help="W&B project name (default: ml_engine_fx).",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from instrument/granularity/model).",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags (e.g., 'tcn,production,hotfix').",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run W&B in offline mode (sync later with `wandb sync`).",
    )
    parser.add_argument(
        "--wandb-no-gradients",
        action="store_false",
        dest="wandb_gradients",
        help="Disable gradient/activation histogram logging (faster, less memory).",
    )
    parser.add_argument(
        "--wandb-no-system",
        action="store_false",
        dest="wandb_system",
        help="Disable system metrics logging (GPU/CPU/memory).",
    )


def _add_multi_pair_arguments(parser: argparse.ArgumentParser) -> None:
    """Add multi-pair training arguments."""
    # Note: --instruments is already defined in the main argument group (line ~467).
    # Do NOT re-add it here — it would conflict with the existing definition.
    parser.add_argument(
        "--multi-pair",
        action="store_true",
        help="For train-buddy: train foundation model on multiple pairs simultaneously",
    )
    parser.add_argument(
        "--foundation-pairs",
        type=str,
        default=None,
        help="For train-buddy --multi-pair: comma-separated pairs (default: EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD)",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="For suggest-improvements: automatically apply approved hyperparam changes to config.",
    )


def _add_correlation_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add correlation-based transfer learning arguments."""
    # Simple transfer command arguments
    parser.add_argument(
        "--from-pair",
        type=str,
        default=None,
        help="For train-transfer: source pair to transfer FROM (e.g., EUR_JPY).",
    )
    parser.add_argument(
        "--to-pairs",
        type=str,
        default=None,
        help="For train-transfer: target pairs to transfer TO (e.g., GBP_JPY,AUD_JPY). Use --auto for auto-detect.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        dest="auto_detect_pairs",
        help="For train-transfer: auto-detect correlated pairs to transfer to.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="For train-transfer: train source model fresh instead of using existing.",
    )
    # Legacy flags (still supported but hidden from main help)
    parser.add_argument("--correlation-transfer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--master-pairs", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-pairs", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--correlation-threshold", type=float, default=0.70, help=argparse.SUPPRESS)
    parser.add_argument("--skip-master-training", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-epochs", type=int, default=50, help=argparse.SUPPRESS)


def _add_learn_arguments(parser: argparse.ArgumentParser) -> None:
    """Add learning loop CLI arguments (buddy learn command)."""
    learn_group = parser.add_argument_group("Learn", "Self-improvement learning loop commands")
    learn_group.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="For learn: run trade outcome analysis on all unanalyzed journal entries",
    )
    learn_group.add_argument(
        "--promote",
        action="store_true",
        default=False,
        help="For learn: run promotion check and promote qualifying patterns to rules",
    )
    learn_group.add_argument(
        "--consolidate",
        action="store_true",
        default=False,
        help="For learn: run learnings consolidation (archive promoted, group by category)",
    )
    learn_group.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="For learn: print improvement tracker report (total trades, win rate, trends)",
    )
    learn_group.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="For learn: print current state from .claude/state.json and recent learnings",
    )


def _add_feedback_arguments(parser: argparse.ArgumentParser) -> None:
    """Add feedback-status and train-offline-sizer arguments."""
    fb_group = parser.add_argument_group("Feedback", "Feedback loop and offline sizer commands")
    fb_group.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=False,
        help="For feedback-status: output as JSON",
    )
    fb_group.add_argument(
        "--iql-journal",
        dest="journal",
        default=None,
        help="For train-offline-sizer: path to trade journal (default: trained_data/trade_journal_rl.json)",
    )
    fb_group.add_argument(
        "--iql-epochs",
        dest="iql_epochs",
        type=int,
        default=100,
        help="For train-offline-sizer: number of training epochs (default: 100)",
    )
    fb_group.add_argument(
        "--iql-dry-run",
        dest="iql_dry_run",
        action="store_true",
        default=False,
        help="For train-offline-sizer: show dataset stats without training",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser.

    Returns:
        Configured ArgumentParser with all CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="ML Engine Trading Bot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )

    # Add all argument groups
    _add_core_arguments(parser)
    _add_training_arguments(parser)
    _add_feature_arguments(parser)
    _add_tier2_arguments(parser)
    _add_data_quality_arguments(parser)
    _add_oanda_arguments(parser)
    _add_broker_arguments(parser)  # US-004: Broker selection and configuration
    _add_validation_arguments(parser)
    _add_scan_arguments(parser)
    _add_journal_arguments(parser)
    _add_rl_arguments(parser)
    _add_execution_arguments(parser)
    _add_tf_performance_arguments(parser)
    _add_intelligent_mode_arguments(parser)
    _add_continual_learning_arguments(parser)
    _add_enterprise_arguments(parser)
    _add_monitoring_arguments(parser)
    _add_wandb_arguments(parser)
    _add_multi_pair_arguments(parser)
    _add_correlation_transfer_arguments(parser)
    _add_learn_arguments(parser)
    _add_feedback_arguments(parser)

    return parser


# =============================================================================
# TRANSFER SUBCOMMAND PARSER (Progressive Disclosure)
# =============================================================================

TRANSFER_EPILOG = """
EXAMPLES:
  buddy transfer                         # Interactive mode - select from existing models
  buddy transfer --auto                  # Auto-select best master and correlated targets
  buddy transfer --from EUR_JPY          # Transfer from EUR_JPY to all correlated pairs
  buddy transfer --from EUR_JPY --to GBP_JPY,AUD_JPY  # Explicit transfer

ADVANCED USAGE:
  buddy transfer --from EUR_JPY --advanced --threshold 0.80 --epochs 30
  buddy transfer --auto --advanced --frozen-layers 3 --ewc-lambda 200
"""


def create_transfer_parser() -> argparse.ArgumentParser:
    """Create the transfer subcommand parser with progressive disclosure.

    Implements tiered CLI complexity:
    - Basic flags always visible (--from, --to, --auto, --fresh)
    - Advanced flags only visible with --advanced flag

    Returns:
        Configured ArgumentParser for transfer command.
    """
    parser = argparse.ArgumentParser(
        prog="buddy transfer",
        description="Transfer learning from a master model to correlated pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TRANSFER_EPILOG,
    )

    # =========================================================================
    # BASIC FLAGS (Always Visible)
    # =========================================================================
    basic_group = parser.add_argument_group("basic options")

    basic_group.add_argument(
        "--from",
        dest="from_pair",
        type=str,
        default=None,
        help="Source pair to transfer from (e.g., EUR_JPY). Default: auto-detect best master.",
    )
    basic_group.add_argument(
        "--to",
        dest="to_pairs",
        type=str,
        default=None,
        help="Target pairs, comma-separated (e.g., GBP_JPY,AUD_JPY). Default: auto-detect correlated.",
    )
    basic_group.add_argument(
        "--auto",
        action="store_true",
        dest="auto_mode",
        help="Zero-config mode: auto-select best master and correlated targets.",
    )
    basic_group.add_argument(
        "--fresh",
        action="store_true",
        help="Train source model fresh instead of using existing.",
    )
    basic_group.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file.",
    )
    basic_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs.",
    )

    # =========================================================================
    # ADVANCED FLAG (Gate for expert options)
    # =========================================================================
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Show advanced options for expert tuning.",
    )

    # =========================================================================
    # ADVANCED FLAGS (Expert tuning - conceptually gated by --advanced)
    # =========================================================================
    advanced_group = parser.add_argument_group(
        "advanced options (requires --advanced for full benefit)"
    )

    advanced_group.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Correlation threshold for target selection (default: 0.70).",
    )
    advanced_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Transfer training epochs (default: 50).",
    )
    advanced_group.add_argument(
        "--lr-factor",
        type=float,
        default=0.03,
        help="Learning rate reduction factor for transfer (default: 0.03 = 33x reduction).",
    )
    advanced_group.add_argument(
        "--frozen-layers",
        type=int,
        default=2,
        help="Number of encoder layers to freeze during transfer (default: 2).",
    )
    advanced_group.add_argument(
        "--ewc-lambda",
        type=float,
        default=100.0,
        help="EWC penalty strength for knowledge preservation (default: 100.0).",
    )
    advanced_group.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience for transfer training (default: 15).",
    )
    advanced_group.add_argument(
        "--granularity",
        "-g",
        default="H1",
        help="OANDA candle granularity for data fetch (default: H1).",
    )
    advanced_group.add_argument(
        "--candles",
        "-n",
        type=int,
        default=5000,
        help="Number of candles to fetch for training (default: 5000).",
    )

    # =========================================================================
    # LEGACY FLAGS (Hidden, for backward compatibility)
    # =========================================================================
    parser.add_argument("--correlation-transfer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--master-pairs", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-pairs", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-master-training", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transfer-epochs", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--correlation-threshold", type=float, default=0.70, help=argparse.SUPPRESS)

    return parser


def parse_args() -> "Namespace":
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = create_argument_parser()
    return parser.parse_args()


__all__ = [
    "COMMANDS",
    "CLI_EPILOG",
    "TRANSFER_EPILOG",
    "create_argument_parser",
    "create_transfer_parser",
    "parse_args",
]
