"""Configuration dataclasses and constants for CLI module.

This module contains all configuration dataclasses and constants
used throughout the CLI, extracted from the original main.py for
better modularity and reduced complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Project root - computed relative to this file's location
_CLI_MODULE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _CLI_MODULE_ROOT.parent.parent

# Default paths
DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "config" / "config_improved_H1.yaml")
MODEL_DIR_PATH = "trained_data/models"

# Formatting constants
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"
DEFAULT_CURRICULUM_KS = "32,64,128,0"
UTC_OFFSET_SUFFIX = "+00:00"

# Model file constants
BUDDY_META_FILENAME = "buddy_tf.meta.json"
BUDDY_KERAS_FILENAME = "buddy_tf.keras"
TRANSFORMER_DIRECTION_FILENAME = "transformer_direction.keras"
TCN_DIRECTION_FILENAME = "tcn_direction.keras"
TRANSFORMER_REGIME_FILENAME = "transformer_regime.keras"
XGB_MOMENTUM_FILENAME = "xgb_momentum.pkl"
RF_RISK_FILENAME = "rf_risk.pkl"
RIDGE_CONFIDENCE_FILENAME = "ridge_confidence.pkl"
TCN_VOLATILITY_REGIME_FILENAME = "tcn_volatility_regime.keras"
MODULAR_ENSEMBLE_META_FILENAME = "modular_ensemble.meta.json"

# UI constants
STATUS_DISABLED = "✗ Disabled"
HEADER_STYLE_CYAN = "bold cyan"
RANDOM_FOREST_NAME = "Random Forest"
CONFIDENCE_OUTPUT = "confidence (0-100)"

# Floating point comparison epsilon
FLOAT_EPSILON = 1e-6

# Valid OANDA FX instruments (major, minor, and exotic pairs)
VALID_OANDA_INSTRUMENTS: set[str] = {
    # Major pairs
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    # Cross pairs
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CHF", "NZD_CAD",
    # Exotic pairs
    "USD_SGD", "USD_HKD", "USD_MXN", "USD_ZAR", "USD_TRY", "USD_SEK",
    "USD_NOK", "USD_DKK", "USD_PLN", "USD_HUF", "USD_CZK", "USD_THB",
    "USD_INR", "USD_CNH", "EUR_SEK", "EUR_NOK", "EUR_DKK", "EUR_PLN",
    "EUR_HUF", "EUR_CZK", "EUR_TRY", "EUR_ZAR",
    "GBP_SGD", "GBP_PLN", "GBP_ZAR",
    "AUD_SGD", "AUD_HKD",
    "SGD_JPY", "SGD_CHF", "HKD_JPY", "TRY_JPY", "ZAR_JPY", "MXN_JPY",
}


@dataclass(frozen=True)
class OandaFetchOptions:
    """Options for fetching data from OANDA API."""

    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 5000
    price: str = "MBA"
    save_csv: str | None = None


@dataclass(frozen=True)
class BuddyTrainingAdvancedOptions:
    """Advanced training options for Buddy model.

    Includes early stopping configuration, feature selection,
    and tier-2 calibration settings.
    """

    init_from: str | None = None
    es_monitor: str = "direction"  # direction|combined|val_loss|loss
    combined_w_dir: float = 0.7
    combined_w_conf: float = 0.3
    train_smoothing: bool = True
    top_features: int | None = None
    median_window: int | None = None
    vol_norm_window: int | None = None
    min_volume: int | None = None
    spread_filter: bool = False
    spread_pctl: float = 0.99
    spread_mult: float = 3.0

    # Tier-2: calibrated win probability (TP-before-SL) using fixed SL/TP.
    tier2_calibrate: bool = True
    tier2_stop_loss_pips: float | None = None
    tier2_take_profit_pips: float | None = None
    tier2_spread_fallback_pips: float | None = None
    tier2_horizon_candles: int | None = None
    tier2_tie_break: str | None = None
    tier2_calibration_bins: int = 20
    tier2_calibration_stride: int = 1


@dataclass(frozen=True)
class BuddyTrainingOptions:
    """Training options for Buddy model.

    M1 Metal Optimizations Applied:
    - model_type: "tcn" - 2-3x faster than LSTM (parallelizable convolutions)
    - batch_size: 128 (was 32) - better GPU utilization
    - mixed_precision: True (was False) - 1.5-2x speedup
    - steps_per_execution: 10 (was 1) - reduces Python overhead
    - lr: 0.0005 (was 0.001) - more stable convergence

    Model Types:
    - "tcn": Temporal Convolutional Network (RECOMMENDED for M1 - fastest)
    - "lstm": Legacy LSTM (slower on Metal, avoid unless necessary)
    - "attention_lstm": LSTM with attention (slower but more accurate)

    Enterprise Training (MLOps):
    - enterprise: Enable MLflow tracking, CV, bootstrap CI
    - cv_folds: Walk-forward cross-validation folds
    - bootstrap: Bootstrap confidence intervals

    Multi-Pair Pre-training:
    - multi_pair: Train foundation model on multiple pairs simultaneously
    - foundation_pairs: Comma-separated list of pairs for pre-training
    """

    oanda_fetch: OandaFetchOptions | None = None
    advanced: BuddyTrainingAdvancedOptions | None = None
    # None => use config (buddy.feature_curriculum)
    feature_curriculum: bool | None = None
    curriculum_ks: str | None = None
    pca_components: int | None = None
    seq_len: int = 60  # M1: Match config_m1_optimized.yaml (was 50)
    epochs: int = 200  # M1: Reduced from 300 with better early stopping
    batch_size: int = 128  # M1 CRITICAL: 128 is optimal for Metal (was 32)
    lr: float = 0.0005  # M1: Lower LR for stability (was 0.001)
    patience: int = 15  # M1: Increased patience (was 10)
    seed: int = 42
    run_tag: str | None = None
    all_features: bool = True
    device: str = "auto"  # auto|cpu|gpu
    ignore_input_mismatches: bool = False
    disable_tier2_on_mismatch: bool = False
    mixed_precision: bool = True  # M1 CRITICAL: Enable for 1.5-2x speedup (was False)
    steps_per_execution: int = 10  # M1 CRITICAL: Reduces Python overhead (was 1)
    jit_compile: bool = False  # Keep False for stability (XLA can cause issues)
    shuffle_buffer: int | None = None
    prefetch: int | None = None
    cache_val: bool = True  # M1: Cache validation data (was False)
    combined_use_predict: bool = True
    shared_encoder: bool = False  # Disabled when using TCN
    model_type: str = "ensemble"  # CRITICAL: Default to ensemble mode
    timing: bool = False
    fit_verbose: int = 1
    # Enterprise training (MLOps) options - ENABLED BY DEFAULT for ensemble
    enterprise: bool = True  # Enable enterprise features (MLflow, CV, bootstrap)
    cv_folds: int = 5  # Walk-forward cross-validation folds (0 to disable)
    bootstrap: bool = True  # Enable bootstrap confidence intervals
    bootstrap_samples: int = 1000  # Bootstrap iterations
    mlflow_experiment: str | None = None  # MLflow experiment name
    generate_report: bool = True  # Generate markdown report
    # Continual learning - EWC enabled by default
    enable_ewc: bool = True  # Enable EWC (Elastic Weight Consolidation)
    # Multi-pair pre-training - foundation model across instruments
    multi_pair: bool = False  # Enable multi-pair foundation training
    foundation_pairs: str | None = None  # Comma-separated pairs (default: majors)
    # RL position sizing training - train RL agent after ensemble
    train_rl_sizer: bool = True  # Enable RL position sizer training after ensemble
    rl_timesteps: int | None = None  # None => use rl_integration.position_sizing.timesteps
