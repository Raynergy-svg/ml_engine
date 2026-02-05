#!/usr/bin/env python3
"""CLI Configuration dataclasses for ML Engine Trading Bot."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class OandaFetchOptions:
    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 5000
    price: str = "MBA"
    save_csv: str | None = None


@dataclass(frozen=True)
class BuddyTrainingAdvancedOptions:
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
    shared_encoder: bool = False  # Disabled when using TCN (TCN doesn't need shared encoder)
    model_type: str = "tcn"  # M1 CRITICAL: TCN is 2-3x faster than LSTM on Metal
    timing: bool = False
    fit_verbose: int = 1
    # Enterprise training (MLOps) options - ENABLED BY DEFAULT for ensemble
    enterprise: bool = True  # Enable enterprise features (MLflow, CV, bootstrap)
    cv_folds: int = 5  # Walk-forward cross-validation folds (0 to disable)
    bootstrap: bool = True  # Enable bootstrap confidence intervals
    bootstrap_samples: int = 1000  # Bootstrap iterations
    mlflow_experiment: str | None = None  # MLflow experiment name
    generate_report: bool = True  # Generate markdown report
    # Continual learning - EWC enabled by default for better multi-instrument learning
    enable_ewc: bool = True  # Enable EWC (Elastic Weight Consolidation)
    # Multi-pair pre-training - foundation model across instruments
    multi_pair: bool = False  # Enable multi-pair foundation training
    foundation_pairs: str | None = None  # Comma-separated pairs (default: majors)
    # RL position sizing training - train RL agent after ensemble
    train_rl_sizer: bool = True  # Enable RL position sizer training after ensemble
    rl_timesteps: int = 500_000  # RL training timesteps (500k ~ 10-15 min on M1)
