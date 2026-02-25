"""Training configuration dataclasses for Buddy model training.

This module contains the core configuration dataclasses used for Buddy model training,
including OANDA fetch options, advanced training options, and main training options.

These configurations are optimized for Apple Silicon (M1/M2/M3) Metal GPU performance.

Example usage:
    from src.training.training_config import BuddyTrainingOptions, OandaFetchOptions

    # Create training options with OANDA fetch
    opts = BuddyTrainingOptions(
        oanda_fetch=OandaFetchOptions(instrument="EUR_USD", granularity="H1"),
        epochs=200,
        batch_size=128,
    )

    # Create from dict (backward compatibility)
    opts = BuddyTrainingOptions.from_dict({"epochs": 100, "batch_size": 64})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OandaFetchOptions:
    """Configuration for fetching data from OANDA API.

    Attributes:
        instrument: Trading instrument (e.g., "EUR_USD", "USD_JPY").
        granularity: Candle timeframe (e.g., "M5", "H1", "D").
        candles: Number of candles to fetch.
        price: Price type - "M" (mid), "B" (bid), "A" (ask), or combinations.
        save_csv: Optional path to save fetched data as CSV.
    """
    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 5000
    price: str = "MBA"
    save_csv: str | None = None

    def __post_init__(self) -> None:
        """Validate OANDA fetch options."""
        if self.candles < 1:
            raise ValueError(f"candles must be positive, got {self.candles}")
        if not self.instrument:
            raise ValueError("instrument cannot be empty")
        valid_granularities = {"S5", "S10", "S15", "S30", "M1", "M2", "M4", "M5",
                               "M10", "M15", "M30", "H1", "H2", "H3", "H4", "H6",
                               "H8", "H12", "D", "W", "M"}
        if self.granularity not in valid_granularities:
            raise ValueError(f"Invalid granularity: {self.granularity}. Must be one of {valid_granularities}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OandaFetchOptions:
        """Create OandaFetchOptions from a dictionary (for backward compatibility).

        Args:
            data: Dictionary with option values.

        Returns:
            New OandaFetchOptions instance.
        """
        return cls(
            instrument=data.get("instrument", "USD_JPY"),
            granularity=data.get("granularity", "M5"),
            candles=data.get("candles", 5000),
            price=data.get("price", "MBA"),
            save_csv=data.get("save_csv"),
        )


@dataclass(frozen=True)
class BuddyTrainingAdvancedOptions:
    """Advanced training options for Buddy model.

    Includes options for:
    - Checkpoint initialization
    - Early stopping configuration
    - Feature engineering (smoothing, top features, filtering)
    - Tier-2 calibration for win probability estimation

    Attributes:
        init_from: Path to checkpoint for weight initialization.
        es_monitor: Early stopping monitor metric (direction|combined|val_loss|loss).
        combined_w_dir: Weight for direction in combined metric.
        combined_w_conf: Weight for confidence in combined metric.
        train_smoothing: Enable training data smoothing.
        top_features: Number of top features to use (None = all).
        median_window: Window size for median filtering.
        vol_norm_window: Window for volatility normalization.
        min_volume: Minimum volume filter threshold.
        spread_filter: Enable spread-based filtering.
        spread_pctl: Percentile for spread filtering.
        spread_mult: Multiplier for spread filter.
        tier2_calibrate: Enable tier-2 win probability calibration.
        tier2_stop_loss_pips: Fixed stop loss in pips for tier-2.
        tier2_take_profit_pips: Fixed take profit in pips for tier-2.
        tier2_spread_fallback_pips: Fallback spread for tier-2 simulation.
        tier2_horizon_candles: Lookforward horizon for tier-2.
        tier2_tie_break: Tie-break strategy for tier-2.
        tier2_calibration_bins: Number of bins for calibration.
        tier2_calibration_stride: Stride for calibration sampling.
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

    def __post_init__(self) -> None:
        """Validate advanced options."""
        valid_monitors = {"direction", "combined", "val_loss", "loss"}
        if self.es_monitor not in valid_monitors:
            raise ValueError(f"es_monitor must be one of {valid_monitors}, got {self.es_monitor}")
        if not 0 <= self.combined_w_dir <= 1:
            raise ValueError(f"combined_w_dir must be in [0, 1], got {self.combined_w_dir}")
        if not 0 <= self.combined_w_conf <= 1:
            raise ValueError(f"combined_w_conf must be in [0, 1], got {self.combined_w_conf}")
        if self.tier2_calibration_bins < 1:
            raise ValueError(f"tier2_calibration_bins must be positive, got {self.tier2_calibration_bins}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuddyTrainingAdvancedOptions:
        """Create BuddyTrainingAdvancedOptions from a dictionary.

        Args:
            data: Dictionary with option values.

        Returns:
            New BuddyTrainingAdvancedOptions instance.
        """
        return cls(
            init_from=data.get("init_from"),
            es_monitor=data.get("es_monitor", "direction"),
            combined_w_dir=data.get("combined_w_dir", 0.7),
            combined_w_conf=data.get("combined_w_conf", 0.3),
            train_smoothing=data.get("train_smoothing", True),
            top_features=data.get("top_features"),
            median_window=data.get("median_window"),
            vol_norm_window=data.get("vol_norm_window"),
            min_volume=data.get("min_volume"),
            spread_filter=data.get("spread_filter", False),
            spread_pctl=data.get("spread_pctl", 0.99),
            spread_mult=data.get("spread_mult", 3.0),
            tier2_calibrate=data.get("tier2_calibrate", True),
            tier2_stop_loss_pips=data.get("tier2_stop_loss_pips"),
            tier2_take_profit_pips=data.get("tier2_take_profit_pips"),
            tier2_spread_fallback_pips=data.get("tier2_spread_fallback_pips"),
            tier2_horizon_candles=data.get("tier2_horizon_candles"),
            tier2_tie_break=data.get("tier2_tie_break"),
            tier2_calibration_bins=data.get("tier2_calibration_bins", 20),
            tier2_calibration_stride=data.get("tier2_calibration_stride", 1),
        )


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

    Attributes:
        oanda_fetch: Options for OANDA data fetching.
        advanced: Advanced training options.
        feature_curriculum: Enable feature curriculum learning (None = use config).
        curriculum_ks: Comma-separated curriculum K values.
        pca_components: Number of PCA components (None = no PCA).
        seq_len: Sequence length for time series.
        epochs: Maximum training epochs.
        batch_size: Training batch size (128 optimal for Metal).
        lr: Learning rate.
        patience: Early stopping patience.
        seed: Random seed for reproducibility.
        run_tag: Optional tag for this training run.
        all_features: Use all available features.
        device: Device selection (auto|cpu|gpu).
        ignore_input_mismatches: Ignore model input dimension mismatches.
        disable_tier2_on_mismatch: Disable tier-2 on feature mismatch.
        mixed_precision: Enable mixed precision training (FP16).
        steps_per_execution: Steps per execution (reduces Python overhead).
        jit_compile: Enable XLA JIT compilation.
        shuffle_buffer: Shuffle buffer size for dataset.
        prefetch: Prefetch buffer size.
        cache_val: Cache validation data in memory.
        combined_use_predict: Use predict for combined metrics.
        shared_encoder: Use shared encoder (disabled for TCN).
        model_type: Model architecture (tcn|lstm|attention_lstm).
        timing: Enable timing instrumentation.
        fit_verbose: Keras fit verbosity level.
        enterprise: Enable enterprise features (MLflow, CV, bootstrap).
        cv_folds: Walk-forward cross-validation folds.
        bootstrap: Enable bootstrap confidence intervals.
        bootstrap_samples: Number of bootstrap iterations.
        mlflow_experiment: MLflow experiment name.
        generate_report: Generate markdown training report.
        enable_ewc: Enable Elastic Weight Consolidation.
        multi_pair: Enable multi-pair foundation training.
        foundation_pairs: Comma-separated pairs for multi-pair training.
        train_rl_sizer: Train RL position sizer after ensemble.
        rl_timesteps: Optional RL training timesteps override.
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
    mixed_precision: bool = False  # Default off; set true in config for M1 Metal speedup
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
    rl_timesteps: int | None = None  # None => use rl_integration.position_sizing.timesteps
    # Weights & Biases (W&B) experiment tracking
    wandb_enabled: bool = False  # Enable W&B tracking (requires `pip install wandb`)
    wandb_project: str = "ml_engine_fx"  # W&B project name
    wandb_name: str | None = None  # Run name (None = auto-generate from instrument/granularity)
    wandb_tags: str | None = None  # Comma-separated tags for the run
    wandb_offline: bool = False  # Run in offline mode (sync later with `wandb sync`)
    wandb_gradients: bool = True  # Log gradient histograms (for diagnosing vanishing/exploding gradients)
    wandb_system: bool = True  # Log system metrics (GPU/CPU/memory)
    # Correlation-based transfer learning
    correlation_transfer: bool = False  # Enable correlation-based transfer learning
    master_pairs: str | None = None  # Comma-separated master pairs (e.g., "EUR_JPY,EUR_USD")
    target_pairs: str | None = None  # Comma-separated target pairs (e.g., "GBP_JPY,AUD_JPY")
    correlation_threshold: float = 0.70  # Min correlation to consider pairs related
    skip_master_training: bool = False  # Skip master training, use existing models
    transfer_epochs: int = 50  # Epochs for fine-tuning target pairs

    def __post_init__(self) -> None:
        """Validate training options."""
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.patience < 1:
            raise ValueError(f"patience must be positive, got {self.patience}")
        valid_devices = {"auto", "cpu", "gpu"}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        valid_model_types = {"tcn", "lstm", "attention_lstm", "transformer", "tft", "ensemble"}
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}, got {self.model_type}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuddyTrainingOptions:
        """Create BuddyTrainingOptions from a dictionary (for backward compatibility).

        Args:
            data: Dictionary with option values. Nested dicts for 'oanda_fetch'
                  and 'advanced' are automatically converted.

        Returns:
            New BuddyTrainingOptions instance.
        """
        oanda_fetch = None
        if "oanda_fetch" in data and data["oanda_fetch"]:
            oanda_fetch = OandaFetchOptions.from_dict(data["oanda_fetch"])

        advanced = None
        if "advanced" in data and data["advanced"]:
            advanced = BuddyTrainingAdvancedOptions.from_dict(data["advanced"])

        return cls(
            oanda_fetch=oanda_fetch,
            advanced=advanced,
            feature_curriculum=data.get("feature_curriculum"),
            curriculum_ks=data.get("curriculum_ks"),
            pca_components=data.get("pca_components"),
            seq_len=data.get("seq_len", 60),
            epochs=data.get("epochs", 200),
            batch_size=data.get("batch_size", 128),
            lr=data.get("lr", 0.0005),
            patience=data.get("patience", 15),
            seed=data.get("seed", 42),
            run_tag=data.get("run_tag"),
            all_features=data.get("all_features", True),
            device=data.get("device", "auto"),
            ignore_input_mismatches=data.get("ignore_input_mismatches", False),
            disable_tier2_on_mismatch=data.get("disable_tier2_on_mismatch", False),
            mixed_precision=data.get("mixed_precision", False),
            steps_per_execution=data.get("steps_per_execution", 10),
            jit_compile=data.get("jit_compile", False),
            shuffle_buffer=data.get("shuffle_buffer"),
            prefetch=data.get("prefetch"),
            cache_val=data.get("cache_val", True),
            combined_use_predict=data.get("combined_use_predict", True),
            shared_encoder=data.get("shared_encoder", False),
            model_type=data.get("model_type", "tcn"),
            timing=data.get("timing", False),
            fit_verbose=data.get("fit_verbose", 1),
            enterprise=data.get("enterprise", True),
            cv_folds=data.get("cv_folds", 5),
            bootstrap=data.get("bootstrap", True),
            bootstrap_samples=data.get("bootstrap_samples", 1000),
            mlflow_experiment=data.get("mlflow_experiment"),
            generate_report=data.get("generate_report", True),
            enable_ewc=data.get("enable_ewc", True),
            multi_pair=data.get("multi_pair", False),
            foundation_pairs=data.get("foundation_pairs"),
            train_rl_sizer=data.get("train_rl_sizer", True),
            rl_timesteps=data.get("rl_timesteps"),
            # W&B options
            wandb_enabled=data.get("wandb_enabled", False),
            wandb_project=data.get("wandb_project", "ml_engine_fx"),
            wandb_name=data.get("wandb_name"),
            wandb_tags=data.get("wandb_tags"),
            wandb_offline=data.get("wandb_offline", False),
            wandb_gradients=data.get("wandb_gradients", True),
            wandb_system=data.get("wandb_system", True),
            # Correlation transfer options
            correlation_transfer=data.get("correlation_transfer", False),
            master_pairs=data.get("master_pairs"),
            target_pairs=data.get("target_pairs"),
            correlation_threshold=data.get("correlation_threshold", 0.70),
            skip_master_training=data.get("skip_master_training", False),
            transfer_epochs=data.get("transfer_epochs", 50),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all option values.
        """
        result = {
            "feature_curriculum": self.feature_curriculum,
            "curriculum_ks": self.curriculum_ks,
            "pca_components": self.pca_components,
            "seq_len": self.seq_len,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "patience": self.patience,
            "seed": self.seed,
            "run_tag": self.run_tag,
            "all_features": self.all_features,
            "device": self.device,
            "ignore_input_mismatches": self.ignore_input_mismatches,
            "disable_tier2_on_mismatch": self.disable_tier2_on_mismatch,
            "mixed_precision": self.mixed_precision,
            "steps_per_execution": self.steps_per_execution,
            "jit_compile": self.jit_compile,
            "shuffle_buffer": self.shuffle_buffer,
            "prefetch": self.prefetch,
            "cache_val": self.cache_val,
            "combined_use_predict": self.combined_use_predict,
            "shared_encoder": self.shared_encoder,
            "model_type": self.model_type,
            "timing": self.timing,
            "fit_verbose": self.fit_verbose,
            "enterprise": self.enterprise,
            "cv_folds": self.cv_folds,
            "bootstrap": self.bootstrap,
            "bootstrap_samples": self.bootstrap_samples,
            "mlflow_experiment": self.mlflow_experiment,
            "generate_report": self.generate_report,
            "enable_ewc": self.enable_ewc,
            "multi_pair": self.multi_pair,
            "foundation_pairs": self.foundation_pairs,
            "train_rl_sizer": self.train_rl_sizer,
            "rl_timesteps": self.rl_timesteps,
            # W&B options
            "wandb_enabled": self.wandb_enabled,
            "wandb_project": self.wandb_project,
            "wandb_name": self.wandb_name,
            "wandb_tags": self.wandb_tags,
            "wandb_offline": self.wandb_offline,
            "wandb_gradients": self.wandb_gradients,
            "wandb_system": self.wandb_system,
            # Correlation transfer options
            "correlation_transfer": self.correlation_transfer,
            "master_pairs": self.master_pairs,
            "target_pairs": self.target_pairs,
            "correlation_threshold": self.correlation_threshold,
            "skip_master_training": self.skip_master_training,
            "transfer_epochs": self.transfer_epochs,
        }

        if self.oanda_fetch:
            result["oanda_fetch"] = {
                "instrument": self.oanda_fetch.instrument,
                "granularity": self.oanda_fetch.granularity,
                "candles": self.oanda_fetch.candles,
                "price": self.oanda_fetch.price,
                "save_csv": self.oanda_fetch.save_csv,
            }

        if self.advanced:
            result["advanced"] = {
                "init_from": self.advanced.init_from,
                "es_monitor": self.advanced.es_monitor,
                "combined_w_dir": self.advanced.combined_w_dir,
                "combined_w_conf": self.advanced.combined_w_conf,
                "train_smoothing": self.advanced.train_smoothing,
                "top_features": self.advanced.top_features,
                "median_window": self.advanced.median_window,
                "vol_norm_window": self.advanced.vol_norm_window,
                "min_volume": self.advanced.min_volume,
                "spread_filter": self.advanced.spread_filter,
                "spread_pctl": self.advanced.spread_pctl,
                "spread_mult": self.advanced.spread_mult,
                "tier2_calibrate": self.advanced.tier2_calibrate,
                "tier2_stop_loss_pips": self.advanced.tier2_stop_loss_pips,
                "tier2_take_profit_pips": self.advanced.tier2_take_profit_pips,
                "tier2_spread_fallback_pips": self.advanced.tier2_spread_fallback_pips,
                "tier2_horizon_candles": self.advanced.tier2_horizon_candles,
                "tier2_tie_break": self.advanced.tier2_tie_break,
                "tier2_calibration_bins": self.advanced.tier2_calibration_bins,
                "tier2_calibration_stride": self.advanced.tier2_calibration_stride,
            }

        return result


# Backward compatibility alias
TrainingOptions = BuddyTrainingOptions


__all__ = [
    "OandaFetchOptions",
    "BuddyTrainingAdvancedOptions",
    "BuddyTrainingOptions",
    "TrainingOptions",
]
