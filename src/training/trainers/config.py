"""
Configuration classes for model trainers.

This module contains:
- TrainerConfig: Main configuration dataclass for all trainer types
- OverfitPreventionConfig: Configuration for overfit prevention callback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainerConfig:
    """Configuration for model trainers."""

    # Common
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20  # Original setting that achieved 60.9%
    min_epochs: int = 30  # Minimum epochs before early stopping can trigger
    verbose: int = 1
    quiet: bool = False  # Quiet mode: minimal console output, logs to file
    seq_len: int = (
        60  # Sequence length for sliding window (from config train_defaults.seq_len)
    )

    # TCN specific - AGGRESSIVE ANTI-OVERFITTING (val was 38%, worse than random)
    tcn_hidden_size: int = 16  # Reduced from 32 - smaller capacity
    tcn_num_layers: int = 1  # Reduced from 2 - simpler model
    tcn_kernel_size: int = 2  # Reduced from 3 - less receptive field
    tcn_dropout: float = 0.6  # Increased from 0.5 - heavy dropout
    tcn_l2_reg: float = 0.01  # NEW: L2 kernel regularization
    tcn_spatial_dropout: float = 0.3  # NEW: Spatial dropout on input
    tcn_noise_std: float = 0.05  # NEW: Input noise level

    # Transformer specific (for direction prediction)
    transformer_d_model: int = 32  # Model dimension
    transformer_num_heads: int = 4  # Number of attention heads
    transformer_num_layers: int = 2  # Number of encoder layers
    transformer_dff: int = 64  # Feedforward network dimension
    transformer_dropout: float = 0.2  # Dropout in transformer encoder layers

    # Transformer regularization — wired from YAML transformer section
    transformer_l2_reg: float = 0.003  # L2 kernel regularization weight
    transformer_input_noise: float = 0.02  # Gaussian noise on input
    transformer_spatial_dropout: float = 0.10  # SpatialDropout1D on input sequence
    transformer_projection_dropout: float = 0.10  # Dropout after input projection
    
    # === TIME-SERIES AUGMENTATION SETTINGS (NEW - Phase 4) ===
    use_augmentation: bool = True  # Enable time-series augmentation during training
    augmentation_noise_std: float = 0.01  # Gaussian noise std dev
    augmentation_scale_range: tuple = (0.98, 1.02)  # Random scaling range
    augmentation_time_mask_prob: float = 0.1  # Probability of time masking
    augmentation_time_mask_max_len: int = 5  # Max length of time mask

    # Output head settings (for direction model) - stored in metadata for fallback rebuild
    final_dense_units: int = 16  # Units in final dense layer before output
    final_dense_activation: str = (
        "tanh"  # Activation: tanh allows [-1,1] range for balanced sigmoid
    )
    final_dense_dropout: float = 0.10  # Dropout before output layer (reduced from 0.15)

    # === CLASS-BALANCED LOSS SETTINGS (NEW - for extreme class imbalance) ===
    use_class_balanced_loss: bool = (
        True  # Use CB Loss instead of Focal Loss for extreme imbalance
    )
    cb_beta: float = 0.9999  # Class-balanced hyperparameter (higher = more aggressive)
    cb_gamma: float = 2.0  # Focusing parameter for CB Loss

    # === FOCAL LOSS SETTINGS (Anti-Bias - fallback if CB Loss disabled) ===
    use_focal_loss: bool = False  # DISABLED: CB Loss is better for extreme imbalance
    focal_gamma: float = (
        2.0  # Focusing parameter - higher = more focus on hard examples
    )
    focal_alpha: float = 0.5  # Class balance (0.5 = balanced, <0.5 = favor SHORT)
    focal_label_smoothing: float = 0.1  # Label smoothing to prevent overconfidence

    # === ANTI-COLLAPSE LOSS SETTINGS (v2) ===
    use_anti_collapse_loss: bool = (
        True  # Use AntiCollapseFocalLoss with variance regularization
    )
    use_hybrid_cb_anticollapse: bool = (
        True  # NEW: Hybrid CB + Anti-Collapse loss (takes priority)
    )
    anti_collapse_base_variance_weight: float = (
        0.2  # INCREASED: Base variance weight (auto-tuned per pair) - was 0.1
    )
    anti_collapse_label_smoothing: float = (
        0.05  # Label smoothing for anti-collapse loss
    )
    sample_weight_max_multiplier: float = 5.0  # Max sample weight (increased from 3.0)

    # === MADL LOSS SETTINGS (Directional Profitability) ===
    use_madl_loss: bool = (
        False  # Use MADL instead of Focal Loss (optimizes for trading profit)
    )
    madl_direction_weight: float = 0.7  # Weight for direction component vs BCE in MADL

    # XGBoost specific
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    use_gpu: bool = False  # Enable GPU acceleration for XGBoost (A100)

    # Random Forest specific
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 10

    # ElasticNet specific (replaces Ridge for better feature selection)
    elasticnet_l1_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
    )
    elasticnet_alphas: Optional[List[float]] = (
        None  # Auto-generate via logspace if None
    )
    elasticnet_cv_splits: int = 5  # TimeSeriesSplit folds
    elasticnet_max_iter: int = 10000

    # Legacy Ridge alpha (deprecated, use elasticnet_* instead)
    ridge_alpha: float = 1.0

    # Checkpoint directory for pair-specific models
    checkpoint_dir: str = "trained_data/checkpoints"

    # === CONTINUAL LEARNING SETTINGS (2025) ===

    # EMA (Exponential Moving Average) for stable inference
    use_ema: bool = True  # Enable EMA shadow weights
    ema_decay: float = 0.999  # α for EMA: θ_ema = α * θ_ema + (1-α) * θ
    ema_update_every: int = 16  # Update EMA every N training steps
    use_ema_for_inference: bool = True  # Use EMA weights for prediction

    # EWC (Elastic Weight Consolidation) for multi-instrument learning
    # Enabled by default for continual learning. Use --disable-ewc to disable.
    use_ewc: bool = True  # Enable EWC penalty on warm-start
    ewc_lambda: float = (
        100.0  # Strength of EWC constraint (λ) - reduced from 1000 to allow learning
    )
    ewc_gamma: float = 0.95  # Decay for old Fisher values when adding new tasks

    # Replay Buffer for catastrophic forgetting prevention
    use_replay_buffer: bool = True  # Enable memory replay
    replay_buffer_ratio: float = 0.10  # Save 10% of training data
    replay_mix_ratio: float = 0.20  # Mix 20% replay samples during training
    replay_buffer_dir: str = "trained_data/replay"  # Replay buffer storage

    # Warm-start settings (CRITICAL for preventing catastrophic forgetting)
    warm_start_lr_factor: float = (
        0.1  # FIXED: 0.1 = 10x LR reduction (was 0.01 = 100x, too aggressive!)
    )
    warm_start_freeze_encoder: bool = (
        True  # Freeze ALL transformer encoder layers (not just transformer_0)
    )
    warm_start_unfreeze_epochs: int = (
        10  # Unfreeze encoder after N epochs (0 = never unfreeze)
    )
    warm_start_gradual_unfreeze: bool = (
        True  # FIXED: Now enabled - gradually unfreeze layers from top to bottom
    )

    # Drift detection for auto-retraining
    drift_threshold: float = 0.03  # Retrain if val_acc drops by >3%

    # === OVERFIT PREVENTION SETTINGS ===
    overfit_threshold: float = 0.08       # 8% train-val gap triggers warning
    critical_threshold: float = 0.15      # 15% gap triggers intervention (LR reduction)
    severe_threshold: float = 0.25        # 25% gap triggers early stop
    max_acceptable_gap: float = 0.12      # Won't save checkpoint if gap > 12%
    overfit_patience_epochs: int = 3      # Consecutive overfit epochs before intervention
    auto_adjust_dropout: bool = True      # Dynamically increase dropout on overfitting
    auto_reduce_lr: bool = True           # Dynamically reduce LR on overfitting
    max_dropout_increase: float = 0.3     # Cap on dropout increase

    # === SWA (Stochastic Weight Averaging) ===
    enable_swa: bool = True               # Average weights in final 25% for flatter optima
    swa_start_fraction: float = 0.75      # Start SWA at 75% of training
    swa_lr_factor: float = 0.5            # SWA constant LR = initial_lr * factor

    # === COSINE ANNEALING WITH WARM RESTARTS ===
    enable_cosine_restarts: bool = True    # Periodic LR resets to escape sharp minima
    cosine_restart_period: int = 10       # Restart every N epochs
    cosine_restart_lr_mult: float = 0.8   # Each restart uses 80% of prev LR

    # === WARM-START OVERFIT RECOVERY ===
    enable_warmstart_detection: bool = True
    warmstart_reset_threshold: float = 0.15  # If initial gap > 15%, intervene
    weight_perturbation_scale: float = 0.02  # Noise to break memorization
    reset_optimizer_on_overfit: bool = True   # Reset momentum on critical overfit

    # Feature selection settings
    use_feature_selection: bool = True  # Enable RF importance-based feature selection
    feature_selection_method: str = "random_forest"  # 'random_forest' or 'f_test'
    top_k_features: int = 50  # Number of top features to select


@dataclass
class OverfitPreventionConfig:
    """Configuration for OverfitPreventionCallback.

    Groups all configurable parameters to reduce __init__ parameter count
    and improve code readability.
    """
    # Threshold settings
    overfit_threshold: float = 0.08  # 8% gap triggers warning
    critical_threshold: float = 0.12  # 12% gap triggers intervention
    severe_threshold: float = 0.20  # 20% gap triggers early stop
    max_acceptable_gap: float = 0.10  # Won't save checkpoint if gap > 10%
    patience_epochs: int = 2  # Epochs before intervention
    auto_adjust_dropout: bool = True
    auto_reduce_lr: bool = True
    max_dropout_increase: float = 0.3

    # SWA settings - helps find flatter minima that generalize better
    enable_swa: bool = True  # Averages weights for better generalization
    swa_start_fraction: float = 0.5  # Start SWA at 50% of training
    swa_lr_factor: float = 0.5  # SWA uses lower constant LR

    # Cosine restart settings - helps escape local minima
    enable_cosine_restarts: bool = True  # Warm restarts improve convergence
    restart_period: int = 15  # Restart every 15 epochs
    restart_lr_mult: float = 0.9  # Each restart uses 90% of prev LR

    # Mixup augmentation (applied at batch level)
    enable_mixup: bool = False  # Disabled by default
    mixup_alpha: float = 0.2

    # Warm-start overfit recovery
    enable_warmstart_detection: bool = True
    warmstart_reset_threshold: float = 0.15  # If initial gap > 15%, intervene
    weight_perturbation_scale: float = 0.02  # Noise to break memorization
    reset_optimizer_on_overfit: bool = True  # Reset momentum on critical overfit

    # Continual learning: Previous best accuracy (prevents saving worse models)
    warm_start_best_acc: float = 0.0  # Set from loaded checkpoint
