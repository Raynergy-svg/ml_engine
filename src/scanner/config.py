"""
Scanner Configuration Module.

Provides ScannerConfig dataclass with:
- Absolute path resolution (no relative path issues)
- Default pairs and pip values
- Non-interactive mode support
- Watch mode incremental caching settings
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve absolute path to project root
# config.py is at src/scanner/config.py, so 3 parents up to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config_improved_H1.yaml"

# Major FX pairs
MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

# Cross pairs
CROSS_PAIRS = [
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
]

DEFAULT_PAIRS = MAJOR_PAIRS + CROSS_PAIRS

# Pip values for position sizing
PIP_VALUES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_JPY": 0.01,
    "USD_CHF": 0.0001, "AUD_USD": 0.0001, "USD_CAD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_JPY": 0.01,
    "GBP_JPY": 0.01, "AUD_JPY": 0.01, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_CHF": 0.0001, "GBP_CHF": 0.0001,
}

SCAN_PROFILE_BALANCED = "balanced"
SCAN_PROFILES: Dict[str, Dict[str, Any]] = {
    # Uses configuration defaults (YAML + dataclass); no extra overrides.
    SCAN_PROFILE_BALANCED: {
        "blocked_pairs": ["EUR_USD"],
        "sub_inference_min_confidence": 0.60,
        "min_agent_consensus_ratio": 0.50,  # 6/12 agents minimum (was 0.33)
    },
    # Fewer trades, higher signal quality requirements.
    "conservative": {
        "blocked_pairs": ["EUR_USD"],
        "min_confidence": 58.0,
        "min_momentum": 0.28,
        "min_tcn_probability": 0.63,
        "max_drawdown_pct": 0.020,
        "final_score_threshold": 0.48,
        "max_uncertainty_std": 0.13,
        "min_atr_pips": 6.0,
        "min_volatility_regime": 2,
        "weighted_vote_threshold": 0.58,
        "sub_inference_min_confidence": 0.62,
        "sub_inference_vote_threshold": 0.72,
        "agent_promotion_min_confidence": 0.56,
        "max_uncertainty_score": 0.35,
        "max_model_disagreement": 0.45,
        "min_agent_consensus_ratio": 0.42,  # 5/12 agents minimum — stricter
        # Phase 33 (US-216): Infrastructure features safe for conservative mode
        "enable_memory_manager": True,
        "enable_health_registry": True,
        "enable_observation_consumer": True,
        "enable_module_activation": True,
        "enable_agent_lifecycle": True,
        "enable_confidence_calibration": True,
        "enable_concept_drift_detection": True,
        "enable_model_calibration": True,
        "enable_replay_validator": True,
        "enable_pair_performance_agent": True,
        "enable_multi_timeframe_agent": True,
    },
    # More trade frequency with looser gates (still risk-bounded).
    "aggressive": {
        "blocked_pairs": ["EUR_USD"],
        "enable_execution": True,  # Phase 30 (US-185)
        "min_confidence": 45.0,
        "min_momentum": 0.12,
        "min_tcn_probability": 0.58,
        "max_drawdown_pct": 0.035,
        "final_score_threshold": 0.42,
        "max_uncertainty_std": 0.18,
        "min_atr_pips": 3.0,
        "min_volatility_regime": 0,
        "weighted_vote_threshold": 0.52,
        "sub_inference_min_confidence": 0.45,
        "sub_inference_vote_threshold": 0.60,
        "agent_promotion_min_confidence": 0.50,
        "max_uncertainty_score": 0.48,
        "max_model_disagreement": 0.60,
        "min_agent_consensus_ratio": 0.33,  # 4/12 agents minimum — was 0.25
        # Phase 30 (US-185): RL features and soft uncertainty for aggressive profile
        "use_rl_sizer": True,
        "use_rl_gates": True,
        "use_rl_exits": True,
        "soft_uncertainty_blocking": True,
        "enable_agent_trade_promotion": True,
        # Phase 33 (US-212): Full feature parity with smart profile
        # --- Agent & signal features ---
        "enable_multi_timeframe_agent": True,
        "enable_pair_performance_agent": True,
        "enable_session_timing_agent": True,
        "enable_support_resistance_agent": True,
        "enable_trader_readiness_agent": True,  # Aura Phase 1 (US-200)
        "enable_news_risk_agent": True,  # Phase 33 (US-214)
        "enable_session_filter": True,  # Phase 33 (US-214)
        # --- LLM & model features ---
        "enable_llm_trade_analysis": True,
        "enable_true_ab_testing": True,
        "enable_confidence_calibration": True,
        "enable_model_calibration": True,
        "enable_model_bandit": True,
        "enable_model_routing": True,
        # --- Risk & sizing features ---
        "enable_dynamic_risk_allocation": True,
        "enable_kelly_sizing": True,
        "enable_entropy_sizing": True,
        "enable_market_impact": True,
        "enable_dynamic_sl_tp": True,
        "enable_dynamic_hedging": True,
        "enable_affinity_portfolio": True,
        "enable_live_position_management": True,
        "enable_position_timeout": True,
        # --- Execution features ---
        "enable_smart_execution": True,
        "enable_execution_routing": True,
        "enable_execution_quality_optimizer": True,
        "enable_execution_quality_tracking": True,
        # --- Intelligence & analysis features ---
        "enable_microstructure_regime": True,
        "enable_multi_horizon_fusion": True,
        "enable_trade_outcome_prediction": True,
        "enable_concept_drift_detection": True,
        "enable_ensemble_disagreement": True,
        "enable_lead_lag_detection": True,
        "enable_feature_attention": True,
        "enable_temporal_attention": True,
        "enable_causal_filtering": True,
        "enable_trade_explainability": True,
        "enable_regime_broadcast": True,
        "enable_regime_reward": True,
        "enable_observational_learning": True,
        "enable_adversarial_training": True,
        "enable_adaptive_lr": True,
        "enable_synthetic_crisis": True,
        "enable_training_augmentation": True,
        "enable_attention_feedback": True,
        "enable_pair_transfer": True,
        # --- Graph & consensus ---
        "use_heterogeneous_agents": True,
        "enable_graph_attention": True,
        "enable_agent_accuracy_matrix": True,
        "enable_pair_regime_agent_matrix": True,
        # --- Infrastructure ---
        "enable_threshold_optimizer": True,
        "enable_signal_timing": True,
        "enable_memory_manager": True,
        "enable_module_activation": True,
        "enable_agent_lifecycle": True,
        "enable_health_registry": True,
        "enable_observation_consumer": True,
        "enable_replay_validator": True,
        # --- ATR settings (aggressive: wider range) ---
        "atr_sl_multiplier": 1.0,
        "atr_tp_multiplier": 1.8,
        "min_sl_pips": 8.0,
        "max_sl_pips": 40.0,
        "min_tp_pips": 12.0,
        "max_tp_pips": 70.0,
        "min_risk_reward_ratio": 1.2,
        "sub_inference_tradeable_only": False,
        "sub_inference_max_candidates": 20,
        "use_hrp": True,
        "execution_strategy": "TWAP",
        "sl_tp_aggressiveness": 1.2,
    },
    # Smart: Agent-driven with RL integration.  Sub-inference agents run on
    # ALL candidates (not just gate-passed) so the agent team can promote
    # borderline setups that have strong multi-timeframe confluence.  Slightly
    # relaxed uncertainty ceiling to let agents make the final call while
    # maintaining risk gates.
    "smart": {
        "blocked_pairs": ["EUR_USD"],
        # Phase 29 (US-174): Enable trade execution in smart profile
        "enable_execution": True,
        "min_confidence": 48.0,
        "min_momentum": 0.15,
        "min_tcn_probability": 0.58,
        "max_drawdown_pct": 0.030,
        "final_score_threshold": 0.44,
        "max_uncertainty_std": 0.16,
        "min_atr_pips": 4.0,
        "min_volatility_regime": 0,
        "weighted_vote_threshold": 0.52,
        "sub_inference_tradeable_only": False,
        "sub_inference_min_confidence": 0.60,
        "sub_inference_vote_threshold": 0.60,
        "sub_inference_max_candidates": 15,
        "agent_promotion_min_confidence": 0.50,
        "max_uncertainty_score": 0.52,
        "max_model_disagreement": 0.65,
        "use_rl_sizer": True,
        "use_rl_gates": True,
        "use_rl_exits": True,
        "enable_agent_trade_promotion": True,
        # Soft uncertainty: penalize confidence instead of hard-blocking trades
        "soft_uncertainty_blocking": True,
        # ATR-based dynamic SL/TP (wider range for volatility-adaptive sizing)
        "atr_sl_multiplier": 1.2,
        "atr_tp_multiplier": 2.0,
        "min_sl_pips": 10.0,
        "max_sl_pips": 35.0,
        "min_tp_pips": 15.0,
        "max_tp_pips": 60.0,
        "high_prob_threshold": 0.65,
        "high_prob_tp_bonus": 15.0,
        # Multi-timeframe confluence agent (H1 → H4 → D1)
        "enable_multi_timeframe_agent": True,
        # Pair performance agent (adjusts confidence from historical W/L)
        "enable_pair_performance_agent": True,
        # Minimum risk:reward ratio to execute a trade
        "min_risk_reward_ratio": 1.2,
        # LLM deep analysis for losing trades
        "enable_llm_trade_analysis": True,
        # True A/B testing: load candidate model at scan time for inference comparison
        "enable_true_ab_testing": True,
        # Agent consensus: 6/12 minimum for smart (was 0.33)
        "min_agent_consensus_ratio": 0.50,
        # HRP portfolio optimization (US-073)
        "use_hrp": True,
        # Microstructure regime detection (US-074)
        "enable_microstructure_regime": True,
        # Smart execution (US-075)
        "enable_smart_execution": True,
        "execution_strategy": "TWAP",
        # Graph-attention consensus (US-076)
        "use_heterogeneous_agents": True,
        "enable_graph_attention": True,
        # Dynamic hedging (US-077)
        "enable_dynamic_hedging": True,
        # Observational learning (US-078)
        "enable_observational_learning": True,
        # Confidence calibration (US-079)
        "enable_confidence_calibration": True,
        # Dynamic SL/TP (US-080)
        "enable_dynamic_sl_tp": True,
        "sl_tp_aggressiveness": 1.0,
        # Multi-horizon fusion (US-081)
        "enable_multi_horizon_fusion": True,
        # Trade outcome prediction (US-082)
        "enable_trade_outcome_prediction": True,
        # Concept drift detection (US-083)
        "enable_concept_drift_detection": True,
        # Kelly portfolio sizing (US-084)
        "enable_kelly_sizing": True,
        # Ensemble disagreement signal (US-085)
        "enable_ensemble_disagreement": True,
        # Position timeout with time-decay (US-086)
        "enable_position_timeout": True,
        # Cross-pair lead-lag detection (US-087)
        "enable_lead_lag_detection": True,
        # Attention-based feature weighting (US-088)
        "enable_feature_attention": True,
        # Adversarial robustness training (US-089)
        "enable_adversarial_training": True,
        # Regime-aware RL reward shaping (US-090)
        "enable_regime_reward": True,
        # Adaptive LR scheduling (US-091)
        "enable_adaptive_lr": True,
        # Market impact position sizing (US-092)
        "enable_market_impact": True,
        # Bandit-based model selection (US-093)
        "enable_model_bandit": True,
        # SHAP-based trade explainability (US-094)
        "enable_trade_explainability": True,
        # Causal signal filtering (US-095)
        "enable_causal_filtering": True,
        # Synthetic crisis event generator (US-096)
        "enable_synthetic_crisis": True,
        # Execution quality tracking (US-097)
        "enable_execution_quality_tracking": True,
        # Entropy-based position sizing (US-098)
        "enable_entropy_sizing": True,
        # Regime event broadcasting (US-099)
        "enable_regime_broadcast": True,
        # Agent accuracy matrix (US-100)
        "enable_agent_accuracy_matrix": True,
        # Pair transfer learning (US-101)
        "enable_pair_transfer": True,
        # Temporal attention (US-102)
        "enable_temporal_attention": True,
        # Live position management (US-103)
        "enable_live_position_management": True,
        # Affinity portfolio sizing (US-104)
        "enable_affinity_portfolio": True,
        # Execution routing (US-105)
        "enable_execution_routing": True,
        # Model routing (US-106)
        "enable_model_routing": True,
        # Training augmentation (US-107)
        "enable_training_augmentation": True,
        # Attention feedback (US-108)
        "enable_attention_feedback": True,
        # Aura Phase 1 (US-200): Enable trader readiness agent for human-side signals
        "enable_trader_readiness_agent": True,
        # Execution quality optimizer (US-109)
        "enable_execution_quality_optimizer": True,  # Phase 29 (US-174): enabled
        # Threshold optimizer (US-110) — enabled Phase 21
        "enable_threshold_optimizer": True,
        # Dynamic risk allocation (US-111) — enabled Phase 22
        "enable_dynamic_risk_allocation": True,
        # Model calibration (US-112) — enabled Phase 22
        "enable_model_calibration": True,
        # Pair-regime-agent matrix (US-113) — enabled Phase 22
        "enable_pair_regime_agent_matrix": True,
        # Signal timing (US-114) — enabled Phase 22
        "enable_signal_timing": True,
        # Memory manager (US-115)
        "enable_memory_manager": True,
        # Module activation (US-116)
        "enable_module_activation": True,
        # Agent lifecycle (US-117)
        "enable_agent_lifecycle": True,
        # Health registry (US-118)
        "enable_health_registry": True,
        # Observation consumer (US-119)
        "enable_observation_consumer": True,
        # Replay validator (US-120)
        "enable_replay_validator": True,
        # Phase 29 (US-174): Enable remaining specialist agents in smart profile
        "enable_session_timing_agent": True,
        "enable_support_resistance_agent": True,
        # Phase 33 (US-214): Enable news risk and session filter agents
        "enable_news_risk_agent": True,
        "enable_session_filter": True,
        # Phase 48: Enable MTF confluence and ensemble conflict gates
        # (were defaulting to False via getattr fallback — never activated)
        "mtf_confluence_enabled": True,
        "ensemble_conflict_enabled": True,
    },
}
VALID_SCAN_PROFILES = tuple(SCAN_PROFILES.keys())


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file (uses DEFAULT_CONFIG_PATH if None)

    Returns:
        Dict containing parsed YAML config
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


@dataclass
class ScannerConfig:
    """Configuration for the FX Scanner.

    Attributes:
        config_path: Path to YAML config file (absolute path recommended)
        pairs: List of FX pairs to scan
        parallel_workers: Number of concurrent pair scans
        lookback_candles: Number of candles to fetch for analysis
        granularity: OANDA timeframe (H1, M15, etc.)
        non_interactive: Skip stdin prompts (for cron/CI)

        # Gate thresholds
        min_confidence: Minimum Ridge confidence score (0-100)
        min_momentum: Minimum XGBoost momentum percentile (0-1)
        max_drawdown_pct: Maximum RF drawdown percentage

        # Position sizing
        account_equity: Account balance (0 = fetch from OANDA)
        risk_per_trade_pct: Risk percentage per trade

        # Watch mode
        watch_interval_seconds: Seconds between rescans in watch mode
        incremental_cache_minutes: Skip re-fetch if scanned within this time
        price_change_threshold: Re-fetch if price moved more than this %
    """
    # Core paths
    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "trained_data" / "models")

    # Pairs to scan
    pairs: List[str] = field(default_factory=lambda: MAJOR_PAIRS.copy())

    # Parallel execution
    parallel_workers: int = 4
    sub_inference_workers: int = 3

    # Data fetching
    lookback_candles: int = 200
    granularity: str = "H1"
    profile: str = SCAN_PROFILE_BALANCED

    # Interactive mode
    non_interactive: bool = field(default_factory=lambda: not sys.stdin.isatty())

    # Gate thresholds (aligned with InferenceConfig)
    min_confidence: float = 50.0  # Ridge ADX score (0-100 scale)
    min_momentum: float = 0.20    # XGBoost percentile (0-1 scale)
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown

    # Meta-labeler threshold (configurable for 0.52-0.53 range)
    # Updated from 0.55 to 0.52 (2024-02) to align with retrained meta-labeler
    # achieving 70.8% accuracy. Lower threshold allows more high-quality signals
    # through while maintaining the all-gates-must-pass safety intact.
    # Valid range: 0.50-0.60 (warning logged if outside this range)
    meta_labeler_threshold: float = 0.52

    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    leverage: int = 50
    min_risk_reward_ratio: float = 1.2  # Minimum R:R to allow execution (TP/SL >= 1.2, per trading rules)

    # Session filter (UTC hours)
    # FX markets are open 24/5 (Sun 22:00 UTC – Fri 22:00 UTC).
    # Enabled by default to restrict scanning to London+NY hours only.
    enable_session_filter: bool = True
    session_start_utc: int = 8   # London open (08:00 UTC)
    session_end_utc: int = 21    # NY close (21:00 UTC = 17:00 ET winter, 16:00 ET summer)
    block_when_market_closed: bool = False  # Live scanner/chat can opt in to weekend closure enforcement

    # Volatility filter
    min_atr_pips: float = 5.0  # Minimum ATR in pips to trade
    min_candles: int = 100  # Minimum candles required

    # TCN Volatility Regime gate (GLOBAL - applies to all pairs)
    # Config-driven minimum regime: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    # Valid values: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    min_volatility_regime: int = 0
    use_tcn_volatility_filter: bool = True
    require_tcn_volatility: bool = False  # If True, block trades when TCN model unavailable

    # Joint-only model loading (scanner uses joint-trained models exclusively)
    use_joint_models_only: bool = True  # Load from trained_data/models/joint/ only

    # Position sizing
    sl_pips: float = 15.0  # Default stop loss
    tp_pips: float = 30.0  # Default take profit
    min_tcn_probability: float = 0.60  # TCN direction gate
    final_score_threshold: float = 0.45  # Ensemble composite score gate (0-1)
    max_uncertainty_std: float = 0.15  # MC-dropout uncertainty ceiling (0-1)

    # Execution settings (from buddy_scanner)
    enable_execution: bool = False  # Enable trade execution
    daily_trade_limit: int = 30  # Max trades per day
    position_sizing_enabled: bool = True
    aggressive_mode: bool = True  # Enable larger positions for compounding
    regime_scaling_enabled: bool = True
    aggressive_scale_high_vol: float = 1.5
    aggressive_scale_extreme_vol: float = 1.75
    aggressive_min_meta_confidence: float = 0.52

    # RL model settings (disabled by default to avoid TF/PyTorch GPU conflicts)
    use_rl_sizer: bool = False  # RL position sizing (PPO model)
    use_rl_gates: bool = False  # RL gate threshold optimizer (SAC model)
    use_rl_exits: bool = False  # RL optimal exit timing (PPO model)

    # ATR-based SL/TP (from buddy_scanner)
    atr_sl_multiplier: float = 1.0  # SL = 1.0x ATR (default; overridden by regime_atr_multipliers)
    atr_tp_multiplier: float = 1.5  # TP = 1.5x ATR (default; overridden by regime_atr_multipliers)

    # Regime-adaptive ATR multipliers (US-050): regime → {sl_mult, tp_mult}
    # LOW=tight stops, EXTREME=wide TP to capture big moves
    regime_atr_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "LOW": {"sl_mult": 0.8, "tp_mult": 1.2},
            "NORMAL": {"sl_mult": 1.0, "tp_mult": 1.5},
            "HIGH": {"sl_mult": 1.1, "tp_mult": 2.0},
            "EXTREME": {"sl_mult": 1.2, "tp_mult": 2.8},
        }
    )
    enable_regime_atr_adaptation: bool = True  # Use regime-adaptive multipliers
    enable_trailing_stop: bool = True  # ATR-based trailing SL for open trades
    trailing_atr_multiplier: float = 1.5  # Trail distance = ATR * this multiplier
    max_data_age_seconds: float = 60.0  # Max age of analysis data before execution skips

    # Execution recovery and fallback params (US-048: extracted from hardcoded values)
    max_order_attempts: int = 2  # Original + N-1 retries on rejection
    rejection_downsize_factor: float = 0.5  # Multiply lots by this on order rejection
    high_slippage_alert_pips: float = 5.0  # Log observation when |slippage| exceeds this
    fallback_tp_pips: float = 25.0  # TP fallback when ATR unavailable
    fallback_atr_pips: float = 15.0  # ATR fallback for trailing stop when actual ATR unavailable
    min_lot_size: float = 0.01  # Minimum order size in lots
    max_lot_size: float = 50.0  # Maximum order size in lots
    trailing_stop_breakeven_pct: float = 0.50  # Progress % to move SL to breakeven
    trailing_stop_lock_pct: float = 0.75  # Progress % to lock 50% of profit

    min_sl_pips: float = 15.0
    max_sl_pips: float = 15.0  # Fixed for tight scalping
    min_tp_pips: float = 20.0
    max_tp_pips: float = 30.0

    # High probability TP bonus (from buddy_scanner)
    high_prob_threshold: float = 0.65  # Confidence threshold for TP bonus
    high_prob_tp_bonus: float = 20.0  # Extra pips at high probability

    # Watch mode incremental caching
    watch_interval_seconds: int = 300  # 5 minutes
    incremental_cache_minutes: int = 5
    incremental_enabled: bool = True  # Enable incremental caching
    price_change_threshold: float = 0.001  # 0.1% price change triggers re-fetch
    max_workers: int = 4  # Default max workers for incremental scan

    # Sub-inference agents (scanner opportunity confirmation pass)
    enable_sub_inference_agents: bool = True
    sub_inference_tradeable_only: bool = True
    sub_inference_min_confidence: float = 0.48
    sub_inference_vote_threshold: float = 0.66
    sub_inference_window_checks: int = 3
    sub_inference_max_candidates: int = 10
    min_agent_consensus_ratio: float = 0.25  # Hard floor: at least 25% of windows must confirm
    enable_agent_trade_promotion: bool = True
    agent_promotion_min_confidence: float = 0.52
    agent_promotion_requires_risk: bool = True
    use_master_pair_models: bool = True

    # Default pairs (for easy access)
    default_pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())
    pip_values: Dict[str, float] = field(default_factory=lambda: PIP_VALUES.copy())

    # Blocked pairs (pairs with model accuracy issues or other constraints)
    blocked_pairs: List[str] = field(default_factory=lambda: ["EUR_USD"])

    # Output
    top_n: int = 5  # Show top N pairs
    show_all: bool = False  # Show all pairs including failed gates

    # =================================================================
    # MULTI-AGENT FRAMEWORK CONFIGURATION
    # Specialized agent evaluators with weighted voting and learning
    # =================================================================

    # --- Agent Toggles (enable/disable each specialized agent) ---
    enable_trend_agent: bool = True
    enable_mean_reversion_agent: bool = True
    enable_volatility_agent: bool = True
    enable_risk_sentinel_agent: bool = True
    enable_news_risk_agent: bool = False  # Default off if no news data
    enable_uncertainty_agent: bool = True
    enable_execution_quality_agent: bool = True
    enable_multi_timeframe_agent: bool = False  # Default off; smart profile enables it
    enable_pair_performance_agent: bool = False  # Default off; smart profile enables it
    enable_momentum_agent: bool = True
    enable_session_timing_agent: bool = False
    enable_support_resistance_agent: bool = False
    enable_trader_readiness_agent: bool = False  # Agent #13: Aura human-side readiness signal

    # --- Graph-Attention Agent Consensus (US-076) ---
    use_heterogeneous_agents: bool = False  # Enable agent specialization categories
    enable_graph_attention: bool = False  # Correlation-aware attention weighting

    # --- Weighted Voting Config ---
    weighted_vote_threshold: float = 0.55  # Minimum weighted score to pass
    agent_weight_learning_rate: float = 0.05  # How fast weights adapt
    agent_weight_decay: float = 0.01  # Decay factor for weight updates
    min_agent_weight: float = 0.1  # Minimum weight floor
    max_agent_weight: float = 2.0  # Maximum weight ceiling

    # --- Regime-Aware Thresholds (regime -> minimum vote threshold) ---
    regime_vote_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "LOW": 0.50,
            "NORMAL": 0.55,
            "HIGH": 0.60,
            "EXTREME": 0.70,
        }
    )

    # --- Regime-Disabled Agents (US-069) ---
    # {regime: [agent_names]} — agents to skip in each volatility regime
    # Default: empty dict preserves current behavior (no agents disabled)
    regime_disabled_agents: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "EXTREME": ["mean_reversion", "session_timing"],
            "LOW": ["volatility"],
        }
    )

    # --- Uncertainty Blocking ---
    max_uncertainty_score: float = 0.4  # Block if uncertainty above this
    max_model_disagreement: float = 0.5  # Block if disagreement above this
    soft_uncertainty_blocking: bool = False  # If True, uncertainty reduces confidence instead of hard blocking

    # --- HRP Portfolio Optimization (US-073) ---
    use_hrp: bool = False  # Replace binary correlation filter with HRP weights
    hrp_linkage_method: str = "average"  # 'average', 'single', 'complete', 'ward'

    # --- Confidence Calibration (US-079) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_confidence_calibration: bool = True  # Platt scaling for raw confidence scores
    calibration_min_trades: int = 30  # Minimum trades before calibration activates

    # --- Dynamic SL/TP Optimization (US-080) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_dynamic_sl_tp: bool = True  # Regime-conditioned SL/TP adaptation
    sl_tp_aggressiveness: float = 1.0  # Multiplier for rule intensity (0.5-2.0)

    # --- Multi-Horizon Signal Fusion (US-081) ---
    enable_multi_horizon_fusion: bool = False  # Bayesian timeframe agreement grading

    # --- Trade Outcome Prediction (US-082) ---
    enable_trade_outcome_prediction: bool = False  # Predict open trade outcomes

    # --- Concept Drift Detection (US-083) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_concept_drift_detection: bool = True  # Rolling z-score drift monitoring

    # --- Kelly Portfolio Sizing (US-084) ---
    enable_kelly_sizing: bool = False  # Portfolio-level Kelly criterion with VaR

    # --- Ensemble Disagreement Signal (US-085) ---
    enable_ensemble_disagreement: bool = False  # Meta-signal from ensemble member disagreement

    # --- Position Timeout with Time-Decay (US-086) ---
    enable_position_timeout: bool = False  # Regime-dependent time-decay for open trades

    # --- Cross-Pair Lead-Lag Detection (US-087) ---
    enable_lead_lag_detection: bool = False  # Lagged correlation entry timing

    # --- Attention-Based Feature Weighting (US-088) ---
    enable_feature_attention: bool = False  # Regime-conditioned softmax attention over features

    # --- Adversarial Robustness Training (US-089) ---
    enable_adversarial_training: bool = False  # Generate adversarial scenarios for ensemble hardening

    # --- Regime-Aware RL Reward Shaping (US-090) ---
    enable_regime_reward: bool = False  # Regime-conditioned reward for RL weight updates

    # --- Adaptive LR Scheduling (US-091) ---
    enable_adaptive_lr: bool = False  # Cosine-annealing + warmup for RL learning rate

    # --- Market Impact Position Sizing (US-092) ---
    enable_market_impact: bool = False  # Almgren-Chriss slippage model for position sizing

    # --- Bandit-Based Model Selection (US-093) ---
    enable_model_bandit: bool = False  # EXP3 bandit for dynamic model routing

    # --- SHAP-Based Trade Explainability (US-094) ---
    enable_trade_explainability: bool = False  # Feature attribution and consistency filtering

    # --- Causal Signal Filtering (US-095) ---
    enable_causal_filtering: bool = False  # Granger causality-based signal consistency

    # --- Synthetic Crisis Event Generator (US-096) ---
    enable_synthetic_crisis: bool = False  # Perturbation-based crisis scenario generation

    # --- Execution Quality Tracking (US-097) ---
    enable_execution_quality_tracking: bool = False  # TCA feedback loop to agent confidence

    # --- Entropy-Based Position Sizing (US-098) ---
    enable_entropy_sizing: bool = False  # Shannon entropy of agent votes → position multiplier

    # --- Regime Event Broadcasting (US-099) ---
    enable_regime_broadcast: bool = False  # Event-driven regime transition propagation

    # --- Agent Accuracy Matrix (US-100) ---
    enable_agent_accuracy_matrix: bool = True  # Phase 25 (US-155): Per-agent accuracy tracking by regime/pair

    # --- Pair Transfer Learning (US-101) ---
    enable_pair_transfer: bool = False  # Share learned weights between correlated FX pairs

    # --- Temporal Attention (US-102) ---
    enable_temporal_attention: bool = False  # Learned attention over multi-timeframe signals

    # --- Live Position Management (US-103) ---
    enable_live_position_management: bool = False  # Prediction-driven open position management

    # --- Affinity Portfolio Sizing (US-104) ---
    enable_affinity_portfolio: bool = False  # Pair affinity → continuous position sizing

    # --- Execution Routing (US-105) ---
    enable_execution_routing: bool = False  # Smart order slicing for large positions

    # --- Model Routing (US-106) ---
    enable_model_routing: bool = False  # Bandit-based pair-specific model selection

    # --- Training Augmentation (US-107) ---
    enable_training_augmentation: bool = False  # Synthetic data augmentation for retrain

    # --- Attention Feedback (US-108) ---
    enable_attention_feedback: bool = False  # Temporal attention agent consensus feedback

    # --- Execution Quality Optimizer (US-109) ---
    enable_execution_quality_optimizer: bool = False  # Cost profiling by model+regime

    # --- Threshold Optimizer (US-110) ---
    enable_threshold_optimizer: bool = True  # Phase 25 (US-153): Adaptive gate threshold tuning

    # --- Dynamic Risk Allocation (US-111) ---
    enable_dynamic_risk_allocation: bool = False  # P/L distribution risk sizing

    # --- Model Calibration (US-112) ---
    enable_model_calibration: bool = False  # Per-model confidence calibration

    # --- Pair-Regime-Agent Matrix (US-113) ---
    enable_pair_regime_agent_matrix: bool = True  # Phase 27 (US-167): 3-way interaction tracking

    # --- Signal Timing (US-114) ---
    enable_signal_timing: bool = False  # Scan frequency optimization

    # --- Memory Manager (US-115) ---
    enable_memory_manager: bool = False  # LRU cache and log rotation

    # --- Module Activation (US-116) ---
    enable_module_activation: bool = False  # Module dispatch scheduling

    # --- Agent Lifecycle (US-117) ---
    enable_agent_lifecycle: bool = True  # Phase 25 (US-151): Agent fitness and quarantine

    # --- Health Registry (US-118) ---
    enable_health_registry: bool = True  # Phase 27 (US-164): Orchestrator health monitoring

    # --- Observation Consumer (US-119) ---
    enable_observation_consumer: bool = True  # Phase 27 (US-163): Pattern detection from observations

    # --- Replay Validator (US-120) ---
    enable_replay_validator: bool = False  # Historical replay and drift detection

    # --- Microstructure Regime Detection (US-074) ---
    enable_microstructure_regime: bool = False  # Augment regime detection with spread signals
    microstructure_confidence_boost: bool = True  # Apply confidence lift from micro signals

    # --- Phase 21: Execution Unblock ---
    enable_regime_consensus: bool = True  # Regime-specific agent consensus thresholds
    enable_extreme_regime_policy: bool = True  # Reduce size instead of blocking in EXTREME
    enable_config_adjuster: bool = True  # Central config adjustment manager
    enable_trade_block_logging: bool = True  # Log every gate failure to observations
    # Phase 29 (US-179): Configurable confidence blend weights
    # Blended confidence = base_conf * W1 + agent_score * W2 + consensus * W3
    confidence_blend_base_weight: float = 0.55  # Weight for base model confidence
    confidence_blend_agent_weight: float = 0.30  # Weight for agent consensus score
    confidence_blend_consensus_weight: float = 0.15  # Weight for consensus ratio
    confidence_boost_cap: float = 0.08  # Max confidence boost when agents pass
    confidence_boost_consensus_factor: float = 0.06  # Consensus multiplier for boost

    # Phase 23 (US-139): Soft consensus gate — penalty instead of hard block
    enable_soft_consensus_gate: bool = True  # Penalty instead of blocking
    soft_consensus_penalty: float = 0.85  # Confidence multiplier when consensus low
    soft_consensus_absolute_min: float = 0.10  # Below this ratio, still hard-block
    # Phase 23 (US-140): Default regime when UNKNOWN persists
    volatility_regime_default: str = "NORMAL"  # Fallback when regime stays UNKNOWN
    # Phase 23 (US-141): Sub-inference opposite-direction penalty
    sub_inference_opposite_penalty: float = 0.50  # Was hardcoded 0.20 (too harsh)
    # Phase 23 (US-142): Execution quality fast-track
    enable_execution_fasttrack: bool = True  # Allow high-quality blocked trades through
    fasttrack_min_quality: float = 0.70  # Minimum execution_quality_score
    fasttrack_min_confidence: float = 0.50  # Minimum confidence for fast-track
    fasttrack_risk_multiplier: float = 0.75  # Reduced risk for fast-track trades
    regime_consensus_map: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 0.50, "NORMAL": 0.33, "HIGH": 0.25, "EXTREME": 0.20,
    })
    extreme_regime_size_multiplier: float = 0.50  # Position size reduction in EXTREME
    extreme_regime_confidence_offset: float = -5.0  # Lower confidence threshold in EXTREME

    # Phase 24 (US-147): Live position management
    enable_position_management: bool = True  # Evaluate open positions via win_prob predictor
    position_management_interval: int = 3  # Run position management every N scan cycles

    # --- Dynamic Hedging (US-077) ---
    enable_dynamic_hedging: bool = False  # Auto-open inverse positions during stress
    hedge_min_correlation: float = 0.65  # Min |correlation| for hedge candidates

    # --- Observational Learning (US-078) ---
    enable_observational_learning: bool = False  # Synthetic agent pattern extraction
    observational_synthetic_weight: float = 0.20  # Blend ratio: synthetic vs live

    # --- Smart Execution (US-075) ---
    enable_smart_execution: bool = False  # Use VWAP/TWAP slicing for larger orders
    execution_strategy: str = "TWAP"  # "VWAP" or "TWAP"
    execution_window_minutes: float = 5.0  # Window for slicing execution

    # --- Circuit-Breaker Config ---
    enable_circuit_breakers: bool = True
    max_concurrent_trades: int = 10  # Phase 29 (US-178): Max open trades before blocking new ones
    max_correlated_exposure: int = 2  # Max trades in correlated pairs
    loss_streak_pause_count: int = 3  # Pause after N consecutive losses
    loss_streak_pause_minutes: int = 60  # Pause duration in minutes
    news_blackout_minutes: int = 30  # Block trades around news events
    session_constraints: List[str] = field(default_factory=list)  # Blocked sessions

    # --- Execution Quality Thresholds ---
    max_spread_pips: float = 3.0  # Max acceptable spread in pips
    max_slippage_pips: float = 2.0  # Max acceptable slippage in pips
    min_liquidity_score: float = 0.3  # Minimum liquidity score (0-1)

    # --- Learning Loop Config (post-trade agent weight updates) ---
    enable_agent_learning: bool = True
    enable_llm_trade_analysis: bool = False  # LLM deep analysis for losing trades (US-009)
    weight_boost_on_win: float = 0.1  # Weight increase on winning trade
    weight_penalty_on_loss: float = 0.15  # Weight decrease on losing trade

    # --- Model A/B Testing (True A/B: load both incumbent + candidate at scan time) ---
    enable_true_ab_testing: bool = False  # Disabled by default; enabled in smart profile

    # --- Phase 46-47: MTF Confluence & Ensemble Conflict ---
    mtf_confluence_enabled: bool = True  # Elder's Triple Screen via MTF module
    ensemble_conflict_enabled: bool = True  # Ensemble disagreement resolver

    # Loaded YAML config (lazy loaded)
    _yaml_config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        """Convert string paths to Path objects and validate thresholds."""
        self.profile = str(self.profile).strip().lower()
        if self.profile not in SCAN_PROFILES:
            logger.warning(
                f"Unknown scan profile '{self.profile}', defaulting to '{SCAN_PROFILE_BALANCED}'"
            )
            self.profile = SCAN_PROFILE_BALANCED

        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

        # Resolve relative paths to absolute
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        if not self.model_dir.is_absolute():
            self.model_dir = PROJECT_ROOT / self.model_dir

        # Validate meta_labeler_threshold is within reasonable bounds
        if not 0.50 <= self.meta_labeler_threshold <= 0.60:
            logger.warning(
                f"meta_labeler_threshold={self.meta_labeler_threshold} is outside "
                f"recommended range [0.50, 0.60]. This may impact signal quality."
            )
        self.sub_inference_vote_threshold = min(
            1.0,
            max(0.34, float(self.sub_inference_vote_threshold)),
        )
        self.min_tcn_probability = min(0.99, max(0.50, float(self.min_tcn_probability)))
        self.final_score_threshold = min(0.99, max(0.20, float(self.final_score_threshold)))
        self.max_uncertainty_std = min(0.50, max(0.01, float(self.max_uncertainty_std)))
        self.weighted_vote_threshold = min(0.95, max(0.40, float(self.weighted_vote_threshold)))
        self.sub_inference_min_confidence = min(0.95, max(0.30, float(self.sub_inference_min_confidence)))
        self.sub_inference_window_checks = max(1, int(self.sub_inference_window_checks))
        self.sub_inference_workers = max(1, int(self.sub_inference_workers))
        self.sub_inference_max_candidates = max(1, int(self.sub_inference_max_candidates))
        self.max_uncertainty_score = min(0.95, max(0.10, float(self.max_uncertainty_score)))
        self.max_model_disagreement = min(0.95, max(0.10, float(self.max_model_disagreement)))
        self.agent_promotion_min_confidence = min(
            0.99,
            max(0.30, float(self.agent_promotion_min_confidence)),
        )

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load and cache YAML configuration.

        Returns:
            Dict containing parsed YAML config

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if self._yaml_config is not None:
            return self._yaml_config

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Expected at: {DEFAULT_CONFIG_PATH}"
            )

        try:
            import yaml
            with open(self.config_path) as f:
                self._yaml_config = yaml.safe_load(f)

            # Update model_dir from config if specified
            paths = self._yaml_config.get("paths", {})
            if "model_dir" in paths:
                model_dir = Path(paths["model_dir"])
                if not model_dir.is_absolute():
                    model_dir = PROJECT_ROOT / model_dir
                self.model_dir = model_dir

            return self._yaml_config

        except Exception as e:
            raise ValueError(f"Failed to load config {self.config_path}: {e}")

    def get_pip_value(self, pair: str) -> float:
        """Get pip value for a pair."""
        return PIP_VALUES.get(pair, 0.0001)

    def apply_profile(self, profile: Optional[str] = None) -> None:
        """Apply scanner threshold overrides for a named profile.

        Args:
            profile: Profile name (conservative|balanced|aggressive).
                If None, reapplies current ``self.profile``.

        Raises:
            ValueError: If profile name is unknown.
        """
        resolved = str(profile or self.profile).strip().lower()
        if resolved not in SCAN_PROFILES:
            valid = ", ".join(VALID_SCAN_PROFILES)
            raise ValueError(f"Unknown scan profile '{resolved}'. Valid profiles: {valid}")

        self.profile = resolved
        overrides = SCAN_PROFILES[resolved]
        # Phase 32 (US-192): Validate keys against known fields before setattr
        _known_fields = {f.name for f in self.__dataclass_fields__.values()} if hasattr(self, "__dataclass_fields__") else set()
        for field_name, value in overrides.items():
            if _known_fields and field_name not in _known_fields:
                logger.warning(f"apply_profile: unknown field '{field_name}' in profile '{resolved}' — skipping")
                continue
            setattr(self, field_name, value)
        # Keep regime in valid classifier range.
        self.min_volatility_regime = max(0, min(3, int(self.min_volatility_regime)))

    def get_trading_session_status(self, now: Optional[Any] = None) -> Dict[str, Any]:
        """Return current market-open/session status for the scanner.

        Weekend market closure is treated separately from the intraday
        liquidity-hour session filter so `force` can bypass session hours
        without pretending the FX market is open on Saturday.
        """
        from datetime import datetime, timezone

        current_time = now or datetime.now(timezone.utc)
        hour = int(current_time.hour)
        weekday = int(current_time.weekday())  # 0=Monday, 6=Sunday
        day_name = current_time.strftime("%A")

        market_open = True
        reason_code = None
        message = None

        if self.block_when_market_closed and weekday == 5:
            market_open = False
            reason_code = "market_closed"
            message = (
                f"FX market closed (weekend: {day_name} {hour:02d}:00 UTC; "
                "reopens Sunday 22:00 UTC)"
            )
        elif self.block_when_market_closed and weekday == 6 and hour < 22:
            market_open = False
            reason_code = "market_closed"
            message = (
                f"FX market closed (weekend: {day_name} {hour:02d}:00 UTC; "
                "reopens Sunday 22:00 UTC)"
            )
        elif self.block_when_market_closed and weekday == 4 and hour >= 22:
            market_open = False
            reason_code = "market_closed"
            message = (
                f"FX market closed (weekend: {day_name} {hour:02d}:00 UTC; "
                "reopens Sunday 22:00 UTC)"
            )

        within_hours = self.session_start_utc <= hour < self.session_end_utc
        session_allowed = market_open and (not self.enable_session_filter or within_hours)

        if reason_code is None and self.enable_session_filter and not within_hours:
            reason_code = "outside_session"
            message = (
                f"Outside trading session ({hour}:00 UTC, "
                f"active: {self.session_start_utc}-{self.session_end_utc} UTC)"
            )

        if market_open and self.enable_session_filter:
            if within_hours:
                logger.debug(
                    f"Within trading session: {hour}:00 UTC "
                    f"(peak hours: {self.session_start_utc}:00-{self.session_end_utc}:00 UTC)"
                )
            else:
                logger.debug(
                    f"Outside peak hours: {hour}:00 UTC "
                    f"(peak hours: {self.session_start_utc}:00-{self.session_end_utc}:00 UTC)"
                )
        elif market_open:
            logger.debug("Session filter disabled, but FX market is open")
        else:
            logger.debug(message or "FX market closed")

        return {
            "now_utc": current_time,
            "hour_utc": hour,
            "weekday": weekday,
            "day_name": day_name,
            "market_open": market_open,
            "within_hours": within_hours,
            "session_filter_enabled": bool(self.enable_session_filter),
            "session_allowed": session_allowed,
            "reason_code": reason_code,
            "message": message,
        }

    def is_market_open(self, now: Optional[Any] = None) -> bool:
        """Return whether the FX market is open right now."""
        return bool(self.get_trading_session_status(now).get("market_open"))

    def is_within_session(self, now: Optional[Any] = None) -> bool:
        """Return whether trading is currently allowed by market-open + session rules."""
        return bool(self.get_trading_session_status(now).get("session_allowed"))

    @classmethod
    def from_cli_args(
        cls,
        config_path: Optional[str] = None,
        pairs: Optional[List[str]] = None,
        top_n: int = 5,
        show_all: bool = False,
        granularity: str = "H1",
        profile: str = SCAN_PROFILE_BALANCED,
        watch: bool = False,
        non_interactive: bool = False,
        force: bool = False,  # Disable session filter
    ) -> "ScannerConfig":
        """Create config from CLI arguments.

        Args:
            config_path: Override config path
            pairs: Override pairs list
            top_n: Number of top pairs to show
            show_all: Show all pairs including failed
            granularity: Timeframe
            profile: Scan profile (conservative|balanced|aggressive)
            watch: Enable watch mode
            non_interactive: Skip interactive prompts
            force: Force scan even outside session hours

        Returns:
            ScannerConfig instance
        """
        config = cls()

        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            config.config_path = path

        if pairs:
            config.pairs = pairs

        config.top_n = top_n
        config.show_all = show_all
        config.granularity = granularity
        config.apply_profile(profile)

        if non_interactive:
            config.non_interactive = True

        # Disable session filter if --force (allow trading outside peak hours)
        if force:
            config.enable_session_filter = False

        return config
