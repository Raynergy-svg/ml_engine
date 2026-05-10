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

from src.brokers.registry import get_registry

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

# Futures contracts — front-month symbols for IBKR
# Symbol format: CME/NYMEX/COMEX standard roots
DEFAULT_FUTURES = [
    "ES",   # S&P 500 E-mini (CME) — equity index, strong trend, high liquidity
    "NQ",   # Nasdaq 100 E-mini (CME) — tech-heavy, momentum-driven
    "CL",   # Crude Oil (NYMEX) — commodity, seasonal + geopolitical
    "GC",   # Gold (COMEX) — safe haven, inversely correlated to equities
    "ZB",   # 30-Year Treasury Bond (CBOT) — rates, macro regime signal
    "6E",   # EUR/USD futures (CME) — same as FX but with real volume
]

# NOTE: PIP_VALUES has been moved to InstrumentRegistry (src.brokers.registry)
# Use get_registry().get(symbol).pip_value to access pip values

SCAN_PROFILE_BALANCED = "balanced"
SCAN_PROFILES: Dict[str, Dict[str, Any]] = {
    # Uses configuration defaults (YAML + dataclass); no extra overrides.
    SCAN_PROFILE_BALANCED: {
        "blocked_pairs": [],
        "sub_inference_min_confidence": 0.30,
        "min_agent_consensus_ratio": 0.50,
        "enable_devil_advocate": True,
        "devil_advocate_block_threshold": 0.60,
        "devil_advocate_warn_threshold": 0.40,
        "soft_uncertainty_blocking": True,
        # Phase 81+: Enable ALL trading modules — Tier 6 full stack
        # Risk & sizing
        "enable_dynamic_risk_allocation": True,
        "enable_kelly_sizing": True,
        "enable_entropy_sizing": True,
        "enable_market_impact": True,
        "enable_affinity_portfolio": True,
        "enable_dynamic_hedging": True,
        "enable_dynamic_sl_tp": True,
        "enable_live_position_management": True,
        "enable_position_timeout": True,
        # Execution
        "enable_smart_execution": True,
        "enable_execution_routing": True,
        "enable_execution_quality_optimizer": True,
        "enable_execution_quality_tracking": True,
        "execution_strategy": "TWAP",
        # Intelligence & learning
        "enable_model_calibration": True,
        "enable_model_bandit": True,
        "enable_model_routing": True,
        "enable_confidence_calibration": True,
        "enable_concept_drift_detection": True,
        "enable_ensemble_disagreement": True,
        "enable_trade_outcome_prediction": True,
        "enable_trade_explainability": True,
        "enable_lead_lag_detection": True,
        "enable_observational_learning": True,
        # Agents & signals
        "enable_multi_timeframe_agent": True,
        "enable_pair_performance_agent": True,
        "enable_session_timing_agent": True,
        "enable_support_resistance_agent": True,
        "enable_news_risk_agent": True,
        "enable_momentum_agent": True,
        "enable_trader_readiness_agent": True,
        "enable_order_flow_agent": True,
        "enable_devil_advocate_agent": True,
        # Features & attention
        "enable_feature_attention": True,
        "enable_temporal_attention": True,
        "enable_causal_filtering": True,
        "enable_microstructure_regime": True,
        "enable_multi_horizon_fusion": True,
        # Infrastructure
        "enable_memory_manager": True,
        "enable_health_registry": True,
        "enable_module_activation": True,
        "enable_agent_lifecycle": True,
        "enable_observation_consumer": True,
        "enable_replay_validator": True,
        "enable_agent_accuracy_matrix": True,
        "enable_pair_regime_agent_matrix": True,
        "enable_signal_timing": True,
        "enable_threshold_optimizer": True,
        "enable_attention_feedback": True,
        "enable_pair_transfer": True,
        "enable_regime_broadcast": True,
        "enable_regime_reward": True,
        "enable_adversarial_training": True,
        "enable_adaptive_lr": True,
        "enable_synthetic_crisis": True,
        "enable_training_augmentation": True,
        "enable_agent_trade_promotion": True,
        # HRP & graph
        "use_hrp": True,
        "use_heterogeneous_agents": True,
        "enable_graph_attention": True,
        # RL
        "use_rl_sizer": True,
        "use_rl_gates": True,
        "use_rl_exits": True,
        # Meta-cybernetic change pipeline (default OFF, see ScannerConfig)
        "enable_meta_manager": False,
        "meta_manager_max_concurrent": 1,
        "meta_manager_constitution_path": ".claude/rules/constitution.json",
        "meta_manager_use_llm": False,
        "staged_deploy_shadow_cycles": 20,
        "staged_deploy_canary_trades": 10,
    },
    # Fewer trades, higher signal quality requirements.
    "conservative": {
        "blocked_pairs": [],
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
        "sub_inference_vote_threshold": 0.55,  # Phase 81: was 0.66, slightly more lenient for conservative
        "agent_promotion_min_confidence": 0.56,
        "max_uncertainty_score": 0.35,
        "max_model_disagreement": 0.45,
        "min_agent_consensus_ratio": 0.42,  # ~42% of voting agents minimum — stricter
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
        "enable_devil_advocate": True,
        "enable_devil_advocate_agent": True,
        "enable_order_flow_agent": True,
        "enable_trader_readiness_agent": True,
        "devil_advocate_block_threshold": 0.60,
        "devil_advocate_warn_threshold": 0.40,
        # Meta-cybernetic change pipeline (default OFF, see ScannerConfig)
        "enable_meta_manager": False,
        "meta_manager_max_concurrent": 1,
        "meta_manager_constitution_path": ".claude/rules/constitution.json",
        "meta_manager_use_llm": False,
        "staged_deploy_shadow_cycles": 20,
        "staged_deploy_canary_trades": 10,
    },
    # More trade frequency with looser gates (still risk-bounded).
    "aggressive": {
        "blocked_pairs": [],
        "enable_execution": True,  # Phase 30 (US-185)
        "min_confidence": 40.0,
        "min_momentum": 0.06,
        "min_tcn_probability": 0.58,
        "max_drawdown_pct": 0.035,
        "final_score_threshold": 0.42,
        "max_uncertainty_std": 0.18,
        "min_atr_pips": 3.0,
        "min_volatility_regime": 0,
        "weighted_vote_threshold": 0.45,
        "sub_inference_min_confidence": 0.45,
        "sub_inference_vote_threshold": 0.34,  # Phase 98: 1/3 vote threshold
        "agent_promotion_min_confidence": 0.50,
        "max_uncertainty_score": 0.48,
        "max_model_disagreement": 0.65,
        "min_agent_consensus_ratio": 0.33,  # ~33% of voting agents minimum — was 0.25
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
        "enable_devil_advocate": True,
        "enable_devil_advocate_agent": True,
        "enable_order_flow_agent": True,
        "devil_advocate_block_threshold": 0.60,
        "devil_advocate_warn_threshold": 0.40,
        # Meta-cybernetic change pipeline (default OFF, see ScannerConfig)
        "enable_meta_manager": False,
        "meta_manager_max_concurrent": 1,
        "meta_manager_constitution_path": ".claude/rules/constitution.json",
        "meta_manager_use_llm": False,
        "staged_deploy_shadow_cycles": 20,
        "staged_deploy_canary_trades": 10,
    },
    # Smart: Agent-driven with RL integration.  Sub-inference agents run on
    # ALL candidates (not just gate-passed) so the agent team can promote
    # borderline setups that have strong multi-timeframe confluence.  Slightly
    # relaxed uncertainty ceiling to let agents make the final call while
    # maintaining risk gates.
    "smart": {
        "blocked_pairs": [],
        "enable_execution": True,
        # ── Entry Quality Gates (high selectivity > high volume) ─────
        # Mythos audit 2026-04-29 — relaxed from 62.0 → 35.0. Context:
        # the 04-15 tightening to 62.0 came from the 10-loss streak when
        # 28-day-stale models still produced confidence 0.55–0.70 and
        # those high-confidence signals were just noise. Today's models
        # are different-stale: per-pair retrain cadence reduced absolute
        # confidence output to 0.10–0.25 range. With threshold 62.0,
        # 100% of scans gate-out (12 days no closed trades, 200+
        # cumulative tightenings). Lowering to 35.0:
        #   * still well above the 0.10–0.25 noise floor
        #   * lets exceptional scans (>0.35) through for execution
        #   * other anti-stale-model gates remain intact (momentum 0.12,
        #     uncertainty 0.38 hard-block, disagreement 0.28, trend
        #     hard-veto, MR composite veto) — those are what actually
        #     prevented the bad trades, not the headline confidence
        #     threshold.
        # Revert toward 50–62 once a healthy retrain cycle restores
        # confidence output to the 0.4–0.7 range the original 62.0
        # was calibrated against.
        "min_confidence": 35.0,
        "min_momentum": 0.12,  # Old: 0.05. Require real directional momentum.
        "min_tcn_probability": 0.62,  # Old: 0.58. Ensemble must agree.
        "max_drawdown_pct": 0.025,  # Old: 0.030. Tighter drawdown leash.
        "final_score_threshold": 0.52,  # Old: 0.44. Raise the floor.
        "max_uncertainty_std": 0.13,  # Old: 0.16. Models must be confident.
        "min_atr_pips": 5.0,  # Old: 4.0. Skip dead-vol pairs.
        "min_volatility_regime": 0,
        # ── Agent Consensus (tighter WVS, hard uncertainty blocking) ──
        # Old WVS: 0.45. Trade 1220 passed at 0.76 despite trend=False.
        # Source: trading.md rule 2026-04-15 + config_adjustments.json CRITICAL.
        "weighted_vote_threshold": 0.72,
        "sub_inference_tradeable_only": False,
        "sub_inference_min_confidence": 0.45,  # Old: 0.30.
        "sub_inference_vote_threshold": 0.42,  # Old: 0.34.
        "sub_inference_max_candidates": 10,  # Old: 15. Focus on fewer, better.
        "agent_promotion_min_confidence": 0.58,  # Old: 0.50.
        # Old: 0.52 soft. Soft penalty docked only 0.074 from confidence —
        # trades at 0.60 uncertainty sailed through (source: trade 1199).
        "max_uncertainty_score": 0.38,
        # Old: 0.60. At 0.50 disagreement models are guessing. Source:
        # config_adjustments.json CRITICAL — 2/3 losses had disagreement=0.5.
        "max_model_disagreement": 0.28,
        # ── RL Adaptive Systems ───────────────────────────────────────
        "use_rl_sizer": True,
        "use_rl_gates": True,
        "use_rl_exits": True,
        "enable_agent_trade_promotion": True,
        # HARD uncertainty blocking. Source: trading.md rule 2026-04-15 —
        # soft penalty proved insufficient during 10-loss streak.
        "soft_uncertainty_blocking": False,
        # ── ATR-Based Dynamic SL/TP (asymmetric R:R) ─────────────────
        # Target: 2.3:1 average R:R. Old: 1.67:1 (1.2 SL / 2.0 TP).
        # Source: trading.md rule — LOW regime sl_mult >= 1.2.
        "atr_sl_multiplier": 1.3,
        "atr_tp_multiplier": 2.8,  # Old: 2.0. Wider TP for asymmetry.
        "min_sl_pips": 12.0,  # Old: 10.0. Survive noise.
        "max_sl_pips": 30.0,  # Old: 35.0. Don't bleed on wrong calls.
        "min_tp_pips": 22.0,  # Old: 15.0. No scalping — hold winners.
        "max_tp_pips": 80.0,  # Old: 60.0. Let runners run.
        "high_prob_threshold": 0.72,  # Old: 0.65. Earn the TP bonus.
        "high_prob_tp_bonus": 20.0,  # Old: 15.0. Reward high conviction.
        # ── Multi-Timeframe & Performance ─────────────────────────────
        "enable_multi_timeframe_agent": True,
        "enable_pair_performance_agent": True,
        # Old: 1.2. Barely above cost. Hedge funds target 2:1+ asymmetry.
        # At 1.8:1 min, even 40% win rate is profitable (expectancy > 0).
        "min_risk_reward_ratio": 1.8,
        "enable_llm_trade_analysis": True,
        "enable_trend_agent_hard_veto": True,
        "enable_mean_reversion_veto": True,  # H2 (Phase 93)
        "mean_reversion_veto_disagree_floor": 0.25,
        "staleness_uncertainty_threshold": 0.35,
        "staleness_age_threshold_days": 7.0,
        "veto_window_ms_default": 0,  # US-511: operator veto window (0=disabled by default in smart)
        "veto_window_ms_per_pair": {},
        # Extended agents 13/14/15 — always on for smart profile; operators can disable via YAML override
        "enable_trader_readiness_agent": True,
        "enable_order_flow_agent": True,
        "enable_devil_advocate_agent": True,
        # Meta-cybernetic change pipeline — enabled in `smart` (live profile)
        # only. Other profiles keep the default-OFF posture from
        # ScannerConfig. use_llm=False keeps the runtime Claude-free; the
        # constitution + scorecard + revert_by_id are pure-Python gates and
        # govern every change without LLM enrichment.
        "enable_meta_manager": True,
        "meta_manager_max_concurrent": 1,
        "meta_manager_constitution_path": ".claude/rules/constitution.json",
        "meta_manager_use_llm": False,
        "staged_deploy_shadow_cycles": 20,
        "staged_deploy_canary_trades": 10,
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
    pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())  # Phase 81: was MAJOR_PAIRS (7), now all 15 pairs

    # Parallel execution
    parallel_workers: int = 4
    sub_inference_workers: int = 3

    # Data fetching
    lookback_candles: int = 200
    # 2026-05-07: switched H1 → M15 default. M15 24-bar-forward direction
    # holdout = 69.2% (validated 22-month window); H1 24-bar = ~49% across
    # all architectures (custom Transformer + Chronos foundation models).
    # The operative trading TF is M15; H1 was the noise floor.
    granularity: str = "M15"
    profile: str = SCAN_PROFILE_BALANCED

    # Interactive mode
    non_interactive: bool = field(default_factory=lambda: not sys.stdin.isatty())

    # Gate thresholds (aligned with InferenceConfig)
    min_confidence: float = 42.0  # Ridge ADX score (0-100 scale) — Phase 81: was 50.0, models output 0.20-0.49
    min_momentum: float = 0.06    # XGBoost percentile (0-1 scale) — Phase 81: was 0.20, avg momentum=0.086
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown

    # Mythos audit 2026-04-30 — auto-halt-on-loss-streak. Operator-
    # observed gap: scanner kept scanning + losing for 14 trades
    # straight (04-15 streak) with no automatic stop. AlertManager
    # raises consecutive_losses alerts but they were observability-
    # only (logged, never acted on). When consecutive losses ≥ this
    # threshold, the engine sets StateEngine.halted=True and emits
    # an incident to the meta-pipeline. Operator must manually
    # un-halt (TUI 'k' or `state.json halted=false`). 0 disables.
    auto_halt_consecutive_loss_threshold: int = 5

    # Meta-labeler threshold (configurable for 0.52-0.53 range)
    # Updated from 0.55 to 0.52 (2024-02) to align with retrained meta-labeler
    # achieving 70.8% accuracy. Lower threshold allows more high-quality signals
    # through while maintaining the all-gates-must-pass safety intact.
    # Valid range: 0.50-0.60 (warning logged if outside this range)
    meta_labeler_threshold: float = 0.52

    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.05  # 5% risk per trade — practice account, $100K
    max_open_risk_pct: float = 0.30  # 30% max portfolio risk — practice account, allows multiple trades
    leverage: int = 50
    min_risk_reward_ratio: float = 1.5  # Phase 81: was 2.0, LOW regime can't produce 2:1 consistently. Adaptive R:R targets 2.5:1+

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
    compound_gate_floor: float = 0.10  # Minimum compound gate pass rate (execution.py)
    max_uncertainty_std: float = 0.15  # MC-dropout uncertainty ceiling (0-1)

    # ═══════════════════════════════════════════════════════════════════════════════
    # BROKER CONFIGURATION (US-004)
    # Select which broker to use for execution and data fetching
    # ═══════════════════════════════════════════════════════════════════════════════
    broker_type: str = "oanda"  # "oanda" or "ibkr" (default: OANDA)
    asset_class: str = "fx"  # "fx" | "futures" | "hybrid" — determines instrument set + sizing logic

    # OANDA-specific settings
    oanda_account_id: str = ""  # Account ID (e.g., "001-001-123456-001")
    oanda_api_key: str = ""  # API key for authentication
    oanda_environment: str = "practice"  # "practice" or "live"

    # Interactive Brokers (IBKR) settings
    ibkr_host: str = "127.0.0.1"  # TWS/IBGateway host (default: localhost)
    ibkr_port: int = 7497  # TWS port (7497) or IBGateway (4002 live)
    ibkr_client_id: int = 1  # Unique client ID for connection

    # Execution settings (from buddy_scanner)
    enable_execution: bool = True  # Enable trade execution
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
    atr_tp_multiplier: float = 2.5  # TP = 2.5x ATR — Phase 81: was 1.5, user wants high R:R (wins >> losses)

    # Regime-adaptive ATR multipliers (US-050): regime → {sl_mult, tp_mult}
    # LOW=tight stops, EXTREME=wide TP to capture big moves
    regime_atr_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "LOW": {"sl_mult": 1.2, "tp_mult": 1.2},
            "NORMAL": {"sl_mult": 1.0, "tp_mult": 1.5},
            "HIGH": {"sl_mult": 1.1, "tp_mult": 2.0},
            "EXTREME": {"sl_mult": 1.2, "tp_mult": 2.8},
        }
    )
    enable_regime_atr_adaptation: bool = True  # Use regime-adaptive multipliers
    enable_trailing_stop: bool = True  # ATR-based trailing SL for open trades
    trailing_atr_multiplier: float = 1.5  # Trail distance = ATR * this multiplier
    max_data_age_seconds: float = 600.0  # Max age of analysis data before execution skips
    # 600s = 10 min. Batch scans across 15 pairs + model loading take 2-8 min on first
    # run. Old 60s default silently killed every trade as stale before OANDA ever saw them.

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

    min_sl_pips: float = 8.0   # Phase 81: was 15, allow tighter SL for higher R:R
    max_sl_pips: float = 30.0  # Phase 81: was 15 (locked), allow ATR-based dynamic SL
    min_tp_pips: float = 20.0
    max_tp_pips: float = 60.0  # Phase 81: was 30, widened for higher R:R targets

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
    sub_inference_min_confidence: float = 0.30  # Phase 81: was 0.48, most pairs have conf 0.23-0.38
    sub_inference_vote_threshold: float = 0.34  # Phase 98: was 0.50, 1/3 vote threshold (3 workers)
    sub_inference_window_checks: int = 3
    sub_inference_max_candidates: int = 10
    min_agent_consensus_ratio: float = 0.25  # Hard floor: at least 25% of windows must confirm
    enable_agent_trade_promotion: bool = True
    agent_promotion_min_confidence: float = 0.52
    agent_promotion_requires_risk: bool = True
    use_master_pair_models: bool = True

    # Default instruments (resolved from asset_class at init)
    default_pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())
    default_futures: List[str] = field(default_factory=lambda: DEFAULT_FUTURES.copy())
    pip_values: Dict[str, float] = field(default_factory=lambda: {
        inst.symbol: inst.pip_value
        for inst in get_registry().get_by_asset_class("FX")
    })

    @property
    def active_instruments(self) -> List[str]:
        """Return instruments for the configured asset_class."""
        if self.asset_class == "futures":
            return list(self.default_futures)
        elif self.asset_class == "hybrid":
            return list(self.default_pairs) + list(self.default_futures)
        else:  # "fx" default
            return list(self.default_pairs)

    @property
    def active_broker_type(self) -> str:
        """Return the broker for the configured asset_class."""
        if self.asset_class == "futures":
            return "ibkr"
        elif self.asset_class == "hybrid":
            return self.broker_type  # user chooses primary
        else:
            return "oanda"

    # Blocked pairs (pairs with model accuracy issues or other constraints)
    blocked_pairs: List[str] = field(default_factory=list)

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
    enable_news_risk_agent: bool = True  # Default off if no news data
    enable_uncertainty_agent: bool = True
    enable_execution_quality_agent: bool = True
    enable_multi_timeframe_agent: bool = True  # Default off; smart profile enables it
    enable_pair_performance_agent: bool = True  # Default off; smart profile enables it
    enable_momentum_agent: bool = True
    enable_session_timing_agent: bool = True
    enable_support_resistance_agent: bool = True
    enable_trader_readiness_agent: bool = True  # Agent #13: Aura human-side readiness signal
    enable_devil_advocate: bool = True  # Agent #14: Adversarial bear-case evaluator (legacy flag — kept for backward compatibility)
    enable_devil_advocate_agent: bool = True  # Agent #14: Canonical _agent-suffixed toggle (supersedes enable_devil_advocate)
    enable_order_flow_agent: bool = True  # Agent #15: Order-flow / book-depth agent
    enable_calendar_blackout: bool = True  # Economic calendar blackout gate (US-301)
    news_sentiment_fast_mode: bool = True  # Fast-path sentiment inference
    enable_news_sentiment: bool = True  # News sentiment integration
    devil_advocate_block_threshold: float = 0.60  # Bear score above this blocks the trade
    devil_advocate_warn_threshold: float = 0.40   # Bear score above this soft-vetoes
    enable_trend_agent_hard_veto: bool = True  # Hard-block directional trades when trend agent fails
    # H2 (Phase 93): mean_reversion composite veto — block when MR fails AND models disagree > floor.
    # MR weight is 0.90 (lowest of core agents) so its NO vote gets drowned in WVS arithmetic.
    # This gives MR real authority only when paired with model disagreement evidence.
    enable_mean_reversion_veto: bool = True
    mean_reversion_veto_disagree_floor: float = 0.25  # Fires when model_disagreement > this value

    # --- Graph-Attention Agent Consensus (US-076) ---
    use_heterogeneous_agents: bool = False  # Enable agent specialization categories
    enable_graph_attention: bool = True  # Correlation-aware attention weighting

    # --- Weighted Voting Config ---
    weighted_vote_threshold: float = 0.45  # Phase 81: was 0.55, gives headroom for adaptive mechanisms
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
    max_model_disagreement: float = 0.65  # Phase 81: was 0.5, hard-blocked all signals at 0.50
    disagreement_hard_floor: float = 0.50  # Phase 81: was 0.30, too many false blocks. Soft penalty handles the range 0.50-0.65
    soft_uncertainty_blocking: bool = True  # Phase 76: Default True — graduated confidence penalty instead of hard block for disagreement between floor and max
    staleness_uncertainty_threshold: float = 0.35  # Tighter uncertainty block when models are stale (US-503)
    staleness_age_threshold_days: float = 7.0  # Days beyond which staleness_uncertainty_threshold applies (US-503)

    # --- HRP Portfolio Optimization (US-073) ---
    use_hrp: bool = False  # Replace binary correlation filter with HRP weights
    hrp_linkage_method: str = "average"  # 'average', 'single', 'complete', 'ward'

    # --- Confidence Calibration (US-079) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_confidence_calibration: bool = True  # Platt scaling for raw confidence scores
    calibration_min_trades: int = 30  # Minimum trades before calibration activates

    # --- Phase 44 Confidence Calibration kill switch (2026-05-09) ---
    # When False, bypasses the Phase 44 ConfidenceCalibrator entirely (no override of
    # weighted_vote_score, no calibration_details writes). Default True preserves
    # current behavior. Added to investigate over-aggressive compression observed
    # in production (raw=0.582 → calibrated=0.099, ~5.9× compression).
    enable_phase44_calibration: bool = True

    # --- Dynamic SL/TP Optimization (US-080) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_dynamic_sl_tp: bool = True  # Regime-conditioned SL/TP adaptation
    sl_tp_aggressiveness: float = 1.0  # Multiplier for rule intensity (0.5-2.0)

    # --- Multi-Horizon Signal Fusion (US-081) ---
    enable_multi_horizon_fusion: bool = True  # Bayesian timeframe agreement grading

    # --- Trade Outcome Prediction (US-082) ---
    enable_trade_outcome_prediction: bool = True  # Predict open trade outcomes

    # --- Concept Drift Detection (US-083) ---
    # Phase 29 (US-179): Enabled by default — stable and tested in smart profile
    enable_concept_drift_detection: bool = True  # Rolling z-score drift monitoring

    # --- Kelly Portfolio Sizing (US-084) ---
    enable_kelly_sizing: bool = True  # Portfolio-level Kelly criterion with VaR

    # --- Ensemble Disagreement Signal (US-085) ---
    enable_ensemble_disagreement: bool = True  # Meta-signal from ensemble member disagreement

    # --- Position Timeout with Time-Decay (US-086) ---
    enable_position_timeout: bool = True  # Regime-dependent time-decay for open trades

    # --- Cross-Pair Lead-Lag Detection (US-087) ---
    enable_lead_lag_detection: bool = True  # Lagged correlation entry timing

    # --- Attention-Based Feature Weighting (US-088) ---
    enable_feature_attention: bool = True  # Regime-conditioned softmax attention over features

    # --- Adversarial Robustness Training (US-089) ---
    enable_adversarial_training: bool = True  # Generate adversarial scenarios for ensemble hardening

    # --- Regime-Aware RL Reward Shaping (US-090) ---
    enable_regime_reward: bool = True  # Regime-conditioned reward for RL weight updates

    # --- Adaptive LR Scheduling (US-091) ---
    enable_adaptive_lr: bool = True  # Cosine-annealing + warmup for RL learning rate

    # --- Market Impact Position Sizing (US-092) ---
    enable_market_impact: bool = True  # Almgren-Chriss slippage model for position sizing

    # --- Bandit-Based Model Selection (US-093) ---
    enable_model_bandit: bool = True  # EXP3 bandit for dynamic model routing

    # --- SHAP-Based Trade Explainability (US-094) ---
    enable_trade_explainability: bool = True  # Feature attribution and consistency filtering

    # --- Causal Signal Filtering (US-095) ---
    enable_causal_filtering: bool = True  # Granger causality-based signal consistency

    # --- Synthetic Crisis Event Generator (US-096) ---
    enable_synthetic_crisis: bool = True  # Perturbation-based crisis scenario generation

    # --- Execution Quality Tracking (US-097) ---
    enable_execution_quality_tracking: bool = True  # TCA feedback loop to agent confidence

    # =================================================================
    # TIER 6 DIMENSION B: STRATEGY INVENTION ENGINE
    # Genetic algorithm discovery of novel trading logic combinations
    # =================================================================

    # Tier 6 Dimension B: Strategy Invention Engine
    enable_strategy_invention: bool = True  # Genetic algorithm strategy discovery (offline only)
    strategy_population_size: int = 30  # Population size for GA evolution
    strategy_evolution_generations: int = 5  # Number of GA generations per cycle

    # --- Entropy-Based Position Sizing (US-098) ---
    enable_entropy_sizing: bool = True  # Shannon entropy of agent votes → position multiplier

    # --- Regime Event Broadcasting (US-099) ---
    enable_regime_broadcast: bool = True  # Event-driven regime transition propagation

    # --- Agent Accuracy Matrix (US-100) ---
    enable_agent_accuracy_matrix: bool = True  # Phase 25 (US-155): Per-agent accuracy tracking by regime/pair

    # --- Meta-Learning Hyperparameter Adaptation (Tier 6) ---
    enable_meta_learning: bool = True  # Online Bayesian adaptation of agent hyperparameters (Tier 6)

    # --- Pair Transfer Learning (US-101) ---
    enable_pair_transfer: bool = True  # Share learned weights between correlated FX pairs

    # --- Temporal Attention (US-102) ---
    enable_temporal_attention: bool = True  # Learned attention over multi-timeframe signals

    # --- Live Position Management (US-103) ---
    enable_live_position_management: bool = True  # Prediction-driven open position management

    # --- Affinity Portfolio Sizing (US-104) ---
    enable_affinity_portfolio: bool = True  # Pair affinity → continuous position sizing

    # --- Execution Routing (US-105) ---
    enable_execution_routing: bool = True  # Smart order slicing for large positions

    # --- Model Routing (US-106) ---
    enable_model_routing: bool = True  # Bandit-based pair-specific model selection

    # --- Training Augmentation (US-107) ---
    enable_training_augmentation: bool = True  # Synthetic data augmentation for retrain

    # --- Attention Feedback (US-108) ---
    enable_attention_feedback: bool = True  # Temporal attention agent consensus feedback

    # --- Execution Quality Optimizer (US-109) ---
    enable_execution_quality_optimizer: bool = True  # Cost profiling by model+regime

    # --- Threshold Optimizer (US-110) ---
    enable_threshold_optimizer: bool = True  # Phase 25 (US-153): Adaptive gate threshold tuning

    # --- Dynamic Risk Allocation (US-111) ---
    enable_dynamic_risk_allocation: bool = True  # P/L distribution risk sizing

    # --- Model Calibration (US-112) ---
    enable_model_calibration: bool = True  # Per-model confidence calibration

    # --- Pair-Regime-Agent Matrix (US-113) ---
    enable_pair_regime_agent_matrix: bool = True  # Phase 27 (US-167): 3-way interaction tracking

    # --- Signal Timing (US-114) ---
    enable_signal_timing: bool = True  # Scan frequency optimization

    # --- Memory Manager (US-115) ---
    enable_memory_manager: bool = True  # LRU cache and log rotation

    # --- Module Activation (US-116) ---
    enable_module_activation: bool = True  # Module dispatch scheduling

    # --- Agent Lifecycle (US-117) ---
    enable_agent_lifecycle: bool = True  # Phase 25 (US-151): Agent fitness and quarantine

    # --- Health Registry (US-118) ---
    enable_health_registry: bool = True  # Phase 27 (US-164): Orchestrator health monitoring

    # --- Observation Consumer (US-119) ---
    enable_observation_consumer: bool = True  # Phase 27 (US-163): Pattern detection from observations

    # --- Replay Validator (US-120) ---
    enable_replay_validator: bool = True  # Historical replay and drift detection

    # --- Microstructure Regime Detection (US-074) ---
    enable_microstructure_regime: bool = True  # Augment regime detection with spread signals
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
    enable_dynamic_hedging: bool = True  # Auto-open inverse positions during stress
    hedge_min_correlation: float = 0.65  # Min |correlation| for hedge candidates

    # --- Observational Learning (US-078) ---
    enable_observational_learning: bool = True  # Synthetic agent pattern extraction
    observational_synthetic_weight: float = 0.20  # Blend ratio: synthetic vs live

    # --- Smart Execution (US-075) ---
    enable_smart_execution: bool = True  # Use VWAP/TWAP slicing for larger orders
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
    off_hours_spread_multiplier: float = 3.0  # Widen spread tolerance when --force (off-hours)
    max_slippage_pips: float = 2.0  # Max acceptable slippage in pips
    min_liquidity_score: float = 0.3  # Minimum liquidity score (0-1)

    # --- Learning Loop Config (post-trade agent weight updates) ---
    enable_agent_learning: bool = True
    enable_llm_trade_analysis: bool = False  # LLM deep analysis for losing trades (US-009)
    weight_boost_on_win: float = 0.1  # Weight increase on winning trade
    weight_penalty_on_loss: float = 0.15  # Weight decrease on losing trade

    # --- Drift Auto-Remediation (US-XXX) ---
    enable_drift_remediator: bool = True  # Auto-retrain when drift is detected
    drift_remediation_cooldown_hours: int = 12  # Hours between remediation attempts
    drift_retrain_timeout_minutes: int = 30  # Max time for retrain subprocess
    drift_remediation_check_interval: int = 20  # Run remediator every N scan cycles

    # --- Model A/B Testing (True A/B: load both incumbent + candidate at scan time) ---
    enable_true_ab_testing: bool = True  # Disabled by default; enabled in smart profile

    # --- Phase 46-47: MTF Confluence & Ensemble Conflict ---
    mtf_confluence_enabled: bool = True  # Elder's Triple Screen via MTF module
    ensemble_conflict_enabled: bool = True  # Ensemble disagreement resolver

    # --- Episodic Market Memory (Pattern Suppression) ---
    enable_episodic_memory: bool = True  # Record and query episodic trade patterns
    episodic_suppression_min_samples: int = 3  # Min similar episodes before suppression activates
    episodic_suppression_loss_rate: float = 0.70  # Loss rate threshold for suppression
    episodic_memory_max_episodes: int = 500  # Max episodes to keep in memory

    # --- Pre-Trade Veto Window (US-511) ---
    veto_window_ms_default: int = 0  # 0 = disabled; 1-5000ms hold before execution
    veto_window_ms_per_pair: Dict[str, int] = field(default_factory=dict)  # pair-specific overrides

    # --- Autonomous Performance-Driven PRD Generation (Ralph Trigger) ---
    enable_auto_ralph: bool = True  # Auto-generate PRD stories from performance gaps
    auto_ralph_win_rate_threshold: float = 0.45  # Win rate below this triggers a story
    auto_ralph_min_trades: int = 10  # Min trades needed before analysis runs
    auto_ralph_check_interval_hours: int = 24  # Hours between autonomous PRD checks
    auto_ralph_max_gaps_per_run: int = 5  # Max stories per auto-generated PRD

    # --- Meta-Cybernetic Change Pipeline ---
    # When enabled, every self-heal / drift / perf-gap signal opens a
    # ChangePackage that walks the 9-stage pipeline (intake → diagnosis →
    # proposal → eval → constitution → human approval → staged deploy →
    # monitor → post-deploy review). Default OFF — flip on after the
    # smoke test on practice. See plan: ".../meta-cybernetic change pipeline".
    enable_meta_manager: bool = False
    meta_manager_max_concurrent: int = 1
    meta_manager_constitution_path: str = ".claude/rules/constitution.json"
    # When False, MetaManager is constructed with a no-op specialist_invoker
    # so the 9-stage pipeline runs without invoking Claude (constitution +
    # scorecard + revert_by_id are pure Python and gate everything that
    # matters). Honors the project rule: Buddy's runtime is Claude-free.
    # Flip to True only when you explicitly want enrichment text in the
    # change ledger and accept LLM in the runtime hot path.
    meta_manager_use_llm: bool = False
    staged_deploy_shadow_cycles: int = 20
    staged_deploy_canary_trades: int = 10

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

        # Validate critical financial/risk fields (Config Validation Gate)
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 0.20:
            logger.warning(
                f"risk_per_trade_pct={self.risk_per_trade_pct} outside safe range (0, 0.20]. "
                f"Clamping to 0.02."
            )
            self.risk_per_trade_pct = 0.02
        if self.max_open_risk_pct < 0 or self.max_open_risk_pct > 0.50:
            logger.warning(
                f"max_open_risk_pct={self.max_open_risk_pct} outside safe range [0, 0.50]. "
                f"Clamping to 0.15."
            )
            self.max_open_risk_pct = 0.15
        if self.leverage <= 0:
            logger.warning(f"leverage={self.leverage} must be positive. Setting to 50.")
            self.leverage = 50
        if self.min_risk_reward_ratio < 1.0:
            logger.warning(
                f"min_risk_reward_ratio={self.min_risk_reward_ratio} below 1.0 violates "
                f"risk management rules. Setting to 1.2."
            )
            self.min_risk_reward_ratio = 1.2
        if self.lookback_candles < 10:
            logger.warning(
                f"lookback_candles={self.lookback_candles} too small for reliable signals. "
                f"Setting to 100."
            )
            self.lookback_candles = 100

        low_sl_mult = self.regime_atr_multipliers.get("LOW", {}).get("sl_mult", 0)
        if low_sl_mult < 1.2:
            raise ValueError(
                "LOW regime sl_mult must be >= 1.2 (ranging markets require wider stops)"
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
        """Get pip value for a pair from InstrumentRegistry.

        Args:
            pair: Trading pair symbol (e.g., "EUR_USD").

        Returns:
            Pip value for the pair, or 0.0001 as default if not found.
        """
        try:
            return get_registry().get(pair).pip_value
        except KeyError:
            # Fallback to default pip value if instrument not in registry
            return 0.0001

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
            # Preserve user-set blocked_pairs — profiles should not overwrite runtime blocks
            if field_name == "blocked_pairs":
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
