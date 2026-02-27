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
    SCAN_PROFILE_BALANCED: {},
    # Fewer trades, higher signal quality requirements.
    "conservative": {
        "min_confidence": 58.0,
        "min_momentum": 0.28,
        "max_drawdown_pct": 0.020,
        "min_atr_pips": 6.0,
        "min_volatility_regime": 2,
    },
    # More trade frequency with looser gates (still risk-bounded).
    "aggressive": {
        "min_confidence": 45.0,
        "min_momentum": 0.12,
        "max_drawdown_pct": 0.035,
        "min_atr_pips": 3.0,
        "min_volatility_regime": 0,
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

    # Session filter (UTC hours)
    # FX markets are open 24/5 (Sun 22:00 UTC – Fri 22:00 UTC).
    # Enabled by default to restrict scanning to London+NY hours only.
    enable_session_filter: bool = True
    session_start_utc: int = 8   # London open (08:00 UTC)
    session_end_utc: int = 21    # NY close (21:00 UTC = 17:00 ET winter, 16:00 ET summer)

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

    # Execution settings (from buddy_scanner)
    enable_execution: bool = False  # Enable trade execution
    daily_trade_limit: int = 30  # Max trades per day
    position_sizing_enabled: bool = True
    aggressive_mode: bool = True  # Enable larger positions for compounding

    # RL model settings (disabled by default to avoid TF/PyTorch GPU conflicts)
    use_rl_sizer: bool = False  # RL position sizing (PPO model)
    use_rl_gates: bool = False  # RL gate threshold optimizer (SAC model)
    use_rl_exits: bool = False  # RL optimal exit timing (PPO model)

    # ATR-based SL/TP (from buddy_scanner)
    atr_sl_multiplier: float = 1.0  # SL = 1.0x ATR
    atr_tp_multiplier: float = 1.5  # TP = 1.5x ATR
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
    enable_agent_trade_promotion: bool = True
    agent_promotion_min_confidence: float = 0.52
    agent_promotion_requires_risk: bool = True
    use_master_pair_models: bool = True

    # Default pairs (for easy access)
    default_pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())
    pip_values: Dict[str, float] = field(default_factory=lambda: PIP_VALUES.copy())

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

    # --- Uncertainty Blocking ---
    max_uncertainty_score: float = 0.4  # Block if uncertainty above this
    max_model_disagreement: float = 0.5  # Block if disagreement above this

    # --- Circuit-Breaker Config ---
    enable_circuit_breakers: bool = True
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
    weight_boost_on_win: float = 0.1  # Weight increase on winning trade
    weight_penalty_on_loss: float = 0.15  # Weight decrease on losing trade

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
        self.sub_inference_window_checks = max(1, int(self.sub_inference_window_checks))
        self.sub_inference_workers = max(1, int(self.sub_inference_workers))
        self.sub_inference_max_candidates = max(1, int(self.sub_inference_max_candidates))
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
        for field_name, value in overrides.items():
            setattr(self, field_name, value)
        # Keep regime in valid classifier range.
        self.min_volatility_regime = max(0, min(3, int(self.min_volatility_regime)))

    def is_within_session(self) -> bool:
        """Check if current time is within Forex trading session.
        
        Forex market hours (UTC):
        - Opens: Sunday 22:00 UTC (5 PM ET Sunday)
        - Closes: Friday 22:00 UTC (5 PM ET Friday)
        - Peak liquidity hours: 08:00-21:00 UTC (London + NY sessions)
        
        The session filter restricts trading to peak liquidity hours when
        major pairs like EUR/USD and NZD/USD have tight spreads and
        good volatility.
        
        Returns:
            True if within peak trading hours AND market is open (not weekend)
        """
        if not self.enable_session_filter:
            logger.debug("Session filter disabled, allowing trade")
            return True

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # === WEEKEND MARKET CLOSURE CHECK ===
        # Forex markets close Friday 22:00 UTC and reopen Sunday 22:00 UTC
        # Saturday (weekday=5) = closed all day
        if weekday == 5:  # Saturday
            logger.debug("Weekend closure: Saturday, blocking trade")
            return False
            
        # Sunday (weekday=6) = opens at 22:00 UTC
        if weekday == 6 and hour < 22:
            logger.debug("Weekend closure: Sunday before 22:00 UTC, blocking trade")
            return False
            
        # Friday (weekday=4) = closes at 22:00 UTC
        if weekday == 4 and hour >= 22:
            logger.debug("Weekend closure: Friday after 22:00 UTC, blocking trade")
            return False
        
        # === PEAK LIQUIDITY HOURS CHECK ===
        # Check if within London+NY overlap (08:00-21:00 UTC by default)
        within_hours = self.session_start_utc <= hour < self.session_end_utc
        
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
        
        return within_hours

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
