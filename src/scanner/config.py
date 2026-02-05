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


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to config file (uses DEFAULT_CONFIG_PATH if None)
        
    Returns:
        Dict containing parsed YAML config
    """
    path = config_path or DEFAULT_CONFIG_PATH
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


@dataclass
class ScannerConfig:
    """Configuration for the FX Scanner.
    
    ALL confidence/probability values use 0-1 scale internally.
    Display methods convert to percentage format (e.g., 65%).
    
    Attributes:
        config_path: Path to YAML config file (absolute path recommended)
        pairs: List of FX pairs to scan
        parallel_workers: Number of concurrent pair scans
        lookback_candles: Number of candles to fetch for analysis
        granularity: OANDA timeframe (H1, M15, etc.)
        non_interactive: Skip stdin prompts (for cron/CI)
        
        # Gate thresholds (ALL 0-1 SCALE)
        min_tcn_probability: Minimum Transformer direction confidence (0-1)
        min_confidence: Minimum Ridge ADX confidence (0-1, internally normalized)
        min_momentum: Minimum XGBoost momentum percentile (0-1)
        max_drawdown_pct: Maximum RF drawdown percentage (0-1)
        
        # TCN Forward Volatility (predicts FUTURE regime)
        # ALLOW: STABLE_NEXT (1), ACTIVE_NEXT (2)
        # BLOCK: QUIET_NEXT (0), EXTREME_NEXT (3)
        
        # Position sizing
        account_equity: Account balance (0 = fetch from OANDA)
        risk_per_trade_pct: Risk percentage per trade (0-1)
        
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
    
    # Data fetching
    lookback_candles: int = 200
    granularity: str = "H1"
    
    # Interactive mode
    non_interactive: bool = field(default_factory=lambda: not sys.stdin.isatty())
    
    # ==========================================================================
    # GATE THRESHOLDS (ALL 0-1 SCALE - from docs optimal values)
    # ==========================================================================
    min_tcn_probability: float = 0.60   # Transformer direction >= 60%
    min_confidence: float = 0.50        # Ridge ADX normalized (0-1), maps to 50/100
    min_momentum: float = 0.20          # XGBoost percentile >= 20%
    max_drawdown_pct: float = 0.025     # RF max drawdown <= 2.5%
    min_meta_confidence: float = 0.55   # Meta-labeler success probability >= 55%
    
    # ==========================================================================
    # TCN FORWARD VOLATILITY GATE (REQUIRED - predicts FUTURE regime)
    # ==========================================================================
    # New forward-looking prediction (48-bar lookahead):
    # ALLOW: STABLE_NEXT (1) - moderate, predictable volatility
    # ALLOW: ACTIVE_NEXT (2) - high, best for entries  
    # BLOCK: QUIET_NEXT (0) - insufficient expected movement
    # BLOCK: EXTREME_NEXT (3) - news events, high risk
    require_tcn_volatility: bool = True  # FAIL scan if TCN model missing
    extreme_regime_warning: bool = True  # Log warning when regime == EXTREME_NEXT (3)
    
    # ==========================================================================
    # JOINT MODEL ENFORCEMENT
    # ==========================================================================
    use_joint_models_only: bool = True  # Load ONLY from trained_data/models/joint/
    
    # ==========================================================================
    # POSITION SIZING
    # ==========================================================================
    account_equity: float = 0.0         # 0 = fetch live NAV from OANDA
    risk_per_trade_pct: float = 0.02    # 2% risk per trade (professional standard)
    leverage: int = 50
    
    # Confidence-tiered position multipliers (0-1 confidence thresholds)
    confidence_tier_low: float = 0.50       # 0.50-0.65 = 0.5x base position
    confidence_tier_medium: float = 0.65    # 0.65-0.80 = 1.0x base position
    confidence_tier_high: float = 0.80      # 0.80+ = 2.0x base position
    position_multiplier_low: float = 0.5
    position_multiplier_medium: float = 1.0
    position_multiplier_high: float = 2.0
    
    # ==========================================================================
    # H1 TIMEFRAME SETTINGS (from docs optimal values)
    # ==========================================================================
    # ATR-based SL/TP
    atr_sl_multiplier: float = 1.5      # SL = 1.5x ATR (professional H1 standard)
    atr_tp_multiplier: float = 3.0      # TP = 3.0x ATR (2:1 R:R ratio)
    min_sl_pips: float = 15.0           # Floor for tight risk
    max_sl_pips: float = 80.0           # Cap for extreme volatility
    min_tp_pips: float = 30.0           # Ensures minimum R:R
    max_tp_pips: float = 200.0          # Reasonable H1 swing target
    
    # Default SL/TP (fallback if ATR unavailable)
    sl_pips: float = 15.0
    tp_pips: float = 30.0
    
    # ==========================================================================
    # SESSION FILTER (UTC hours - London/NY overlap)
    # ==========================================================================
    enable_session_filter: bool = False  # Only block weekends by default
    session_filter_enabled: bool = False  # Alias for compatibility
    session_start_utc: int = 8           # London open (only if enable_session_filter=True)
    session_end_utc: int = 21            # NY close (only if enable_session_filter=True)
    
    # ==========================================================================
    # VOLATILITY FILTER
    # ==========================================================================
    min_atr_pips: float = 8.0           # Skip dead markets (from docs)
    min_candles: int = 100              # Minimum candles required
    
    # ==========================================================================
    # BACKTEST GATE (Phase 2: Scanner Accuracy)
    # ==========================================================================
    require_backtest: bool = True       # Run backtest before showing tradeable signals
    min_backtest_win_rate: float = 0.45 # Minimum win rate to pass gate (0-1)
    min_backtest_trades: int = 10       # Minimum simulated trades for valid backtest
    backtest_window: int = 50           # Number of candles for backtest
    
    # ==========================================================================
    # TRADE LIMITS
    # ==========================================================================
    max_trades_per_day: int = 3         # H1 = swing trading (from docs)
    daily_trade_limit: int = 3          # Alias for compatibility
    enable_execution: bool = False      # Enable trade execution
    position_sizing_enabled: bool = True
    aggressive_mode: bool = True        # Enable compounding with Kelly-based sizing
    
    # High probability TP bonus
    high_prob_threshold: float = 0.65   # Confidence threshold for TP bonus (0-1)
    high_prob_tp_bonus: float = 20.0    # Extra pips at high probability
    
    # ==========================================================================
    # WATCH MODE / INCREMENTAL CACHING
    # ==========================================================================
    watch_interval_seconds: int = 300   # 5 minutes
    incremental_cache_minutes: int = 5
    incremental_enabled: bool = True
    price_change_threshold: float = 0.001  # 0.1% price change triggers re-fetch
    max_workers: int = 4
    
    # ==========================================================================
    # OUTPUT
    # ==========================================================================
    default_pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())
    pip_values: Dict[str, float] = field(default_factory=lambda: PIP_VALUES.copy())
    top_n: int = 5                      # Show top N pairs
    show_all: bool = False              # Show all pairs including failed gates
    
    # Loaded YAML config (lazy loaded)
    _yaml_config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)
        
        # Resolve relative paths to absolute
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        if not self.model_dir.is_absolute():
            self.model_dir = PROJECT_ROOT / self.model_dir
    
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
            with open(self.config_path, 'r') as f:
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
    
    def is_within_session(self) -> bool:
        """Check if current time is within trading session.
        
        FX markets are open 24/5:
        - Open: Sunday 5pm EST (22:00 UTC)
        - Close: Friday 5pm EST (22:00 UTC)
        
        Only blocks on weekends (Saturday after 22:00 UTC to Sunday 22:00 UTC).
        Hourly filter is optional and DISABLED by default now.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        # Weekend check (FX market closed)
        # Saturday = 5, Sunday = 6
        # Market closes Friday 22:00 UTC, opens Sunday 22:00 UTC
        if now.weekday() == 5:  # Saturday - always closed
            return False
        if now.weekday() == 6 and now.hour < 22:  # Sunday before 22:00 UTC - closed
            return False
        if now.weekday() == 4 and now.hour >= 22:  # Friday after 22:00 UTC - closed
            return False
        
        # Optional hourly filter (for optimal London/NY overlap)
        if self.enable_session_filter:
            hour = now.hour
            return self.session_start_utc <= hour < self.session_end_utc
        
        return True  # Weekday, market is open
    
    def get_position_multiplier(self, confidence: float) -> float:
        """Get position multiplier based on confidence tier.
        
        Confidence tiers (all 0-1 scale):
        - 0.50-0.65: 0.5x base position (conservative)
        - 0.65-0.80: 1.0x base position (standard)
        - 0.80+: 2.0x base position (aggressive)
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Position multiplier
        """
        if confidence >= self.confidence_tier_high:
            return self.position_multiplier_high
        elif confidence >= self.confidence_tier_medium:
            return self.position_multiplier_medium
        elif confidence >= self.confidence_tier_low:
            return self.position_multiplier_low
        else:
            return 0.0  # Below minimum, no trade
    
    @staticmethod
    def format_confidence_pct(confidence: float) -> str:
        """Format confidence (0-1) as percentage string.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Formatted string like "65%"
        """
        return f"{confidence * 100:.0f}%"
    
    @staticmethod
    def normalize_confidence(score: float, from_scale: int = 100) -> float:
        """Normalize confidence from legacy scale to 0-1.
        
        Args:
            score: Raw score (e.g., 50 on 0-100 scale)
            from_scale: Source scale maximum (default 100)
            
        Returns:
            Normalized score (0-1)
        """
        return score / from_scale if from_scale > 0 else 0.0
    
    @classmethod
    def from_cli_args(
        cls,
        config_path: Optional[str] = None,
        pairs: Optional[List[str]] = None,
        top_n: int = 5,
        show_all: bool = False,
        granularity: str = "H1",
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
        
        if non_interactive:
            config.non_interactive = True
        
        # Disable session filter if --force or non-interactive
        if force or non_interactive:
            config.enable_session_filter = False
            config.session_filter_enabled = False
        
        return config
