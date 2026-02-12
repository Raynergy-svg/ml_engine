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

    # Data fetching
    lookback_candles: int = 200
    granularity: str = "H1"

    # Interactive mode
    non_interactive: bool = field(default_factory=lambda: not sys.stdin.isatty())

    # Gate thresholds (aligned with InferenceConfig)
    min_confidence: float = 50.0  # Ridge ADX score (0-100 scale)
    min_momentum: float = 0.20    # XGBoost percentile (0-1 scale)
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown

    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    leverage: int = 50

    # Session filter (UTC hours)
    # FX markets are open 24/5 (Sun 22:00 UTC – Fri 22:00 UTC).
    # Enabled by default to restrict scanning to London+NY hours only.
    enable_session_filter: bool = True
    session_filter_enabled: bool = True  # Alias for compatibility
    session_start_utc: int = 8   # London open
    session_end_utc: int = 21    # NY close (17:00 ET ≈ 21:00 UTC in summer)

    # Volatility filter
    min_atr_pips: float = 5.0  # Minimum ATR in pips to trade
    min_candles: int = 100  # Minimum candles required

    # TCN Volatility Regime gate (GLOBAL - applies to all pairs)
    # Only allow trades in HIGH (2) or EXTREME (3) volatility regimes
    # Valid values: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    min_volatility_regime: int = 2
    require_tcn_volatility: bool = True  # Block ALL trades if TCN model unavailable

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

    # Default pairs (for easy access)
    default_pairs: List[str] = field(default_factory=lambda: DEFAULT_PAIRS.copy())
    pip_values: Dict[str, float] = field(default_factory=lambda: PIP_VALUES.copy())

    # Output
    top_n: int = 5  # Show top N pairs
    show_all: bool = False  # Show all pairs including failed gates

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

    def is_within_session(self) -> bool:
        """Check if current time is within trading session."""
        if not self.enable_session_filter:
            return True

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour

        return self.session_start_utc <= hour < self.session_end_utc

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
