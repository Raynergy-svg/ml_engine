"""
Buddy Scanner - Enhanced Multi-Pair FX Scanner
===============================================
Upgraded scanner with:
- Parallel pair scanning (ThreadPoolExecutor)
- Integrated position sizing
- 50-candle quick backtesting
- Drift detection with retraining prompts
- Correlation analysis for diversification
- Historical accuracy tracking via MemoryClient

Usage:
    from buddy_scanner import BuddyScanner
    
    scanner = BuddyScanner(config_path="config_improved_H1.yaml")
    results = scanner.scan(pairs=["EUR_USD", "GBP_USD"], top_n=5)
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*serialized model.*')
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging():
    """Context manager to suppress verbose logging during scan."""
    # Suppress TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Loggers to quiet
    loggers_to_quiet = [
        'utils', 'modular_trainers', 'modular_inference', 'feature_engineering',
        'modular_data_loaders', 'tensorflow', 'absl', 'xgboost', 'oanda_practice',
        'position_sizing', 'risk_management', 'memory_client', 'fx_paper',
    ]
    
    # Store original levels
    original_levels = {}
    for name in loggers_to_quiet:
        log = logging.getLogger(name)
        original_levels[name] = log.level
        log.setLevel(logging.ERROR)
    
    try:
        yield
    finally:
        # Restore original levels
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm
    console = Console()
except ImportError:
    console = None

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

ALL_PAIRS = MAJOR_PAIRS + CROSS_PAIRS

# Pip values for position sizing
PIP_VALUES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_JPY": 0.01,
    "USD_CHF": 0.0001, "AUD_USD": 0.0001, "USD_CAD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_JPY": 0.01,
    "GBP_JPY": 0.01, "AUD_JPY": 0.01, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_CHF": 0.0001, "GBP_CHF": 0.0001,
}


@dataclass
class ScanConfig:
    """Configuration for buddy scanner.
    
    H1 Timeframe Optimizations:
    - ATR-based SL/TP (1.5x ATR for SL, 3x ATR for TP = 2:1 R:R)
    - Session filtering (London/NY best)
    - Volatility filter (skip low-ATR periods)
    - Trailing stop after 1R profit
    """
    lookback_candles: int = 200
    parallel_workers: int = 4
    backtest_window: int = 50
    auto_retrain_prompt: bool = True
    drift_threshold: float = 0.03
    min_confidence: float = 0.52  # 52%+ (above random)
    min_gate_confidence: float = 0.55  # 55%+ required for gates to pass
    
    # H1 ATR-based SL/TP settings
    atr_period: int = 14  # Standard H1 ATR period
    atr_sl_multiplier: float = 1.0  # SL = 1.0x ATR (tighter for scalping)
    atr_tp_multiplier: float = 1.5  # TP = 1.5x ATR (tighter targets)
    rr_multiplier: float = 1.5  # Fallback R:R if ATR unavailable
    
    # Tight pip ranges for quick trades (user requested)
    min_sl_pips: float = 15.0  # Fixed 15 pip SL as requested
    max_sl_pips: float = 15.0  # Cap at 15 pips - don't exceed
    min_tp_pips: float = 20.0  # Minimum 20 pip TP
    max_tp_pips: float = 30.0  # Max 30 pip TP (base)
    
    # High probability TP bonus (>65% confidence)
    high_prob_threshold: float = 0.65  # Probability threshold for bonus TP
    high_prob_tp_bonus: float = 20.0  # Extra pips added when prob > 65%
    
    # Position sizing - optimized for $101k account
    position_sizing_enabled: bool = True
    account_equity: float = 0.0  # 0 = fetch from OANDA live
    risk_per_trade_pct: float = 0.02  # 2% risk per trade for larger lots
    leverage: int = 50  # 50:1 leverage
    aggressive_mode: bool = True  # Enabled for larger position sizes
    max_trades_per_day: int = 30  # Max 30 trades/day as requested
    
    # Session filter (UTC hours)
    enable_session_filter: bool = True
    session_start_utc: int = 8  # London open
    session_end_utc: int = 21  # NY close
    
    # Volatility filter
    min_atr_pips: float = 8.0  # Skip if H1 ATR < 8 pips (dead market)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScanConfig":
        """Create config from dict (YAML scan section)."""
        return cls(
            lookback_candles=d.get("lookback_candles", 200),
            parallel_workers=d.get("parallel_workers", 4),
            backtest_window=d.get("backtest_window", 50),
            auto_retrain_prompt=d.get("auto_retrain_prompt", True),
            drift_threshold=d.get("drift_threshold", 0.03),
            min_confidence=d.get("min_confidence", 0.52),
            min_gate_confidence=d.get("min_gate_confidence", 0.55),
            atr_period=d.get("atr_period", 14),
            atr_sl_multiplier=d.get("atr_sl_multiplier", 1.5),
            atr_tp_multiplier=d.get("atr_tp_multiplier", 3.0),
            rr_multiplier=d.get("rr_multiplier", 2.0),
            min_sl_pips=d.get("min_sl_pips", 15.0),
            max_sl_pips=d.get("max_sl_pips", 15.0),
            min_tp_pips=d.get("min_tp_pips", 20.0),
            max_tp_pips=d.get("max_tp_pips", 30.0),
            high_prob_threshold=d.get("high_prob_threshold", 0.65),
            high_prob_tp_bonus=d.get("high_prob_tp_bonus", 20.0),
            position_sizing_enabled=d.get("position_sizing_enabled", True),
            account_equity=d.get("account_equity", 101000.0),
            risk_per_trade_pct=d.get("risk_per_trade_pct", 0.02),
            leverage=d.get("leverage", 50),
            aggressive_mode=d.get("aggressive_mode", False),
            max_trades_per_day=d.get("max_trades_per_day", 3),
            enable_session_filter=d.get("enable_session_filter", True),
            session_start_utc=d.get("session_start_utc", 8),
            session_end_utc=d.get("session_end_utc", 21),
            min_atr_pips=d.get("min_atr_pips", 8.0),
        )


@dataclass
class EnhancedScanResult:
    """Enhanced scan result with position sizing and backtest data."""
    # Core fields
    pair: str
    direction: str
    confidence: float
    tcn_confidence: float
    ridge_confidence: float
    volatility_percentile: float
    trend_strength: float
    entry_score: float
    current_price: float
    atr: float
    gates_passed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Position sizing fields
    recommended_lots: float = 0.0
    risk_pct: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    confidence_level: str = "unknown"
    
    # Model status per pair
    has_pair_model: bool = False  # Whether pair has dedicated trained model
    model_accuracy: Optional[float] = None  # Model accuracy for this pair
    needs_training: bool = False  # Whether pair needs (re)training
    training_reason: Optional[str] = None  # Why training is needed
    
    # Backtest fields (populated for top 3 only)
    backtest_win_rate: Optional[float] = None
    backtest_sharpe: Optional[float] = None
    backtest_trades: Optional[int] = None
    
    # Warning fields
    correlation_warning: Optional[str] = None
    drift_warning: bool = False
    historical_accuracy: Optional[float] = None
    
    @property
    def overall_score(self) -> float:
        """Combined score for ranking pairs."""
        hist_acc = self.historical_accuracy if self.historical_accuracy else 0.5
        score = (
            self.confidence * 0.30 +
            self.entry_score * 0.25 +
            self.trend_strength * 0.20 +
            hist_acc * 0.15 +
            self.volatility_percentile * 0.10
        )
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "confidence": self.confidence,
            "tcn_confidence": self.tcn_confidence,
            "ridge_confidence": self.ridge_confidence,
            "volatility_percentile": self.volatility_percentile,
            "trend_strength": self.trend_strength,
            "entry_score": self.entry_score,
            "overall_score": self.overall_score,
            "current_price": self.current_price,
            "atr": self.atr,
            "gates_passed": self.gates_passed,
            "recommended_lots": self.recommended_lots,
            "risk_pct": self.risk_pct,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "confidence_level": self.confidence_level,
            "backtest_win_rate": self.backtest_win_rate,
            "backtest_sharpe": self.backtest_sharpe,
            "correlation_warning": self.correlation_warning,
            "drift_warning": self.drift_warning,
            "timestamp": self.timestamp.isoformat(),
        }


class BuddyScanner:
    """
    Enhanced multi-pair FX scanner with parallel processing,
    position sizing, backtesting, and drift detection.
    """
    
    def __init__(
        self,
        config_path: str = "config_improved_H1.yaml",
        account_equity: Optional[float] = None,
        use_rl_sizer: bool = True,
    ):
        """
        Initialize the scanner.
        
        Args:
            config_path: Path to config YAML file
            account_equity: Override account equity for position sizing
            use_rl_sizer: Use RL-based position sizing instead of Kelly
        """
        self.config_path = config_path
        self._cfg = None
        self._scan_config: Optional[ScanConfig] = None
        self._oanda_client = None
        self._feature_engineer = None
        self._modular_ensemble = None
        self._model_78 = None  # Verified 78% model from Colab
        self._position_sizer = None
        self._risk_manager = None
        self._memory_client = None
        self._model_meta = None
        self._account_equity = account_equity
        self._use_rl_sizer = use_rl_sizer
        
        # Cached historical accuracy per pair
        self._historical_accuracy: Dict[str, float] = {}
        
        # Cached returns for correlation analysis
        self._pair_returns: Dict[str, pd.Series] = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML."""
        if self._cfg is not None:
            return self._cfg
        
        with suppress_logging():
            from utils import load_config
            self._cfg = load_config(self.config_path)
        
        # Load scan-specific config
        scan_dict = self._cfg.get("scan", {})
        self._scan_config = ScanConfig.from_dict(scan_dict)
        
        # Override account equity if provided explicitly
        if self._account_equity:
            self._scan_config.account_equity = self._account_equity
        
        return self._cfg
    
    def _fetch_live_nav(self) -> Optional[float]:
        """Fetch live NAV from OANDA for proper compounding.
        
        Returns:
            Account NAV or None if unavailable
        """
        try:
            self._init_oanda_client()
            result = self._oanda_client.get_account_summary()
            account = result.get('account', {})
            nav = float(account.get('NAV', 0))
            if nav > 0:
                logger.debug(f"Fetched live NAV: ${nav:,.2f}")
                return nav
        except Exception as e:
            logger.debug(f"Could not fetch live NAV: {e}")
        return None
    
    def _fetch_trades_today(self) -> int:
        """Fetch count of trades opened today from OANDA.
        
        Returns:
            Number of trades opened today (0 if unable to fetch)
        """
        try:
            self._init_oanda_client()
            from datetime import datetime, timezone
            
            # Get today's date in UTC
            today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            # Fetch recent trades
            result = self._oanda_client._request(
                'GET',
                f'/accounts/{self._oanda_client._config.account_id}/trades',
                params={'state': 'ALL', 'count': 100}
            )
            trades = result.get('trades', [])
            
            # Count trades opened today
            trades_today = 0
            for t in trades:
                open_time = t.get('openTime', '')
                if open_time.startswith(today_utc):
                    trades_today += 1
            
            return trades_today
            
        except Exception as e:
            logger.debug(f"Could not fetch trades today: {e}")
            return 0
    
    def _update_equity_from_oanda(self) -> Tuple[bool, float, int, int]:
        """Update account equity from OANDA and get trade limits.
        
        Returns:
            Tuple of (was_updated, nav, trades_today, trades_remaining)
        """
        nav = self._fetch_live_nav()
        trades_today = self._fetch_trades_today()
        max_trades = self._scan_config.max_trades_per_day
        trades_remaining = max(0, max_trades - trades_today)
        
        # Update equity if fetched successfully and no explicit override
        was_updated = False
        if nav and nav > 0:
            if not self._account_equity:  # Only update if not explicitly set
                self._scan_config.account_equity = nav
                was_updated = True
            else:
                nav = self._account_equity  # Use explicitly set equity
        elif self._account_equity:
            nav = self._account_equity
        else:
            nav = self._scan_config.account_equity or 10000.0  # Fallback
        
        return was_updated, nav, trades_today, trades_remaining
    
    def _fetch_actual_win_rate(self) -> Tuple[float, int]:
        """Fetch actual win rate from OANDA closed trades.
        
        Returns:
            Tuple of (win_rate, total_trades)
        """
        try:
            self._init_oanda_client()
            # Use _request directly as OandaPracticeClient doesn't have get_trades
            result = self._oanda_client._request(
                'GET',
                f'/accounts/{self._oanda_client._config.account_id}/trades',
                params={'state': 'CLOSED', 'count': 100}
            )
            trades = result.get('trades', [])
            
            if not trades:
                return 0.0, 0
            
            wins = sum(1 for t in trades if float(t.get('realizedPL', 0)) > 0)
            total = len(trades)
            win_rate = wins / total if total > 0 else 0.0
            
            logger.info(f"Actual trading performance: {win_rate:.1%} ({wins}W/{total-wins}L)")
            return win_rate, total
            
        except Exception as e:
            logger.debug(f"Could not fetch actual win rate: {e}")
            return 0.0, 0
    
    def _init_oanda_client(self):
        """Initialize OANDA API client."""
        if self._oanda_client is not None:
            return
        
        from oanda_practice import OandaPracticeClient
        self._oanda_client = OandaPracticeClient.from_env()
    
    def _init_feature_engineer(self):
        """Initialize feature engineering."""
        if self._feature_engineer is not None:
            return
        
        cfg = self._load_config()
        with suppress_logging():
            from feature_engineering import FeatureEngineering
            self._feature_engineer = FeatureEngineering(cfg.get("feature_engineering", {}))
    
    def _init_modular_ensemble(self) -> bool:
        """Initialize modular ensemble model. Returns True if loaded."""
        if self._modular_ensemble is not None:
            return True
        
        # Try 78% verified model FIRST (from Colab A100 training)
        try:
            with suppress_logging():
                from model_78_inference import Model78Inference
                model_78 = Model78Inference()
                if model_78.load():
                    self._model_78 = model_78
                    self._model_meta = {
                        "results": {"transformer": {"val_accuracy": model_78.accuracy}}
                    }
                    logger.debug(f"Loaded verified 78% model")
                    return True
        except Exception as e:
            logger.debug(f"78% model not available: {e}")
        
        # Fallback to modular ensemble
        meta_path = Path("trained_data/models/modular_ensemble.meta.json")
        if not meta_path.exists():
            return False
        
        try:
            with suppress_logging():
                from modular_inference import ModularEnsembleInference
                self._modular_ensemble = ModularEnsembleInference(use_rl_sizer=self._use_rl_sizer)
                self._modular_ensemble.load_models()
            
            # Load meta for model info
            self._model_meta = json.loads(meta_path.read_text())
            return True
        except Exception as e:
            logger.warning(f"Could not load modular ensemble: {e}")
            return False
    
    def _init_position_sizer(self):
        """Initialize position sizer."""
        if self._position_sizer is not None:
            return
        
        try:
            with suppress_logging():
                from position_sizing import (
                    DynamicPositionSizer, PositionSizingConfig,
                    create_aggressive_position_sizer, create_kelly_position_sizer
                )
                
                if self._scan_config.aggressive_mode:
                    # AGGRESSIVE Kelly-based sizing for $100k→$1M compounding
                    # 78% win rate, 1.5:1 R:R → Kelly f* = 0.52, half-Kelly = 0.26
                    # With 50:1 leverage, position sizes 5-15 lots
                    config = PositionSizingConfig(
                        risk_per_trade_pct=self._scan_config.risk_per_trade_pct,
                        min_confidence_threshold=0.53,  # 53%+ is above random
                        max_position_multiplier=10.0,  # Allow up to 10x scaling
                        low_confidence_band=(0.53, 0.60),
                        medium_confidence_band=(0.60, 0.75),
                        high_confidence_band=(0.75, 1.0),
                        low_confidence_multiplier=1.5,    # 1.5x at low confidence
                        medium_confidence_multiplier=3.0,  # 3x at medium
                        high_confidence_multiplier=5.0,    # 5x at high confidence
                        max_position_pct=0.80,  # Max 80% margin usage per trade
                        min_position_size=500000,  # Min 5.0 lots
                    )
                else:
                    config = PositionSizingConfig(
                        risk_per_trade_pct=self._scan_config.risk_per_trade_pct,
                        min_confidence_threshold=0.5,
                    )
                self._position_sizer = DynamicPositionSizer(config)
        except ImportError:
            pass  # Silent - position sizing not available
    
    def _init_risk_manager(self):
        """Initialize risk manager."""
        if self._risk_manager is not None:
            return
        
        try:
            with suppress_logging():
                from risk_management import ConfidenceBasedRiskManager, RiskManagementConfig
                self._risk_manager = ConfidenceBasedRiskManager(RiskManagementConfig())
        except ImportError:
            pass  # Silent - risk management not available
    
    def _init_memory_client(self):
        """Initialize memory client for historical accuracy tracking."""
        if self._memory_client is not None:
            return
        
        try:
            with suppress_logging():
                from memory_client import MLEngineMemory
                self._memory_client = MLEngineMemory()
            
            # Load historical accuracy from memory
            self._load_historical_accuracy()
        except ImportError:
            pass  # Silent - memory client not available
    
    def _load_historical_accuracy(self):
        """Load historical accuracy from memory client."""
        if self._memory_client is None:
            return
        
        try:
            self._historical_accuracy = self._memory_client.get_pair_accuracy()
        except Exception:
            pass  # Silent - could not load historical accuracy
    
    def _fetch_pair_data(
        self,
        pair: str,
        granularity: str,
        count: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single pair.
        
        Args:
            pair: Instrument name (e.g., "EUR_USD")
            granularity: Timeframe (e.g., "H1")
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None on error
        """
        self._init_oanda_client()
        
        try:
            from fx_paper import candles_to_ohlcv_df
            resp = self._oanda_client.get_candles(
                pair, granularity=granularity, count=count, price="MBA"
            )
            df = candles_to_ohlcv_df(resp)
            
            if len(df) < 100:
                return None
            
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch {pair}: {e}")
            return None
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a dataframe.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self._init_feature_engineer()
        
        # Create features (suppress logging)
        with suppress_logging():
            df_feat = self._feature_engineer.create_features(df.copy(), include_all=True)
        
        # Clean up
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        df_feat = df_feat.ffill().bfill().fillna(0.0)
        
        return df_feat
    
    def _run_inference(
        self,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[str, float, float, float, bool]:
        """
        Run model inference on prepared features.
        
        Args:
            df_raw: Raw OHLCV DataFrame (for 78% model)
            df_feat: DataFrame with engineered features (for modular ensemble)
            pair: Pair name for logging
            
        Returns:
            Tuple of (direction, confidence, tcn_conf, ridge_conf, gates_passed)
        """
        direction = "LONG"
        confidence = 0.5
        tcn_conf = 0.0
        ridge_conf = 0.0
        gates_passed = False
        
        # Use 78% verified model FIRST
        if self._model_78 is not None:
            try:
                with suppress_logging():
                    direction, prob = self._model_78.predict(df_raw, pair)
                    confidence = prob
                    tcn_conf = prob * 100
                    ridge_conf = prob * 100
                    # Gates pass if confidence >= min_gate_confidence (stricter than min_confidence)
                    # This improves WR by only taking higher-confidence trades
                    gates_passed = confidence >= self._scan_config.min_gate_confidence
                    return direction, min(confidence, 0.95), tcn_conf, ridge_conf, gates_passed
            except Exception as e:
                logger.warning(f"78% model inference failed for {pair}: {e}")
        
        # Fallback to modular ensemble
        if self._modular_ensemble is not None:
            try:
                with suppress_logging():
                    signal = self._modular_ensemble.predict(df_feat)
                
                if signal.tcn_direction is not None:
                    direction = "LONG" if signal.tcn_direction == 1 else "SHORT"
                    tcn_conf = abs(signal.tcn_probability - 0.5) * 200
                    ridge_conf = signal.ridge_confidence
                    confidence = max(tcn_conf, ridge_conf) / 100
                    gates_passed = signal.trade
                elif signal.direction is not None:
                    direction = signal.direction.upper()
                    confidence = signal.confidence if signal.confidence > 0 else 0.5
                    ridge_conf = signal.ridge_confidence
                    tcn_conf = confidence * 100
                    gates_passed = signal.trade
                    
            except Exception as e:
                logger.warning(f"Model inference failed for {pair}: {e}")
        
        # Fallback to technical indicators
        if confidence == 0.5 and "rsi" in df_feat.columns:
            rsi = df_feat["rsi"].iloc[-1]
            if rsi < 30:
                direction = "LONG"
                confidence = 0.5 + (30 - rsi) / 60
            elif rsi > 70:
                direction = "SHORT"
                confidence = 0.5 + (rsi - 70) / 60
            elif "macd" in df_feat.columns:
                macd = df_feat["macd"].iloc[-1]
                direction = "LONG" if macd > 0 else "SHORT"
                confidence = 0.5 + min(abs(macd) * 10, 0.2)
        
        return direction, min(confidence, 0.95), tcn_conf, ridge_conf, gates_passed
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        """
        Calculate technical metrics from features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (vol_pct, trend_str, entry_score, current_price, atr)
        """
        vol_pct = 0.5
        trend_str = 0.5
        entry_score = 0.5
        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df.columns else 0.0
        
        # Volatility percentile
        if "atr" in df.columns:
            atr_series = df["atr"].dropna()
            if len(atr_series) > 20:
                vol_pct = float((atr_series < atr_series.iloc[-1]).mean())
        
        # Trend strength from ADX
        if "adx" in df.columns:
            adx_val = df["adx"].iloc[-1]
            trend_str = min(float(adx_val) / 60, 1.0)
        
        # Entry score based on MA alignment and RSI
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if 30 < rsi < 70:
                entry_score += 0.15
        
        if "sma_20" in df.columns and "sma_50" in df.columns:
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1]
            close = df["close"].iloc[-1]
            
            if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                entry_score += 0.15
        
        return vol_pct, trend_str, entry_score, current_price, atr
    
    def _calculate_position_size(
        self,
        confidence: float,
        atr: float,
        pair: str,
    ) -> Tuple[float, float, float, float, str]:
        """
        Calculate position sizing based on confidence.
        
        Args:
            confidence: Model confidence (0-1)
            atr: ATR value for SL calculation
            pair: Instrument name
            
        Returns:
            Tuple of (lots, risk_pct, sl_pips, tp_pips, confidence_level)
        """
        if not self._scan_config.position_sizing_enabled:
            return 0.0, 0.0, 0.0, 0.0, "disabled"
        
        self._init_position_sizer()
        self._init_risk_manager()
        
        # TIGHT SCALPING: Fixed SL/TP with probability-based TP bonus
        pip_value = PIP_VALUES.get(pair, 0.0001)
        
        # Fixed 15 pip SL (user requested - lower risk)
        sl_pips = self._scan_config.max_sl_pips  # Fixed at 15 pips
        
        # Base TP: 20-30 pips (user requested - quick targets)
        base_tp = (atr * self._scan_config.atr_tp_multiplier) / pip_value if pip_value > 0 else 25.0
        tp_pips = max(self._scan_config.min_tp_pips, min(base_tp, self._scan_config.max_tp_pips))
        
        # HIGH PROBABILITY BONUS: If confidence > 65%, add bonus TP pips
        if confidence >= self._scan_config.high_prob_threshold:
            tp_bonus = self._scan_config.high_prob_tp_bonus
            tp_pips = tp_pips + tp_bonus
            logger.info(f"High probability ({confidence:.1%}) - TP bonus +{tp_bonus} pips → {tp_pips:.1f} pips")
        
        confidence_level = "medium"
        
        if self._risk_manager is not None:
            try:
                risk_result = self._risk_manager.calculate_risk_levels(
                    entry_price=1.0,  # Placeholder
                    raw_confidence=confidence,
                    base_stop_loss_pips=sl_pips,
                    instrument=pair,
                )
                if risk_result.is_valid:
                    sl_pips = risk_result.stop_loss_pips
                    tp_pips = risk_result.take_profit_pips
                    confidence_level = risk_result.confidence_level
            except Exception as e:
                logger.warning(f"Risk calculation failed: {e}")
        
        # Calculate position size
        lots = 0.0
        risk_pct = self._scan_config.risk_per_trade_pct
        
        if self._position_sizer is not None:
            try:
                pos_result = self._position_sizer.calculate_position_size(
                    account_equity=self._scan_config.account_equity,
                    stop_loss_pips=sl_pips,
                    instrument=pair,
                    raw_confidence=confidence,
                )
                if pos_result.is_valid:
                    lots = pos_result.units / 100000  # Convert units to lots
                    risk_pct = pos_result.risk_amount / self._scan_config.account_equity
                    confidence_level = pos_result.confidence_level
            except Exception as e:
                logger.warning(f"Position sizing failed: {e}")
        
        return lots, risk_pct, sl_pips, tp_pips, confidence_level
    
    def _quick_backtest(
        self,
        df: pd.DataFrame,
        direction: str,
        window: int = 50,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Run quick backtest on recent data.
        
        Args:
            df: DataFrame with features
            direction: Predicted direction ("LONG" or "SHORT")
            window: Number of candles to backtest
            
        Returns:
            Tuple of (win_rate, sharpe, num_trades) or (None, None, None) on failure
        """
        if len(df) < window + 10:
            return None, None, None
        
        try:
            # Get recent slice
            df_backtest = df.tail(window + 10).copy()
            
            # Calculate returns
            df_backtest["returns"] = df_backtest["close"].pct_change()
            
            # Generate simple signals based on model direction bias
            # We simulate: enter at each bar, hold for 1 bar
            if direction == "LONG":
                df_backtest["signal_returns"] = df_backtest["returns"]
            else:
                df_backtest["signal_returns"] = -df_backtest["returns"]
            
            # Remove NaN
            signal_returns = df_backtest["signal_returns"].dropna().tail(window)
            
            if len(signal_returns) < 10:
                return None, None, None
            
            # Calculate metrics
            wins = (signal_returns > 0).sum()
            trades = len(signal_returns)
            win_rate = wins / trades if trades > 0 else 0.0
            
            # Sharpe ratio (annualized for hourly data)
            mean_ret = signal_returns.mean()
            std_ret = signal_returns.std()
            sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24) if std_ret > 0 else 0.0
            
            return win_rate, sharpe, trades
            
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return None, None, None
    
    def _calculate_correlations(
        self,
        results: List[EnhancedScanResult],
    ) -> Dict[str, str]:
        """
        Calculate pair correlations and return warnings.
        
        Args:
            results: List of scan results
            
        Returns:
            Dict mapping pair to correlation warning (if any)
        """
        warnings_dict: Dict[str, str] = {}
        
        if len(self._pair_returns) < 2:
            return warnings_dict
        
        try:
            # Build correlation matrix
            pairs_with_data = [r.pair for r in results if r.pair in self._pair_returns]
            if len(pairs_with_data) < 2:
                return warnings_dict
            
            # Create returns DataFrame
            returns_df = pd.DataFrame({
                pair: self._pair_returns[pair]
                for pair in pairs_with_data
                if pair in self._pair_returns
            })
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Check for high correlations (>0.8)
            for i, pair1 in enumerate(pairs_with_data):
                for pair2 in pairs_with_data[i+1:]:
                    if pair1 in corr_matrix.columns and pair2 in corr_matrix.columns:
                        corr = corr_matrix.loc[pair1, pair2]
                        if abs(corr) > 0.8:
                            warnings_dict[pair1] = f"High correlation ({corr:.2f}) with {pair2}"
                            warnings_dict[pair2] = f"High correlation ({corr:.2f}) with {pair1}"
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
        
        return warnings_dict
    
    def _check_model_drift(self) -> Tuple[bool, float, float]:
        """
        Check for model drift.
        
        Returns:
            Tuple of (drift_detected, current_acc, baseline_acc)
        """
        if self._model_meta is None:
            return False, 0.0, 0.0
        
        # Get baseline accuracy from meta
        baseline_acc = 0.0
        if "results" in self._model_meta:
            baseline_acc = self._model_meta.get("results", {}).get("transformer", {}).get("val_accuracy", 0)
        if baseline_acc == 0 and "models" in self._model_meta:
            baseline_acc = self._model_meta.get("models", {}).get("direction", {}).get("metrics", {}).get("val_accuracy", 0)
        
        # For now, we use the same baseline (real drift detection would compare live performance)
        # This is a placeholder - in production you'd track live accuracy
        current_acc = baseline_acc  # Placeholder
        
        drift = abs(current_acc - baseline_acc)
        drift_detected = drift > self._scan_config.drift_threshold
        
        return drift_detected, current_acc, baseline_acc
    
    def _check_pair_model_status(self, pair: str) -> Tuple[bool, Optional[float], bool, Optional[str]]:
        """
        Check model status for a specific pair.
        
        Returns:
            Tuple of (has_pair_model, model_accuracy, needs_training, training_reason)
        """
        from pathlib import Path
        
        # Check for pair-specific model directory
        pair_model_dir = Path("trained_data/models") / pair
        has_pair_model = pair_model_dir.exists() and any(pair_model_dir.glob("*.keras"))
        
        model_accuracy = None
        needs_training = False
        training_reason = None
        
        if has_pair_model:
            # Check for model metadata
            meta_path = pair_model_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    model_accuracy = meta.get("val_accuracy", meta.get("accuracy"))
                    
                    # Check if model is old (trained >7 days ago)
                    trained_date = meta.get("trained_at", meta.get("timestamp"))
                    if trained_date:
                        from datetime import datetime
                        try:
                            trained_dt = datetime.fromisoformat(trained_date.replace("Z", "+00:00"))
                            days_old = (datetime.now(trained_dt.tzinfo) - trained_dt).days
                            if days_old > 7:
                                needs_training = True
                                training_reason = f"Model is {days_old} days old"
                        except:
                            pass
                    
                    # Check if accuracy is below threshold
                    if model_accuracy and model_accuracy < 0.55:
                        needs_training = True
                        training_reason = f"Low accuracy ({model_accuracy:.1%})"
                        
                except json.JSONDecodeError:
                    pass
        else:
            # No pair-specific model - check if generic model covers it
            generic_meta_path = Path("trained_data/models/modular_ensemble.meta.json")
            if generic_meta_path.exists():
                try:
                    meta = json.loads(generic_meta_path.read_text())
                    trained_pairs = meta.get("selected_pairs", [])
                    if pair not in trained_pairs:
                        needs_training = True
                        training_reason = "No model trained for this pair"
                except:
                    needs_training = True
                    training_reason = "No model trained for this pair"
            else:
                needs_training = True
                training_reason = "No model available"
        
        return has_pair_model, model_accuracy, needs_training, training_reason

    def _scan_pair(
        self,
        pair: str,
        granularity: str,
    ) -> Optional[EnhancedScanResult]:
        """
        Scan a single pair (atomic unit for parallel execution).
        
        Args:
            pair: Instrument name
            granularity: Timeframe
            
        Returns:
            EnhancedScanResult or None on failure
        """
        try:
            # Check model status for this pair
            has_pair_model, model_accuracy, needs_training, training_reason = self._check_pair_model_status(pair)
            
            # Fetch data
            df_raw = self._fetch_pair_data(
                pair,
                granularity,
                self._scan_config.lookback_candles + 200,  # Extra for feature warmup
            )
            
            if df_raw is None:
                return None
            
            # Compute features
            df = self._compute_features(df_raw)
            
            # Store returns for correlation analysis
            if "close" in df.columns:
                self._pair_returns[pair] = df["close"].pct_change().dropna()
            
            # Run inference (pass both raw and featured for 78% model compatibility)
            direction, confidence, tcn_conf, ridge_conf, gates_passed = self._run_inference(df_raw, df, pair)
            
            # Calculate metrics
            vol_pct, trend_str, entry_score, current_price, atr = self._calculate_metrics(df)
            
            # H1 FILTERS: Session and volatility checks
            # Filter 1: Session timing (skip outside London/NY hours)
            if self._scan_config.enable_session_filter:
                from datetime import datetime, timezone
                current_utc_hour = datetime.now(timezone.utc).hour
                if not (self._scan_config.session_start_utc <= current_utc_hour <= self._scan_config.session_end_utc):
                    logger.debug(f"{pair}: Outside active session ({current_utc_hour}:00 UTC)")
                    return None
            
            # Filter 2: Volatility check (skip dead markets)
            pip_value = PIP_VALUES.get(pair, 0.0001)
            atr_pips = atr / pip_value if pip_value > 0 else 0
            if atr_pips < self._scan_config.min_atr_pips:
                logger.debug(f"{pair}: Low volatility (ATR={atr_pips:.1f} pips < {self._scan_config.min_atr_pips})")
                return None
            
            # Calculate position sizing
            lots, risk_pct, sl_pips, tp_pips, conf_level = self._calculate_position_size(
                confidence, atr, pair
            )
            
            # Get historical accuracy
            hist_acc = self._historical_accuracy.get(pair)
            
            # Build result
            result = EnhancedScanResult(
                pair=pair,
                direction=direction,
                confidence=confidence,
                tcn_confidence=tcn_conf,
                ridge_confidence=ridge_conf,
                volatility_percentile=vol_pct,
                trend_strength=trend_str,
                entry_score=entry_score,
                current_price=current_price,
                atr=atr,
                gates_passed=gates_passed,
                recommended_lots=lots,
                risk_pct=risk_pct,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                confidence_level=conf_level,
                historical_accuracy=hist_acc,
                has_pair_model=has_pair_model,
                model_accuracy=model_accuracy,
                needs_training=needs_training,
                training_reason=training_reason,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {pair}: {e}")
            return None
    
    def scan(
        self,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
        top_n: int = 5,
        verbose: bool = True,
        prompt_train: bool = False,
    ) -> List[EnhancedScanResult]:
        """
        Scan multiple pairs in parallel.
        
        Args:
            pairs: List of pairs to scan (default: MAJOR_PAIRS)
            granularity: Timeframe
            top_n: Number of top results to return
            verbose: Print progress and results
            prompt_train: Prompt to train on best pair (deprecated, default False)
            
        Returns:
            List of EnhancedScanResult sorted by overall_score
        """
        # Initialize components (suppress all logging)
        with suppress_logging():
            self._load_config()
            self._init_memory_client()
            model_loaded = self._init_modular_ensemble()
        
        # CRITICAL: Fetch live account data for proper position sizing
        # This pulls: NAV (for position sizing), trades today, remaining trade slots
        equity_updated, nav, trades_today, trades_remaining = self._update_equity_from_oanda()
        
        # Block scan if daily limit reached
        if trades_remaining <= 0:
            if verbose and console:
                console.print(f"\n[bold red]⛔ DAILY TRADE LIMIT REACHED[/bold red]")
                console.print(f"[red]Trades today: {trades_today}/{self._scan_config.max_trades_per_day}[/red]")
                console.print("[dim]Come back tomorrow or adjust max_trades_per_day in config[/dim]")
            return []
        
        pair_list = pairs or MAJOR_PAIRS
        
        if verbose and console:
            console.print("\n" + "=" * 70)
            console.print("[bold cyan]📡 BUDDY SCANNER v2.0[/bold cyan]")
            console.print("=" * 70)
            
            # Show account info with live balance
            console.print(f"\n[bold]💰 ACCOUNT STATUS[/bold]")
            if equity_updated:
                console.print(f"   [green]✓ Live Balance:[/green] ${nav:,.2f} (fetched from OANDA)")
            else:
                console.print(f"   [dim]Balance:[/dim] ${nav:,.2f}")
            
            console.print(f"   [dim]Trades Today:[/dim] {trades_today}/{self._scan_config.max_trades_per_day}")
            if trades_remaining <= 5:
                console.print(f"   [yellow]Trades Remaining:[/yellow] {trades_remaining}")
            else:
                console.print(f"   [green]Trades Remaining:[/green] {trades_remaining}")
            console.print(f"   [dim]Risk per Trade:[/dim] {self._scan_config.risk_per_trade_pct:.1%}")
            
            # Calculate example position size for display
            example_risk_amount = nav * self._scan_config.risk_per_trade_pct
            console.print(f"   [dim]Max Risk Amount:[/dim] ${example_risk_amount:,.2f} per trade")
            console.print()
            
            if model_loaded:
                val_acc = 0
                if self._model_meta:
                    if "results" in self._model_meta:
                        val_acc = self._model_meta.get("results", {}).get("transformer", {}).get("val_accuracy", 0)
                    if val_acc == 0 and "models" in self._model_meta:
                        val_acc = self._model_meta.get("models", {}).get("direction", {}).get("metrics", {}).get("val_accuracy", 0)
                console.print(f"[green]✓ MODEL_78_v2[/green] ({val_acc:.1%} accuracy) | {len(pair_list)} pairs | {granularity}")
            else:
                console.print(f"[dim]Technical indicators | {len(pair_list)} pairs | {granularity}[/dim]")
            console.print("-" * 70)
            
            # Session hours check with user prompt
            from datetime import datetime, timezone
            current_utc_hour = datetime.now(timezone.utc).hour
            in_session = self._scan_config.session_start_utc <= current_utc_hour <= self._scan_config.session_end_utc
            
            if not in_session and self._scan_config.enable_session_filter:
                console.print()
                console.print("[bold yellow]⚠️  OUTSIDE OPTIMAL TRADING HOURS[/bold yellow]")
                console.print(f"[yellow]Current time: {current_utc_hour}:00 UTC[/yellow]")
                console.print(f"[dim]Best practice for H1: Trade during London/NY session (08:00-21:00 UTC)[/dim]")
                console.print(f"[dim]Reason: Higher liquidity = better fills, cleaner price action[/dim]")
                console.print()
                
                # Prompt user using standard input (more reliable than rich console.input)
                try:
                    import sys
                    sys.stdout.write("\033[93mContinue scanning anyway? [y/N]: \033[0m")
                    sys.stdout.flush()
                    response = sys.stdin.readline().strip().lower()
                    if response not in ('y', 'yes'):
                        console.print("[dim]Scan cancelled. Try again during London/NY hours for best results.[/dim]")
                        return []
                    console.print("[green]Proceeding with scan...[/green]")
                    console.print("-" * 70)
                    # Disable the per-pair session filter since user chose to proceed
                    self._scan_config.enable_session_filter = False
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Scan cancelled.[/dim]")
                    return []
        
        # Scan pairs in parallel (with logging suppressed)
        results: List[EnhancedScanResult] = []
        
        if verbose and console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Scanning...", total=len(pair_list))
                
                with suppress_logging():
                    with ThreadPoolExecutor(max_workers=self._scan_config.parallel_workers) as executor:
                        futures = {
                            executor.submit(self._scan_pair, pair, granularity): pair
                            for pair in pair_list
                        }
                        
                        for future in as_completed(futures):
                            pair = futures[future]
                            progress.update(task, advance=1, description=f"{pair}")
                            
                            result = future.result()
                            if result is not None:
                                results.append(result)
        else:
            # Non-verbose parallel scan
            with suppress_logging():
                with ThreadPoolExecutor(max_workers=self._scan_config.parallel_workers) as executor:
                    futures = {
                        executor.submit(self._scan_pair, pair, granularity): pair
                        for pair in pair_list
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            results.append(result)
        
        # Sort by overall score
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Run backtest on top 3 (silently)
        for i, result in enumerate(results[:3]):
            with suppress_logging():
                df_raw = self._fetch_pair_data(result.pair, granularity, self._scan_config.backtest_window + 60)
                if df_raw is not None:
                    df = self._compute_features(df_raw)
                    win_rate, sharpe, trades = self._quick_backtest(
                        df, result.direction, self._scan_config.backtest_window
                    )
                    result.backtest_win_rate = win_rate
                    result.backtest_sharpe = sharpe
                    result.backtest_trades = trades
        
        # Calculate correlations and add warnings
        corr_warnings = self._calculate_correlations(results)
        for result in results:
            if result.pair in corr_warnings:
                result.correlation_warning = corr_warnings[result.pair]
        
        # Check for model drift
        drift_detected, current_acc, baseline_acc = self._check_model_drift()
        if drift_detected:
            for result in results:
                result.drift_warning = True
        
        # Display results
        if verbose and console:
            self._display_results(results, top_n, model_loaded, drift_detected, current_acc, baseline_acc)
        
        # Store scan results in memory (silently)
        if self._memory_client is not None:
            try:
                with suppress_logging():
                    # Convert to legacy format for storage
                    analyses = [(result, result.gates_passed) for result in results]
                    self._memory_client.store_scan(analyses, granularity)
            except Exception:
                pass  # Silent - failed to store
        
        return results[:top_n]
    
    def _display_results(
        self,
        results: List[EnhancedScanResult],
        top_n: int,
        model_loaded: bool,
        drift_detected: bool,
        current_acc: float,
        baseline_acc: float,
    ):
        """Display scan results in formatted table."""
        if not console:
            return
        
        console.print("\n" + "=" * 70)
        model_label = "MODEL_78_v2" if model_loaded else "Technical"
        console.print(f"[bold green]📊 SCAN RESULTS ({model_label}) - Top {min(top_n, len(results))}[/bold green]")
        console.print(f"[dim]Account: ${self._scan_config.account_equity:,.0f} | Risk: {self._scan_config.risk_per_trade_pct:.0%}/trade | Leverage: {self._scan_config.leverage}:1[/dim]")
        console.print("=" * 70)
        
        # Separate tradeable and needs-training pairs
        tradeable_results = [r for r in results[:top_n] if r.gates_passed]
        needs_training = [r for r in results[:top_n] if r.needs_training]
        
        # === SECTION 1: TRADEABLE PAIRS ===
        if tradeable_results:
            console.print(f"\n[bold green]✅ TRADEABLE NOW ({len(tradeable_results)} pairs)[/bold green]")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=3)
            table.add_column("Pair", style="bold")
            table.add_column("Signal", justify="center")
            table.add_column("Conf", justify="right")
            table.add_column("Model", justify="center")
            table.add_column("Lots", justify="right")
            table.add_column("SL/TP", justify="right")
            table.add_column("Est. $", justify="right")
            
            for i, r in enumerate(tradeable_results, 1):
                signal_color = "green" if r.direction == "LONG" else "red"
                
                # Model status indicator
                if r.has_pair_model:
                    if r.model_accuracy and r.model_accuracy > 0.65:
                        model_display = f"[green]✓{r.model_accuracy:.0%}[/green]"
                    else:
                        model_display = f"[yellow]✓{r.model_accuracy:.0%}[/yellow]" if r.model_accuracy else "[yellow]✓[/yellow]"
                else:
                    model_display = "[dim]generic[/dim]"
                
                sl_tp = f"{r.sl_pips:.0f}/{r.tp_pips:.0f}" if r.sl_pips > 0 else "-"
                lots_display = f"{r.recommended_lots:.2f}" if r.recommended_lots > 0 else "-"
                pip_value = 10.0 if not r.pair.endswith("JPY") else 7.5
                est_return = r.recommended_lots * r.tp_pips * pip_value if r.tp_pips > 0 else 0
                est_display = f"[green]+${est_return:,.0f}[/green]" if est_return > 0 else "-"
                
                table.add_row(
                    str(i),
                    r.pair.replace("_", "/"),
                    f"[{signal_color}]{r.direction}[/{signal_color}]",
                    f"{r.confidence:.0%}",
                    model_display,
                    lots_display,
                    sl_tp,
                    est_display,
                )
            console.print(table)
        else:
            console.print(f"\n[yellow]⚠️ NO TRADEABLE PAIRS - All gates failed[/yellow]")
        
        # === SECTION 2: NON-TRADEABLE (Gates Failed) ===
        non_tradeable = [r for r in results[:top_n] if not r.gates_passed]
        if non_tradeable:
            console.print(f"\n[bold yellow]⚠️ GATES FAILED ({len(non_tradeable)} pairs)[/bold yellow]")
            table2 = Table(show_header=True, header_style="dim")
            table2.add_column("Pair", style="dim")
            table2.add_column("Signal", justify="center")
            table2.add_column("Conf", justify="right")
            table2.add_column("Model", justify="center")
            table2.add_column("Reason", justify="left")
            
            for r in non_tradeable:
                signal_color = "dim green" if r.direction == "LONG" else "dim red"
                model_display = "[green]✓[/green]" if r.has_pair_model else "[dim]generic[/dim]"
                
                # Determine why gates failed
                reason = []
                if r.confidence < 0.55:
                    reason.append(f"low conf ({r.confidence:.0%})")
                if r.tcn_confidence < 55:
                    reason.append(f"TCN weak")
                if r.ridge_confidence < 45:
                    reason.append(f"Ridge weak")
                reason_str = ", ".join(reason) if reason else "gate thresholds"
                
                table2.add_row(
                    r.pair.replace("_", "/"),
                    f"[{signal_color}]{r.direction}[/{signal_color}]",
                    f"[dim]{r.confidence:.0%}[/dim]",
                    model_display,
                    f"[dim]{reason_str}[/dim]",
                )
            console.print(table2)
        
        # === SECTION 3: NEEDS TRAINING ===
        if needs_training:
            console.print(f"\n[bold magenta]📚 NEEDS TRAINING ({len(needs_training)} pairs)[/bold magenta]")
            console.print("[dim]Run: buddy train -I <PAIR> to train these pairs[/dim]")
            table3 = Table(show_header=True, header_style="magenta")
            table3.add_column("Pair")
            table3.add_column("Reason")
            table3.add_column("Command")
            
            for r in needs_training:
                reason = r.training_reason or "No model"
                cmd = f"buddy train -I {r.pair}"
                table3.add_row(r.pair.replace("_", "/"), reason, f"[cyan]{cmd}[/cyan]")
            console.print(table3)
        
        # === SECTION 4: WARNINGS ===
        if drift_detected:
            console.print(f"\n[bold yellow]⚠️ MODEL DRIFT DETECTED[/bold yellow] (current: {current_acc:.1%}, baseline: {baseline_acc:.1%})")
            console.print("[dim]Consider retraining the model[/dim]")
        
        corr_pairs = [r.pair for r in results[:top_n] if r.correlation_warning]
        if corr_pairs:
            console.print(f"\n[yellow]⚠️ Correlation Warning:[/yellow] {', '.join(corr_pairs)}")
            console.print("[dim]Consider diversifying to reduce correlated exposure[/dim]")
        
        # === SECTION 5: SUMMARY & PROJECTIONS ===
        if results:
            longs = [r for r in results if r.direction == "LONG" and r.gates_passed]
            shorts = [r for r in results if r.direction == "SHORT" and r.gates_passed]
            tradeable = [r for r in results if r.gates_passed]
            
            if tradeable:
                console.print(f"\n[bold]🎯 QUICK SUMMARY[/bold]")
            
            if longs:
                best_long = max(longs, key=lambda x: x.overall_score)
                console.print(f"  [green]Best LONG:[/green]  {best_long.pair.replace('_', '/')} (conf: {best_long.confidence:.0%}, lots: {best_long.recommended_lots:.2f})")
            
            if shorts:
                best_short = max(shorts, key=lambda x: x.overall_score)
                console.print(f"  [red]Best SHORT:[/red] {best_short.pair.replace('_', '/')} (conf: {best_short.confidence:.0%}, lots: {best_short.recommended_lots:.2f})")
            
            # Compounding projection based on ACTUAL trading performance
            if tradeable:
                max_trades = self._scan_config.max_trades_per_day
                trades_to_use = tradeable[:max_trades]
                
                # CRITICAL: Use ACTUAL win rate from OANDA, not model accuracy
                actual_wr, actual_trades = self._fetch_actual_win_rate()
                if actual_trades >= 10:  # Only use if we have enough data
                    avg_win_rate = actual_wr
                    wr_source = f"actual ({actual_trades} trades)"
                else:
                    # Fall back to backtest or conservative estimate
                    backtest_wrs = [r.backtest_win_rate for r in trades_to_use if r.backtest_win_rate is not None]
                    if backtest_wrs:
                        avg_win_rate = sum(backtest_wrs) / len(backtest_wrs)
                        wr_source = "backtest"
                    else:
                        avg_win_rate = 0.50  # Conservative default, NOT 78%
                        wr_source = "conservative"
                
                # Calculate avg SL and TP from actual trades
                avg_tp = sum(r.tp_pips for r in trades_to_use) / len(trades_to_use) if trades_to_use else 30
                avg_sl = sum(r.sl_pips for r in trades_to_use) / len(trades_to_use) if trades_to_use else 20
                avg_lots = sum(r.recommended_lots for r in trades_to_use) / len(trades_to_use) if trades_to_use else 5.0
                rr_ratio = avg_tp / avg_sl if avg_sl > 0 else 1.5
                
                # Expected value per trade: (WR × TP) - ((1-WR) × SL)
                pip_value = 10.0  # $10 per pip per lot
                avg_win_usd = avg_lots * avg_tp * pip_value
                avg_loss_usd = avg_lots * avg_sl * pip_value
                ev_per_trade = (avg_win_rate * avg_win_usd) - ((1 - avg_win_rate) * avg_loss_usd)
                
                # Daily estimate: trades/day × EV (assume avg 3 quality setups)
                quality_trades_per_day = min(3, len(trades_to_use))  # Conservative: 3 quality trades/day
                net_daily = ev_per_trade * quality_trades_per_day
                daily_pct = (net_daily / self._scan_config.account_equity) * 100
                
                # Project to 6 months (130 trading days)
                trading_days_6mo = 130
                projected_6mo = self._scan_config.account_equity * ((1 + daily_pct/100) ** trading_days_6mo)
                
                # H1 candles in 6 months for reference
                candles_6mo = 24 * 180  # ~4320 H1 candles
                
                # Color-code win rate based on source
                if wr_source.startswith("actual"):
                    wr_color = "yellow" if avg_win_rate < 0.5 else "green"
                else:
                    wr_color = "dim"
                
                # Calculate breakeven win rate for current R:R
                breakeven_wr = 1 / (1 + rr_ratio)  # e.g., 1.5:1 R:R needs 40% WR to break even
                
                console.print(f"\n[bold cyan]📈 COMPOUNDING PROJECTION[/bold cyan] ([{wr_color}]{avg_win_rate:.0%} WR[/{wr_color}] {wr_source}, {quality_trades_per_day} trades/day)")
                console.print(f"  Avg Trade: {avg_lots:.1f} lots, SL:{avg_sl:.0f}/TP:{avg_tp:.0f} pips (R:R {rr_ratio:.1f}:1)")
                
                # Color-code EV properly (red if negative, yellow if near zero, green if positive)
                if ev_per_trade < -10:
                    ev_color = "red"
                    ev_sign = "-"
                    ev_display = abs(ev_per_trade)
                elif ev_per_trade < 50:
                    ev_color = "yellow"
                    ev_sign = "+" if ev_per_trade >= 0 else "-"
                    ev_display = abs(ev_per_trade)
                else:
                    ev_color = "green"
                    ev_sign = "+"
                    ev_display = ev_per_trade
                
                # Same for daily
                if net_daily < -10:
                    daily_color = "red"
                    daily_sign = "-"
                    daily_display = abs(net_daily)
                elif net_daily < 50:
                    daily_color = "yellow"
                    daily_sign = "+" if net_daily >= 0 else "-"
                    daily_display = abs(net_daily)
                else:
                    daily_color = "green"
                    daily_sign = "+"
                    daily_display = net_daily
                
                console.print(f"  EV/Trade:  [{ev_color}]{ev_sign}${ev_display:,.0f}[/{ev_color}] | Daily: [{daily_color}]{daily_sign}${daily_display:,.0f}[/{daily_color}] ({daily_pct:+.2f}%)")
                
                # Show breakeven analysis
                if avg_win_rate < breakeven_wr - 0.02:
                    console.print(f"  [bold red]⚠️ NEGATIVE EDGE:[/bold red] {avg_win_rate:.0%} WR < {breakeven_wr:.0%} breakeven for {rr_ratio:.1f}:1 R:R")
                elif avg_win_rate < breakeven_wr + 0.05:
                    console.print(f"  [yellow]⚠️ Near breakeven:[/yellow] {avg_win_rate:.0%} WR ≈ {breakeven_wr:.0%} breakeven for {rr_ratio:.1f}:1 R:R")
                else:
                    console.print(f"  [green]✓ Positive edge:[/green] {avg_win_rate:.0%} WR > {breakeven_wr:.0%} breakeven for {rr_ratio:.1f}:1 R:R")
                
                # 6-month projection with proper coloring
                if projected_6mo < self._scan_config.account_equity * 0.95:
                    proj_color = "red"
                elif projected_6mo < self._scan_config.account_equity * 1.2:
                    proj_color = "yellow"
                else:
                    proj_color = "green"
                    
                console.print(f"  6-Month:   [bold {proj_color}]${projected_6mo:,.0f}[/bold {proj_color}] ({trading_days_6mo} trading days, ~{candles_6mo:,} H1 candles)")
                if projected_6mo >= 1_000_000:
                    console.print(f"  [bold yellow]🎯 $1M TARGET: ACHIEVABLE[/bold yellow]")
                else:
                    needed_daily = ((1_000_000 / self._scan_config.account_equity) ** (1/trading_days_6mo) - 1) * 100
                    console.print(f"  [yellow]Need {needed_daily:.2f}%/day for $1M (current: {daily_pct:+.2f}%)[/yellow]")
                
                # Actionable recommendation when EV is zero or negative
                if ev_per_trade < 100:  # Show recommendation if EV is less than $100/trade
                    needed_wr = breakeven_wr + 0.10  # 10% above breakeven for decent edge
                    console.print(f"\n  [bold yellow]💡 RECOMMENDATION:[/bold yellow]")
                    if ev_per_trade < 0:
                        console.print(f"     [red]NEGATIVE EDGE - Consider pausing live trading[/red]")
                    elif ev_per_trade < 50:
                        console.print(f"     [yellow]MARGINAL EDGE - Profitability not guaranteed[/yellow]")
                    console.print(f"     Option 1: Improve WR to {needed_wr:.0%}+ (currently {avg_win_rate:.0%})")
                    needed_rr = (1 - avg_win_rate) / avg_win_rate  # R:R needed for current WR to break even
                    console.print(f"     Option 2: Improve R:R to {needed_rr:.1f}:1+ (currently {rr_ratio:.1f}:1)")
                    console.print(f"     Option 3: Use higher-confidence signals only (>60%)")
        
        # === SECTION 6: ACTION COMMANDS ===
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]📋 NEXT STEPS[/bold cyan]")
        
        tradeable = [r for r in results if r.gates_passed]
        needs_train = [r for r in results if r.needs_training]
        
        if tradeable:
            console.print(f"\n[green]To execute trades:[/green]")
            for r in tradeable[:3]:  # Show top 3 tradeable
                console.print(f"  buddy -I {r.pair} --execute")
        
        if needs_train:
            console.print(f"\n[magenta]To train missing models:[/magenta]")
            for r in needs_train[:3]:  # Show top 3 needing training
                console.print(f"  buddy train -I {r.pair}")
        
        console.print(f"\n[dim]View journal: buddy journal[/dim]")
        console.print("=" * 70)
    
    def _prompt_execute_trades(
        self,
        tradeable: List[EnhancedScanResult],
        granularity: str,
    ):
        """Prompt user to select trades to execute."""
        if not console or not tradeable:
            return
        
        # Build choices for questionary
        choices = []
        for r in tradeable:
            signal_color = "🟢" if r.direction == "LONG" else "🔴"
            label = f"{signal_color} {r.pair.replace('_', '/')} {r.direction} ({r.confidence:.0%}, {r.recommended_lots:.2f} lots, SL:{r.sl_pips:.0f}/TP:{r.tp_pips:.0f})"
            choices.append({"name": label, "value": r})
        
        # Add execute all and skip options
        if len(tradeable) > 1:
            choices.insert(0, {"name": f"✅ Execute ALL {len(tradeable)} trades", "value": "ALL"})
        choices.append({"name": "❌ Skip (no trades)", "value": None})
        
        console.print(f"\n[bold yellow]🎯 Execute Trades:[/bold yellow]")
        console.print("[dim]Use ↑/↓ arrows, Enter to select[/dim]\n")
        
        try:
            import questionary
            from questionary import Style
            
            custom_style = Style([
                ('qmark', 'fg:yellow bold'),
                ('question', 'bold'),
                ('answer', 'fg:green bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
            ])
            
            selected = questionary.select(
                "Trade:",
                choices=[c["name"] for c in choices],
                style=custom_style,
                instruction="",
            ).ask()
            
            if selected is None or "Skip" in selected:
                console.print("[dim]No trades executed[/dim]")
                return
            
            # Determine which trades to execute
            trades_to_execute = []
            if "ALL" in selected:
                trades_to_execute = tradeable
            else:
                for c in choices:
                    if c["name"] == selected and c["value"] not in [None, "ALL"]:
                        trades_to_execute = [c["value"]]
                        break
            
            if not trades_to_execute:
                return
            
            # Execute the trades
            self._execute_trades(trades_to_execute, granularity)
            
        except ImportError:
            console.print("[yellow]questionary not installed - run: pip install questionary[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def _execute_trades(
        self,
        trades: List[EnhancedScanResult],
        granularity: str,
    ):
        """Execute a list of trades on OANDA with daily limit enforcement."""
        if not trades:
            return
        
        # Check daily limit before execution
        trades_today = self._fetch_trades_today()
        max_trades = self._scan_config.max_trades_per_day
        trades_remaining = max(0, max_trades - trades_today)
        
        if trades_remaining <= 0:
            console.print(f"\n[bold red]⛔ DAILY TRADE LIMIT REACHED[/bold red]")
            console.print(f"[red]Trades today: {trades_today}/{max_trades}[/red]")
            console.print("[dim]Come back tomorrow or adjust max_trades_per_day in config[/dim]")
            return
        
        # Limit trades to execute to remaining daily slots
        if len(trades) > trades_remaining:
            console.print(f"[yellow]⚠ Only executing {trades_remaining} of {len(trades)} trades (daily limit)[/yellow]")
            trades = trades[:trades_remaining]
        
        console.print(f"\n[bold]Executing {len(trades)} trade(s)...[/bold]")
        console.print(f"[dim]Trades today after execution: {trades_today + len(trades)}/{max_trades}[/dim]")
        
        try:
            from oanda_practice import OandaPracticeClient
            from fx_paper import pip_size
            from datetime import datetime
            
            client = OandaPracticeClient.from_env()
            
            # Fetch live balance for proper position sizing
            _, nav, _, _ = self._update_equity_from_oanda()
            
            executed = 0
            for r in trades:
                try:
                    # Calculate SL/TP prices
                    pip = pip_size(r.pair)
                    if r.direction == "LONG":
                        sl_price = r.current_price - (r.sl_pips * pip)
                        tp_price = r.current_price + (r.tp_pips * pip)
                        units = int(r.recommended_lots * 100_000)
                    else:
                        sl_price = r.current_price + (r.sl_pips * pip)
                        tp_price = r.current_price - (r.tp_pips * pip)
                        units = -int(r.recommended_lots * 100_000)
                    
                    # Execute order
                    result = client.create_market_order(
                        instrument=r.pair,
                        units=units,
                        take_profit_price=round(tp_price, 5),
                        stop_loss_price=round(sl_price, 5),
                    )
                    
                    if result and "orderFillTransaction" in result:
                        fill = result["orderFillTransaction"]
                        fill_price = float(fill.get("price", r.current_price))
                        trade_id = fill.get("tradeOpened", {}).get("tradeID", "N/A")
                        
                        signal = "🟢" if r.direction == "LONG" else "🔴"
                        console.print(f"  {signal} [green]✓ FILLED[/green] {r.pair.replace('_', '/')} {r.direction} @ {fill_price:.5f} (#{trade_id})")
                        console.print(f"      [dim]Lots: {r.recommended_lots:.2f} | SL: {r.sl_pips:.0f} pips | TP: {r.tp_pips:.0f} pips[/dim]")
                        executed += 1
                        
                        # Log to memory
                        if self._memory_client:
                            try:
                                self._memory_client.log_trade({
                                    "timestamp": datetime.now().isoformat(),
                                    "instrument": r.pair,
                                    "direction": r.direction,
                                    "confidence": r.confidence,
                                    "lots": r.recommended_lots,
                                    "entry": fill_price,
                                    "sl": sl_price,
                                    "tp": tp_price,
                                    "trade_id": trade_id,
                                    "model": "MODEL_78_v2",
                                    "granularity": granularity,
                                    "account_balance": nav,
                                })
                            except Exception:
                                pass
                    else:
                        console.print(f"  [yellow]⚠ {r.pair}: {result}[/yellow]")
                        
                except Exception as e:
                    console.print(f"  [red]✗ {r.pair}: {e}[/red]")
            
            if executed > 0:
                console.print(f"\n[bold green]✓ {executed}/{len(trades)} trade(s) executed[/bold green]")
                console.print(f"[dim]Trades remaining today: {trades_remaining - executed}[/dim]")
                console.print("[dim]Use 'buddy journal' to track positions[/dim]")
            
        except Exception as e:
            console.print(f"[red]✗ Execution failed: {e}[/red]")

    def _prompt_training(
        self,
        results: List[EnhancedScanResult],
        granularity: str,
        drift_detected: bool,
    ):
        """Prompt user to select a pair to train on using arrow keys."""
        if not console:
            return
        
        if not results:
            return
        
        # Check for warm-start
        model_path = Path("trained_data/models/transformer_direction.keras")
        warm_label = " (warm-start)" if model_path.exists() else ""
        
        # Build choices for questionary
        choices = []
        for r in results:
            gate_icon = "✓" if r.gates_passed else "⚠"
            label = f"{r.pair.replace('_', '/')} ({r.direction}, {r.confidence:.0%}) {gate_icon}"
            choices.append({"name": label, "value": r.pair})
        choices.append({"name": "❌ Skip training", "value": None})
        
        if drift_detected:
            console.print(f"\n[yellow]⚠️ Model drift detected - retraining recommended[/yellow]")
        
        console.print(f"\n[bold yellow]🎯 Select pair to train{warm_label}:[/bold yellow]")
        console.print("[dim]Use ↑/↓ arrows, Enter to select[/dim]\n")
        
        try:
            import questionary
            from questionary import Style
            
            # Custom style for better visibility
            custom_style = Style([
                ('qmark', 'fg:yellow bold'),
                ('question', 'bold'),
                ('answer', 'fg:green bold'),
                ('pointer', 'fg:cyan bold'),
                ('highlighted', 'fg:cyan bold'),
                ('selected', 'fg:green'),
            ])
            
            selected = questionary.select(
                "Train pair:",
                choices=[c["name"] for c in choices],
                style=custom_style,
                instruction="",
            ).ask()
            
            if selected is None or "Skip" in selected:
                return
            
            # Extract pair from selection
            selected_pair = None
            for c in choices:
                if c["name"] == selected:
                    selected_pair = c["value"]
                    break
            
            if selected_pair is None:
                return
            
            console.print(f"\n[bold green]Starting training for {selected_pair}...[/bold green]")
            
            # Launch training via CLI to match exact 'buddy train' behavior
            try:
                import subprocess
                import sys
                
                # Use same CLI path as 'buddy train' for consistent behavior
                cmd = [
                    sys.executable, "main.py", "train-buddy",
                    "--config", str(self.config_path),
                    "--instrument", selected_pair,
                    "--granularity", granularity,
                    "--candles", "12000",
                    "--oanda-live",
                ]
                
                # Run training in same process for interactive output
                subprocess.run(cmd, check=True)
                
                # Log training to memory
                if self._memory_client is not None:
                    self._memory_client.log_training(selected_pair, {"granularity": granularity})
                    
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Training failed with exit code {e.returncode}[/red]")
            except Exception as e:
                console.print(f"[red]Training failed: {e}[/red]")
                    
        except ImportError:
            # Fallback to simple input if questionary not available
            console.print("[dim]Install questionary for arrow-key selection: pip install questionary[/dim]")
            for i, r in enumerate(results, 1):
                gate_icon = "✓" if r.gates_passed else "⚠"
                console.print(f"  [{i}] {r.pair.replace('_', '/')} ({r.direction}, {r.confidence:.0%}) {gate_icon}")
            console.print(f"  [0] Skip")
            
            choice = console.input("  Select [0]: ").strip() or "0"
            try:
                choice_num = int(choice)
                if choice_num > 0 and choice_num <= len(results):
                    selected = results[choice_num - 1]
                    console.print(f"\n[bold green]Starting training for {selected.pair}...[/bold green]")
            except ValueError:
                pass
        except Exception:
            # Non-interactive mode
            pass


# Convenience function for CLI usage
def buddy_scan_v2(
    config_path: str = "config_improved_H1.yaml",
    pairs: Optional[str] = None,
    granularity: str = "H1",
    top_n: int = 5,
    verbose: bool = True,
    prompt_train: bool = False,
    account_equity: float = 10000.0,
) -> List[EnhancedScanResult]:
    """
    Enhanced buddy scan with parallel processing and full features.
    
    Args:
        config_path: Path to config file
        pairs: Comma-separated pairs (e.g., "EUR_USD,GBP_USD")
        granularity: Timeframe
        top_n: Number of results
        verbose: Print output
        prompt_train: Prompt for training (deprecated)
        account_equity: Account equity for position sizing
        
    Returns:
        List of EnhancedScanResult
    """
    scanner = BuddyScanner(config_path, account_equity=account_equity)
    
    pair_list = None
    if pairs:
        pair_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]
    
    return scanner.scan(
        pairs=pair_list,
        granularity=granularity,
        top_n=top_n,
        verbose=verbose,
        prompt_train=prompt_train,
    )


if __name__ == "__main__":
    # Quick test
    results = buddy_scan_v2(top_n=3, prompt_train=False)
    print(f"\nGot {len(results)} results")
    for r in results:
        print(f"  {r.pair}: {r.direction} @ {r.confidence:.0%} (score: {r.overall_score:.2f})")
