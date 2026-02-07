"""
Scanner Execution Module.

Provides ExecutionManager for trade execution with:
- Daily trade limit enforcement
- ATR-based SL/TP calculation (H1 timeframe optimized)
- Confidence-tiered position multipliers (0.5x/1.0x/2.0x)
- Kelly-based position sizing
- Live NAV fetching for compounding
- RL position sizer integration
- Trade journal sync for retraining

ALL confidence values are 0-1 scale internally.
Position multiplier tiers:
- 0.50-0.65: 0.5x base position (conservative)
- 0.65-0.80: 1.0x base position (standard)  
- 0.80+: 2.0x base position (aggressive)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pip values for position sizing
PIP_VALUES = {
    "EUR_USD": 0.0001, "GBP_USD": 0.0001, "USD_JPY": 0.01,
    "USD_CHF": 0.0001, "AUD_USD": 0.0001, "USD_CAD": 0.0001,
    "NZD_USD": 0.0001, "EUR_GBP": 0.0001, "EUR_JPY": 0.01,
    "GBP_JPY": 0.01, "AUD_JPY": 0.01, "EUR_AUD": 0.0001,
    "GBP_AUD": 0.0001, "EUR_CHF": 0.0001, "GBP_CHF": 0.0001,
    "EUR_NZD": 0.0001, "GBP_NZD": 0.0001, "AUD_NZD": 0.0001,
    "NZD_JPY": 0.01, "CAD_JPY": 0.01, "CHF_JPY": 0.01,
    "USD_SGD": 0.0001, "EUR_CAD": 0.0001, "GBP_CAD": 0.0001,
}


@dataclass
class ExecutionConfig:
    """Configuration for trade execution (H1 timeframe optimized).
    
    ALL confidence values are 0-1 scale.
    
    Attributes:
        # Position sizing
        account_equity: Account balance (0 = fetch from OANDA)
        risk_per_trade_pct: Risk percentage per trade (0.02 = 2%)
        leverage: Account leverage (default 50:1)
        position_sizing_enabled: Enable dynamic position sizing
        aggressive_mode: Enable larger position sizes for compounding
        
        # Confidence-tiered position multipliers (0-1 scale thresholds)
        confidence_tier_low: Min confidence for 0.5x position (0.50)
        confidence_tier_medium: Min confidence for 1.0x position (0.65)
        confidence_tier_high: Min confidence for 2.0x position (0.80)
        position_multiplier_low: Multiplier for low tier (0.5)
        position_multiplier_medium: Multiplier for medium tier (1.0)
        position_multiplier_high: Multiplier for high tier (2.0)
        
        # SL/TP settings (H1 optimized)
        atr_sl_multiplier: ATR multiplier for stop loss (1.5x)
        atr_tp_multiplier: ATR multiplier for take profit (3.0x)
        min_sl_pips: Minimum stop loss in pips (15)
        max_sl_pips: Maximum stop loss in pips (80)
        min_tp_pips: Minimum take profit in pips (30)
        max_tp_pips: Maximum base take profit in pips (160)
        
        # High probability bonus
        high_prob_threshold: Confidence threshold for TP bonus (0.80)
        high_prob_tp_bonus: Extra pips added at high probability (20)
        
        # Daily limits
        max_trades_per_day: Maximum trades allowed per day (3)
        
        # Execution
        fill_timeout_seconds: Max wait for fill (10)
    """
    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.02  # 2% risk per trade (H1 standard)
    leverage: int = 50
    position_sizing_enabled: bool = True
    aggressive_mode: bool = False  # Conservative by default for H1
    
    # Confidence-tiered position multipliers (0-1 scale)
    confidence_tier_low: float = 0.50
    confidence_tier_medium: float = 0.65
    confidence_tier_high: float = 0.80
    position_multiplier_low: float = 0.5
    position_multiplier_medium: float = 1.0
    position_multiplier_high: float = 2.0
    
    # SL/TP settings (H1 optimized)
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    min_sl_pips: float = 15.0
    max_sl_pips: float = 80.0  # H1 allows wider stops
    min_tp_pips: float = 30.0
    max_tp_pips: float = 160.0  # H1 allows wider targets
    
    # High probability bonus
    high_prob_threshold: float = 0.80  # 0-1 scale
    high_prob_tp_bonus: float = 20.0
    
    # Daily limits
    max_trades_per_day: int = 3  # H1 timeframe = fewer trades
    
    # Execution
    fill_timeout_seconds: float = 10.0
    
    def get_position_multiplier(self, confidence: float) -> float:
        """Get position multiplier based on confidence tier.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Position multiplier (0.5, 1.0, or 2.0)
        """
        if confidence >= self.confidence_tier_high:
            return self.position_multiplier_high
        elif confidence >= self.confidence_tier_medium:
            return self.position_multiplier_medium
        elif confidence >= self.confidence_tier_low:
            return self.position_multiplier_low
        else:
            return 0.0  # Below minimum, no trade


@dataclass 
class ExecutionResult:
    """Result of trade execution.
    
    ALL confidence values are 0-1 scale.
    
    Attributes:
        success: Whether execution succeeded
        trade_id: OANDA trade ID if filled
        fill_price: Actual fill price
        units: Position size in units
        lots: Position size in lots
        sl_price: Stop loss price
        tp_price: Take profit price
        sl_pips: Stop loss distance in pips
        tp_pips: Take profit distance in pips
        risk_pct: Risk percentage of account
        confidence: Trade confidence (0-1)
        confidence_level: Confidence tier name
        position_multiplier: Applied position multiplier (0.5/1.0/2.0)
        error: Error message if failed
        journal_synced: Whether trade was logged to journal
    """
    success: bool
    trade_id: Optional[str] = None
    fill_price: float = 0.0
    units: int = 0
    lots: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    risk_pct: float = 0.0
    confidence: float = 0.0
    confidence_level: str = "medium"
    position_multiplier: float = 1.0
    error: Optional[str] = None
    journal_synced: bool = False
    
    def format_confidence_pct(self) -> str:
        """Format confidence as percentage for display."""
        return f"{self.confidence * 100:.0f}%"


class ExecutionManager:
    """Manages trade execution with position sizing and daily limits.
    
    Features:
    - Daily trade limit enforcement from OANDA
    - Live NAV fetching for proper compounding
    - ATR-based SL/TP calculation (H1 optimized)
    - Confidence-tiered position multipliers (0.5x/1.0x/2.0x)
    - High probability TP bonus
    - Kelly-based position sizing
    - RL position sizer integration
    - Trade journal sync for retraining
    
    ALL confidence values are 0-1 scale internally.
    """
    
    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        oanda_client: Optional[Any] = None,
    ):
        """Initialize execution manager.
        
        Args:
            config: Execution configuration
            oanda_client: OANDA API client (created if None)
        """
        self.config = config or ExecutionConfig()
        self._oanda = oanda_client
        
        # Lazy-loaded components
        self._position_sizer = None
        self._risk_manager = None
        self._rl_sizer = None
        self._memory_client = None
        self._trade_journal = None
        
        # Cached account info
        self._cached_nav: Optional[float] = None
        self._trades_today: int = 0
        self._last_nav_fetch: float = 0.0
    
    def _init_oanda_client(self) -> bool:
        """Initialize OANDA client.
        
        Returns:
            True if client initialized successfully
        """
        if self._oanda is not None:
            return True
        
        try:
            from src.utils.oanda_practice import OandaPracticeClient
            self._oanda = OandaPracticeClient.from_env()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OANDA client: {e}")
            return False
    
    def _init_position_sizer(self) -> None:
        """Initialize position sizer."""
        if self._position_sizer is not None:
            return
        
        try:
            from src.risk.position_sizing import (
                DynamicPositionSizer, PositionSizingConfig
            )
            
            if self.config.aggressive_mode:
                # Aggressive Kelly-based sizing for compounding
                config = PositionSizingConfig(
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                    min_confidence_threshold=0.53,
                    max_position_multiplier=10.0,
                    low_confidence_band=(0.53, 0.60),
                    medium_confidence_band=(0.60, 0.75),
                    high_confidence_band=(0.75, 1.0),
                    low_confidence_multiplier=1.5,
                    medium_confidence_multiplier=3.0,
                    high_confidence_multiplier=5.0,
                    max_position_pct=0.80,
                    min_position_size=500000,
                )
            else:
                config = PositionSizingConfig(
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                    min_confidence_threshold=0.5,
                )
            self._position_sizer = DynamicPositionSizer(config)
        except ImportError:
            logger.debug("DynamicPositionSizer not available")
    
    def _init_risk_manager(self) -> None:
        """Initialize risk manager."""
        if self._risk_manager is not None:
            return
        
        try:
            from src.risk.risk_management import (
                ConfidenceBasedRiskManager, RiskManagementConfig
            )
            self._risk_manager = ConfidenceBasedRiskManager(RiskManagementConfig())
        except ImportError:
            logger.debug("ConfidenceBasedRiskManager not available")
    
    def _init_memory_client(self) -> None:
        """Initialize memory client for trade logging."""
        if self._memory_client is not None:
            return
        
        try:
            from memory_client import MLEngineMemory
            self._memory_client = MLEngineMemory()
        except ImportError:
            logger.debug("MemoryClient not available")
    
    def _init_trade_journal(self) -> None:
        """Initialize trade journal for logging."""
        if self._trade_journal is not None:
            return
        
        try:
            from src.utils.trade_journal import TradeJournal
            self._trade_journal = TradeJournal()
        except ImportError:
            logger.debug("TradeJournal not available")
    
    def _wait_for_fill(
        self, 
        order_id: str, 
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Wait for order fill with timeout.
        
        Args:
            order_id: OANDA order ID
            timeout: Max wait time in seconds (uses config default if None)
            
        Returns:
            Fill transaction dict or None if timeout/error
        """
        if not self._init_oanda_client():
            return None
        
        timeout = timeout or self.config.fill_timeout_seconds
        start_time = time.time()
        poll_interval = 0.5
        
        while (time.time() - start_time) < timeout:
            try:
                # Check order status
                result = self._oanda._request(
                    'GET',
                    f'/accounts/{self._oanda._config.account_id}/orders/{order_id}'
                )
                order = result.get('order', {})
                state = order.get('state', '')
                
                if state == 'FILLED':
                    # Order filled, get fill transaction
                    tx_id = order.get('fillingTransactionID')
                    if tx_id:
                        tx_result = self._oanda._request(
                            'GET',
                            f'/accounts/{self._oanda._config.account_id}/transactions/{tx_id}'
                        )
                        return tx_result.get('transaction', {})
                    return order
                elif state in ('CANCELLED', 'REJECTED'):
                    logger.warning(f"Order {order_id} {state}")
                    return None
                    
            except Exception as e:
                logger.debug(f"Error checking order {order_id}: {e}")
            
            time.sleep(poll_interval)
        
        logger.warning(f"Order {order_id} fill timeout after {timeout}s")
        return None
    
    def fetch_live_nav(self) -> Optional[float]:
        """Fetch live NAV from OANDA for proper compounding.
        
        Returns:
            Account NAV or None if unavailable
        """
        if not self._init_oanda_client():
            return None
        
        try:
            result = self._oanda.get_account_summary()
            account = result.get('account', {})
            nav = float(account.get('NAV', 0))
            if nav > 0:
                self._cached_nav = nav
                logger.debug(f"Fetched live NAV: ${nav:,.2f}")
                return nav
        except Exception as e:
            logger.debug(f"Could not fetch live NAV: {e}")
        return self._cached_nav
    
    def fetch_trades_today(self) -> int:
        """Fetch count of trades opened today from OANDA.
        
        Returns:
            Number of trades opened today (0 if unable to fetch)
        """
        if not self._init_oanda_client():
            return 0
        
        try:
            today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            result = self._oanda._request(
                'GET',
                f'/accounts/{self._oanda._config.account_id}/trades',
                params={'state': 'ALL', 'count': 100}
            )
            trades = result.get('trades', [])
            
            trades_today = sum(
                1 for t in trades 
                if t.get('openTime', '').startswith(today_utc)
            )
            
            self._trades_today = trades_today
            return trades_today
            
        except Exception as e:
            logger.debug(f"Could not fetch trades today: {e}")
            return self._trades_today
    
    def get_account_status(self) -> Tuple[float, int, int]:
        """Get account NAV and trade limits.
        
        Returns:
            Tuple of (nav, trades_today, trades_remaining)
        """
        nav = self.fetch_live_nav() or self.config.account_equity or 10000.0
        trades_today = self.fetch_trades_today()
        trades_remaining = max(0, self.config.max_trades_per_day - trades_today)
        
        return nav, trades_today, trades_remaining
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed.
        
        Returns:
            Tuple of (can_trade, reason)
        """
        trades_today = self.fetch_trades_today()
        
        if trades_today >= self.config.max_trades_per_day:
            return False, f"Daily limit reached ({trades_today}/{self.config.max_trades_per_day})"
        
        return True, f"OK ({trades_today}/{self.config.max_trades_per_day} trades today)"
    
    def _calculate_base_tp_pips(self, atr: float, pip_value: float, confidence: float) -> float:
        """Calculate base take profit pips from ATR with high probability bonus.
        
        Args:
            atr: ATR value
            pip_value: Pip value for the pair
            confidence: Model confidence (0-1)
            
        Returns:
            Take profit in pips
        """
        if pip_value > 0 and atr > 0:
            base_tp = (atr * self.config.atr_tp_multiplier) / pip_value
        else:
            base_tp = 25.0
        
        tp_pips = max(self.config.min_tp_pips, min(base_tp, self.config.max_tp_pips))
        
        # High probability TP bonus
        if confidence >= self.config.high_prob_threshold:
            tp_pips += self.config.high_prob_tp_bonus
            logger.debug(
                f"High probability ({confidence:.1%}) - "
                f"TP bonus +{self.config.high_prob_tp_bonus} pips → {tp_pips:.1f} pips"
            )
        
        return tp_pips
    
    def _apply_risk_manager(
        self, pair: str, confidence: float, sl_pips: float, tp_pips: float
    ) -> Tuple[float, float, str]:
        """Apply risk manager adjustments to SL/TP.
        
        Args:
            pair: Instrument name
            confidence: Model confidence
            sl_pips: Base stop loss pips
            tp_pips: Base take profit pips
            
        Returns:
            Tuple of (adjusted_sl_pips, adjusted_tp_pips, confidence_level)
        """
        if self._risk_manager is None:
            return sl_pips, tp_pips, "medium"
        
        try:
            risk_result = self._risk_manager.calculate_risk_levels(
                entry_price=1.0,
                raw_confidence=confidence,
                base_stop_loss_pips=sl_pips,
                instrument=pair,
            )
            if risk_result.is_valid:
                return risk_result.stop_loss_pips, risk_result.take_profit_pips, risk_result.confidence_level
        except Exception as e:
            logger.warning(f"Risk calculation failed: {e}")
        
        return sl_pips, tp_pips, "medium"
    
    def _calculate_lots_from_sizer(
        self, equity: float, sl_pips: float, pair: str, confidence: float
    ) -> Tuple[float, float, str]:
        """Calculate lots using position sizer.
        
        Args:
            equity: Account equity
            sl_pips: Stop loss pips
            pair: Instrument name
            confidence: Model confidence
            
        Returns:
            Tuple of (lots, risk_pct, confidence_level)
        """
        if self._position_sizer is None:
            return 0.0, self.config.risk_per_trade_pct, "medium"
        
        try:
            pos_result = self._position_sizer.calculate_position_size(
                account_equity=equity,
                stop_loss_pips=sl_pips,
                instrument=pair,
                raw_confidence=confidence,
            )
            if pos_result.is_valid:
                lots = pos_result.units / 100_000
                risk_pct = pos_result.risk_amount / equity if equity > 0 else 0
                return lots, risk_pct, pos_result.confidence_level
        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")
        
        return 0.0, self.config.risk_per_trade_pct, "medium"

    def calculate_position_size(
        self,
        pair: str,
        confidence: float,
        atr: float,
        account_equity: Optional[float] = None,
    ) -> Tuple[float, float, float, float, str]:
        """Calculate position sizing based on confidence.
        
        Args:
            pair: Instrument name
            confidence: Model confidence (0-1)
            atr: ATR value for SL calculation
            account_equity: Account equity (fetches from OANDA if None)
            
        Returns:
            Tuple of (lots, risk_pct, sl_pips, tp_pips, confidence_level)
        """
        if not self.config.position_sizing_enabled:
            return 0.0, 0.0, 0.0, 0.0, "disabled"
        
        self._init_position_sizer()
        self._init_risk_manager()
        
        # Get pip value for pair
        pip_value = PIP_VALUES.get(pair, 0.0001)
        
        # Get account equity
        equity = account_equity or self.fetch_live_nav() or self.config.account_equity or 10000.0
        
        # Fixed SL (tight scalping)
        sl_pips = self.config.max_sl_pips
        
        # Calculate base TP with high probability bonus
        tp_pips = self._calculate_base_tp_pips(atr, pip_value, confidence)
        
        # Apply risk manager adjustments
        sl_pips, tp_pips, confidence_level = self._apply_risk_manager(pair, confidence, sl_pips, tp_pips)
        
        # Calculate position size
        lots, risk_pct, confidence_level = self._calculate_lots_from_sizer(
            equity, sl_pips, pair, confidence
        )
        
        return lots, risk_pct, sl_pips, tp_pips, confidence_level
    
    def execute_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        current_price: float,
        atr: float,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        lots: Optional[float] = None,
        position_multiplier: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a single trade on OANDA.
        
        Args:
            pair: Instrument name (e.g., "EUR_USD")
            direction: Trade direction ("LONG" or "SHORT")
            confidence: Model confidence (0-1)
            current_price: Current market price
            atr: ATR value for SL/TP calculation
            sl_pips: Override stop loss pips
            tp_pips: Override take profit pips
            lots: Override position size in lots
            position_multiplier: Override position multiplier (0.5/1.0/2.0)
            
        Returns:
            ExecutionResult with trade details
        """
        # Check daily limit
        can_trade, reason = self.can_trade()
        if not can_trade:
            return ExecutionResult(success=False, error=reason, confidence=confidence)
        
        # Initialize OANDA
        if not self._init_oanda_client():
            return ExecutionResult(success=False, error="OANDA client not available", confidence=confidence)
        
        # Get position multiplier from confidence tier if not provided
        if position_multiplier is None:
            position_multiplier = self.config.get_position_multiplier(confidence)
        
        # Check if confidence is below minimum tier
        if position_multiplier <= 0:
            return ExecutionResult(
                success=False, 
                error=f"Confidence {confidence:.0%} below minimum tier ({self.config.confidence_tier_low:.0%})",
                confidence=confidence,
                position_multiplier=0.0,
            )
        
        # Calculate position sizing if not provided
        if lots is None or sl_pips is None or tp_pips is None:
            calc_lots, risk_pct, calc_sl, calc_tp, conf_level = self.calculate_position_size(
                pair, confidence, atr
            )
            lots = lots or calc_lots
            sl_pips = sl_pips or calc_sl
            tp_pips = tp_pips or calc_tp
        else:
            risk_pct = self.config.risk_per_trade_pct
            conf_level = "custom"
        
        # Apply position multiplier
        lots = lots * position_multiplier
        
        if lots <= 0:
            return ExecutionResult(
                success=False, 
                error="Invalid position size",
                confidence=confidence,
                position_multiplier=position_multiplier,
            )
        
        # Determine confidence level from tier
        if confidence >= self.config.confidence_tier_high:
            conf_level = "high"
        elif confidence >= self.config.confidence_tier_medium:
            conf_level = "medium"
        else:
            conf_level = "low"
        
        # Calculate SL/TP prices
        pip_value = PIP_VALUES.get(pair, 0.0001)
        
        if direction.upper() == "LONG":
            sl_price = current_price - (sl_pips * pip_value)
            tp_price = current_price + (tp_pips * pip_value)
            units = int(lots * 100_000)
        else:
            sl_price = current_price + (sl_pips * pip_value)
            tp_price = current_price - (tp_pips * pip_value)
            units = -int(lots * 100_000)
        
        journal_synced = False
        
        try:
            result = self._oanda.create_market_order(
                instrument=pair,
                units=units,
                take_profit_price=round(tp_price, 5),
                stop_loss_price=round(sl_price, 5),
            )
            
            if result and "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                fill_price = float(fill.get("price", current_price))
                trade_id = fill.get("tradeOpened", {}).get("tradeID", "N/A")
                
                # Log to memory client
                self._log_trade(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    lots=lots,
                    entry=fill_price,
                    sl=sl_price,
                    tp=tp_price,
                    trade_id=trade_id,
                )
                
                # Sync trade journal
                journal_synced = self._sync_trade_to_journal(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    lots=lots,
                    entry=fill_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    trade_id=trade_id,
                    position_multiplier=position_multiplier,
                )
                
                logger.info(
                    f"Trade executed: {pair} {direction} @ {fill_price:.5f} | "
                    f"Conf: {confidence:.0%} ({conf_level}) | "
                    f"Mult: {position_multiplier}x | "
                    f"Lots: {lots:.2f} | SL: {sl_pips:.0f} TP: {tp_pips:.0f}"
                )
                
                return ExecutionResult(
                    success=True,
                    trade_id=trade_id,
                    fill_price=fill_price,
                    units=abs(units),
                    lots=lots,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    risk_pct=risk_pct,
                    confidence=confidence,
                    confidence_level=conf_level,
                    position_multiplier=position_multiplier,
                    journal_synced=journal_synced,
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Order rejected: {result}",
                    confidence=confidence,
                    position_multiplier=position_multiplier,
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False, 
                error=str(e),
                confidence=confidence,
                position_multiplier=position_multiplier,
            )
    
    def _sync_trade_to_journal(
        self,
        pair: str,
        direction: str,
        confidence: float,
        lots: float,
        entry: float,
        sl_price: float,
        tp_price: float,
        trade_id: str,
        position_multiplier: float,
    ) -> bool:
        """Sync trade to trade journal.
        
        Args:
            All trade details
            
        Returns:
            True if synced successfully
        """
        self._init_trade_journal()
        
        if self._trade_journal is None:
            return False
        
        try:
            self._trade_journal.add_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instrument": pair,
                "direction": direction,
                "confidence": confidence,
                "confidence_pct": f"{confidence * 100:.0f}%",
                "position_multiplier": position_multiplier,
                "lots": lots,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "trade_id": trade_id,
                "model": "scanner",
                "account_balance": self._cached_nav or self.config.account_equity,
            })
            return True
        except Exception as e:
            logger.debug(f"Failed to sync trade to journal: {e}")
            return False
    
    def execute_trades(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Execute multiple trades with daily limit enforcement.
        
        Args:
            trades: List of trade dicts with keys:
                - pair: Instrument name
                - direction: "LONG" or "SHORT"
                - confidence: Model confidence
                - current_price: Current price
                - atr: ATR value
                - sl_pips (optional): Stop loss pips
                - tp_pips (optional): Take profit pips
                - recommended_lots (optional): Position size
            
        Returns:
            List of ExecutionResult for each trade
        """
        if not trades:
            return []
        
        # Check daily limit
        _, trades_today, trades_remaining = self.get_account_status()
        
        if trades_remaining <= 0:
            logger.warning(f"Daily trade limit reached ({trades_today}/{self.config.max_trades_per_day})")
            return [ExecutionResult(success=False, error="Daily limit reached") for _ in trades]
        
        # Limit trades to remaining slots
        if len(trades) > trades_remaining:
            logger.warning(f"Executing {trades_remaining} of {len(trades)} trades (daily limit)")
            trades = trades[:trades_remaining]
        
        results = []
        for trade in trades:
            result = self.execute_trade(
                pair=trade.get("pair", ""),
                direction=trade.get("direction", "HOLD"),
                confidence=trade.get("confidence", 0.5),
                current_price=trade.get("current_price", 0.0),
                atr=trade.get("atr", 0.0),
                sl_pips=trade.get("sl_pips"),
                tp_pips=trade.get("tp_pips"),
                lots=trade.get("recommended_lots"),
            )
            results.append(result)
        
        return results
    
    def _log_trade(
        self,
        pair: str,
        direction: str,
        confidence: float,
        lots: float,
        entry: float,
        sl: float,
        tp: float,
        trade_id: str,
    ) -> None:
        """Log trade to memory client."""
        self._init_memory_client()
        
        if self._memory_client is None:
            return
        
        try:
            nav = self._cached_nav or self.config.account_equity or 10000.0
            self._memory_client.log_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "instrument": pair,
                "direction": direction,
                "confidence": confidence,
                "lots": lots,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "trade_id": trade_id,
                "model": "scanner",
                "account_balance": nav,
            })
        except Exception as e:
            logger.debug(f"Failed to log trade: {e}")
    
    def fetch_actual_win_rate(self) -> Tuple[float, int]:
        """Fetch actual win rate from OANDA closed trades.
        
        Returns:
            Tuple of (win_rate, total_trades)
        """
        if not self._init_oanda_client():
            return 0.0, 0
        
        try:
            result = self._oanda._request(
                'GET',
                f'/accounts/{self._oanda._config.account_id}/trades',
                params={'state': 'CLOSED', 'count': 100}
            )
            trades = result.get('trades', [])
            
            if not trades:
                return 0.0, 0
            
            wins = sum(1 for t in trades if float(t.get('realizedPL', 0)) > 0)
            total = len(trades)
            win_rate = wins / total if total > 0 else 0.0
            
            logger.debug(f"Actual trading performance: {win_rate:.1%} ({wins}W/{total-wins}L)")
            return win_rate, total
            
        except Exception as e:
            logger.debug(f"Could not fetch actual win rate: {e}")
            return 0.0, 0
    
    def sync_journal(self) -> int:
        """Sync trade journal with OANDA and check for retraining.
        
        Returns:
            Number of trades synced
        """
        if not self._init_oanda_client():
            return 0
        
        try:
            from src.utils.trade_journal import TradeJournal
            
            journal = TradeJournal()
            updated = journal.update_from_oanda(self._oanda)
            
            if updated > 0:
                logger.info(f"Journal synced: {updated} trade(s) updated")
            
            return updated
            
        except ImportError:
            logger.debug("TradeJournal not available")
            return 0
        except Exception as e:
            logger.debug(f"Journal sync failed: {e}")
            return 0
