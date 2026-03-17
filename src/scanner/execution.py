"""
Scanner Execution Module.

Provides ExecutionManager for trade execution with:
- Daily trade limit enforcement
- ATR-based SL/TP calculation
- High probability TP bonus
- Kelly-based position sizing
- Live NAV fetching for compounding
- RL position sizer integration
"""

from __future__ import annotations

import logging
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
    """Configuration for trade execution.

    Attributes:
        # Position sizing
        account_equity: Account balance (0 = fetch from OANDA)
        risk_per_trade_pct: Risk percentage per trade (0.05 = 5%)
        leverage: Account leverage (default 50:1)
        position_sizing_enabled: Enable dynamic position sizing
        aggressive_mode: Enable larger position sizes for compounding

        # SL/TP settings
        atr_sl_multiplier: ATR multiplier for stop loss (1.0 = 1x ATR)
        atr_tp_multiplier: ATR multiplier for take profit (1.5 = 1.5x ATR)
        min_sl_pips: Minimum stop loss in pips
        max_sl_pips: Maximum stop loss in pips (fixed for tight scalping)
        min_tp_pips: Minimum take profit in pips
        max_tp_pips: Maximum base take profit in pips

        # High probability bonus
        high_prob_threshold: Confidence threshold for TP bonus (0.65 = 65%)
        high_prob_tp_bonus: Extra pips added at high probability

        # Daily limits
        max_trades_per_day: Maximum trades allowed per day
    """
    # Position sizing
    account_equity: float = 0.0  # 0 = fetch from OANDA
    risk_per_trade_pct: float = 0.05  # 5% risk per trade
    leverage: int = 50
    position_sizing_enabled: bool = True
    aggressive_mode: bool = True
    regime_scaling_enabled: bool = True
    aggressive_scale_high_vol: float = 1.5
    aggressive_scale_extreme_vol: float = 1.75
    aggressive_min_meta_confidence: float = 0.52

    # SL/TP settings (tight scalping)
    atr_sl_multiplier: float = 1.0
    atr_tp_multiplier: float = 1.5
    min_sl_pips: float = 15.0
    max_sl_pips: float = 15.0  # Fixed for tight scalping
    min_tp_pips: float = 20.0
    max_tp_pips: float = 30.0

    # High probability bonus
    high_prob_threshold: float = 0.65
    high_prob_tp_bonus: float = 20.0

    # Daily limits
    max_trades_per_day: int = 30

    # Portfolio risk limits
    max_open_risk_pct: float = 0.15  # Max 15% of NAV at risk across all open trades


@dataclass
class ExecutionResult:
    """Result of trade execution.

    Attributes:
        success: Whether execution succeeded
        trade_id: OANDA trade ID if filled
        fill_price: Actual fill price
        units: Position size in units
        lots: Position size in lots
        sl_price: Stop loss price
        tp_price: Take profit price
        error: Error message if failed
        regime_scale: Regime-based scaling factor applied
        regime_name: Volatility regime name
        aggressive_scaling_reason: Reason for aggressive scaling if applied
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
    confidence_level: str = "medium"
    error: Optional[str] = None
    # Regime-based scaling fields
    regime_scale: float = 1.0
    regime_name: str = "UNKNOWN"
    aggressive_scaling_reason: str = ""


class ExecutionManager:
    """Manages trade execution with position sizing and daily limits.

    Features:
    - Daily trade limit enforcement from OANDA
    - Live NAV fetching for proper compounding
    - ATR-based SL/TP calculation
    - High probability TP bonus
    - Kelly-based position sizing
    - RL position sizer integration
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
        """Initialize position sizer with regime-aware scaling."""
        if self._position_sizer is not None:
            return

        try:
            from src.risk.position_sizing import (
                DynamicPositionSizer, PositionSizingConfig,
                create_regime_aware_position_sizer
            )

            if self.config.aggressive_mode:
                # Use regime-aware position sizer with aggressive scaling
                # This is the "highest-ROI toggle in the entire system"
                self._position_sizer = create_regime_aware_position_sizer(
                    aggressive_scale_high_vol=float(self.config.aggressive_scale_high_vol),
                    aggressive_scale_extreme_vol=float(self.config.aggressive_scale_extreme_vol),
                    aggressive_min_meta_confidence=float(self.config.aggressive_min_meta_confidence),
                    regime_scaling_enabled=bool(self.config.regime_scaling_enabled),
                )
                logger.info("✓ Regime-aware position sizer initialized (aggressive mode)")
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

    def _check_portfolio_risk_limit(self) -> Tuple[bool, str]:
        """Check if total open risk across all positions is within limit.

        Uses SL distance * position size to estimate risk per trade.

        Returns:
            Tuple of (within_limit, reason)
        """
        max_risk_pct = self.config.max_open_risk_pct
        if max_risk_pct <= 0:
            return True, "risk limit disabled"

        try:
            statuses = self.monitor_open_trades()
            if not statuses:
                return True, "no open trades"

            nav = self.fetch_live_nav() or self.config.account_equity or 100000.0

            total_risk = 0.0
            for s in statuses:
                sl_dist = s.get("sl_dist_pips", 0)
                units = abs(s.get("units", 0))
                pip_val = PIP_VALUES.get(s.get("pair", ""), 0.0001)
                # Risk = SL distance in price * units
                risk_amount = sl_dist * pip_val * units
                total_risk += risk_amount

            risk_pct = total_risk / nav if nav > 0 else 0.0

            if risk_pct >= max_risk_pct:
                return False, (
                    f"Portfolio risk limit ({risk_pct:.1%} >= {max_risk_pct:.0%}): "
                    f"${total_risk:,.0f} at risk on ${nav:,.0f} NAV"
                )
            return True, f"risk {risk_pct:.1%} of {max_risk_pct:.0%} limit"

        except Exception as e:
            logger.debug(f"Risk limit check failed: {e}")
            return True, "risk check unavailable"

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
        def _fallback_lots() -> Tuple[float, float, str]:
            risk_pct = float(self.config.risk_per_trade_pct)
            risk_amount = max(0.0, float(equity) * risk_pct)
            pip_value_usd = 7.5 if str(pair).upper().endswith("JPY") else 10.0
            denom = max(float(sl_pips), 1.0) * pip_value_usd
            lots = (risk_amount / denom) if denom > 0 else 0.0
            lots = round(min(max(lots, 0.01), 50.0), 2)
            return lots, risk_pct, "fallback"

        if self._position_sizer is None:
            return _fallback_lots()

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
                if lots > 0:
                    return lots, risk_pct, pos_result.confidence_level
        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")

        return _fallback_lots()
    
    def calculate_regime_aware_position_size(
        self,
        pair: str,
        confidence: float,
        atr: float,
        volatility_regime: int,
        meta_confidence: float,
        account_equity: Optional[float] = None,
    ) -> Tuple[float, float, float, float, str, float, str]:
        """Calculate position sizing with regime-based aggressive scaling.
        
        This is the key method for enabling aggressive volatility-regime sizing.
        When regime == HIGH/EXTREME and meta_confidence > threshold, positions
        are scaled UP to capitalize on high-quality signals.
        
        Args:
            pair: Instrument name
            confidence: Model confidence (0-1)
            atr: ATR value for SL calculation
            volatility_regime: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
            meta_confidence: Meta-labeler confidence (0-1)
            account_equity: Account equity (fetches from OANDA if None)
            
        Returns:
            Tuple of (lots, risk_pct, sl_pips, tp_pips, confidence_level, regime_scale, regime_name)
        """
        if not self.config.position_sizing_enabled:
            return 0.0, 0.0, 0.0, 0.0, "disabled", 1.0, "UNKNOWN"
        
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
        sl_pips, tp_pips, confidence_level = self._apply_risk_manager(
            pair, confidence, sl_pips, tp_pips
        )
        
        # Calculate position size with regime-based scaling
        if hasattr(self._position_sizer, 'calculate_regime_scaled_position_size'):
            try:
                pos_result = self._position_sizer.calculate_regime_scaled_position_size(
                    account_equity=equity,
                    stop_loss_pips=sl_pips,
                    instrument=pair,
                    volatility_regime=volatility_regime,
                    meta_confidence=meta_confidence,
                    raw_confidence=confidence,
                )
                if pos_result.is_valid:
                    lots = pos_result.units / 100_000
                    risk_pct = pos_result.risk_amount / equity if equity > 0 else 0
                    return (
                        lots,
                        risk_pct,
                        sl_pips,
                        tp_pips,
                        pos_result.confidence_level,
                        pos_result.regime_scale_applied,
                        pos_result.regime_name,
                    )
            except Exception as e:
                logger.warning(f"Regime-aware position sizing failed: {e}")
        
        # Fallback to standard calculation
        lots, risk_pct, confidence_level = self._calculate_lots_from_sizer(
            equity, sl_pips, pair, confidence
        )
        
        regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
        regime_name = regime_names[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"
        
        return lots, risk_pct, sl_pips, tp_pips, confidence_level, 1.0, regime_name

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
        volatility_regime: Optional[int] = None,
        meta_confidence: Optional[float] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
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
            volatility_regime: Volatility regime for aggressive scaling (0-3)
            meta_confidence: Meta-labeler confidence for aggressive scaling

        Returns:
            ExecutionResult with trade details
        """
        # Check daily limit
        can_trade, reason = self.can_trade()
        if not can_trade:
            return ExecutionResult(success=False, error=reason)

        # Check portfolio risk limit
        risk_ok, risk_reason = self._check_portfolio_risk_limit()
        if not risk_ok:
            return ExecutionResult(success=False, error=risk_reason)

        # Initialize OANDA
        if not self._init_oanda_client():
            return ExecutionResult(success=False, error="OANDA client not available")

        # Initialize regime scaling info
        regime_scale = 1.0
        regime_name = "UNKNOWN"
        aggressive_reason = ""

        # Calculate position sizing if not provided
        if (
            lots is None or float(lots) <= 0.0 or
            sl_pips is None or float(sl_pips) <= 0.0 or
            tp_pips is None or float(tp_pips) <= 0.0
        ):
            # When scan already provides SL/TP but not lots, use the scan SL
            # for sizing (not the internal max_sl_pips) so risk is accurate.
            scan_sl = float(sl_pips) if sl_pips and float(sl_pips) > 0 else None

            # Use regime-aware sizing if regime info provided
            if volatility_regime is not None and meta_confidence is not None:
                (
                    calc_lots,
                    risk_pct,
                    calc_sl,
                    calc_tp,
                    conf_level,
                    regime_scale,
                    regime_name,
                ) = self.calculate_regime_aware_position_size(
                    pair=pair,
                    confidence=confidence,
                    atr=atr,
                    volatility_regime=volatility_regime,
                    meta_confidence=meta_confidence,
                )
                # Re-calculate lots using actual SL if scan provided one
                if scan_sl and lots is None and calc_lots > 0:
                    equity = self.fetch_live_nav() or self.config.account_equity or 10000.0
                    calc_lots, risk_pct, conf_level = self._calculate_lots_from_sizer(
                        equity, scan_sl, pair, confidence
                    )
            else:
                calc_lots, risk_pct, calc_sl, calc_tp, conf_level = self.calculate_position_size(
                    pair, confidence, atr
                )
                # Re-calculate lots using actual SL if scan provided one
                if scan_sl and lots is None:
                    equity = self.fetch_live_nav() or self.config.account_equity or 10000.0
                    calc_lots, risk_pct, conf_level = self._calculate_lots_from_sizer(
                        equity, scan_sl, pair, confidence
                    )
            lots = lots or calc_lots
            sl_pips = sl_pips or calc_sl
            tp_pips = tp_pips or calc_tp
        else:
            risk_pct = self.config.risk_per_trade_pct
            conf_level = "custom"

        if lots <= 0:
            return ExecutionResult(success=False, error="Invalid position size")

        # Minimum risk:reward ratio gate
        min_rr = getattr(self.config, "min_risk_reward_ratio", 1.0)
        if sl_pips > 0 and tp_pips > 0:
            rr_ratio = tp_pips / sl_pips
            if rr_ratio < min_rr:
                return ExecutionResult(
                    success=False,
                    error=f"R:R {rr_ratio:.2f}:1 below minimum {min_rr}:1 (SL {sl_pips:.1f}p / TP {tp_pips:.1f}p)",
                )

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

                # Log to memory client with full analysis context
                self._log_trade(
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    lots=lots,
                    entry=fill_price,
                    sl=sl_price,
                    tp=tp_price,
                    trade_id=trade_id,
                    analysis_context=analysis_context,
                )
                
                # Log aggressive scaling if applied
                if regime_scale > 1.0:
                    logger.info(
                        f"🚀 Trade executed with AGGRESSIVE SCALING: {regime_name} vol, "
                        f"{regime_scale:.2f}x scale, {lots:.2f} lots"
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
                    confidence_level=conf_level,
                    regime_scale=regime_scale,
                    regime_name=regime_name,
                    aggressive_scaling_reason=aggressive_reason,
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Order rejected: {result}",
                )

        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

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
            # Extract volatility regime and meta-confidence from analysis context
            ctx = trade.get("analysis_context") or {}
            vol_regime = None
            meta_conf = None
            regime_str = str(ctx.get("volatility_regime", "") or "").upper()
            regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
            if regime_str in regime_map:
                vol_regime = regime_map[regime_str]
            meta_conf = ctx.get("tcn_confidence") or ctx.get("weighted_vote_score")

            result = self.execute_trade(
                pair=trade.get("pair", ""),
                direction=trade.get("direction", "HOLD"),
                confidence=trade.get("confidence", 0.5),
                current_price=trade.get("current_price", 0.0),
                atr=trade.get("atr", 0.0),
                sl_pips=trade.get("sl_pips"),
                tp_pips=trade.get("tp_pips"),
                lots=trade.get("recommended_lots"),
                volatility_regime=vol_regime,
                meta_confidence=float(meta_conf) if meta_conf is not None else None,
                analysis_context=trade.get("analysis_context"),
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
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log trade to memory client with full gate/agent context."""
        self._init_memory_client()

        if self._memory_client is None:
            return

        try:
            nav = self._cached_nav or self.config.account_equity or 10000.0
            record: Dict[str, Any] = {
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
            }
            if analysis_context:
                record["metadata"] = analysis_context
            self._memory_client.log_trade(record)
        except Exception as e:
            logger.debug(f"Failed to log trade: {e}")

        # Also append to trade_journal.json for RL feedback
        self._append_journal_entry(trade_id, pair, direction, confidence, entry, sl, tp, lots, analysis_context)

    def _append_journal_entry(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        lots: float,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append trade entry to trained_data/trade_journal.json for RL feedback."""
        import json
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        journal_path.parent.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        if journal_path.exists():
            try:
                entries = json.loads(journal_path.read_text())
            except Exception:
                entries = []

        ctx = analysis_context or {}
        entry_record = {
            "trade_id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "lots": lots,
            "gates": {
                "momentum_passed": ctx.get("momentum_passed"),
                "confidence_passed": ctx.get("confidence_passed"),
                "risk_passed": ctx.get("risk_passed"),
                "gate_summary": ctx.get("gate_summary"),
            },
            "agents": {
                "agent_votes": ctx.get("agent_votes"),
                "agent_total": ctx.get("agent_total"),
                "agent_passed": ctx.get("agent_passed"),
                "weighted_vote_score": ctx.get("weighted_vote_score"),
                "agent_reasons": ctx.get("agent_reasons", []),
            },
            "regime": {
                "volatility_regime": ctx.get("volatility_regime"),
                "atr_pips": ctx.get("atr_pips"),
                "uncertainty_score": ctx.get("uncertainty_score"),
                "model_disagreement": ctx.get("model_disagreement"),
            },
            "outcome": None,  # Filled by sync_closed_trades_rl
        }

        # Deduplicate by trade_id
        entries = [e for e in entries if e.get("trade_id") != trade_id]
        entries.append(entry_record)

        journal_path.write_text(json.dumps(entries, indent=2, default=str))
        logger.debug(f"Journal entry appended for trade #{trade_id}")

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

    def monitor_open_trades(self) -> List[Dict[str, Any]]:
        """Monitor open trades and return status for each.

        Returns:
            List of dicts with trade status: pair, direction, entry, current_pl,
            distance_to_sl_pips, distance_to_tp_pips, time_in_trade_minutes.
        """
        if not self._init_oanda_client():
            return []

        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = requests.get(f"{base}/v3/accounts/{acct}/openTrades", headers=headers, timeout=10)
            trades = resp.json().get("trades", [])
        except Exception as e:
            logger.warning(f"Failed to fetch open trades: {e}")
            return []

        statuses = []
        for t in trades:
            pair = t.get("instrument", "")
            pip_value = PIP_VALUES.get(pair, 0.0001)
            units = int(t.get("currentUnits", 0))
            direction = "LONG" if units > 0 else "SHORT"
            entry = float(t.get("price", 0))
            unrealized_pl = float(t.get("unrealizedPL", 0))

            sl_price = float(t.get("stopLossOrder", {}).get("price", 0))
            tp_price = float(t.get("takeProfitOrder", {}).get("price", 0))

            # Calculate distances in pips
            current_price = entry  # Approximate; exact would need live price
            if sl_price > 0:
                sl_dist_pips = abs(entry - sl_price) / pip_value
            else:
                sl_dist_pips = 0
            if tp_price > 0:
                tp_dist_pips = abs(tp_price - entry) / pip_value
            else:
                tp_dist_pips = 0

            # Time in trade
            open_time = t.get("openTime", "")
            time_in_minutes = 0
            if open_time:
                try:
                    open_dt = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                    time_in_minutes = int((datetime.now(timezone.utc) - open_dt).total_seconds() / 60)
                except Exception:
                    pass

            statuses.append({
                "trade_id": t.get("id"),
                "pair": pair,
                "direction": direction,
                "units": units,
                "entry": entry,
                "unrealized_pl": unrealized_pl,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_dist_pips": round(sl_dist_pips, 1),
                "tp_dist_pips": round(tp_dist_pips, 1),
                "time_in_minutes": time_in_minutes,
            })

        return statuses

    def apply_drawdown_guardian(self) -> List[str]:
        """Check open trades and tighten SL when trade moves in favor.

        - At 50% of TP distance: move SL to breakeven
        - At 75% of TP distance: lock in 50% of unrealized profit

        Returns:
            List of modification messages.
        """
        import requests
        import os

        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        statuses = self.monitor_open_trades()
        modifications = []

        for status in statuses:
            trade_id = status["trade_id"]
            pair = status["pair"]
            direction = status["direction"]
            entry = status["entry"]
            sl_price = status["sl_price"]
            tp_price = status["tp_price"]
            pip_value = PIP_VALUES.get(pair, 0.0001)

            if tp_price <= 0 or sl_price <= 0:
                continue

            # Get current price
            try:
                resp = requests.get(
                    f"{base}/v3/accounts/{acct}/pricing?instruments={pair}",
                    headers=headers,
                    timeout=10,
                )
                prices = resp.json().get("prices", [])
                if not prices:
                    continue
                mid = (float(prices[0].get("asks", [{}])[0].get("price", 0)) +
                       float(prices[0].get("bids", [{}])[0].get("price", 0))) / 2
            except Exception:
                continue

            # Calculate progress toward TP
            if direction == "LONG":
                tp_distance = tp_price - entry
                current_progress = mid - entry
            else:
                tp_distance = entry - tp_price
                current_progress = entry - mid

            if tp_distance <= 0:
                continue

            progress_pct = current_progress / tp_distance

            new_sl = None
            reason = ""

            if progress_pct >= 0.75:
                # Lock in 50% of unrealized profit
                if direction == "LONG":
                    new_sl = entry + (current_progress * 0.50)
                    if new_sl > sl_price:
                        reason = f"75% TP reached, locking 50% profit (SL {sl_price:.5f} -> {new_sl:.5f})"
                    else:
                        new_sl = None
                else:
                    new_sl = entry - (current_progress * 0.50)
                    if new_sl < sl_price:
                        reason = f"75% TP reached, locking 50% profit (SL {sl_price:.5f} -> {new_sl:.5f})"
                    else:
                        new_sl = None
            elif progress_pct >= 0.50:
                # Move SL to breakeven
                if direction == "LONG" and sl_price < entry:
                    new_sl = entry + (pip_value * 2)  # Breakeven + 2 pips
                    reason = f"50% TP reached, SL to breakeven (SL {sl_price:.5f} -> {new_sl:.5f})"
                elif direction == "SHORT" and sl_price > entry:
                    new_sl = entry - (pip_value * 2)
                    reason = f"50% TP reached, SL to breakeven (SL {sl_price:.5f} -> {new_sl:.5f})"

            if new_sl is not None:
                try:
                    # Modify the trade's SL via OANDA API
                    modify_data = {
                        "stopLoss": {"price": f"{new_sl:.5f}"}
                    }
                    resp = requests.put(
                        f"{base}/v3/accounts/{acct}/trades/{trade_id}/orders",
                        headers=headers,
                        json=modify_data,
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        modifications.append(f"{pair} #{trade_id}: {reason}")
                        logger.info(f"Drawdown guardian: {pair} #{trade_id}: {reason}")
                    else:
                        logger.warning(f"SL modification failed for {pair} #{trade_id}: {resp.text}")
                except Exception as e:
                    logger.warning(f"SL modification error for {pair} #{trade_id}: {e}")

        return modifications

    def sync_closed_trades_rl(self) -> Dict[str, Any]:
        """Check OANDA for recently closed trades and run RL feedback on agent weights.

        Reads the RL journal, matches against OANDA closed trades, updates outcomes,
        and calls ScannerAgentTeam.update_weights_from_outcome for each closed trade.

        Returns:
            Dict with sync results: trades_synced, weights_updated, new_weights.
        """
        import json
        import requests
        import os
        from pathlib import Path

        journal_path = Path("trained_data/trade_journal_rl.json")
        if not journal_path.exists():
            return {"trades_synced": 0, "weights_updated": False, "detail": "no journal"}

        try:
            entries = json.loads(journal_path.read_text())
        except Exception:
            return {"trades_synced": 0, "weights_updated": False, "detail": "journal parse error"}

        # Find entries without outcomes
        pending = [e for e in entries if e.get("outcome") is None]
        if not pending:
            return {"trades_synced": 0, "weights_updated": False, "detail": "no pending trades"}

        # Fetch closed trades from OANDA
        token = os.getenv("OANDA_API_TOKEN", "")
        acct = os.getenv("OANDA_ACCOUNT_ID", "")
        base = "https://api-fxpractice.oanda.com"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = requests.get(
                f"{base}/v3/accounts/{acct}/trades",
                params={"state": "CLOSED", "count": 50},
                headers=headers,
                timeout=10,
            )
            closed_trades = {t["id"]: t for t in resp.json().get("trades", [])}
        except Exception as e:
            return {"trades_synced": 0, "weights_updated": False, "detail": f"oanda error: {e}"}

        synced = 0
        rl_updates = []

        for entry in pending:
            tid = str(entry.get("trade_id", ""))
            if tid not in closed_trades:
                continue

            ct = closed_trades[tid]
            realized_pl = float(ct.get("realizedPL", 0))
            trade_won = realized_pl > 0
            pip_value = PIP_VALUES.get(entry.get("pair", ""), 0.0001)
            pnl_pips = 0.0
            close_price = float(ct.get("averageClosePrice", 0))
            entry_price = float(ct.get("price", entry.get("entry_price", 0)))
            if pip_value > 0 and entry_price > 0:
                pnl_pips = (close_price - entry_price) / pip_value
                if entry.get("direction", "").upper() == "SHORT":
                    pnl_pips = -pnl_pips

            entry["outcome"] = {
                "realized_pl": realized_pl,
                "pnl_pips": round(pnl_pips, 1),
                "trade_won": trade_won,
                "close_price": close_price,
                "close_time": ct.get("closeTime"),
            }
            synced += 1

            # Collect agent verdicts for RL feedback
            agents = entry.get("agents", {})
            if agents.get("agent_reasons"):
                rl_updates.append({
                    "agent_verdicts": agents["agent_reasons"],
                    "trade_won": trade_won,
                })

        # Write updated journal
        journal_path.write_text(json.dumps(entries, indent=2, default=str))

        # Run RL agent weight updates
        weights_updated = False
        new_weights: Dict[str, float] = {}
        if rl_updates:
            try:
                from src.scanner.agents import ScannerAgentTeam
                agent_team = ScannerAgentTeam(config=None)
                for upd in rl_updates:
                    new_weights = agent_team.update_weights_from_outcome(
                        agent_verdicts=upd["agent_verdicts"],
                        trade_won=upd["trade_won"],
                    )
                weights_updated = True
                logger.info(f"RL feedback: updated weights from {len(rl_updates)} closed trades")
            except Exception as e:
                logger.warning(f"RL weight update failed: {e}")

        # Update per-pair performance summary after syncing
        if synced > 0:
            try:
                self._update_pair_performance()
            except Exception as perf_err:
                logger.debug(f"Pair performance update error: {perf_err}")

        return {
            "trades_synced": synced,
            "weights_updated": weights_updated,
            "new_weights": new_weights,
            "detail": f"synced {synced} trades, {len(rl_updates)} RL updates",
        }

    def _update_pair_performance(self) -> None:
        """Aggregate per-pair performance stats from the trade journal."""
        import json
        from pathlib import Path
        from collections import defaultdict

        journal_path = Path("trained_data/trade_journal_rl.json")
        perf_path = Path("trained_data/models/pair_performance.json")

        if not journal_path.exists():
            return

        entries = json.loads(journal_path.read_text())
        closed = [e for e in entries if e.get("outcome") is not None]
        if not closed:
            return

        stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
            "total_pnl_pips": 0.0, "avg_confidence": 0.0,
            "best_pnl_pips": 0.0, "worst_pnl_pips": 0.0,
        })

        for e in closed:
            pair = e.get("pair", "UNKNOWN")
            outcome = e["outcome"]
            s = stats[pair]
            s["trades"] += 1
            pnl_pips = float(outcome.get("pnl_pips", 0))
            realized_pl = float(outcome.get("realized_pl", 0))
            won = bool(outcome.get("trade_won", False))

            if won:
                s["wins"] += 1
            else:
                s["losses"] += 1

            s["total_pnl"] += realized_pl
            s["total_pnl_pips"] += pnl_pips
            s["avg_confidence"] += float(e.get("confidence", 0))
            s["best_pnl_pips"] = max(s["best_pnl_pips"], pnl_pips)
            s["worst_pnl_pips"] = min(s["worst_pnl_pips"], pnl_pips)

        # Finalize averages
        for pair, s in stats.items():
            if s["trades"] > 0:
                s["avg_confidence"] = round(s["avg_confidence"] / s["trades"], 3)
                s["win_rate"] = round(s["wins"] / s["trades"], 3)
            else:
                s["win_rate"] = 0.0
            s["total_pnl"] = round(s["total_pnl"], 2)
            s["total_pnl_pips"] = round(s["total_pnl_pips"], 1)
            s["best_pnl_pips"] = round(s["best_pnl_pips"], 1)
            s["worst_pnl_pips"] = round(s["worst_pnl_pips"], 1)

        perf_path.parent.mkdir(parents=True, exist_ok=True)
        perf_path.write_text(json.dumps(dict(stats), indent=2))

    def get_pair_performance(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """Read per-pair performance stats.

        Args:
            pair: Specific pair to query, or None for all pairs.

        Returns:
            Dict with pair -> stats mapping, or single pair stats.
        """
        import json
        from pathlib import Path

        perf_path = Path("trained_data/models/pair_performance.json")
        if not perf_path.exists():
            return {}

        try:
            data = json.loads(perf_path.read_text())
        except Exception:
            return {}

        if pair:
            return data.get(pair, {})
        return data
