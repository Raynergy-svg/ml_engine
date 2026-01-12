"""Trade Journal - Track and analyze trade performance.

Logs all trades placed by Buddy and fetches results from OANDA to calculate:
- Win rate
- Average win/loss
- Profit factor
- Total P/L
- Slippage analysis
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("trained_data") / "trade_journal.json"


@dataclass
class TradeEntry:
    """Record of a single trade."""
    # Identifiers
    trade_id: str
    timestamp: str
    instrument: str
    
    # Entry details
    direction: str  # 'long' or 'short'
    entry_price: float
    expected_price: float
    units: int
    lots: float
    
    # Risk management
    stop_loss: float
    take_profit: float
    trailing_stop: float
    atr: float
    
    # Signal info
    tcn_probability: float
    ridge_confidence: float
    xgb_momentum: float
    rf_drawdown_pips: float
    
    # Execution
    slippage_pips: float
    fill_price: float
    
    # Outcome (filled later when trade closes)
    status: str = "open"  # 'open', 'win', 'loss', 'breakeven'
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'ts', 'manual'


class TradeJournal:
    """Manage trade journal - log, update, and analyze trades."""
    
    def __init__(self, path: Path = JOURNAL_PATH):
        self.path = path
        self.trades: List[TradeEntry] = []
        self._load()
    
    def _load(self):
        """Load existing journal from disk."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    self.trades = [TradeEntry(**t) for t in data.get('trades', [])]
                logger.info(f"Loaded {len(self.trades)} trades from journal")
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
                self.trades = []
        else:
            self.trades = []
    
    def _save(self):
        """Save journal to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'trades': [asdict(t) for t in self.trades],
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_trade(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        units: int,
        lots: float,
        expected_price: float,
        fill_price: float,
        stop_loss: float,
        take_profit: float,
        trailing_stop: float,
        atr: float,
        tcn_probability: float,
        ridge_confidence: float,
        xgb_momentum: float,
        rf_drawdown_pips: float,
    ) -> TradeEntry:
        """Log a new trade entry."""
        # Calculate slippage
        is_jpy = instrument.endswith('_JPY')
        pip_mult = 100 if is_jpy else 10000
        slippage = fill_price - expected_price
        slippage_pips = slippage * pip_mult
        
        entry = TradeEntry(
            trade_id=trade_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            instrument=instrument,
            direction=direction,
            entry_price=expected_price,
            expected_price=expected_price,
            units=units,
            lots=lots,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            atr=atr,
            tcn_probability=tcn_probability,
            ridge_confidence=ridge_confidence,
            xgb_momentum=xgb_momentum,
            rf_drawdown_pips=rf_drawdown_pips,
            slippage_pips=slippage_pips,
            fill_price=fill_price,
            status="open",
        )
        
        self.trades.append(entry)
        self._save()
        logger.info(f"Logged trade {trade_id}: {direction} {instrument}")
        return entry
    
    def update_from_oanda(self, client) -> int:
        """Update trade outcomes from OANDA.
        
        Args:
            client: OandaPracticeClient instance
            
        Returns:
            Number of trades updated
        """
        updated = 0
        
        for trade in self.trades:
            if trade.status != "open":
                continue
            
            try:
                # Fetch trade from OANDA
                result = client._request(
                    'GET',
                    f'/accounts/{client._config.account_id}/trades/{trade.trade_id}'
                )
                oanda_trade = result.get('trade', {})
                
                state = oanda_trade.get('state', 'OPEN')
                
                if state == 'CLOSED':
                    # Trade is closed - get P/L
                    pnl = float(oanda_trade.get('realizedPL', 0))
                    close_time = oanda_trade.get('closeTime', '')
                    
                    # Determine exit reason and price
                    exit_price = None
                    exit_reason = 'manual'
                    
                    # Check for TP/SL orders that filled
                    if 'takeProfitOrder' in oanda_trade:
                        tp_order = oanda_trade['takeProfitOrder']
                        if tp_order.get('state') == 'FILLED':
                            exit_reason = 'tp'
                            exit_price = float(tp_order.get('filledPrice', trade.take_profit))
                    
                    if 'stopLossOrder' in oanda_trade:
                        sl_order = oanda_trade['stopLossOrder']
                        if sl_order.get('state') == 'FILLED':
                            exit_reason = 'sl'
                            exit_price = float(sl_order.get('filledPrice', trade.stop_loss))
                    
                    if 'trailingStopLossOrder' in oanda_trade:
                        ts_order = oanda_trade['trailingStopLossOrder']
                        if ts_order.get('state') == 'FILLED':
                            exit_reason = 'ts'
                            exit_price = float(ts_order.get('filledPrice', 0))
                    
                    # Calculate pips
                    if exit_price:
                        is_jpy = trade.instrument.endswith('_JPY')
                        pip_mult = 100 if is_jpy else 10000
                        if trade.direction == 'long':
                            pnl_pips = (exit_price - trade.fill_price) * pip_mult
                        else:
                            pnl_pips = (trade.fill_price - exit_price) * pip_mult
                    else:
                        pnl_pips = pnl / (trade.lots * 10)  # Rough estimate
                    
                    # Determine win/loss
                    if pnl > 0:
                        status = 'win'
                    elif pnl < 0:
                        status = 'loss'
                    else:
                        status = 'breakeven'
                    
                    # Update trade
                    trade.status = status
                    trade.pnl = pnl
                    trade.pnl_pips = pnl_pips
                    trade.exit_price = exit_price
                    trade.exit_time = close_time
                    trade.exit_reason = exit_reason
                    
                    updated += 1
                    logger.info(f"Updated trade {trade.trade_id}: {status} ${pnl:.2f}")
                    
            except Exception as e:
                error_msg = str(e)
                # If trade not found (404), mark as cancelled/not found
                if "404" in error_msg or "does not exist" in error_msg.lower():
                    trade.status = "cancelled"
                    trade.pnl = 0.0
                    trade.pnl_pips = 0.0
                    trade.exit_reason = "not_found"
                    updated += 1
                    logger.warning(f"Trade {trade.trade_id} not found in OANDA - marked as cancelled")
                else:
                    logger.warning(f"Failed to update trade {trade.trade_id}: {e}")
                continue
        
        if updated > 0:
            self._save()
        
        return updated
    
    def get_statistics(self, days: Optional[int] = None) -> Dict[str, Any]:
        """Calculate performance statistics.
        
        Args:
            days: Only include trades from last N days. If None, include all trades.
        """
        # Filter by date if days specified
        trades_to_analyze = self.trades
        if days is not None:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            trades_to_analyze = [
                t for t in self.trades
                if datetime.fromisoformat(t.timestamp.replace('Z', '+00:00')) >= cutoff
            ]
        
        closed_trades = [t for t in trades_to_analyze if t.status in ('win', 'loss', 'breakeven')]
        
        if not closed_trades:
            return {
                'total_trades': len(trades_to_analyze),
                'open_trades': len([t for t in trades_to_analyze if t.status == 'open']),
                'closed_trades': 0,
                'message': 'No closed trades yet'
            }
        
        wins = [t for t in closed_trades if t.status == 'win']
        losses = [t for t in closed_trades if t.status == 'loss']
        
        total_pnl = sum(t.pnl or 0 for t in closed_trades)
        total_wins = sum(t.pnl or 0 for t in wins)
        total_losses = abs(sum(t.pnl or 0 for t in losses))
        
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Pips analysis
        total_pips = sum(t.pnl_pips or 0 for t in closed_trades)
        avg_win_pips = sum(t.pnl_pips or 0 for t in wins) / len(wins) if wins else 0
        avg_loss_pips = abs(sum(t.pnl_pips or 0 for t in losses)) / len(losses) if losses else 0
        
        # Slippage analysis
        avg_slippage = sum(abs(t.slippage_pips) for t in trades_to_analyze) / len(trades_to_analyze) if trades_to_analyze else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in closed_trades:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # By instrument
        by_instrument = {}
        for t in closed_trades:
            if t.instrument not in by_instrument:
                by_instrument[t.instrument] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.status == 'win':
                by_instrument[t.instrument]['wins'] += 1
            elif t.status == 'loss':
                by_instrument[t.instrument]['losses'] += 1
            by_instrument[t.instrument]['pnl'] += t.pnl or 0
        
        return {
            'total_trades': len(self.trades),
            'open_trades': len([t for t in self.trades if t.status == 'open']),
            'closed_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pips': total_pips,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'profit_factor': profit_factor,
            'avg_slippage_pips': avg_slippage,
            'exit_reasons': exit_reasons,
            'by_instrument': by_instrument,
        }
    
    def get_recent_trades(self, n: int = 10) -> List[TradeEntry]:
        """Get most recent trades."""
        return sorted(self.trades, key=lambda t: t.timestamp, reverse=True)[:n]
    
    def clear_journal(self):
        """Clear all trades from journal."""
        self.trades = []
        self._save()


def format_journal_stats(stats: Dict[str, Any]) -> str:
    """Format statistics for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 TRADE JOURNAL STATISTICS")
    lines.append("=" * 60)
    
    if stats.get('message'):
        lines.append(f"\n{stats['message']}")
        lines.append(f"Open trades: {stats['open_trades']}")
        return "\n".join(lines)
    
    # Summary
    lines.append(f"\n📈 SUMMARY")
    lines.append(f"  Total trades: {stats['total_trades']}")
    lines.append(f"  Open: {stats['open_trades']} | Closed: {stats['closed_trades']}")
    lines.append(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
    
    # Performance
    lines.append(f"\n💰 PERFORMANCE")
    win_rate = stats['win_rate'] * 100
    lines.append(f"  Win Rate: {win_rate:.1f}%")
    lines.append(f"  Total P/L: ${stats['total_pnl']:.2f}")
    lines.append(f"  Total Pips: {stats['total_pips']:.1f}")
    
    if stats['profit_factor'] != float('inf'):
        lines.append(f"  Profit Factor: {stats['profit_factor']:.2f}")
    else:
        lines.append(f"  Profit Factor: ∞ (no losses)")
    
    # Averages
    lines.append(f"\n📊 AVERAGES")
    lines.append(f"  Avg Win: ${sta