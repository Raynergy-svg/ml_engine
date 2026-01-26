"""
Trade Analyzer - Grounded reasoning about trade outcomes.

Analyzes completed trades using only recorded data (no LLM speculation).
Categorizes failures, detects patterns, and generates reports.

Usage:
    analyzer = TradeAnalyzer()
    report = analyzer.analyze_recent(n=50)
    print(analyzer.format_markdown(report))
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

# Gate thresholds (from modular_inference.py)
GATE_THRESHOLDS = {
    'tcn_probability': 0.55,
    'ridge_confidence': 45.0,
    'xgb_momentum': 0.15,
    'rf_drawdown_pips': 250.0,  # 250 pips = 2.5% for most pairs
}


@dataclass
class FailureMode:
    """A detected pattern in losing trades."""
    mode: str  # e.g., 'momentum_marginal', 'confidence_overestimate'
    count: int
    description: str
    avg_value: float
    threshold: float
    trades: List[str]  # trade_ids


@dataclass
class TradeAnalysisReport:
    """Complete analysis of trade history."""
    # Summary
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    total_pnl: float
    total_pips: float
    avg_win_pips: float
    avg_loss_pips: float
    profit_factor: float
    
    # By gate performance
    gate_performance: Dict[str, Dict[str, float]]
    
    # Failure modes
    failure_modes: List[FailureMode]
    
    # Calibration
    calibration: Dict[str, float]
    
    # Time-based analysis
    time_analysis: Dict[str, Any]
    
    # Instrument breakdown
    instrument_breakdown: Dict[str, Dict[str, float]]
    
    # Recent trend
    recent_trend: Dict[str, float]


class TradeAnalyzer:
    """
    Analyze trade outcomes using only recorded data.
    
    No LLM speculation - all insights are data-driven pattern matching.
    """
    
    def __init__(self, journal_path: Path = None):
        self.journal_path = journal_path or Path("trained_data/trade_journal.json")
        self._trades = None
    
    def _load_trades(self) -> List[Dict]:
        """Load trades from journal."""
        if self._trades is not None:
            return self._trades
        
        if not self.journal_path.exists():
            logger.warning(f"Trade journal not found: {self.journal_path}")
            return []
        
        try:
            with open(self.journal_path) as f:
                data = json.load(f)
            self._trades = data.get('trades', [])
            logger.info(f"Loaded {len(self._trades)} trades from journal")
            return self._trades
        except Exception as e:
            logger.error(f"Failed to load journal: {e}")
            return []
    
    def get_closed_trades(self, n: Optional[int] = None) -> List[Dict]:
        """Get closed trades (win/loss/breakeven), optionally limited to last N."""
        trades = self._load_trades()
        closed = [t for t in trades if t.get('status') in ('win', 'loss', 'breakeven')]
        
        # Sort by timestamp descending
        closed.sort(key=lambda t: t.get('timestamp', ''), reverse=True)
        
        if n:
            closed = closed[:n]
        
        return closed
    
    def analyze_recent(self, n: int = 50) -> TradeAnalysisReport:
        """
        Analyze the most recent N closed trades.
        
        Returns:
            TradeAnalysisReport with all analysis results
        """
        trades = self.get_closed_trades(n)
        
        if not trades:
            return self._empty_report()
        
        # Basic stats
        wins = [t for t in trades if t.get('status') == 'win']
        losses = [t for t in trades if t.get('status') == 'loss']
        breakeven = [t for t in trades if t.get('status') == 'breakeven']
        
        total_pnl = sum(t.get('pnl', 0) or 0 for t in trades)
        total_pips = sum(t.get('pnl_pips', 0) or 0 for t in trades)
        
        win_pips = [t.get('pnl_pips', 0) or 0 for t in wins]
        loss_pips = [abs(t.get('pnl_pips', 0) or 0) for t in losses]
        
        avg_win_pips = np.mean(win_pips) if win_pips and np is not None else 0
        avg_loss_pips = np.mean(loss_pips) if loss_pips and np is not None else 0
        
        total_wins_pips = sum(win_pips)
        total_losses_pips = sum(loss_pips)
        profit_factor = total_wins_pips / total_losses_pips if total_losses_pips > 0 else float('inf')
        
        # Gate performance
        gate_performance = self._analyze_gate_performance(trades)
        
        # Failure modes
        failure_modes = self._detect_failure_modes(losses)
        
        # Calibration
        calibration = self._analyze_calibration(trades)
        
        # Time analysis
        time_analysis = self._analyze_time_patterns(trades)
        
        # Instrument breakdown
        instrument_breakdown = self._analyze_by_instrument(trades)
        
        # Recent trend (last 10 vs previous)
        recent_trend = self._analyze_trend(trades)
        
        return TradeAnalysisReport(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            breakeven=len(breakeven),
            win_rate=len(wins) / len(trades) if trades else 0,
            total_pnl=total_pnl,
            total_pips=total_pips,
            avg_win_pips=avg_win_pips,
            avg_loss_pips=avg_loss_pips,
            profit_factor=profit_factor,
            gate_performance=gate_performance,
            failure_modes=failure_modes,
            calibration=calibration,
            time_analysis=time_analysis,
            instrument_breakdown=instrument_breakdown,
            recent_trend=recent_trend,
        )
    
    def _empty_report(self) -> TradeAnalysisReport:
        """Return empty report when no trades available."""
        return TradeAnalysisReport(
            total_trades=0,
            wins=0,
            losses=0,
            breakeven=0,
            win_rate=0,
            total_pnl=0,
            total_pips=0,
            avg_win_pips=0,
            avg_loss_pips=0,
            profit_factor=0,
            gate_performance={},
            failure_modes=[],
            calibration={},
            time_analysis={},
            instrument_breakdown={},
            recent_trend={},
        )
    
    def _analyze_gate_performance(self, trades: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze win rate by gate signal strength."""
        gates = {
            'tcn': 'tcn_probability',
            'momentum': 'xgb_momentum', 
            'confidence': 'ridge_confidence',
            'drawdown': 'rf_drawdown_pips',
        }
        
        performance = {}
        
        for gate_name, field in gates.items():
            # Group trades by signal strength quartile
            values = [(t.get(field, 0) or 0, t.get('status') == 'win') for t in trades]
            values = [(v, w) for v, w in values if v != 0]  # Filter zero values
            
            if not values:
                performance[gate_name] = {'win_rate': 0, 'avg_value': 0, 'count': 0}
                continue
            
            win_rate = sum(1 for _, w in values if w) / len(values)
            avg_value = sum(v for v, _ in values) / len(values) if np is None else float(np.mean([v for v, _ in values]))
            
            # Marginal trades (within 10% of threshold)
            threshold = GATE_THRESHOLDS.get(field, 0)
            if threshold > 0:
                marginal = [v for v, _ in values if 0.9 * threshold <= v <= 1.1 * threshold]
                marginal_pct = len(marginal) / len(values) if values else 0
            else:
                marginal_pct = 0
            
            performance[gate_name] = {
                'win_rate': win_rate,
                'avg_value': avg_value,
                'count': len(values),
                'threshold': threshold,
                'marginal_pct': marginal_pct,
            }
        
        return performance
    
    def _detect_failure_modes(self, losses: List[Dict]) -> List[FailureMode]:
        """Detect patterns in losing trades."""
        if not losses:
            return []
        
        modes = []
        
        # 1. Momentum marginal (momentum just above threshold)
        momentum_marginal = [
            t for t in losses
            if 0.15 <= (t.get('xgb_momentum', 0) or 0) <= 0.20
        ]
        if len(momentum_marginal) >= 2:
            avg_val = sum(t.get('xgb_momentum', 0) or 0 for t in momentum_marginal) / len(momentum_marginal)
            modes.append(FailureMode(
                mode='momentum_marginal',
                count=len(momentum_marginal),
                description='Momentum gate was marginally above threshold (0.15-0.20)',
                avg_value=avg_val,
                threshold=0.15,
                trades=[t.get('trade_id', '') for t in momentum_marginal],
            ))
        
        # 2. Confidence overestimate (high confidence but lost)
        confidence_overestimate = [
            t for t in losses
            if (t.get('ridge_confidence', 0) or 0) >= 65
        ]
        if len(confidence_overestimate) >= 2:
            avg_val = sum(t.get('ridge_confidence', 0) or 0 for t in confidence_overestimate) / len(confidence_overestimate)
            modes.append(FailureMode(
                mode='confidence_overestimate',
                count=len(confidence_overestimate),
                description='High confidence (≥65) but trade lost',
                avg_value=avg_val,
                threshold=65.0,
                trades=[t.get('trade_id', '') for t in confidence_overestimate],
            ))
        
        # 3. TCN probability marginal
        tcn_marginal = [
            t for t in losses
            if 0.55 <= (t.get('tcn_probability', 0) or 0) <= 0.60
        ]
        if len(tcn_marginal) >= 2:
            avg_val = sum(t.get('tcn_probability', 0) or 0 for t in tcn_marginal) / len(tcn_marginal)
            modes.append(FailureMode(
                mode='tcn_marginal',
                count=len(tcn_marginal),
                description='TCN probability was marginally above threshold (0.55-0.60)',
                avg_value=avg_val,
                threshold=0.55,
                trades=[t.get('trade_id', '') for t in tcn_marginal],
            ))
        
        # 4. High drawdown expected but still took trade
        high_drawdown = [
            t for t in losses
            if (t.get('rf_drawdown_pips', 0) or 0) >= 200.0  # 200 pips = elevated risk
        ]
        if len(high_drawdown) >= 2:
            avg_val = sum(t.get('rf_drawdown_pips', 0) or 0 for t in high_drawdown) / len(high_drawdown)
            modes.append(FailureMode(
                mode='high_drawdown_risk',
                count=len(high_drawdown),
                description='Trade taken despite elevated drawdown risk (≥200 pips)',
                avg_value=avg_val,
                threshold=200.0,
                trades=[t.get('trade_id', '') for t in high_drawdown],
            ))
        
        # 5. Stop loss hit (not TP)
        sl_exits = [t for t in losses if t.get('exit_reason') == 'sl']
        if len(sl_exits) >= 3:
            avg_atr = sum(t.get('atr', 0) or 0 for t in sl_exits) / len(sl_exits)
            modes.append(FailureMode(
                mode='stop_loss_triggered',
                count=len(sl_exits),
                description='Trade exited via stop loss',
                avg_value=avg_atr,
                threshold=0,
                trades=[t.get('trade_id', '') for t in sl_exits],
            ))
        
        # Sort by count descending
        modes.sort(key=lambda m: m.count, reverse=True)
        
        return modes
    
    def _analyze_calibration(self, trades: List[Dict]) -> Dict[str, float]:
        """Check if confidence scores are well-calibrated."""
        # Group by confidence bucket and check actual win rate
        buckets = {
            '45-55': (45, 55),
            '55-65': (55, 65),
            '65-75': (65, 75),
            '75-85': (75, 85),
            '85+': (85, 100),
        }
        
        calibration = {}
        
        for bucket_name, (low, high) in buckets.items():
            bucket_trades = [
                t for t in trades
                if low <= (t.get('ridge_confidence', 0) or 0) < high
            ]
            
            if len(bucket_trades) >= 3:
                actual_win_rate = sum(1 for t in bucket_trades if t.get('status') == 'win') / len(bucket_trades)
                expected_win_rate = (low + high) / 200  # Midpoint as expected
                drift = actual_win_rate - expected_win_rate
                
                calibration[bucket_name] = {
                    'count': len(bucket_trades),
                    'expected': expected_win_rate,
                    'actual': actual_win_rate,
                    'drift': drift,
                }
        
        return calibration
    
    def _analyze_time_patterns(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze win rate by time of day, day of week."""
        hour_wins = defaultdict(lambda: {'wins': 0, 'total': 0})
        day_wins = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for t in trades:
            try:
                ts = t.get('timestamp', '')
                if not ts:
                    continue
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                hour = dt.hour
                day = dt.strftime('%A')
                
                is_win = t.get('status') == 'win'
                hour_wins[hour]['total'] += 1
                hour_wins[hour]['wins'] += 1 if is_win else 0
                day_wins[day]['total'] += 1
                day_wins[day]['wins'] += 1 if is_win else 0
            except Exception:
                continue
        
        # Calculate win rates
        hour_performance = {
            h: {'win_rate': d['wins'] / d['total'] if d['total'] > 0 else 0, 'count': d['total']}
            for h, d in sorted(hour_wins.items())
        }
        
        day_performance = {
            d: {'win_rate': data['wins'] / data['total'] if data['total'] > 0 else 0, 'count': data['total']}
            for d, data in day_wins.items()
        }
        
        # Find best/worst
        best_hour = max(hour_performance.items(), key=lambda x: x[1]['win_rate']) if hour_performance else (None, {})
        worst_hour = min(hour_performance.items(), key=lambda x: x[1]['win_rate']) if hour_performance else (None, {})
        
        return {
            'by_hour': hour_performance,
            'by_day': day_performance,
            'best_hour': best_hour[0],
            'worst_hour': worst_hour[0],
        }
    
    def _analyze_by_instrument(self, trades: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by instrument."""
        by_instrument = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0, 'pips': 0})
        
        for t in trades:
            inst = t.get('instrument', 'UNKNOWN')
            by_instrument[inst]['wins'] += 1 if t.get('status') == 'win' else 0
            by_instrument[inst]['losses'] += 1 if t.get('status') == 'loss' else 0
            by_instrument[inst]['pnl'] += t.get('pnl', 0) or 0
            by_instrument[inst]['pips'] += t.get('pnl_pips', 0) or 0
        
        result = {}
        for inst, data in by_instrument.items():
            total = data['wins'] + data['losses']
            result[inst] = {
                'total': total,
                'win_rate': data['wins'] / total if total > 0 else 0,
                'pnl': data['pnl'],
                'pips': data['pips'],
            }
        
        return result
    
    def _analyze_trend(self, trades: List[Dict], recent_n: int = 10) -> Dict[str, float]:
        """Compare recent trades to previous period."""
        if len(trades) < recent_n * 2:
            return {'recent_win_rate': 0, 'previous_win_rate': 0, 'trend': 'insufficient_data'}
        
        recent = trades[:recent_n]
        previous = trades[recent_n:recent_n * 2]
        
        recent_wr = sum(1 for t in recent if t.get('status') == 'win') / len(recent)
        previous_wr = sum(1 for t in previous if t.get('status') == 'win') / len(previous)
        
        diff = recent_wr - previous_wr
        if diff > 0.1:
            trend = 'improving'
        elif diff < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'recent_win_rate': recent_wr,
            'previous_win_rate': previous_wr,
            'difference': diff,
            'trend': trend,
        }
    
    def format_markdown(self, report: TradeAnalysisReport) -> str:
        """Format report as Markdown for CLI display."""
        if report.total_trades == 0:
            return "# Trade Analysis Report\n\n⚠️ No closed trades found in journal."
        
        lines = [
            "# Trade Analysis Report",
            "",
            f"**Period:** Last {report.total_trades} trades",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {report.total_trades} |",
            f"| Wins | {report.wins} |",
            f"| Losses | {report.losses} |",
            f"| Win Rate | {report.win_rate:.1%} |",
            f"| Total P/L | ${report.total_pnl:,.2f} |",
            f"| Total Pips | {report.total_pips:+.1f} |",
            f"| Avg Win | {report.avg_win_pips:+.1f} pips |",
            f"| Avg Loss | {report.avg_loss_pips:-.1f} pips |",
            f"| Profit Factor | {report.profit_factor:.2f} |",
            "",
        ]
        
        # Gate performance
        if report.gate_performance:
            lines.extend([
                "## Gate Performance",
                "",
                "| Gate | Win Rate | Avg Value | Threshold | Marginal % |",
                "|------|----------|-----------|-----------|------------|",
            ])
            for gate, data in report.gate_performance.items():
                wr = data.get('win_rate', 0)
                avg = data.get('avg_value', 0)
                thresh = data.get('threshold', 0)
                marg = data.get('marginal_pct', 0)
                lines.append(f"| {gate} | {wr:.1%} | {avg:.3f} | {thresh:.2f} | {marg:.1%} |")
            lines.append("")
        
        # Failure modes
        if report.failure_modes:
            lines.extend([
                "## Detected Failure Modes",
                "",
            ])
            for mode in report.failure_modes:
                lines.append(f"### {mode.mode} ({mode.count} trades)")
                lines.append(f"- **Description:** {mode.description}")
                lines.append(f"- **Average value:** {mode.avg_value:.3f} (threshold: {mode.threshold})")
                lines.append("")
        
        # Calibration
        if report.calibration:
            lines.extend([
                "## Confidence Calibration",
                "",
                "| Bucket | Count | Expected | Actual | Drift |",
                "|--------|-------|----------|--------|-------|",
            ])
            for bucket, data in report.calibration.items():
                if isinstance(data, dict):
                    lines.append(
                        f"| {bucket} | {data['count']} | {data['expected']:.1%} | "
                        f"{data['actual']:.1%} | {data['drift']:+.1%} |"
                    )
            lines.append("")
        
        # Instrument breakdown
        if report.instrument_breakdown:
            lines.extend([
                "## Performance by Instrument",
                "",
                "| Instrument | Trades | Win Rate | P/L | Pips |",
                "|------------|--------|----------|-----|------|",
            ])
            for inst, data in sorted(report.instrument_breakdown.items(), key=lambda x: -x[1]['pnl']):
                lines.append(
                    f"| {inst} | {data['total']} | {data['win_rate']:.1%} | "
                    f"${data['pnl']:,.2f} | {data['pips']:+.1f} |"
                )
            lines.append("")
        
        # Trend
        if report.recent_trend:
            trend = report.recent_trend
            emoji = "📈" if trend.get('trend') == 'improving' else "📉" if trend.get('trend') == 'declining' else "➡️"
            lines.extend([
                "## Recent Trend",
                "",
                f"- Recent win rate (last 10): {trend.get('recent_win_rate', 0):.1%}",
                f"- Previous win rate: {trend.get('previous_win_rate', 0):.1%}",
                f"- Trend: {emoji} **{trend.get('trend', 'unknown')}**",
                "",
            ])
        
        return "\n".join(lines)
    
    def to_dict(self, report: TradeAnalysisReport) -> Dict[str, Any]:
        """Convert report to dictionary for JSON export."""
        result = asdict(report)
        # Convert FailureMode dataclasses
        result['failure_modes'] = [asdict(fm) for fm in report.failure_modes]
        return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for trade analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trade history")
    parser.add_argument("--last", type=int, default=50, help="Number of recent trades to analyze")
    parser.add_argument("--export", type=str, help="Export report to JSON file")
    args = parser.parse_args()
    
    analyzer = TradeAnalyzer()
    report = analyzer.analyze_recent(n=args.last)
    
    # Print markdown report
    print(analyzer.format_markdown(report))
    
    # Export if requested
    if args.export:
        export_path = Path(args.export)
        with open(export_path, 'w') as f:
            json.dump(analyzer.to_dict(report), f, indent=2)
        print(f"\n✅ Report exported to {export_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
