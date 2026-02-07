#!/usr/bin/env python3
"""
Monitoring Dashboard for FX Trading Bot.

This script provides a simple CLI dashboard for monitoring trading performance,
model health, and alerts.

Usage:
    python scripts/monitoring_dashboard.py              # Show current status
    python scripts/monitoring_dashboard.py --report     # Generate full report
    python scripts/monitoring_dashboard.py --alerts     # Show only alerts
    python scripts/monitoring_dashboard.py --drift      # Show model drift history
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.monitoring import MonitoringSystem, create_monitoring_report, AlertLevel
    import yaml
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def load_config(config_path: str = "config/config_improved_H1.yaml") -> dict:
    """Load configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {'paths': {'log_dir': 'trained_data/logs'}}
    
    with open(config_file) as f:
        return yaml.safe_load(f)


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts_str


def print_dashboard(monitor: MonitoringSystem) -> None:
    """Print dashboard summary."""
    print("\n" + "=" * 80)
    print("FX TRADING BOT - MONITORING DASHBOARD".center(80))
    print("=" * 80)
    
    # Get dashboard data
    data = monitor.get_dashboard_data()
    
    # Timestamp
    print(f"\nLast Updated: {format_timestamp(data['timestamp'])}")
    
    # Alert summary
    print("\n📊 ALERT SUMMARY")
    print("-" * 80)
    alerts = data['alerts']
    print(f"  Total Alerts:    {alerts['total']}")
    print(f"  🔴 Critical:     {alerts['critical']}")
    print(f"  ⚠️  Warning:      {alerts['warning']}")
    print(f"  ℹ️  Info:         {alerts['info']}")
    
    # Recent alerts (last 5)
    recent_alerts = monitor.get_alerts()[-5:]
    if recent_alerts:
        print("\n  Recent Alerts:")
        for alert in recent_alerts:
            icon = "🔴" if alert.level == AlertLevel.CRITICAL else "⚠️" if alert.level == AlertLevel.WARNING else "ℹ️"
            print(f"    {icon} [{format_timestamp(alert.timestamp)}] {alert.category}: {alert.message}")
    
    # Model drift
    print("\n📈 MODEL HEALTH")
    print("-" * 80)
    recent_drift = data['recent_drift']
    if recent_drift:
        latest = recent_drift[-1]
        print(f"  Last Check:      {format_timestamp(latest['timestamp'])}")
        print(f"  Confidence:      {latest['confidence_mean']:.3f}")
        print(f"  Drift Score:     {latest['drift_score']:.3f}")
        status = "🔴 ALERT" if latest['alert_triggered'] else "✅ OK"
        print(f"  Status:          {status}")
        
        # Trend over last 10 checks
        if len(recent_drift) >= 2:
            first_conf = recent_drift[0]['confidence_mean']
            last_conf = latest['confidence_mean']
            trend = "📈 IMPROVING" if last_conf > first_conf else "📉 DECLINING"
            print(f"  Trend:           {trend}")
    else:
        print("  No drift data available")
    
    # Baseline performance
    print("\n⚡ PERFORMANCE BASELINE")
    print("-" * 80)
    baseline = data.get('baseline_metrics')
    if baseline and baseline.get('win_rate'):
        print(f"  Win Rate:        {baseline['win_rate']:.1%}")
        print(f"  Sharpe Ratio:    {baseline.get('sharpe_ratio', 0):.2f}")
    else:
        print("  No baseline metrics available")
        print("  Run with --set-baseline to establish baseline")
    
    print("\n" + "=" * 80)


def print_alerts(monitor: MonitoringSystem, level: str = None) -> None:
    """Print detailed alert list."""
    print("\n" + "=" * 80)
    print("ALERTS".center(80))
    print("=" * 80)
    
    # Filter by level if specified
    if level:
        level_enum = AlertLevel[level.upper()]
        alerts = monitor.get_alerts(level=level_enum)
        print(f"\nFiltering by level: {level.upper()}")
    else:
        alerts = monitor.get_alerts()
    
    if not alerts:
        print("\n✅ No alerts to display")
        return
    
    # Group by category
    by_category = {}
    for alert in alerts:
        if alert.category not in by_category:
            by_category[alert.category] = []
        by_category[alert.category].append(alert)
    
    # Print by category
    for category, cat_alerts in sorted(by_category.items()):
        print(f"\n{category.upper().replace('_', ' ')}")
        print("-" * 80)
        for alert in cat_alerts:
            icon = "🔴" if alert.level == AlertLevel.CRITICAL else "⚠️" if alert.level == AlertLevel.WARNING else "ℹ️"
            print(f"{icon} [{format_timestamp(alert.timestamp)}] {alert.message}")
            if alert.data:
                for key, value in alert.data.items():
                    print(f"   └─ {key}: {value}")
    
    print("\n" + "=" * 80)


def print_drift_history(monitor: MonitoringSystem, limit: int = 20) -> None:
    """Print model drift history."""
    print("\n" + "=" * 80)
    print("MODEL DRIFT HISTORY".center(80))
    print("=" * 80)
    
    if not monitor.drift_history:
        print("\n❌ No drift history available")
        return
    
    # Get recent history
    recent = monitor.drift_history[-limit:]
    
    print(f"\nShowing last {len(recent)} drift checks:\n")
    print(f"{'Timestamp':<20} {'Confidence':>12} {'Drift Score':>12} {'Status':>10}")
    print("-" * 80)
    
    for metrics in recent:
        status = "🔴 ALERT" if metrics.alert_triggered else "✅ OK"
        print(
            f"{format_timestamp(metrics.timestamp):<20} "
            f"{metrics.confidence_mean:>12.3f} "
            f"{metrics.feature_drift_score:>12.3f} "
            f"{status:>10}"
        )
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="FX Trading Bot Monitoring Dashboard")
    parser.add_argument(
        '--config',
        default='config/config_improved_H1.yaml',
        help='Config file path (default: config/config_improved_H1.yaml)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate full monitoring report'
    )
    parser.add_argument(
        '--alerts',
        action='store_true',
        help='Show detailed alerts'
    )
    parser.add_argument(
        '--alert-level',
        choices=['info', 'warning', 'critical'],
        help='Filter alerts by level'
    )
    parser.add_argument(
        '--drift',
        action='store_true',
        help='Show model drift history'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limit for drift history (default: 20)'
    )
    parser.add_argument(
        '--set-baseline',
        action='store_true',
        help='Set current performance as baseline'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize monitoring system
    monitor = MonitoringSystem(config)
    
    # Execute requested action
    if args.report:
        report_path = create_monitoring_report(monitor)
        print(f"\n✅ Full monitoring report generated: {report_path}")
        
    elif args.alerts:
        print_alerts(monitor, level=args.alert_level)
        
    elif args.drift:
        print_drift_history(monitor, limit=args.limit)
        
    elif args.set_baseline:
        print("\n⚠️  Setting baseline requires trade data.")
        print("This feature should be called programmatically after a trading session.")
        print("Example: monitor.save_baseline_metrics(metrics)")
        
    else:
        # Default: show dashboard
        print_dashboard(monitor)
        
        # Suggestion
        print("\n💡 TIP: Use --report to generate a full report, --alerts to see details, or --drift for model health")


if __name__ == "__main__":
    main()
