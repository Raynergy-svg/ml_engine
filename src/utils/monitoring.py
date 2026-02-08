"""
Monitoring and Alerting System for FX Trading Bot.

This module provides:
1. Performance monitoring and anomaly detection
2. Alert system for critical events
3. Trading metrics dashboard data aggregation
4. Model drift detection

Usage:
    from src.utils.monitoring import MonitoringSystem, Alert, AlertLevel

    monitor = MonitoringSystem(config)
    monitor.check_daily_performance()
    monitor.check_model_drift(predictions)
    monitor.send_alerts()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    category: str
    message: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    date: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration_hours: float
    largest_win: float
    largest_loss: float


@dataclass
class ModelDriftMetrics:
    """Model drift detection metrics."""
    timestamp: str
    prediction_mean: float
    prediction_std: float
    confidence_mean: float
    feature_drift_score: float
    performance_degradation: float
    alert_triggered: bool


class MonitoringSystem:
    """
    Comprehensive monitoring system for FX trading bot.

    Features:
    - Daily performance tracking
    - Model drift detection
    - Alert generation and management
    - Trading metrics aggregation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize monitoring system.

        Args:
            config: Configuration dictionary
            log_dir: Directory for monitoring logs
        """
        self.config = config
        self.log_dir = Path(log_dir or config.get('paths', {}).get('log_dir', 'trained_data/logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Alert settings
        self.alerts: List[Alert] = []
        self.alert_thresholds = self._load_alert_thresholds()

        # Performance baselines
        self.baseline_metrics = self._load_baseline_metrics()

        # Model drift tracking
        self.drift_history: List[ModelDriftMetrics] = []

        logger.info(f"Monitoring system initialized. Log dir: {self.log_dir}")

    def _load_alert_thresholds(self) -> Dict[str, Any]:
        """Load alert threshold configuration."""
        defaults = {
            'daily_loss_threshold': 0.05,  # 5% daily loss triggers alert
            'max_drawdown_threshold': 0.10,  # 10% max drawdown
            'win_rate_degradation': 0.10,  # 10% drop in win rate
            'sharpe_ratio_min': 0.3,  # Minimum acceptable Sharpe
            'model_drift_threshold': 0.15,  # 15% drift score
            'confidence_drop_threshold': 0.10,  # 10% drop in avg confidence
            'consecutive_losses_max': 5,  # Max consecutive losses
        }

        # Override with config if available
        monitoring_cfg = self.config.get('monitoring', {})
        return {**defaults, **monitoring_cfg.get('alert_thresholds', {})}

    def _load_baseline_metrics(self) -> Optional[PerformanceMetrics]:
        """Load baseline performance metrics."""
        baseline_path = self.log_dir / 'baseline_metrics.json'
        if not baseline_path.exists():
            return None

        try:
            data = json.loads(baseline_path.read_text())
            return PerformanceMetrics(**data)
        except Exception as e:
            logger.warning(f"Failed to load baseline metrics: {e}")
            return None

    def save_baseline_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save baseline performance metrics."""
        baseline_path = self.log_dir / 'baseline_metrics.json'
        data = {
            'date': metrics.date,
            'total_pnl': metrics.total_pnl,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'avg_trade_duration_hours': metrics.avg_trade_duration_hours,
            'largest_win': metrics.largest_win,
            'largest_loss': metrics.largest_loss,
        }
        baseline_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Baseline metrics saved: {baseline_path}")

    def check_daily_performance(
        self,
        pnl: float,
        trades: List[Dict[str, Any]],
        starting_balance: float,
    ) -> None:
        """
        Check daily performance and generate alerts if needed.

        Args:
            pnl: Total profit/loss for the day
            trades: List of trades
            starting_balance: Starting balance for the day
        """
        if not trades:
            return

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        sum(1 for t in trades if t.get('pnl', 0) < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate drawdown
        cumulative_pnl = np.cumsum([t.get('pnl', 0) for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = (running_max - cumulative_pnl) / starting_balance
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        # Daily loss check
        daily_loss_pct = abs(pnl / starting_balance) if pnl < 0 else 0
        if daily_loss_pct >= self.alert_thresholds['daily_loss_threshold']:
            self.add_alert(
                AlertLevel.CRITICAL,
                "daily_loss",
                f"Daily loss threshold exceeded: {daily_loss_pct:.1%} "
                f"(threshold: {self.alert_thresholds['daily_loss_threshold']:.1%})",
                {'pnl': pnl, 'pct': daily_loss_pct}
            )

        # Max drawdown check
        if max_drawdown >= self.alert_thresholds['max_drawdown_threshold']:
            self.add_alert(
                AlertLevel.CRITICAL,
                "max_drawdown",
                f"Maximum drawdown exceeded: {max_drawdown:.1%} "
                f"(threshold: {self.alert_thresholds['max_drawdown_threshold']:.1%})",
                {'max_drawdown': max_drawdown}
            )

        # Consecutive losses check
        consecutive_losses = 0
        max_consecutive = 0
        for trade in trades:
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0

        if max_consecutive >= self.alert_thresholds['consecutive_losses_max']:
            self.add_alert(
                AlertLevel.WARNING,
                "consecutive_losses",
                f"Consecutive losses: {max_consecutive} "
                f"(threshold: {self.alert_thresholds['consecutive_losses_max']})",
                {'consecutive_losses': max_consecutive}
            )

        # Win rate degradation (if baseline exists)
        if self.baseline_metrics and win_rate > 0:
            degradation = self.baseline_metrics.win_rate - win_rate
            if degradation >= self.alert_thresholds['win_rate_degradation']:
                self.add_alert(
                    AlertLevel.WARNING,
                    "win_rate_degradation",
                    f"Win rate degraded by {degradation:.1%} "
                    f"(current: {win_rate:.1%}, baseline: {self.baseline_metrics.win_rate:.1%})",
                    {'current_win_rate': win_rate, 'baseline_win_rate': self.baseline_metrics.win_rate}
                )

        logger.info(
            f"Daily performance check: PnL={pnl:.2f}, "
            f"Trades={total_trades}, Win Rate={win_rate:.1%}, "
            f"Max DD={max_drawdown:.1%}"
        )

    def check_model_drift(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        feature_stats: Optional[Dict[str, float]] = None,
    ) -> ModelDriftMetrics:
        """
        Check for model drift using prediction statistics.

        Args:
            predictions: Model predictions
            confidences: Model confidence scores
            feature_stats: Optional feature distribution statistics

        Returns:
            ModelDriftMetrics object
        """
        timestamp = datetime.now().isoformat()

        # Calculate current statistics
        pred_mean = float(np.mean(predictions))
        pred_std = float(np.std(predictions))
        conf_mean = float(np.mean(confidences))

        # Feature drift score (if stats provided)
        feature_drift_score = 0.0
        if feature_stats:
            # Simple drift score based on feature mean shift
            baseline_stats = self._load_feature_baseline()
            if baseline_stats:
                drifts = []
                for key, value in feature_stats.items():
                    baseline_val = baseline_stats.get(key, value)
                    if baseline_val != 0:
                        drift = abs(value - baseline_val) / abs(baseline_val)
                        drifts.append(drift)
                feature_drift_score = float(np.mean(drifts)) if drifts else 0.0

        # Performance degradation (if baseline exists)
        performance_degradation = 0.0
        if self.baseline_metrics:
            # Use confidence as a proxy for performance
            performance_degradation = max(0, self.baseline_metrics.win_rate - conf_mean)

        # Check thresholds
        alert_triggered = False
        if feature_drift_score >= self.alert_thresholds['model_drift_threshold']:
            alert_triggered = True
            self.add_alert(
                AlertLevel.WARNING,
                "model_drift",
                f"Model drift detected: score={feature_drift_score:.3f} "
                f"(threshold: {self.alert_thresholds['model_drift_threshold']:.3f})",
                {'drift_score': feature_drift_score}
            )

        if performance_degradation >= self.alert_thresholds['confidence_drop_threshold']:
            alert_triggered = True
            self.add_alert(
                AlertLevel.WARNING,
                "confidence_drop",
                f"Model confidence dropped: {performance_degradation:.1%}",
                {'degradation': performance_degradation}
            )

        # Create metrics object
        metrics = ModelDriftMetrics(
            timestamp=timestamp,
            prediction_mean=pred_mean,
            prediction_std=pred_std,
            confidence_mean=conf_mean,
            feature_drift_score=feature_drift_score,
            performance_degradation=performance_degradation,
            alert_triggered=alert_triggered,
        )

        self.drift_history.append(metrics)
        self._save_drift_metrics(metrics)

        logger.info(
            f"Model drift check: pred_mean={pred_mean:.3f}, "
            f"conf_mean={conf_mean:.3f}, drift={feature_drift_score:.3f}"
        )

        return metrics

    def _load_feature_baseline(self) -> Optional[Dict[str, float]]:
        """Load baseline feature statistics."""
        baseline_path = self.log_dir / 'feature_baseline.json'
        if not baseline_path.exists():
            return None

        try:
            return json.loads(baseline_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load feature baseline: {e}")
            return None

    def _save_drift_metrics(self, metrics: ModelDriftMetrics) -> None:
        """Save drift metrics to log."""
        drift_log = self.log_dir / 'model_drift.jsonl'

        data = {
            'timestamp': metrics.timestamp,
            'prediction_mean': metrics.prediction_mean,
            'prediction_std': metrics.prediction_std,
            'confidence_mean': metrics.confidence_mean,
            'feature_drift_score': metrics.feature_drift_score,
            'performance_degradation': metrics.performance_degradation,
            'alert_triggered': metrics.alert_triggered,
        }

        # Append to JSONL file
        with drift_log.open('a') as f:
            f.write(json.dumps(data) + '\n')

    def add_alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an alert to the queue."""
        alert = Alert(
            level=level,
            category=category,
            message=message,
            timestamp=datetime.now().isoformat(),
            data=data or {},
        )
        self.alerts.append(alert)
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else
            logging.WARNING if level == AlertLevel.WARNING else
            logging.INFO,
            f"[{level.value.upper()}] {category}: {message}"
        )

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None,
    ) -> List[Alert]:
        """Get alerts, optionally filtered by level or category."""
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []

    def export_alerts(self, path: Optional[Path] = None) -> Path:
        """Export alerts to JSON file."""
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.log_dir / f'alerts_{timestamp}.json'

        data = [
            {
                'level': a.level.value,
                'category': a.category,
                'message': a.message,
                'timestamp': a.timestamp,
                'data': a.data,
            }
            for a in self.alerts
        ]

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Alerts exported to {path}")
        return path

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get aggregated data for monitoring dashboard.

        Returns:
            Dictionary with dashboard metrics
        """
        # Recent drift metrics
        recent_drift = self.drift_history[-10:] if self.drift_history else []

        # Alert summary
        alert_summary = {
            'total': len(self.alerts),
            'critical': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            'warning': len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
            'info': len([a for a in self.alerts if a.level == AlertLevel.INFO]),
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'alerts': alert_summary,
            'recent_drift': [
                {
                    'timestamp': m.timestamp,
                    'confidence_mean': m.confidence_mean,
                    'drift_score': m.feature_drift_score,
                    'alert_triggered': m.alert_triggered,
                }
                for m in recent_drift
            ],
            'baseline_metrics': {
                'win_rate': self.baseline_metrics.win_rate if self.baseline_metrics else None,
                'sharpe_ratio': self.baseline_metrics.sharpe_ratio if self.baseline_metrics else None,
            } if self.baseline_metrics else None,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def create_monitoring_report(
    monitor: MonitoringSystem,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Create a comprehensive monitoring report.

    Args:
        monitor: MonitoringSystem instance
        output_path: Optional output path

    Returns:
        Path to report file
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = monitor.log_dir / f'monitoring_report_{timestamp}.json'

    # Gather all data
    dashboard_data = monitor.get_dashboard_data()
    all_alerts = [
        {
            'level': a.level.value,
            'category': a.category,
            'message': a.message,
            'timestamp': a.timestamp,
            'data': a.data,
        }
        for a in monitor.alerts
    ]

    report = {
        'generated_at': datetime.now().isoformat(),
        'dashboard': dashboard_data,
        'alerts': all_alerts,
        'drift_history': [
            {
                'timestamp': m.timestamp,
                'prediction_mean': m.prediction_mean,
                'prediction_std': m.prediction_std,
                'confidence_mean': m.confidence_mean,
                'feature_drift_score': m.feature_drift_score,
                'performance_degradation': m.performance_degradation,
                'alert_triggered': m.alert_triggered,
            }
            for m in monitor.drift_history
        ],
    }

    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Monitoring report saved to {output_path}")
    return output_path


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Monitoring System Module")
    print("=" * 60)

    # Create test config
    test_config = {
        'paths': {'log_dir': '/tmp/ml_engine_monitoring_test'},
        'monitoring': {
            'alert_thresholds': {
                'daily_loss_threshold': 0.05,
                'max_drawdown_threshold': 0.10,
            }
        }
    }

    # Initialize monitoring system
    monitor = MonitoringSystem(test_config)

    # Test daily performance check
    print("\nTesting daily performance check...")
    test_trades = [
        {'pnl': 100},
        {'pnl': -50},
        {'pnl': 200},
        {'pnl': -30},
        {'pnl': -20},
    ]
    monitor.check_daily_performance(
        pnl=200,
        trades=test_trades,
        starting_balance=10000,
    )

    # Test model drift check
    print("\nTesting model drift check...")
    predictions = np.random.rand(100) * 0.5 + 0.3  # Mean around 0.55
    confidences = np.random.rand(100) * 0.2 + 0.6  # Mean around 0.70
    drift_metrics = monitor.check_model_drift(predictions, confidences)
    print(f"  Drift score: {drift_metrics.feature_drift_score:.3f}")
    print(f"  Confidence mean: {drift_metrics.confidence_mean:.3f}")

    # Test alerts
    print(f"\nTotal alerts: {len(monitor.alerts)}")
    for alert in monitor.alerts:
        print(f"  [{alert.level.value}] {alert.category}: {alert.message}")

    # Test dashboard data
    print("\nDashboard data:")
    dashboard = monitor.get_dashboard_data()
    print(f"  Alerts: {dashboard['alerts']}")
    print(f"  Recent drift checks: {len(dashboard['recent_drift'])}")

    # Export report
    report_path = create_monitoring_report(monitor)
    print(f"\nReport exported to: {report_path}")

    print("\n✓ Monitoring system module ready")
