"""
Tests for monitoring and alerting system.
"""

import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.monitoring import (
    MonitoringSystem,
    Alert,
    AlertLevel,
    PerformanceMetrics,
    ModelDriftMetrics,
    create_monitoring_report,
)


class TestMonitoringSystem(unittest.TestCase):
    """Test monitoring system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.config = {
            'paths': {'log_dir': self.temp_dir.name},
            'monitoring': {
                'alert_thresholds': {
                    'daily_loss_threshold': 0.05,
                    'max_drawdown_threshold': 0.10,
                    'consecutive_losses_max': 5,
                }
            }
        }
        self.monitor = MonitoringSystem(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test monitoring system initialization."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(len(self.monitor.alerts), 0)
        self.assertIsNotNone(self.monitor.alert_thresholds)
    
    def test_add_alert(self):
        """Test adding alerts."""
        self.monitor.add_alert(
            AlertLevel.WARNING,
            "test_category",
            "Test alert message",
            {'key': 'value'}
        )
        
        self.assertEqual(len(self.monitor.alerts), 1)
        alert = self.monitor.alerts[0]
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.category, "test_category")
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.data['key'], 'value')
    
    def test_get_alerts_filtered(self):
        """Test filtering alerts by level and category."""
        self.monitor.add_alert(AlertLevel.INFO, "cat1", "Info 1")
        self.monitor.add_alert(AlertLevel.WARNING, "cat1", "Warning 1")
        self.monitor.add_alert(AlertLevel.CRITICAL, "cat2", "Critical 1")
        
        # Filter by level
        warnings = self.monitor.get_alerts(level=AlertLevel.WARNING)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].level, AlertLevel.WARNING)
        
        # Filter by category
        cat1_alerts = self.monitor.get_alerts(category="cat1")
        self.assertEqual(len(cat1_alerts), 2)
    
    def test_clear_alerts(self):
        """Test clearing alerts."""
        self.monitor.add_alert(AlertLevel.INFO, "test", "Test")
        self.assertEqual(len(self.monitor.alerts), 1)
        
        self.monitor.clear_alerts()
        self.assertEqual(len(self.monitor.alerts), 0)
    
    def test_daily_performance_check_normal(self):
        """Test daily performance check with normal performance."""
        trades = [
            {'pnl': 100},
            {'pnl': 50},
            {'pnl': -30},
            {'pnl': 80},
        ]
        
        self.monitor.check_daily_performance(
            pnl=200,
            trades=trades,
            starting_balance=10000,
        )
        
        # Should not trigger alerts for good performance
        critical_alerts = self.monitor.get_alerts(level=AlertLevel.CRITICAL)
        self.assertEqual(len(critical_alerts), 0)
    
    def test_daily_performance_check_loss_threshold(self):
        """Test daily performance check triggers alert on loss threshold."""
        trades = [{'pnl': -600}]
        
        self.monitor.check_daily_performance(
            pnl=-600,
            trades=trades,
            starting_balance=10000,
        )
        
        # Should trigger daily loss alert (6% > 5% threshold)
        alerts = self.monitor.get_alerts(category="daily_loss")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].level, AlertLevel.CRITICAL)
    
    def test_daily_performance_check_consecutive_losses(self):
        """Test consecutive loss detection."""
        trades = [
            {'pnl': -10},
            {'pnl': -20},
            {'pnl': -15},
            {'pnl': -25},
            {'pnl': -30},
            {'pnl': -5},  # 6 consecutive losses
        ]
        
        self.monitor.check_daily_performance(
            pnl=-105,
            trades=trades,
            starting_balance=10000,
        )
        
        # Should trigger consecutive losses alert
        alerts = self.monitor.get_alerts(category="consecutive_losses")
        self.assertEqual(len(alerts), 1)
    
    def test_model_drift_check(self):
        """Test model drift detection."""
        predictions = np.random.rand(100) * 0.4 + 0.3  # Mean ~0.5
        confidences = np.random.rand(100) * 0.2 + 0.6  # Mean ~0.7
        
        metrics = self.monitor.check_model_drift(predictions, confidences)
        
        self.assertIsInstance(metrics, ModelDriftMetrics)
        self.assertGreater(metrics.prediction_mean, 0)
        self.assertGreater(metrics.confidence_mean, 0)
        self.assertEqual(len(self.monitor.drift_history), 1)
    
    def test_baseline_metrics_save_load(self):
        """Test saving and loading baseline metrics."""
        metrics = PerformanceMetrics(
            date='2025-01-01',
            total_pnl=1000.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.60,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            avg_trade_duration_hours=2.5,
            largest_win=200.0,
            largest_loss=-100.0,
        )
        
        self.monitor.save_baseline_metrics(metrics)
        
        # Create new monitor to test loading
        new_monitor = MonitoringSystem(self.config)
        self.assertIsNotNone(new_monitor.baseline_metrics)
        self.assertEqual(new_monitor.baseline_metrics.win_rate, 0.60)
        self.assertEqual(new_monitor.baseline_metrics.sharpe_ratio, 1.5)
    
    def test_export_alerts(self):
        """Test exporting alerts to JSON."""
        self.monitor.add_alert(AlertLevel.WARNING, "test", "Test alert")
        
        export_path = self.monitor.export_alerts()
        
        self.assertTrue(export_path.exists())
        
        # Verify content
        data = json.loads(export_path.read_text())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['category'], 'test')
    
    def test_dashboard_data(self):
        """Test getting dashboard data."""
        self.monitor.add_alert(AlertLevel.CRITICAL, "test", "Critical alert")
        self.monitor.add_alert(AlertLevel.WARNING, "test", "Warning alert")
        
        data = self.monitor.get_dashboard_data()
        
        self.assertIn('timestamp', data)
        self.assertIn('alerts', data)
        self.assertEqual(data['alerts']['total'], 2)
        self.assertEqual(data['alerts']['critical'], 1)
        self.assertEqual(data['alerts']['warning'], 1)
    
    def test_create_monitoring_report(self):
        """Test creating full monitoring report."""
        self.monitor.add_alert(AlertLevel.INFO, "test", "Test alert")
        
        predictions = np.random.rand(50)
        confidences = np.random.rand(50) * 0.2 + 0.6
        self.monitor.check_model_drift(predictions, confidences)
        
        report_path = create_monitoring_report(self.monitor)
        
        self.assertTrue(report_path.exists())
        
        # Verify report content
        report = json.loads(report_path.read_text())
        self.assertIn('generated_at', report)
        self.assertIn('dashboard', report)
        self.assertIn('alerts', report)
        self.assertIn('drift_history', report)


class TestAlertLevels(unittest.TestCase):
    """Test alert level enum."""
    
    def test_alert_levels(self):
        """Test alert level values."""
        self.assertEqual(AlertLevel.INFO.value, "info")
        self.assertEqual(AlertLevel.WARNING.value, "warning")
        self.assertEqual(AlertLevel.CRITICAL.value, "critical")


if __name__ == '__main__':
    unittest.main()
