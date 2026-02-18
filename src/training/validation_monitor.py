"""
Validation Monitoring System for Financial ML Ensemble.

Tracks validation metrics over time, detects degradation trends,
and provides early warning alerts before validation gates fail.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationSnapshot:
    """Single validation measurement snapshot."""
    timestamp: str
    model_version: str
    metrics: Dict[str, float]
    passed: bool
    failed_checks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'model_version': self.model_version,
            'metrics': self.metrics,
            'passed': self.passed,
            'failed_checks': self.failed_checks,
        }


@dataclass
class DegradationAlert:
    """Alert for detected degradation."""
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'degradation_pct': self.degradation_pct,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp,
        }


class ValidationMonitor:
    """Monitors validation metrics over time and detects degradation."""
    
    def __init__(
        self,
        history_path: Optional[Path] = None,
        warning_threshold_pct: float = 0.10,
        critical_threshold_pct: float = 0.20,
        trend_window: int = 5,
    ):
        """Initialize validation monitor.
        
        Args:
            history_path: Path to save validation history
            warning_threshold_pct: % degradation for warning alert
            critical_threshold_pct: % degradation for critical alert
            trend_window: Number of snapshots for trend analysis
        """
        self.history_path = history_path or Path("trained_data/validation_history.json")
        self.warning_threshold_pct = warning_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.trend_window = trend_window
        self.history: List[ValidationSnapshot] = []
        self.baselines: Dict[str, float] = {}
        
        self._load_history()
    
    def _load_history(self) -> None:
        """Load validation history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    self.history = [
                        ValidationSnapshot(**snapshot) 
                        for snapshot in data.get('snapshots', [])
                    ]
                    self.baselines = data.get('baselines', {})
                logger.info(f"Loaded {len(self.history)} validation snapshots")
            except Exception as e:
                logger.warning(f"Failed to load validation history: {e}")
    
    def _save_history(self) -> None:
        """Save validation history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'snapshots': [s.to_dict() for s in self.history],
            'baselines': self.baselines,
        }
        with open(self.history_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved validation history to {self.history_path}")
    
    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set baseline value for a metric.
        
        Args:
            metric_name: Name of the metric
            value: Baseline value
        """
        self.baselines[metric_name] = value
        self._save_history()
        logger.info(f"Set baseline for {metric_name}: {value:.4f}")
    
    def record_validation(
        self,
        model_version: str,
        metrics: Dict[str, float],
        passed: bool,
        failed_checks: Optional[List[str]] = None,
    ) -> ValidationSnapshot:
        """Record a validation result.
        
        Args:
            model_version: Model version identifier
            metrics: Dictionary of metric name → value
            passed: Whether validation passed
            failed_checks: List of failed check names
            
        Returns:
            Created ValidationSnapshot
        """
        snapshot = ValidationSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            model_version=model_version,
            metrics=metrics,
            passed=passed,
            failed_checks=failed_checks or [],
        )
        
        self.history.append(snapshot)
        self._save_history()
        
        logger.info(f"Recorded validation for {model_version}: passed={passed}")
        return snapshot
    
    def check_degradation(
        self,
        metrics: Dict[str, float]
    ) -> List[DegradationAlert]:
        """Check metrics for degradation against baselines.
        
        Args:
            metrics: Current metrics to check
            
        Returns:
            List of DegradationAlert for any detected issues
        """
        alerts = []
        timestamp = datetime.utcnow().isoformat()
        
        for metric_name, current_value in metrics.items():
            if metric_name not in self.baselines:
                continue
            
            baseline = self.baselines[metric_name]
            
            # Calculate degradation percentage
            if baseline != 0:
                degradation_pct = abs(current_value - baseline) / abs(baseline)
            else:
                degradation_pct = 0.0 if current_value == 0 else 1.0
            
            # Determine severity
            if degradation_pct >= self.critical_threshold_pct:
                severity = 'critical'
            elif degradation_pct >= self.warning_threshold_pct:
                severity = 'warning'
            else:
                continue  # No alert needed
            
            # Determine if this is improvement or degradation
            is_improvement = (
                (metric_name.startswith('min_') and current_value > baseline) or
                (metric_name.startswith('max_') and current_value < baseline)
            )
            
            if not is_improvement:
                alert = DegradationAlert(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline,
                    degradation_pct=degradation_pct,
                    severity=severity,
                    message=f"{metric_name} degraded by {degradation_pct:.1%}: {baseline:.4f} → {current_value:.4f}",
                    timestamp=timestamp,
                )
                alerts.append(alert)
                
                if severity == 'critical':
                    logger.error(alert.message)
                else:
                    logger.warning(alert.message)
        
        return alerts
    
    def get_trend(self, metric_name: str) -> Tuple[str, float]:
        """Analyze trend for a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            
        Returns:
            Tuple of (trend_direction, slope)
            trend_direction: 'improving', 'degrading', 'stable'
        """
        if len(self.history) < 2:
            return ('stable', 0.0)
        
        # Get recent values
        recent = self.history[-self.trend_window:]
        values = [h.metrics.get(metric_name, 0) for h in recent if metric_name in h.metrics]
        
        if len(values) < 2:
            return ('stable', 0.0)
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            return ('stable', slope)
        elif (metric_name.startswith('min_') and slope > 0) or \
             (metric_name.startswith('max_') and slope < 0):
            return ('improving', slope)
        else:
            return ('degrading', slope)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation history.
        
        Returns:
            Summary dictionary with statistics
        """
        if not self.history:
            return {'total_validations': 0}
        
        recent = self.history[-self.trend_window:]
        pass_rate = sum(1 for h in recent if h.passed) / len(recent) if recent else 0
        
        # Get trends for key metrics
        trends = {}
        for metric in ['min_direction_accuracy', 'max_cv_degradation', 'min_bootstrap_ci_lower']:
            direction, slope = self.get_trend(metric)
            trends[metric] = {'direction': direction, 'slope': slope}
        
        return {
            'total_validations': len(self.history),
            'recent_pass_rate': pass_rate,
            'last_validation': self.history[-1].timestamp if self.history else None,
            'baselines': self.baselines,
            'trends': trends,
        }
    
    def generate_report(self) -> str:
        """Generate human-readable validation report.
        
        Returns:
            Formatted report string
        """
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "VALIDATION MONITORING REPORT",
            "=" * 60,
            f"Total Validations: {summary['total_validations']}",
            f"Recent Pass Rate: {summary['recent_pass_rate']:.1%}",
            f"Last Validation: {summary.get('last_validation', 'N/A')}",
            "",
            "BASELINES:",
        ]
        
        for metric, value in summary.get('baselines', {}).items():
            trend = summary.get('trends', {}).get(metric, {})
            trend_str = f" ({trend.get('direction', 'unknown')})"
            lines.append(f"  {metric}: {value:.4f}{trend_str}")
        
        if self.history:
            latest = self.history[-1]
            lines.extend([
                "",
                "LATEST VALIDATION:",
                f"  Model Version: {latest.model_version}",
                f"  Passed: {latest.passed}",
                f"  Failed Checks: {', '.join(latest.failed_checks) or 'None'}",
                "",
                "METRICS:",
            ])
            for metric, value in latest.metrics.items():
                lines.append(f"  {metric}: {value:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
