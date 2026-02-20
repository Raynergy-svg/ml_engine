"""
Training Stability Analyzer - MLOps Tool for Crash Detection and Optimal Performance Identification

This module provides comprehensive analysis of training logs to identify:
1. Training divergence/crash points
2. Peak stable performance before crashes
3. Optimal hyperparameters at stability
4. Root cause analysis of training failures

Usage:
    from src.training.training_stability_analyzer import TrainingStabilityAnalyzer
    
    analyzer = TrainingStabilityAnalyzer()
    report = analyzer.analyze_wandb_run("path/to/wandb/run")
    print(report.to_markdown())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CrashPoint:
    """Represents a detected training crash/divergence point."""
    step: int
    epoch: Optional[int]
    crash_type: str  # "loss_spike", "accuracy_drop", "nan_detected", "infinity_detected"
    severity: str  # "critical", "warning", "minor"
    trigger_metric: str
    trigger_value: float
    previous_stable_value: float
    description: str


@dataclass
class OptimalMetrics:
    """Optimal metrics achieved before crash."""
    step: int
    epoch: Optional[int]
    learning_rate: float
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    gradient_norm: Optional[float]
    is_stable: bool  # Verified by surrounding steps


@dataclass
class StabilityReport:
    """Comprehensive training stability analysis report."""
    run_id: str
    run_name: str
    total_steps: int
    total_epochs: int
    crash_points: List[CrashPoint]
    optimal_metrics: Optional[OptimalMetrics]
    stability_score: float  # 0-100, higher = more stable
    recommendations: List[str]
    analysis_timestamp: str
    
    def to_markdown(self) -> str:
        """Generate formatted Markdown report."""
        lines = [
            "# Training Stability Report",
            "",
            f"**Run ID:** `{self.run_id}`",
            f"**Run Name:** {self.run_name}",
            f"**Analysis Time:** {self.analysis_timestamp}",
            f"**Stability Score:** {self.stability_score:.1f}/100",
            "",
            "---",
            "",
            "## 📊 Stability Summary",
            "",
            f"- **Total Steps:** {self.total_steps}",
            f"- **Total Epochs:** {self.total_epochs}",
            f"- **Crash Points Detected:** {len(self.crash_points)}",
            "",
        ]
        
        # Crash points section
        if self.crash_points:
            lines.extend([
                "## 🚨 Crash Analysis",
                "",
                "| Step | Epoch | Type | Severity | Trigger | Value | Previous |",
                "|------|-------|------|----------|---------|-------|----------|",
            ])
            for cp in self.crash_points:
                lines.append(
                    f"| {cp.step} | {cp.epoch or 'N/A'} | {cp.crash_type} | "
                    f"{cp.severity} | {cp.trigger_metric} | {cp.trigger_value:.4f} | "
                    f"{cp.previous_stable_value:.4f} |"
                )
            lines.append("")
            
            # Detailed crash descriptions
            lines.extend(["### Crash Details", ""])
            for i, cp in enumerate(self.crash_points, 1):
                lines.extend([
                    f"**Crash #{i} (Step {cp.step}):**",
                    f"- **Type:** {cp.crash_type}",
                    f"- **Severity:** {cp.severity}",
                    f"- **Description:** {cp.description}",
                    "",
                ])
        else:
            lines.extend([
                "## ✅ No Crashes Detected",
                "",
                "Training completed without significant divergence.",
                "",
            ])
        
        # Optimal metrics section
        if self.optimal_metrics:
            om = self.optimal_metrics
            lines.extend([
                "## 🎯 Optimal Metrics (Peak Stable Performance)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| **Step** | {om.step} |",
                f"| **Epoch** | {om.epoch or 'N/A'} |",
                f"| **Learning Rate** | {om.learning_rate:.2e} |",
                f"| **Train Loss** | {om.train_loss:.4f} |",
                f"| **Train Accuracy** | {om.train_accuracy:.2%} |",
            ])
            if om.val_loss is not None:
                lines.append(f"| **Val Loss** | {om.val_loss:.4f} |")
            if om.val_accuracy is not None:
                lines.append(f"| **Val Accuracy** | {om.val_accuracy:.2%} |")
            if om.gradient_norm is not None:
                lines.append(f"| **Gradient Norm** | {om.gradient_norm:.4f} |")
            lines.append(f"| **Verified Stable** | {'✅ Yes' if om.is_stable else '⚠️ No'} |")
            lines.append("")
        
        # Recommendations section
        if self.recommendations:
            lines.extend([
                "## 💡 Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class TrainingStabilityAnalyzer:
    """
    Analyze training logs for stability, crash detection, and optimal performance.
    
    This analyzer implements the step-by-step reasoning process:
    1. Preprocessing: Clean and sort data, handle missing values
    2. Anomaly Detection: Identify first instance of training instability
    3. Optimal Window Extraction: Filter to steps preceding crash
    4. Metric Optimization: Find maximum accuracy and corresponding LR
    5. Reflexion Check: Verify optimal step was not an outlier
    
    Attributes:
        loss_spike_threshold: Multiplier for detecting loss spikes
        accuracy_drop_threshold: Percentage drop for accuracy collapse
        stability_window: Number of steps to check for stability verification
    """
    
    def __init__(
        self,
        loss_spike_threshold: float = 2.0,
        accuracy_drop_threshold: float = 0.20,
        stability_window: int = 5,
        nan_detection: bool = True,
        infinity_detection: bool = True,
    ):
        """
        Initialize the stability analyzer.
        
        Args:
            loss_spike_threshold: Loss increase ratio to trigger crash detection
            accuracy_drop_threshold: Accuracy drop fraction (0.20 = 20% drop)
            stability_window: Steps before/after to verify stability
            nan_detection: Enable NaN value detection
            infinity_detection: Enable Infinity value detection
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.stability_window = stability_window
        self.nan_detection = nan_detection
        self.infinity_detection = infinity_detection
    
    def analyze_wandb_run(self, run_path: str) -> StabilityReport:
        """
        Analyze a wandb run directory for training stability.
        
        Args:
            run_path: Path to wandb run directory
            
        Returns:
            StabilityReport with crash analysis and optimal metrics
        """
        run_path = Path(run_path)
        
        # Load summary and config
        summary = self._load_wandb_summary(run_path)
        config = self._load_wandb_config(run_path)
        
        # Extract run metadata
        run_id = run_path.name.split("-")[-1] if "-" in run_path.name else run_path.name
        run_name = config.get("experiment_name", {}).get("value", run_id)
        
        # Get total steps/epochs
        total_steps = summary.get("_step", 0)
        total_epochs = config.get("num_epochs", {}).get("value", 0)
        
        # Detect crashes from summary metrics
        crash_points = self._detect_crashes_from_summary(summary, config)
        
        # Find optimal metrics
        optimal_metrics = self._find_optimal_metrics_from_summary(summary, config)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            total_steps, crash_points, optimal_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(crash_points, config, summary)
        
        return StabilityReport(
            run_id=run_id,
            run_name=run_name,
            total_steps=total_steps,
            total_epochs=total_epochs,
            crash_points=crash_points,
            optimal_metrics=optimal_metrics,
            stability_score=stability_score,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat(),
        )
    
    def analyze_metrics_dict(
        self,
        metrics: Dict[str, List[float]],
        config: Optional[Dict[str, Any]] = None,
        run_id: str = "unknown",
        run_name: str = "Unknown Run",
    ) -> StabilityReport:
        """
        Analyze training metrics from a dictionary.
        
        Expected keys:
            - 'step' or 'steps': List of step numbers
            - 'train_loss' or 'loss': Training loss values
            - 'train_accuracy' or 'accuracy': Training accuracy values
            - 'val_loss' (optional): Validation loss values
            - 'val_accuracy' (optional): Validation accuracy values
            - 'learning_rate' or 'lr' (optional): Learning rate values
        
        Args:
            metrics: Dictionary of metric lists
            config: Optional configuration dictionary
            run_id: Run identifier
            run_name: Human-readable run name
            
        Returns:
            StabilityReport with analysis results
        """
        config = config or {}
        
        # Normalize metric names
        steps = metrics.get("step", metrics.get("steps", list(range(len(metrics.get("train_loss", metrics.get("loss", [])))))))
        train_loss = metrics.get("train_loss", metrics.get("loss", []))
        train_accuracy = metrics.get("train_accuracy", metrics.get("accuracy", []))
        val_loss = metrics.get("val_loss", [])
        val_accuracy = metrics.get("val_accuracy", [])
        learning_rate = metrics.get("learning_rate", metrics.get("lr", []))
        
        # Detect crashes
        crash_points = self._detect_crashes_from_lists(
            steps, train_loss, train_accuracy, val_loss, val_accuracy
        )
        
        # Find optimal metrics
        optimal_metrics = self._find_optimal_metrics_from_lists(
            steps, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate
        )
        
        # Calculate stability score
        total_steps = len(steps)
        stability_score = self._calculate_stability_score(
            total_steps, crash_points, optimal_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations_from_metrics(
            crash_points, config, train_loss, train_accuracy
        )
        
        return StabilityReport(
            run_id=run_id,
            run_name=run_name,
            total_steps=total_steps,
            total_epochs=max(steps) // 100 if steps else 0,  # Estimate
            crash_points=crash_points,
            optimal_metrics=optimal_metrics,
            stability_score=stability_score,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat(),
        )
    
    def _load_wandb_summary(self, run_path: Path) -> Dict[str, Any]:
        """Load wandb-summary.json from run directory."""
        summary_path = run_path / "files" / "wandb-summary.json"
        if not summary_path.exists():
            summary_path = run_path / "wandb-summary.json"
        
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)
        return {}
    
    def _load_wandb_config(self, run_path: Path) -> Dict[str, Any]:
        """Load config.yaml from wandb run directory."""
        import yaml
        
        config_path = run_path / "files" / "config.yaml"
        if not config_path.exists():
            config_path = run_path / "config.yaml"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _detect_crashes_from_summary(
        self,
        summary: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[CrashPoint]:
        """Detect crashes from summary metrics."""
        crashes = []
        
        # Check for NaN/Infinity in final metrics
        train_loss = summary.get("train/loss", summary.get("train_loss"))
        val_loss = summary.get("val/loss", summary.get("val_loss"))
        train_acc = summary.get("train/accuracy", summary.get("train_accuracy"))
        val_acc = summary.get("val/accuracy", summary.get("val_accuracy"))
        
        step = summary.get("_step", 0)
        
        # NaN detection
        if self.nan_detection:
            for metric_name, value in [
                ("train/loss", train_loss),
                ("val/loss", val_loss),
                ("train/accuracy", train_acc),
                ("val/accuracy", val_acc),
            ]:
                if value is not None and np.isnan(value):
                    crashes.append(CrashPoint(
                        step=step,
                        epoch=None,
                        crash_type="nan_detected",
                        severity="critical",
                        trigger_metric=metric_name,
                        trigger_value=value,
                        previous_stable_value=0.0,
                        description=f"NaN detected in {metric_name} at step {step}. "
                                   f"This indicates numerical instability in training.",
                    ))
        
        # Infinity detection
        if self.infinity_detection:
            for metric_name, value in [
                ("train/loss", train_loss),
                ("val/loss", val_loss),
            ]:
                if value is not None and np.isinf(value):
                    crashes.append(CrashPoint(
                        step=step,
                        epoch=None,
                        crash_type="infinity_detected",
                        severity="critical",
                        trigger_metric=metric_name,
                        trigger_value=value,
                        previous_stable_value=0.0,
                        description=f"Infinity detected in {metric_name} at step {step}. "
                                   f"Training has diverged completely.",
                    ))
        
        # Check for high loss values (potential divergence)
        lr = config.get("learning_rate", {}).get("value", 0.001)
        if train_loss is not None and train_loss > 10.0:
            crashes.append(CrashPoint(
                step=step,
                epoch=None,
                crash_type="loss_spike",
                severity="warning",
                trigger_metric="train/loss",
                trigger_value=train_loss,
                previous_stable_value=1.0,
                description=f"High loss value ({train_loss:.2f}) detected. "
                           f"Learning rate ({lr}) may be too high.",
            ))
        
        # Check for low accuracy (potential collapse)
        if train_acc is not None and train_acc < 0.3:
            crashes.append(CrashPoint(
                step=step,
                epoch=None,
                crash_type="accuracy_drop",
                severity="warning",
                trigger_metric="train/accuracy",
                trigger_value=train_acc,
                previous_stable_value=0.5,
                description=f"Low accuracy ({train_acc:.2%}) suggests model collapse. "
                           f"Consider reducing learning rate or checking data quality.",
            ))
        
        return crashes
    
    def _detect_crashes_from_lists(
        self,
        steps: List[int],
        train_loss: List[float],
        train_accuracy: List[float],
        val_loss: List[float],
        val_accuracy: List[float],
    ) -> List[CrashPoint]:
        """Detect crashes from step-wise metric lists."""
        crashes = []
        
        if not train_loss:
            return crashes
        
        # Calculate rolling statistics for anomaly detection
        window_size = min(self.stability_window, len(train_loss))
        
        for i in range(window_size, len(train_loss)):
            # Get rolling statistics
            recent_losses = train_loss[i-window_size:i]
            mean_loss = np.mean(recent_losses)
            current_loss = train_loss[i]
            
            # Check for loss spike
            if mean_loss > 0 and current_loss > mean_loss * self.loss_spike_threshold:
                crashes.append(CrashPoint(
                    step=steps[i] if i < len(steps) else i,
                    epoch=None,
                    crash_type="loss_spike",
                    severity="critical" if current_loss > mean_loss * 3 else "warning",
                    trigger_metric="train_loss",
                    trigger_value=current_loss,
                    previous_stable_value=mean_loss,
                    description=f"Loss spike detected: {current_loss:.4f} vs "
                               f"rolling mean {mean_loss:.4f} ({current_loss/mean_loss:.1f}x increase).",
                ))
                break  # Report first crash only
            
            # Check for NaN
            if self.nan_detection and np.isnan(current_loss):
                crashes.append(CrashPoint(
                    step=steps[i] if i < len(steps) else i,
                    epoch=None,
                    crash_type="nan_detected",
                    severity="critical",
                    trigger_metric="train_loss",
                    trigger_value=current_loss,
                    previous_stable_value=mean_loss,
                    description=f"NaN detected in training loss at step {steps[i] if i < len(steps) else i}.",
                ))
                break
            
            # Check for Infinity
            if self.infinity_detection and np.isinf(current_loss):
                crashes.append(CrashPoint(
                    step=steps[i] if i < len(steps) else i,
                    epoch=None,
                    crash_type="infinity_detected",
                    severity="critical",
                    trigger_metric="train_loss",
                    trigger_value=current_loss,
                    previous_stable_value=mean_loss,
                    description=f"Infinity detected in training loss at step {steps[i] if i < len(steps) else i}.",
                ))
                break
        
        # Check for accuracy collapse
        if train_accuracy:
            for i in range(window_size, len(train_accuracy)):
                recent_acc = train_accuracy[i-window_size:i]
                mean_acc = np.mean(recent_acc)
                current_acc = train_accuracy[i]
                
                if mean_acc > 0.1 and current_acc < mean_acc * (1 - self.accuracy_drop_threshold):
                    crashes.append(CrashPoint(
                        step=steps[i] if i < len(steps) else i,
                        epoch=None,
                        crash_type="accuracy_drop",
                        severity="warning",
                        trigger_metric="train_accuracy",
                        trigger_value=current_acc,
                        previous_stable_value=mean_acc,
                        description=f"Accuracy collapse: {current_acc:.2%} vs "
                                   f"rolling mean {mean_acc:.2%} "
                                   f"({(mean_acc - current_acc)/mean_acc:.0%} drop).",
                    ))
                    break
        
        return crashes
    
    def _find_optimal_metrics_from_summary(
        self,
        summary: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Optional[OptimalMetrics]:
        """Extract optimal metrics from summary."""
        try:
            return OptimalMetrics(
                step=summary.get("_step", 0),
                epoch=None,
                learning_rate=config.get("learning_rate", {}).get("value", 0.001),
                train_loss=summary.get("train/loss", summary.get("train_loss", 0.0)),
                train_accuracy=summary.get("train/accuracy", summary.get("train_accuracy", 0.0)),
                val_loss=summary.get("val/loss", summary.get("val_loss")),
                val_accuracy=summary.get("val/accuracy", summary.get("val_accuracy")),
                gradient_norm=None,
                is_stable=True,  # Assume stable for summary-only analysis
            )
        except Exception as e:
            logger.warning(f"Could not extract optimal metrics: {e}")
            return None
    
    def _find_optimal_metrics_from_lists(
        self,
        steps: List[int],
        train_loss: List[float],
        train_accuracy: List[float],
        val_loss: List[float],
        val_accuracy: List[float],
        learning_rate: List[float],
    ) -> Optional[OptimalMetrics]:
        """Find optimal metrics from step-wise data."""
        if not train_accuracy:
            return None
        
        # Find step with best accuracy
        best_idx = np.argmax(train_accuracy)
        
        # Verify stability (check surrounding steps)
        is_stable = True
        if best_idx >= self.stability_window:
            prev_acc = train_accuracy[best_idx - self.stability_window:best_idx]
            if prev_acc:
                prev_mean = np.mean(prev_acc)
                # Stable if within 5% of best
                is_stable = abs(train_accuracy[best_idx] - prev_mean) < 0.05
        
        return OptimalMetrics(
            step=steps[best_idx] if best_idx < len(steps) else best_idx,
            epoch=None,
            learning_rate=learning_rate[best_idx] if best_idx < len(learning_rate) else 0.001,
            train_loss=train_loss[best_idx] if best_idx < len(train_loss) else 0.0,
            train_accuracy=train_accuracy[best_idx],
            val_loss=val_loss[best_idx] if best_idx < len(val_loss) else None,
            val_accuracy=val_accuracy[best_idx] if best_idx < len(val_accuracy) else None,
            gradient_norm=None,
            is_stable=is_stable,
        )
    
    def _calculate_stability_score(
        self,
        total_steps: int,
        crash_points: List[CrashPoint],
        optimal_metrics: Optional[OptimalMetrics],
    ) -> float:
        """Calculate overall training stability score (0-100)."""
        score = 100.0
        
        # Deduct for each crash
        for crash in crash_points:
            if crash.severity == "critical":
                score -= 30
            elif crash.severity == "warning":
                score -= 15
            else:
                score -= 5
        
        # Bonus for good accuracy
        if optimal_metrics and optimal_metrics.train_accuracy > 0.6:
            score += 10
        if optimal_metrics and optimal_metrics.val_accuracy and optimal_metrics.val_accuracy > 0.55:
            score += 10
        
        # Penalty for low accuracy
        if optimal_metrics and optimal_metrics.train_accuracy < 0.4:
            score -= 20
        
        return max(0, min(100, score))
    
    def _generate_recommendations(
        self,
        crash_points: List[CrashPoint],
        config: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        lr = config.get("learning_rate", {}).get("value", 0.001)
        
        for crash in crash_points:
            if crash.crash_type == "nan_detected":
                recommendations.extend([
                    "**Critical:** NaN detected - reduce learning rate by 10x",
                    "Enable gradient clipping (gradient_clip_val=1.0)",
                    "Check for data quality issues (infinite values, NaNs in input)",
                    "Consider using mixed precision training with loss scaling",
                ])
            
            elif crash.crash_type == "infinity_detected":
                recommendations.extend([
                    "**Critical:** Infinity detected - training has diverged",
                    f"Reduce learning rate from {lr:.2e} to {lr/10:.2e}",
                    "Enable gradient clipping immediately",
                    "Check model architecture for numerical instability",
                ])
            
            elif crash.crash_type == "loss_spike":
                spike_ratio = crash.trigger_value / crash.previous_stable_value if crash.previous_stable_value > 0 else 0
                recommendations.extend([
                    f"Loss spike detected ({spike_ratio:.1f}x increase)",
                    f"Consider reducing learning rate from {lr:.2e}",
                    "Add learning rate warmup for first 5-10% of training",
                    "Monitor gradient norms for explosion",
                ])
            
            elif crash.crash_type == "accuracy_drop":
                recommendations.extend([
                    "Accuracy collapse detected - model may be overfitting or underfitting",
                    "Increase regularization (weight_decay, dropout)",
                    "Check for label noise or data corruption",
                    "Consider early stopping with patience=10",
                ])
        
        # General recommendations based on config
        if lr > 0.01:
            recommendations.append(
                f"Learning rate ({lr:.2e}) is high - consider 1e-4 to 1e-3 range"
            )
        
        if not crash_points:
            if summary.get("train/accuracy", 0) > 0.65:
                recommendations.append(
                    "✅ Training appears stable with good accuracy - "
                    "consider this configuration for production"
                )
            else:
                recommendations.append(
                    "Training stable but accuracy could improve - "
                    "try increasing model capacity or training longer"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_recommendations_from_metrics(
        self,
        crash_points: List[CrashPoint],
        config: Dict[str, Any],
        train_loss: List[float],
        train_accuracy: List[float],
    ) -> List[str]:
        """Generate recommendations from metric lists."""
        recommendations = []
        
        # Use base recommendations
        summary = {
            "train/accuracy": train_accuracy[-1] if train_accuracy else 0,
        }
        recommendations = self._generate_recommendations(crash_points, config, summary)
        
        # Add trend-based recommendations
        if len(train_loss) > 10:
            loss_trend = np.polyfit(range(len(train_loss)), train_loss, 1)[0]
            if loss_trend > 0:
                recommendations.append(
                    "⚠️ Loss trend is increasing - model may be overfitting. "
                    "Consider early stopping or more regularization."
                )
        
        return recommendations


# =============================================================================
# CLI UTILITY
# =============================================================================

def analyze_wandb_runs(base_path: str = "wandb") -> None:
    """
    Analyze all wandb runs in a directory and print reports.
    
    Args:
        base_path: Path to wandb directory
    """
    analyzer = TrainingStabilityAnalyzer()
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return
    
    # Find all run directories
    run_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and (item.name.startswith("run-") or item.name == "latest-run"):
            run_dirs.append(item)
    
    if not run_dirs:
        print(f"No wandb runs found in {base_path}")
        return
    
    print(f"Found {len(run_dirs)} wandb runs\n")
    
    for run_dir in sorted(run_dirs):
        print("=" * 80)
        try:
            report = analyzer.analyze_wandb_run(str(run_dir))
            print(report.to_markdown())
        except Exception as e:
            print(f"Error analyzing {run_dir.name}: {e}")
        print()


def analyze_training_csv(csv_path: str) -> StabilityReport:
    """
    Analyze a training history CSV file for crash detection.
    
    Args:
        csv_path: Path to training_history.csv file
        
    Returns:
        StabilityReport with analysis results
    """
    import pandas as pd
    from pathlib import Path
    
    df = pd.read_csv(csv_path)
    
    # Detect crash point (first NaN in loss)
    crash_epoch = None
    for i, row in df.iterrows():
        loss_val = row.get('loss', row.get('train_loss', 0))
        if pd.isna(loss_val) or str(loss_val).lower() == 'nan':
            crash_epoch = int(row.get('epoch', i))
            break
    
    # Get metrics lists
    epochs = df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df)))
    
    # Handle different column naming conventions
    train_loss = df.get('loss', df.get('train_loss', df.get('train/loss', [0]))).tolist()
    val_loss = df.get('val_loss', df.get('val/loss', [None])).tolist()
    
    if 'direction_dir_acc' in df.columns:
        train_acc = df['direction_dir_acc'].tolist()
        val_acc = df.get('val_direction_dir_acc', [None]).tolist()
    else:
        train_acc = df.get('accuracy', df.get('train_accuracy', [0])).tolist()
        val_acc = df.get('val_accuracy', [None]).tolist()
    
    lr = df.get('lr', df.get('learning_rate', [0.001])).tolist()
    
    # Build metrics dict
    metrics = {
        'step': epochs,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss if val_loss and not all(v is None for v in val_loss) else [],
        'val_accuracy': val_acc if val_acc and not all(v is None for v in val_acc) else [],
        'learning_rate': lr,
    }
    
    # Create analyzer and generate report
    analyzer = TrainingStabilityAnalyzer()
    
    run_name = Path(csv_path).parent.name
    run_id = Path(csv_path).stem
    
    report = analyzer.analyze_metrics_dict(
        metrics,
        config={},
        run_id=run_id,
        run_name=run_name,
    )
    
    # Add crash point if detected
    if crash_epoch is not None:
        crash = CrashPoint(
            step=crash_epoch,
            epoch=crash_epoch,
            crash_type="nan_detected",
            severity="critical",
            trigger_metric="loss",
            trigger_value=float('nan'),
            previous_stable_value=train_loss[crash_epoch - 1] if crash_epoch > 0 else 0,
            description=f"NaN detected in loss at epoch {crash_epoch}. Training crashed."
        )
        report.crash_points.append(crash)
        report.stability_score = max(0, report.stability_score - 30)
    
    # Find optimal metrics from stable region
    if crash_epoch is not None and crash_epoch > 0:
        stable_df = df[df['epoch'] < crash_epoch]
        if len(stable_df) > 0:
            # Find best val loss
            if 'val_loss' in stable_df.columns:
                best_idx = stable_df['val_loss'].idxmin()
                best_row = stable_df.loc[best_idx]
                
                report.optimal_metrics = OptimalMetrics(
                    step=int(best_row['epoch']),
                    epoch=int(best_row['epoch']),
                    learning_rate=best_row.get('lr', 0.001),
                    train_loss=best_row.get('loss', 0),
                    train_accuracy=best_row.get('direction_dir_acc', 0),
                    val_loss=best_row.get('val_loss'),
                    val_accuracy=best_row.get('val_direction_dir_acc'),
                    gradient_norm=None,
                    is_stable=True,
                )
    
    return report


def find_all_training_logs(base_path: str = ".") -> List[str]:
    """
    Find all training history files in a directory.
    
    Args:
        base_path: Base directory to search
        
    Returns:
        List of paths to training history files
    """
    from pathlib import Path
    import glob
    
    patterns = [
        "**/training_history.csv",
        "**/history.csv",
        "**/*training*.csv",
        "**/wandb/**/wandb-summary.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(Path(base_path) / pattern), recursive=True))
    
    return list(set(files))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.endswith('.csv'):
            # Analyze single CSV file
            report = analyze_training_csv(path)
            print(report.to_markdown())
        else:
            # Analyze wandb directory
            analyze_wandb_runs(path)
    else:
        # Find and analyze all training logs
        print("=" * 80)
        print("SCANNING FOR TRAINING LOGS...")
        print("=" * 80)
        
        logs = find_all_training_logs(".")
        
        # Separate CSVs and wandb summaries
        csvs = [log for log in logs if log.endswith('.csv')]
        wandb_summaries = [log for log in logs if 'wandb-summary' in log]
        
        print(f"\nFound {len(csvs)} CSV training logs")
        print(f"Found {len(wandb_summaries)} wandb summaries")
        
        # Analyze CSVs first (they have more detail)
        for csv in csvs:
            print(f"\n{'=' * 80}")
            print(f"ANALYZING: {csv}")
            print("=" * 80)
            try:
                report = analyze_training_csv(csv)
                print(report.to_markdown())
            except Exception as e:
                print(f"Error: {e}")
        
        # Also analyze wandb runs
        if wandb_summaries:
            print(f"\n{'=' * 80}")
            print("WANDB RUNS SUMMARY")
            print("=" * 80)
            analyze_wandb_runs("wandb")
