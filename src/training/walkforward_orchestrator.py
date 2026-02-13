"""
Walk-Forward Cross-Validation Orchestrator.

This module orchestrates walk-forward cross-validation training for the ML Engine.
It manages the fold-wise training loop, metric aggregation, and model selection.

Key Features:
1. Orchestrates training across multiple walk-forward folds
2. Supports rolling and expanding window modes
3. Aggregates metrics using best/average/ensemble strategies
4. Provides comprehensive logging and progress reporting
5. Maintains backward compatibility with standard training

Usage:
    from src.training.walkforward_orchestrator import WalkForwardOrchestrator
    
    orchestrator = WalkForwardOrchestrator(
        trainer_class=TransformerDirectionTrainer,
        trainer_config=config,
        wf_config=wf_config_dict,
    )
    
    best_trainer, metrics = orchestrator.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
        instrument="EUR_USD",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.training.walkforward_validation import (
    WalkForwardConfig,
    WalkForwardValidator,
    WalkForwardResult,
    WalkForwardSummary,
)

logger = logging.getLogger(__name__)


class WalkForwardOrchestrator:
    """
    Orchestrator for walk-forward cross-validation training.
    
    This class manages the entire walk-forward training pipeline:
    - Creates fold splits using WalkForwardValidator
    - Trains models on each fold
    - Aggregates metrics across folds
    - Selects best model based on configured strategy
    
    Attributes:
        trainer_class: The trainer class to instantiate for each fold
        trainer_config: Configuration object for the trainer
        wf_config: Walk-forward validation configuration
        console: Rich console for output (optional)
        fold_trainers: List of trained models (one per fold)
        fold_results: List of WalkForwardResult objects
        summary: WalkForwardSummary with aggregated metrics
    """
    
    def __init__(
        self,
        trainer_class: Type,
        trainer_config: Any,
        wf_config: Dict[str, Any],
        console: Optional[Console] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            trainer_class: Trainer class to use (e.g., TransformerDirectionTrainer)
            trainer_config: Configuration object for the trainer
            wf_config: Walk-forward configuration dict from YAML
            console: Optional Rich console for output
        """
        self.trainer_class = trainer_class
        self.trainer_config = trainer_config
        self.wf_config = WalkForwardConfig.from_dict(wf_config)
        self.console = console or Console()
        
        # Results storage
        self.fold_trainers: List[Any] = []
        self.fold_results: List[WalkForwardResult] = []
        self.summary: Optional[WalkForwardSummary] = None
        
        # Validator
        self.validator = WalkForwardValidator(
            n_splits=self.wf_config.n_splits,
            train_size=self.wf_config.train_size,
            val_size=self.wf_config.val_size,
            test_size=self.wf_config.test_size,
            gap=self.wf_config.gap,
            mode=self.wf_config.mode,
            min_train_size=self.wf_config.min_train_size,
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        instrument: str = "UNKNOWN",
        **trainer_kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train using walk-forward cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Optional feature names
            instrument: Instrument name for logging
            **trainer_kwargs: Additional kwargs to pass to trainer.train()
        
        Returns:
            Tuple of (best_trainer, aggregated_metrics)
        """
        # Display configuration
        self._print_config_panel(instrument)
        
        # Combine train and val for walk-forward splitting
        X_combined = np.concatenate([X_train, X_val], axis=0)
        y_combined = np.concatenate([y_train, y_val], axis=0)
        
        # Extract sample weights if present
        w_train = trainer_kwargs.pop('w_train', None)
        w_val = trainer_kwargs.pop('w_val', None)
        w_combined = None
        if w_train is not None and w_val is not None:
            w_combined = np.concatenate([w_train, w_val], axis=0)
        
        # Train on each fold
        self._train_all_folds(
            X_combined,
            y_combined,
            w_combined,
            feature_names,
            instrument,
            trainer_kwargs,
        )
        
        # Compute summary statistics
        self._compute_summary()
        
        # Display results
        self._print_results_table()
        
        # Select and return best model
        best_trainer, aggregated_metrics = self._select_best_model()
        
        return best_trainer, aggregated_metrics
    
    def _print_config_panel(self, instrument: str) -> None:
        """Display configuration panel."""
        panel = Panel(
            f"[bold]Walk-Forward Cross-Validation[/bold]\n\n"
            f"[dim]Instrument:[/dim] {instrument}\n"
            f"[dim]Mode:[/dim] {self.wf_config.mode.upper()}\n"
            f"[dim]Folds:[/dim] {self.wf_config.n_splits}\n"
            f"[dim]Train Size:[/dim] {self.wf_config.train_size*100:.0f}%\n"
            f"[dim]Gap:[/dim] {self.wf_config.gap} samples\n"
            f"[dim]Retrain per fold:[/dim] {'✓' if self.wf_config.retrain_per_fold else '✗'}\n"
            f"[dim]Aggregate method:[/dim] {self.wf_config.aggregate_method}",
            title="📊 WFCV Configuration",
            border_style="cyan",
        )
        self.console.print(panel)
    
    def _train_all_folds(
        self,
        X_combined: np.ndarray,
        y_combined: np.ndarray,
        w_combined: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        instrument: str,
        trainer_kwargs: Dict[str, Any],
    ) -> None:
        """Train models on all folds."""
        self.console.print(f"\n[cyan]Training {self.wf_config.n_splits} folds...[/cyan]\n")
        
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(self.validator.split(X_combined)):
            fold_num = fold_idx + 1
            
            self.console.print(
                f"[bold cyan]Fold {fold_num}/{self.wf_config.n_splits}[/bold cyan] | "
                f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}"
            )
            
            # Split data for this fold
            X_train_fold = X_combined[train_idx]
            y_train_fold = y_combined[train_idx]
            X_val_fold = X_combined[val_idx]
            y_val_fold = y_combined[val_idx]
            X_test_fold = X_combined[test_idx]
            y_test_fold = y_combined[test_idx]
            
            # Split weights if available
            w_train_fold = w_combined[train_idx] if w_combined is not None else None
            w_val_fold = w_combined[val_idx] if w_combined is not None else None
            
            # Train on this fold
            fold_trainer, fold_metrics = self._train_single_fold(
                fold_idx=fold_idx,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                X_test=X_test_fold,
                y_test=y_test_fold,
                w_train=w_train_fold,
                w_val=w_val_fold,
                feature_names=feature_names,
                instrument=instrument,
                trainer_kwargs=trainer_kwargs,
            )
            
            # Store results
            self.fold_trainers.append(fold_trainer)
            
            # Create WalkForwardResult
            result = WalkForwardResult(
                fold=fold_num,
                train_size=len(train_idx),
                val_size=len(val_idx),
                test_size=len(test_idx),
                train_metrics=fold_metrics.get('train_metrics', {}),
                val_metrics=fold_metrics,
                test_metrics=self._evaluate_on_test(fold_trainer, X_test_fold, y_test_fold),
                train_period=(int(train_idx[0]), int(train_idx[-1])),
                val_period=(int(val_idx[0]), int(val_idx[-1])),
                test_period=(int(test_idx[0]), int(test_idx[-1])),
            )
            self.fold_results.append(result)
            
            # Print fold result
            val_acc = fold_metrics.get('val_accuracy', 0.0)
            test_acc = result.test_metrics.get('test_accuracy', 0.0)
            self.console.print(
                f"  [green]✓ Fold {fold_num}: "
                f"Val Acc={val_acc:.1%}, Test Acc={test_acc:.1%}[/green]\n"
            )
    
    def _train_single_fold(
        self,
        fold_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        w_train: Optional[np.ndarray],
        w_val: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        instrument: str,
        trainer_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single fold.
        
        Args:
            fold_idx: Fold index (0-based)
            X_train: Training features for this fold
            y_train: Training labels for this fold
            X_val: Validation features for this fold
            y_val: Validation labels for this fold
            X_test: Test features for this fold (for evaluation)
            y_test: Test labels for this fold (for evaluation)
            w_train: Training sample weights
            w_val: Validation sample weights
            feature_names: Feature names
            instrument: Instrument name
            trainer_kwargs: Additional trainer arguments
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        if self.wf_config.retrain_per_fold or fold_idx == 0:
            # Create fresh trainer for this fold
            fold_trainer = self.trainer_class(self.trainer_config)
            
            # Train on this fold
            fold_metrics = fold_trainer.train(
                X_train,
                y_train,
                X_val,
                y_val,
                feature_names=feature_names,
                w_train=w_train,
                w_val=w_val,
                instrument=instrument,
                fold_id=fold_idx,
                **trainer_kwargs,
            )
        else:
            # Reuse trainer from fold 0 (not recommended, but supported)
            fold_trainer = self.fold_trainers[0]
            # Just evaluate without retraining
            fold_metrics = {
                'val_accuracy': self._compute_accuracy(fold_trainer, X_val, y_val),
            }
        
        return fold_trainer, fold_metrics
    
    def _evaluate_on_test(
        self,
        trainer: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate trainer on test set.
        
        Args:
            trainer: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of test metrics
        """
        try:
            test_acc = self._compute_accuracy(trainer, X_test, y_test)
            return {
                'test_accuracy': test_acc,
                'test_samples': len(X_test),
            }
        except Exception as e:
            logger.warning(f"Test evaluation failed: {e}")
            return {'test_accuracy': 0.0, 'test_samples': len(X_test)}
    
    def _compute_accuracy(
        self,
        trainer: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute accuracy for a trainer on given data.
        
        Args:
            trainer: Trained model
            X: Features
            y: True labels
        
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        try:
            pred_dict = trainer.predict(X)
            
            # Handle different prediction formats
            if 'prediction' in pred_dict:
                y_pred = pred_dict['prediction']
            elif 'predictions' in pred_dict:
                y_pred = pred_dict['predictions']
            elif 'direction' in pred_dict:
                y_pred = pred_dict['direction']
            else:
                # Try to get the first array-like value
                for v in pred_dict.values():
                    if hasattr(v, '__len__'):
                        y_pred = v
                        break
                else:
                    return 0.0
            
            # Convert probabilities to binary predictions if needed
            if len(y_pred.shape) > 1:
                y_pred = (y_pred[:, 1] > 0.5).astype(int) if y_pred.shape[1] == 2 else y_pred.argmax(axis=1)
            elif y_pred.dtype == float:
                y_pred = (y_pred > 0.5).astype(int)
            
            # Ensure same length
            min_len = min(len(y_pred), len(y))
            accuracy = np.mean(y_pred[:min_len] == y[:min_len])
            
            return float(accuracy)
        except Exception as e:
            logger.warning(f"Accuracy computation failed: {e}")
            return 0.0
    
    def _compute_summary(self) -> None:
        """Compute summary statistics from fold results."""
        self.summary = WalkForwardSummary(
            n_folds=len(self.fold_results),
            fold_results=self.fold_results,
        )
        self.summary.compute_summary()
    
    def _print_results_table(self) -> None:
        """Display results table."""
        if not self.summary:
            return
        
        table = Table(
            title="Walk-Forward Cross-Validation Results",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("Metric", style="white", width=20)
        table.add_column("Mean", style="green", width=12)
        table.add_column("Std", style="yellow", width=12)
        table.add_column("Stability", style="cyan", width=12)
        
        # Add test metrics
        for metric_name, mean_val in self.summary.mean_test_metrics.items():
            std_val = self.summary.std_test_metrics.get(metric_name, 0.0)
            stability = self.summary.metric_stability.get(metric_name, 0.0)
            
            table.add_row(
                metric_name,
                f"{mean_val:.4f}",
                f"{std_val:.4f}",
                f"{stability:.4f}",
            )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
    
    def _select_best_model(self) -> Tuple[Any, Dict[str, float]]:
        """
        Select best model based on configured strategy.
        
        Returns:
            Tuple of (best_trainer, aggregated_metrics)
        """
        if self.wf_config.aggregate_method == "best":
            # Select fold with best validation accuracy
            best_idx = 0
            best_val_acc = 0.0
            
            for idx, result in enumerate(self.fold_results):
                val_acc = result.val_metrics.get('val_accuracy', 0.0)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_idx = idx
            
            best_trainer = self.fold_trainers[best_idx]
            aggregated_metrics = {
                **self.fold_results[best_idx].val_metrics,
                'wfcv_mean_test_accuracy': self.summary.mean_test_metrics.get('test_accuracy', 0.0),
                'wfcv_std_test_accuracy': self.summary.std_test_metrics.get('test_accuracy', 0.0),
                'wfcv_best_fold': best_idx + 1,
                'wfcv_n_folds': len(self.fold_results),
            }
            
            self.console.print(
                f"[bold green]✓ Selected best fold: {best_idx + 1} "
                f"(Val Acc: {best_val_acc:.1%})[/bold green]\n"
            )
            
        elif self.wf_config.aggregate_method == "average":
            # Use last fold's trainer with averaged metrics
            best_trainer = self.fold_trainers[-1]
            aggregated_metrics = {
                'val_accuracy': self.summary.mean_test_metrics.get('test_accuracy', 0.0),
                'wfcv_mean_test_accuracy': self.summary.mean_test_metrics.get('test_accuracy', 0.0),
                'wfcv_std_test_accuracy': self.summary.std_test_metrics.get('test_accuracy', 0.0),
                'wfcv_n_folds': len(self.fold_results),
            }
            
            self.console.print(
                f"[bold green]✓ Using averaged metrics from {len(self.fold_results)} folds[/bold green]\n"
            )
            
        elif self.wf_config.aggregate_method == "ensemble":
            # Return ensemble wrapper (not implemented yet)
            logger.warning("Ensemble method not yet implemented, falling back to 'best'")
            return self._select_best_model_helper_best()
            
        else:
            raise ValueError(f"Unknown aggregate_method: {self.wf_config.aggregate_method}")
        
        return best_trainer, aggregated_metrics
    
    def _select_best_model_helper_best(self) -> Tuple[Any, Dict[str, float]]:
        """Helper to avoid recursion in ensemble fallback."""
        best_idx = 0
        best_val_acc = 0.0
        
        for idx, result in enumerate(self.fold_results):
            val_acc = result.val_metrics.get('val_accuracy', 0.0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_idx = idx
        
        best_trainer = self.fold_trainers[best_idx]
        aggregated_metrics = {
            **self.fold_results[best_idx].val_metrics,
            'wfcv_mean_test_accuracy': self.summary.mean_test_metrics.get('test_accuracy', 0.0),
            'wfcv_std_test_accuracy': self.summary.std_test_metrics.get('test_accuracy', 0.0),
            'wfcv_best_fold': best_idx + 1,
            'wfcv_n_folds': len(self.fold_results),
        }
        
        return best_trainer, aggregated_metrics
    
    def save_summary(self, output_dir: Path) -> None:
        """
        Save WFCV summary to disk.
        
        Args:
            output_dir: Directory to save summary
        """
        if not self.summary:
            logger.warning("No summary to save")
            return
        
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / "wfcv_summary.json"
        
        summary_dict = {
            'n_folds': self.summary.n_folds,
            'mean_test_metrics': self.summary.mean_test_metrics,
            'std_test_metrics': self.summary.std_test_metrics,
            'metric_stability': self.summary.metric_stability,
            'config': self.wf_config.to_dict(),
            'fold_results': [
                {
                    'fold': r.fold,
                    'train_size': r.train_size,
                    'val_size': r.val_size,
                    'test_size': r.test_size,
                    'val_metrics': r.val_metrics,
                    'test_metrics': r.test_metrics,
                }
                for r in self.summary.fold_results
            ],
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logger.info(f"Saved WFCV summary to {summary_path}")
