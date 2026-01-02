#!/usr/bin/env python3
"""
Training Diagnostic Script
Checks for common training issues and suggests fixes.

Usage:
    python diagnose_training.py
    python diagnose_training.py --config config_m1_optimized.yaml
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np


class TrainingDiagnostic:
    def __init__(self, config_path: str = "config_m1_optimized.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.issues: List[Tuple[str, str, str]] = []  # (severity, issue, fix)
        self.passes: List[str] = []
        
    def _load_config(self) -> Dict:
        """Load YAML config."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)
    
    def check_model_architecture(self):
        """Check if model architecture is appropriate."""
        print("\n🔍 Checking Model Architecture...")
        
        model_cfg = self.config.get("model", {})
        model_type = model_cfg.get("type", "unknown")
        hidden_size = model_cfg.get("hidden_size", 0)
        num_layers = model_cfg.get("num_layers", 0)
        dropout = model_cfg.get("dropout", 0)
        
        # Check model type
        if model_type == "tcn":
            self.passes.append("✅ Model type: TCN (M1 Metal optimized)")
        elif model_type in ("lstm", "shared_encoder"):
            self.issues.append((
                "🔴 CRITICAL",
                f"Model type is {model_type} (slower, prone to overfitting on M1)",
                "Set model.type: tcn in config"
            ))
        else:
            self.issues.append((
                "🟡 WARNING",
                f"Unknown model type: {model_type}",
                "Set model.type: tcn in config"
            ))
        
        # Check hidden size
        if hidden_size > 64:
            self.issues.append((
                "🟡 WARNING",
                f"Hidden size {hidden_size} is large (may overfit with limited data)",
                "Reduce model.hidden_size to 32-48"
            ))
        elif 24 <= hidden_size <= 48:
            self.passes.append(f"✅ Hidden size: {hidden_size} (good for regularization)")
        
        # Check num layers
        if num_layers > 3:
            self.issues.append((
                "🟡 WARNING",
                f"Too many layers ({num_layers}) for ~3k training samples",
                "Reduce model.num_layers to 2-3"
            ))
        elif num_layers in (2, 3):
            self.passes.append(f"✅ Num layers: {num_layers} (appropriate)")
        
        # Check dropout
        if dropout < 0.3:
            self.issues.append((
                "🟡 WARNING",
                f"Dropout {dropout} is low (may lead to overfitting)",
                "Increase model.dropout to 0.3-0.5"
            ))
        elif 0.3 <= dropout <= 0.5:
            self.passes.append(f"✅ Dropout: {dropout} (good regularization)")
    
    def check_training_config(self):
        """Check training configuration."""
        print("\n🔍 Checking Training Configuration...")
        
        training_cfg = self.config.get("training", {})
        buddy_train_cfg = self.config.get("buddy", {}).get("train_defaults", {})
        
        # Check early stopping patience
        patience = training_cfg.get("early_stopping_patience", 0)
        buddy_patience = buddy_train_cfg.get("patience", 0)
        
        if patience < 30:
            self.issues.append((
                "🔴 CRITICAL",
                f"Early stopping patience ({patience}) is too low",
                "Set training.early_stopping_patience: 40 in config"
            ))
        elif patience >= 30:
            self.passes.append(f"✅ Early stopping patience: {patience} (good)")
        
        if buddy_patience < 30:
            self.issues.append((
                "🔴 CRITICAL",
                f"Buddy patience ({buddy_patience}) is too low",
                "Set buddy.train_defaults.patience: 40 in config"
            ))
        elif buddy_patience >= 30:
            self.passes.append(f"✅ Buddy patience: {buddy_patience} (good)")
        
        # Check min_epochs
        min_epochs = training_cfg.get("min_epochs", 0)
        if min_epochs < 30:
            self.issues.append((
                "🟡 WARNING",
                f"Min epochs ({min_epochs}) is low (model needs time to learn)",
                "Set training.min_epochs: 30-40 in config"
            ))
        elif min_epochs >= 30:
            self.passes.append(f"✅ Min epochs: {min_epochs} (good)")
        
        # Check early stop monitor
        es_monitor = training_cfg.get("early_stop_monitor", "unknown")
        if es_monitor == "direction":
            self.issues.append((
                "🟡 WARNING",
                "Early stop monitor is 'direction' (noisy metric)",
                "Set training.early_stop_monitor: val_loss in config"
            ))
        elif es_monitor == "val_loss":
            self.passes.append("✅ Early stop monitor: val_loss (smooth metric)")
        
        # Check batch size
        batch_size = buddy_train_cfg.get("batch_size", 0)
        if batch_size < 32:
            self.issues.append((
                "🟡 WARNING",
                f"Batch size ({batch_size}) is small (noisy gradients)",
                "Set buddy.train_defaults.batch_size: 64-128"
            ))
        elif 32 <= batch_size <= 128:
            self.passes.append(f"✅ Batch size: {batch_size} (good for M1)")
        elif batch_size > 128:
            self.issues.append((
                "🟡 WARNING",
                f"Batch size ({batch_size}) is large (may slow convergence)",
                "Set buddy.train_defaults.batch_size: 64-128"
            ))
        
        # Check learning rate
        lr = buddy_train_cfg.get("lr", 0)
        if lr > 0.001:
            self.issues.append((
                "🟡 WARNING",
                f"Learning rate ({lr}) is high (may cause instability)",
                "Set buddy.train_defaults.lr: 0.0001-0.0005"
            ))
        elif 0.00005 <= lr <= 0.0005:
            self.passes.append(f"✅ Learning rate: {lr} (good)")
    
    def check_loss_configuration(self):
        """Check loss function configuration."""
        print("\n🔍 Checking Loss Configuration...")
        
        direction_loss = self.config.get("direction_loss", "unknown")
        focal_gamma = self.config.get("focal_gamma", 0)
        auto_focal_alpha = self.config.get("auto_focal_alpha", False)
        
        if direction_loss == "focal":
            self.passes.append("✅ Direction loss: focal (hard example mining enabled)")
            
            if focal_gamma >= 2.0:
                self.passes.append(f"✅ Focal gamma: {focal_gamma} (strong focus on hard examples)")
            else:
                self.issues.append((
                    "🟡 WARNING",
                    f"Focal gamma ({focal_gamma}) is low",
                    "Set focal_gamma: 2.0 in config"
                ))
            
            if auto_focal_alpha:
                self.passes.append("✅ Auto focal alpha: enabled (adaptive class weighting)")
            else:
                self.issues.append((
                    "🟡 WARNING",
                    "Auto focal alpha is disabled",
                    "Set auto_focal_alpha: true in config"
                ))
        else:
            self.issues.append((
                "🔴 CRITICAL",
                f"Direction loss is {direction_loss} (not using focal loss)",
                "Set direction_loss: focal in config AND update main.py code (see QUICK_FIX_TRAINING.md)"
            ))
        
        # Check unified head loss weights
        loss_weights = self.config.get("unified_head_loss_weights", {})
        direction_weight = loss_weights.get("direction", 0)
        
        if direction_weight >= 10:
            self.passes.append(f"✅ Direction loss weight: {direction_weight} (high priority)")
        elif direction_weight > 0:
            self.issues.append((
                "🟡 WARNING",
                f"Direction loss weight ({direction_weight}) is low",
                "Set unified_head_loss_weights.direction: 15.0 in config"
            ))
    
    def check_feature_configuration(self):
        """Check feature selection and engineering."""
        print("\n🔍 Checking Feature Configuration...")
        
        buddy_train_cfg = self.config.get("buddy", {}).get("train_defaults", {})
        top_features = buddy_train_cfg.get("top_features")
        pca_components = buddy_train_cfg.get("pca_components")
        
        if top_features is None and pca_components is None:
            self.issues.append((
                "🔴 CRITICAL",
                "No feature selection enabled (using all 196+ features)",
                "Add --top-features 40 to training command OR set buddy.train_defaults.top_features: 40"
            ))
        elif top_features is not None:
            if 30 <= top_features <= 60:
                self.passes.append(f"✅ Feature selection: top {top_features} features (reduces noise)")
            elif top_features < 30:
                self.issues.append((
                    "🟡 WARNING",
                    f"Too few features selected ({top_features})",
                    "Increase top_features to 40-50"
                ))
            elif top_features > 80:
                self.issues.append((
                    "🟡 WARNING",
                    f"Too many features selected ({top_features})",
                    "Reduce top_features to 40-60"
                ))
    
    def check_data_quality(self):
        """Check data quality settings."""
        print("\n🔍 Checking Data Quality Settings...")
        
        # Check spread filter
        spread_filter = self.config.get("data", {}).get("spread_filter", False)
        if spread_filter:
            self.passes.append("✅ Spread filter: enabled (removes wide-spread candles)")
        else:
            self.issues.append((
                "🟡 WARNING",
                "Spread filter is disabled",
                "Consider enabling spread_filter in data config"
            ))
        
        # Check normalization
        normalize_features = self.config.get("normalize_features", False)
        enable_data_scaling = self.config.get("enable_data_scaling", False)
        
        if normalize_features or enable_data_scaling:
            self.passes.append("✅ Feature normalization: enabled")
        else:
            self.issues.append((
                "🔴 CRITICAL",
                "Feature normalization is disabled",
                "Set normalize_features: true in config"
            ))
    
    def check_training_results(self):
        """Check recent training results if available."""
        print("\n🔍 Checking Recent Training Results...")
        
        model_dir = Path("trained_data/models")
        meta_files = list(model_dir.glob("buddy_tf*.meta.json"))
        
        if not meta_files:
            self.issues.append((
                "🟡 INFO",
                "No training results found",
                "Run a training session to generate results"
            ))
            return
        
        # Get most recent meta file
        latest_meta = max(meta_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_meta) as f:
                meta = json.load(f)
            
            val_acc = meta.get("val_direction_accuracy", 0)
            best_epoch = meta.get("best_epoch", 0)
            trained_epochs = meta.get("trained_epochs", 0)
            early_stopped = meta.get("early_stopped", False)
            
            print(f"\n📊 Latest Training: {latest_meta.name}")
            print(f"   Val Accuracy: {val_acc:.1%}")
            print(f"   Best Epoch: {best_epoch}")
            print(f"   Total Epochs: {trained_epochs}")
            print(f"   Early Stopped: {early_stopped}")
            
            # Analyze results
            if val_acc < 0.55:
                self.issues.append((
                    "🔴 CRITICAL",
                    f"Validation accuracy is low ({val_acc:.1%})",
                    "Model is not learning - apply all critical fixes"
                ))
            elif 0.55 <= val_acc < 0.60:
                self.issues.append((
                    "🟡 WARNING",
                    f"Validation accuracy is marginal ({val_acc:.1%})",
                    "Apply remaining fixes for +2-5% improvement"
                ))
            elif val_acc >= 0.60:
                self.passes.append(f"✅ Validation accuracy: {val_acc:.1%} (good!)")
            
            if best_epoch < 20:
                self.issues.append((
                    "🔴 CRITICAL",
                    f"Best epoch is very early ({best_epoch})",
                    "Model stopped too soon - increase patience"
                ))
            elif 20 <= best_epoch < 40:
                self.issues.append((
                    "🟡 WARNING",
                    f"Best epoch is early ({best_epoch})",
                    "Consider increasing patience to 40-50"
                ))
            elif best_epoch >= 40:
                self.passes.append(f"✅ Best epoch: {best_epoch} (good convergence)")
            
            # Check train/val gap (overfitting indicator)
            train_acc = meta.get("train_direction_accuracy", val_acc)
            if train_acc - val_acc > 0.15:
                self.issues.append((
                    "🔴 CRITICAL",
                    f"Severe overfitting detected (train: {train_acc:.1%}, val: {val_acc:.1%})",
                    "Reduce model capacity, increase dropout, or add more data"
                ))
            elif train_acc - val_acc > 0.08:
                self.issues.append((
                    "🟡 WARNING",
                    f"Moderate overfitting (train: {train_acc:.1%}, val: {val_acc:.1%})",
                    "Consider increasing regularization"
                ))
            
        except Exception as e:
            print(f"   ⚠️ Could not load metadata: {e}")
    
    def print_report(self):
        """Print diagnostic report."""
        print("\n" + "="*70)
        print("📋 TRAINING DIAGNOSTIC REPORT")
        print("="*70)
        
        # Run all checks
        self.check_model_architecture()
        self.check_training_config()
        self.check_loss_configuration()
        self.check_feature_configuration()
        self.check_data_quality()
        self.check_training_results()
        
        # Print passes
        if self.passes:
            print("\n✅ PASSING CHECKS:")
            for check in self.passes:
                print(f"   {check}")
        
        # Print issues
        if self.issues:
            print("\n🔍 ISSUES FOUND:")
            
            # Group by severity
            critical = [i for i in self.issues if i[0].startswith("🔴")]
            warnings = [i for i in self.issues if i[0].startswith("🟡")]
            info = [i for i in self.issues if "INFO" in i[0]]
            
            if critical:
                print("\n   🔴 CRITICAL ISSUES (Fix These First):")
                for severity, issue, fix in critical:
                    print(f"      ❌ {issue}")
                    print(f"         → {fix}")
            
            if warnings:
                print("\n   🟡 WARNINGS (Should Fix):")
                for severity, issue, fix in warnings:
                    print(f"      ⚠️  {issue}")
                    print(f"         → {fix}")
            
            if info:
                print("\n   ℹ️  INFORMATION:")
                for severity, issue, fix in info:
                    print(f"      • {issue}")
                    print(f"        → {fix}")
        
        # Summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"   ✅ Passing Checks: {len(self.passes)}")
        print(f"   🔴 Critical Issues: {sum(1 for i in self.issues if i[0].startswith('🔴'))}")
        print(f"   🟡 Warnings: {sum(1 for i in self.issues if i[0].startswith('🟡'))}")
        
        critical_count = sum(1 for i in self.issues if i[0].startswith('🔴'))
        
        if critical_count == 0:
            print("\n🎉 No critical issues found! Ready to train.")
            print("   Run: ./Buddy train --model-type tcn --top-features 40 --candles 30000")
        else:
            print(f"\n⚠️  {critical_count} critical issue(s) found. Fix these before training.")
            print("   See: QUICK_FIX_TRAINING.md for detailed instructions")
        
        print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Training diagnostic tool")
    parser.add_argument("--config", default="config_m1_optimized.yaml", help="Path to config file")
    args = parser.parse_args()
    
    print("🔬 BUDDY TRAINING DIAGNOSTIC TOOL")
    print("="*70)
    
    diagnostic = TrainingDiagnostic(args.config)
    diagnostic.print_report()


if __name__ == "__main__":
    main()

