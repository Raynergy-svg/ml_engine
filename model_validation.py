#!/usr/bin/env python3
"""
Model Validation Suite for ML Engine
=====================================
Comprehensive tests to validate 78% accuracy claim:
1. USD_JPY cross-pair validation
2. Market regime analysis (bull/bear/ranging)
3. Post-transaction-cost Sharpe ratio
4. Feature importance / leakage detection
"""

import numpy as np
import pandas as pd
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()


# ============================================================
# Configuration
# ============================================================
class ValidationConfig:
    """Validation test configuration"""
    
    # Data settings
    PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    TIMEFRAME = 'H1'
    MIN_SAMPLES = 1000
    
    # Transaction costs (realistic)
    SPREADS = {
        'EUR_USD': 0.00010,  # 1.0 pip
        'GBP_USD': 0.00012,  # 1.2 pips
        'USD_JPY': 0.00010,  # ~1.0 pip (converted)
    }
    COMMISSION_PER_SIDE = 0.000005  # $5 per 100k
    SLIPPAGE_PIPS = 0.3
    
    # Regime detection
    TREND_THRESHOLD = 0.002  # 0.2% move defines trend
    VOLATILITY_LOOKBACK = 20  # bars for vol calculation
    
    # Paths
    MODEL_DIR = 'trained_data/models'
    DATA_DIR = 'data'


# ============================================================
# 1. USD_JPY Cross-Pair Validation
# ============================================================
class CrossPairValidator:
    """Test if model generalizes across currency pairs"""
    
    def __init__(self, model, scaler, config: ValidationConfig):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.results = {}
    
    def load_pair_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Load data for a specific pair"""
        # Try multiple data sources
        paths = [
            f"{self.config.DATA_DIR}/{pair}_{self.config.TIMEFRAME}.csv",
            f"{self.config.DATA_DIR}/{pair}_data.csv",
            f"data/{pair}.csv",
        ]
        
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                console.print(f"  ✅ Loaded {pair}: {len(df)} samples from {path}")
                return df
        
        console.print(f"  ⚠️ No data found for {pair}")
        return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features matching training pipeline"""
        from feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        features_df = fe.create_features(df)
        
        # Create labels (direction)
        features_df['future_return'] = features_df['close'].pct_change().shift(-1)
        features_df['direction'] = (features_df['future_return'] > 0).astype(int)
        
        # Drop NaN
        features_df = features_df.dropna()
        
        # Get feature columns (exclude targets)
        exclude_cols = ['direction', 'future_return', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df['direction'].values
        
        return X, y
    
    def evaluate_pair(self, pair: str) -> Dict:
        """Evaluate model on a specific pair"""
        console.print(f"\n📊 Evaluating {pair}...")
        
        df = self.load_pair_data(pair)
        if df is None:
            return {'pair': pair, 'status': 'no_data'}
        
        try:
            X, y = self.prepare_features(df)
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Reshape for Transformer (batch, seq, features)
            # Use last 60 bars as sequence
            seq_len = 60
            X_seq = []
            y_seq = []
            
            for i in range(seq_len, len(X_scaled)):
                X_seq.append(X_scaled[i-seq_len:i])
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Predict
            predictions = self.model.predict(X_seq, verbose=0)
            pred_classes = (predictions.flatten() > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(pred_classes == y_seq)
            
            # Class distribution
            pred_up = np.mean(pred_classes == 1)
            actual_up = np.mean(y_seq == 1)
            
            # Check for collapse
            collapse = pred_up < 0.1 or pred_up > 0.9
            
            result = {
                'pair': pair,
                'status': 'success',
                'samples': len(y_seq),
                'accuracy': accuracy,
                'pred_up_pct': pred_up,
                'actual_up_pct': actual_up,
                'collapse': collapse,
            }
            
            self.results[pair] = result
            
            console.print(f"  Accuracy: {accuracy:.1%}")
            console.print(f"  Predictions: {pred_up:.1%} UP | Actual: {actual_up:.1%} UP")
            if collapse:
                console.print(f"  ⚠️ PREDICTION COLLAPSE DETECTED")
            
            return result
            
        except Exception as e:
            console.print(f"  ❌ Error: {e}")
            return {'pair': pair, 'status': 'error', 'error': str(e)}
    
    def run_all(self) -> Dict:
        """Run validation on all pairs"""
        console.print(Panel("[bold cyan]1. Cross-Pair Validation[/bold cyan]"))
        
        for pair in self.config.PAIRS:
            self.evaluate_pair(pair)
        
        # Summary
        console.print("\n📋 Cross-Pair Summary:")
        table = Table()
        table.add_column("Pair")
        table.add_column("Accuracy")
        table.add_column("Samples")
        table.add_column("Status")
        
        for pair, result in self.results.items():
            if result.get('status') == 'success':
                acc = f"{result['accuracy']:.1%}"
                status = "✅" if result['accuracy'] > 0.55 else "⚠️"
                if result.get('collapse'):
                    status = "❌ COLLAPSE"
            else:
                acc = "N/A"
                status = result.get('status', 'unknown')
            
            table.add_row(pair, acc, str(result.get('samples', 0)), status)
        
        console.print(table)
        
        return self.results


# ============================================================
# 2. Market Regime Analysis
# ============================================================
class RegimeAnalyzer:
    """Test model performance across different market regimes"""
    
    def __init__(self, model, scaler, config: ValidationConfig):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.results = {}
    
    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each period into bull/bear/ranging"""
        df = df.copy()
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.config.VOLATILITY_LOOKBACK).std()
        df['trend'] = df['close'].pct_change(self.config.VOLATILITY_LOOKBACK)
        
        # Classify regime
        def classify_regime(row):
            if pd.isna(row['trend']) or pd.isna(row['volatility']):
                return 'unknown'
            
            if row['trend'] > self.config.TREND_THRESHOLD:
                return 'bull'
            elif row['trend'] < -self.config.TREND_THRESHOLD:
                return 'bear'
            else:
                return 'ranging'
        
        df['regime'] = df.apply(classify_regime, axis=1)
        
        return df
    
    def evaluate_by_regime(self, X: np.ndarray, y: np.ndarray, 
                           regimes: np.ndarray) -> Dict:
        """Evaluate model on each regime"""
        results = {}
        
        for regime in ['bull', 'bear', 'ranging']:
            mask = regimes == regime
            if np.sum(mask) < 50:
                console.print(f"  ⚠️ {regime}: Only {np.sum(mask)} samples, skipping")
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            
            # Predict
            predictions = self.model.predict(X_regime, verbose=0)
            pred_classes = (predictions.flatten() > 0.5).astype(int)
            
            accuracy = np.mean(pred_classes == y_regime)
            
            results[regime] = {
                'samples': int(np.sum(mask)),
                'accuracy': accuracy,
                'pred_up': np.mean(pred_classes),
                'actual_up': np.mean(y_regime),
            }
            
            emoji = "📈" if regime == 'bull' else "📉" if regime == 'bear' else "↔️"
            console.print(f"  {emoji} {regime.upper()}: {accuracy:.1%} ({np.sum(mask)} samples)")
        
        return results
    
    def run(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> Dict:
        """Run regime analysis"""
        console.print(Panel("[bold cyan]2. Market Regime Analysis[/bold cyan]"))
        
        # Detect regimes
        df_with_regime = self.detect_regimes(df)
        
        # Align regimes with features (accounting for sequence length)
        seq_len = 60
        regimes = df_with_regime['regime'].values[seq_len:]
        
        # Trim to match
        min_len = min(len(X), len(y), len(regimes))
        X = X[:min_len]
        y = y[:min_len]
        regimes = regimes[:min_len]
        
        # Show regime distribution
        regime_counts = pd.Series(regimes).value_counts()
        console.print(f"\nRegime Distribution:")
        for regime, count in regime_counts.items():
            console.print(f"  {regime}: {count} ({count/len(regimes):.1%})")
        
        console.print("\nAccuracy by Regime:")
        self.results = self.evaluate_by_regime(X, y, regimes)
        
        # Check for regime-specific issues
        if self.results:
            accuracies = [r['accuracy'] for r in self.results.values()]
            if max(accuracies) - min(accuracies) > 0.15:
                console.print("\n⚠️ WARNING: Large accuracy variance across regimes")
                console.print("   Model may only work in specific market conditions")
        
        return self.results


# ============================================================
# 3. Post-Transaction Cost Analysis
# ============================================================
class TransactionCostAnalyzer:
    """Calculate actual profitability after spreads and commissions"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
    
    def calculate_pnl(self, predictions: np.ndarray, actual_returns: np.ndarray,
                      pair: str = 'EUR_USD') -> Dict:
        """Calculate P&L with transaction costs"""
        
        spread = self.config.SPREADS.get(pair, 0.0001)
        commission = self.config.COMMISSION_PER_SIDE * 2  # Round trip
        slippage = self.config.SLIPPAGE_PIPS * 0.0001
        
        total_cost = spread + commission + slippage
        
        # Convert predictions to positions (-1, 0, 1)
        # Assuming predictions > 0.5 = long, < 0.5 = short
        positions = np.where(predictions > 0.5, 1, -1)
        
        # Raw P&L (before costs)
        raw_pnl = positions * actual_returns
        
        # P&L after costs (pay cost on each trade)
        position_changes = np.abs(np.diff(positions, prepend=0))  # 0 = hold, 2 = flip
        trade_costs = (position_changes > 0) * total_cost
        
        net_pnl = raw_pnl - trade_costs
        
        # Calculate metrics
        cumulative_pnl = np.cumsum(net_pnl)
        
        # Sharpe ratio (annualized for H1 = ~6240 bars/year)
        bars_per_year = 6240  # 24 * 260
        sharpe = np.mean(net_pnl) / (np.std(net_pnl) + 1e-10) * np.sqrt(bars_per_year)
        
        # Max drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / (np.abs(peak) + 1e-10)
        max_dd = np.max(drawdown)
        
        # Win rate
        wins = np.sum(net_pnl > 0)
        losses = np.sum(net_pnl < 0)
        win_rate = wins / (wins + losses + 1e-10)
        
        # Trade count
        trades = np.sum(position_changes > 0)
        
        results = {
            'pair': pair,
            'total_return': float(np.sum(net_pnl)),
            'raw_return': float(np.sum(raw_pnl)),
            'total_costs': float(np.sum(trade_costs)),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate),
            'trades': int(trades),
            'avg_trade': float(np.sum(net_pnl) / (trades + 1e-10)),
        }
        
        return results
    
    def run(self, predictions: np.ndarray, actual_returns: np.ndarray,
            pair: str = 'EUR_USD') -> Dict:
        """Run transaction cost analysis"""
        console.print(Panel("[bold cyan]3. Post-Transaction Cost Analysis[/bold cyan]"))
        
        self.results = self.calculate_pnl(predictions, actual_returns, pair)
        
        # Display results
        table = Table(title=f"Trading Metrics - {pair}")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_column("Assessment")
        
        # Sharpe assessment
        sharpe = self.results['sharpe_ratio']
        if sharpe > 2.0:
            sharpe_status = "🌟 Excellent"
        elif sharpe > 1.5:
            sharpe_status = "✅ Very Good"
        elif sharpe > 1.0:
            sharpe_status = "✅ Good"
        elif sharpe > 0.5:
            sharpe_status = "⚠️ Marginal"
        else:
            sharpe_status = "❌ Poor"
        
        table.add_row("Sharpe Ratio", f"{sharpe:.2f}", sharpe_status)
        table.add_row("Max Drawdown", f"{self.results['max_drawdown']:.1%}", 
                      "✅" if self.results['max_drawdown'] < 0.15 else "⚠️")
        table.add_row("Win Rate", f"{self.results['win_rate']:.1%}",
                      "✅" if self.results['win_rate'] > 0.52 else "⚠️")
        table.add_row("Total Return", f"{self.results['total_return']:.4f}", "")
        table.add_row("Trading Costs", f"{self.results['total_costs']:.4f}", "")
        table.add_row("Number of Trades", str(self.results['trades']), "")
        
        console.print(table)
        
        # Profitability check
        if self.results['total_return'] > self.results['total_costs']:
            console.print("\n✅ Strategy is profitable AFTER transaction costs")
        else:
            console.print("\n❌ Strategy is NOT profitable after transaction costs")
            console.print(f"   Costs ({self.results['total_costs']:.4f}) exceed returns ({self.results['total_return']:.4f})")
        
        return self.results


# ============================================================
# 4. Feature Importance / Leakage Detection
# ============================================================
class LeakageDetector:
    """Check for potential data leakage via feature importance"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.results = {}
    
    def permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                n_repeats: int = 5) -> Dict:
        """Calculate permutation importance for each feature"""
        console.print(Panel("[bold cyan]4. Feature Importance & Leakage Detection[/bold cyan]"))
        
        # Baseline accuracy
        base_pred = self.model.predict(X, verbose=0)
        base_acc = np.mean((base_pred.flatten() > 0.5) == y)
        console.print(f"Baseline accuracy: {base_acc:.1%}")
        
        importances = {}
        
        with Progress() as progress:
            task = progress.add_task("Testing features...", total=X.shape[2])
            
            for i in range(X.shape[2]):
                # Permute feature i
                drops = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    # Shuffle feature i across all samples
                    np.random.shuffle(X_permuted[:, :, i].flat)
                    
                    perm_pred = self.model.predict(X_permuted, verbose=0)
                    perm_acc = np.mean((perm_pred.flatten() > 0.5) == y)
                    drops.append(base_acc - perm_acc)
                
                feat_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                importances[feat_name] = np.mean(drops)
                
                progress.update(task, advance=1)
        
        # Sort by importance
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # Display top features
        console.print("\n📊 Top 15 Most Important Features:")
        table = Table()
        table.add_column("Rank")
        table.add_column("Feature")
        table.add_column("Importance")
        table.add_column("Leakage Risk")
        
        # Known leakage-prone features
        leakage_keywords = ['future', 'target', 'label', 'next', '_1', 'close_t1', 'return_1']
        
        for rank, (feat, imp) in enumerate(sorted_imp[:15], 1):
            # Check for leakage risk
            risk = "⚠️ CHECK" if any(kw in feat.lower() for kw in leakage_keywords) else "✅"
            table.add_row(str(rank), feat, f"{imp:.4f}", risk)
        
        console.print(table)
        
        # Leakage warnings
        top_features = [f for f, _ in sorted_imp[:5]]
        suspicious = [f for f in top_features if any(kw in f.lower() for kw in leakage_keywords)]
        
        if suspicious:
            console.print(f"\n🚨 POTENTIAL LEAKAGE DETECTED!")
            console.print(f"   Suspicious top features: {suspicious}")
            console.print("   These features may contain future information")
        else:
            console.print("\n✅ No obvious leakage detected in top features")
        
        self.results = {
            'importances': dict(sorted_imp),
            'top_features': top_features,
            'suspicious': suspicious,
            'base_accuracy': base_acc,
        }
        
        return self.results


# ============================================================
# Main Validation Runner
# ============================================================
def run_full_validation(
    model_path: str = None,
    scaler_path: str = None,
    data_path: str = None,
    pair: str = 'EUR_USD'
):
    """Run complete validation suite"""
    
    console.print(Panel(
        "[bold green]ML Engine Model Validation Suite[/bold green]\n"
        "Testing 78% accuracy claim with comprehensive validation",
        title="🔬 Validation"
    ))
    
    config = ValidationConfig()
    
    # ========== Load Model ==========
    console.print("\n📦 Loading model and data...")
    
    import tensorflow as tf
    
    # Custom loss functions that may have been used during training
    def focal_loss(gamma=2.0, alpha=0.5):
        def loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            return -tf.reduce_mean(alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt))
        return loss_fn
    
    def ewc_loss(y_true, y_pred):
        """Placeholder for EWC loss - actual regularization was applied during training"""
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    def anti_collapse_loss(y_true, y_pred):
        """Anti-collapse loss placeholder"""
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    custom_objects = {
        'focal_loss': focal_loss,
        'ewc_loss': ewc_loss,
        'anti_collapse_loss': anti_collapse_loss,
        'loss_fn': focal_loss(),
    }
    
    # Try to find model
    model_paths = [
        model_path,
        f"{config.MODEL_DIR}/transformer_direction.keras",
        "trained_data/models/transformer_direction.keras",
        "/content/ml_engine/trained_data/models/transformer_direction.keras",
    ]
    
    model = None
    for path in model_paths:
        if path and os.path.exists(path):
            try:
                # Try loading with custom objects
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                console.print(f"  ✅ Loaded model from {path}")
                break
            except Exception as e:
                # Try loading without compiling (just architecture + weights)
                console.print(f"  ⚠️ Standard load failed, trying compile=False...")
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    # Recompile with standard loss
                    model.compile(
                        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    console.print(f"  ✅ Loaded model (recompiled) from {path}")
                    break
                except Exception as e2:
                    console.print(f"  ❌ Failed to load {path}: {e2}")
    
    if model is None:
        console.print("❌ Could not find model. Please provide model_path.")
        return None
    
    # Load scaler
    scaler = None
    scaler_paths = [
        scaler_path,
        f"{config.MODEL_DIR}/direction_scaler.pkl",
        "trained_data/models/direction_scaler.pkl",
    ]
    
    for path in scaler_paths:
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            console.print(f"  ✅ Loaded scaler from {path}")
            break
    
    # ========== Run Tests ==========
    results = {}
    
    # 1. Cross-Pair Validation
    try:
        cross_validator = CrossPairValidator(model, scaler, config)
        results['cross_pair'] = cross_validator.run_all()
    except Exception as e:
        console.print(f"❌ Cross-pair validation failed: {e}")
    
    # 2-4 require prepared data - provide template for Colab
    console.print("\n" + "="*60)
    console.print("[yellow]For regime analysis and transaction cost testing,[/yellow]")
    console.print("[yellow]run this in your Colab notebook after training:[/yellow]")
    console.print("="*60)
    
    code_template = '''
# After training, run validation:
from model_validation import RegimeAnalyzer, TransactionCostAnalyzer, LeakageDetector

# 2. Regime Analysis
regime_analyzer = RegimeAnalyzer(transformer_model, scaler, ValidationConfig())
regime_results = regime_analyzer.run(test_df, X_test_seq, y_test_seq)

# 3. Transaction Cost Analysis
cost_analyzer = TransactionCostAnalyzer(ValidationConfig())
predictions = transformer_model.predict(X_test_seq)
actual_returns = test_df['close'].pct_change().values[-len(predictions):]
cost_results = cost_analyzer.run(predictions, actual_returns, pair='EUR_USD')

# 4. Feature Importance (takes a while)
# leakage_detector = LeakageDetector(transformer_model, feature_columns)
# leakage_results = leakage_detector.permutation_importance(X_test_seq, y_test_seq)
'''
    console.print(code_template)
    
    # ========== Summary ==========
    console.print(Panel("[bold green]Validation Complete[/bold green]"))
    
    return results


# ============================================================
# Quick Validation (for Colab cell)
# ============================================================
def quick_validate(model, scaler, X_test, y_test, test_df, pair='EUR_USD'):
    """
    Quick validation after training - paste this in Colab
    
    Usage:
        from model_validation import quick_validate
        results = quick_validate(transformer_model, scaler, X_test_seq, y_test, test_df)
    """
    config = ValidationConfig()
    results = {}
    
    console.print(Panel("[bold cyan]Quick Validation Suite[/bold cyan]"))
    
    # 1. Basic accuracy check
    predictions = model.predict(X_test, verbose=0)
    pred_classes = (predictions.flatten() > 0.5).astype(int)
    accuracy = np.mean(pred_classes == y_test)
    
    console.print(f"\n📊 Accuracy: {accuracy:.1%}")
    console.print(f"   Predictions: {np.mean(pred_classes):.1%} UP | Actual: {np.mean(y_test):.1%} UP")
    
    # 2. Regime Analysis
    regime_analyzer = RegimeAnalyzer(model, scaler, config)
    results['regime'] = regime_analyzer.run(test_df, X_test, y_test)
    
    # 3. Transaction Cost Analysis
    cost_analyzer = TransactionCostAnalyzer(config)
    actual_returns = test_df['close'].pct_change().values[-len(predictions):]
    results['costs'] = cost_analyzer.run(predictions.flatten(), actual_returns, pair)
    
    # 4. Summary Assessment
    console.print(Panel("[bold green]Assessment[/bold green]"))
    
    issues = []
    if accuracy < 0.55:
        issues.append("Low accuracy - may be random")
    if np.mean(pred_classes) < 0.2 or np.mean(pred_classes) > 0.8:
        issues.append("Prediction collapse detected")
    if results['costs']['sharpe_ratio'] < 0.5:
        issues.append("Low Sharpe ratio - not profitable after costs")
    if results['costs']['max_drawdown'] > 0.20:
        issues.append("High drawdown risk")
    
    if not issues:
        console.print("✅ All checks passed! Model looks viable for paper trading.")
    else:
        console.print("⚠️ Issues found:")
        for issue in issues:
            console.print(f"   • {issue}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = None
    
    run_full_validation(model_path=model_path)
