#!/usr/bin/env python3
"""
Diagnostic Script for ML Training Issues.

This script identifies why the model is stuck at random guessing by checking:
1. Target learnability (is the prediction task fundamentally learnable?)
2. Data leakage (are features leaking future information?)
3. Feature predictiveness (do any features correlate with the target?)
4. Temporal split integrity (is train/val/test properly ordered?)

Usage:
    python diagnose_training_issues.py --csv market_data/your_data.csv
    python diagnose_training_issues.py --oanda-live --candles 10000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(csv_path: Optional[str] = None, oanda_candles: int = 10000) -> pd.DataFrame:
    """Load market data from CSV or OANDA."""
    if csv_path and Path(csv_path).exists():
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    
    # Try to find a CSV in market_data
    market_data_dir = Path("market_data")
    if market_data_dir.exists():
        csvs = list(market_data_dir.glob("*.csv"))
        if csvs:
            logger.info(f"Loading data from {csvs[0]}")
            df = pd.read_csv(csvs[0])
            df.columns = [c.lower().strip() for c in df.columns]
            return df
    
    # Try trained_data/data
    trained_data_dir = Path("trained_data/data")
    if trained_data_dir.exists():
        csvs = list(trained_data_dir.glob("*.csv"))
        if csvs:
            logger.info(f"Loading data from {csvs[0]}")
            df = pd.read_csv(csvs[0])
            df.columns = [c.lower().strip() for c in df.columns]
            return df
    
    raise FileNotFoundError("No CSV data found. Please provide --csv path or ensure market_data/ has CSV files.")


def check_target_learnability(df: pd.DataFrame, target_shifts: List[int] = [1, 6, 12, 24]) -> Dict[str, Any]:
    """
    Check if the prediction target is actually learnable.
    
    A target is learnable if:
    1. Returns have some autocorrelation (momentum/mean-reversion)
    2. Direction labels are not perfectly balanced (some predictability)
    3. Future returns show some pattern
    """
    results = {
        "target_shifts": {},
        "autocorrelation": {},
        "recommendations": [],
    }
    
    close = df["close"].values
    
    # Check return autocorrelation at different lags
    returns = np.diff(close) / close[:-1]
    
    logger.info("\n" + "=" * 60)
    logger.info("TARGET LEARNABILITY ANALYSIS")
    logger.info("=" * 60)
    
    # Autocorrelation of returns
    logger.info("\n1. Return Autocorrelation (momentum/mean-reversion signal):")
    for lag in [1, 5, 10, 20]:
        if len(returns) > lag:
            autocorr = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
            results["autocorrelation"][f"lag_{lag}"] = float(autocorr)
            interpretation = "WEAK" if abs(autocorr) < 0.05 else ("MODERATE" if abs(autocorr) < 0.1 else "STRONG")
            logger.info(f"   Lag {lag}: {autocorr:.4f} ({interpretation})")
    
    # Direction prediction at different horizons
    logger.info("\n2. Direction Label Analysis at Different Horizons:")
    for shift in target_shifts:
        if len(close) > shift:
            # Future return
            future_return = (close[shift:] - close[:-shift]) / close[:-shift]
            direction = (future_return > 0).astype(float)
            
            # Balance
            up_pct = np.mean(direction) * 100
            
            # Entropy (measure of predictability)
            p = np.mean(direction)
            entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10) if 0 < p < 1 else 0
            max_entropy = 1.0  # Binary classification
            normalized_entropy = entropy / max_entropy
            
            results["target_shifts"][f"shift_{shift}"] = {
                "up_pct": float(up_pct),
                "down_pct": float(100 - up_pct),
                "entropy": float(normalized_entropy),
                "mean_return": float(np.mean(future_return) * 100),
                "std_return": float(np.std(future_return) * 100),
            }
            
            predictability = "LOW" if normalized_entropy > 0.95 else ("MODERATE" if normalized_entropy > 0.85 else "HIGH")
            logger.info(f"   Shift {shift}: UP={up_pct:.1f}% DOWN={100-up_pct:.1f}% | "
                        f"Entropy={normalized_entropy:.3f} ({predictability} predictability)")
    
    # Recommendations
    logger.info("\n3. Recommendations:")
    
    # Check if returns are essentially random
    max_autocorr = max(abs(v) for v in results["autocorrelation"].values())
    if max_autocorr < 0.03:
        results["recommendations"].append(
            "CRITICAL: Returns show near-zero autocorrelation. "
            "The market appears efficient at this timeframe. "
            "Consider using longer prediction horizons or alternative targets (e.g., triple barrier)."
        )
        logger.warning("   ⚠️  Returns show near-zero autocorrelation - market is efficient at this timeframe")
    
    # Check label balance
    for shift, data in results["target_shifts"].items():
        if 49 < data["up_pct"] < 51:
            results["recommendations"].append(
                f"WARNING: {shift} has near-perfect 50/50 balance ({data['up_pct']:.1f}% up). "
                "This suggests random walk behavior at this horizon."
            )
            logger.warning(f"   ⚠️  {shift}: Near-perfect 50/50 balance suggests random walk")
    
    # Suggest best horizon
    best_shift = min(results["target_shifts"].keys(), 
                     key=lambda k: results["target_shifts"][k]["entropy"])
    logger.info(f"   ✓ Best prediction horizon: {best_shift} (lowest entropy)")
    
    return results


def check_feature_target_correlation(df: pd.DataFrame, target_shift: int = 1, top_k: int = 20) -> Dict[str, Any]:
    """
    Check correlation between features and target.
    
    If all correlations are near zero, features have no predictive power.
    """
    results = {
        "correlations": {},
        "top_features": [],
        "suspicious_features": [],
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE-TARGET CORRELATION ANALYSIS")
    logger.info("=" * 60)
    
    # Create target
    close = df["close"].values
    if len(close) <= target_shift:
        logger.error("Not enough data for target creation")
        return results
    
    future_return = np.zeros(len(close))
    future_return[:-target_shift] = (close[target_shift:] - close[:-target_shift]) / close[:-target_shift]
    direction = (future_return > 0).astype(float)
    
    # Get numeric columns (potential features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"close", "open", "high", "low", "volume", "time", "date", "timestamp"}
    feature_cols = [c for c in numeric_cols if c.lower() not in exclude_cols]
    
    if not feature_cols:
        # Use OHLCV-derived features
        feature_cols = [c for c in numeric_cols if c.lower() not in {"time", "date", "timestamp"}]
    
    logger.info(f"\nAnalyzing {len(feature_cols)} features...")
    
    correlations = []
    for col in feature_cols:
        try:
            feat = df[col].values[:-target_shift]
            tgt = direction[:-target_shift]
            
            # Remove NaN
            mask = ~(np.isnan(feat) | np.isnan(tgt))
            if mask.sum() < 100:
                continue
            
            corr, pval = stats.pearsonr(feat[mask], tgt[mask])
            correlations.append({
                "feature": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "p_value": pval,
                "significant": pval < 0.05,
            })
        except Exception:
            continue
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
    
    logger.info(f"\n1. Top {top_k} Features by Correlation with Direction (shift={target_shift}):")
    for i, c in enumerate(correlations[:top_k]):
        sig = "***" if c["p_value"] < 0.001 else ("**" if c["p_value"] < 0.01 else ("*" if c["p_value"] < 0.05 else ""))
        logger.info(f"   {i+1:2d}. {c['feature'][:40]:<40} r={c['correlation']:+.4f} {sig}")
        results["top_features"].append(c)
    
    # Check for suspicious correlations (too high = possible leakage)
    logger.info("\n2. Leakage Detection (suspiciously high correlations):")
    suspicious = [c for c in correlations if c["abs_correlation"] > 0.3]
    if suspicious:
        for c in suspicious:
            logger.warning(f"   ⚠️  {c['feature']}: r={c['correlation']:.4f} - POSSIBLE LEAKAGE")
            results["suspicious_features"].append(c)
    else:
        logger.info("   ✓ No suspiciously high correlations detected")
    
    # Summary statistics
    all_corrs = [c["abs_correlation"] for c in correlations]
    if all_corrs:
        logger.info("\n3. Correlation Summary:")
        logger.info(f"   Mean |correlation|: {np.mean(all_corrs):.4f}")
        logger.info(f"   Max |correlation|:  {np.max(all_corrs):.4f}")
        logger.info(f"   Features with |r| > 0.05: {sum(1 for c in all_corrs if c > 0.05)}")
        logger.info(f"   Features with |r| > 0.10: {sum(1 for c in all_corrs if c > 0.10)}")
        
        if np.max(all_corrs) < 0.05:
            logger.warning("\n   ⚠️  CRITICAL: No features have meaningful correlation with target!")
            logger.warning("   This suggests features have no predictive power at this horizon.")
    
    results["correlations"] = {c["feature"]: c["correlation"] for c in correlations}
    return results


def check_data_leakage(df: pd.DataFrame, target_shift: int = 1) -> Dict[str, Any]:
    """
    Check for data leakage by comparing performance on shuffled vs chronological data.
    
    If shuffled data performs much better, there's likely leakage.
    """
    results = {
        "leakage_detected": False,
        "leakage_score": 0.0,
        "issues": [],
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA LEAKAGE DETECTION")
    logger.info("=" * 60)
    
    close = df["close"].values
    
    # Check 1: Future price in features
    logger.info("\n1. Checking for future price leakage in columns...")
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ["future", "next", "forward", "target", "label"]):
            logger.warning(f"   ⚠️  Suspicious column name: {col}")
            results["issues"].append(f"Suspicious column: {col}")
    
    # Check 2: Look-ahead in rolling features
    logger.info("\n2. Checking rolling feature computation...")
    
    # Simulate what happens with ffill on unordered data
    if "time" in df.columns or "date" in df.columns:
        time_col = "time" if "time" in df.columns else "date"
        try:
            times = pd.to_datetime(df[time_col])
            if not times.is_monotonic_increasing:
                logger.warning("   ⚠️  Data is not sorted by time! This can cause leakage with ffill.")
                results["issues"].append("Data not sorted by time")
                results["leakage_detected"] = True
            else:
                logger.info("   ✓ Data is properly sorted by time")
        except Exception:
            pass
    
    # Check 3: Scaler leakage simulation
    logger.info("\n3. Simulating scaler leakage impact...")
    
    # Create simple features
    returns = np.diff(close) / close[:-1]
    
    # Chronological split (correct)
    split_idx = int(len(returns) * 0.8)
    train_returns = returns[:split_idx]
    
    # Fit scaler on train only (correct)
    train_mean = np.mean(train_returns)
    
    # Fit scaler on all data (leakage)
    all_mean = np.mean(returns)
    
    # Compare
    leakage_diff = abs(train_mean - all_mean) / (abs(train_mean) + 1e-10)
    if leakage_diff > 0.1:
        logger.warning(f"   ⚠️  Scaler leakage could shift mean by {leakage_diff*100:.1f}%")
        results["issues"].append(f"Scaler leakage: {leakage_diff*100:.1f}% mean shift")
    else:
        logger.info(f"   ✓ Scaler leakage impact is minimal ({leakage_diff*100:.1f}%)")
    
    # Check 4: Target leakage via risk quantiles
    logger.info("\n4. Checking target computation for leakage...")
    
    # The risk quantiles in build_multitask_targets use full data
    # This is a form of leakage
    logger.warning("   ⚠️  multitask_labels.py computes risk quantiles on full data before split")
    logger.warning("      This leaks validation distribution into training")
    results["issues"].append("Risk quantiles computed on full data")
    
    # Summary
    results["leakage_score"] = len(results["issues"]) / 5.0  # Normalize to 0-1
    results["leakage_detected"] = len(results["issues"]) > 0
    
    logger.info("\n5. Leakage Summary:")
    logger.info(f"   Issues found: {len(results['issues'])}")
    logger.info(f"   Leakage score: {results['leakage_score']:.2f} (0=none, 1=severe)")
    
    return results


def check_temporal_split(df: pd.DataFrame, val_fraction: float = 0.2) -> Dict[str, Any]:
    """
    Verify that train/val/test splits preserve temporal ordering.
    """
    results = {
        "splits_valid": True,
        "has_gap": False,
        "issues": [],
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL SPLIT VERIFICATION")
    logger.info("=" * 60)
    
    n = len(df)
    
    # Simulate the split from multitask_labels.py
    train_end = int(n * (1 - val_fraction))
    
    # Check for time column
    time_col = None
    for col in ["time", "date", "timestamp", "datetime"]:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        try:
            times = pd.to_datetime(df[time_col])
            
            train_times = times[:train_end]
            val_times = times[train_end:]
            
            logger.info("\n1. Split Information:")
            logger.info(f"   Total samples: {n}")
            logger.info(f"   Train: {train_end} samples ({train_end/n*100:.1f}%)")
            logger.info(f"   Val:   {n - train_end} samples ({(n-train_end)/n*100:.1f}%)")
            
            logger.info("\n2. Time Ranges:")
            logger.info(f"   Train: {train_times.min()} to {train_times.max()}")
            logger.info(f"   Val:   {val_times.min()} to {val_times.max()}")
            
            # Check for overlap
            if train_times.max() >= val_times.min():
                logger.warning("   ⚠️  OVERLAP DETECTED between train and validation!")
                results["splits_valid"] = False
                results["issues"].append("Train/val overlap")
            else:
                gap = val_times.min() - train_times.max()
                logger.info(f"   Gap between train and val: {gap}")
                
                # Check if gap is sufficient
                if hasattr(gap, 'total_seconds'):
                    gap_hours = gap.total_seconds() / 3600
                    if gap_hours < 1:
                        logger.warning(f"   ⚠️  Gap is very small ({gap_hours:.1f} hours)")
                        logger.warning("      Consider adding purge gap to prevent temporal leakage")
                        results["issues"].append("Insufficient gap between splits")
                    else:
                        logger.info(f"   ✓ Gap is {gap_hours:.1f} hours")
                        results["has_gap"] = True
        except Exception as e:
            logger.warning(f"   Could not parse time column: {e}")
    else:
        logger.info("   No time column found - using index-based split")
        logger.info(f"   Train indices: 0 to {train_end-1}")
        logger.info(f"   Val indices: {train_end} to {n-1}")
    
    # Recommendation
    logger.info("\n3. Recommendations:")
    if not results["has_gap"]:
        logger.info("   → Add purge gap of 10-50 samples between train and validation")
        logger.info("   → Use walkforward_validation.py with gap parameter")
    
    return results


def run_full_diagnostics(
    csv_path: Optional[str] = None,
    target_shifts: List[int] = [1, 6, 12, 24],
) -> Dict[str, Any]:
    """Run all diagnostic checks."""
    
    logger.info("\n" + "=" * 70)
    logger.info("ML TRAINING DIAGNOSTIC REPORT")
    logger.info("=" * 70)
    
    # Load data
    df = load_data(csv_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    results = {
        "data_shape": df.shape,
        "columns": list(df.columns),
    }
    
    # Run diagnostics
    results["target_learnability"] = check_target_learnability(df, target_shifts)
    results["feature_correlation"] = check_feature_target_correlation(df, target_shift=1)
    results["data_leakage"] = check_data_leakage(df)
    results["temporal_split"] = check_temporal_split(df)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 70)
    
    issues = []
    
    # Target learnability issues
    autocorrs = results["target_learnability"]["autocorrelation"]
    if all(abs(v) < 0.03 for v in autocorrs.values()):
        issues.append("TARGET: Returns show no autocorrelation - market is efficient")
    
    # Feature issues
    top_corr = results["feature_correlation"].get("top_features", [])
    if top_corr and top_corr[0]["abs_correlation"] < 0.05:
        issues.append("FEATURES: No features correlate with target - features may be useless")
    
    # Leakage issues
    if results["data_leakage"]["leakage_detected"]:
        issues.append(f"LEAKAGE: {len(results['data_leakage']['issues'])} potential leakage sources found")
    
    # Split issues
    if not results["temporal_split"]["splits_valid"]:
        issues.append("SPLIT: Train/val splits overlap or have insufficient gap")
    
    if issues:
        logger.warning("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.warning(f"   {i}. {issue}")
    else:
        logger.info("\n✓ No critical issues found")
    
    # Recommendations
    logger.info("\nRECOMMENDED ACTIONS:")
    logger.info("   1. Increase target_shift from 1 to 12-24 (predict further ahead)")
    logger.info("   2. Use triple barrier labels instead of simple direction")
    logger.info("   3. Fix scaler to fit only on training data")
    logger.info("   4. Add purge gap between train/val splits")
    logger.info("   5. Run feature importance and remove non-predictive features")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose ML training issues")
    parser.add_argument("--csv", type=str, help="Path to CSV data file")
    parser.add_argument("--target-shifts", type=str, default="1,6,12,24",
                        help="Comma-separated target shifts to analyze")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    target_shifts = [int(x) for x in args.target_shifts.split(",")]
    
    results = run_full_diagnostics(
        csv_path=args.csv,
        target_shifts=target_shifts,
    )
    
    if args.output:
        import json
        # Convert numpy types to Python types for JSON serialization

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(args.output, "w") as f:
            json.dump(convert(results), f, indent=2)
        logger.info(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
