#!/usr/bin/env python3
"""
Feature Importance Analysis for Trading Model.

This script analyzes which features are actually useful for prediction
and identifies features that should be removed to improve generalization.

Usage:
    python analyze_feature_importance.py --csv market_data/your_data.csv
    python analyze_feature_importance.py --top 30 --output feature_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


def load_and_engineer_features(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load data and apply feature engineering."""
    from feature_engineering import FeatureEngineering
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    logger.info(f"Applying feature engineering...")
    fe = FeatureEngineering({})
    df = fe.create_features(df, include_all=True)
    
    # Get numeric feature columns
    exclude_cols = {"close", "open", "high", "low", "volume", "time", "date", "timestamp", "datetime"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    logger.info(f"Generated {len(feature_cols)} features")
    return df, feature_cols


def compute_correlation_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_shift: int = 12,
) -> pd.DataFrame:
    """
    Compute feature importance based on correlation with future direction.
    
    This is a simple but effective method that doesn't require a trained model.
    """
    logger.info(f"Computing correlation importance (target_shift={target_shift})...")
    
    close = df["close"].values
    n = len(close)
    
    if n <= target_shift:
        raise ValueError("Not enough data for target computation")
    
    # Create direction target
    future_return = np.zeros(n)
    future_return[:-target_shift] = (close[target_shift:] - close[:-target_shift]) / close[:-target_shift]
    direction = (future_return > 0).astype(float)
    
    results = []
    for col in feature_cols:
        try:
            feat = df[col].values[:-target_shift]
            tgt = direction[:-target_shift]
            
            # Remove NaN
            mask = ~(np.isnan(feat) | np.isnan(tgt) | np.isinf(feat))
            if mask.sum() < 100:
                continue
            
            # Pearson correlation
            corr, pval = stats.pearsonr(feat[mask], tgt[mask])
            
            # Spearman correlation (rank-based, more robust)
            spearman_corr, spearman_pval = stats.spearmanr(feat[mask], tgt[mask])
            
            results.append({
                "feature": col,
                "pearson_corr": corr,
                "pearson_pval": pval,
                "spearman_corr": spearman_corr,
                "spearman_pval": spearman_pval,
                "abs_pearson": abs(corr),
                "abs_spearman": abs(spearman_corr),
                "significant": pval < 0.05,
            })
        except Exception as e:
            logger.debug(f"Skipping {col}: {e}")
            continue
    
    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values("abs_pearson", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    return importance_df


def compute_mutual_information_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_shift: int = 12,
) -> pd.DataFrame:
    """
    Compute feature importance using mutual information.
    
    Mutual information captures non-linear relationships that correlation misses.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        logger.warning("sklearn not available, skipping mutual information")
        return pd.DataFrame()
    
    logger.info(f"Computing mutual information importance...")
    
    close = df["close"].values
    n = len(close)
    
    # Create direction target
    future_return = np.zeros(n)
    future_return[:-target_shift] = (close[target_shift:] - close[:-target_shift]) / close[:-target_shift]
    direction = (future_return > 0).astype(int)
    
    # Prepare feature matrix
    X = df[feature_cols].values[:-target_shift]
    y = direction[:-target_shift]
    
    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    # Replace remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    
    results = []
    for i, col in enumerate(feature_cols):
        results.append({
            "feature": col,
            "mutual_info": mi_scores[i],
        })
    
    mi_df = pd.DataFrame(results)
    mi_df = mi_df.sort_values("mutual_info", ascending=False)
    mi_df["mi_rank"] = range(1, len(mi_df) + 1)
    
    return mi_df


def compute_variance_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute feature variance (low variance = likely useless).
    """
    logger.info("Computing feature variance...")
    
    results = []
    for col in feature_cols:
        try:
            values = df[col].dropna().values
            variance = np.var(values)
            std = np.std(values)
            
            # Coefficient of variation (normalized variance)
            mean_val = np.mean(values)
            cv = std / (abs(mean_val) + 1e-10)
            
            results.append({
                "feature": col,
                "variance": variance,
                "std": std,
                "coef_variation": cv,
            })
        except Exception:
            continue
    
    var_df = pd.DataFrame(results)
    var_df = var_df.sort_values("variance", ascending=False)
    
    return var_df


def identify_redundant_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    correlation_threshold: float = 0.95,
) -> List[Tuple[str, str, float]]:
    """
    Identify highly correlated feature pairs (redundant features).
    """
    logger.info(f"Identifying redundant features (threshold={correlation_threshold})...")
    
    # Compute correlation matrix
    X = df[feature_cols].values
    
    # Remove NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    corr_matrix = np.corrcoef(X.T)
    
    redundant_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr_matrix[i, j]) >= correlation_threshold:
                redundant_pairs.append((
                    feature_cols[i],
                    feature_cols[j],
                    corr_matrix[i, j]
                ))
    
    # Sort by correlation
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return redundant_pairs


def recommend_features(
    correlation_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    redundant_pairs: List[Tuple[str, str, float]],
    top_n: int = 50,
    min_correlation: float = 0.02,
    min_variance_percentile: float = 10,
) -> Dict[str, Any]:
    """
    Generate feature recommendations based on all analyses.
    """
    logger.info("Generating feature recommendations...")
    
    recommendations = {
        "keep": [],
        "remove_low_importance": [],
        "remove_redundant": [],
        "remove_low_variance": [],
    }
    
    # Features to keep: high correlation and/or high MI
    if not correlation_df.empty:
        high_corr = correlation_df[correlation_df["abs_pearson"] >= min_correlation]["feature"].tolist()
    else:
        high_corr = []
    
    if not mi_df.empty:
        # Top 50% by MI
        mi_threshold = mi_df["mutual_info"].quantile(0.5)
        high_mi = mi_df[mi_df["mutual_info"] >= mi_threshold]["feature"].tolist()
    else:
        high_mi = []
    
    # Union of high correlation and high MI
    useful_features = set(high_corr) | set(high_mi)
    
    # Remove redundant features (keep one from each pair)
    redundant_to_remove = set()
    for f1, f2, corr in redundant_pairs:
        # Keep the one with higher importance
        if f1 in useful_features and f2 in useful_features:
            # Keep f1, remove f2 (arbitrary choice)
            redundant_to_remove.add(f2)
        elif f2 in useful_features:
            redundant_to_remove.add(f1)
        elif f1 in useful_features:
            redundant_to_remove.add(f2)
        else:
            # Neither is useful, remove both from consideration
            pass
    
    # Remove low variance features
    if not variance_df.empty:
        var_threshold = variance_df["variance"].quantile(min_variance_percentile / 100)
        low_var = variance_df[variance_df["variance"] < var_threshold]["feature"].tolist()
    else:
        low_var = []
    
    # Final recommendations
    all_features = set(correlation_df["feature"].tolist()) if not correlation_df.empty else set()
    
    recommendations["keep"] = [f for f in useful_features if f not in redundant_to_remove][:top_n]
    recommendations["remove_low_importance"] = [f for f in all_features if f not in useful_features]
    recommendations["remove_redundant"] = list(redundant_to_remove)
    recommendations["remove_low_variance"] = low_var
    
    # Summary stats
    recommendations["summary"] = {
        "total_features": len(all_features),
        "useful_features": len(useful_features),
        "redundant_features": len(redundant_to_remove),
        "low_variance_features": len(low_var),
        "recommended_features": len(recommendations["keep"]),
    }
    
    return recommendations


def run_feature_analysis(
    csv_path: str,
    target_shift: int = 12,
    top_n: int = 50,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete feature analysis."""
    
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 70)
    
    # Load and engineer features
    df, feature_cols = load_and_engineer_features(csv_path)
    
    logger.info(f"\nAnalyzing {len(feature_cols)} features with target_shift={target_shift}")
    
    # Compute various importance metrics
    correlation_df = compute_correlation_importance(df, feature_cols, target_shift)
    mi_df = compute_mutual_information_importance(df, feature_cols, target_shift)
    variance_df = compute_variance_importance(df, feature_cols)
    redundant_pairs = identify_redundant_features(df, feature_cols)
    
    # Generate recommendations
    recommendations = recommend_features(
        correlation_df, mi_df, variance_df, redundant_pairs, top_n=top_n
    )
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    logger.info(f"\n1. Top {min(20, len(correlation_df))} Features by Correlation:")
    if not correlation_df.empty:
        for _, row in correlation_df.head(20).iterrows():
            sig = "***" if row["pearson_pval"] < 0.001 else ("**" if row["pearson_pval"] < 0.01 else ("*" if row["pearson_pval"] < 0.05 else ""))
            logger.info(f"   {row['rank']:2d}. {row['feature'][:40]:<40} r={row['pearson_corr']:+.4f} {sig}")
    
    logger.info(f"\n2. Top {min(20, len(mi_df))} Features by Mutual Information:")
    if not mi_df.empty:
        for _, row in mi_df.head(20).iterrows():
            logger.info(f"   {row['mi_rank']:2d}. {row['feature'][:40]:<40} MI={row['mutual_info']:.4f}")
    
    logger.info(f"\n3. Redundant Feature Pairs (correlation > 0.95):")
    if redundant_pairs:
        for f1, f2, corr in redundant_pairs[:10]:
            logger.info(f"   {f1[:30]:<30} <-> {f2[:30]:<30} r={corr:.3f}")
    else:
        logger.info("   None found")
    
    logger.info(f"\n4. Summary:")
    for key, value in recommendations["summary"].items():
        logger.info(f"   {key}: {value}")
    
    logger.info(f"\n5. Recommended Features ({len(recommendations['keep'])}):")
    for i, feat in enumerate(recommendations["keep"][:30], 1):
        logger.info(f"   {i:2d}. {feat}")
    
    if len(recommendations["keep"]) > 30:
        logger.info(f"   ... and {len(recommendations['keep']) - 30} more")
    
    # Compile results
    results = {
        "correlation_importance": correlation_df.to_dict(orient="records") if not correlation_df.empty else [],
        "mutual_info_importance": mi_df.to_dict(orient="records") if not mi_df.empty else [],
        "redundant_pairs": [(f1, f2, float(c)) for f1, f2, c in redundant_pairs],
        "recommendations": recommendations,
        "config": {
            "csv_path": csv_path,
            "target_shift": target_shift,
            "top_n": top_n,
            "n_features_analyzed": len(feature_cols),
        },
    }
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance for trading model")
    parser.add_argument("--csv", type=str, help="Path to CSV data file")
    parser.add_argument("--target-shift", type=int, default=12, help="Target prediction horizon")
    parser.add_argument("--top", type=int, default=50, help="Number of top features to recommend")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Find CSV if not provided
    csv_path = args.csv
    if not csv_path:
        for search_dir in ["market_data", "trained_data/data"]:
            search_path = Path(search_dir)
            if search_path.exists():
                csvs = list(search_path.glob("*.csv"))
                if csvs:
                    csv_path = str(csvs[0])
                    break
    
    if not csv_path:
        logger.error("No CSV file found. Please provide --csv path")
        sys.exit(1)
    
    results = run_feature_analysis(
        csv_path=csv_path,
        target_shift=args.target_shift,
        top_n=args.top,
        output_path=args.output,
    )
    
    # Print final recommendation
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION")
    logger.info("=" * 70)
    logger.info(f"Use the top {len(results['recommendations']['keep'])} features for training.")
    logger.info("This can be done by passing --top-features flag to training:")
    logger.info(f"  buddy train --top-features {len(results['recommendations']['keep'])}")
    
    return results


if __name__ == "__main__":
    main()

