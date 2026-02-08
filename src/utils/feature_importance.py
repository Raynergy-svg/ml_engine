"""
Feature Importance Analysis using SHAP.

SHAP (SHapley Additive exPlanations) provides:
1. Global feature importance - which features matter most overall
2. Local explanations - why a specific prediction was made
3. Feature interactions - which features work together
4. Direction of impact - positive or negative effect

This helps us:
- Remove noisy features that hurt generalization
- Understand what the model is learning
- Debug unexpected predictions
- Build simpler, more robust models
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Check for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using SHAP values.

    Works with:
    - XGBoost models (TreeExplainer - fast)
    - Keras/TensorFlow models (DeepExplainer or GradientExplainer)
    - Scikit-learn models (various explainers)
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_type: str = "auto",
        background_samples: int = 100,
    ):
        """
        Initialize the analyzer.

        Args:
            model: Trained model (XGBoost, Keras, sklearn)
            feature_names: List of feature column names
            model_type: "xgboost", "keras", "sklearn", or "auto"
            background_samples: Number of background samples for SHAP
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.background_samples = background_samples
        self.explainer = None
        self.shap_values = None
        self._background_data = None

    def _detect_model_type(self) -> str:
        """Auto-detect model type."""
        model_class = type(self.model).__name__.lower()

        if "xgb" in model_class or "lightgbm" in model_class or "catboost" in model_class:
            return "tree"
        elif "keras" in model_class or "tensorflow" in model_class or "sequential" in model_class or "functional" in model_class:
            return "deep"
        elif "forest" in model_class or "tree" in model_class:
            return "tree"
        else:
            return "kernel"  # Fallback to KernelExplainer

    def fit(self, X: np.ndarray, sample_size: Optional[int] = None) -> "FeatureImportanceAnalyzer":
        """
        Fit the SHAP explainer with background data.

        Args:
            X: Feature data (2D or 3D array)
            sample_size: Number of samples to use (default: background_samples)

        Returns:
            self
        """
        sample_size = sample_size or self.background_samples

        # Handle 3D sequences (batch, seq_len, features) - flatten or use last timestep
        if X.ndim == 3:
            logger.info("Flattening 3D sequences for SHAP (using last timestep)")
            X = X[:, -1, :]  # Use last timestep

        # Sample background data
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            self._background_data = X[indices]
        else:
            self._background_data = X

        # Detect model type
        if self.model_type == "auto":
            self.model_type = self._detect_model_type()

        logger.info(f"Creating SHAP explainer for model type: {self.model_type}")

        # Create appropriate explainer
        try:
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "deep":
                # For Keras models, use GradientExplainer
                self.explainer = shap.GradientExplainer(
                    self.model,
                    self._background_data
                )
            elif self.model_type == "kernel":
                # Fallback for any model
                self.explainer = shap.KernelExplainer(
                    self.model.predict if hasattr(self.model, 'predict') else self.model,
                    self._background_data
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logger.warning(f"Failed to create {self.model_type} explainer: {e}")
            logger.info("Falling back to KernelExplainer")

            # Define prediction function
            def predict_fn(x):
                pred = self.model.predict(x)
                if isinstance(pred, dict):
                    pred = pred.get("direction", list(pred.values())[0])
                return np.asarray(pred).flatten()

            self.explainer = shap.KernelExplainer(predict_fn, self._background_data)

        return self

    def analyze(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values and feature importance.

        Args:
            X: Feature data to analyze
            max_samples: Maximum samples to analyze

        Returns:
            Dict with importance metrics
        """
        if self.explainer is None:
            raise RuntimeError("Must call fit() before analyze()")

        # Handle 3D sequences
        if X.ndim == 3:
            X = X[:, -1, :]

        # Sample if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        logger.info(f"Computing SHAP values for {len(X)} samples...")

        # Compute SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shap_values = self.explainer.shap_values(X)

        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            # Multi-output: use first output or average
            self.shap_values = self.shap_values[0] if len(self.shap_values) == 2 else self.shap_values[1]

        # Ensure 2D
        if self.shap_values.ndim > 2:
            self.shap_values = self.shap_values.reshape(self.shap_values.shape[0], -1)

        # Calculate importance metrics
        abs_shap = np.abs(self.shap_values)

        # Global importance (mean |SHAP|)
        global_importance = abs_shap.mean(axis=0)

        # Ensure feature names match
        n_features = len(global_importance)
        if len(self.feature_names) != n_features:
            logger.warning(f"Feature names mismatch: {len(self.feature_names)} names vs {n_features} SHAP values")
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": self.feature_names[:n_features],
            "importance": global_importance,
            "importance_std": abs_shap.std(axis=0),
            "mean_shap": self.shap_values.mean(axis=0),  # Direction of effect
        })
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["cumulative_importance"] = importance_df["importance"].cumsum() / importance_df["importance"].sum()

        # Feature interactions (correlation of SHAP values)
        shap_corr = pd.DataFrame(
            self.shap_values,
            columns=self.feature_names[:n_features]
        ).corr()

        return {
            "importance_df": importance_df,
            "shap_values": self.shap_values,
            "feature_data": X,
            "shap_correlation": shap_corr,
            "top_features": importance_df.head(20)["feature"].tolist(),
            "bottom_features": importance_df.tail(20)["feature"].tolist(),
        }

    def get_top_features(
        self,
        n: int = 50,
        cumulative_threshold: float = 0.9,
    ) -> List[str]:
        """
        Get top N features or features explaining threshold% of importance.

        Args:
            n: Maximum number of features
            cumulative_threshold: Cumulative importance threshold (0-1)

        Returns:
            List of top feature names
        """
        if self.shap_values is None:
            raise RuntimeError("Must call analyze() first")

        abs_shap = np.abs(self.shap_values)
        importance = abs_shap.mean(axis=0)

        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]

        # Get cumulative importance
        sorted_importance = importance[sorted_indices]
        cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()

        # Find cutoff
        n_by_threshold = np.searchsorted(cumulative, cumulative_threshold) + 1
        n_features = min(n, n_by_threshold, len(self.feature_names))

        return [self.feature_names[i] for i in sorted_indices[:n_features]]

    def get_feature_interactions(
        self,
        top_n: int = 10,
        threshold: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """
        Get pairs of features that interact (correlated SHAP values).

        Args:
            top_n: Number of top interactions to return
            threshold: Minimum correlation threshold

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        if self.shap_values is None:
            raise RuntimeError("Must call analyze() first")

        n_features = min(len(self.feature_names), self.shap_values.shape[1])
        shap_df = pd.DataFrame(
            self.shap_values[:, :n_features],
            columns=self.feature_names[:n_features]
        )
        corr = shap_df.corr()

        # Get upper triangle (excluding diagonal)
        interactions = [
            (corr.index[i], corr.columns[j], corr.iloc[i, j])
            for i in range(len(corr))
            for j in range(i + 1, len(corr))
            if abs(corr.iloc[i, j]) >= threshold
        ]

        # Sort by absolute correlation
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)

        return interactions[:top_n]

    def plot_summary(
        self,
        max_display: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create SHAP summary plot.

        Args:
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        if self.shap_values is None:
            raise RuntimeError("Must call analyze() first")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=self._background_data if self._background_data is not None else None,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {save_path}")

        plt.close()

    def plot_importance(
        self,
        max_display: int = 30,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create bar plot of feature importance.

        Args:
            max_display: Maximum features to display
            save_path: Path to save plot (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return

        if self.shap_values is None:
            raise RuntimeError("Must call analyze() first")

        plt.figure(figsize=(10, 10))
        shap.summary_plot(
            self.shap_values,
            features=self._background_data if self._background_data is not None else None,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved importance plot to {save_path}")

        plt.close()


def quick_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 50,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Quick feature importance analysis.

    Args:
        model: Trained model
        X: Feature data
        feature_names: Feature names
        top_n: Number of top features to return

    Returns:
        Tuple of (top_feature_names, importance_df)
    """
    analyzer = FeatureImportanceAnalyzer(model, feature_names)
    analyzer.fit(X)
    results = analyzer.analyze(X)

    top_features = analyzer.get_top_features(n=top_n)

    return top_features, results["importance_df"]


def select_features_by_importance(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_features: int = 50,
    method: str = "correlation",
) -> List[str]:
    """
    Select top features without SHAP (faster, simpler).

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        n_features: Number of features to select
        method: "correlation", "mutual_info", or "variance"

    Returns:
        List of selected feature names
    """
    X = df[feature_cols].values
    y = df[target_col].values

    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if method == "correlation":
        # Absolute correlation with target
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        correlations = np.nan_to_num(correlations, 0)
        top_indices = np.argsort(correlations)[::-1][:n_features]

    elif method == "variance":
        # Feature variance (remove low-variance features)
        variances = X.var(axis=0)
        top_indices = np.argsort(variances)[::-1][:n_features]

    elif method == "mutual_info":
        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X, (y > 0.5).astype(int), random_state=42)
            top_indices = np.argsort(mi)[::-1][:n_features]
        except ImportError:
            logger.warning("sklearn not available, falling back to correlation")
            return select_features_by_importance(df, feature_cols, target_col, n_features, "correlation")
    else:
        raise ValueError(f"Unknown method: {method}")

    return [feature_cols[i] for i in top_indices]


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Feature Importance Test")
    print("-" * 50)

    if not SHAP_AVAILABLE:
        print("SHAP not installed. Run: pip install shap")
    else:
        # Test with simple sklearn model
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        n_samples = 500
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        # Make some features important
        y = ((X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5) > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Analyze
        analyzer = FeatureImportanceAnalyzer(model, feature_names)
        analyzer.fit(X)
        results = analyzer.analyze(X)

        print("\nTop 10 Features:")
        print(results["importance_df"].head(10)[["feature", "importance", "rank"]].to_string(index=False))

        top_features = analyzer.get_top_features(n=5)
        print(f"\nTop 5 features: {top_features}")

        interactions = analyzer.get_feature_interactions(top_n=5, threshold=0.3)
        print(f"\nTop interactions: {interactions}")

        print("\nFeature importance test complete!")
