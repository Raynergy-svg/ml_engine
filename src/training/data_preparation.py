"""
Data Preparation Pipeline for Buddy Model Training.

Extracted from _train_buddy_impl() to provide a reusable, testable data
preparation pipeline. Handles:
- Data loading (OANDA API or CSV)
- Feature engineering and normalization
- Train/validation splitting
- Sequence preparation for time-series models
- Preprocessing (scaling, NaN handling)

Usage:
    from src.training.data_preparation import DataPreparationPipeline, PreparedData

    pipeline = DataPreparationPipeline(config=cfg, console=console)
    prepared = pipeline.prepare_for_training(options)
    # Now use prepared.X_train, prepared.y_train, etc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from rich.panel import Panel

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS FOR DEPENDENCY INJECTION
# =============================================================================

class ConsoleLike(Protocol):
    """Protocol for Rich console-like objects."""
    def print(self, *args: Any, **kwargs: Any) -> Any:
        ...


class OandaFetchOptionsLike(Protocol):
    """Protocol for OANDA fetch options."""
    instrument: str
    granularity: str
    candles: int
    price: str
    save_csv: Optional[str]


class TrainingOptionsLike(Protocol):
    """Protocol for training options (subset needed for data prep)."""
    oanda_fetch: Optional[OandaFetchOptionsLike]
    seq_len: int
    all_features: bool
    pca_components: Optional[int]
    seed: int


class AdvancedOptionsLike(Protocol):
    """Protocol for advanced training options."""
    train_smoothing: bool
    median_window: Optional[int]
    vol_norm_window: Optional[int]
    min_volume: Optional[int]
    spread_filter: bool
    spread_pctl: float
    spread_mult: float
    top_features: Optional[int]
    tier2_calibrate: bool


# =============================================================================
# PREPARED DATA CONTAINER
# =============================================================================

@dataclass
class PreparedData:
    """Container for prepared training data.

    Attributes:
        X_train: Training features (n_train, n_features) - 2D standardized features
        y_train: Training labels (n_train,) - direction labels (0/1)
        X_val: Validation features (n_val, n_features)
        y_val: Validation labels (n_val,)
        feature_names: List of feature column names
        scaler: StandardScaler-like object with mean_, scale_ attributes
        df: Original DataFrame before feature extraction

    Extended attributes for advanced training:
        y_train_conf: Confidence labels for training (Tier-2 TP/SL outcome)
        y_val_conf: Confidence labels for validation
        conf_weights_train: Sample weights for confidence labels
        conf_weights_val: Sample weights for confidence labels
        closes_train: Close prices for training period
        closes_val: Close prices for validation period
        ohlc_df: OHLC DataFrame aligned to features (for Tier-2 labeling)
        pca_model: Optional PCA model if dimensionality reduction applied
        ranked_features: Feature ranking for curriculum learning
        instrument: Inferred instrument name
    """
    # Core data
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]
    scaler: Any  # StandardScaler-compatible object
    df: pd.DataFrame

    # Extended data for advanced training
    y_train_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    y_val_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    conf_weights_train: np.ndarray = field(default_factory=lambda: np.array([]))
    conf_weights_val: np.ndarray = field(default_factory=lambda: np.array([]))
    closes_train: np.ndarray = field(default_factory=lambda: np.array([]))
    closes_val: np.ndarray = field(default_factory=lambda: np.array([]))
    ohlc_df: Optional[pd.DataFrame] = None
    pca_model: Any = None
    ranked_features: Optional[List[str]] = None
    instrument: Optional[str] = None

    # Sequence parameters
    seq_len: int = 60
    split_ratio: float = 0.8

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.X_val)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X_train.shape[1] if len(self.X_train) > 0 else 0

    @property
    def train_dir_pos_rate(self) -> float:
        """Fraction of positive (up) direction labels in training set."""
        return float(np.mean(self.y_train)) if len(self.y_train) > 0 else 0.0

    @property
    def val_dir_pos_rate(self) -> float:
        """Fraction of positive (up) direction labels in validation set."""
        return float(np.mean(self.y_val)) if len(self.y_val) > 0 else 0.0

    def validate(self) -> List[str]:
        """Validate prepared data and return list of warnings."""
        warnings = []

        if len(self.X_train) == 0:
            warnings.append("Empty training set")
        if len(self.X_val) == 0:
            warnings.append("Empty validation set")
        if not np.isfinite(self.X_train).all():
            warnings.append("Non-finite values in training features")
        if not np.isfinite(self.X_val).all():
            warnings.append("Non-finite values in validation features")
        if not np.isfinite(self.y_train).all():
            warnings.append("Non-finite values in training labels")
        if not np.isfinite(self.y_val).all():
            warnings.append("Non-finite values in validation labels")

        # Check for extreme class imbalance
        if self.train_dir_pos_rate < 0.1 or self.train_dir_pos_rate > 0.9:
            warnings.append(f"Extreme class imbalance in training: {self.train_dir_pos_rate:.1%} positive")

        return warnings


@dataclass
class ScalerState:
    """Container for scaler state (mean and std)."""
    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse standardization."""
        return X * self.scale_ + self.mean_


# =============================================================================
# DATA PREPARATION PIPELINE
# =============================================================================

class DataPreparationPipeline:
    """Pipeline for preparing training data from raw sources.

    This class encapsulates all data preparation logic extracted from
    _train_buddy_impl(). It handles:

    1. Data Loading:
       - OANDA API fetch (with pagination for >5000 candles)
       - CSV file loading
       - Multi-pair data loading for foundation models

    2. Data Validation:
       - Required column checks (OHLCV)
       - Time column normalization
       - Quality filters (min volume, spread)

    3. Feature Engineering:
       - Full feature engineering via FeatureEngineering class
       - Normalized feature computation via modular_data_loaders
       - Noise reduction (EMA smoothing, median filtering)

    4. Train/Val Split:
       - Chronological split (no shuffle for time-series)
       - Default 80/20 split

    5. Preprocessing:
       - Standardization (fit on train only to avoid leakage)
       - NaN/Inf handling
       - Optional PCA dimensionality reduction
       - Feature ranking for curriculum learning
    """

    def __init__(
        self,
        config: Dict[str, Any],
        console: Optional[ConsoleLike] = None,
        oanda_fetch_to_csv: Optional[Callable[[Any], str]] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Configuration dictionary (from YAML config file)
            console: Rich console for output (optional)
            oanda_fetch_to_csv: Function to fetch OANDA data and save to CSV
        """
        self.config = config
        self.console = console or _DummyConsole()
        self.oanda_fetch_to_csv = oanda_fetch_to_csv

        # Extract relevant config sections
        self.buddy_cfg = config.get("buddy", {}) if isinstance(config, dict) else {}
        self.fe_cfg = config.get("feature_engineering", {}) if isinstance(config, dict) else {}

    def prepare_for_training(
        self,
        options: TrainingOptionsLike,
        *,
        csv_path: Optional[str] = None,
        advanced: Optional[AdvancedOptionsLike] = None,
        multi_pair: bool = False,
        foundation_pairs: Optional[str] = None,
    ) -> PreparedData:
        """Prepare data for training from options.

        This is the main entry point that orchestrates the full pipeline.

        Args:
            options: Training options containing data source and parameters
            csv_path: Optional path to CSV file (overrides options)
            advanced: Advanced training options for filters and Tier-2 labeling
            multi_pair: Enable multi-pair foundation model training
            foundation_pairs: Comma-separated list of pairs for multi-pair mode

        Returns:
            PreparedData containing all prepared training data
        """
        t_start = time.perf_counter()

        # Extract parameters from options
        oanda_fetch = getattr(options, 'oanda_fetch', None)
        seq_len = getattr(options, 'seq_len', 60)
        all_features = getattr(options, 'all_features', True)
        pca_components = getattr(options, 'pca_components', None)
        self._seed = getattr(options, 'seed', 42)

        # Extract advanced options
        train_smoothing = True
        median_window = None
        vol_norm_window = None
        min_volume = None
        spread_filter = False
        spread_pctl = 0.99
        spread_mult = 3.0
        top_features = None
        tier2_calibrate = True

        if advanced is not None:
            train_smoothing = getattr(advanced, 'train_smoothing', True)
            median_window = getattr(advanced, 'median_window', None)
            vol_norm_window = getattr(advanced, 'vol_norm_window', None)
            min_volume = getattr(advanced, 'min_volume', None)
            spread_filter = getattr(advanced, 'spread_filter', False)
            spread_pctl = getattr(advanced, 'spread_pctl', 0.99)
            spread_mult = getattr(advanced, 'spread_mult', 3.0)
            top_features = getattr(advanced, 'top_features', None)
            tier2_calibrate = getattr(advanced, 'tier2_calibrate', True)

        # Step 1: Load raw data
        if multi_pair:
            df = self._load_multi_pair_data(
                pairs=foundation_pairs,
                oanda_fetch=oanda_fetch,
            )
            instrument = "MULTI_PAIR"
        else:
            df = self._load_data(
                csv_path=csv_path,
                oanda_fetch=oanda_fetch,
                min_volume=min_volume,
                spread_filter=spread_filter,
                spread_pctl=spread_pctl,
                spread_mult=spread_mult,
            )
            instrument = self._infer_instrument(csv_path, oanda_fetch)

        # Step 2: Apply feature engineering
        df = self._compute_features(
            df=df,
            all_features=all_features,
            train_smoothing=train_smoothing,
            median_window=median_window,
        )

        # Step 3: Extract numeric features and handle NaNs
        numeric_df, ohlc_df, closes = self._extract_numeric_features(df)

        # Step 4: Optional feature ranking
        ranked_features = None
        if top_features is not None and top_features > 0:
            numeric_df, ranked_features = self._rank_and_select_features(
                numeric_df=numeric_df,
                closes=closes,
                top_features=top_features,
                vol_norm_window=vol_norm_window,
            )

        # Step 5: Extract features array and handle alignment
        feats, feature_names = self._features_to_array(numeric_df)

        # Align features to targets (targets are for t->t+1, so drop last row)
        feats = feats[:-1]
        closes_aligned = closes[:-1]

        n = feats.shape[0]
        if n <= seq_len + 10:
            raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

        # Step 6: Create direction labels
        direction_y = self._create_direction_labels(closes)

        # Step 7: Train/val split (chronological)
        split_idx = int(n * 0.8)

        train_feats_raw = feats[:split_idx]
        val_feats_raw = feats[split_idx:]
        train_dir = direction_y[:split_idx]
        val_dir = direction_y[split_idx:]
        closes_train = closes_aligned[:split_idx]
        closes_val = closes_aligned[split_idx:]

        # Step 8: Standardize features (fit on train only)
        train_feats, val_feats, scaler = self._standardize_features(
            train_feats_raw, val_feats_raw
        )

        # Step 9: Optional PCA
        pca_model = None
        if pca_components is not None and pca_components > 0:
            train_feats, val_feats, pca_model = self._apply_pca(
                train_feats, val_feats, n_components=pca_components
            )

        # Step 10: Create confidence labels (Tier-2) if enabled
        conf_y_all = np.zeros(n, dtype=np.float32)
        conf_w_all = np.zeros(n, dtype=np.float32)

        if tier2_calibrate and ohlc_df is not None:
            try:
                conf_y_all, conf_w_all = self._create_confidence_labels(
                    ohlc_df=ohlc_df,
                    direction_y=direction_y,
                    instrument=instrument,
                    seq_len=seq_len,
                )
            except Exception as e:
                self.console.print(f"[yellow]Tier-2 labeling failed: {e}[/yellow]")

        train_conf = conf_y_all[:split_idx]
        val_conf = conf_y_all[split_idx:]
        train_conf_w = conf_w_all[:split_idx]
        val_conf_w = conf_w_all[split_idx:]

        elapsed = time.perf_counter() - t_start
        self.console.print(Panel(
            f"[bold]Data Preparation Complete[/bold]\n\n"
            f"[dim]Train:[/dim] {len(train_feats):,} samples  [dim]Val:[/dim] {len(val_feats):,} samples\n"
            f"[dim]Features:[/dim] {train_feats.shape[1]}  [dim]Time:[/dim] {elapsed:.2f}s",
            title="📊 Prepared Data",
            border_style="green",
        ))

        return PreparedData(
            X_train=train_feats,
            y_train=train_dir,
            X_val=val_feats,
            y_val=val_dir,
            feature_names=feature_names,
            scaler=scaler,
            df=df,
            y_train_conf=train_conf,
            y_val_conf=val_conf,
            conf_weights_train=train_conf_w,
            conf_weights_val=val_conf_w,
            closes_train=closes_train,
            closes_val=closes_val,
            ohlc_df=ohlc_df,
            pca_model=pca_model,
            ranked_features=ranked_features,
            instrument=instrument,
            seq_len=seq_len,
            split_ratio=0.8,
        )

    # =========================================================================
    # PRIVATE METHODS - DATA LOADING
    # =========================================================================

    def _load_data(
        self,
        *,
        csv_path: Optional[str],
        oanda_fetch: Optional[Any],
        min_volume: Optional[int],
        spread_filter: bool,
        spread_pctl: float,
        spread_mult: float,
    ) -> pd.DataFrame:
        """Load data from OANDA API or CSV file."""
        from src.training.buddy_training_helpers import _buddy_load_and_validate_csv

        df = _buddy_load_and_validate_csv(
            csv_path=csv_path,
            oanda_fetch=oanda_fetch,
            min_volume=min_volume,
            spread_filter=spread_filter,
            spread_pctl=spread_pctl,
            spread_mult=spread_mult,
            oanda_fetch_to_csv=self.oanda_fetch_to_csv,
            console=self.console,
        )

        return df

    def _load_multi_pair_data(
        self,
        pairs: Optional[str],
        oanda_fetch: Optional[Any],
    ) -> pd.DataFrame:
        """Load and combine data from multiple currency pairs."""
        from src.training.buddy_training_helpers import _load_multi_pair_data

        DEFAULT_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"]

        if pairs:
            pairs_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]
        else:
            pairs_list = DEFAULT_PAIRS

        # Determine settings from oanda_fetch if available
        candles_per_pair = 5000
        granularity = "H1"
        if oanda_fetch is not None:
            try:
                candles_per_pair = int(getattr(oanda_fetch, "candles", 5000) or 5000)
                granularity = str(getattr(oanda_fetch, "granularity", "H1") or "H1")
            except Exception:
                pass

        self.console.print(Panel(
            f"[bold]Multi-Pair Foundation Training[/bold]\n\n"
            f"[dim]Pairs:[/dim] {', '.join(pairs_list)}\n"
            f"[dim]Candles per pair:[/dim] {candles_per_pair:,}\n"
            f"[dim]Granularity:[/dim] {granularity}",
            title="🌍 Multi-Pair Mode",
            border_style="cyan",
        ))

        df = _load_multi_pair_data(
            pairs=pairs_list,
            granularity=granularity,
            candles_per_pair=candles_per_pair,
            console=self.console,
        )

        if df is None or len(df) < 1000:
            raise ValueError(f"Multi-pair data loading failed: {len(df) if df is not None else 0} rows")

        return df

    def _fetch_oanda_data(
        self,
        oanda_fetch: Any,
    ) -> pd.DataFrame:
        """Fetch data directly from OANDA API.

        This method is provided for future use when oanda_fetcher module is available.
        Currently, data fetching goes through _buddy_load_and_validate_csv.
        """
        if self.oanda_fetch_to_csv is None:
            raise ValueError("oanda_fetch_to_csv function not provided")

        csv_path = self.oanda_fetch_to_csv(oanda_fetch)
        return pd.read_csv(csv_path)

    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        t_csv = time.perf_counter()
        df = pd.read_csv(csv_path)
        elapsed = time.perf_counter() - t_csv

        self.console.print(Panel(
            f"[bold]Loading Training Data[/bold]\n\n"
            f"[dim]Source:[/dim] {csv_path}\n"
            f"[dim]Rows:[/dim] {len(df):,}  [dim]Columns:[/dim] {df.shape[1]}  [dim]Time:[/dim] {elapsed:.2f}s",
            title="📄 CSV Data",
            border_style="blue",
        ))

        return df

    # =========================================================================
    # PRIVATE METHODS - FEATURE ENGINEERING
    # =========================================================================

    def _compute_features(
        self,
        df: pd.DataFrame,
        *,
        all_features: bool,
        train_smoothing: bool,
        median_window: Optional[int],
    ) -> pd.DataFrame:
        """Apply feature engineering to raw OHLCV data."""
        if all_features:
            t_fe = time.perf_counter()

            # Suppress verbose logging
            import logging as _logging
            _fe_logger = _logging.getLogger('feature_engineering')
            _fe_prev = _fe_logger.level
            _fe_logger.setLevel(_logging.WARNING)

            try:
                from feature_engineering import FeatureEngineering
                fe = FeatureEngineering(self.fe_cfg)
                df = fe.create_features(
                    df,
                    include_all=True,
                    apply_candle_smoothing=train_smoothing,
                    median_window=median_window,
                )
            finally:
                _fe_logger.setLevel(_fe_prev)

            elapsed = time.perf_counter() - t_fe
            self.console.print(Panel(
                f"[bold]Feature Engineering Complete[/bold]\n\n"
                f"[dim]Smoothing:[/dim] {train_smoothing}  [dim]Median Window:[/dim] {median_window}\n"
                f"[dim]Output:[/dim] {len(df):,} rows × {df.shape[1]} columns\n"
                f"[dim]Time:[/dim] {elapsed:.2f}s",
                title="⚙️ Feature Engineering",
                border_style="blue",
            ))
        elif train_smoothing:
            try:
                from src.data.candle_smoothing import resample_5min_ohlcv_and_ema_close
                df = resample_5min_ohlcv_and_ema_close(
                    df,
                    ema_span=14,
                    time_col="time",
                    median_window=median_window,
                )
            except Exception as e:
                self.console.print(f"[yellow]Training noise reduction skipped: {e}[/yellow]")

        return df

    # =========================================================================
    # PRIVATE METHODS - FEATURE EXTRACTION & SPLITTING
    # =========================================================================

    def _extract_numeric_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Extract numeric features and OHLC data."""
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=["number"]).copy()

        if numeric_df.empty:
            raise ValueError("No numeric features found after preprocessing")

        # Handle NaN/Inf
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # Extract OHLC for labeling
        ohlc_cols = ["open", "high", "low", "close"]
        extra_cols = [c for c in ("bid_close", "ask_close") if c in df.columns]

        ohlc_df = df.loc[numeric_df.index, ohlc_cols + extra_cols].copy()
        closes = df.loc[numeric_df.index, "close"].to_numpy(dtype=np.float32)

        return numeric_df, ohlc_df, closes

    def _rank_and_select_features(
        self,
        numeric_df: pd.DataFrame,
        closes: np.ndarray,
        top_features: int,
        vol_norm_window: Optional[int],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Rank features by correlation with direction and select top K."""
        # Direction labels
        y_dir = (closes[1:] > closes[:-1]).astype(np.float32)
        r0 = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-8)

        # Optional volatility normalization
        if vol_norm_window is not None:
            w = int(vol_norm_window)
            ret_std = (
                pd.Series(r0)
                .rolling(w, min_periods=max(10, w // 5))
                .std()
                .to_numpy(dtype=np.float32)
            )
            if not np.isfinite(ret_std).all():
                finite = ret_std[np.isfinite(ret_std)]
                fill = float(np.median(finite)) if finite.size else 1.0
                ret_std = np.where(np.isfinite(ret_std), ret_std, fill)
            r0 = r0 / np.maximum(ret_std, 1e-6)

        abs_ret = np.nan_to_num(np.abs(r0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        # Train-only ranking (first 80% to avoid leaking future info)
        x_rank_all = numeric_df.iloc[:-1].astype(np.float32)
        n_rank = len(x_rank_all)
        split_rank = int(n_rank * 0.8)

        if split_rank <= 10:
            self.console.print("[yellow]Not enough rows for feature ranking[/yellow]")
            return numeric_df, None

        x_rank = x_rank_all.iloc[:split_rank]
        y_dir_rank = pd.Series(y_dir[:split_rank])
        abs_ret_rank = pd.Series(abs_ret[:split_rank])

        # Correlation-based ranking
        s_dir = x_rank.corrwith(y_dir_rank, method="pearson").abs().fillna(0.0)
        s_conf = x_rank.corrwith(abs_ret_rank, method="pearson").abs().fillna(0.0)
        scores = (0.7 * s_dir) + (0.3 * s_conf)
        ranked_cols = scores.sort_values(ascending=False).index.tolist()

        # Select top features
        if top_features < len(ranked_cols):
            keep = ranked_cols[:top_features]
            numeric_df = numeric_df[keep]
            self.console.print(
                f"[cyan]✓ Feature Selection:[/cyan] kept {len(keep)} / {len(ranked_cols)} ranked features"
            )

        return numeric_df, ranked_cols

    def _features_to_array(
        self,
        numeric_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert DataFrame to numpy array."""
        feats = numeric_df.to_numpy(dtype=np.float32)
        feature_names = list(numeric_df.columns)

        if feats.shape[1] < 6:
            self.console.print(f"[yellow]⚠ Warning: only {feats.shape[1]} numeric features[/yellow]")
        else:
            self.console.print(f"[cyan]✓ Features:[/cyan] {feats.shape[1]} numeric columns")

        return feats, feature_names

    def _create_direction_labels(self, closes: np.ndarray) -> np.ndarray:
        """Create direction labels (up=1, down=0)."""
        next_close = closes[1:]
        cur_close = closes[:-1]
        direction_y = (next_close > cur_close).astype(np.float32)
        return direction_y

    def _split_train_val(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        split_ratio: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data chronologically into train and validation sets."""
        n = len(feats)
        split_idx = int(n * split_ratio)

        train_feats = feats[:split_idx]
        val_feats = feats[split_idx:]
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]

        return train_feats, val_feats, train_labels, val_labels

    # =========================================================================
    # PRIVATE METHODS - PREPROCESSING
    # =========================================================================

    def _standardize_features(
        self,
        train_feats: np.ndarray,
        val_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, ScalerState]:
        """Standardize features (fit on train only)."""
        # Compute mean and std from training data only
        train_feats = train_feats.astype(np.float32, copy=False)

        mu = train_feats.mean(axis=0, keepdims=True, dtype=np.float32)
        sigma = train_feats.std(axis=0, keepdims=True, dtype=np.float32)
        sigma = np.where(sigma < 1e-6, np.float32(1.0), sigma).astype(np.float32)

        # Transform both train and val
        train_standardized = ((train_feats - mu) / sigma).astype(np.float32)
        val_standardized = ((val_feats.astype(np.float32) - mu) / sigma).astype(np.float32)

        scaler = ScalerState(mean_=mu.flatten(), scale_=sigma.flatten())

        return train_standardized, val_standardized, scaler

    def _apply_pca(
        self,
        train_feats: np.ndarray,
        val_feats: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Apply PCA dimensionality reduction."""
        if n_components >= train_feats.shape[1]:
            return train_feats, val_feats, None

        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
        pca.fit(train_feats)

        train_pca = pca.transform(train_feats).astype(np.float32)
        val_pca = pca.transform(val_feats).astype(np.float32)

        self.console.print(f"[cyan]PCA:[/cyan] {train_feats.shape[1]} -> {n_components} components")

        return train_pca, val_pca, pca

    def _create_confidence_labels(
        self,
        ohlc_df: pd.DataFrame,
        direction_y: np.ndarray,
        instrument: str,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create Tier-2 confidence labels (TP-before-SL outcome)."""
        from fx_paper import pip_size, simulate_tp_sl_outcome

        n = len(direction_y)
        conf_y = np.zeros(n, dtype=np.float32)
        conf_w = np.zeros(n, dtype=np.float32)

        # Get Tier-2 parameters from config
        sl_pips = float(self.buddy_cfg.get("stop_loss_pips", 15.0))
        tp_pips = float(self.buddy_cfg.get("take_profit_pips", 30.0))
        spread_fallback = float(self.buddy_cfg.get("tier2_spread_fallback_pips", 2.0))
        horizon_candles = int(self.buddy_cfg.get("tier2_horizon_candles", 288))
        tie_break = str(self.buddy_cfg.get("tier2_tie_break", "sl"))
        label_stride = int(self.buddy_cfg.get("tier2_calibration_stride", 5))

        pip = float(pip_size(instrument))
        max_signal = min(n - 1, len(ohlc_df) - 3)

        if max_signal < 0:
            return conf_y, conf_w

        def _get_spread(idx: int) -> float:
            sp = spread_fallback
            try:
                if "bid_close" in ohlc_df.columns and "ask_close" in ohlc_df.columns:
                    bid = float(ohlc_df["bid_close"].iloc[idx + 1])
                    ask = float(ohlc_df["ask_close"].iloc[idx + 1])
                    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid > 0:
                        sp = (ask - bid) / max(pip, 1e-9)
            except Exception:
                pass
            return sp

        start_idx = max(0, seq_len - 1)
        n_sim = 0

        for gi in range(start_idx, max_signal + 1, label_stride):
            spread_pips = _get_spread(gi)
            true_dir = "long" if direction_y[gi] >= 0.5 else "short"

            res = simulate_tp_sl_outcome(
                ohlc_df,
                instrument=instrument,
                signal_index=gi,
                direction=true_dir,
                stop_loss_pips=sl_pips,
                take_profit_pips=tp_pips,
                spread_pips=spread_pips,
                max_horizon_candles=horizon_candles,
                tie_break=tie_break,
            )

            conf_y[gi] = float(int(res.to_label()))
            conf_w[gi] = 1.0
            n_sim += 1

        self.console.print(
            f"[cyan]✓ Tier-2 Labels:[/cyan] {instrument} | SL={sl_pips:g} TP={tp_pips:g} | "
            f"horizon={horizon_candles} stride={label_stride} | {n_sim} samples"
        )

        return conf_y, conf_w

    # =========================================================================
    # PRIVATE METHODS - UTILITIES
    # =========================================================================

    def _infer_instrument(
        self,
        csv_path: Optional[str],
        oanda_fetch: Optional[Any],
    ) -> str:
        """Infer instrument name from options."""
        if oanda_fetch is not None:
            try:
                return str(getattr(oanda_fetch, "instrument", None) or "USD_JPY")
            except Exception:
                pass

        if csv_path:
            up = str(csv_path).upper()
            if "EUR_USD" in up:
                return "EUR_USD"
            elif "GBP_USD" in up:
                return "GBP_USD"
            elif "USD_JPY" in up or "USDJPY" in up:
                return "USD_JPY"

        return "USD_JPY"


# =============================================================================
# SEQUENCE DATASET CREATION
# =============================================================================

def create_sequence_dataset(
    X: np.ndarray,
    y_direction: np.ndarray,
    y_confidence: np.ndarray,
    confidence_weights: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    cache: bool = False,
    shuffle_buffer: Optional[int] = None,
    prefetch: Optional[int] = None,
):
    """Create a TensorFlow Dataset for sequence training.

    This function creates windowed sequences from 2D feature arrays,
    suitable for LSTM, TCN, or Transformer models.

    Args:
        X: Feature array (n_samples, n_features)
        y_direction: Direction labels (n_samples,)
        y_confidence: Confidence labels (n_samples,)
        confidence_weights: Sample weights for confidence (n_samples,)
        seq_len: Sequence length for windowing
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed
        cache: Whether to cache dataset
        shuffle_buffer: Buffer size for shuffling (None = auto)
        prefetch: Prefetch buffer (None = auto)

    Returns:
        tf.data.Dataset yielding (x_seq, {direction, confidence}, {weights})
    """
    import os
    import tensorflow as tf

    # Ensure float32
    X = np.asarray(X).astype(np.float32, copy=False)
    y_direction = np.asarray(y_direction).reshape(-1).astype(np.float32, copy=False)
    y_confidence = np.asarray(y_confidence).reshape(-1).astype(np.float32, copy=False)
    confidence_weights = np.asarray(confidence_weights).reshape(-1).astype(np.float32, copy=False)

    if len(X) <= seq_len + 1:
        raise ValueError(f"Not enough samples ({len(X)}) for seq_len={seq_len}")

    # Drop last row (label alignment)
    X = X[:-1]
    y_direction = y_direction[:-1]
    y_confidence = y_confidence[:-1]
    confidence_weights = confidence_weights[:-1]

    n_windows = len(X) - seq_len + 1

    # Determine shuffle buffer
    if shuffle_buffer is None:
        shuffle_buffer_env = os.environ.get("BUDDY_SHUFFLE_BUFFER")
        if shuffle_buffer_env:
            try:
                shuffle_buffer = max(1, min(int(shuffle_buffer_env), n_windows))
            except Exception:
                shuffle_buffer = min(256, n_windows)
        else:
            shuffle_buffer = min(256, n_windows)

    # Determine prefetch
    if prefetch is None:
        prefetch_env = os.environ.get("BUDDY_PREFETCH")
        if prefetch_env:
            try:
                prefetch = int(prefetch_env)
            except Exception:
                prefetch = 1
        else:
            prefetch = 1

    # Use fast timeseries_dataset_from_array if available
    try:
        ts = getattr(tf.keras.utils, "timeseries_dataset_from_array", None)
        if ts is None:
            raise AttributeError("timeseries_dataset_from_array unavailable")

        # Labels at end of each window
        y_dir_end = y_direction[seq_len - 1:]
        y_conf_end = y_confidence[seq_len - 1:]
        w_conf_end = confidence_weights[seq_len - 1:]

        x_seq = ts(
            X,
            targets=None,
            sequence_length=seq_len,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False,
            batch_size=None,
        )
        y_seq = tf.data.Dataset.from_tensor_slices((y_dir_end, y_conf_end, w_conf_end))
        ds = tf.data.Dataset.zip((x_seq, y_seq))

        def _map_fn(xw, ypack):
            y_d, y_c, w_c = ypack
            return (
                tf.cast(xw, tf.float32),
                {
                    "direction": tf.expand_dims(tf.cast(y_d, tf.float32), axis=-1),
                    "confidence": tf.expand_dims(tf.cast(y_c, tf.float32), axis=-1),
                },
                {
                    "direction": tf.constant(1.0, dtype=tf.float32),
                    "confidence": tf.expand_dims(tf.cast(w_c, tf.float32), axis=-1),
                },
            )

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    except Exception:
        # Fallback to window-based approach
        ds = tf.data.Dataset.from_tensor_slices((X, y_direction, y_confidence, confidence_weights))
        ds = ds.window(seq_len, shift=1, drop_remainder=True)
        ds = ds.flat_map(
            lambda xw, ydw, ycw, wcw: tf.data.Dataset.zip((
                xw.batch(seq_len, drop_remainder=True),
                ydw.batch(seq_len, drop_remainder=True),
                ycw.batch(seq_len, drop_remainder=True),
                wcw.batch(seq_len, drop_remainder=True),
            ))
        )
        ds = ds.map(
            lambda xw, ydw, ycw, wcw: (
                xw,
                {
                    "direction": tf.expand_dims(ydw[-1], axis=-1),
                    "confidence": tf.expand_dims(ycw[-1], axis=-1),
                },
                {
                    "direction": tf.constant(1.0, dtype=tf.float32),
                    "confidence": tf.expand_dims(tf.cast(wcw[-1], tf.float32), axis=-1),
                },
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    if prefetch == 0:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.prefetch(max(1, prefetch))

    return ds


# =============================================================================
# HELPER CLASSES
# =============================================================================

class _DummyConsole:
    """Dummy console for when Rich is not available."""
    def print(self, *args: Any, **kwargs: Any) -> None:
        """No-op print method."""
        # Intentionally empty - silent console for testing
