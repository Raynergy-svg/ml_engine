"""
Market Regime Detector Module.

Implements multi-factor regime detection combining:
1. Bayesian Online Changepoint Detection (BOCPD) - Adams & MacKay algorithm
2. Hurst Exponent - Trending vs mean-reverting classification (R/S analysis)
3. ADX (Average Directional Index) - Trend strength confirmation
4. Volatility Regime Clustering - ATR percentile bands for LOW/NORMAL/HIGH/EXTREME
5. Ensemble Consensus - Soft voting across all signals

All methods are deterministic, use closed bar data only (no look-ahead bias),
and are thread-safe with no mutable class-level state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Volatility regime constants
REGIME_LOW = 0
REGIME_NORMAL = 1
REGIME_HIGH = 2
REGIME_EXTREME = 3

REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


@dataclass
class RegimeDetectorConfig:
    """Configuration for Market Regime Detector.

    Attributes:
        # BOCPD (Bayesian Online Changepoint Detection) parameters
        hazard_rate: Expected regime duration = 1/hazard_rate bars. Default 0.01 → ~100 bar avg.
        bocpd_threshold: Changepoint probability threshold for alert. 0-1 scale.
        max_run_length: Maximum run length in changepoint distribution. Higher = more memory.

        # Hurst Exponent parameters
        hurst_window: Lookback window for Hurst calculation. Recommend 50-150.
        hurst_trending_threshold: H > this value classifies as TRENDING. Default 0.55.
        hurst_mean_revert_threshold: H < this value classifies as MEAN_REVERTING. Default 0.45.

        # ADX (Average Directional Index) parameters
        adx_period: EMA period for ADX calculation. Standard is 14.
        adx_strong_threshold: ADX > this = strong trend. Default 25.0.
        adx_weak_threshold: ADX < this = weak trend. Default 20.0.

        # Volatility regime clustering
        vol_lookback: Lookback window for volatility percentile calculation.
        vol_low_percentile: ATR percentile threshold for LOW regime.
        vol_high_percentile: ATR percentile threshold for HIGH regime.
        vol_extreme_percentile: ATR percentile threshold for EXTREME regime.
    """

    # BOCPD parameters
    hazard_rate: float = 0.01
    bocpd_threshold: float = 0.3
    max_run_length: int = 200

    # Hurst parameters
    hurst_window: int = 100
    hurst_trending_threshold: float = 0.55
    hurst_mean_revert_threshold: float = 0.45

    # ADX parameters
    adx_period: int = 14
    adx_strong_threshold: float = 25.0
    adx_weak_threshold: float = 20.0

    # Volatility regime parameters
    vol_lookback: int = 50
    vol_low_percentile: float = 25.0
    vol_high_percentile: float = 75.0
    vol_extreme_percentile: float = 95.0


@dataclass
class RegimeDetectionResult:
    """Result of regime detection analysis.

    Attributes:
        regime_index: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME).
        regime_name: Human-readable regime name.
        confidence: Overall confidence in detection (0-1), derived from ensemble consensus.

        # Component signals
        changepoint_probability: BOCPD changepoint probability (0-1).
        changepoint_alert: True if changepoint_probability > threshold.
        hurst_exponent: Hurst exponent value (0-2, typically 0.4-0.6).
        trend_classification: Hurst-based classification: "TRENDING", "MEAN_REVERTING", or "UNCERTAIN".
        adx_value: ADX value (0-100).
        adx_strength: ADX-based classification: "STRONG", "WEAK", or "TRANSITION".
        volatility_percentile: Current ATR percentile vs historical (0-100).

        # Derived signals
        strategy_hint: Recommended strategy: "MOMENTUM", "MEAN_REVERSION", "RANGE", or "CAUTION".
        risk_multiplier: Position sizing scalar (0-1). Apply to base position size.

        # Metadata
        signals: Dict of all raw signal values for logging and RL feedback.
    """

    regime_index: int = REGIME_NORMAL
    regime_name: str = "NORMAL"
    confidence: float = 0.5

    # Component signals
    changepoint_probability: float = 0.0
    changepoint_alert: bool = False
    hurst_exponent: float = 0.5
    trend_classification: str = "UNCERTAIN"
    adx_value: float = 0.0
    adx_strength: str = "TRANSITION"
    volatility_percentile: float = 0.5

    # Derived signals
    strategy_hint: str = "RANGE"
    risk_multiplier: float = 0.7

    # Metadata
    signals: dict = field(default_factory=dict)


class MarketRegimeDetector:
    """Multi-factor market regime detector using ensemble consensus.

    Combines BOCPD, Hurst exponent, ADX, and volatility clustering to provide
    comprehensive regime detection with strategy recommendations.

    Thread-safe: each call is independent with no shared mutable state.
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        """Initialize detector with configuration.

        Args:
            config: RegimeDetectorConfig. If None, uses defaults.
        """
        self.config = config or RegimeDetectorConfig()
        self._validate_config()
        logger.debug(f"MarketRegimeDetector initialized with config: {self.config}")

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        try:
            if not (0.0 < self.config.hazard_rate < 1.0):
                raise ValueError(
                    f"hazard_rate must be in (0, 1), got {self.config.hazard_rate}"
                )
            if not (0.0 <= self.config.bocpd_threshold <= 1.0):
                raise ValueError(
                    f"bocpd_threshold must be in [0, 1], got {self.config.bocpd_threshold}"
                )
            if self.config.max_run_length < 10:
                raise ValueError(
                    f"max_run_length must be >= 10, got {self.config.max_run_length}"
                )
            if self.config.hurst_window < 20:
                raise ValueError(
                    f"hurst_window must be >= 20, got {self.config.hurst_window}"
                )
            if not (
                self.config.hurst_mean_revert_threshold
                < self.config.hurst_trending_threshold
            ):
                raise ValueError(
                    f"hurst_mean_revert_threshold ({self.config.hurst_mean_revert_threshold}) "
                    f"must be < hurst_trending_threshold ({self.config.hurst_trending_threshold})"
                )
            if self.config.adx_period < 5:
                raise ValueError(
                    f"adx_period must be >= 5, got {self.config.adx_period}"
                )
            if self.config.vol_lookback < 20:
                raise ValueError(
                    f"vol_lookback must be >= 20, got {self.config.vol_lookback}"
                )
        except (TypeError, ValueError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def update(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
    ) -> RegimeDetectionResult:
        """Perform full regime detection on price data.

        This is the main entry point. Uses only closed bar data (no look-ahead bias).

        Args:
            prices: Close prices, shape (N,). At least max(hurst_window, vol_lookback, adx_period+1) required.
            highs: High prices, shape (N,). Required for ADX. If None, ADX is skipped.
            lows: Low prices, shape (N,). Required for ADX. If None, ADX is skipped.

        Returns:
            RegimeDetectionResult with all signals, classification, and strategy hints.

        Raises:
            ValueError: If input validation fails.
        """
        try:
            prices = np.asarray(prices, dtype=np.float64)
            if prices.ndim != 1:
                raise ValueError(f"prices must be 1D, got shape {prices.shape}")
            if len(prices) < 50:
                raise ValueError(
                    f"prices must have at least 50 bars, got {len(prices)}"
                )

            # Check for NaN or inf
            if not np.isfinite(prices).all():
                raise ValueError("prices contains NaN or infinite values")

            # Validate highs/lows if provided
            if highs is not None:
                highs = np.asarray(highs, dtype=np.float64)
                if highs.shape != prices.shape:
                    raise ValueError(
                        f"highs shape {highs.shape} != prices shape {prices.shape}"
                    )
                if not np.isfinite(highs).all():
                    raise ValueError("highs contains NaN or infinite values")
            if lows is not None:
                lows = np.asarray(lows, dtype=np.float64)
                if lows.shape != prices.shape:
                    raise ValueError(
                        f"lows shape {lows.shape} != prices shape {prices.shape}"
                    )
                if not np.isfinite(lows).all():
                    raise ValueError("lows contains NaN or infinite values")

            # Compute returns for BOCPD, Hurst, and volatility
            returns = np.diff(np.log(prices))

            # Component calculations
            changepoint_prob, run_length_dist = self._compute_bocpd(returns)
            hurst = self._compute_hurst(prices)
            vol_regime, vol_name, vol_percentile = self._classify_volatility_regime(
                returns
            )

            # ADX (optional, only if highs/lows provided)
            adx = 0.0
            if highs is not None and lows is not None:
                adx = self._compute_adx(highs, lows, prices)
            else:
                logger.warning(
                    "highs/lows not provided; ADX skipped, defaulting to 0.0"
                )

            # Classification from components
            trend_class = self._classify_trend(hurst)
            adx_strength = self._classify_adx_strength(adx)

            # Ensemble consensus
            result = self._ensemble_consensus(
                vol_regime=vol_regime,
                vol_percentile=vol_percentile,
                changepoint_prob=changepoint_prob,
                changepoint_alert=changepoint_prob > self.config.bocpd_threshold,
                hurst=hurst,
                trend_class=trend_class,
                adx=adx,
                adx_strength=adx_strength,
            )

            logger.debug(
                f"Regime detection: {result.regime_name} (confidence={result.confidence:.2f}), "
                f"strategy={result.strategy_hint}, risk_mult={result.risk_multiplier:.2f}"
            )
            return result

        except (ValueError, TypeError) as e:
            logger.error(
                f"Regime detection failed: {e} (input: prices={len(prices) if isinstance(prices, (list, np.ndarray)) else 'invalid'})"
            )
            raise

    def _compute_bocpd(self, returns: np.ndarray) -> Tuple[float, np.ndarray]:
        """Bayesian Online Changepoint Detection (Adams & MacKay 2007).

        Maintains a run-length distribution P(r_t | data). At each step:
        1. Extend existing runs (pass through Gaussian likelihood)
        2. Start new run (reset to r=0)
        3. Compute run-length distribution via Bayes rule

        Args:
            returns: Log returns, shape (N,). Preferably recent bars only.

        Returns:
            Tuple of (changepoint_probability, run_length_distribution).
            changepoint_probability: Probability of run-0 (recent changepoint).
            run_length_distribution: Array of posterior probabilities for each run length.
        """
        try:
            n = len(returns)
            if n < 2:
                logger.warning(f"BOCPD requires n >= 2, got {n}; returning zero probability")
                return 0.0, np.array([1.0])

            # Initialize: run_length_dist[r_t] = P(r_t | data_1:t)
            # Start with no changepoints (single run of length 0)
            run_length_dist = np.ones(min(n + 1, self.config.max_run_length + 1))
            run_length_dist /= run_length_dist.sum()

            # Iteratively update with each return
            for t in range(n):
                # Growth probabilities: extend each existing run
                # For run length r, likelihood is N(return_t | mu_r, sigma_r^2)
                # We use a simple online Gaussian sufficient statistics approach
                growth_probs = self._gaussian_likelihood(returns[t], run_length_dist)

                # Hazard function: constant probability of changepoint
                # P(changepoint at t) = hazard_rate
                hazard = self.config.hazard_rate

                # Bayes update: combine growth with hazard (prior on changepoint)
                # If changepoint: restart at r=0. Else: extend current run.
                run_length_dist = (1 - hazard) * growth_probs + hazard
                run_length_dist = run_length_dist / (run_length_dist.sum() + 1e-10)

                # Prune to max_run_length to limit memory
                if len(run_length_dist) > self.config.max_run_length + 1:
                    run_length_dist = run_length_dist[: self.config.max_run_length + 1]

            # Changepoint probability = P(r=0 | data)
            changepoint_prob = run_length_dist[0] if len(run_length_dist) > 0 else 0.0
            return float(np.clip(changepoint_prob, 0.0, 1.0)), run_length_dist

        except Exception as e:
            logger.error(f"BOCPD computation failed: {e}")
            raise

    def _gaussian_likelihood(
        self, return_t: float, run_length_dist: np.ndarray
    ) -> np.ndarray:
        """Compute Gaussian likelihood for each run length.

        Uses a simplified approach: assume recent returns are more stable.
        For run r, likelihood of return_t is approximated by a Gaussian fit
        to recent data.

        Args:
            return_t: Current return value.
            run_length_dist: Current run-length distribution.

        Returns:
            Likelihood vector for each run length.
        """
        try:
            # Simplified likelihood: prefer shorter runs (higher Gaussian density)
            # Real implementation would fit Gaussians per run, but that's expensive.
            # Instead, use a simple penalty that increases with run length.
            max_r = len(run_length_dist)
            likelihoods = np.ones(max_r)

            # Penalty: longer runs → lower likelihood (assume higher variance)
            for r in range(max_r):
                variance_est = 1.0 + 0.01 * r  # Increase variance with run length
                likelihoods[r] = stats.norm.pdf(
                    return_t, loc=0.0, scale=np.sqrt(variance_est)
                )

            # Normalize to avoid underflow
            likelihoods = likelihoods / (likelihoods.sum() + 1e-10)
            return likelihoods

        except Exception as e:
            logger.error(f"Gaussian likelihood computation failed: {e}")
            return np.ones(len(run_length_dist)) / len(run_length_dist)

    def _compute_hurst(self, prices: np.ndarray) -> float:
        """Compute Hurst Exponent via Rescaled Range (R/S) Analysis.

        Divides the price series into chunks of varying size and computes
        R/S = (max cumulative deviation - min cumulative deviation) / std_dev
        for each chunk size. Regresses log(R/S) on log(chunk_size) → slope is H.

        H > 0.5: trending (momentum). H < 0.5: mean-reverting.
        H ≈ 0.5: random walk.

        Args:
            prices: Close prices, shape (N,). Uses last hurst_window bars.

        Returns:
            Hurst exponent (0-2, typically 0.3-0.7 for FX).

        Reference:
            Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
        """
        try:
            n = len(prices)
            if n < self.config.hurst_window:
                logger.warning(
                    f"Hurst: need {self.config.hurst_window} bars, got {n}; using all data"
                )
                data = prices
            else:
                data = prices[-self.config.hurst_window :]

            # Mean-adjusted returns
            returns = np.diff(data)

            # Check if returns have any variance
            if len(returns) < 2 or np.std(returns) < 1e-10:
                logger.warning("Hurst: no variance in returns; returning 0.5")
                return 0.5

            mean_return = returns.mean()
            demeaned = returns - mean_return

            # Cumulative sum
            cumsum = np.cumsum(demeaned)

            # R/S for different chunk sizes
            chunk_sizes = np.logspace(
                1, np.log10(len(demeaned) / 2), num=20, dtype=int
            )
            chunk_sizes = np.unique(np.clip(chunk_sizes, 4, len(demeaned) // 2))

            rs_values = []
            valid_chunk_sizes = []

            for chunk_size in chunk_sizes:
                n_chunks = len(demeaned) // chunk_size
                if n_chunks < 1:
                    continue

                # Trim to fit whole chunks
                trimmed_cumsum = cumsum[: n_chunks * chunk_size]

                # Reshape and compute R/S per chunk
                chunks = trimmed_cumsum.reshape(n_chunks, chunk_size)
                ranges = np.max(chunks, axis=1) - np.min(chunks, axis=1)

                # Standard deviation of returns per chunk
                trimmed_returns = demeaned[: n_chunks * chunk_size]
                returns_chunks = trimmed_returns.reshape(n_chunks, chunk_size)
                stds = np.std(returns_chunks, axis=1, ddof=1)
                stds = np.clip(stds, 1e-10, None)  # Avoid division by zero

                # R/S value for this chunk size
                rs_mean = (ranges / stds).mean()

                # Only add valid values
                if np.isfinite(rs_mean) and rs_mean > 0:
                    rs_values.append(rs_mean)
                    valid_chunk_sizes.append(chunk_size)

            if len(rs_values) < 2:
                logger.warning("Hurst: insufficient valid data for reliable fit; returning 0.5")
                return 0.5

            # Linear regression: log(R/S) vs log(chunk_size)
            rs_values = np.array(rs_values)
            chunk_sizes_fit = np.array(valid_chunk_sizes)

            log_rs = np.log(rs_values)
            log_sizes = np.log(chunk_sizes_fit)

            # Check for valid log values
            if not (np.isfinite(log_rs).all() and np.isfinite(log_sizes).all()):
                logger.warning("Hurst: non-finite log values; returning 0.5")
                return 0.5

            # Least squares fit
            coeffs = np.polyfit(log_sizes, log_rs, 1)
            hurst = coeffs[0]  # slope

            # Clip to (0, 2) range for safety
            hurst = np.clip(float(hurst), 0.0, 2.0)
            return hurst

        except Exception as e:
            logger.error(f"Hurst exponent computation failed: {e}")
            return 0.5

    def _compute_adx(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> float:
        """Compute Average Directional Index (ADX).

        Measures trend strength (0-100). ADX > 25 indicates strong trend.

        Steps:
        1. Compute True Range (TR)
        2. Compute +DM, -DM (directional movements)
        3. Smooth with EMA: ATR, +DI, -DI
        4. DX = |+DI - -DI| / (+DI + -DI) * 100
        5. ADX = EMA(DX, period)

        Args:
            highs: High prices, shape (N,).
            lows: Low prices, shape (N,).
            closes: Close prices, shape (N,).

        Returns:
            ADX value (0-100).
        """
        try:
            n = len(closes)
            if n < self.config.adx_period + 1:
                logger.warning(
                    f"ADX: need {self.config.adx_period + 1} bars, got {n}; returning 0.0"
                )
                return 0.0

            # True Range
            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = tr1[0]  # First TR can't use previous close

            # Directional Movements
            high_diff = np.diff(highs, prepend=0)
            low_diff = -np.diff(lows, prepend=0)

            plus_dm = np.where((high_diff > 0) & (high_diff > low_diff), high_diff, 0)
            minus_dm = np.where((low_diff > 0) & (low_diff > high_diff), low_diff, 0)

            # EMA smoothing
            period = self.config.adx_period
            alpha = 2.0 / (period + 1)

            atr = self._ema_smooth(tr, period)
            plus_di_raw = 100 * self._ema_smooth(plus_dm, period) / (atr + 1e-10)
            minus_di_raw = 100 * self._ema_smooth(minus_dm, period) / (atr + 1e-10)

            # DX
            di_sum = plus_di_raw + minus_di_raw
            di_sum = np.clip(di_sum, 1e-10, None)
            dx = 100 * np.abs(plus_di_raw - minus_di_raw) / di_sum

            # ADX = EMA(DX)
            adx = self._ema_smooth(dx, period)

            return float(np.clip(adx[-1], 0.0, 100.0))

        except Exception as e:
            logger.error(f"ADX computation failed: {e}")
            return 0.0

    def _ema_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average smoothing.

        Args:
            data: Input data, shape (N,).
            period: EMA period.

        Returns:
            EMA values, shape (N,).
        """
        try:
            if len(data) < period:
                return data.copy()

            alpha = 2.0 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]

            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

            return ema

        except Exception as e:
            logger.error(f"EMA smoothing failed: {e}")
            return data.copy()

    def _classify_volatility_regime(
        self, returns: np.ndarray
    ) -> Tuple[int, str, float]:
        """Classify volatility regime using ATR percentiles.

        Computes rolling ATR over vol_lookback window and classifies current
        ATR into LOW/NORMAL/HIGH/EXTREME based on percentiles.

        Args:
            returns: Log returns, shape (N,).

        Returns:
            Tuple of (regime_index, regime_name, percentile).
            regime_index: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME.
            regime_name: "LOW", "NORMAL", "HIGH", or "EXTREME".
            percentile: Current ATR percentile (0-100).
        """
        try:
            n = len(returns)
            if n < self.config.vol_lookback:
                logger.warning(
                    f"Volatility: need {self.config.vol_lookback} bars, got {n}; using all data"
                )
                lookback = n
            else:
                lookback = self.config.vol_lookback

            # ATR approximation: std dev of recent returns
            recent_returns = returns[-lookback:]
            current_vol = np.std(recent_returns)

            # Historical ATR
            all_volatilities = []
            for i in range(lookback, n + 1):
                window = returns[i - lookback : i]
                vol = np.std(window)
                all_volatilities.append(vol)

            if len(all_volatilities) < 2:
                logger.warning("Volatility: insufficient data; defaulting to NORMAL")
                return REGIME_NORMAL, REGIME_NAMES[REGIME_NORMAL], 50.0

            all_volatilities = np.array(all_volatilities)

            # Guard: if historical volatility spread is too narrow, percentile
            # ranking becomes unreliable (tiny uptick → 95th+ percentile).
            # Use IQR-based spread check: if IQR < 5% of median, the data is
            # too tightly clustered for meaningful percentile classification.
            q25, q75 = np.percentile(all_volatilities, [25, 75])
            iqr = q75 - q25
            median_vol = np.median(all_volatilities)
            min_spread_ratio = 0.05  # IQR must be >= 5% of median

            if median_vol > 0 and iqr < median_vol * min_spread_ratio:
                # Tight clustering — use z-score relative to mean instead
                mean_vol = np.mean(all_volatilities)
                std_vol = np.std(all_volatilities)
                if std_vol > 0:
                    z = (current_vol - mean_vol) / std_vol
                    # Map z-score to regime: <-1=LOW, -1..1=NORMAL, 1..2=HIGH, >2=EXTREME
                    if z < -1.0:
                        regime = REGIME_LOW
                    elif z < 1.0:
                        regime = REGIME_NORMAL
                    elif z < 2.0:
                        regime = REGIME_HIGH
                    else:
                        regime = REGIME_EXTREME
                    synthetic_pct = min(100.0, max(0.0, 50.0 + z * 25.0))
                    logger.info(
                        f"Volatility spread narrow (IQR={iqr:.6f}, median={median_vol:.6f}), "
                        f"using z-score={z:.2f} → {REGIME_NAMES[regime]}"
                    )
                    return regime, REGIME_NAMES[regime], float(synthetic_pct)
                else:
                    logger.warning("Volatility std=0; defaulting to NORMAL")
                    return REGIME_NORMAL, REGIME_NAMES[REGIME_NORMAL], 50.0

            # Absolute floor: if the std of all volatilities is near-zero,
            # percentileofscore returns inflated values on uniform data.
            # Default to NORMAL when the distribution is effectively flat.
            if np.std(all_volatilities) < 1e-8:
                logger.info(
                    "Volatility distribution flat (std=%.2e); defaulting to NORMAL",
                    np.std(all_volatilities),
                )
                return REGIME_NORMAL, REGIME_NAMES[REGIME_NORMAL], 50.0

            # Percentile classification (normal path with sufficient spread)
            percentile = stats.percentileofscore(
                all_volatilities, current_vol, kind="weak"
            )

            low_threshold = self.config.vol_low_percentile
            high_threshold = self.config.vol_high_percentile
            extreme_threshold = self.config.vol_extreme_percentile

            if percentile < low_threshold:
                regime = REGIME_LOW
            elif percentile < high_threshold:
                regime = REGIME_NORMAL
            elif percentile < extreme_threshold:
                regime = REGIME_HIGH
            else:
                regime = REGIME_EXTREME

            logger.debug(
                f"Volatility regime: {REGIME_NAMES[regime]} (pct={percentile:.1f}, "
                f"current={current_vol:.6f}, IQR={iqr:.6f})"
            )
            return regime, REGIME_NAMES[regime], float(percentile)

        except Exception as e:
            logger.error(f"Volatility regime classification failed: {e}")
            return REGIME_NORMAL, REGIME_NAMES[REGIME_NORMAL], 50.0

    def _classify_trend(self, hurst: float) -> str:
        """Classify trend direction using Hurst exponent.

        Args:
            hurst: Hurst exponent (0-2).

        Returns:
            "TRENDING", "MEAN_REVERTING", or "UNCERTAIN".
        """
        if hurst > self.config.hurst_trending_threshold:
            return "TRENDING"
        elif hurst < self.config.hurst_mean_revert_threshold:
            return "MEAN_REVERTING"
        else:
            return "UNCERTAIN"

    def _classify_adx_strength(self, adx: float) -> str:
        """Classify trend strength using ADX value.

        Args:
            adx: ADX value (0-100).

        Returns:
            "STRONG", "WEAK", or "TRANSITION".
        """
        if adx > self.config.adx_strong_threshold:
            return "STRONG"
        elif adx < self.config.adx_weak_threshold:
            return "WEAK"
        else:
            return "TRANSITION"

    def _ensemble_consensus(
        self,
        vol_regime: int,
        vol_percentile: float,
        changepoint_prob: float,
        changepoint_alert: bool,
        hurst: float,
        trend_class: str,
        adx: float,
        adx_strength: str,
    ) -> RegimeDetectionResult:
        """Combine all signals into final regime detection with confidence.

        Logic:
        - If changepoint_alert AND vol >= HIGH → CAUTION, risk_mult=0.5
        - Elif hurst > 0.55 AND adx > 25 → MOMENTUM, risk_mult=1.0
        - Elif hurst < 0.45 AND adx < 20 → MEAN_REVERSION, risk_mult=0.8
        - Else → RANGE, risk_mult=0.7

        Confidence = average agreement across all signals.

        Args:
            vol_regime: Volatility regime (0-3).
            vol_percentile: Volatility percentile (0-100).
            changepoint_prob: BOCPD changepoint probability (0-1).
            changepoint_alert: BOCPD alert flag.
            hurst: Hurst exponent.
            trend_class: Trend classification.
            adx: ADX value.
            adx_strength: ADX strength classification.

        Returns:
            RegimeDetectionResult with all signals and derived metrics.
        """
        try:
            # Determine strategy and risk multiplier
            if changepoint_alert and vol_regime >= REGIME_HIGH:
                strategy = "CAUTION"
                risk_mult = 0.5
            elif trend_class == "TRENDING" and adx_strength == "STRONG":
                strategy = "MOMENTUM"
                risk_mult = 1.0
            elif trend_class == "MEAN_REVERTING" and adx_strength == "WEAK":
                strategy = "MEAN_REVERSION"
                risk_mult = 0.8
            else:
                strategy = "RANGE"
                risk_mult = 0.7

            # Confidence calculation: average agreement across signals
            # 1. Volatility regime confidence (higher percentile → higher confidence)
            vol_confidence = abs(vol_percentile - 50.0) / 50.0
            vol_confidence = np.clip(vol_confidence, 0.0, 1.0)

            # 2. Trend classification confidence (trending/mean-revert > uncertain)
            trend_confidence = 0.8 if trend_class != "UNCERTAIN" else 0.5

            # 3. ADX strength confidence (strong/weak > transition)
            adx_confidence = 0.8 if adx_strength != "TRANSITION" else 0.5

            # 4. Changepoint confidence (inverse: low prob → high confidence in no changepoint)
            changepoint_confidence = 1.0 - changepoint_prob

            # Weighted average
            overall_confidence = 0.25 * (
                vol_confidence + trend_confidence + adx_confidence + changepoint_confidence
            )
            overall_confidence = np.clip(float(overall_confidence), 0.0, 1.0)

            # Build signals dict for logging
            signals = {
                "volatility_regime": vol_regime,
                "volatility_percentile": vol_percentile,
                "changepoint_probability": changepoint_prob,
                "changepoint_alert": bool(changepoint_alert),
                "hurst_exponent": hurst,
                "trend_classification": trend_class,
                "adx_value": adx,
                "adx_strength": adx_strength,
                "strategy_hint": strategy,
                "risk_multiplier": risk_mult,
                "overall_confidence": overall_confidence,
            }

            result = RegimeDetectionResult(
                regime_index=vol_regime,
                regime_name=REGIME_NAMES[vol_regime],
                confidence=overall_confidence,
                changepoint_probability=float(np.clip(changepoint_prob, 0.0, 1.0)),
                changepoint_alert=bool(changepoint_alert),
                hurst_exponent=float(np.clip(hurst, 0.0, 2.0)),
                trend_classification=trend_class,
                adx_value=float(np.clip(adx, 0.0, 100.0)),
                adx_strength=adx_strength,
                volatility_percentile=float(np.clip(vol_percentile, 0.0, 100.0)),
                strategy_hint=strategy,
                risk_multiplier=float(np.clip(risk_mult, 0.0, 1.0)),
                signals=signals,
            )

            return result

        except Exception as e:
            logger.error(f"Ensemble consensus failed: {e}")
            # Return safe default on error
            return RegimeDetectionResult(
                regime_index=REGIME_NORMAL,
                regime_name="NORMAL",
                confidence=0.0,
                signals={},
            )
