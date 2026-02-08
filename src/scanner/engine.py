"""
Scanner Engine Module.

Core scanning logic using ThreadPoolExecutor for parallel pair analysis.
Handles data fetching, feature engineering, and incremental caching.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ScannerConfig, load_yaml_config
from .gates import GateEvaluator
from .results import PairAnalysis, ScanResult
from .execution import ExecutionManager, ExecutionConfig, ExecutionResult
from .analysis import QuickBacktester, CorrelationAnalyzer, DriftDetector
from .filters import VolatilityFilter, DiversificationFilter

logger = logging.getLogger(__name__)


class Scanner:
    """FX Pair Scanner with parallel execution and incremental updates.

    Key features:
    - ThreadPoolExecutor for parallel pair scanning
    - Incremental caching to skip unchanged pairs
    - CatBoost/XGBoost gate models
    - Non-interactive mode for scripts
    - Rich display integration
    """

    def __init__(
        self,
        config: Optional[ScannerConfig] = None,
        oanda_client: Optional[Any] = None,
    ):
        """Initialize scanner.

        Args:
            config: Scanner configuration (loads default if None)
            oanda_client: OANDA API client (created if None)
        """
        self.config = config or ScannerConfig()
        self._oanda = oanda_client

        # Lazy-loaded components
        self._gate_evaluator: Optional[GateEvaluator] = None
        self._feature_engineer = None
        self._modular_ensemble = None
        self._executor: Optional[ExecutionManager] = None

        # Analysis and filter components
        self._backtester: Optional[QuickBacktester] = None
        self._correlation_analyzer: Optional[CorrelationAnalyzer] = None
        self._drift_detector: Optional[DriftDetector] = None
        self._volatility_filter: Optional[VolatilityFilter] = None
        self._diversification_filter: Optional[DiversificationFilter] = None

        # Incremental caching
        self._last_scan_times: Dict[str, float] = {}
        self._last_prices: Dict[str, float] = {}
        self._cached_results: Dict[str, PairAnalysis] = {}
        self._pair_returns: Dict[str, pd.Series] = {}

        # Load config
        self._load_yaml_config()

    def _load_yaml_config(self) -> None:
        """Load YAML configuration and update settings."""
        try:
            yaml_config = load_yaml_config(self.config.config_path)

            # Update config from YAML
            if yaml_config:
                buddy_config = yaml_config.get("buddy", {})
                self.config.risk_per_trade_pct = buddy_config.get(
                    "risk_per_trade_pct", self.config.risk_per_trade_pct
                )
                self.config.sl_pips = buddy_config.get("stop_loss_pips", self.config.sl_pips)
                self.config.tp_pips = buddy_config.get("take_profit_pips", self.config.tp_pips)

                # Gate thresholds from inference config
                inference_config = yaml_config.get("inference", {})
                self.config.min_tcn_probability = inference_config.get(
                    "min_tcn_probability", self.config.min_tcn_probability
                )
                self.config.min_confidence = inference_config.get(
                    "min_confidence", self.config.min_confidence
                )
                self.config.min_momentum = inference_config.get(
                    "min_momentum", self.config.min_momentum
                )
                self.config.max_drawdown_pct = inference_config.get(
                    "max_drawdown_pct", self.config.max_drawdown_pct
                )

                logger.debug(f"Loaded config from {self.config.config_path}")

        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}")

    def _init_oanda_client(self) -> bool:
        """Initialize OANDA client.

        Returns:
            True if client initialized successfully
        """
        if self._oanda is not None:
            return True

        try:
            from src.utils.oanda_practice import OandaPracticeClient
            self._oanda = OandaPracticeClient.from_env()
            return True
        except ImportError:
            logger.error("OANDA client not available - install oanda-api-v20")
            return False
        except OSError as e:
            logger.warning(f"OANDA credentials not configured: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OANDA client: {e}")
            return False

    def _init_gate_evaluator(self) -> bool:
        """Initialize gate evaluator with JOINT models (preferred) or per-pair fallback.

        Scanner prefers joint-trained models from:
        trained_data/models/joint/

        Falls back to per-pair models if joint models not available.

        Returns:
            True if gates loaded successfully
        """
        if self._gate_evaluator is not None:
            return self._gate_evaluator.is_loaded

        model_dir = Path(self.config.model_dir)
        joint_dir = model_dir / "joint"

        # Check if joint models exist
        use_joint_only = getattr(self.config, 'use_joint_models_only', True)
        require_tcn = getattr(self.config, 'require_tcn_volatility', True)

        # If joint directory doesn't exist, fall back to per-pair models
        if use_joint_only and not joint_dir.exists():
            logger.warning(
                f"Joint model directory not found: {joint_dir}\n"
                f"Falling back to per-pair models. For best results, run:\n"
                f"  python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY"
            )
            use_joint_only = False
            require_tcn = False  # Can't require TCN if using fallback

        # Check if TCN model exists in joint dir - if not, don't require it
        tcn_path = joint_dir / "tcn_volatility_regime.keras"
        if use_joint_only and require_tcn and not tcn_path.exists():
            logger.warning(
                "TCN Volatility Regime model not found in joint directory.\n"
                "Scanner will run without volatility filtering.\n"
                "To train TCN model, run: python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY"
            )
            require_tcn = False

        self._gate_evaluator = GateEvaluator(model_dir, use_joint_only=use_joint_only)

        try:
            status = self._gate_evaluator.load_models(require_tcn=require_tcn)
            return any(status.values())
        except FileNotFoundError as e:
            logger.warning(f"Model loading issue: {e}")
            # Still return False but don't raise - allow scan to continue with reduced functionality
            return False

    def _init_feature_engineer(self) -> bool:
        """Initialize feature engineering module.

        Returns:
            True if initialized successfully
        """
        if self._feature_engineer is not None:
            return True

        try:
            from src.core.modular_data_loaders import compute_normalized_features
            # Store function reference directly
            self._feature_engineer = compute_normalized_features
            return True
        except ImportError:
            logger.debug("compute_normalized_features not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize feature engineer: {e}")
            return False

    def _init_modular_ensemble(self) -> bool:
        """Initialize modular ensemble for direction prediction.

        Returns:
            True if ensemble loaded successfully
        """
        if self._modular_ensemble is not None:
            return True

        try:
            from src.core.modular_inference import ModularEnsembleInference

            self._modular_ensemble = ModularEnsembleInference(
                model_dir=str(self.config.model_dir),
            )
            # Check if at least the main direction model loaded
            return self._modular_ensemble.tcn is not None or self._modular_ensemble.histgb is not None

        except ImportError:
            logger.debug("ModularEnsembleInference not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize ensemble: {e}")
            return False

    def _init_analysis_tools(self) -> None:
        """Initialize analysis and filter components."""
        if self._backtester is None:
            self._backtester = QuickBacktester(
                window=50,
                granularity=self.config.granularity,
            )

        if self._correlation_analyzer is None:
            self._correlation_analyzer = CorrelationAnalyzer(
                threshold=0.7,
                lookback_periods=100,
            )

        if self._drift_detector is None:
            self._drift_detector = DriftDetector(
                threshold=0.03,
                model_dir=self.config.model_dir,
            )

        if self._volatility_filter is None:
            self._volatility_filter = VolatilityFilter(
                min_atr_pips=self.config.min_atr_pips,
                model_dir=self.config.model_dir,
            )

        if self._diversification_filter is None:
            self._diversification_filter = DiversificationFilter(
                correlation_threshold=0.7,
            )

    def _fetch_pair_data(
        self,
        pair: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a pair.

        Args:
            pair: Instrument name (e.g., "EUR_USD")
            count: Number of candles to fetch

        Returns:
            DataFrame with OHLCV columns or None on failure
        """
        if not self._init_oanda_client():
            return None

        try:
            raw = self._oanda.get_candles(
                instrument=pair,
                granularity=self.config.granularity,
                count=count,
            )

            # Parse OANDA JSON response into DataFrame
            candles = raw.get("candles", [])
            if not candles:
                logger.debug(f"{pair}: No candles returned")
                return None

            # Extract OHLCV from mid prices
            data = []
            for c in candles:
                if not c.get("complete", True):
                    continue
                mid = c.get("mid", {})
                data.append({
                    "time": c.get("time"),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })

            if not data:
                logger.debug(f"{pair}: No complete candles")
                return None

            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")

            if len(df) < self.config.min_candles:
                logger.debug(f"{pair}: Insufficient data ({len(df)} candles)")
                return None

            return df

        except Exception as e:
            logger.debug(f"{pair}: Data fetch failed - {e}")
            return None

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features for analysis.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self._init_feature_engineer():
            # Basic fallback features
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(14).mean()
            return df.ffill().bfill().fillna(0)

        try:
            # _feature_engineer is now compute_normalized_features function
            df_feat = self._feature_engineer(df.copy())
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            df_feat = df_feat.ffill().bfill().fillna(0.0)
            return df_feat
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            return df.ffill().bfill().fillna(0)

    def _run_inference(
        self,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        pair: str,
    ) -> Tuple[str, float, float, float, bool, Optional[int]]:
        """Run model inference on features.

        IMPORTANT: TCN Volatility Regime is evaluated FIRST as a global filter.
        If volatility is LOW or NORMAL, gates_passed is False regardless of other signals.

        Args:
            df_raw: Raw OHLCV data
            df_feat: Engineered features
            pair: Instrument name

        Returns:
            Tuple of (direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime)
        """
        direction = "LONG"
        confidence = 0.5
        tcn_conf = 0.0
        ridge_conf = 0.0
        gates_passed = False
        volatility_regime = None

        # === FIRST: Check TCN Volatility Regime (GLOBAL GATE) ===
        if self._init_gate_evaluator() and self._gate_evaluator is not None:
            if self._gate_evaluator.tcn_volatility_available:
                regime, _vol_conf, vol_allowed = self._gate_evaluator.evaluate_volatility_regime(df_feat)
                volatility_regime = regime

                if not vol_allowed:
                    # LOW or NORMAL volatility - block ALL trades
                    regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
                    logger.info(f"{pair}: Blocked by TCN Volatility Regime ({regime_names[regime]})")
                    return direction, confidence, tcn_conf, ridge_conf, False, volatility_regime
            # If TCN not available, just skip the volatility check (don't block)

        # === SECOND: Evaluate direction and other gates ===
        # Try modular ensemble first
        if self._init_modular_ensemble() and self._modular_ensemble is not None:
            try:
                signal = self._modular_ensemble.predict(df_feat)

                if signal.tcn_direction is not None:
                    direction = "LONG" if signal.tcn_direction == 1 else "SHORT"
                    tcn_conf = abs(signal.tcn_probability - 0.5) * 200
                    ridge_conf = signal.ridge_confidence
                    confidence = max(tcn_conf, ridge_conf) / 100
                    gates_passed = signal.trade
                elif signal.direction is not None:
                    direction = signal.direction.upper()
                    confidence = signal.confidence if signal.confidence > 0 else 0.5
                    ridge_conf = signal.ridge_confidence
                    tcn_conf = confidence * 100
                    gates_passed = signal.trade

                return direction, min(confidence, 0.95), tcn_conf, ridge_conf, gates_passed, volatility_regime

            except Exception as e:
                logger.debug(f"{pair}: Ensemble inference failed - {e}")

        # Evaluate gates directly if ensemble not available
        if self._init_gate_evaluator() and self._gate_evaluator is not None:
            try:
                # Pass full DataFrame for Transformer (needs sequence)
                # but last row features for XGB/Ridge/RF
                gate_result = self._gate_evaluator.evaluate_all_gates(
                    df_feat,  # Full DataFrame for Transformer sequence
                    min_confidence=self.config.min_confidence,
                    min_momentum=self.config.min_momentum,
                    max_drawdown_pct=self.config.max_drawdown_pct,
                )

                confidence = gate_result["momentum"]
                ridge_conf = gate_result["confidence"]
                gates_passed = gate_result["all_passed"]

                # Use direction from gates if Transformer provided it
                if gate_result.get("transformer_direction"):
                    direction = gate_result["transformer_direction"]
                    tcn_conf = gate_result.get("transformer_prob", 0.5) * 100
                elif "rsi" in df_feat.columns:
                    # Fallback to RSI-based direction
                    rsi = df_feat["rsi"].iloc[-1]
                    direction = "LONG" if rsi < 50 else "SHORT"

            except Exception as e:
                logger.debug(f"{pair}: Gate evaluation failed - {e}")

        # Technical indicator fallback
        if "rsi" in df_feat.columns:
            rsi = df_feat["rsi"].iloc[-1]
            if rsi < 30:
                direction = "LONG"
                confidence = 0.5 + (30 - rsi) / 60
            elif rsi > 70:
                direction = "SHORT"
                confidence = 0.5 + (rsi - 70) / 60
            elif "macd" in df_feat.columns:
                macd = df_feat["macd"].iloc[-1]
                direction = "LONG" if macd > 0 else "SHORT"
                confidence = 0.5 + min(abs(macd) * 10, 0.2)

        return direction, min(confidence, 0.95), tcn_conf, ridge_conf, gates_passed, volatility_regime

    def _calculate_metrics(
        self,
        df: pd.DataFrame,
        pair: str,
    ) -> Dict[str, float]:
        """Calculate technical metrics.

        Args:
            df: DataFrame with features
            pair: Instrument name

        Returns:
            Dict of metric values
        """
        metrics = {
            "current_price": df["close"].iloc[-1],
            "atr": df["atr"].iloc[-1] if "atr" in df.columns else 0.0,
            "volatility_percentile": 0.5,
            "trend_strength": 0.5,
            "entry_score": 0.5,
        }

        # Volatility percentile
        if "atr" in df.columns:
            atr_series = df["atr"].dropna()
            if len(atr_series) > 20:
                metrics["volatility_percentile"] = float(
                    (atr_series < atr_series.iloc[-1]).mean()
                )

        # Trend strength from ADX
        if "adx" in df.columns:
            adx_val = df["adx"].iloc[-1]
            metrics["trend_strength"] = min(float(adx_val) / 60, 1.0)

        # Entry score
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            if 30 < rsi < 70:
                metrics["entry_score"] += 0.15

        if "sma_20" in df.columns and "sma_50" in df.columns:
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1]
            close = df["close"].iloc[-1]

            if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                metrics["entry_score"] += 0.15

        return metrics

    def _check_price_changed(
        self,
        pair: str,
        current_price: float,
        threshold: float = 0.0001,
    ) -> bool:
        """Check if price changed enough to warrant rescan.

        Args:
            pair: Instrument name
            current_price: Current price
            threshold: Minimum price change ratio

        Returns:
            True if price changed significantly
        """
        if pair not in self._last_prices:
            self._last_prices[pair] = current_price
            return True

        last_price = self._last_prices[pair]
        change_ratio = abs(current_price - last_price) / last_price

        if change_ratio >= threshold:
            self._last_prices[pair] = current_price
            return True

        return False

    def _scan_pair(self, pair: str) -> Optional[PairAnalysis]:
        """Scan a single pair (atomic unit for parallel execution).

        Args:
            pair: Instrument name

        Returns:
            PairAnalysis or None on failure
        """
        try:
            # Check session timing
            if self.config.session_filter_enabled:
                if not self.config.is_within_session():
                    return PairAnalysis(
                        pair=pair,
                        direction="HOLD",
                        confidence=0.0,
                        error="Outside trading session",
                    )

            # Fetch data
            df_raw = self._fetch_pair_data(pair, self.config.lookback_candles + 200)
            if df_raw is None:
                return PairAnalysis(
                    pair=pair,
                    direction="HOLD",
                    confidence=0.0,
                    error="Data fetch failed",
                )

            # Check incremental cache
            current_price = df_raw["close"].iloc[-1]
            if self.config.incremental_enabled:
                if not self._check_price_changed(pair, current_price, self.config.price_change_threshold):
                    # Return cached result if available
                    if pair in self._cached_results:
                        cached = self._cached_results[pair]
                        return cached

            # Compute features
            df_feat = self._compute_features(df_raw)

            # Store returns for correlation analysis
            if "close" in df_feat.columns:
                self._pair_returns[pair] = df_feat["close"].pct_change().dropna()

            # Check minimum volatility
            # compute_normalized_features produces atr_pct_14, convert to absolute ATR
            pip_value = self.config.pip_values.get(pair, 0.0001)
            if "atr" in df_feat.columns:
                atr = df_feat["atr"].iloc[-1]
            elif "atr_pct_14" in df_feat.columns and "close" in df_raw.columns:
                # Convert percentage ATR back to absolute
                atr = df_feat["atr_pct_14"].iloc[-1] * df_raw["close"].iloc[-1]
            else:
                atr = 0.0
            atr_pips = atr / pip_value if pip_value > 0 else 0

            if atr_pips < self.config.min_atr_pips:
                return PairAnalysis(
                    pair=pair,
                    direction="HOLD",
                    confidence=0.0,
                    error=f"Low volatility (ATR={atr_pips:.1f} pips)",
                )

            # Run inference (TCN Volatility Regime is checked FIRST inside)
            direction, confidence, tcn_conf, ridge_conf, gates_passed, volatility_regime = self._run_inference(
                df_raw, df_feat, pair
            )

            # Calculate metrics
            metrics = self._calculate_metrics(df_feat, pair)

            # Calculate position sizing
            sl_pips = self.config.sl_pips
            tp_pips = self.config.tp_pips

            # Kelly-based position sizing (simplified)
            risk_pct = self.config.risk_per_trade_pct
            if confidence > 0.65:
                # Higher confidence = slightly larger position
                risk_pct = min(risk_pct * 1.25, 0.03)

            # Build error message if volatility regime blocked the trade
            error_msg = None
            regime_name = "UNKNOWN"
            if volatility_regime is not None:
                regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
                regime_name = regime_names[volatility_regime] if 0 <= volatility_regime <= 3 else "UNKNOWN"
                if volatility_regime < getattr(self.config, 'min_volatility_regime', 2):
                    error_msg = f"Blocked: {regime_name} volatility regime"

            # Create analysis result
            result = PairAnalysis(
                pair=pair,
                direction=direction,
                confidence=confidence,
                tcn_confidence=tcn_conf,
                ridge_confidence=ridge_conf,
                momentum=confidence,  # Use as proxy
                momentum_acceleration=gates_passed,
                momentum_passed=confidence >= self.config.min_momentum,
                confidence_score=ridge_conf,
                confidence_passed=ridge_conf >= self.config.min_confidence,
                drawdown=0.02,  # Placeholder
                risk_passed=True,  # Simplified
                gates_passed=gates_passed,
                current_price=metrics["current_price"],
                atr=metrics["atr"],
                atr_pips=atr_pips,
                volatility_percentile=metrics["volatility_percentile"],
                trend_strength=metrics["trend_strength"],
                entry_score=metrics["entry_score"],
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                risk_pct=risk_pct,
                scan_time=datetime.now(timezone.utc),
                volatility_regime=regime_name,  # TCN volatility regime as string
                error=error_msg,
            )

            # Cache result
            self._cached_results[pair] = result
            self._last_scan_times[pair] = time.time()

            return result

        except Exception as e:
            logger.error(f"Error scanning {pair}: {e}")
            return PairAnalysis(
                pair=pair,
                direction="HOLD",
                confidence=0.0,
                error=str(e),
            )

    def scan(
        self,
        pairs: Optional[List[str]] = None,
        max_workers: int = 4,
        on_pair_complete: Optional[Callable[[PairAnalysis], None]] = None,
    ) -> ScanResult:
        """Scan multiple pairs in parallel.

        Args:
            pairs: List of pairs to scan (uses default if None)
            max_workers: Number of parallel workers
            on_pair_complete: Callback when each pair completes (for live updates)

        Returns:
            ScanResult with all pair analyses
        """
        pair_list = pairs or self.config.default_pairs

        # Check session before starting
        if self.config.session_filter_enabled and not self.config.non_interactive:
            if not self.config.is_within_session():
                current_hour = datetime.now(timezone.utc).hour
                logger.warning(
                    f"Outside trading session ({current_hour}:00 UTC). "
                    f"Active hours: {self.config.session_start_utc}-{self.config.session_end_utc} UTC"
                )

        # Initialize components
        self._init_oanda_client()
        self._init_gate_evaluator()
        self._init_feature_engineer()
        self._init_modular_ensemble()

        # Parallel scan
        analyses: List[PairAnalysis] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._scan_pair, pair): pair
                for pair in pair_list
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        analyses.append(result)

                        # Callback for live display updates
                        if on_pair_complete:
                            on_pair_complete(result)

                except Exception as e:
                    logger.error(f"Scan failed for {pair}: {e}")
                    analyses.append(PairAnalysis(
                        pair=pair,
                        direction="HOLD",
                        confidence=0.0,
                        error=str(e),
                    ))

        # Build scan result
        model_type = "technical"
        if self._modular_ensemble is not None:
            model_type = "ensemble"
        elif self._gate_evaluator is not None and self._gate_evaluator.is_loaded:
            model_type = self._gate_evaluator.momentum_model_type

        return ScanResult(
            analyses=analyses,
            model_type=model_type,
            granularity=self.config.granularity,
        )

    def scan_with_analysis(
        self,
        pairs: Optional[List[str]] = None,
        max_workers: int = 4,
        run_backtest: bool = True,
        run_correlation: bool = True,
        run_drift_check: bool = True,
        apply_diversification: bool = True,
    ) -> ScanResult:
        """Scan with full analysis pipeline.

        Runs the standard scan plus:
        - Quick backtests on tradeable pairs
        - Correlation analysis and diversification filter
        - Model drift detection

        Args:
            pairs: List of pairs to scan
            max_workers: Parallel workers
            run_backtest: Run quick backtests
            run_correlation: Analyze correlations
            run_drift_check: Check model drift
            apply_diversification: Filter correlated pairs

        Returns:
            ScanResult with enhanced analytics
        """
        # Initialize analysis tools
        self._init_analysis_tools()

        # Run base scan
        result = self.scan(pairs=pairs, max_workers=max_workers)

        # Build data cache for analysis
        data_cache: Dict[str, pd.DataFrame] = {}
        for pair in self._cached_results:
            if pair in self._pair_returns:
                # We need full DataFrame - fetch if needed
                df = self._fetch_pair_data(pair, self.config.lookback_candles + 200)
                if df is not None:
                    data_cache[pair] = self._compute_features(df)

        # Run quick backtests on tradeable pairs
        if run_backtest and self._backtester:
            for analysis in result.analyses:
                if analysis.is_tradeable and analysis.pair in data_cache:
                    bt_result = self._backtester.run(
                        data_cache[analysis.pair],
                        analysis.direction,
                    )
                    if bt_result.success:
                        analysis.backtest_pnl = bt_result.total_pnl
                        analysis.backtest_sharpe = bt_result.sharpe

        # Correlation analysis
        if run_correlation and self._correlation_analyzer:
            confidence_lookup = {a.pair: a.confidence for a in result.analyses}
            corr_results = self._correlation_analyzer.analyze(
                self._pair_returns,
                confidence_lookup,
            )

            # Update analyses with correlation info
            for analysis in result.analyses:
                if analysis.pair in corr_results:
                    cr = corr_results[analysis.pair]
                    analysis.correlation_group = cr.correlation_group

            # Apply diversification filter
            if apply_diversification and self._diversification_filter:
                result.analyses = self._diversification_filter.filter(
                    result.analyses,
                    returns_data=self._pair_returns,
                    apply_position_reduction=True,
                )

        # Model drift detection
        if run_drift_check and self._drift_detector:
            global_drift = self._drift_detector.detect()

            if global_drift.needs_retrain:
                for analysis in result.analyses:
                    analysis.needs_training = True
                    analysis.training_reason = global_drift.retrain_reason
                    analysis.model_drift_score = global_drift.drift_amount

        return result

    def scan_incremental(
        self,
        pairs: Optional[List[str]] = None,
        interval_seconds: float = 60.0,
        max_iterations: Optional[int] = None,
        on_update: Optional[Callable[[ScanResult], None]] = None,
    ) -> None:
        """Run incremental scanning loop.

        Only re-scans pairs whose price has changed significantly.

        Args:
            pairs: List of pairs to scan
            interval_seconds: Time between scans
            max_iterations: Stop after N iterations (None = infinite)
            on_update: Callback with updated results
        """
        pair_list = pairs or self.config.default_pairs
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            try:
                # Scan all pairs (incremental cache handles skipping)
                result = self.scan(
                    pairs=pair_list,
                    max_workers=self.config.max_workers,
                )

                # Callback with results
                if on_update:
                    on_update(result)

                # Wait for next interval
                time.sleep(interval_seconds)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("Scan loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scan iteration failed: {e}")
                time.sleep(interval_seconds)

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information.

        Returns:
            Dict with account details (equity, trades today, etc.)
        """
        if not self._init_oanda_client():
            return {"error": "OANDA client not available"}

        try:
            account = self._oanda.get_account_summary()
            return {
                "balance": account.get("balance", 0),
                "nav": account.get("NAV", 0),
                "unrealized_pl": account.get("unrealizedPL", 0),
                "margin_used": account.get("marginUsed", 0),
                "margin_available": account.get("marginAvailable", 0),
                "open_positions": account.get("openPositionCount", 0),
                "open_trades": account.get("openTradeCount", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _init_executor(self) -> bool:
        """Initialize execution manager.

        Returns:
            True if initialized successfully
        """
        if self._executor is not None:
            return True

        try:
            exec_config = ExecutionConfig(
                account_equity=self.config.account_equity,
                risk_per_trade_pct=self.config.risk_per_trade_pct,
                leverage=self.config.leverage,
                position_sizing_enabled=self.config.position_sizing_enabled,
                aggressive_mode=self.config.aggressive_mode,
                atr_sl_multiplier=self.config.atr_sl_multiplier,
                atr_tp_multiplier=self.config.atr_tp_multiplier,
                min_sl_pips=self.config.min_sl_pips,
                max_sl_pips=self.config.max_sl_pips,
                min_tp_pips=self.config.min_tp_pips,
                max_tp_pips=self.config.max_tp_pips,
                high_prob_threshold=self.config.high_prob_threshold,
                high_prob_tp_bonus=self.config.high_prob_tp_bonus,
                max_trades_per_day=self.config.daily_trade_limit,
            )
            self._executor = ExecutionManager(
                config=exec_config,
                oanda_client=self._oanda,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            return False

    def execute_trades(
        self,
        analyses: List[PairAnalysis],
        max_trades: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """Execute trades for tradeable pairs.

        Args:
            analyses: List of PairAnalysis results to execute
            max_trades: Maximum number of trades to execute

        Returns:
            List of ExecutionResult for each trade attempted
        """
        if not self.config.enable_execution:
            logger.warning("Execution disabled in config. Set enable_execution=True")
            return []

        if not self._init_executor():
            return []

        # Filter to tradeable pairs only
        tradeable = [a for a in analyses if a.is_tradeable]

        if max_trades:
            tradeable = tradeable[:max_trades]

        if not tradeable:
            logger.info("No tradeable pairs to execute")
            return []

        # Convert to execution format
        trades = [
            {
                "pair": a.pair,
                "direction": a.direction,
                "confidence": a.confidence,
                "current_price": a.current_price,
                "atr": a.atr,
                "sl_pips": a.sl_pips,
                "tp_pips": a.tp_pips,
                "recommended_lots": a.recommended_lots,
            }
            for a in tradeable
        ]

        return self._executor.execute_trades(trades, granularity=self.config.granularity)

    def scan_and_execute(
        self,
        pairs: Optional[List[str]] = None,
        max_trades: int = 3,
    ) -> Tuple[ScanResult, List[ExecutionResult]]:
        """Scan pairs and execute top trades.

        Args:
            pairs: List of pairs to scan (uses default if None)
            max_trades: Maximum number of trades to execute

        Returns:
            Tuple of (ScanResult, list of ExecutionResult)
        """
        # Scan first
        result = self.scan(pairs=pairs)

        # Execute if enabled
        if self.config.enable_execution:
            executions = self.execute_trades(result.tradeable, max_trades=max_trades)
        else:
            executions = []

        return result, executions

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status including daily limits.

        Returns:
            Dict with execution status
        """
        if not self._init_executor():
            return {"error": "Executor not available"}

        nav, trades_today, trades_remaining = self._executor.get_account_status()
        can_trade, reason = self._executor.can_trade()

        return {
            "can_trade": can_trade,
            "reason": reason,
            "nav": nav,
            "trades_today": trades_today,
            "trades_remaining": trades_remaining,
            "daily_limit": self.config.daily_trade_limit,
            "execution_enabled": self.config.enable_execution,
        }
