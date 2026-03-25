"""Walk-Forward Validation — Rolling window model evaluation to detect overfitting.

Splits historical data into N rolling windows. For each fold:
  - Train on window K (train_days candles)
  - Test on window K+1 (test_days candles)
  - Record accuracy, Sharpe, max drawdown

A model must pass walk-forward (>53% accuracy across ALL folds) to be promoted.
This catches overfitting that single train/test splits miss.

Default config: 5 folds, 60-day train, 20-day test, 10-day step.
At H1 granularity: ~1440 candles train, ~480 candles test, ~240 candle step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# H1 candles per day (24h market for FX)
CANDLES_PER_DAY = 24


@dataclass
class FoldResult:
    """Result for a single walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    accuracy: float  # Direction prediction accuracy (0-1)
    sharpe: float  # Annualized Sharpe on test period
    max_drawdown: float  # Maximum drawdown on test period (negative)
    wins: int
    losses: int


@dataclass
class WalkForwardResult:
    """Aggregate result across all folds."""

    pair: str
    n_folds: int
    fold_results: List[FoldResult] = field(default_factory=list)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    min_accuracy: float = 0.0
    mean_sharpe: float = 0.0
    mean_max_drawdown: float = 0.0
    passed: bool = False
    pass_threshold: float = 0.53
    timestamp: str = ""


class WalkForwardValidator:
    """Walk-forward validation framework for model evaluation.

    Usage:
        validator = WalkForwardValidator()
        result = validator.validate(df, pair="EUR_USD", predict_fn=model.predict)
        if result.passed:
            print("Model generalizes well!")
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_days: int = 60,
        test_days: int = 20,
        step_days: int = 10,
        min_accuracy: float = 0.53,
        results_path: str = "trained_data/models/walk_forward_results.json",
    ):
        self.n_folds = n_folds
        self.train_candles = train_days * CANDLES_PER_DAY
        self.test_candles = test_days * CANDLES_PER_DAY
        self.step_candles = step_days * CANDLES_PER_DAY
        self.min_accuracy = min_accuracy
        self.results_path = Path(results_path)
        self._results: Dict[str, WalkForwardResult] = {}
        self._load_results()

    def validate(
        self,
        df: pd.DataFrame,
        pair: str,
        predict_fn,
        feature_fn=None,
        lookahead: int = 5,
    ) -> WalkForwardResult:
        """Run walk-forward validation on historical data.

        Args:
            df: DataFrame with OHLCV columns (close required, time-indexed)
            pair: Currency pair name
            predict_fn: Function(features_df) -> direction_str ("LONG"/"SHORT"/"HOLD")
            feature_fn: Optional function(df) -> features_df. If None, passes df directly.
            lookahead: Number of candles ahead to check for direction (default 5 = 5h at H1)

        Returns:
            WalkForwardResult with per-fold and aggregate statistics
        """
        if len(df) < self.train_candles + self.test_candles:
            logger.warning(
                f"Insufficient data for walk-forward: {len(df)} candles "
                f"(need {self.train_candles + self.test_candles})"
            )
            return WalkForwardResult(
                pair=pair,
                n_folds=0,
                passed=False,
                pass_threshold=self.min_accuracy,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        fold_results = []
        total_needed = self.train_candles + self.test_candles

        for fold_idx in range(self.n_folds):
            offset = fold_idx * self.step_candles

            # Check we have enough data for this fold
            if offset + total_needed > len(df):
                logger.debug(f"Not enough data for fold {fold_idx}, stopping")
                break

            train_start = offset
            train_end = offset + self.train_candles
            test_start = train_end
            test_end = min(train_end + self.test_candles, len(df) - lookahead)

            if test_end <= test_start + 10:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            fold_result = self._evaluate_fold(
                fold_idx=fold_idx,
                train_df=train_df,
                test_df=test_df,
                full_df=df,
                test_end_abs=test_end,
                predict_fn=predict_fn,
                feature_fn=feature_fn,
                lookahead=lookahead,
            )
            fold_results.append(fold_result)

        if not fold_results:
            return WalkForwardResult(
                pair=pair,
                n_folds=0,
                passed=False,
                pass_threshold=self.min_accuracy,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Compute aggregates
        accuracies = [f.accuracy for f in fold_results]
        sharpes = [f.sharpe for f in fold_results]
        drawdowns = [f.max_drawdown for f in fold_results]

        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        min_acc = float(np.min(accuracies))

        # Pass if mean accuracy above threshold AND no fold is catastrophically bad
        passed = mean_acc >= self.min_accuracy and min_acc >= (self.min_accuracy - 0.05)

        result = WalkForwardResult(
            pair=pair,
            n_folds=len(fold_results),
            fold_results=fold_results,
            mean_accuracy=round(mean_acc, 4),
            std_accuracy=round(std_acc, 4),
            min_accuracy=round(min_acc, 4),
            mean_sharpe=round(float(np.mean(sharpes)), 4),
            mean_max_drawdown=round(float(np.mean(drawdowns)), 4),
            passed=passed,
            pass_threshold=self.min_accuracy,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Persist result
        self._results[pair] = result
        self._save_results()

        logger.info(
            f"Walk-forward {pair}: {mean_acc:.1%} mean accuracy "
            f"({'PASSED' if passed else 'FAILED'}) across {len(fold_results)} folds"
        )

        return result

    def _evaluate_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        test_end_abs: int,
        predict_fn,
        feature_fn,
        lookahead: int,
    ) -> FoldResult:
        """Evaluate a single walk-forward fold."""
        wins = 0
        losses = 0
        returns = []

        # Compute features for test set
        features_df = feature_fn(test_df) if feature_fn else test_df

        close_col = "close" if "close" in test_df.columns else test_df.columns[-1]

        for i in range(0, len(features_df) - lookahead, max(1, lookahead)):
            try:
                row = features_df.iloc[i : i + 1]
                prediction = predict_fn(row)

                if prediction is None:
                    continue

                pred_dir = str(prediction).upper()
                if pred_dir not in ("LONG", "SHORT"):
                    continue

                # Actual direction from price movement
                current_price = float(test_df[close_col].iloc[i])
                future_price = float(test_df[close_col].iloc[min(i + lookahead, len(test_df) - 1)])
                actual_dir = "LONG" if future_price > current_price else "SHORT"
                pct_return = (future_price - current_price) / current_price

                if pred_dir == actual_dir:
                    wins += 1
                    returns.append(abs(pct_return))
                else:
                    losses += 1
                    returns.append(-abs(pct_return))

            except Exception:
                continue

        total = wins + losses
        accuracy = wins / total if total > 0 else 0.0

        # Sharpe from returns
        if returns and len(returns) > 1:
            ret_arr = np.array(returns)
            sharpe = (
                float(np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252 * 24))
                if np.std(ret_arr) > 0
                else 0.0
            )
        else:
            sharpe = 0.0

        # Max drawdown from cumulative returns
        if returns:
            cum = np.cumsum(returns)
            peak = np.maximum.accumulate(cum)
            drawdown = float(np.min(cum - peak))
        else:
            drawdown = 0.0

        # Timestamps for reporting
        train_start_ts = str(train_df.index[0]) if hasattr(train_df.index, '__getitem__') else ""
        train_end_ts = str(train_df.index[-1]) if hasattr(train_df.index, '__getitem__') else ""
        test_start_ts = str(test_df.index[0]) if hasattr(test_df.index, '__getitem__') else ""
        test_end_ts = str(test_df.index[-1]) if hasattr(test_df.index, '__getitem__') else ""

        return FoldResult(
            fold=fold_idx,
            train_start=train_start_ts,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            train_samples=len(train_df),
            test_samples=len(test_df),
            accuracy=round(accuracy, 4),
            sharpe=round(sharpe, 4),
            max_drawdown=round(drawdown, 6),
            wins=wins,
            losses=losses,
        )

    def _load_results(self) -> None:
        """Load previous walk-forward results."""
        try:
            if self.results_path.exists():
                data = json.loads(self.results_path.read_text())
                for pair, rec in data.items():
                    folds = [FoldResult(**f) for f in rec.get("fold_results", [])]
                    rec_copy = {k: v for k, v in rec.items() if k != "fold_results"}
                    self._results[pair] = WalkForwardResult(fold_results=folds, **rec_copy)
        except Exception as e:
            logger.debug(f"Walk-forward results load failed: {e}")

    def _save_results(self) -> None:
        """Persist walk-forward results to JSON."""
        try:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for pair, result in self._results.items():
                d = asdict(result)
                data[pair] = d
            self.results_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Walk-forward results save failed: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary of all walk-forward results."""
        if not self._results:
            return "No walk-forward validation results yet."

        lines = [
            "Walk-Forward Validation Summary",
            "=" * 90,
            f"{'Pair':12} {'Folds':>6} {'Mean Acc':>10} {'Min Acc':>10} "
            f"{'Sharpe':>8} {'MaxDD':>8} {'Status':>8}",
            "-" * 90,
        ]

        for pair, res in sorted(self._results.items()):
            status = "PASS" if res.passed else "FAIL"
            lines.append(
                f"{pair:12} {res.n_folds:>6} {res.mean_accuracy:>10.1%} "
                f"{res.min_accuracy:>10.1%} {res.mean_sharpe:>8.2f} "
                f"{res.mean_max_drawdown:>8.4f} {status:>8}"
            )

        return "\n".join(lines)
