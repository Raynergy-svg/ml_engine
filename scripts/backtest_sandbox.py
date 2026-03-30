#!/usr/bin/env python3
"""
Backtest Sandbox — Walk-forward backtester for Buddy FX Engine.

Replays CSV market data through the full Scanner pipeline:
  CSV → Features → Inference → Gates → Agents → Simulated Execution

Usage:
  python scripts/backtest_sandbox.py --mode aggressive --equity 100
  python scripts/backtest_sandbox.py --mode realistic --pairs EUR_USD GBP_USD --equity 10000
  python scripts/backtest_sandbox.py --mode aggressive --equity 100 --scan-interval 4
"""

from __future__ import annotations

# --- TF/OpenMP thread safety (must be before TF/numpy imports) ---
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['KMP_AFFINITY'] = 'disabled'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
# Force TF single-threaded before anything else imports it
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# -----------------------------------------------------------

import argparse
import glob
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scanner.engine import Scanner
from src.scanner.config import ScannerConfig, MAJOR_PAIRS, SCAN_PROFILES
from src.brokers.registry import get_registry

logger = logging.getLogger(__name__)
print("[STARTUP] backtest_sandbox.py loaded", flush=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    pair: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    units: int
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    confidence: float
    risk_amount: float


@dataclass
class ClosedTrade:
    pair: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    units: int
    sl_pips: float
    tp_pips: float
    pnl_pips: float
    pnl_dollars: float
    exit_reason: str  # SL, TP, END
    confidence: float
    transaction_cost: float


@dataclass
class BacktestResult:
    starting_equity: float
    final_equity: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown_pct: float
    max_drawdown_amount: float
    sharpe_ratio: float
    profit_factor: float
    avg_win_pips: float
    avg_loss_pips: float
    best_trade: float
    worst_trade: float
    equity_curve: List[Tuple[datetime, float]]
    trades: List[ClosedTrade]
    pair_breakdown: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# CSV Data Loader
# ---------------------------------------------------------------------------

def load_pair_data(pair: str, data_dir: str = "market_data") -> Optional[pd.DataFrame]:
    """Load and merge all CSV files for a pair, deduplicate by timestamp."""
    pattern = os.path.join(PROJECT_ROOT, data_dir, f"oanda_{pair}_H1_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning(f"No CSV files found for {pair}")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "time" in df.columns:
                dfs.append(df)
        except Exception as e:
            logger.debug(f"Skipping {f}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"], utc=True)
    merged = merged.drop_duplicates(subset=["time"], keep="last")
    merged = merged.sort_values("time").reset_index(drop=True)
    merged = merged.set_index("time")

    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in merged.columns:
            logger.warning(f"{pair}: Missing column '{col}'")
            return None

    logger.info(f"{pair}: Loaded {len(merged)} candles ({merged.index[0]} → {merged.index[-1]})")
    return merged


def load_all_pairs(pairs: List[str], data_dir: str = "market_data") -> Dict[str, pd.DataFrame]:
    """Load data for all requested pairs."""
    pair_data = {}
    for pair in pairs:
        df = load_pair_data(pair, data_dir)
        if df is not None and len(df) >= 400:
            pair_data[pair] = df
        elif df is not None:
            logger.warning(f"{pair}: Only {len(df)} candles (need 400+), skipping")
    return pair_data


# ---------------------------------------------------------------------------
# Backtest Scanner (subclass with CSV data injection)
# ---------------------------------------------------------------------------

class BacktestScanner(Scanner):
    """Scanner subclass that reads from pre-loaded DataFrames instead of OANDA."""

    def __init__(self, config: ScannerConfig):
        # Set oanda_client to a sentinel so _init_oanda_client returns False
        super().__init__(config=config, oanda_client=None)
        self._backtest_data: Dict[str, pd.DataFrame] = {}

    def _init_oanda_client(self) -> bool:
        """Never connect to OANDA in backtest mode."""
        return False

    def _fetch_pair_data(self, pair: str, count: int = 500) -> Optional[pd.DataFrame]:
        """Return pre-loaded CSV data instead of fetching from OANDA."""
        df = self._backtest_data.get(pair)
        if df is None:
            return None
        # Return the last `count` rows (mimics OANDA lookback)
        if len(df) > count:
            return df.iloc[-count:].copy()
        return df.copy()

    def set_pair_window(self, pair: str, df_window: pd.DataFrame) -> None:
        """Inject a data window for the next scan."""
        self._backtest_data[pair] = df_window

    def _resolve_model_source_pair(self, pair: str) -> str:
        """Always use joint models — pair-specific models have feature mismatch."""
        return "GENERIC"

    def _init_modular_ensemble(self) -> bool:
        """Override to load MEI without TF models (avoids OMP/SIGSEGV on macOS).

        The standard _init_modular_ensemble loads TF models which cause a fatal
        SIGSEGV from libomp conflicts (TF + sklearn + torch all load separate
        copies of libomp on macOS).  We skip TF entirely and rely on Ridge/XGB/RF.
        """
        if self._modular_ensemble is not None:
            self._sync_modular_ensemble_config()
            return self._ensemble_loaded

        try:
            from src.core.modular_inference import InferenceConfig, ModularEnsembleInference

            scan_inference_cfg = InferenceConfig(
                min_tcn_probability=float(self.config.min_tcn_probability),
                min_confidence=float(self.config.min_confidence),
                min_momentum=float(self.config.min_momentum),
                max_drawdown_pct=float(self.config.max_drawdown_pct),
                max_uncertainty_std=float(self.config.max_uncertainty_std),
                use_tcn_volatility_filter=False,
                min_volatility_regime=0,
                tcn_required=False,
                enforce_model_contracts=False,
                require_direction_model=False,
                require_volatility_model=False,
                require_regime_model=False,
                require_xgb_model=False,
                require_rf_model=False,
                require_ridge_model=False,
                sentiment_block_enabled=False,
                enable_meta_labeling=False,
                overlay_meta_enabled=False,
                final_score_threshold=float(self.config.final_score_threshold),
                mc_dropout_samples=0,
            )

            self._modular_ensemble = ModularEnsembleInference(
                model_dir=str(self.config.model_dir),
                config=scan_inference_cfg,
                use_rl_sizer=False,
                use_rl_gates=False,
                use_rl_exits=False,
                enable_market_intelligence=False,
            )
            self._modular_ensemble.load_models()

            # Disable TF-based models to avoid OMP SIGSEGV
            mei = self._modular_ensemble
            mei.tcn = None
            mei.regime_model = None
            mei.tcn_vol = None
            mei.use_hybrid = False
            mei.histgb = None
            mei._loaded_instrument = "GENERIC"
            def _safe_technical_fallback(df):
                """RSI + MACD fallback (pure pandas, no OMP-triggering ops)."""
                try:
                    close = df["close"]
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss.replace(0, 1e-8)
                    rsi = (100 - (100 / (1 + rs))).iloc[-1]
                    ema12 = close.ewm(span=12).mean()
                    ema26 = close.ewm(span=26).mean()
                    macd = (ema12 - ema26).iloc[-1]
                    macd_hist = (ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()).iloc[-1]
                    import math
                    if math.isnan(rsi): rsi = 50.0
                    if math.isnan(macd): macd = 0.0
                    if math.isnan(macd_hist): macd_hist = 0.0
                    if rsi < 30:
                        return "LONG", min(0.55 + (30 - rsi) / 100, 0.75)
                    elif rsi > 70:
                        return "SHORT", min(0.55 + (rsi - 70) / 100, 0.75)
                    if macd_hist > 0 and macd > 0:
                        return "LONG", 0.52
                    elif macd_hist < 0 and macd < 0:
                        return "SHORT", 0.52
                    return "HOLD", 0.5
                except Exception:
                    return "HOLD", 0.5
            mei._predict_technical_fallback = _safe_technical_fallback

            self._ensemble_loaded = (
                mei.xgb is not None or mei.ridge is not None or mei.rf is not None
            )
            if self._ensemble_loaded:
                self._sync_modular_ensemble_config()
                self._ensemble_type = "ModularEnsembleInference"
                self._assess_ensemble_health()
                logger.info("✓ Backtest MEI loaded (TF disabled, Ridge/XGB/RF only)")
            return self._ensemble_loaded

        except Exception as e:
            logger.warning(f"Backtest MEI init failed: {e}")
            self._ensemble_loaded = False
            return False


# ---------------------------------------------------------------------------
# Trade Simulator
# ---------------------------------------------------------------------------

class TradeSimulator:
    """Simulates trade execution, position management, and equity tracking."""

    # Transaction cost: spread + commission + slippage
    TRANSACTION_COST_PCT = 0.0004  # 0.04% per round-trip

    def __init__(
        self,
        starting_equity: float,
        mode: str = "aggressive",
        max_open_positions: int = 7,
    ):
        self.equity = starting_equity
        self.peak_equity = starting_equity
        self.mode = mode
        self.max_open_positions = max_open_positions

        self.open_positions: List[OpenPosition] = []
        self.closed_trades: List[ClosedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Risk parameters by mode
        if mode == "aggressive":
            self.risk_pct = 0.05  # 5% risk per trade
            self.max_risk_multiplier = 4.0  # High-confidence multiplier
            self.max_drawdown_halt = 0.50  # 50% DD before halting (aggressive)
        elif mode == "smart":
            self.risk_pct = 0.02  # 2% risk — smart profile is conservative
            self.max_risk_multiplier = 2.5  # Moderate confidence multiplier
            self.max_drawdown_halt = 0.20  # 20% DD for backtest (live smart uses 3% but $100 equity needs room)
        else:
            self.risk_pct = 0.02  # 2% risk per trade
            self.max_risk_multiplier = 2.5
            self.max_drawdown_halt = 0.12  # 12% DD

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def is_halted(self) -> bool:
        return self.drawdown_pct >= self.max_drawdown_halt

    def has_position(self, pair: str) -> bool:
        return any(p.pair == pair for p in self.open_positions)

    def check_sl_tp(self, pair: str, candle: pd.Series, timestamp: datetime) -> None:
        """Check if any open positions hit SL or TP on this candle."""
        remaining = []
        for pos in self.open_positions:
            if pos.pair != pair:
                remaining.append(pos)
                continue

            high = candle["high"]
            low = candle["low"]
            try:
                pip_value = get_registry().get(pair).pip_value
            except KeyError:
                pip_value = 0.0001
            hit_sl = False
            hit_tp = False

            if pos.direction == "LONG":
                hit_sl = low <= pos.sl_price
                hit_tp = high >= pos.tp_price
            else:  # SHORT
                hit_sl = high >= pos.sl_price
                hit_tp = low <= pos.tp_price

            if hit_sl and hit_tp:
                # Conservative: assume SL hit first
                self._close_position(pos, pos.sl_price, timestamp, "SL", pip_value)
            elif hit_sl:
                self._close_position(pos, pos.sl_price, timestamp, "SL", pip_value)
            elif hit_tp:
                self._close_position(pos, pos.tp_price, timestamp, "TP", pip_value)
            else:
                remaining.append(pos)

        self.open_positions = remaining

    def open_position(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        timestamp: datetime,
    ) -> bool:
        """Try to open a position. Returns True if opened."""
        # Guard: halted, max positions, duplicate pair
        if self.is_halted:
            return False
        if len(self.open_positions) >= self.max_open_positions:
            return False
        if self.has_position(pair):
            return False

        try:
            pip_value = get_registry().get(pair).pip_value
        except KeyError:
            pip_value = 0.0001

        # Position sizing: risk-based
        risk_amount = self.equity * self.risk_pct

        # Confidence multiplier (scale up for high-confidence trades)
        if confidence >= 0.75:
            risk_amount *= min(self.max_risk_multiplier, 1.0 + (confidence - 0.5) * 4.0)
        elif confidence >= 0.60:
            risk_amount *= 1.5

        # Cap at 30% of equity per position
        risk_amount = min(risk_amount, self.equity * 0.30)

        # Calculate units from risk and SL
        sl_value = sl_pips * pip_value
        if sl_value <= 0:
            return False
        units = int(risk_amount / sl_value)
        if units < 1:
            return False

        # Apply entry slippage (half spread)
        half_spread = pip_value * 1.0  # ~1 pip spread
        if direction == "LONG":
            actual_entry = entry_price + half_spread
            sl_price = actual_entry - sl_pips * pip_value
            tp_price = actual_entry + tp_pips * pip_value
        else:
            actual_entry = entry_price - half_spread
            sl_price = actual_entry + sl_pips * pip_value
            tp_price = actual_entry - tp_pips * pip_value

        pos = OpenPosition(
            pair=pair,
            direction=direction,
            entry_price=actual_entry,
            entry_time=timestamp,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=confidence,
            risk_amount=risk_amount,
        )
        self.open_positions.append(pos)
        return True

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        pip_value: float,
    ) -> None:
        """Close a position and update equity."""
        if pos.direction == "LONG":
            pnl_pips = (exit_price - pos.entry_price) / pip_value
        else:
            pnl_pips = (pos.entry_price - exit_price) / pip_value

        # Dollar P&L
        pnl_dollars = pnl_pips * pip_value * pos.units

        # Transaction cost
        notional = pos.entry_price * pos.units
        tx_cost = notional * self.TRANSACTION_COST_PCT
        pnl_dollars -= tx_cost

        self.equity += pnl_dollars
        self.peak_equity = max(self.peak_equity, self.equity)

        trade = ClosedTrade(
            pair=pos.pair,
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_time=pos.entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            units=pos.units,
            sl_pips=pos.sl_pips,
            tp_pips=pos.tp_pips,
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_dollars,
            exit_reason=reason,
            confidence=pos.confidence,
            transaction_cost=tx_cost,
        )
        self.closed_trades.append(trade)

    def close_all_at_market(self, pair_prices: Dict[str, float], timestamp: datetime) -> None:
        """Close all remaining positions at current market price."""
        for pos in list(self.open_positions):
            price = pair_prices.get(pos.pair, pos.entry_price)
            try:
                pip_value = get_registry().get(pos.pair).pip_value
            except KeyError:
                pip_value = 0.0001
            self._close_position(pos, price, timestamp, "END", pip_value)
        self.open_positions = []

    def record_equity(self, timestamp: datetime) -> None:
        self.equity_curve.append((timestamp, self.equity))


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(sim: TradeSimulator, starting_equity: float, mode: str) -> BacktestResult:
    """Compute summary statistics from the simulation."""
    trades = sim.closed_trades
    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]

    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0.001

    # Max drawdown from equity curve
    max_dd_pct = 0.0
    max_dd_amount = 0.0
    peak = starting_equity
    for _, eq in sim.equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd)
        max_dd_amount = max(max_dd_amount, peak - eq)

    # Sharpe ratio from equity curve returns
    sharpe = 0.0
    if len(sim.equity_curve) > 10:
        equities = [e for _, e in sim.equity_curve]
        returns = np.diff(equities) / np.array(equities[:-1])
        returns = returns[np.isfinite(returns)]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))

    # Per-pair breakdown
    pair_breakdown: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        if t.pair not in pair_breakdown:
            pair_breakdown[t.pair] = {"trades": 0, "wins": 0, "pnl": 0.0}
        pair_breakdown[t.pair]["trades"] += 1
        pair_breakdown[t.pair]["pnl"] += t.pnl_dollars
        if t.pnl_dollars > 0:
            pair_breakdown[t.pair]["wins"] += 1

    return BacktestResult(
        starting_equity=starting_equity,
        final_equity=sim.equity,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        total_pnl=total_pnl,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_amount=max_dd_amount,
        sharpe_ratio=sharpe,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0.0,
        avg_win_pips=float(np.mean([t.pnl_pips for t in wins])) if wins else 0.0,
        avg_loss_pips=float(np.mean([t.pnl_pips for t in losses])) if losses else 0.0,
        best_trade=max((t.pnl_dollars for t in trades), default=0.0),
        worst_trade=min((t.pnl_dollars for t in trades), default=0.0),
        equity_curve=sim.equity_curve,
        trades=trades,
        pair_breakdown=pair_breakdown,
    )


def print_report(result: BacktestResult, mode: str) -> None:
    """Print a formatted summary to console."""
    return_pct = ((result.final_equity - result.starting_equity) / result.starting_equity) * 100
    win_rate = (result.winning_trades / result.total_trades * 100) if result.total_trades > 0 else 0

    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULTS — {mode.upper()} MODE")
    print("=" * 70)
    print(f"  Starting Equity:    ${result.starting_equity:,.2f}")
    print(f"  Final Equity:       ${result.final_equity:,.2f}")
    print(f"  Total Return:       {return_pct:+,.2f}%")
    print(f"  Total P&L:          ${result.total_pnl:+,.2f}")
    print("-" * 70)
    print(f"  Total Trades:       {result.total_trades}")
    print(f"  Winning:            {result.winning_trades} ({win_rate:.1f}%)")
    print(f"  Losing:             {result.losing_trades}")
    print(f"  Profit Factor:      {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print("-" * 70)
    print(f"  Avg Win:            {result.avg_win_pips:+.1f} pips")
    print(f"  Avg Loss:           {result.avg_loss_pips:.1f} pips")
    print(f"  Best Trade:         ${result.best_trade:+,.2f}")
    print(f"  Worst Trade:        ${result.worst_trade:+,.2f}")
    print("-" * 70)
    print(f"  Max Drawdown:       {result.max_drawdown_pct*100:.1f}% (${result.max_drawdown_amount:,.2f})")
    print("-" * 70)

    target = 1_000_000
    if result.final_equity >= target:
        print(f"  >>> TARGET $1M REACHED! Final: ${result.final_equity:,.2f} <<<")
    else:
        if result.total_pnl > 0 and result.total_trades > 0:
            avg_return_per_trade = result.total_pnl / result.total_trades
            if avg_return_per_trade > 0:
                trades_needed = (target - result.final_equity) / avg_return_per_trade
                print(f"  $1M Target:         Need ~{int(trades_needed)} more trades at avg ${avg_return_per_trade:,.2f}/trade")
            else:
                print(f"  $1M Target:         Not achievable with current performance")
        else:
            print(f"  $1M Target:         Strategy not profitable")

    if result.pair_breakdown:
        print("\n  Per-Pair Breakdown:")
        for pair, stats in sorted(result.pair_breakdown.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            print(f"    {pair:12s}  {stats['trades']:3d} trades  {wr:5.1f}% WR  ${stats['pnl']:+10,.2f}")

    print("=" * 70 + "\n")


def save_results(result: BacktestResult, mode: str) -> None:
    """Save trade log and equity curve to files."""
    out_dir = PROJECT_ROOT / "trained_data" / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Trade log
    trade_log = []
    for t in result.trades:
        trade_log.append({
            "pair": t.pair,
            "direction": t.direction,
            "entry_price": round(t.entry_price, 5),
            "exit_price": round(t.exit_price, 5),
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "units": t.units,
            "sl_pips": round(t.sl_pips, 1),
            "tp_pips": round(t.tp_pips, 1),
            "pnl_pips": round(t.pnl_pips, 1),
            "pnl_dollars": round(t.pnl_dollars, 2),
            "exit_reason": t.exit_reason,
            "confidence": round(t.confidence, 3),
            "tx_cost": round(t.transaction_cost, 2),
        })

    log_path = out_dir / f"trades_{mode}_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(trade_log, f, indent=2)

    # Equity curve
    curve_path = out_dir / f"equity_{mode}_{ts}.csv"
    with open(curve_path, "w") as f:
        f.write("time,equity\n")
        for t, eq in result.equity_curve:
            f.write(f"{t.isoformat()},{eq:.2f}\n")

    # Summary
    summary = {
        "mode": mode,
        "starting_equity": result.starting_equity,
        "final_equity": round(result.final_equity, 2),
        "return_pct": round(((result.final_equity - result.starting_equity) / result.starting_equity) * 100, 2),
        "total_trades": result.total_trades,
        "win_rate": round(result.winning_trades / result.total_trades * 100, 1) if result.total_trades > 0 else 0,
        "profit_factor": round(result.profit_factor, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct * 100, 1),
        "reached_1m": bool(result.final_equity >= 1_000_000),
    }
    summary_path = out_dir / f"summary_{mode}_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved to {out_dir}/")
    print(f"    Trade log:    {log_path.name}")
    print(f"    Equity curve: {curve_path.name}")
    print(f"    Summary:      {summary_path.name}")


# ---------------------------------------------------------------------------
# Main Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest(
    pairs: List[str],
    starting_equity: float = 100.0,
    mode: str = "aggressive",
    scan_interval: int = 1,
    max_open: int = 7,
    verbose: bool = False,
) -> BacktestResult:
    """Run the walk-forward backtest."""

    print(f"\n{'='*70}")
    print(f"  BACKTEST SANDBOX — ${starting_equity:,.0f} → $1M Challenge")
    print(f"  Mode: {mode} | Pairs: {', '.join(pairs)} | Scan interval: {scan_interval}")
    print(f"{'='*70}\n")

    # --- Load data ---
    print("Loading CSV data...")
    pair_data = load_all_pairs(pairs)
    if not pair_data:
        print("ERROR: No valid pair data found!")
        sys.exit(1)

    available_pairs = list(pair_data.keys())
    print(f"Loaded {len(available_pairs)} pairs: {', '.join(available_pairs)}")

    # --- Build unified timeline ---
    all_times = set()
    for df in pair_data.values():
        all_times.update(df.index.tolist())
    timeline = sorted(all_times)
    print(f"Timeline: {len(timeline)} candles ({timeline[0]} → {timeline[-1]})")

    # --- Configure scanner for backtest ---
    config = ScannerConfig()

    # Profile overrides for backtest
    if mode == "smart":
        # Apply the full smart profile from config — all gates and agents enabled
        smart_profile = SCAN_PROFILES["smart"]
        for key, value in smart_profile.items():
            if hasattr(config, key):
                setattr(config, key, value)
        print("  Applied SMART profile — full autonomous pipeline with all gates")
    elif mode == "aggressive":
        config.min_confidence = 40.0
        config.min_momentum = 0.12
        config.min_tcn_probability = 0.55
        config.max_drawdown_pct = 0.05
        config.final_score_threshold = 0.40
        config.max_uncertainty_std = 0.20
        config.min_atr_pips = 3.0
        config.min_volatility_regime = 0
        config.min_agent_consensus_ratio = 0.25
        config.max_model_disagreement = 0.60
        config.max_uncertainty_score = 0.50
    else:
        config.min_confidence = 50.0
        config.min_momentum = 0.20
        config.max_drawdown_pct = 0.025

    # Backtest-specific overrides (critical — applied AFTER profile)
    config.blocked_pairs = []  # Don't block any pairs in backtest
    config.non_interactive = True
    config.incremental_enabled = False
    config.enable_execution = False  # We handle execution ourselves
    config.block_when_market_closed = False  # Process all candles

    # Disable features that need live infrastructure
    config.enable_session_filter = False
    config.enable_llm_trade_analysis = False
    config.enable_news_risk_agent = False  # No live news feed
    config.enable_live_position_management = False
    config.enable_smart_execution = False
    config.enable_execution_routing = False
    config.use_tcn_volatility_filter = False  # TCN has shape mismatch with current features
    config.sub_inference_enabled = False  # No sub-inference in backtest
    config.use_master_pair_models = False  # Use joint models only (pair-specific have feature mismatch)
    config.use_rl_sizer = False  # RL subprocesses cause OMP conflicts
    config.use_rl_gates = False
    config.use_rl_exits = False
    config.enable_meta_learning = False  # Torch-based ensemble weighter triggers OMP crash

    print("Initializing Scanner (may take a moment for model loading)...")
    scanner = BacktestScanner(config)

    # Force-clear blocked pairs AFTER construction (profiles may re-set it)
    scanner.config.blocked_pairs = []

    # Eagerly init MEI (our override disables TF models to avoid OMP SIGSEGV)
    scanner._init_modular_ensemble()
    if scanner._modular_ensemble is not None:
        print("  ✓ MEI loaded (TF disabled, Ridge/XGB/RF only)")
    else:
        print("  ⚠ MEI failed to load — will use technical fallback only")

    # --- Initialize simulator ---
    sim = TradeSimulator(
        starting_equity=starting_equity,
        mode=mode,
        max_open_positions=max_open,
    )

    # --- Warmup period ---
    WARMUP = 400
    if len(timeline) <= WARMUP:
        print(f"ERROR: Only {len(timeline)} candles, need {WARMUP}+ for warmup")
        sys.exit(1)

    tradeable_timeline = timeline[WARMUP:]
    total_steps = len(tradeable_timeline)
    print(f"\nStarting walk-forward ({total_steps} steps, warmup={WARMUP})...")
    print(f"Starting equity: ${starting_equity:,.2f}")
    print("-" * 70)

    t_start = time.time()
    signals_generated = 0
    positions_opened = 0
    scan_errors = 0

    for step_idx, current_time in enumerate(tradeable_timeline):
        # --- 1. Check SL/TP on open positions ---
        for pair in available_pairs:
            if pair not in pair_data:
                continue
            df = pair_data[pair]
            if current_time in df.index:
                candle = df.loc[current_time]
                sim.check_sl_tp(pair, candle, current_time)

        # --- 2. Scan for new signals (every scan_interval candles) ---
        if step_idx % scan_interval == 0 and not sim.is_halted:
            for pair in available_pairs:
                if sim.has_position(pair):
                    continue
                if len(sim.open_positions) >= sim.max_open_positions:
                    break

                df = pair_data[pair]
                # Find the position of current_time in this pair's data
                if current_time not in df.index:
                    continue

                idx = df.index.get_loc(current_time)
                if idx < WARMUP:
                    continue

                # Extract lookback window
                window = df.iloc[max(0, idx - WARMUP + 1):idx + 1].copy()
                if len(window) < config.min_candles:
                    continue

                # Inject data and scan
                scanner.set_pair_window(pair, window)
                try:
                    analysis = scanner._scan_pair(pair)
                except Exception as e:
                    scan_errors += 1
                    if verbose:
                        logger.debug(f"Scan error {pair} at {current_time}: {e}")
                    continue

                if analysis is None or analysis.error:
                    continue

                signals_generated += 1

                # Check if tradeable
                tradeable = False
                if mode == "aggressive":
                    # Relaxed: direction + confidence above threshold
                    tradeable = (
                        analysis.direction in ("LONG", "SHORT")
                        and analysis.confidence >= 0.40
                        and analysis.sl_pips > 0
                        and analysis.tp_pips > 0
                    )
                else:
                    # Smart and realistic: use full pipeline is_tradeable check
                    tradeable = analysis.is_tradeable

                if tradeable:
                    entry_price = window["close"].iloc[-1]

                    # Fallback ATR-based SL/TP if scanner returned defaults
                    sl_pips = analysis.sl_pips
                    tp_pips = analysis.tp_pips
                    if sl_pips <= 15.0 and tp_pips <= 20.0:
                        # Compute ATR ourselves from raw data
                        try:
                            pip_val = get_registry().get(pair).pip_value
                        except KeyError:
                            pip_val = 0.0001
                        if len(window) >= 14:
                            tr = pd.concat([
                                window["high"] - window["low"],
                                (window["high"] - window["close"].shift(1)).abs(),
                                (window["low"] - window["close"].shift(1)).abs(),
                            ], axis=1).max(axis=1)
                            atr_raw = tr.rolling(14).mean().iloc[-1]
                            atr_pips_calc = atr_raw / pip_val if pip_val > 0 else 0
                            if atr_pips_calc > 3:
                                sl_pips = max(8.0, min(atr_pips_calc * 1.0, 40.0))
                                tp_pips = max(12.0, min(atr_pips_calc * 1.8, 70.0))

                    opened = sim.open_position(
                        pair=pair,
                        direction=analysis.direction,
                        entry_price=entry_price,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        confidence=analysis.confidence,
                        timestamp=current_time,
                    )
                    if opened:
                        positions_opened += 1
                        if verbose:
                            print(
                                f"  [{current_time}] OPEN {analysis.direction} {pair} "
                                f"@ {entry_price:.5f} SL={analysis.sl_pips:.0f} TP={analysis.tp_pips:.0f} "
                                f"conf={analysis.confidence:.2f} eq=${sim.equity:,.2f}"
                            )

        # --- 3. Record equity ---
        sim.record_equity(current_time)

        # --- 4. Progress reporting ---
        if (step_idx + 1) % 500 == 0 or step_idx == total_steps - 1:
            elapsed = time.time() - t_start
            pct = (step_idx + 1) / total_steps * 100
            trades_closed = len(sim.closed_trades)
            open_count = len(sim.open_positions)
            print(
                f"  Step {step_idx+1:>6}/{total_steps} ({pct:5.1f}%) | "
                f"Equity: ${sim.equity:>12,.2f} | "
                f"Open: {open_count} | Closed: {trades_closed} | "
                f"DD: {sim.drawdown_pct*100:5.1f}% | "
                f"Time: {elapsed:.0f}s"
            )

        # Early termination if equity hits $1M
        if sim.equity >= 1_000_000:
            print(f"\n  >>> $1M TARGET REACHED at step {step_idx+1}! <<<\n")
            break

        # Early termination if account blown (equity <= 0)
        if sim.equity <= 0:
            print(f"\n  >>> ACCOUNT BLOWN at step {step_idx+1} <<<\n")
            break

    # --- Close remaining positions ---
    if sim.open_positions:
        last_prices = {}
        for pair in available_pairs:
            if pair in pair_data:
                last_prices[pair] = pair_data[pair]["close"].iloc[-1]
        sim.close_all_at_market(last_prices, tradeable_timeline[-1])

    elapsed_total = time.time() - t_start
    print(f"\nBacktest complete in {elapsed_total:.1f}s")
    print(f"  Signals generated: {signals_generated}")
    print(f"  Positions opened:  {positions_opened}")
    print(f"  Scan errors:       {scan_errors}")

    # --- Generate report ---
    result = generate_report(sim, starting_equity, mode)
    print_report(result, mode)
    save_results(result, mode)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Sandbox — Walk-forward backtester for Buddy FX Engine"
    )
    parser.add_argument(
        "--mode", choices=["realistic", "aggressive", "smart"], default="aggressive",
        help="Trading mode (default: aggressive). 'smart' uses full autonomous pipeline."
    )
    parser.add_argument(
        "--equity", type=float, default=100.0,
        help="Starting equity in USD (default: 100)"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Pairs to trade (default: all available in market_data/)"
    )
    parser.add_argument(
        "--scan-interval", type=int, default=24,
        help="Scan every N candles (default: 24 = daily on H1, use 12 for 2x/day)"
    )
    parser.add_argument(
        "--max-open", type=int, default=7,
        help="Max simultaneous open positions (default: 7)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print individual trade entries"
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        help="Logging level (default: WARNING)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine pairs
    if args.pairs:
        pairs = args.pairs
    else:
        # Auto-detect from CSV files
        csv_pattern = os.path.join(PROJECT_ROOT, "market_data", "oanda_*_H1_*.csv")
        csv_files = glob.glob(csv_pattern)
        detected = set()
        for f in csv_files:
            basename = os.path.basename(f)
            # Pattern: oanda_{PAIR}_H1_live_{timestamp}.csv
            parts = basename.replace("oanda_", "").split("_H1_")
            if parts:
                detected.add(parts[0])
        pairs = sorted(detected) if detected else MAJOR_PAIRS
        print(f"Auto-detected pairs: {', '.join(pairs)}")

    run_backtest(
        pairs=pairs,
        starting_equity=args.equity,
        mode=args.mode,
        scan_interval=args.scan_interval,
        max_open=args.max_open,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
