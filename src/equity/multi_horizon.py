"""Multi-horizon harvester VARIANT — offline evaluation only (running:NO).

The baseline harvester (`ship_gate.evaluate_harvester` → `backtest.overlay`)
vol-targets off a SINGLE 21-day realized-vol window. This variant blends the
vol-target scalar across genuinely different lookbacks (1M/3M/6M/12M =
21/63/126/252 trading days), averaging only the horizons that have enough
history at each date — strictly causal (every per-horizon vol is `shift(1)`,
no lookahead).

It is a SEPARATE variant: it does NOT modify or regress the baseline. The
baseline arm is reproduced exactly by `horizons=(21,)` (proven in tests), so the
side-by-side ship-gate comparison differs only in the overlay.

This module is OFFLINE EVALUATION, NOT a live loop: nothing here runs in a
process, places an order, or unhalts. `run_comparison` fetches historical prices
(yfinance) and writes a result JSON for the operator + verifier to read. It earns
no promotion — the ship gate + operator decide that. NON-hot-path: imports
nothing from `src.scanner.execution`/`main`/`embedded_scanner`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.equity.backtest import ANN, run_portfolio_backtest
from src.equity.ship_gate import (
    DEFAULT_COST_BPS,
    DEFAULT_DD_HARD,
    DEFAULT_DD_SOFT,
    DEFAULT_EXECUTION_LAG,
    DEFAULT_MAX_LEV,
    DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV,
    DEFAULT_TARGET_VOL,
    _derive_returns,
    build_equal_weight_panel,
)
from src.factor.ship_gate import evaluate_gate

if TYPE_CHECKING:  # pragma: no cover
    from src.equity.universe import UniverseSnapshot

logger = logging.getLogger(__name__)

BASELINE_HORIZONS = (21,)
VARIANT_HORIZONS = (21, 63, 126, 252)  # 1M / 3M / 6M / 12M — canonical, untuned
COMPARE_PATH_DEFAULT = "trained_data/backtests/multi_horizon_compare.json"


def overlay_multi_horizon(
    base_ret: pd.Series,
    *,
    target_vol: float,
    dd_soft: float,
    dd_hard: float,
    max_lev: float = 1.0,
    horizons: Sequence[int] = VARIANT_HORIZONS,
) -> pd.Series:
    """Causal multi-horizon vol-target + drawdown circuit-breaker scalar.

    Per horizon h: ``s_h = clip(target_vol / (rolling(h).std()*sqrt(ANN)).shift(1),
    upper=max_lev)``. Blend = row-wise mean over horizons that have enough history
    (NaN-skipping; all-NaN warmup rows -> 0). Then × the same shift(1) drawdown
    breaker as the baseline. With ``horizons=(21,)`` this equals `backtest.overlay`.
    """
    hs = tuple(int(h) for h in horizons)
    if not hs:
        raise ValueError("horizons must be non-empty")
    cols = []
    for h in hs:
        rvol = base_ret.rolling(h).std().mul(np.sqrt(ANN)).shift(1)
        cols.append((target_vol / rvol).clip(upper=max_lev))  # NaN where no history
    s_vol = pd.concat(cols, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    eq = (1 + base_ret).cumprod()
    dd = (1 - eq / eq.cummax()).shift(1).fillna(0.0)
    s_dd = np.where(
        dd <= dd_soft,
        1.0,
        np.where(dd >= dd_hard, 0.0, (dd_hard - dd) / (dd_hard - dd_soft)),
    )
    return (s_vol * pd.Series(s_dd, index=base_ret.index)).clip(0.0, max_lev)


def evaluate_multi_horizon(
    snapshot: "UniverseSnapshot",
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = VARIANT_HORIZONS,
    adv: Optional[pd.DataFrame] = None,
    target_vol: float = DEFAULT_TARGET_VOL,
    dd_soft: float = DEFAULT_DD_SOFT,
    dd_hard: float = DEFAULT_DD_HARD,
    max_lev: float = DEFAULT_MAX_LEV,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps_per_pct_adv: float = DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
) -> Dict[str, object]:
    """Run the harvester with the multi-horizon overlay and score the ship gate.

    Mirrors `ship_gate.evaluate_harvester` exactly except for the overlay, so
    ``horizons=(21,)`` reproduces the baseline. Returns the gate metrics dict.
    """
    base_weights = build_equal_weight_panel(snapshot, prices)
    base_returns = _derive_returns(prices.reindex(index=base_weights.index))
    scalar = overlay_multi_horizon(
        base_returns.fillna(0.0),
        target_vol=target_vol,
        dd_soft=dd_soft,
        dd_hard=dd_hard,
        max_lev=max_lev,
        horizons=horizons,
    ).reindex(base_weights.index).fillna(0.0)
    managed = base_weights.mul(scalar, axis=0)

    eff_slip = float(slippage_bps_per_pct_adv)
    if adv is None and eff_slip > 0.0:
        eff_slip = 0.0
    result = run_portfolio_backtest(
        managed,
        prices.reindex(index=base_weights.index),
        cost_bps=cost_bps,
        slippage_bps_per_pct_adv=eff_slip,
        adv=adv,
        execution_lag=execution_lag,
    )
    s = result.summary
    net_sharpe = float(s.get("net_sharpe", 0.0))
    max_dd = float(s.get("max_dd", 1.0))
    per_year = s.get("per_year", {}) or {}
    positive_years = sum(1 for v in per_year.values() if float(v) > 0)
    total_years = len(per_year)
    verdict = evaluate_gate(
        {
            "net_sharpe": net_sharpe,
            "positive_years": positive_years,
            "total_years": total_years,
            "max_drawdown": max_dd,
            "walk_forward": True,
        }
    )
    return {
        "horizons": list(int(h) for h in horizons),
        "net_sharpe": round(net_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "positive_years": int(positive_years),
        "total_years": int(total_years),
        "gate_pass": bool(verdict.passed),
    }


def compare_windows(
    snapshot: "UniverseSnapshot",
    prices: pd.DataFrame,
    *,
    variant_horizons: Sequence[int] = VARIANT_HORIZONS,
    adv: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Side-by-side: baseline (single 21d) vs variant (multi-horizon) on the same
    snapshot+prices — only the overlay differs."""
    baseline = evaluate_multi_horizon(
        snapshot, prices, horizons=BASELINE_HORIZONS, adv=adv
    )
    variant = evaluate_multi_horizon(
        snapshot, prices, horizons=variant_horizons, adv=adv
    )
    return {
        "baseline": baseline,
        "variant": variant,
        "variant_beats_baseline_net_sharpe": (
            float(variant["net_sharpe"]) > float(baseline["net_sharpe"])
        ),
        "net_sharpe_margin": round(
            float(variant["net_sharpe"]) - float(baseline["net_sharpe"]), 4
        ),
    }


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_comparison(
    *,
    snapshot_path: Path = Path("market_data/equity/universe_snapshot_pit.json"),
    out_path: Path = Path(COMPARE_PATH_DEFAULT),
    oos_fraction: float = 0.34,
    variant_horizons: Sequence[int] = VARIANT_HORIZONS,
    generated_at: str = "",
) -> Dict[str, object]:
    """OFFLINE eval: fetch member prices, compare baseline vs variant on the full
    history AND a held-out out-of-sample tail; write + return the result JSON.

    running:NO — this is a backtest, not a live loop. No promotion, no unhalt.
    """
    import yfinance as yf  # lazy: only needed for the offline fetch

    from src.equity.universe import UniverseSnapshot

    snap = UniverseSnapshot.load(Path(snapshot_path))
    members = list(snap.membership.columns)
    idx = snap.membership.index
    start = idx.min().date().isoformat()
    end = (idx.max() + pd.Timedelta(days=1)).date().isoformat()
    logger.info("multi_horizon offline fetch: %d members %s..%s", len(members), start, end)
    raw = yf.download(
        members, start=start, end=end, progress=False, auto_adjust=True, threads=True
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=members)
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    prices = close.reindex(index=idx).ffill()

    full = compare_windows(snap, prices, variant_horizons=variant_horizons)

    split = int(len(idx) * (1.0 - float(oos_fraction)))
    oos_idx = idx[split:]
    oos_snap = UniverseSnapshot(
        membership=snap.membership.loc[oos_idx],
        rules=snap.rules,
        built_at=snap.built_at,
        pipeline_version=snap.pipeline_version,
        universe_hash=snap.universe_hash,
    )
    oos = compare_windows(oos_snap, prices.loc[oos_idx], variant_horizons=variant_horizons)

    payload = {
        "generated_at": generated_at,
        "running": "NO (offline backtest evaluation)",
        "universe_hash": snap.universe_hash,
        "members": len(members),
        "variant_horizons": list(int(h) for h in variant_horizons),
        "baseline_horizons": list(BASELINE_HORIZONS),
        "full_sample": {"start": start, "end": end, **full},
        "out_of_sample": {
            "start": oos_idx.min().date().isoformat(),
            "end": oos_idx.max().date().isoformat(),
            "fraction": float(oos_fraction),
            **oos,
        },
    }
    _atomic_write_json(payload, Path(out_path))
    return payload
