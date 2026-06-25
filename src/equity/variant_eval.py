"""Cross-sectional variant bake-off harness — offline evaluation only (running:NO).

Each variant REPLACES the equal-weight base book with a different long-only
cross-sectional construction, then the SAME vol-target + drawdown overlay applies
(the overlay scalar is derived from the EW-diversified return series in
`ship_gate._derive_returns`, identical across arms — so the ONLY difference
between arms is the cross-sectional weighting, not the overlay or the universe
handling). `evaluate_book` with the EW panel reproduces `ship_gate.evaluate_harvester`.

SEPARATE module: the baseline (`ship_gate`/`backtest`) is NOT modified. OFFLINE
backtest, NOT a live loop — nothing here runs in a process, places an order, or
unhalts. NON-hot-path. Promotion/unhalt on a win is a separate operator decision.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from src.equity.backtest import overlay as baseline_overlay
from src.equity.backtest import run_portfolio_backtest
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

BAKEOFF_PATH_DEFAULT = "trained_data/backtests/variant_bakeoff.json"


def aligned_inputs(snapshot: "UniverseSnapshot", prices: pd.DataFrame):
    """Shared, membership-respecting alignment for every variant constructor.

    Returns ``(ew, active, asset_returns)`` all on the EW panel's normalized
    index/columns: ``ew`` = baseline equal-weight panel (membership + NaN-price
    discipline), ``active`` = boolean ew>0 mask, ``asset_returns`` = per-ticker
    pct_change aligned to the same grid. Variants tilt within the active set and
    renormalize, so they inherit the baseline's universe handling exactly.
    """
    ew = build_equal_weight_panel(snapshot, prices)
    px = prices.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    else:
        px.index = px.index.tz_convert("UTC")
    px.index = px.index.normalize()
    px = px.reindex(index=ew.index, columns=ew.columns)
    asset_returns = px.pct_change()
    active = ew > 0
    return ew, active, asset_returns


def evaluate_book(
    snapshot: "UniverseSnapshot",
    prices: pd.DataFrame,
    base_weights: pd.DataFrame,
    *,
    adv: Optional[pd.DataFrame] = None,
    target_vol: float = DEFAULT_TARGET_VOL,
    dd_soft: float = DEFAULT_DD_SOFT,
    dd_hard: float = DEFAULT_DD_HARD,
    max_lev: float = DEFAULT_MAX_LEV,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps_per_pct_adv: float = DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
) -> Dict[str, object]:
    """Run the harvester overlay+backtest+gate on a given long-only base book.

    ``base_weights`` is a date×ticker panel (0 for inactive, rows sum to 1 over
    active members). The overlay scalar is the EW-diversified one (same as the
    baseline) so only the cross-sectional construction differs.
    """
    base_returns = _derive_returns(prices.reindex(index=base_weights.index))
    scalar = baseline_overlay(
        base_returns.fillna(0.0),
        target_vol=target_vol,
        dd_soft=dd_soft,
        dd_hard=dd_hard,
        max_lev=max_lev,
    ).reindex(base_weights.index).fillna(0.0)
    managed = base_weights.mul(scalar, axis=0)
    if (managed < -1e-9).any().any():
        raise ValueError("base_weights contain negatives — long-only violated")

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
        "net_sharpe": round(net_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "positive_years": int(positive_years),
        "total_years": int(total_years),
        "gate_pass": bool(verdict.passed),
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


def run_bakeoff(
    *,
    snapshot_path: Path = Path("market_data/equity/universe_snapshot_pit.json"),
    out_path: Path = Path(BAKEOFF_PATH_DEFAULT),
    oos_fraction: float = 0.34,
    quality_panel: Optional[pd.DataFrame] = None,
    generated_at: str = "",
) -> Dict[str, object]:
    """OFFLINE 4-arm bake-off: baseline EW vs low-vol vs HRP (vs quality if a real
    PIT quality panel is supplied) on the ship gate, full-sample AND OOS tail.

    running:NO — backtest only. quality_panel is REQUIRED to evaluate the quality
    arm; if None, the quality arm is reported as not-evaluated (no fabrication).
    """
    import yfinance as yf  # lazy: offline fetch only

    from src.equity.hrp import hrp_weights
    from src.equity.lowvol import lowvol_weights
    from src.equity.quality import quality_tilt_weights
    from src.equity.universe import UniverseSnapshot

    snap = UniverseSnapshot.load(Path(snapshot_path))
    members = list(snap.membership.columns)
    idx = snap.membership.index
    start = idx.min().date().isoformat()
    end = (idx.max() + pd.Timedelta(days=1)).date().isoformat()
    raw = yf.download(
        members, start=start, end=end, progress=False, auto_adjust=True, threads=True
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=members)
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    prices_full = close.reindex(index=idx).ffill()

    def _arms(snp, prc, qpanel):
        ew = build_equal_weight_panel(snp, prc)
        arms = {
            "baseline_ew": evaluate_book(snp, prc, ew),
            "lowvol": evaluate_book(snp, prc, lowvol_weights(snp, prc)),
            "hrp": evaluate_book(snp, prc, hrp_weights(snp, prc)),
        }
        if qpanel is not None:
            arms["quality"] = evaluate_book(
                snp, prc, quality_tilt_weights(snp, prc, qpanel)
            )
        else:
            arms["quality"] = {
                "status": "NOT_EVALUATED",
                "reason": "no point-in-time fundamental data available for the "
                "universe (financial-datasets returns latest-FY snapshot only; "
                "yfinance is current-snapshot only). A current ROE applied "
                "historically would be lookahead + survivorship (L-018). Supply "
                "a real PIT quality panel to evaluate this arm.",
            }
        return arms

    full = _arms(snap, prices_full, quality_panel)
    split = int(len(idx) * (1.0 - float(oos_fraction)))
    oos_idx = idx[split:]
    oos_snap = UniverseSnapshot(
        membership=snap.membership.loc[oos_idx],
        rules=snap.rules,
        built_at=snap.built_at,
        pipeline_version=snap.pipeline_version,
        universe_hash=snap.universe_hash,
    )
    q_oos = quality_panel.loc[oos_idx] if quality_panel is not None else None
    oos = _arms(oos_snap, prices_full.loc[oos_idx], q_oos)

    def _rank(arms):
        base = float(arms["baseline_ew"]["net_sharpe"])
        out = {}
        for name, a in arms.items():
            if "net_sharpe" not in a:
                continue
            out[name] = {
                "net_sharpe": a["net_sharpe"],
                "max_dd": a["max_dd"],
                "gate_pass": a["gate_pass"],
                "beats_baseline": float(a["net_sharpe"]) > base and name != "baseline_ew",
                "margin_vs_baseline": round(float(a["net_sharpe"]) - base, 4),
            }
        return out

    payload = {
        "generated_at": generated_at,
        "running": "NO (offline backtest evaluation)",
        "universe_hash": snap.universe_hash,
        "members": len(members),
        "full_sample": {"start": start, "end": end, "arms": full, "ranking": _rank(full)},
        "out_of_sample": {
            "start": oos_idx.min().date().isoformat(),
            "end": oos_idx.max().date().isoformat(),
            "fraction": float(oos_fraction),
            "arms": oos,
            "ranking": _rank(oos),
        },
    }
    _atomic_write_json(payload, Path(out_path))
    return payload
