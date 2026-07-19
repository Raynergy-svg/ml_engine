"""37-asset multi-asset TREND shadow lane — strategy-owned forward ledger.

2026-07-18 readiness report, execution-order step 2: the 37-asset diversified
trend book is "the strongest existing risk-control construction for forward
testing: good OOS return shape (OOS Sharpe 0.829/0.788 across rounds 3-4),
low drawdown (~16-18%), causal verification, and a significance miss
consistent with limited holdout power. Keep its parameters frozen." This
module gives it what it lacked: a dedicated strategy-owned forward ledger so
genuine out-of-sample observations accumulate under the FROZEN construction.
Target per the acceptance rules: 24-30 completed monthly rebalances before a
serious read; interim results are operational, never "statistically proven".

FROZEN construction (verbatim reuse — nothing re-derived or tuned here):
  * Universe: the Round-3 37-ticker cross-asset set, imported from the
    pre-registered ``scripts/experiment_edge_round3_leadB.py`` (never
    retyped). SHA-256 of the sorted universe is stamped into every ledger
    row so a silent universe change is detectable.
  * Signal/holding: ``src.equity.multi_asset_trend.single_asset_trend_returns``
    — price > 200d SMA, long-or-flat, shift(1) causal, monthly (step=21),
    2 bps/side. ONE rule, uniform, no per-asset tuning.
  * Combiner: ``sleeve_combiner.combine_sleeves`` (HRP across sleeves,
    causal, re-estimated every 21d from a trailing 252d window).
  * Overlay: ``backtest.overlay`` at the PRE-REGISTERED target_vol=0.10
    (round-3 doc section 2 — explicitly NOT the code default 0.12), dd
    circuit-breaker soft 0.10 / hard 0.20, max_lev 3.0.
  * ``tests/test_strategy_lanes_2026_07_18.py`` regression-locks this
    module's net series bit-for-bit against ``combined_portfolio`` so the
    lane can never silently drift from the pre-registered construction.

SHADOW ONLY: zero orders, zero broker imports. The book this lane logs is a
paper book. Data source is the free yfinance daily cache
(``market_data/multi_asset/panel_round3.parquet``); ``load_panel(refresh=
True)`` re-downloads when yfinance is installed, otherwise the cached panel
is used as-is and staleness shows up honestly as an unchanged ``asof_date``
(duplicate-asof rows are refused, same guard as the crypto lanes).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.equity.backtest import overlay
from src.equity.multi_asset_trend import (
    COST_BPS_PER_SIDE,
    SMA_WINDOW,
    STEP,
    combined_portfolio,
    trend_streams,
)
from src.equity.ship_gate import DEFAULT_DD_HARD, DEFAULT_DD_SOFT
from src.equity.sleeve_combiner import combine_sleeves

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Frozen pre-registered spec — imported from the round-3 experiment script,
# never retyped here. TARGET_VOL is the round-3 doc's explicit 0.10.
import experiment_edge_round3_leadB as _r3  # noqa: E402

UNIVERSE: List[str] = list(_r3.UNIVERSE)
TARGET_VOL: float = float(_r3.TARGET_VOL)          # 0.10 — pre-registered, NOT the code default 0.12
PANEL_CACHE: Path = Path(_r3.CACHE)
MAX_LEV: float = 3.0                               # frozen in the round-3 construction (combined_portfolio)
MIN_BARS: int = 250                                # round-3 eligibility floor

SOURCE_DOCS = ("docs/experiment-multi-asset-trend-2026-06-29.md",
               "docs/experiment-edge-hunt-round3-2026-06-29.md",
               "docs/experiment-edge-hunt-round4-2026-06-30.md")
GATE_VERDICT = ("absolute-gate PASS (full Sharpe 0.744, maxDD 18%; OOS 0.829); "
                "significance near-miss is power-limited — forward-test the spec "
                "UNCHANGED (readiness report 2026-07-18)")

LEDGER_PATH_DEFAULT = REPO_ROOT / "trained_data" / "trend" / "shadow_multi_asset_trend_ledger.jsonl"


def universe_sha256(universe: List[str] = UNIVERSE) -> str:
    return hashlib.sha256(",".join(sorted(universe)).encode("utf-8")).hexdigest()


def construction_manifest() -> Dict[str, Any]:
    """The frozen construction, exactly as reused — stamped into every row."""
    return {
        "signal": f"price_gt_sma{SMA_WINDOW}_long_or_flat",
        "step_days": STEP,
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "combiner": "hrp_across_sleeves_252d_lookback_21d_step",
        "vol_target_ann": TARGET_VOL,
        "dd_soft": DEFAULT_DD_SOFT,
        "dd_hard": DEFAULT_DD_HARD,
        "max_leverage": MAX_LEV,
        "universe_size": len(UNIVERSE),
        "universe_sha256": universe_sha256(),
        "min_bars": MIN_BARS,
        "source_docs": list(SOURCE_DOCS),
        "gate_verdict": GATE_VERDICT,
    }


def load_panel(refresh: bool = False) -> pd.DataFrame:
    """The frozen-universe daily close panel. Cached parquet by default;
    ``refresh=True`` re-downloads via yfinance when available. Missing cache
    AND no yfinance -> explicit error (never a fabricated panel)."""
    if not refresh and PANEL_CACHE.exists():
        return pd.read_parquet(PANEL_CACHE)
    try:
        import yfinance as yf
    except ImportError as exc:
        if PANEL_CACHE.exists():
            logger.warning("multi_asset_trend_lane: yfinance unavailable (%s) — using cached panel", exc)
            return pd.read_parquet(PANEL_CACHE)
        raise RuntimeError(
            "multi_asset_trend_lane: no cached panel at "
            f"{PANEL_CACHE} and yfinance is not installed — cannot compute a cycle"
        ) from exc
    raw = yf.download(UNIVERSE, start="1990-01-01", progress=False,
                      auto_adjust=True, group_by="ticker", threads=True)
    close = pd.concat({t: raw[t]["Close"] for t in UNIVERSE if t in raw}, axis=1).sort_index()
    close = close.dropna(how="all")
    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(PANEL_CACHE)
    return close


# --------------------------------------------------------------------------- #
# Cycle computation — exposes the book the frozen construction implies today.
# The net series is regression-locked bit-for-bit against combined_portfolio.
# --------------------------------------------------------------------------- #
def _held_state(price: pd.Series) -> Optional[pd.Series]:
    """The per-asset held weight series (0/1, monthly-held) EXACTLY as
    ``single_asset_trend_returns`` computes it internally (that function
    returns only the net-return stream; the lane also needs the state).
    Regression-locked against it in the lane tests."""
    px = pd.Series(price).dropna().astype(float)
    if len(px) < SMA_WINDOW + STEP:
        return None
    sma = px.rolling(SMA_WINDOW, min_periods=max(20, SMA_WINDOW // 2)).mean().shift(1)
    on = (px.shift(1) > sma).fillna(False)
    w = pd.Series(0.0, index=px.index)
    last = 0.0
    for i in range(len(px)):
        if i % STEP == 0:
            last = 1.0 if bool(on.iloc[i]) else 0.0
        w.iloc[i] = last
    return w


@dataclass
class LaneCycleResult:
    asof_date: str
    universe_size: int          # sleeves with enough history to trade
    longs: Dict[str, float]     # asset -> portfolio weight (hrp x held x overlay lev)
    gross_leverage: float
    overlay_leverage: float
    today_net_return: float
    cumulative_note: str = "net stream embeds 2 bps/side per-sleeve costs (frozen)"
    construction: Dict[str, Any] = field(default_factory=construction_manifest)


def compute_lane_cycle(prices: Optional[pd.DataFrame] = None, *,
                       refresh: bool = False) -> LaneCycleResult:
    """Compute the frozen 37-asset trend book + today's mark.

    Book weight per asset at the latest bar = HRP sleeve weight (causal panel)
    x held state (0/1, monthly schedule) x overlay exposure scalar. The net
    return logged is the SAME number ``combined_portfolio`` produces for the
    latest bar (regression-locked)."""
    if prices is None:
        prices = load_panel(refresh=refresh)
    if prices is None or prices.empty:
        raise RuntimeError("multi_asset_trend_lane: empty price panel")

    streams = trend_streams(prices)
    if streams.empty or streams.shape[1] < 2:
        raise RuntimeError("multi_asset_trend_lane: fewer than 2 tradeable sleeves — no portfolio")
    combined, weight_panel = combine_sleeves(streams)
    combined = combined.dropna()
    if combined.empty:
        raise RuntimeError("multi_asset_trend_lane: combined stream empty")
    scalar = overlay(combined.fillna(0.0), target_vol=TARGET_VOL,
                     dd_soft=DEFAULT_DD_SOFT, dd_hard=DEFAULT_DD_HARD, max_lev=MAX_LEV)
    net = (combined * scalar.reindex(combined.index).fillna(0.0)).dropna()
    if net.empty:
        raise RuntimeError("multi_asset_trend_lane: net stream empty")

    asof = net.index[-1]
    lev = float(scalar.reindex(net.index).fillna(0.0).loc[asof])
    hrp_row = weight_panel.reindex(index=[asof]).iloc[0].fillna(0.0)
    book: Dict[str, float] = {}
    for asset in streams.columns:
        held = _held_state(prices[asset])
        if held is None or asof not in held.index:
            continue
        w = float(hrp_row.get(asset, 0.0)) * float(held.loc[asof]) * lev
        if abs(w) > 1e-12:
            book[str(asset)] = w

    return LaneCycleResult(
        asof_date=str(pd.Timestamp(asof).date()),
        universe_size=int(streams.shape[1]),
        longs=book,   # long-or-flat construction: no shorts by design
        gross_leverage=float(sum(abs(v) for v in book.values())),
        overlay_leverage=lev,
        today_net_return=float(net.loc[asof]),
    )


# --------------------------------------------------------------------------- #
# Ledger — append-only JSONL, flush+fsync (same convention as the crypto
# shadow lanes / equity live_gate audit log).
# --------------------------------------------------------------------------- #
def _read_ledger_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("multi_asset_trend ledger unreadable at %s: %s", path, exc)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping corrupt ledger line in %s", path)
    return rows


def _append_ledger_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        logger.error("multi_asset_trend ledger append failed for %s: %s", path, exc, exc_info=True)
        raise


def record_lane_cycle(result: LaneCycleResult, *,
                      ledger_path: Path = LEDGER_PATH_DEFAULT,
                      cycle_ts_iso: str) -> Optional[Dict[str, Any]]:
    """Append one forward cycle row. Duplicate ``asof_date`` -> honest no-op
    (stale cache, not a bug). ``cumulative_shadow_return`` compounds ONLY the
    rows in this ledger — true forward-from-activation P&L, never backtest
    history."""
    rows = _read_ledger_rows(ledger_path)
    if rows and rows[-1].get("asof_date") == result.asof_date:
        logger.info("multi_asset_trend shadow: no new trading day (asof_date=%s) — skipping",
                    result.asof_date)
        return None
    prior_cum = float(rows[-1]["cumulative_shadow_return"]) if rows else 0.0
    cum = (1.0 + prior_cum) * (1.0 + result.today_net_return) - 1.0
    row = {
        "cycle_ts": cycle_ts_iso,
        "asof_date": result.asof_date,
        "universe_size": result.universe_size,
        "book": {"longs": result.longs, "shorts": {}},
        "n_longs": len(result.longs),
        "n_shorts": 0,
        "gross_leverage": result.gross_leverage,
        "overlay_leverage": result.overlay_leverage,
        "today_net_return": result.today_net_return,
        "cumulative_shadow_return": cum,
        "forward_cycle_seq": len(rows) + 1,
        "construction": result.construction,
        "orders_placed": 0,
        "broker": None,
    }
    _append_ledger_row(ledger_path, row)
    return row


def forward_summary(ledger_path: Path = LEDGER_PATH_DEFAULT) -> Dict[str, Any]:
    """Honest forward-record summary; empty ledger -> explicit zero-state."""
    rows = _read_ledger_rows(ledger_path)
    if not rows:
        return {"n_cycles": 0, "first_asof_date": None, "last_asof_date": None,
                "cumulative_return": 0.0, "note": "no shadow cycles recorded yet"}
    rets = [float(r.get("today_net_return", 0.0)) for r in rows]
    n = len(rets)
    mean = sum(rets) / n
    sharpe = None
    if n >= 2:
        var = sum((x - mean) ** 2 for x in rets) / (n - 1)
        if var > 0:
            sharpe = (mean / var ** 0.5) * (252.0 ** 0.5)
    return {
        "n_cycles": n,
        "first_asof_date": rows[0].get("asof_date"),
        "last_asof_date": rows[-1].get("asof_date"),
        "cumulative_return": float(rows[-1].get("cumulative_shadow_return", 0.0)),
        "forward_sharpe_annualized": sharpe,
        "forward_sharpe_note": ("n<2 forward cycles — Sharpe undefined, not reported as zero"
                                if sharpe is None else
                                "annualized from forward shadow cycles only, NOT the backtest"),
        "note": None,
    }


# Regression seam: the exact frozen-portfolio series this lane's numbers must
# match (used by the lane tests; import kept so the seam is one symbol).
FROZEN_PORTFOLIO = combined_portfolio

__all__ = [
    "FROZEN_PORTFOLIO", "GATE_VERDICT", "LEDGER_PATH_DEFAULT", "MAX_LEV",
    "MIN_BARS", "PANEL_CACHE", "SOURCE_DOCS", "TARGET_VOL", "UNIVERSE",
    "LaneCycleResult", "compute_lane_cycle", "construction_manifest",
    "forward_summary", "load_panel", "record_lane_cycle", "universe_sha256",
]
