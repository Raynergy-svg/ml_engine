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
import logging
import sys
from dataclasses import dataclass
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
from src.evidence.forward_ledger import (
    append_unseen_bars,
    cadence_counts as fl_cadence_counts,
    forward_rows as fl_forward_rows,
    read_rows as fl_read_rows,
)

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
# THIS lane's own pre-registered result — the 37-asset Round-3 Lead B run at
# the spec-correct 10% target vol, 2 bps (round-3 doc section 5). rev 2 fix
# (2026-07-19 review): the first manifest wrongly stamped the ORIGINAL
# 21-asset experiment's numbers (0.744/18%/0.829) as this lane's verdict.
PRE_REGISTERED_37 = {
    "full_sharpe": 0.665, "full_max_dd": 0.167,
    "full_dsr_n20": 0.987, "full_bootstrap_p": 0.0,
    "oos_sharpe": 0.788, "oos_max_dd": 0.162,
    "oos_dsr_n20": 0.843, "oos_bootstrap_p": 0.002,
    "clears_gate": False,
}
GATE_VERDICT = ("clears_gate=FALSE — OOS DSR 0.843 < 0.95 (short 0.107; the OOS "
                "bootstrap-p half PASSES at 0.002 < 0.0025 Bonferroni); full-sample "
                "significance clears (DSR 0.987, p 0.0). Power-limited near-miss -> "
                "forward-test the spec UNCHANGED (readiness report 2026-07-18). "
                "Risk-control book, NOT alpha: OOS loses to EW/60-40 on return "
                "(beta-to-SPY 0.128, maxDD 0.162 vs EW -0.765).")
# Background context ONLY — a DIFFERENT construction (the original 21-asset
# experiment), never this lane's verdict.
BACKGROUND_21_ASSET = {"full_sharpe": 0.744, "full_max_dd": 0.18, "oos_sharpe": 0.829,
                       "source": "docs/experiment-multi-asset-trend-2026-06-29.md"}

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
        "pre_registered_37_asset": dict(PRE_REGISTERED_37),
        "background_21_asset_context": dict(BACKGROUND_21_ASSET),
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
class LaneFrames:
    """The frozen construction's full aligned state (regression-locked)."""
    net: pd.Series               # == combined_portfolio(prices, target_vol=0.10) bit-for-bit
    scalar: pd.Series            # overlay exposure scalar (causal)
    hrp: pd.DataFrame            # causal HRP sleeve-weight panel
    held: pd.DataFrame           # per-asset 0/1 monthly-held state (frozen rule)
    universe_size: int


def compute_frames(prices: Optional[pd.DataFrame] = None, *,
                   refresh: bool = False) -> LaneFrames:
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
    held = pd.DataFrame({a: h for a in streams.columns
                         if (h := _held_state(prices[a])) is not None})
    held = held.reindex(net.index).fillna(0.0)
    return LaneFrames(net=net,
                      scalar=scalar.reindex(net.index).fillna(0.0),
                      hrp=weight_panel.reindex(net.index).fillna(0.0),
                      held=held,
                      universe_size=int(streams.shape[1]))


def _book_at(frames: LaneFrames, i: int, *, applied: bool) -> Dict[str, float]:
    """Period-aligned book (2026-07-19 review fix 5).

    The frozen construction is causal: ``net[t]`` was earned by the held
    state decided at ``t-1`` (``w.shift(1)``), weighted by the HRP row and
    overlay scalar assigned AT ``t`` (both estimated causally from <= t-1).
    ``applied=True``  -> the book that GENERATED net[t]:
                         hrp[t] x held[t-1] x scalar[t].
    ``applied=False`` -> the STANDING next-bar target as of t:
                         hrp[t] x held[t] x scalar[t] (HRP/scalar may
                         re-estimate on the next bar — noted, not hidden)."""
    t = frames.net.index[i]
    held_row = (frames.held.iloc[i - 1] if applied and i > 0
                else frames.held.iloc[i] * 0.0 if applied
                else frames.held.iloc[i])
    lev = float(frames.scalar.iloc[i])
    hrp_row = frames.hrp.loc[t].fillna(0.0)
    book = {}
    for a in frames.held.columns:
        w = float(hrp_row.get(a, 0.0)) * float(held_row.get(a, 0.0)) * lev
        if abs(w) > 1e-12:
            book[str(a)] = w
    return book


def capture_forward(prices: Optional[pd.DataFrame] = None, *,
                    refresh: bool = False,
                    ledger_path: Path = LEDGER_PATH_DEFAULT,
                    cycle_ts_iso: str) -> List[Dict[str, Any]]:
    """Evidence-safe forward capture (review fixes 2+3+5): activation
    baseline on the first run (no realized return), deterministic backfill
    of every unseen bar afterwards, applied vs next-target books split per
    row. See ``src.evidence.forward_ledger`` for the engine contract."""
    frames = compute_frames(prices, refresh=refresh)
    dates = [str(pd.Timestamp(t).date()) for t in frames.net.index]
    pos = {d: i for i, d in enumerate(dates)}

    def payload_for(d: str) -> Dict[str, Any]:
        i = pos[d]
        nxt = _book_at(frames, i, applied=False)
        return {
            "today_net_return": float(frames.net.iloc[i]),
            "applied_book": {"longs": _book_at(frames, i, applied=True), "shorts": {}},
            "book": {"longs": nxt, "shorts": {}},
            "return_period_start": dates[i - 1] if i > 0 else None,
            "overlay_leverage": float(frames.scalar.iloc[i]),
            "gross_leverage": float(sum(abs(v) for v in nxt.values())),
            "universe_size": frames.universe_size,
            "n_longs": len(nxt),
            "n_shorts": 0,
            "construction": construction_manifest(),
        }

    def activation_payload_for(d: str) -> Dict[str, Any]:
        i = pos[d]
        nxt = _book_at(frames, i, applied=False)
        return {
            "book": {"longs": nxt, "shorts": {}},
            "overlay_leverage": float(frames.scalar.iloc[i]),
            "gross_leverage": float(sum(abs(v) for v in nxt.values())),
            "universe_size": frames.universe_size,
            "n_longs": len(nxt),
            "n_shorts": 0,
            "construction": construction_manifest(),
        }

    return append_unseen_bars(ledger_path, dates=dates, payload_for=payload_for,
                              activation_payload_for=activation_payload_for,
                              cycle_ts_iso=cycle_ts_iso)


def forward_summary(ledger_path: Path = LEDGER_PATH_DEFAULT) -> Dict[str, Any]:
    """Honest forward record; activation excluded; cadence in
    rebalances/months (24-30 completed rebalances is the evidence bar,
    NEVER raw daily row count)."""
    rows = fl_read_rows(ledger_path)
    fwd = fl_forward_rows(rows)
    rets = [float(r["today_net_return"]) for r in fwd]
    n = len(rets)
    sharpe = None
    if n >= 2:
        mean = sum(rets) / n
        var = sum((x - mean) ** 2 for x in rets) / (n - 1)
        if var > 0:
            sharpe = (mean / var ** 0.5) * (252.0 ** 0.5)
    return {
        "n_cycles": n,
        "first_asof_date": fwd[0]["asof_date"] if fwd else None,
        "last_asof_date": rows[-1]["asof_date"] if rows else None,
        "activated": bool(rows),
        "cumulative_return": float(rows[-1].get("cumulative_shadow_return", 0.0)) if rows else 0.0,
        "forward_sharpe_annualized": sharpe,
        "forward_sharpe_note": ("n<2 forward bars — Sharpe undefined, not reported as zero"
                                if sharpe is None else
                                "annualized from forward bars only (post-activation), NOT the backtest"),
        "cadence": fl_cadence_counts(rows),
        "evidence_target": "24-30 completed MONTHLY rebalances (readiness report) — see cadence",
    }


# Regression seam: the exact frozen-portfolio series this lane's numbers must
# match (used by the lane tests; import kept so the seam is one symbol).
FROZEN_PORTFOLIO = combined_portfolio

__all__ = [
    "BACKGROUND_21_ASSET", "FROZEN_PORTFOLIO", "GATE_VERDICT",
    "LEDGER_PATH_DEFAULT", "MAX_LEV", "MIN_BARS", "PANEL_CACHE",
    "PRE_REGISTERED_37", "SOURCE_DOCS", "TARGET_VOL", "UNIVERSE",
    "LaneFrames", "capture_forward", "compute_frames",
    "construction_manifest", "forward_summary", "load_panel",
    "universe_sha256",
]
