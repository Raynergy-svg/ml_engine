"""Crypto XS-momentum SHADOW forward-tracking (research-adjacent, NOT execution).

Runs the pre-registered H2/H4 crypto cross-sectional momentum construction
FORWARD in shadow (paper) mode so a genuine, un-p-hackable live out-of-sample
track record accumulates for the campaign's one surviving lead: XS 14-day
momentum, dollar-neutral, vol-targeted overlay (the fix that brought full-
sample drawdown from -49% to -20% and turned +0.75 OOS Sharpe into a
4-of-5-criteria-clearing book — see docs/experiment-crypto-edge-hunt-
round2-2026-06-29.md section 3, "H4 — infra-corrected cross-sectional
momentum"). The signal FAILED the significance criterion (DSR < 0.95) in
both the Round-1 and Round-2 pre-registrations, so it is NOT a verified edge
— this module's entire purpose is to let the live forward record (which no
backtest can fake) eventually answer whether the OOS Sharpe holds up.

HARD LINE — everything below is SHADOW ONLY:
  * Zero orders. Zero exchange/broker client. No import of any broker/
    execution module (verified: no `src.brokers`, `execution.py`,
    `place_*_order` reference anywhere in this file).
  * The construction is REUSED VERBATIM, not re-derived. Every frozen
    parameter (lookback, quintile, vol-target, leverage cap, rebalance
    frequency, cost) is imported from the verifier-confirmed harness
    modules (`scripts/experiment_crypto_xs_signals.py`,
    `scripts/experiment_crypto_h2_infra_stress.py`) — never redefined or
    tuned here. `test_momentum_shadow_regression_matches_frozen_harness`
    (tests/test_crypto_momentum_shadow.py) asserts the P&L series this
    module computes is bit-for-bit identical to `backtest_flex`'s output
    for the same inputs, so this can never silently drift from the
    pre-registered construction.
  * Real-money / live-arming is gated the same way the equity harvester is
    gated (`src.crypto.crypto_live_gate` mirrors `src.equity.live_gate`
    exactly) — but nothing in this module or its caller ever calls
    `LiveGate.arm()`. See `src/crypto/crypto_live_gate.py` docstring for
    why arming is structurally unreachable today (no ship-gate pass, no
    broker integration exists for crypto in this repo).

Data cadence caveat (honest, not hidden): the underlying data source
(`src.crypto.data_layer`, Binance monthly static dumps) lags ~1 month behind
the wall clock. Consecutive shadow cycles run on the same calendar day (or
week) will often see no new trading day in the cached data and the cycle is
correctly a no-op (see `record_shadow_cycle`'s duplicate-`asof_date` guard) —
this is not a bug, it is the free-data source's real resolution. Pass
`refresh_klines=True` to force a fresh pull.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_xs_signals as _h            # noqa: E402  frozen H2 harness (verifier-confirmed)
import experiment_crypto_h2_infra_stress as _h2i     # noqa: E402  frozen vol-target/weekly overlay (verifier-confirmed)
import experiment_crypto_round2 as _round2           # noqa: E402  frozen H4 weekly-rebalance constant (verifier-confirmed)

# --------------------------------------------------------------------------- #
# Frozen construction — imported constants ONLY. Never redefine/tune here.
# Source: docs/experiment-crypto-edge-hunt-round2-2026-06-29.md section 3 (H4),
# reused byte-for-byte from scripts/experiment_crypto_round2.py's H4 config
# (REBALANCE_D, vol_target=True, cost_bps=10.0).
# --------------------------------------------------------------------------- #
SIGNAL_NAME = "momentum"
LOOKBACK_D: int = _h.SIGNALS[SIGNAL_NAME]["lookback"]        # 14
DIRECTION: int = _h.SIGNALS[SIGNAL_NAME]["direction"]         # +1
QUINTILE: float = _h.QUINTILE                                 # 0.20
COST_BPS: float = _h.COST_BPS                                 # 10.0
ANN: float = _h2i.ANN                                          # 365.0
TARGET_ANN_VOL: float = _h2i.TARGET_ANN_VOL                    # 0.10
VOL_WINDOW: int = _h2i.VOL_WINDOW                               # 30
MAX_LEV: float = _h2i.MAX_LEV                                   # 3.0
REBALANCE_DAYS: int = _round2.REBALANCE_D                      # 7 (weekly, imported not hardcoded)
VOL_TARGET_ENABLED: bool = True                                 # frozen: the DD-fix overlay is always on

SOURCE_DOC = "docs/experiment-crypto-edge-hunt-round2-2026-06-29.md#3-h4"
PRE_REGISTERED_OOS_SHARPE = 0.75   # Round-1 H2 headline (context only, NOT a live guarantee)
GATE_VERDICT = "clears_ex_history=FALSE (significance FAILS at N=15 and N=3 — see source_doc)"

LEDGER_PATH_DEFAULT = REPO_ROOT / "trained_data" / "crypto" / "shadow_momentum_ledger.jsonl"


def construction_manifest() -> Dict[str, Any]:
    """The frozen construction, exactly as reused — for ledger rows + AXIOM."""
    return {
        "signal": f"xs_momentum_{LOOKBACK_D}d",
        "direction": DIRECTION,
        "quintile": QUINTILE,
        "vol_target_ann": TARGET_ANN_VOL,
        "vol_window_d": VOL_WINDOW,
        "max_leverage": MAX_LEV,
        "rebalance_days": REBALANCE_DAYS,
        "cost_bps": COST_BPS,
        "source_doc": SOURCE_DOC,
        "pre_registered_oos_sharpe": PRE_REGISTERED_OOS_SHARPE,
        "gate_verdict": GATE_VERDICT,
    }


# --------------------------------------------------------------------------- #
# Book + P&L construction — identical math to h2i.backtest_flex, ALSO exposing
# the weight matrix (backtest_flex computes it internally but discards it).
# Regression-tested against backtest_flex's "net" series in
# tests/test_crypto_momentum_shadow.py.
# --------------------------------------------------------------------------- #
def _compute_book_and_returns(
    close: pd.DataFrame,
    funding_daily: pd.DataFrame,
    eligible: pd.DataFrame,
    cols: List[str],
    signal_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce h2i.backtest_flex(..., rebalance_days=REBALANCE_DAYS,
    vol_target=True) exactly, but also return the (levered) weight matrix so a
    caller can read off "today's would-be book" — a value backtest_flex
    computes internally (as `Wlev`) but never returns.
    """
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols]
    price_ret = px.pct_change()
    sig = signal_df.where(elig.shift(1).fillna(False))

    dates = list(px.index)
    target_w = pd.Series(0.0, index=cols)
    weights: Dict[Any, pd.Series] = {}
    for i, d in enumerate(dates):
        if i % REBALANCE_DAYS == 0:
            s = sig.loc[d].dropna()
            w = pd.Series(0.0, index=cols)
            if len(s) >= 10:
                k = max(1, int(len(s) * QUINTILE))
                ranked = s.sort_values()
                low, high = ranked.index[:k], ranked.index[-k:]
                longs, shorts = (high, low) if DIRECTION > 0 else (low, high)
                w[longs] = 0.5 / k
                w[shorts] = -0.5 / k
            target_w = w
        weights[d] = target_w.copy()
    W = pd.DataFrame(weights).T.reindex(columns=cols).fillna(0.0)
    Wprev = W.shift(1).fillna(0.0)

    pr = price_ret.reindex(columns=cols).fillna(0.0)
    fr = fund.reindex(columns=cols).fillna(0.0)
    price_pnl = (Wprev * pr).sum(axis=1)
    carry_pnl = (-Wprev * fr).sum(axis=1)
    gross = price_pnl + carry_pnl

    target_daily = TARGET_ANN_VOL / (ANN ** 0.5)
    realized = gross.rolling(VOL_WINDOW, min_periods=10).std().shift(1)
    lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
    Wlev = W.mul(lev, axis=0)
    Wlevprev = Wlev.shift(1).fillna(0.0)
    turnover = (Wlev - Wlevprev).abs().sum(axis=1)
    price_pnl = (Wlevprev * pr).sum(axis=1)
    carry_pnl = (-Wlevprev * fr).sum(axis=1)
    gross = price_pnl + carry_pnl
    book_weights = Wlev

    cost = turnover * (COST_BPS / 1e4)
    out = pd.DataFrame({
        "price": price_pnl, "carry": carry_pnl, "turnover": turnover,
        "cost": cost, "gross": gross, "net": gross - cost,
    })
    out.index = px.index
    return out, book_weights


@dataclass
class ShadowCycleResult:
    asof_date: str
    universe_size: int
    longs: Dict[str, float]
    shorts: Dict[str, float]
    gross_leverage: float
    today_net_return: float
    today_price_return: float
    today_carry_return: float
    today_cost: float
    today_turnover: float
    # Snapshots the module's constants AT CONSTRUCTION TIME (default_factory
    # runs per-instance, not once at class-definition time). Since these are
    # meant to be a permanently frozen pre-registration, this is intentional —
    # but if a constant were ever revised, historical ledger rows written
    # before the change and re-serialized after it could report the wrong
    # value. Never re-derive this field from a `ShadowCycleResult` created in
    # a different process run than the one that logged it.
    construction: Dict[str, Any] = field(default_factory=construction_manifest)


MIN_SIGNAL_UNIVERSE = 10  # matches the frozen harness's own len(s)>=10 floor (h.backtest / h2i.backtest_flex)


def _last_populated_date(close: pd.DataFrame, cols: List[str], min_valid: int = MIN_SIGNAL_UNIVERSE) -> Any:
    """Latest date where at least `min_valid` symbols have a real close price.

    The free monthly-dump data source (`src.crypto.data_layer`) updates
    unevenly across symbols — a handful of prematurely-cached symbols can
    extend the panel's raw last index by weeks past where the bulk of the
    universe actually has data, producing a spuriously sparse tail (most
    symbols NaN). This trims the panel to the latest date with a healthy
    raw-price count, using the SAME numeric floor (10) the construction's
    own signal-eligibility gate uses (`len(s) >= 10` in `h.backtest` /
    `h2i.backtest_flex`) — not a new threshold. Note this is raw close-price
    population, a coarser/earlier-stage check than the harness's own gate
    (which additionally requires ADV/history eligibility + a non-null signal
    after dropna) — the two can diverge in either direction on an unusual
    date, but sharing the numeric floor keeps the intent aligned: don't rank
    a near-empty cross-section. No signal/overlay parameter changes.
    """
    counts = close[cols].notna().sum(axis=1)
    populated = counts[counts >= min_valid]
    return populated.index[-1] if not populated.empty else close.index[-1]


def compute_shadow_cycle(refresh_klines: bool = False) -> ShadowCycleResult:
    """Pull the crypto universe (cached unless refresh_klines=True), compute
    the frozen H4 book as of the latest well-populated date, and return
    today's would-be positions + today's realized (mark-to-market) P&L.
    """
    close, funding_daily, eligible, _btc_ret, cols, _taker = _h.build_panels(False, refresh_klines)
    if close.empty or not cols:
        raise RuntimeError("crypto momentum shadow: empty universe/panel — data layer returned no data")
    cutoff = _last_populated_date(close, cols)
    close = close.loc[:cutoff]
    funding_daily = funding_daily.loc[:cutoff]
    eligible = eligible.loc[:cutoff]
    sig = _h.make_signal(SIGNAL_NAME, close, None, cols)
    returns, book = _compute_book_and_returns(close, funding_daily, eligible, cols, sig)

    asof = book.index[-1]
    row = book.loc[asof]
    longs = {str(s): float(w) for s, w in row.items() if w > 1e-12}
    shorts = {str(s): float(w) for s, w in row.items() if w < -1e-12}
    ret_row = returns.loc[asof]
    # Eligibility is consumed via `elig.shift(1)` inside _compute_book_and_returns
    # (today's signal mask uses YESTERDAY's eligibility, matching backtest_flex's
    # causal convention) — so the universe that actually gated today's book is
    # the prior date's eligible count, not today's (which can be legitimately 0
    # on the most recent date if the current month's funding dump hasn't posted
    # yet, even though klines already have a bar for it).
    elig_shifted = eligible[cols].shift(1)
    universe_size = int(elig_shifted.loc[asof].sum()) if asof in elig_shifted.index else len(cols)

    return ShadowCycleResult(
        asof_date=str(pd.Timestamp(asof).date()),
        universe_size=universe_size,
        longs=longs,
        shorts=shorts,
        gross_leverage=float(row.abs().sum()),
        today_net_return=float(ret_row["net"]),
        today_price_return=float(ret_row["price"]),
        today_carry_return=float(ret_row["carry"]),
        today_cost=float(ret_row["cost"]),
        today_turnover=float(ret_row["turnover"]),
    )


# --------------------------------------------------------------------------- #
# Ledger — append-only JSONL, atomic append (matches src/equity/live_gate.py's
# `_append_audit_jsonl` convention: flush + fsync, no mkstemp since this is a
# genuinely append-only log, not a rewrite-in-place file).
# --------------------------------------------------------------------------- #
def _read_ledger_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("crypto momentum ledger unreadable at %s: %s", path, exc)
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
        logger.error("crypto momentum ledger append failed for %s: %s", path, exc, exc_info=True)
        raise


def record_shadow_cycle(
    result: ShadowCycleResult,
    *,
    ledger_path: Path = LEDGER_PATH_DEFAULT,
    cycle_ts_iso: str,
) -> Optional[Dict[str, Any]]:
    """Append one shadow-cycle row to the forward-OOS ledger.

    Returns the appended row, or None if this cycle's `asof_date` matches the
    ledger's last entry (no new trading day in the cached data since the last
    cycle — an honest no-op, not an error; see the module docstring's cadence
    caveat). `cumulative_shadow_return` compounds ONLY the rows already in
    this ledger (true forward-from-activation P&L), never the pre-activation
    backtest history that `compute_shadow_cycle` necessarily also touches
    internally to derive today's book.
    """
    rows = _read_ledger_rows(ledger_path)
    if rows and rows[-1].get("asof_date") == result.asof_date:
        logger.info(
            "crypto momentum shadow: no new trading day since last cycle "
            "(asof_date=%s unchanged) — skipping duplicate ledger row",
            result.asof_date,
        )
        return None

    prior_cum = float(rows[-1]["cumulative_shadow_return"]) if rows else 0.0
    cum = (1.0 + prior_cum) * (1.0 + result.today_net_return) - 1.0

    row = {
        "cycle_ts": cycle_ts_iso,
        "asof_date": result.asof_date,
        "universe_size": result.universe_size,
        "book": {"longs": result.longs, "shorts": result.shorts},
        "n_longs": len(result.longs),
        "n_shorts": len(result.shorts),
        "gross_leverage": result.gross_leverage,
        "today_net_return": result.today_net_return,
        "today_price_return": result.today_price_return,
        "today_carry_return": result.today_carry_return,
        "today_cost": result.today_cost,
        "today_turnover": result.today_turnover,
        "cumulative_shadow_return": cum,
        "forward_cycle_seq": len(rows) + 1,
        "construction": result.construction,
        "orders_placed": 0,
        "broker": None,
    }
    _append_ledger_row(ledger_path, row)
    return row


def forward_oos_summary(ledger_path: Path = LEDGER_PATH_DEFAULT) -> Dict[str, Any]:
    """Honest summary of the live-forward track record accumulated so far.

    Empty ledger -> explicit zero-state (never fabricates a Sharpe on n<2).
    """
    rows = _read_ledger_rows(ledger_path)
    if not rows:
        return {
            "n_cycles": 0, "first_asof_date": None, "last_asof_date": None,
            "cumulative_return": 0.0, "note": "no shadow cycles recorded yet",
        }
    daily_rets = [float(r.get("today_net_return", 0.0)) for r in rows]
    n = len(daily_rets)
    mean = sum(daily_rets) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in daily_rets) / (n - 1)
        std = var ** 0.5
        forward_sharpe = (mean / std) * (ANN ** 0.5) if std > 0 else None
    else:
        forward_sharpe = None
    return {
        "n_cycles": n,
        "first_asof_date": rows[0].get("asof_date"),
        "last_asof_date": rows[-1].get("asof_date"),
        "cumulative_return": float(rows[-1].get("cumulative_shadow_return", 0.0)),
        "forward_sharpe_annualized": forward_sharpe,
        "forward_sharpe_note": (
            "n<2 forward cycles — Sharpe undefined, not reported as zero"
            if forward_sharpe is None else
            "annualized from forward shadow cycles only (post-activation), NOT the backtest"
        ),
        "note": None,
    }


__all__ = [
    "ANN", "COST_BPS", "DIRECTION", "GATE_VERDICT", "LEDGER_PATH_DEFAULT",
    "LOOKBACK_D", "MAX_LEV", "PRE_REGISTERED_OOS_SHARPE", "QUINTILE",
    "REBALANCE_DAYS", "SIGNAL_NAME", "SOURCE_DOC", "TARGET_ANN_VOL",
    "VOL_TARGET_ENABLED", "VOL_WINDOW", "ShadowCycleResult",
    "compute_shadow_cycle", "construction_manifest", "forward_oos_summary",
    "record_shadow_cycle",
]
