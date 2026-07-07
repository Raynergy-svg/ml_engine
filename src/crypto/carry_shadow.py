"""Crypto cash-and-carry SHADOW forward-tracking (research-adjacent, NOT execution).

Runs the pre-registered cash-and-carry construction (long spot proxy / short
perpetual, positive-funding-only, delta-neutral by construction) FORWARD in
shadow (paper) mode so a genuine, un-p-hackable live out-of-sample track
record accumulates. See docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md
for the frozen construction and docs/ENGINEERING_BRAIN.md P3 for why this
lever was prioritized: crypto funding/cash-and-carry is "the strongest
genuinely-live retail lever" (~8-18%/yr in normal regimes per BIS WP 1087) --
but a **risk premium with a real exchange-solvency/liquidation tail, not free
money**. The frozen backtest (docs/prereg-crypto-cash-and-carry-shadow-
2026-07-06.md section 5) FAILED the ship gate on turnover cost (not price
adverse-selection like H1) -- this module's entire purpose is to let the live
forward record answer whether a real deployment (with holding-period
discipline this naive daily-threshold backtest doesn't have) would fare
differently. It is NOT a verified edge.

HARD LINE -- everything below is SHADOW ONLY:
  * Zero orders. Zero exchange/broker client. No import of any broker/
    execution module (verified: no `src.brokers`, `execution.py`,
    `place_*_order` reference anywhere in this file).
  * The construction is REUSED VERBATIM, not re-derived. Every frozen
    parameter (lookback, cost, vol-target, leverage cap) is imported from the
    pre-registered harness module (`scripts/experiment_crypto_cash_and_carry.py`,
    which itself imports the universe/signal from
    `scripts/experiment_crypto_funding_carry.py` (H1) and the vol-target
    overlay from `scripts/experiment_crypto_h2_infra_stress.py`) -- never
    redefined or tuned here. `test_carry_shadow_regression_matches_frozen_harness`
    (tests/test_crypto_carry_shadow.py) asserts the P&L series this module
    computes is bit-for-bit identical to the frozen harness's `backtest()`
    output for the same inputs.
  * Real-money / live-arming is gated the same way every other lane is gated
    (`src.crypto.crypto_carry_live_gate` mirrors `src.equity.live_gate`
    exactly) -- but nothing in this module or its caller ever calls
    `LiveGate.arm()`. No `trained_data/crypto_carry/SHIP_GATE.json` exists and
    none should be fabricated -- the frozen backtest's own verdict is
    `clears_ex_history=FALSE`. No exchange/broker client exists for crypto in
    this repo -- even an armed gate has no execution path to route through.

Delta-neutral-by-construction caveat (honest, not hidden): this module
assumes the spot leg tracks the perp price essentially exactly, so there is
NO price P&L term -- `today_price_return` is always 0.0 by design (kept in
the ledger schema for cross-lane consistency, not because a price P&L
component exists here). Real spot-perp basis carries small residual noise
this construction does not model. The single most important caveat is the
**exchange-solvency / liquidation tail**: a real cash-and-carry position is
exposed to counterparty failure and perp-leg margin/liquidation risk during a
violent basis blowout, neither of which a daily mark-to-market backtest or
this shadow ledger can capture. This is a risk premium, not free money.

Data cadence caveat (same as `momentum_shadow.py`): the underlying data source
(`src.crypto.data_layer`, Binance monthly static dumps) lags ~1 month behind
the wall clock. Consecutive shadow cycles run on the same calendar day (or
week) will often see no new trading day in the cached data and the cycle is
correctly a no-op (see `record_shadow_cycle`'s duplicate-`asof_date` guard).
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

import experiment_crypto_cash_and_carry as _cc         # noqa: E402  frozen carry construction (verifier-confirmed)
import experiment_crypto_funding_carry as _h1           # noqa: E402  frozen universe+signal (H1, verifier-confirmed)

# --------------------------------------------------------------------------- #
# Frozen construction -- imported constants ONLY. Never redefine/tune here.
# Source: docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md (frozen
# BEFORE the backtest ran), reused byte-for-byte from
# scripts/experiment_crypto_cash_and_carry.py's module-level constants.
# --------------------------------------------------------------------------- #
ANN: float = _cc.ANN                                    # 365.0
LOOKBACK_D: int = _cc.LOOKBACK_D                        # 3
MIN_ADV_USD: float = _cc.MIN_ADV_USD                    # 10_000_000
MIN_HISTORY_D: int = _cc.MIN_HISTORY_D                  # 90
COST_BPS_ROUNDTRIP: float = _cc.COST_BPS_ROUNDTRIP      # 20.0 (two legs)
TARGET_ANN_VOL: float = _cc.TARGET_ANN_VOL              # 0.10
VOL_WINDOW: int = _cc.VOL_WINDOW                         # 30
MAX_LEV: float = _cc.MAX_LEV                             # 3.0
REBALANCE_DAYS: int = 1                                  # daily -- matches the signal's own cadence
VOL_TARGET_ENABLED: bool = True                          # frozen: overlay is always on

SOURCE_DOC = "docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md"
GATE_VERDICT = "clears_ex_history=FALSE (turnover cost dominates a real funding premium -- see source_doc section 5)"
RISK_PREMIUM_NOTE = (
    "Cash-and-carry harvests a real, market-neutral funding premium (BIS WP 1087) "
    "but carries a real exchange-solvency / liquidation tail. This is a risk "
    "premium with a fat tail, not free money -- see source_doc sections 0-2."
)

MIN_SIGNAL_UNIVERSE = 10  # matches the frozen harness's own selection-eligibility spirit

LEDGER_PATH_DEFAULT = REPO_ROOT / "trained_data" / "crypto_carry" / "shadow_carry_ledger.jsonl"


def construction_manifest() -> Dict[str, Any]:
    """The frozen construction, exactly as reused -- for ledger rows + AXIOM."""
    return {
        "signal": "cash_and_carry_positive_funding_only",
        "lookback_d": LOOKBACK_D,
        "cost_bps_roundtrip": COST_BPS_ROUNDTRIP,
        "vol_target_ann": TARGET_ANN_VOL,
        "vol_window_d": VOL_WINDOW,
        "max_leverage": MAX_LEV,
        "rebalance_days": REBALANCE_DAYS,
        "source_doc": SOURCE_DOC,
        "gate_verdict": GATE_VERDICT,
        "risk_premium_note": RISK_PREMIUM_NOTE,
    }


# --------------------------------------------------------------------------- #
# Book + P&L construction -- identical math to _cc.backtest, ALSO exposing the
# weight matrix (backtest computes it internally but discards it). Regression-
# tested against _cc.backtest's "net" series in tests/test_crypto_carry_shadow.py.
# --------------------------------------------------------------------------- #
def _compute_book_and_returns(
    funding_daily: pd.DataFrame,
    eligible: pd.DataFrame,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce _cc.backtest(..., cost_bps_roundtrip=COST_BPS_ROUNDTRIP,
    vol_target=True) exactly, but also return the (levered) weight matrix so a
    caller can read off "today's would-be book" -- a value the frozen
    backtest computes internally but never returns.
    """
    fund = funding_daily[cols]
    elig = eligible[cols]
    sig = fund.rolling(LOOKBACK_D, min_periods=LOOKBACK_D).mean().shift(1)
    sig = sig.where(elig.shift(1).fillna(False))

    dates = list(fund.index)
    prev_w = pd.Series(0.0, index=cols)
    unlevered_rows = []
    weights: Dict[Any, pd.Series] = {}
    for d in dates:
        s = sig.loc[d].dropna()
        positive = s[s > 0.0]
        w = pd.Series(0.0, index=cols)
        if len(positive) > 0:
            w[positive.index] = 1.0 / len(positive)
        weights[d] = w.copy()
        fr = fund.loc[d].reindex(cols).fillna(0.0)
        carry_pnl = float((prev_w * fr).sum())
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * (COST_BPS_ROUNDTRIP / 1e4)
        unlevered_rows.append({"date": d, "carry": carry_pnl, "turnover": turnover,
                               "cost": cost, "net": carry_pnl - cost})
        prev_w = w
    r = pd.DataFrame(unlevered_rows).set_index("date")
    W = pd.DataFrame(weights).T.reindex(columns=cols).fillna(0.0)

    target_daily = TARGET_ANN_VOL / (ANN ** 0.5)
    realized = r["net"].rolling(VOL_WINDOW, min_periods=10).std().shift(1)
    lev = (target_daily / realized).clip(upper=MAX_LEV).fillna(0.0)
    r["carry"] = r["carry"] * lev
    r["cost"] = r["cost"] * lev
    r["gross"] = r["carry"]
    r["net"] = r["carry"] - r["cost"]
    r["turnover"] = r["turnover"] * lev
    r.index = fund.index
    book_weights = W.mul(lev, axis=0)
    return r, book_weights


@dataclass
class ShadowCycleResult:
    asof_date: str
    universe_size: int
    longs: Dict[str, float]  # every position is a "long the carry" (long spot / short perp)
    shorts: Dict[str, float]  # always empty -- no reverse-carry leg (see module docstring)
    gross_leverage: float
    today_net_return: float
    today_price_return: float  # always 0.0 -- delta-neutral by construction, see module docstring
    today_carry_return: float
    today_cost: float
    today_turnover: float
    construction: Dict[str, Any] = field(default_factory=construction_manifest)


def _last_populated_date(eligible: pd.DataFrame, cols: List[str],
                         min_valid: int = MIN_SIGNAL_UNIVERSE) -> Any:
    """Latest date where at least `min_valid` symbols are eligible.

    Mirrors `momentum_shadow._last_populated_date`'s rationale: the free
    monthly-dump data source updates unevenly across symbols, so this trims
    the panel to the latest date with a healthy eligible-symbol count rather
    than ranking/selecting off a near-empty cross-section on a stale tail.
    """
    counts = eligible[cols].sum(axis=1)
    populated = counts[counts >= min_valid]
    return populated.index[-1] if not populated.empty else eligible.index[-1]


def compute_shadow_cycle(refresh_klines: bool = False) -> ShadowCycleResult:
    """Pull the crypto universe (cached unless refresh_klines=True), compute
    the frozen cash-and-carry book as of the latest well-populated date, and
    return today's would-be positions + today's realized (mark-to-market) P&L.
    """
    close, _adv, funding_daily, eligible, _btc_ret, cols = _h1.build_panels(refresh_klines)
    if funding_daily.empty or not cols:
        raise RuntimeError("crypto carry shadow: empty universe/panel -- data layer returned no data")
    cutoff = _last_populated_date(eligible, cols)
    funding_daily = funding_daily.loc[:cutoff]
    eligible = eligible.loc[:cutoff]
    returns, book = _compute_book_and_returns(funding_daily, eligible, cols)

    asof = book.index[-1]
    row = book.loc[asof]
    longs = {str(s): float(w) for s, w in row.items() if w > 1e-12}
    ret_row = returns.loc[asof]
    elig_shifted = eligible[cols].shift(1)
    universe_size = int(elig_shifted.loc[asof].sum()) if asof in elig_shifted.index else len(cols)

    return ShadowCycleResult(
        asof_date=str(pd.Timestamp(asof).date()),
        universe_size=universe_size,
        longs=longs,
        shorts={},
        gross_leverage=float(row.abs().sum()),
        today_net_return=float(ret_row["net"]),
        today_price_return=0.0,
        today_carry_return=float(ret_row["carry"]),
        today_cost=float(ret_row["cost"]),
        today_turnover=float(ret_row["turnover"]),
    )


# --------------------------------------------------------------------------- #
# Ledger -- append-only JSONL, atomic append (matches momentum_shadow.py's /
# src/equity/live_gate.py's `_append_audit_jsonl` convention).
# --------------------------------------------------------------------------- #
def _read_ledger_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("crypto carry ledger unreadable at %s: %s", path, exc)
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
        logger.error("crypto carry ledger append failed for %s: %s", path, exc, exc_info=True)
        raise


def record_shadow_cycle(
    result: ShadowCycleResult,
    *,
    ledger_path: Path = LEDGER_PATH_DEFAULT,
    cycle_ts_iso: str,
) -> Optional[Dict[str, Any]]:
    """Append one shadow-cycle row to the forward-OOS ledger.

    Returns the appended row, or None if this cycle's `asof_date` matches the
    ledger's last entry (no new trading day since the last cycle -- an
    honest no-op, not an error; see the module docstring's cadence caveat).
    """
    rows = _read_ledger_rows(ledger_path)
    if rows and rows[-1].get("asof_date") == result.asof_date:
        logger.info(
            "crypto carry shadow: no new trading day since last cycle "
            "(asof_date=%s unchanged) -- skipping duplicate ledger row",
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
            "n<2 forward cycles -- Sharpe undefined, not reported as zero"
            if forward_sharpe is None else
            "annualized from forward shadow cycles only (post-activation), NOT the backtest"
        ),
        "note": None,
    }


__all__ = [
    "ANN", "COST_BPS_ROUNDTRIP", "GATE_VERDICT", "LEDGER_PATH_DEFAULT", "LOOKBACK_D",
    "MAX_LEV", "MIN_ADV_USD", "MIN_HISTORY_D", "MIN_SIGNAL_UNIVERSE", "REBALANCE_DAYS",
    "RISK_PREMIUM_NOTE", "SOURCE_DOC", "TARGET_ANN_VOL", "VOL_TARGET_ENABLED", "VOL_WINDOW",
    "ShadowCycleResult", "compute_shadow_cycle", "construction_manifest",
    "forward_oos_summary", "record_shadow_cycle",
]
