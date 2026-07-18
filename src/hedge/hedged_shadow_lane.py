"""AXIOM Hedge Layer — Phase 4: raw-vs-hedged shadow lane (SHADOW / ANALYSIS-ONLY).

The capstone question the roadmap set up Phases 1-3 to answer: **does hedging
improve after-cost expectancy** for Buddy's existing shadow strategies, or is
their return just market beta wearing a stock-picking costume?

For each covered shadow strategy (equity harvester, crypto momentum,
Track B), this module:

1. Reads that strategy's OWN forward-shadow book/ledger (never re-derives or
   forks its construction — ``src.equity.rebalance``, ``src.crypto.
   momentum_shadow``, ``src.equity.track_b_shadow`` remain the single source
   of truth for what the book actually is).
2. Converts the book into Phase 1/2 ``ExposurePosition`` records and runs the
   REAL ``portfolio_exposure.build_exposure_report`` + ``hedge_candidates.
   build_hedge_report`` (Phase 1-3, not forked) to get a ranked hedge
   proposal for the book as a whole.
3. Marks BOTH lanes forward: **Lane A** = the strategy's own raw book,
   **Lane B** = raw + the top-ranked structurally-valid hedge applied
   (preferring the ``market`` style, since "does this book beat the market"
   is the decisive alpha-vs-beta question). Marking uses only cached/free
   price data already vetted in this repo (``market_data/equity/
   sp500_prices.parquet`` — same panel ``track_b_shadow`` reads).
4. Appends one row per cycle to ``trained_data/hedge/
   raw_vs_hedged_ledger.jsonl`` (the raw-vs-hedged forward P&L record) and
   ``trained_data/hedge/hedge_decision_log.jsonl`` (the full P3 structured
   report, for audit).

HARD LINE — everything below is SHADOW/ANALYSIS-ONLY, same triplet as
Phases 1-3:
  * Zero orders. No broker/execution import anywhere in this file.
  * Never unhalts a lane, never arms a live gate, never bypasses a gate,
    never changes leverage.
  * Every row carries ``runtime_allowed=False, paper_only=True,
    human_review_required=True`` — fixed module constants, never a
    parameter, so no call site can override them.

Honesty notes worth stating explicitly:

* **Notional is a modeling constant, not a real account.** Every strategy's
  book is weight-fraction-of-NAV; there is no single shared real NAV across
  three independent shadow lanes. ``SHADOW_NOTIONAL_DEFAULT`` converts
  fractional weights to a dollar ``risk_home`` figure ONLY so Phase 1-3's
  dollar-denominated exposure/cost machinery can run — it does not imply any
  of these lanes control real capital (they never have).
* **Equity hedge cost is genuinely unknown, not faked as zero.** The P2 cost
  model (``src.data.execution_cost_model.estimate_execution_cost``) reads
  OANDA ``ORDER_FILL`` history — there are no OANDA fills for SPY / sector
  ETFs, so equity-class hedge legs come back ``cost_known=False`` by
  construction (the same real model Phase 3 already uses, not forked or
  special-cased here). The scorecard (``hedge_scorecard.py``) surfaces GROSS
  hedged performance in that case and explicitly flags cost as unmodeled for
  that venue — never silently assumes it's cheap.
* **Crypto is not a covered asset class.** ``portfolio_exposure.
  VALID_ASSET_CLASSES = ("fx", "equity")`` — extending exposure tagging to
  crypto is a real, separate data-engineering task (bucket maps, beta
  proxies) out of scope here. Crypto momentum's book is still run through
  the SAME pipeline as the other two lanes (no special-cased branch); it
  naturally comes back ``fail_closed`` / ``DECISION_SHADOW_ONLY`` via P1-P3's
  own existing fail-closed contract, which this module reports honestly
  (Lane B == Lane A, ``hedge.status = "unsupported_asset_class"``) rather
  than fabricating a crypto hedge.
* **Market/sector hedge marks are a real, cached, honestly-labeled PROXY.**
  Neither SPY nor any SPDR sector ETF is in the cached S&P constituent price
  panel this repo has on disk. Rather than fetch live (network-dependent,
  unusable for deterministic no-mock tests) or fabricate a number, the
  market/sector hedge overlay return is the cross-sectional equal-weight
  mean forward return of the REAL cached constituent panel (all names, or
  the subset mapped to the flagged sector) — a genuine, disk-verifiable
  number, just not the literal ETF print. Every overlay return this produces
  carries ``is_proxy=True`` and a ``proxy_note`` so it is never mistaken for
  a real ETF fill.
* **Stale/missing forward price data degrades honestly.** If the cached
  price panel has no bar for a book's ``asof_date`` (or no bar strictly
  after it), the raw/overlay return for that leg is reported ``None`` with
  an explicit ``insufficient_forward_price_data`` note — never zero-filled,
  never silently skipped from the ledger row.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.hedge.exposure_tags import (
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
)
from src.hedge.hedge_candidates import MARKET_HEDGE_INSTRUMENT, build_hedge_report
from src.hedge.portfolio_exposure import (
    ExposurePosition,
    ExposureReport,
    build_exposure_report,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------- #
# Fixed safety fields — SHADOW/ANALYSIS-ONLY, matching Phases 1-3.       #
# --------------------------------------------------------------------- #
RUNTIME_ALLOWED = False
PAPER_ONLY = True
HUMAN_REVIEW_REQUIRED = True

# --------------------------------------------------------------------- #
# Paths                                                                  #
# --------------------------------------------------------------------- #
HEDGE_LEDGER_DIR = REPO_ROOT / "trained_data" / "hedge"
RAW_VS_HEDGED_LEDGER_PATH = HEDGE_LEDGER_DIR / "raw_vs_hedged_ledger.jsonl"
HEDGE_DECISION_LOG_PATH = HEDGE_LEDGER_DIR / "hedge_decision_log.jsonl"

EQUITY_HARVESTER_STATE_PATH = REPO_ROOT / "trained_data" / "equity" / "rebalance_state.json"
CRYPTO_MOMENTUM_LEDGER_PATH = REPO_ROOT / "trained_data" / "crypto" / "shadow_momentum_ledger.jsonl"
TRACK_B_LEDGER_PATH = REPO_ROOT / "trained_data" / "research" / "track_b_shadow_ledger.jsonl"
FX_TREND_ACCOUNT_STATE_PATH = REPO_ROOT / "trained_data" / "oanda" / "account_state.json"
EQUITY_PRICE_PANEL_PATH = REPO_ROOT / "market_data" / "equity" / "sp500_prices.parquet"
# Per-instrument daily OHLCV CSVs (OANDA-sourced, cached): {PAIR}_D.csv. The FX
# forward-price source for marking an FX hedge leg's overlay P&L — the FX
# counterpart of the equity price panel. 19 majors/crosses, 2014-01-01 onward.
FX_PRICE_PANEL_DIR = REPO_ROOT / "market_data" / "factor"

# Weight-fraction-of-NAV -> dollar risk_home conversion constant. See module
# docstring "Notional is a modeling constant, not a real account."
SHADOW_NOTIONAL_DEFAULT = 100_000.0

# A sector proxy averaged over fewer than this many cached constituent names
# is too thin to trust as a sector-return estimate.
MIN_SECTOR_PROXY_TICKERS = 3

# A single-day |return| above this is treated as a bad tick (unadjusted
# corporate action, data-vendor glitch), never a real print, and excluded
# from any average. See ``_ticker_forward_return`` docstring — found via a
# real cached-panel artifact while building this module (one ticker printed
# +1844% in a day).
MAX_SANE_DAILY_RETURN = 0.50

STRATEGIES = ("equity_harvester", "crypto_momentum", "track_b", "fx_trend", "oanda_fx")

# The 15-agent scanner's own trade journal — the ONLY scanner-attributed
# position source (account_state.json is the whole shared practice account,
# which also carries the trend lane's positions; using it here would
# mis-attribute trend exposure to the scanner's agents).
SCANNER_JOURNAL_PATH = REPO_ROOT / "trained_data" / "trade_journal_rl.json"


# --------------------------------------------------------------------- #
# Data shapes                                                           #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class BookSnapshot:
    """One strategy's current (or ledger-latest) book, translated into a
    shape this module's exposure/hedge/price machinery can consume."""
    strategy: str
    asset_class: str  # "equity" | "crypto" | "fx"
    asof_date: str
    weights: Dict[str, float]  # ticker -> signed fraction-of-NAV
    raw_net_return: Optional[float]  # from the strategy's OWN ledger, if already computed there
    raw_gross_return: Optional[float]
    raw_cost: Optional[float]
    return_source: str  # "own_ledger" | "must_compute" | "no_data"
    meta: Dict[str, Any] = field(default_factory=dict)


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    """Tolerant JSONL reader — matches the convention already used by
    ``momentum_shadow._read_ledger_rows`` / ``track_b_shadow._read_ledger_rows``.
    Missing/corrupt file -> [] (fail soft), never crashes the caller."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("hedged_shadow_lane: skipping corrupt line in %s", path)
    except OSError as exc:
        logger.warning("hedged_shadow_lane: ledger unreadable at %s (%s)", path, exc)
        return []
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append-only JSONL write, flush + fsync — same durability convention
    as ``momentum_shadow._append_ledger_row``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        logger.error("hedged_shadow_lane: ledger append failed for %s: %s", path, exc, exc_info=True)


# --------------------------------------------------------------------- #
# Strategy loaders — read-only, reuse each strategy's own persisted state #
# --------------------------------------------------------------------- #
def load_equity_harvester_book(state_path: Path = EQUITY_HARVESTER_STATE_PATH) -> Optional[BookSnapshot]:
    """Equity harvester has no per-cycle forward-return ledger (its
    ``cycle_ledger.jsonl`` stores tamper-evident hashes only, not the raw
    weights) — the live book lives in ``rebalance_state.json``'s
    ``current_actual_weights``. Return ``None`` (never a fabricated empty
    book) if the state file is missing/corrupt/empty."""
    if not state_path.exists():
        logger.warning("hedged_shadow_lane: equity harvester state missing at %s", state_path)
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("hedged_shadow_lane: equity harvester state unreadable (%s)", exc)
        return None
    weights = payload.get("current_actual_weights") or {}
    if not isinstance(weights, dict) or not weights:
        logger.warning("hedged_shadow_lane: equity harvester has no current_actual_weights — flat book")
        return None
    asof = payload.get("last_rebalance_asof") or (payload.get("active_plan") or {}).get("asof")
    if not asof:
        logger.warning("hedged_shadow_lane: equity harvester state has no asof — cannot mark forward")
        return None
    return BookSnapshot(
        strategy="equity_harvester", asset_class="equity",
        asof_date=str(pd.Timestamp(asof).date()),
        weights={str(k): float(v) for k, v in weights.items()},
        raw_net_return=None, raw_gross_return=None, raw_cost=None,
        return_source="must_compute",
        meta={"source_path": str(state_path)},
    )


def load_crypto_momentum_book(ledger_path: Path = CRYPTO_MOMENTUM_LEDGER_PATH) -> Optional[BookSnapshot]:
    rows = _read_jsonl_rows(ledger_path)
    if not rows:
        logger.warning("hedged_shadow_lane: crypto momentum ledger empty/missing at %s", ledger_path)
        return None
    latest = rows[-1]
    book = latest.get("book") or {}
    weights = {**(book.get("longs") or {}), **(book.get("shorts") or {})}
    if not weights:
        logger.warning("hedged_shadow_lane: crypto momentum latest cycle has an empty book")
        return None
    return BookSnapshot(
        strategy="crypto_momentum", asset_class="crypto",
        asof_date=str(latest.get("asof_date", "")),
        weights={str(k): float(v) for k, v in weights.items()},
        raw_net_return=_safe_float(latest.get("today_net_return")),
        raw_gross_return=_safe_float(latest.get("today_price_return")),
        raw_cost=_safe_float(latest.get("today_cost")),
        return_source="own_ledger",
        meta={"source_path": str(ledger_path), "gross_leverage": latest.get("gross_leverage"),
              "construction": latest.get("construction", {})},
    )


def load_track_b_book(ledger_path: Path = TRACK_B_LEDGER_PATH) -> Optional[BookSnapshot]:
    rows = _read_jsonl_rows(ledger_path)
    if not rows:
        logger.warning("hedged_shadow_lane: track_b ledger empty/missing at %s", ledger_path)
        return None
    latest = rows[-1]
    book = latest.get("book") or {}
    weights = {**(book.get("longs") or {}), **(book.get("shorts") or {})}
    if not weights:
        logger.warning("hedged_shadow_lane: track_b latest cycle has an empty book")
        return None
    return BookSnapshot(
        strategy="track_b", asset_class="equity",
        asof_date=str(latest.get("asof_date", "")),
        weights={str(k): float(v) for k, v in weights.items()},
        raw_net_return=_safe_float(latest.get("today_net_return")),
        raw_gross_return=_safe_float(latest.get("today_gross_return")),
        raw_cost=_safe_float(latest.get("today_cost")),
        return_source="own_ledger",
        meta={"source_path": str(ledger_path), "gross_leverage": latest.get("gross_leverage"),
              "construction": latest.get("construction", {}),
              "gate_verdict": (latest.get("construction") or {}).get("gate_verdict")},
    )


def load_fx_trend_book(account_state_path: Path = FX_TREND_ACCOUNT_STATE_PATH,
                       ticks_root: Optional[Path] = None) -> Optional[BookSnapshot]:
    """The FX trend lane — the ONE lane actually placing (practice) trades.

    Unlike the equity/crypto loaders (which read a strategy-owned rebalance
    ledger of fraction-of-NAV weights), the FX lane's live book is the real
    OANDA open positions in ``account_state.json``, denominated in UNITS. We
    translate units -> signed home-currency notional as a fraction of NAV,
    reusing ``exposure_history.resolve_current_book`` (the SAME
    freshness-check + tick-rate resolution the exposure-history capture uses —
    one rate path, no drift). The imported call is function-local to break the
    import cycle (``exposure_history`` imports ``_append_jsonl`` from here).

    Raw return: the lane's OWN live P&L — total unrealized P&L / NAV, a
    point-in-time open-book MARK (not a period return; difference consecutive
    ledger rows for a period P&L). ``return_source="own_ledger"`` because it
    comes from the lane's real account, never a marked-forward proxy.

    Hedged overlay marking is honestly UNRESOLVED for FX right now: the
    Phase-3 ``generate_fx_hedges`` proposal is logged, but
    ``hedge_overlay_pnl`` can only mark forward through the equity price panel,
    so an FX hedge leg resolves to ``None`` (documented, never fabricated).
    Wiring an FX forward-price source into ``hedge_overlay_pnl`` is the next
    lever. Returns ``None`` (never a fabricated book) on a stale/missing/flat
    account state or one with no NAV.
    """
    from src.hedge.exposure_history import TICKS_ROOT, resolve_current_book

    book = resolve_current_book(account_state_path=account_state_path,
                                ticks_root=ticks_root or TICKS_ROOT)
    if book is None:
        return None
    resolved = book.resolved
    if not resolved:
        logger.warning("hedged_shadow_lane: fx_trend book is flat / no positions resolved "
                       "a rate — nothing to score")
        return None
    nav = _safe_float(book.state.get("nav"))
    if not nav or nav <= 0:
        logger.warning("hedged_shadow_lane: fx_trend account has no positive NAV — cannot "
                       "express fraction-of-NAV weights")
        return None

    weights: Dict[str, float] = {}
    for rec in resolved:
        signed_notional = rec.notional_home if rec.direction == "long" else -rec.notional_home
        weights[rec.instrument] = signed_notional / nav

    unrealized = _safe_float(book.state.get("unrealized_pl"))
    raw_net = (unrealized / nav) if unrealized is not None else None
    skipped = sorted(r.instrument for r in book.records if r.notional_home is None)
    return BookSnapshot(
        strategy="fx_trend", asset_class="fx",
        asof_date=str(book.mtime.date()),
        weights=weights,
        raw_net_return=raw_net, raw_gross_return=raw_net, raw_cost=None,
        return_source="own_ledger",
        meta={"source_path": str(account_state_path), "nav": nav,
              "open_trade_count": book.state.get("open_trade_count"),
              "raw_return_basis": "open_book_unrealized_pl_over_nav_point_in_time_mark",
              "notional_home_total": sum(abs(r.notional_home) for r in resolved),
              "skipped_instruments": skipped,
              "hedged_overlay": "resolves_via_daily_fx_panel_when_forward_bar_available"},
    )


def load_oanda_fx_book(journal_path: Optional[Path] = None,
                       account_state_path: Optional[Path] = None) -> Optional[BookSnapshot]:
    """The 15-agent scanner's book — scanner-attributed OPEN trades only.

    Layer-8 lane coverage (2026-07-18): sourced from the scanner's own
    ``trade_journal_rl.json`` (entries whose ``outcome`` is still None — the
    exact pending-set the RL sync itself uses), NOT from account_state.json:
    the practice account is shared with the trend lane, and attributing the
    trend lane's exposure to the scanner's agents would corrupt the residual
    credit signal this coverage exists to produce.

    Weights: signed fraction-of-NAV notional per pair, aggregated across
    same-pair entries. Units = lots x 100,000; base->USD via the pair's own
    entry price when the quote is USD (exact), 1.0 when the base is USD
    (exact), else the position-sizing module's documented static estimate
    (same approximation family the sizing path uses; noted in meta).
    NAV from the shared account state (one account, one NAV).

    Fail-closed: no journal / no open FX entries / no positive NAV / a
    malformed open entry (missing pair or non-positive lots) -> None with a
    warning — never a partial or fabricated book. ``raw_net_return`` is None
    (the journal carries no live mark); lane A is marked forward from asof by
    run_cycle_for_strategy's F1-aligned ``fx_basket_forward_return``.
    """
    from datetime import datetime, timezone

    journal_path = journal_path or SCANNER_JOURNAL_PATH
    account_state_path = account_state_path or FX_TREND_ACCOUNT_STATE_PATH
    try:
        entries = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("hedged_shadow_lane: scanner journal unreadable (%s) — no oanda_fx book", exc)
        return None
    if not isinstance(entries, list):
        logger.warning("hedged_shadow_lane: scanner journal is not a list — no oanda_fx book")
        return None

    open_fx = [
        e for e in entries
        if isinstance(e, dict) and e.get("outcome") is None
        and str(e.get("asset_class", "FX")).upper() == "FX"
    ]
    if not open_fx:
        logger.info("hedged_shadow_lane: scanner book is flat (no open FX journal entries)")
        return None

    state = {}
    try:
        state = json.loads(Path(account_state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    nav = _safe_float(state.get("nav")) if isinstance(state, dict) else None
    if not nav or nav <= 0:
        logger.warning("hedged_shadow_lane: no positive NAV for the scanner book — "
                       "cannot express fraction-of-NAV weights")
        return None

    from src.risk.position_sizing import _approx_ccy_to_usd

    weights: Dict[str, float] = {}
    approx_legs = 0
    for e in open_fx:
        pair = str(e.get("pair") or "")
        lots = _safe_float(e.get("lots"))
        direction = str(e.get("direction") or "").upper()
        parts = pair.split("_")
        if len(parts) != 2 or not lots or lots <= 0 or direction not in ("LONG", "SHORT"):
            logger.warning("hedged_shadow_lane: malformed open scanner entry "
                           "(pair=%r lots=%r direction=%r) — refusing partial book",
                           pair, e.get("lots"), direction)
            return None
        base, quote = parts
        units = lots * 100_000.0
        entry_price = _safe_float(e.get("entry_price"))
        if base == "USD":
            base_to_usd = 1.0
        elif quote == "USD" and entry_price and entry_price > 0:
            base_to_usd = entry_price
        else:
            base_to_usd = _approx_ccy_to_usd(base)
            approx_legs += 1
        signed = units * base_to_usd * (1.0 if direction == "LONG" else -1.0)
        weights[pair] = weights.get(pair, 0.0) + signed / nav

    # Same-pair netting can cancel to a flat book — nothing to score.
    weights = {p: w for p, w in weights.items() if abs(w) > 1e-12}
    if not weights:
        logger.info("hedged_shadow_lane: scanner book nets flat across open entries")
        return None

    return BookSnapshot(
        strategy="oanda_fx", asset_class="fx",
        asof_date=datetime.now(timezone.utc).date().isoformat(),
        weights=weights,
        raw_net_return=None, raw_gross_return=None, raw_cost=None,
        return_source="open_positions_journal",
        meta={
            "source": str(journal_path),
            "n_open_entries": len(open_fx),
            "n_approx_base_usd_legs": approx_legs,
            "nav": nav,
        },
    )


_LOADERS = {
    "equity_harvester": load_equity_harvester_book,
    "crypto_momentum": load_crypto_momentum_book,
    "track_b": load_track_b_book,
    "fx_trend": load_fx_trend_book,
    "oanda_fx": load_oanda_fx_book,
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- #
# Book -> Phase 1/2 exposure positions                                  #
# --------------------------------------------------------------------- #
def positions_from_weights(weights: Dict[str, float], asset_class: str,
                           notional: float = SHADOW_NOTIONAL_DEFAULT,
                           source: str = "shadow_open") -> List[ExposurePosition]:
    positions = []
    for instrument, weight in weights.items():
        if weight == 0:
            continue
        direction = "long" if weight > 0 else "short"
        positions.append(ExposurePosition(
            asset_class=asset_class, instrument=instrument, direction=direction,
            risk_home=abs(weight) * notional, source=source))
    return positions


# --------------------------------------------------------------------- #
# Cached-price forward marking. See module docstring "Stale/missing      #
# forward price data degrades honestly" and "real, cached, honestly-     #
# labeled PROXY".                                                        #
# --------------------------------------------------------------------- #
def load_equity_price_panel(panel_path: Path = EQUITY_PRICE_PANEL_PATH) -> pd.DataFrame:
    """Cached S&P constituent adjusted-close panel — same file
    ``track_b_shadow`` reads. Missing file -> empty DataFrame (fail soft;
    every downstream lookup then honestly reports insufficient data)."""
    if not panel_path.exists():
        logger.warning("hedged_shadow_lane: equity price panel missing at %s", panel_path)
        return pd.DataFrame()
    try:
        df = pd.read_parquet(panel_path)
    except (OSError, ValueError) as exc:
        logger.warning("hedged_shadow_lane: equity price panel unreadable (%s)", exc)
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _forward_bar_date(panel: pd.DataFrame, asof_date: str) -> Optional[pd.Timestamp]:
    """The first panel index strictly AFTER ``asof_date`` — the bar whose
    return marks "what happened forward from this book snapshot". ``None``
    when the panel has no bar past asof (stale cache; expected, not a bug)."""
    if panel.empty:
        return None
    asof_ts = pd.Timestamp(asof_date)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("UTC")
    later = panel.index[panel.index > asof_ts]
    return later[0] if len(later) else None


def _ticker_forward_return(panel: pd.DataFrame, ticker: str, asof_date: str) -> Optional[float]:
    if ticker not in panel.columns:
        return None
    fwd_date = _forward_bar_date(panel, asof_date)
    if fwd_date is None:
        return None
    prior = panel[ticker][panel.index <= fwd_date].dropna()
    if len(prior) < 2:
        return None
    prev_price, fwd_price = float(prior.iloc[-2]), float(prior.iloc[-1])
    if prev_price <= 0:
        return None
    ret = (fwd_price / prev_price) - 1.0
    if abs(ret) > MAX_SANE_DAILY_RETURN:
        # A real cache-quality issue this module found while building the
        # equity-harvester demo: the cached S&P panel has un-adjusted bad
        # ticks (e.g. one ticker printed +1844% in a single day — a
        # corporate-action artifact, not a real return). One outlier like
        # that dominates an equal-weight cross-sectional average. Never
        # feed a return this implausible into a proxy computation — drop it
        # (the caller's None-handling already reports "no forward price"
        # for this name), don't silently let it distort a whole basket.
        logger.warning("hedged_shadow_lane: implausible forward return for %s (%.2f) on %s — "
                       "treating as a bad tick, excluding from any average", ticker, ret, asof_date)
        return None
    return ret


def basket_forward_return(panel: pd.DataFrame, weights: Dict[str, float],
                          asof_date: str) -> Tuple[Optional[float], List[str]]:
    """Weighted forward return of a strategy's own book.

    ALL-OR-NOTHING fail-closed, matching ``build_exposure_report``'s own
    contract elsewhere in this package (a single unresolved position fails
    the whole report, never a silent partial). A weighted sum over only the
    RESOLVED legs is mathematically indistinguishable from zero-filling the
    unresolved ones' contribution (the missing weight's return is implicitly
    treated as 0) -- that is exactly the silent zero-fill this project's
    rules forbid, even though the ticker gap itself is named in a note. So:
    if ANY ticker in ``weights`` can't be marked forward, the WHOLE basket
    return is ``None`` -- never a partial number that looks complete."""
    notes: List[str] = []
    total = 0.0
    unresolved: List[str] = []
    for ticker, weight in weights.items():
        ret = _ticker_forward_return(panel, ticker, asof_date)
        if ret is None:
            unresolved.append(ticker)
            continue
        total += weight * ret
    if unresolved:
        notes.append(f"insufficient_forward_price_data:{len(unresolved)}_of_{len(weights)}_tickers_unresolved")
        notes.extend(f"no_forward_price:{t}" for t in unresolved)
        return None, notes
    return total, notes


def load_fx_price_panel(instrument: str,
                        panel_dir: Path = FX_PRICE_PANEL_DIR) -> pd.DataFrame:
    """One FX instrument's cached daily OHLCV (``{PAIR}_D.csv``), date-indexed
    UTC. Missing/unreadable file -> empty DataFrame (fail soft; the caller then
    honestly reports no forward price for this leg). No module-level cache: a
    cycle applies at most one hedge (1-2 legs), and caching would break test
    isolation (tmp CSVs) for no real gain."""
    path = panel_dir / f"{instrument}_D.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("hedged_shadow_lane: FX panel unreadable for %s (%s)", instrument, exc)
        return pd.DataFrame()
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def fx_basket_forward_return(weights: Dict[str, float], asof_date: str,
                             panel_dir: Path = FX_PRICE_PANEL_DIR
                             ) -> Tuple[Optional[float], List[str]]:
    """Weighted forward return of an FX book, mirroring ``basket_forward_return``.

    2026-07-18 (F1 fix): gives the FX ``own_ledger`` book a forward mark over
    the SAME period the hedge overlay uses, so lane A and lane B are period-
    and book-aligned. Same ALL-OR-NOTHING contract: any unresolved instrument
    fails the whole basket (silent partial == implicit zero-fill, forbidden).
    """
    notes: List[str] = []
    total = 0.0
    unresolved: List[str] = []
    for instrument, weight in weights.items():
        ret, leg_notes = fx_instrument_forward_return(instrument, asof_date, panel_dir)
        if ret is None:
            unresolved.append(instrument)
            notes.extend(leg_notes)
            continue
        total += weight * ret
    if unresolved:
        notes.append(
            f"insufficient_forward_price_data:{len(unresolved)}_of_{len(weights)}_instruments_unresolved")
        notes.extend(f"no_forward_price:{i}" for i in unresolved)
        return None, notes
    return total, notes


def fx_instrument_forward_return(instrument: str, asof_date: str,
                                 panel_dir: Path = FX_PRICE_PANEL_DIR
                                 ) -> Tuple[Optional[float], List[str]]:
    """Forward one-bar close-to-close return of an FX hedge leg, marked from
    the cached daily panel with the SAME leak-safety as
    ``_ticker_forward_return``: the forward bar is strictly AFTER ``asof_date``
    (never reads the asof bar's own future), and an implausible daily move is
    dropped as a bad tick rather than fed through.

    The cache stores ONE canonical direction per pair (e.g. ``CHF_JPY``, not
    ``JPY_CHF``); an FX hedge generator may name either. If the direct pair
    isn't cached, the INVERSE pair is used and its return exactly inverted
    (``BASE_QUOTE`` return = ``prev/fwd - 1`` on the ``QUOTE_BASE`` closes) --
    never left unresolved just because the naming direction differs.

    ``None`` (with a reason) when NEITHER direction is cached or the panel has
    no bar past asof (stale cache -- expected for a live 'today' book, not a
    bug)."""
    invert = False
    panel = load_fx_price_panel(instrument, panel_dir)
    if panel.empty:
        parts = str(instrument).split("_")
        if len(parts) == 2:
            panel = load_fx_price_panel(f"{parts[1]}_{parts[0]}", panel_dir)
            invert = True
        if panel.empty:
            return None, [f"no_fx_panel:{instrument}"]
    fwd_date = _forward_bar_date(panel, asof_date)
    if fwd_date is None:
        return None, [f"no_forward_bar:{instrument}"]
    prior = panel["close"][panel.index <= fwd_date].dropna()
    if len(prior) < 2:
        return None, [f"insufficient_history:{instrument}"]
    prev_price, fwd_price = float(prior.iloc[-2]), float(prior.iloc[-1])
    if prev_price <= 0 or fwd_price <= 0:
        return None, [f"nonpositive_price:{instrument}"]
    # Direct: fwd/prev - 1. Inverse (BASE_QUOTE from cached QUOTE_BASE closes):
    # (1/fwd)/(1/prev) - 1 == prev/fwd - 1.
    ret = (prev_price / fwd_price - 1.0) if invert else (fwd_price / prev_price - 1.0)
    if abs(ret) > MAX_SANE_DAILY_RETURN:
        logger.warning("hedged_shadow_lane: implausible FX forward return for %s (%.4f) on %s "
                       "— treating as a bad tick", instrument, ret, asof_date)
        return None, [f"implausible_return:{instrument}"]
    return ret, []


def _is_fx_instrument(instrument: str,
                      currency_bucket_map: Optional[Dict[str, dict]]) -> bool:
    """True iff ``instrument`` is an FX pair ``BASE_QUOTE`` with both legs in
    the currency bucket map. Distinguishes an FX hedge leg (e.g. ``GBP_USD``)
    from an equity market/sector hedge (``SPY``/``XLK`` -> no underscore)."""
    if not currency_bucket_map:
        return False
    parts = str(instrument).split("_")
    if len(parts) != 2:
        return False
    return all(p in currency_bucket_map for p in parts)


def market_proxy_return(panel: pd.DataFrame, asof_date: str,
                        sector_bucket_map: Dict[str, dict]) -> Tuple[Optional[float], List[str]]:
    """Cross-sectional equal-weight mean forward return of AXIOM's OWN
    curated large-cap universe (``sector_bucket_map.json`` tickers, ~35
    names this module already vouches for elsewhere via sector/beta
    tagging) — the honest proxy for SPY (not cached in this repo).

    Deliberately NOT an average over every column in the raw cached panel:
    that panel has real un-adjusted bad ticks (see ``_ticker_forward_return``
    docstring) which a diversified 600+ name equal-weight average has no
    protection against — one corrupted print swings the whole "market" by
    percentage points. The curated set is both smaller-and-vetted AND still
    outlier-guarded by ``_ticker_forward_return``, belt and suspenders."""
    fwd_date = _forward_bar_date(panel, asof_date)
    if fwd_date is None:
        return None, ["insufficient_forward_price_data"]
    universe = [t for t in sector_bucket_map if t in panel.columns]
    rets = []
    for ticker in universe:
        r = _ticker_forward_return(panel, ticker, asof_date)
        if r is not None:
            rets.append(r)
    if len(rets) < MIN_SECTOR_PROXY_TICKERS:
        return None, [f"insufficient_market_proxy_coverage:n={len(rets)}"]
    return float(sum(rets) / len(rets)), [f"market_proxy_n={len(rets)}_of_curated_universe"]


def sector_proxy_return(panel: pd.DataFrame, sector: str, asof_date: str,
                        sector_bucket_map: Dict[str, dict]) -> Tuple[Optional[float], List[str]]:
    tickers = [t for t, tag in sector_bucket_map.items() if tag.get("sector") == sector]
    rets = []
    for ticker in tickers:
        r = _ticker_forward_return(panel, ticker, asof_date)
        if r is not None:
            rets.append(r)
    if len(rets) < MIN_SECTOR_PROXY_TICKERS:
        return None, [f"insufficient_sector_proxy_coverage:{sector}:n={len(rets)}"]
    return float(sum(rets) / len(rets)), [f"sector_proxy_n={len(rets)}:{sector}"]


# --------------------------------------------------------------------- #
# Applying the P3 hedge report to a book                                #
# --------------------------------------------------------------------- #
def select_applied_hedge(hedge_action: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the ONE hedge proposal Lane B applies. Prefers ``market`` style
    (the decisive alpha-vs-beta test — "does this book beat the market");
    falls back to the top cost-ranked structurally-valid proposal
    (``hedge_action`` is already ranked by cost, per
    ``hedge_candidates.build_hedge_report``); ``None`` if every proposal is
    ``fail_closed``."""
    for proposal in hedge_action:
        if proposal.get("style") == "market" and not proposal.get("fail_closed"):
            return proposal
    for proposal in hedge_action:
        if not proposal.get("fail_closed"):
            return proposal
    return None


def hedge_overlay_pnl(applied_hedge: Optional[Dict[str, Any]], notional: float,
                      asof_date: str, panel: pd.DataFrame,
                      sector_bucket_map: Dict[str, dict], *,
                      currency_bucket_map: Optional[Dict[str, dict]] = None,
                      fx_panel_dir: Path = FX_PRICE_PANEL_DIR) -> Dict[str, Any]:
    """Dollar P&L of the applied hedge's legs, marked forward via the cached
    proxy price data. Returns gross overlay P&L always; cost only when the
    P2 model actually knows it (see module docstring).

    An FX hedge leg (``BASE_QUOTE`` with both currencies in
    ``currency_bucket_map``) is marked from the cached daily FX panel via
    ``fx_instrument_forward_return``; equity legs keep the market/sector proxy
    path. Same ALL-OR-NOTHING contract for both: any unresolved leg -> the
    whole overlay is ``None`` (never a silent partial)."""
    if applied_hedge is None:
        return {"gross_pnl_dollars": 0.0, "cost_known": True, "cost_dollars": 0.0,
                "net_pnl_dollars": 0.0, "notes": ["no_hedge_applied"]}

    total_gross = 0.0
    notes: List[str] = []
    unresolved_legs: List[str] = []
    for leg in applied_hedge.get("legs", []):
        instrument = leg["instrument"]
        sign = 1.0 if leg["direction"] == "long" else -1.0
        if _is_fx_instrument(instrument, currency_bucket_map):
            ret, leg_notes = fx_instrument_forward_return(instrument, asof_date, fx_panel_dir)
        elif instrument == MARKET_HEDGE_INSTRUMENT:
            ret, leg_notes = market_proxy_return(panel, asof_date, sector_bucket_map)
        else:
            # Any other equity leg is a sector-benchmark ticker (e.g. XLK) —
            # find its sector via the benchmark_hedge reverse-mapping.
            sector = next((tag.get("sector") for tag in sector_bucket_map.values()
                          if tag.get("benchmark_hedge") == instrument), None)
            if sector is None:
                ret, leg_notes = None, [f"unmapped_hedge_instrument:{instrument}"]
            else:
                ret, leg_notes = sector_proxy_return(panel, sector, asof_date, sector_bucket_map)
        notes.extend(f"{instrument}:{n}" for n in leg_notes)
        if ret is None:
            unresolved_legs.append(instrument)
            continue
        total_gross += sign * leg["risk_home"] * ret

    cost_known = bool(applied_hedge.get("cost_known"))
    cost_dollars = applied_hedge.get("cost_estimate_dollars")

    # ALL-OR-NOTHING fail-closed, same contract as basket_forward_return: a
    # sum over only the resolved legs is mathematically identical to
    # zero-filling the unresolved legs' contribution (a 2-leg relative-value
    # hedge with one leg unresolved would otherwise silently report the
    # OTHER leg's return alone as "the overlay P&L"). Currently only the
    # single-leg market/sector styles are ever applied by this module (see
    # select_applied_hedge), so this branch is reachable only if a future
    # caller wires relative_strength through to build_hedge_report.
    if unresolved_legs:
        notes.append(f"overlay_unresolved:{len(unresolved_legs)}_of_"
                     f"{len(applied_hedge.get('legs', []))}_legs_unresolved")
        return {"gross_pnl_dollars": None, "cost_known": cost_known, "cost_dollars": cost_dollars,
                "net_pnl_dollars": None, "notes": notes}

    gross_pnl_dollars = total_gross
    net_pnl = (
        (gross_pnl_dollars - cost_dollars)
        if (cost_known and cost_dollars is not None)
        else None
    )
    return {
        "gross_pnl_dollars": gross_pnl_dollars,
        "cost_known": cost_known,
        "cost_dollars": cost_dollars,
        "net_pnl_dollars": net_pnl,
        "notes": notes,
    }


# --------------------------------------------------------------------- #
# Orchestration — one cycle per strategy                                #
# --------------------------------------------------------------------- #
def run_cycle_for_strategy(strategy: str, *, notional: float = SHADOW_NOTIONAL_DEFAULT,
                           price_panel: Optional[pd.DataFrame] = None,
                           sector_bucket_map: Optional[Dict[str, dict]] = None,
                           currency_bucket_map: Optional[Dict[str, dict]] = None,
                           ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH,
                           decision_log_path: Path = HEDGE_DECISION_LOG_PATH,
                           persist: bool = True,
                           book: Optional[BookSnapshot] = None) -> Optional[Dict[str, Any]]:
    """Run one raw-vs-hedged cycle for ``strategy``, log both artifacts, and
    return the ledger row. ``None`` (never a fabricated row) if the
    strategy's own book/ledger has no data to run against.

    ``book`` is an injectable override (bypasses the real loader) — used by
    tests (synthetic reproduce-then-resolve fixtures, no real disk state
    required) and by the historical-window demo script (marks a REAL book
    forward through REAL cached prices at a past ``asof_date``, since the
    live cache is stale past "today"). Production callers never pass it."""
    if strategy not in _LOADERS:
        raise ValueError(f"unknown strategy: {strategy} (expected one of {STRATEGIES})")

    if book is None:
        book = _LOADERS[strategy]()
    if book is None:
        return None

    if sector_bucket_map is None:
        sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    if currency_bucket_map is None:
        currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    if price_panel is None:
        price_panel = load_equity_price_panel()

    positions = positions_from_weights(book.weights, book.asset_class, notional)
    exposure_report: ExposureReport = build_exposure_report(
        positions, sector_bucket_map=sector_bucket_map, currency_bucket_map=currency_bucket_map)

    # Book-level synthetic candidate: the hedge legs Phase 3 proposes for
    # ``market``/``sector`` styles are derived entirely from the aggregate
    # ``exposure_report`` (whole book already netted above), not from any
    # single trade — this candidate exists only to satisfy build_hedge_report's
    # asset-class routing + report-formatting contract. risk_home=0 so it can
    # never itself distort exposure_report's own already-computed nets.
    synthetic_candidate = ExposurePosition(
        asset_class=book.asset_class, instrument=f"__{strategy}_book__",
        direction="long", risk_home=0.0, source="book_level_synthetic")

    hedge_report = build_hedge_report(
        synthetic_candidate, exposure_report,
        currency_bucket_map=currency_bucket_map, sector_bucket_map=sector_bucket_map)

    applied = select_applied_hedge(hedge_report["hedge_action"])
    if book.asset_class not in ("fx", "equity"):
        hedge_status = "unsupported_asset_class"
    elif applied is None:
        hedge_status = "no_valid_hedge" if not exposure_report.fail_closed else "fail_closed"
    else:
        hedge_status = "applied"

    # --- Lane A: raw ------------------------------------------------------
    # 2026-07-18 audit fix (F1): both lanes MUST span the SAME forward period
    # on the SAME book. ``own_ledger`` previously used the ledger's realized
    # return AT asof — a D-1→D return of the PRIOR book — while the hedge
    # overlay is marked strictly forward (D→D+1) on the CURRENT book, so
    # hedged−raw reduced to an unrelated period's hedge P&L and the
    # alpha-vs-beta verdict was noise. The raw lane is now marked forward
    # from asof over the same weights the exposure report / overlay use
    # (equity via the price panel, FX via the cached daily panels, both
    # ALL-OR-NOTHING). The ledger's own figure is preserved separately in
    # the row as ``ledger_net_return`` for reference — it is no longer the
    # lane-A comparison number.
    ledger_net_return = book.raw_net_return if book.return_source == "own_ledger" else None
    if book.asset_class == "fx":
        raw_net_return, raw_notes = fx_basket_forward_return(book.weights, book.asof_date)
        if book.return_source == "own_ledger":
            raw_notes = ["aligned_forward_mark_replaces_ledger_return(F1_2026-07-18)"] + raw_notes
    elif book.asset_class == "equity":
        raw_net_return, raw_notes = basket_forward_return(price_panel, book.weights, book.asof_date)
        if book.return_source == "own_ledger":
            raw_notes = ["aligned_forward_mark_replaces_ledger_return(F1_2026-07-18)"] + raw_notes
    else:
        # Unsupported hedge asset class (e.g. crypto): no overlay can apply, so
        # hedged == raw by construction and there is nothing to align — keep
        # the ledger's own figure rather than failing it against a price panel
        # that cannot cover it.
        raw_net_return, raw_notes = book.raw_net_return, []

    # --- Lane B: raw + applied hedge --------------------------------------
    overlay = hedge_overlay_pnl(applied, notional, book.asof_date, price_panel, sector_bucket_map,
                                currency_bucket_map=currency_bucket_map)
    hedged_gross_return = None
    hedged_net_return = None
    if raw_net_return is not None and overlay["gross_pnl_dollars"] is not None:
        hedged_gross_return = raw_net_return + (overlay["gross_pnl_dollars"] / notional)
        if overlay["net_pnl_dollars"] is not None:
            hedged_net_return = raw_net_return + (overlay["net_pnl_dollars"] / notional)

    cycle_ts = datetime.now(timezone.utc).isoformat()
    row: Dict[str, Any] = {
        "cycle_ts": cycle_ts,
        "strategy": strategy,
        "asset_class": book.asset_class,
        "asof_date": book.asof_date,
        "notional": notional,
        # 2026-07-18 (F8): the book's signed fraction-of-NAV weights, persisted
        # so an unresolved row (stale price cache at cycle time) can be
        # re-marked later by ``reconcile_unresolved`` once forward bars exist.
        # Without this the ledger row is unreproducible: the raw lane's mark
        # depends on weights that lived only in memory.
        "weights": {str(k): float(v) for k, v in book.weights.items()},
        "raw": {
            "return_source": book.return_source,
            "net_return": raw_net_return,
            # F1 (2026-07-18): the ledger's own realized-at-asof figure, kept
            # for reference only — net_return above is the period-aligned
            # forward mark used for the lane A/B comparison.
            "ledger_net_return": ledger_net_return,
            "gross_return": book.raw_gross_return,
            "cost": book.raw_cost,
            "n_long": sum(1 for w in book.weights.values() if w > 0),
            "n_short": sum(1 for w in book.weights.values() if w < 0),
            "gross_leverage": sum(abs(w) for w in book.weights.values()),
            "notes": raw_notes,
        },
        "exposure": exposure_report.to_dict(),
        "hedge": {
            "status": hedge_status,
            "decision": hedge_report["decision"],
            "applied_proposal": applied,
            "overlay_gross_pnl_dollars": overlay["gross_pnl_dollars"],
            "overlay_cost_known": overlay["cost_known"],
            "overlay_cost_dollars": overlay["cost_dollars"],
            "notes": overlay["notes"],
        },
        "hedged": {
            "gross_return": hedged_gross_return,
            "net_return": hedged_net_return,
            "return_basis": (
                "net" if hedged_net_return is not None else
                "gross_cost_unknown" if hedged_gross_return is not None else
                "unresolved"
            ),
        },
        "meta": book.meta,
        "runtime_allowed": RUNTIME_ALLOWED,
        "paper_only": PAPER_ONLY,
        "human_review_required": HUMAN_REVIEW_REQUIRED,
    }

    if persist:
        _append_jsonl(ledger_path, row)
        _append_jsonl(decision_log_path, {
            "cycle_ts": cycle_ts, "strategy": strategy, "asof_date": book.asof_date,
            "hedge_report": hedge_report,
            "runtime_allowed": RUNTIME_ALLOWED, "paper_only": PAPER_ONLY,
            "human_review_required": HUMAN_REVIEW_REQUIRED,
        })
    return row


def _row_is_unresolved(row: Dict[str, Any]) -> bool:
    """True when a ledger row's twin-lane marks are incomplete: the raw lane
    never resolved, or the hedged lane is still ``unresolved`` (overlay legs
    lacked forward bars). Rows whose hedged basis is ``gross_cost_unknown``
    are NOT unresolved — the mark exists; only the cost model is honest about
    not knowing the overlay cost, and no amount of waiting changes that."""
    raw_unresolved = (row.get("raw") or {}).get("net_return") is None
    hedged_unresolved = (row.get("hedged") or {}).get("return_basis") == "unresolved"
    return raw_unresolved or hedged_unresolved


def reconcile_unresolved(ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH, *,
                         price_panel: Optional[pd.DataFrame] = None,
                         sector_bucket_map: Optional[Dict[str, dict]] = None,
                         currency_bucket_map: Optional[Dict[str, dict]] = None,
                         today: Optional[str] = None) -> Dict[str, int]:
    """Re-mark ledger rows whose lanes were unresolved at cycle time (F8,
    2026-07-18 review finding: 'snapshot at cycle time, never reconciled').

    A cycle that runs on the live 'today' book cannot mark t→t+1 forward
    returns — the forward bar doesn't exist yet — so the row is honestly
    persisted with ``None`` lanes. Before this function existed those rows
    stayed unresolved FOREVER, silently shrinking every downstream aligned
    sample (scorecard, attribution, promotion gate) and biasing it toward
    cycles that happened to run on stale asof dates. This revisits every
    unresolved row whose ``asof_date`` is strictly before ``today`` (a
    forward bar could now exist), re-marks BOTH lanes from the row's own
    persisted ``weights`` / ``hedge.applied_proposal`` / ``notional`` —
    the exact same functions the live cycle uses — and atomically rewrites
    the ledger (tmp + fsync + os.replace, same durability convention as the
    append path).

    Fail-closed properties preserved: reconciliation never FABRICATES a
    partial hedge result — a raw lane that genuinely resolves is persisted
    even when the overlay's own legs still lack forward data (real progress,
    each lane marked only from real prices), but such a row keeps
    ``return_basis == "unresolved"``, gets no ``reconciled_at`` stamp, and
    stays excluded from residual evidence until BOTH lanes resolve; rows
    persisted before weights were captured are counted
    ``skipped_no_weights`` and left untouched;
    a resolved row is never touched again (``reconciled_at`` stamps the
    revisit). Order and line count never change, and (rev 2.1/2.2, review
    fixes) every line NOT reconciled is preserved BYTE-FOR-BYTE — the file is
    read and rewritten in binary with original terminators kept, so corrupt
    lines, blank lines, undecodable bytes, CRLF endings and a missing final
    newline all survive verbatim. Only a line whose row was actually
    reconciled is re-serialized (LF-terminated, the append convention)."""
    summary = {"checked": 0, "resolved": 0, "still_unresolved": 0, "skipped_no_weights": 0}
    if not ledger_path.exists():
        return summary
    try:
        with open(ledger_path, "rb") as fh:
            raw_lines = fh.read().splitlines(keepends=True)
    except OSError as exc:
        logger.warning("hedged_shadow_lane: ledger unreadable at %s (%s)", ledger_path, exc)
        return summary
    indexed_rows: List[Tuple[int, Dict[str, Any]]] = []
    for i, line_bytes in enumerate(raw_lines):
        try:
            stripped = line_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            logger.warning("hedged_shadow_lane: preserving undecodable ledger line %d in %s "
                           "verbatim (reconcile never rewrites what it cannot read)",
                           i + 1, ledger_path)
            continue
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("hedged_shadow_lane: preserving unparseable ledger line %d in %s "
                           "verbatim (reconcile never rewrites what it cannot read)",
                           i + 1, ledger_path)
            continue
        if isinstance(parsed, dict):
            indexed_rows.append((i, parsed))
    if not indexed_rows:
        return summary
    if today is None:
        today = datetime.now(timezone.utc).date().isoformat()

    panel = price_panel
    sector_map = sector_bucket_map
    currency_map = currency_bucket_map
    modified_lines: set = set()

    for line_no, row in indexed_rows:
        if not _row_is_unresolved(row):
            continue
        asof = row.get("asof_date")
        if not asof or str(asof) >= today:
            continue  # forward bar still cannot exist — nothing to reconcile yet
        summary["checked"] += 1
        weights = row.get("weights")
        if not isinstance(weights, dict) or not weights:
            summary["skipped_no_weights"] += 1
            continue

        # Lazy-load shared inputs only once a reconcilable row exists.
        if sector_map is None:
            sector_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
        if currency_map is None:
            currency_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
        if panel is None:
            panel = load_equity_price_panel()

        asset_class = row.get("asset_class")
        notional = float(row.get("notional") or SHADOW_NOTIONAL_DEFAULT)
        raw = row.setdefault("raw", {})
        hedged = row.setdefault("hedged", {})
        hedge = row.get("hedge") or {}

        # --- Lane A: raw, re-marked from the persisted book ----------------
        raw_net = raw.get("net_return")
        if raw_net is None:
            if asset_class == "fx":
                raw_net, raw_notes = fx_basket_forward_return(weights, asof)
            elif asset_class == "equity":
                raw_net, raw_notes = basket_forward_return(panel, weights, asof)
            else:
                raw_net, raw_notes = None, [f"reconcile_unsupported_asset_class:{asset_class}"]
            if raw_net is not None:
                raw["net_return"] = raw_net
                raw["notes"] = list(raw.get("notes") or []) + \
                    ["reconciled_forward_mark(F8)"] + raw_notes
                modified_lines.add(line_no)

        # --- Lane B: overlay re-marked with the row's own applied proposal --
        if raw_net is not None and hedged.get("return_basis") in (None, "unresolved"):
            overlay = hedge_overlay_pnl(
                hedge.get("applied_proposal"), notional, asof, panel, sector_map,
                currency_bucket_map=currency_map)
            if overlay["gross_pnl_dollars"] is not None:
                hedged["gross_return"] = raw_net + (overlay["gross_pnl_dollars"] / notional)
                hedged["net_return"] = (
                    raw_net + (overlay["net_pnl_dollars"] / notional)
                    if overlay["net_pnl_dollars"] is not None else None)
                hedged["return_basis"] = (
                    "net" if hedged["net_return"] is not None else "gross_cost_unknown")
                hedge["overlay_gross_pnl_dollars"] = overlay["gross_pnl_dollars"]
                hedge["overlay_cost_known"] = overlay["cost_known"]
                hedge["overlay_cost_dollars"] = overlay["cost_dollars"]
                hedge["notes"] = list(hedge.get("notes") or []) + ["reconciled_overlay_mark(F8)"]
                modified_lines.add(line_no)

        if _row_is_unresolved(row):
            summary["still_unresolved"] += 1
        else:
            summary["resolved"] += 1
            row["reconciled_at"] = datetime.now(timezone.utc).isoformat()

    if modified_lines:
        row_by_line = dict(indexed_rows)
        tmp_path = ledger_path.with_name(ledger_path.name + ".reconcile.tmp")
        try:
            with open(tmp_path, "wb") as fh:
                for i, line_bytes in enumerate(raw_lines):
                    if i in modified_lines:
                        fh.write(json.dumps(row_by_line[i], sort_keys=True)
                                 .encode("utf-8") + b"\n")
                    else:
                        # verbatim BYTES — blank/corrupt/undecodable lines,
                        # original terminators (CRLF kept as CRLF), and a
                        # missing final newline all pass through unchanged
                        fh.write(line_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, ledger_path)
        except OSError as exc:
            logger.error("hedged_shadow_lane: reconcile rewrite failed for %s: %s",
                         ledger_path, exc, exc_info=True)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return summary


def run_all(strategies: Sequence[str] = STRATEGIES, *,
            notional: float = SHADOW_NOTIONAL_DEFAULT,
            ledger_path: Path = RAW_VS_HEDGED_LEDGER_PATH,
            decision_log_path: Path = HEDGE_DECISION_LOG_PATH,
            persist: bool = True) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run one raw-vs-hedged cycle for every covered strategy. Loads the
    price panel + config maps once and reuses across strategies."""
    price_panel = load_equity_price_panel()
    sector_bucket_map = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
    currency_bucket_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    if persist:
        # F8: before appending today's snapshots, revisit any prior rows whose
        # lanes were unresolved at their own cycle time — forward bars may
        # exist by now. Fail-soft: a reconcile failure never blocks the cycle.
        try:
            summary = reconcile_unresolved(
                ledger_path, price_panel=price_panel,
                sector_bucket_map=sector_bucket_map,
                currency_bucket_map=currency_bucket_map)
            if summary["checked"]:
                logger.info("hedged_shadow_lane: reconcile_unresolved %s", summary)
        except Exception:
            logger.exception("hedged_shadow_lane: reconcile_unresolved failed — continuing cycle")
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for strategy in strategies:
        try:
            results[strategy] = run_cycle_for_strategy(
                strategy, notional=notional, price_panel=price_panel,
                sector_bucket_map=sector_bucket_map, currency_bucket_map=currency_bucket_map,
                ledger_path=ledger_path, decision_log_path=decision_log_path, persist=persist)
        except Exception:
            logger.exception("hedged_shadow_lane: cycle failed for strategy=%s", strategy)
            results[strategy] = None
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for _strategy, _row in run_all().items():
        if _row is None:
            print(f"{_strategy}: no data — skipped")
        else:
            print(f"{_strategy}: asof={_row['asof_date']} raw_net={_row['raw']['net_return']} "
                  f"hedge_status={_row['hedge']['status']} hedged_net={_row['hedged']['net_return']}")
