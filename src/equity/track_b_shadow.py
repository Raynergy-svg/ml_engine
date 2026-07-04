"""Track B (SEC filing-text research-alpha) SHADOW forward-tracking (research-
adjacent, NOT execution).

Runs the pre-registered agentic filing-text factor FORWARD in shadow (paper)
mode so a genuine, un-p-hackable live out-of-sample track record accumulates
for the campaign's last untested equity lever: an LLM research read of 10-K/
10-Q primary-document text, composited into fundamental_quality (+0.5) minus
accounting_red_flags (-0.5) (forward_outlook logged, zero-weighted), formed
into a long-only Q5 (top-quintile) equal-weight cross-sectional book — see
``docs/experiment-equity-research-alpha-prereg-2026-06-30.md`` (the frozen
pre-registration) and ``docs/track-b-postcutoff-scaleup-2026-07-02.md`` (the
2026-07-02 scale-up run).

The signal's frozen ``overall_verdict`` is **INSUFFICIENT** — and as of the
2026-07-04 N-scale-up, that is a well-POWERED null, not an underpowered "can't
tell." The post-cutoff scored universe was pushed from N=48 to **N=420** (300
pre-registered + 120 content-blind extension; 419 priced), which was enough to
answer the load-bearing question: the +0.16 rank-IC point estimate **FADES**
toward zero as N grows — +0.160 (n=26) -> +0.123 (n=202) -> +0.091 (n=291),
never significant (best p=0.08 at N=300, 0.12 at N=420), with a clean IC-placebo
throughout. The single valid cross-section is now n=291 — ample power for a true
0.16 — and the effect is +0.09 non-significant. The binding constraint is now
the number of rebalance DATES (still only one qualifies in the ~5-month
post-cutoff price window), not cross-sectional N; only forward time can add
rebalances. See ``docs/track-b-postcutoff-scaleup-2026-07-02.md`` (2026-07-04
update) and ``docs/adversarial-review-no-edge-verdicts-2026-07-02.md``. This
module's entire purpose is to let a genuine live forward record — which no
backtest or power calculation can substitute for — accumulate against this
specific frozen construction.

HARD LINE — everything below is SHADOW ONLY:
  * Zero orders. Zero broker/execution client. No import of any broker/
    execution module (verified: no ``src.brokers`` / ``execution.py`` /
    ``place_*_order`` reference anywhere in this file).
  * The construction is REUSED VERBATIM from the frozen, pre-registered,
    verifier-confirmed research harness (``src.equity.research.harness`` /
    ``.contracts``) — composite weights, quintile cut, overlay knobs,
    rebalance cadence, and cost are all IMPORTED, never redefined or tuned
    here. This module adds only the "read today's book off the frozen
    construction and mark it forward" seam the harness (an offline evaluator)
    doesn't itself provide.
  * Real-money / live-arming is gated the same way the equity harvester and
    the crypto momentum lane are gated (``src.equity.track_b_live_gate``
    mirrors ``src.equity.live_gate`` exactly) — but nothing in this module or
    its caller ever calls ``LiveGate.arm()``. No ``trained_data/equity/
    track_b/SHIP_GATE.json`` exists (and none should be fabricated to force
    an arm): the frozen harness's own ``overall_verdict`` for this signal is
    INSUFFICIENT, never REAL.

Score coverage is a STATIC, EXPLICIT registry (:data:`SCORE_ARTIFACT_PATHS`),
not an auto-discovered directory scan — scoring new filings is an explicit,
separate, currently-manual step (in-session LLM batches; full autonomy needs
a wired ``ANTHROPIC_API_KEY`` — see ``scripts/track_b_run_harness_postcutoff.py``
and ``scripts/track_b_collect_scores.py``). This module never fakes a score:
a filing with no artifact entry is simply absent from the cross-section, the
same "abstain, never zero-fill" contract as the pipeline's own scorer.

Price data cadence caveat (honest, not hidden): this module reads the cached
S&P 500 adjusted-close panel at ``market_data/equity/sp500_prices.parquet``
(built for the PIT fundamentals work, last refreshed 2026-06-24 as of writing)
rather than hitting Yahoo live on every cycle — matching the crypto momentum
shadow lane's "cached data source, not a live feed" precedent. Pass
``refresh_prices=True`` to instead pull a fresh panel via
``src.equity.research.price_loader.load_price_panel`` for the held tickers
(network-dependent, best-effort).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.equity.backtest import run_portfolio_backtest              # noqa: E402
from src.equity.research import harness as H                        # noqa: E402  frozen harness (verifier-confirmed)
from src.equity.research.contracts import (                          # noqa: E402
    PRIMARY_WEIGHTS,
    ResearchScore,
)
import track_b_run_harness_postcutoff as _tb_postcutoff              # noqa: E402  frozen scale-up driver (constants only; guarded by __main__)

# --------------------------------------------------------------------------- #
# Frozen construction — imported constants ONLY. Never redefine/tune here.
# Weights/quintile/overlay come from the pre-registered harness (contracts.py
# / harness.py); cost/vol-target/rebalance cadence are read off
# run_research_backtest's own defaults via introspection so this can never
# silently drift from the frozen function signature.
# --------------------------------------------------------------------------- #
_RRB_PARAMS = inspect.signature(H.run_research_backtest).parameters
COST_BPS: float = float(_RRB_PARAMS["cost_bps"].default)                  # 2.0 (primary; matches harness_result_primary_2bps.json)
VOL_TARGET: float = float(_RRB_PARAMS["vol_target"].default)              # 0.10
REBALANCE_STEP_DAYS: int = int(_RRB_PARAMS["rebalance_step_days"].default)  # 21 (~monthly)
COMPOSITE_WEIGHTS: Dict[str, float] = dict(PRIMARY_WEIGHTS)
QUINTILE: int = H.QUINTILE                                                 # 5 (top 20%)
MIN_NAMES_FOR_QUINTILE: int = H.MIN_NAMES_FOR_QUINTILE                     # 5
OVERLAY_DD_SOFT: float = H.OVERLAY_DD_SOFT                                 # 0.10
OVERLAY_DD_HARD: float = H.OVERLAY_DD_HARD                                 # 0.20
OVERLAY_MAX_LEV: float = H.OVERLAY_MAX_LEV                                 # 1.0
MODEL_CUTOFF: str = _tb_postcutoff.MODEL_CUTOFF                            # "2026-02-01"
PRICE_START: str = _tb_postcutoff.PRICE_START                              # "2025-11-01" (overlay rolling-window buffer)

SOURCE_PREREG_DOC = "docs/experiment-equity-research-alpha-prereg-2026-06-30.md"
SOURCE_SCALEUP_DOC = "docs/track-b-postcutoff-scaleup-2026-07-02.md"
SOURCE_ADVERSARIAL_REVIEW_DOC = "docs/adversarial-review-no-edge-verdicts-2026-07-02.md"
GATE_VERDICT = (
    "overall_verdict=INSUFFICIENT (post-cutoff full-arm never clears the gate's "
    "history-length criterion; frozen harness result at "
    "trained_data/research/track_b_postcutoff_2026_07_02/harness_result_primary_2bps.json). "
    "Cross-sectional rank-IC point estimate +0.16 (p=0.43, n=26) is NOT significant "
    "at current N — self-labeled INSUFFICIENT, NOT a measured NO EDGE. See "
    + SOURCE_ADVERSARIAL_REVIEW_DOC
)
POWER_REQUIRED_N_FILINGS = 405  # Fisher-z, 80% power, rank-IC=0.16, Bonferroni alpha=0.05/3 — see SOURCE_ADVERSARIAL_REVIEW_DOC

# Explicit, static registry of known scored-filing batches. Adding a new batch
# (after running a fresh, non-rate-limited scoring pass — see module
# docstring) means appending its scores_artifact.json path here; this module
# never auto-discovers or fabricates scores.
SCORE_ARTIFACT_PATHS: List[Path] = [
    REPO_ROOT / "trained_data" / "research" / "track_b_pilot_2026_07_01" / "scores_artifact.json",
    REPO_ROOT / "trained_data" / "research" / "track_b_postcutoff_2026_07_02" / "scores_artifact.json",
]

CACHED_PRICE_PANEL_PATH = REPO_ROOT / "market_data" / "equity" / "sp500_prices.parquet"
LEDGER_PATH_DEFAULT = REPO_ROOT / "trained_data" / "research" / "track_b_shadow_ledger.jsonl"


class TrackBShadowError(RuntimeError):
    """Raised when shadow-cycle inputs are missing or mis-shaped."""


def load_frozen_scores(paths: Optional[List[Path]] = None) -> List[ResearchScore]:
    """Load + combine every known scored batch, keeping ONLY strictly
    post-cutoff (``as_of >= MODEL_CUTOFF``) scores.

    Pre-cutoff scores are excluded on purpose: they are the exact
    contamination arm (``ARM_PRE_CUTOFF``) the pre-registration exists to rule
    out — a live shadow book must never be built from a score the scoring
    model could have derived from memorized training data about that
    company's actual subsequent outcome. Deduplicates by ``(ticker, as_of)``
    in case a future batch re-scores an existing filing.
    """
    paths = paths if paths is not None else SCORE_ARTIFACT_PATHS
    seen: Dict[tuple, ResearchScore] = {}
    for path in paths:
        if not path.exists():
            logger.warning("track_b shadow: scores artifact missing at %s — skipping", path)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("track_b shadow: unreadable scores artifact %s: %s", path, exc)
            continue
        for raw in payload.get("scores", []):
            if raw.get("as_of", "") < MODEL_CUTOFF:
                continue  # pre-cutoff — contamination arm, never used for the live book
            score = ResearchScore(
                ticker=raw["ticker"],
                as_of=raw["as_of"],
                fundamental_quality=raw["fundamental_quality"],
                accounting_red_flags=raw["accounting_red_flags"],
                forward_outlook=raw["forward_outlook"],
                conviction=raw["conviction"],
                rationale=raw.get("rationale", ""),
                spans_used=raw.get("spans_used", []),
            )
            score.validate()
            seen[(score.ticker, score.as_of)] = score  # last-write-wins on true dupes
    return list(seen.values())


def construction_manifest(n_scored: int) -> Dict[str, Any]:
    """The frozen construction, exactly as reused — for ledger rows + AXIOM."""
    return {
        "signal": "sec_edgar_filing_text_llm_composite",
        "composite_weights": COMPOSITE_WEIGHTS,
        "book": "Q5 (top 20%) equal-weight, LONG-ONLY",
        "quintile_divisor": QUINTILE,
        "min_names_for_quintile": MIN_NAMES_FOR_QUINTILE,
        "rebalance_step_days": REBALANCE_STEP_DAYS,
        "vol_target_ann": VOL_TARGET,
        "dd_soft": OVERLAY_DD_SOFT,
        "dd_hard": OVERLAY_DD_HARD,
        "max_leverage": OVERLAY_MAX_LEV,
        "cost_bps": COST_BPS,
        "model_cutoff": MODEL_CUTOFF,
        "n_scored_filings": n_scored,
        "power_required_n_filings": POWER_REQUIRED_N_FILINGS,
        "source_prereg_doc": SOURCE_PREREG_DOC,
        "source_scaleup_doc": SOURCE_SCALEUP_DOC,
        "source_adversarial_review_doc": SOURCE_ADVERSARIAL_REVIEW_DOC,
        "gate_verdict": GATE_VERDICT,
        "coverage_note": (
            f"{n_scored} post-cutoff scored filings. As of the 2026-07-04 "
            f"scale-up (N=420 scored, 419 priced) the cross-sectional-N power "
            f"target (~{POWER_REQUIRED_N_FILINGS}) is MET — and the answer is a "
            "well-powered NULL: the +0.16 rank-IC FADED to +0.09 (n=291, "
            "non-significant) as N grew, clean placebo throughout. The frozen "
            "gate stays INSUFFICIENT. The remaining power gap is now TEMPORAL "
            "(only one valid rebalance date exists in the ~5-month post-cutoff "
            "window; pooled IC significance is uncomputable at n_rebalances=1) — "
            "only forward shadow time can close it, which is exactly what this "
            "lane accumulates. New filings are scored via an explicit, separate, "
            "currently-manual step (in-session blinded LLM batches). This lane "
            "exists to accumulate a genuine forward record on the frozen "
            "construction, not because the signal is verified."
        ),
    }


def _load_cached_price_panel(tickers: List[str]) -> pd.DataFrame:
    if not CACHED_PRICE_PANEL_PATH.exists():
        raise TrackBShadowError(f"cached price panel missing at {CACHED_PRICE_PANEL_PATH}")
    df = pd.read_parquet(CACHED_PRICE_PANEL_PATH)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    cols = [t for t in tickers if t in df.columns]
    missing = sorted(set(tickers) - set(cols))
    if missing:
        logger.warning("track_b shadow: no cached price data for %d tickers: %s", len(missing), missing)
    return df[cols].sort_index()


def _load_refreshed_price_panel(tickers: List[str]) -> pd.DataFrame:
    from src.equity.research.price_loader import load_price_panel

    today = pd.Timestamp.now(tz="UTC").date().isoformat()
    return load_price_panel(tickers, PRICE_START, today)


@dataclass
class ShadowCycleResult:
    asof_date: str
    n_scored_filings: int
    universe_size: int
    longs: Dict[str, float]
    gross_leverage: float
    today_net_return: float
    today_gross_return: float
    today_cost: float
    today_turnover: float
    construction: Dict[str, Any] = field(default_factory=dict)


def compute_shadow_cycle(refresh_prices: bool = False) -> ShadowCycleResult:
    """Form the frozen-construction Q5 book from every known post-cutoff
    score, mark it forward on real prices, and return today's would-be
    positions + today's realized (mark-to-market) return.
    """
    scores = load_frozen_scores()
    if not scores:
        raise TrackBShadowError("track_b shadow: no post-cutoff scores available — nothing to form a book from")

    tickers = sorted({s.ticker for s in scores})
    prices = (
        _load_refreshed_price_panel(tickers) if refresh_prices else _load_cached_price_panel(tickers)
    )
    prices = prices.loc[prices.index >= pd.Timestamp(PRICE_START, tz="UTC")]
    if prices.empty:
        raise TrackBShadowError("track_b shadow: empty price panel — no price data for the scored universe")

    # Price-coverage exclusion (pre-reg §2 + the frozen backtest's no-fill
    # contract): the backtest refuses to forward-fill ANY NaN in the panel, so
    # a ticker whose cached/fetched series has a missing bar in the eval window
    # is DROPPED here (both its column and its score) — logged, abstain, never
    # forward-filled, never dropped for the value of its score (content-blind
    # data hygiene). Mirrors the harness driver's identical exclusion so the
    # shadow book is formed on the same priced universe the offline gate uses.
    nan_cols = sorted(prices.columns[prices.isna().any()].tolist())
    if nan_cols:
        logger.warning(
            "track_b shadow: %d tickers with NaN price bars excluded "
            "(abstain, not forward-filled): %s", len(nan_cols), nan_cols,
        )
        prices = prices.drop(columns=nan_cols)
        scores = [s for s in scores if s.ticker in prices.columns]
        if not scores or prices.empty:
            raise TrackBShadowError(
                "track_b shadow: no priced scores remain after NaN-coverage exclusion"
            )

    scores_by_ticker: Dict[str, List[ResearchScore]] = {}
    for s in scores:
        scores_by_ticker.setdefault(s.ticker, []).append(s)

    rebalance_dates = H._rebalance_dates(prices.index, REBALANCE_STEP_DAYS)
    rebalance_dates = [d for d in rebalance_dates if d >= H._to_utc_ts(MODEL_CUTOFF)]
    if not rebalance_dates:
        raise TrackBShadowError("track_b shadow: no post-cutoff rebalance dates in the price index")

    panel = H._build_weight_panel(
        scores_by_ticker, prices, rebalance_dates, COMPOSITE_WEIGHTS,
        long_only=True, placebo=False,
    )
    managed = H._apply_overlay(panel, prices, vol_target=VOL_TARGET)
    bt = run_portfolio_backtest(managed, prices, cost_bps=COST_BPS, execution_lag=1)

    if bt.returns.empty:
        raise TrackBShadowError("track_b shadow: backtest produced no realized return bars")
    asof = bt.returns.index[-1]
    row = managed.loc[asof] if asof in managed.index else managed.iloc[-1]
    longs = {str(t): float(w) for t, w in row.items() if w > 1e-12}

    return ShadowCycleResult(
        asof_date=str(pd.Timestamp(asof).date()),
        n_scored_filings=len(scores),
        universe_size=len(tickers),
        longs=longs,
        gross_leverage=float(row.abs().sum()),
        today_net_return=float(bt.returns.loc[asof]),
        today_gross_return=float(bt.gross_returns.loc[asof]),
        today_cost=float(bt.costs.loc[asof]),
        today_turnover=float(bt.turnover.loc[asof]),
        construction=construction_manifest(len(scores)),
    )


# --------------------------------------------------------------------------- #
# Ledger — append-only JSONL, atomic append (matches
# src/crypto/momentum_shadow.py's ledger convention exactly).
# --------------------------------------------------------------------------- #
def _read_ledger_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("track_b shadow ledger unreadable at %s: %s", path, exc)
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
        logger.error("track_b shadow ledger append failed for %s: %s", path, exc, exc_info=True)
        raise


def record_shadow_cycle(
    result: ShadowCycleResult,
    *,
    ledger_path: Path = LEDGER_PATH_DEFAULT,
    cycle_ts_iso: str,
) -> Optional[Dict[str, Any]]:
    """Append one shadow-cycle row to the forward-OOS ledger.

    Returns the appended row, or None if this cycle's ``asof_date`` matches
    the ledger's last entry (no new trading day since the last cycle — an
    honest no-op, not an error). ``cumulative_shadow_return`` compounds ONLY
    the rows already in this ledger (true forward-from-activation P&L).
    """
    rows = _read_ledger_rows(ledger_path)
    if rows and rows[-1].get("asof_date") == result.asof_date:
        logger.info(
            "track_b shadow: no new trading day since last cycle "
            "(asof_date=%s unchanged) — skipping duplicate ledger row",
            result.asof_date,
        )
        return None

    prior_cum = float(rows[-1]["cumulative_shadow_return"]) if rows else 0.0
    cum = (1.0 + prior_cum) * (1.0 + result.today_net_return) - 1.0

    row = {
        "cycle_ts": cycle_ts_iso,
        "asof_date": result.asof_date,
        "n_scored_filings": result.n_scored_filings,
        "universe_size": result.universe_size,
        "book": {"longs": result.longs, "shorts": {}},
        "n_longs": len(result.longs),
        "n_shorts": 0,
        "gross_leverage": result.gross_leverage,
        "today_net_return": result.today_net_return,
        "today_gross_return": result.today_gross_return,
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
        # ~252 trading days/yr (equities, not crypto's 365) — matches
        # src.equity.backtest.stats' own ANN constant.
        ann = 252.0
        forward_sharpe = (mean / std) * (ann ** 0.5) if std > 0 else None
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
    "COMPOSITE_WEIGHTS", "COST_BPS", "GATE_VERDICT", "LEDGER_PATH_DEFAULT",
    "MIN_NAMES_FOR_QUINTILE", "MODEL_CUTOFF", "OVERLAY_DD_HARD",
    "OVERLAY_DD_SOFT", "OVERLAY_MAX_LEV", "POWER_REQUIRED_N_FILINGS",
    "PRICE_START", "QUINTILE", "REBALANCE_STEP_DAYS", "SCORE_ARTIFACT_PATHS",
    "ShadowCycleResult", "TrackBShadowError", "compute_shadow_cycle",
    "construction_manifest", "forward_oos_summary", "load_frozen_scores",
    "record_shadow_cycle",
]
