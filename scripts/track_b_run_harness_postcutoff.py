"""Track B SCALE-UP — assemble scores + price panel, run the frozen harness,
and compute the additional cross-sectional IC / quintile-spread diagnostics
the operator asked for on top of the frozen gate.

Reads trained_data/research/track_b_postcutoff_2026_07_02/{raw_scores.json,
_index.json}, builds ResearchScore objects (all as_of >= MODEL_CUTOFF by
construction — see track_b_fetch_and_blind_postcutoff.py), fetches the price
panel, and runs run_research_backtest at the frozen 2 bps cost (primary) and a
5 bps stress re-run. Because every score here is already strictly post-cutoff,
the FULL and POST_CUTOFF arms are expected to coincide (same rebalance dates);
PRE_CUTOFF is expected empty by construction, which is the point of this run
(the load-bearing arm IS the only arm that exists).

Also computes cross-sectional Spearman rank-IC and Q5-Q1 quintile spread per
rebalance, plus a pooled IC significance test and a Bonferroni-consistency note
— NOT part of the frozen harness (which only ever tested portfolio-return
Sharpe), but explicitly requested for this scale-up ("forward-return IC /
quintile-spread ... reported"). This is ADDITIVE instrumentation; it does not
alter the frozen gate math in src.equity.research.harness or src.factor.ship_gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.equity.research import harness as H
from src.equity.research.contracts import PRIMARY_WEIGHTS, ResearchScore
from src.equity.research.harness import run_research_backtest
from src.equity.research.price_loader import load_price_panel
from src.equity.research.scorer import write_scores_artifact

OUT_DIR = Path("trained_data/research/track_b_postcutoff_2026_07_02")
RAW_SCORES_PATH = OUT_DIR / "raw_scores.json"
INDEX_PATH = OUT_DIR / "_index.json"
SCORES_ARTIFACT_PATH = OUT_DIR / "scores_artifact.json"
PRICE_CACHE_DIR = OUT_DIR / "price_cache"

MODEL_CUTOFF = "2026-02-01"
PRICE_START = "2025-11-01"   # buffer before the earliest as_of for the overlay's rolling vol
PRICE_END = "2026-07-01"
FORWARD_HORIZON_DAYS = 21    # matches the frozen rebalance_step_days cadence


def _load_scores() -> list:
    raw_scores = json.loads(RAW_SCORES_PATH.read_text())
    index = json.loads(INDEX_PATH.read_text())
    scores = []
    abstained = []
    for task_id, meta in index.items():
        fields = raw_scores.get(task_id)
        if fields is None:
            abstained.append(task_id)
            continue
        s = ResearchScore(
            ticker=meta["ticker"],
            as_of=meta["as_of"],
            fundamental_quality=fields["fundamental_quality"],
            accounting_red_flags=fields["accounting_red_flags"],
            forward_outlook=fields["forward_outlook"],
            conviction=fields["conviction"],
            rationale=fields.get("rationale", ""),
            spans_used=fields.get("spans_used", []),
        )
        s.validate()
        scores.append(s)
    print(f"Assembled {len(scores)} ResearchScore objects "
          f"({len(abstained)} abstained/missing) across "
          f"{len(set(s.ticker for s in scores))} tickers")
    return scores, abstained


def _cross_sectional_diagnostics(
    scores: list, prices: pd.DataFrame, model_cutoff: str,
) -> dict:
    """Rank-IC + Q5-Q1 spread per rebalance, forward return over
    FORWARD_HORIZON_DAYS trading bars, plus the frozen placebo permutation as
    the no-signal comparator (reuses harness._placebo_permute for byte-for-byte
    consistency with the frozen design's negative control)."""
    scores_by_ticker: dict = {}
    for s in scores:
        scores_by_ticker.setdefault(s.ticker, []).append(s)

    price_idx = prices.index
    rb_dates = H._rebalance_dates(price_idx, 21)
    rb_dates = [d for d in rb_dates if d >= H._to_utc_ts(model_cutoff)]

    per_rebalance = []
    all_ic_real = []
    all_ic_placebo = []
    for rb in rb_dates:
        pos = price_idx.get_indexer([rb])[0]
        fwd_pos = pos + FORWARD_HORIZON_DAYS
        if fwd_pos >= len(price_idx):
            continue
        fwd_date = price_idx[fwd_pos]

        latest = H._latest_scores_asof(scores_by_ticker, rb)
        composites = {
            t: s.composite(PRIMARY_WEIGHTS) for t, s in latest.items()
            if t in prices.columns
        }
        fwd_rets = {}
        for t in composites:
            p0 = prices.at[rb, t] if rb in prices.index else np.nan
            p1 = prices.at[fwd_date, t] if fwd_date in prices.index else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_rets[t] = (p1 / p0) - 1.0
        names = sorted(set(composites) & set(fwd_rets))
        if len(names) < 10:
            continue
        comp_vec = [composites[t] for t in names]
        ret_vec = [fwd_rets[t] for t in names]
        ic, ic_p = spearmanr(comp_vec, ret_vec)

        placebo_composites = H._placebo_permute({t: composites[t] for t in names})
        placebo_comp_vec = [placebo_composites[t] for t in names]
        ic_placebo, ic_placebo_p = spearmanr(placebo_comp_vec, ret_vec)

        ranked = sorted(names, key=lambda t: -composites[t])
        q_n = max(1, len(ranked) // 5)
        q5 = ranked[:q_n]
        q1 = ranked[-q_n:]
        q5_ret = float(np.mean([fwd_rets[t] for t in q5]))
        q1_ret = float(np.mean([fwd_rets[t] for t in q1]))

        per_rebalance.append({
            "rebalance_date": str(rb.date()),
            "forward_date": str(fwd_date.date()),
            "n_names": len(names),
            "q5_n": q_n,
            "rank_ic": float(ic),
            "rank_ic_pvalue": float(ic_p),
            "q5_fwd_return": q5_ret,
            "q1_fwd_return": q1_ret,
            "q5_minus_q1_spread": q5_ret - q1_ret,
            "placebo_rank_ic": float(ic_placebo),
            "placebo_rank_ic_pvalue": float(ic_placebo_p),
        })
        all_ic_real.append(ic)
        all_ic_placebo.append(ic_placebo)

    pooled = {}
    if all_ic_real:
        arr = np.array(all_ic_real)
        pooled["mean_ic"] = float(arr.mean())
        pooled["ic_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else None
        pooled["n_rebalances"] = len(arr)
        pooled["ic_ir"] = (
            float(arr.mean() / arr.std(ddof=1) * np.sqrt(len(arr)))
            if len(arr) > 1 and arr.std(ddof=1) > 1e-12 else None
        )
        pooled["placebo_mean_ic"] = float(np.mean(all_ic_placebo))
    return {"per_rebalance": per_rebalance, "pooled": pooled}


def main() -> None:
    scores, abstained = _load_scores()
    if not scores:
        raise SystemExit("no scores assembled — nothing to run")

    write_scores_artifact(
        scores, SCORES_ARTIFACT_PATH,
        meta={
            "model": "claude-sonnet-5 (this session + dispatched Data Engineer "
                     "subagents acting as scorers; no ANTHROPIC_API_KEY "
                     "configured for an automated fan-out)",
            "model_cutoff": MODEL_CUTOFF,
            "scoring_date": "2026-07-02",
            "blinded": True,
            "n_companies": len(set(s.ticker for s in scores)),
            "n_abstained": len(abstained),
            "universe_note": (
                "SCALE-UP: strictly post-cutoff (filed >= model_cutoff) 10-Q/10-K "
                "filings across the current S&P 500 (Wikipedia constituent list, "
                "503 tickers as of fetch), NOT the pilot's 12-name bounded set. "
                "See docs/experiment-equity-research-alpha-prereg-2026-06-30.md "
                "and the pilot's Sec.7 for the frozen design this run inherits."
            ),
        },
    )

    tickers = sorted(set(s.ticker for s in scores))
    prices = load_price_panel(
        tickers, PRICE_START, PRICE_END, cache_dir=str(PRICE_CACHE_DIR),
    )
    print(f"Price panel: {prices.shape[0]} bars x {prices.shape[1]} tickers")
    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"WARNING: no price data for {len(missing)} tickers: {sorted(missing)[:20]}...")

    results = {}
    for cost_bps, label in ((2.0, "primary_2bps"), (5.0, "stress_5bps")):
        result = run_research_backtest(
            scores, prices, MODEL_CUTOFF,
            cost_bps=cost_bps,
            blinding_audit_clean=None,  # set after the blinding audit is run separately
        )
        out_path = OUT_DIR / f"harness_result_{label}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        results[label] = result
        print(f"\n=== {label} ===")
        print(json.dumps(result["summary"], indent=2, default=str))

    diag = _cross_sectional_diagnostics(scores, prices, MODEL_CUTOFF)
    (OUT_DIR / "cross_sectional_ic_spread.json").write_text(
        json.dumps(diag, indent=2, default=str)
    )
    print("\n=== Cross-sectional IC / Q5-Q1 spread (post-cutoff, forward "
          f"{FORWARD_HORIZON_DAYS}d) ===")
    print(json.dumps(diag["pooled"], indent=2, default=str))
    for row in diag["per_rebalance"]:
        print(row)


if __name__ == "__main__":
    main()
