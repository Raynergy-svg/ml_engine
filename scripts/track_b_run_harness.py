"""Track B — assemble scores + price panel and run the four-arm harness.

Reads trained_data/research/track_b_pilot_2026_07_01/{raw_scores.json,_index.json},
builds ResearchScore objects, fetches the price panel for the pilot universe,
and runs run_research_backtest at the frozen 2 bps cost (primary) and a 5 bps
stress re-run (robustness, not a new trial per prereg §3.5).

model_cutoff = 2026-02-01: the scoring session's stated knowledge cutoff is
January 2026 (system-declared), so 2026-02-01 is the first date the post-cutoff
arm can be considered clean of pretraining hindsight about REALIZED RETURNS in
that window.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.equity.research.contracts import ResearchScore
from src.equity.research.harness import run_research_backtest
from src.equity.research.price_loader import load_price_panel
from src.equity.research.scorer import write_scores_artifact

OUT_DIR = Path("trained_data/research/track_b_pilot_2026_07_01")
RAW_SCORES_PATH = OUT_DIR / "raw_scores.json"
INDEX_PATH = OUT_DIR / "_index.json"
SCORES_ARTIFACT_PATH = OUT_DIR / "scores_artifact.json"
PRICE_CACHE_DIR = OUT_DIR / "price_cache"

MODEL_CUTOFF = "2026-02-01"
PRICE_START = "2023-06-01"   # buffer before the earliest as_of for the overlay's rolling vol
PRICE_END = "2026-06-30"


def main() -> None:
    raw_scores = json.loads(RAW_SCORES_PATH.read_text())
    index = json.loads(INDEX_PATH.read_text())

    scores = []
    for task_id, fields in raw_scores.items():
        meta = index[task_id]
        s = ResearchScore(
            ticker=meta["ticker"],
            as_of=meta["as_of"],
            fundamental_quality=fields["fundamental_quality"],
            accounting_red_flags=fields["accounting_red_flags"],
            forward_outlook=fields["forward_outlook"],
            conviction=fields["conviction"],
            rationale=fields["rationale"],
            spans_used=fields["spans_used"],
        )
        s.validate()
        scores.append(s)
    print(f"Assembled {len(scores)} ResearchScore objects across "
          f"{len(set(s.ticker for s in scores))} tickers")

    write_scores_artifact(
        scores, SCORES_ARTIFACT_PATH,
        meta={
            "model": "claude-sonnet-5 (this session, acting as scorer; no "
                     "ANTHROPIC_API_KEY configured for an automated fan-out)",
            "model_cutoff": MODEL_CUTOFF,
            "scoring_date": "2026-07-01",
            "blinded": True,
            "n_companies": len(set(s.ticker for s in scores)),
            "universe_note": (
                "BOUNDED PILOT (12 mega-cap names, 3 annual 10-Ks each) -- "
                "NOT the frozen full S&P500 PIT universe; see "
                "docs/experiment-equity-research-alpha-prereg-2026-06-30.md "
                "Sec.7 for the scale-down rationale."
            ),
        },
    )

    tickers = sorted(set(s.ticker for s in scores))
    prices = load_price_panel(
        tickers, PRICE_START, PRICE_END, cache_dir=str(PRICE_CACHE_DIR),
    )
    print(f"Price panel: {prices.shape[0]} bars x {prices.shape[1]} tickers "
          f"({list(prices.columns)})")
    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"WARNING: no price data for {missing}")

    for cost_bps, label in ((2.0, "primary_2bps"), (5.0, "stress_5bps")):
        result = run_research_backtest(
            scores, prices, MODEL_CUTOFF,
            cost_bps=cost_bps,
            blinding_audit_clean=False,  # from blinding_audit.json (2026-07-01)
        )
        out_path = OUT_DIR / f"harness_result_{label}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n=== {label} ===")
        print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
