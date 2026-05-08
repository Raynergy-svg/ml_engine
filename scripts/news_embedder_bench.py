"""Benchmark FinBERTEmbedder on M1 (CPU vs MPS) + PCA n_components decision.

Run: python scripts/news_embedder_bench.py

Outputs:
  - Latency (ms) for 32 sample finance headlines on cpu, then mps if available
  - Throughput (headlines/sec) on the better device
  - PCA explained-variance-ratio cumulative for k in {16, 32, 64} on 500
    synthesized financial headlines
  - One-line PCA recommendation

No mocks. Real model load. Per design doc §2 + .claude/rules/improvement.md.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.data.news import FinBERTEmbedder, NewsEvent


# ---------------------------------------------------------------------------
# Sample text generators — realistic finance headlines, NOT lorem ipsum
# ---------------------------------------------------------------------------

REAL_32_HEADLINES = [
    "Federal Reserve raises interest rates by 25 basis points",
    "ECB signals dovish pivot as inflation cools to 2.1 percent",
    "Non-Farm Payrolls beat expectations at 270k vs 200k forecast",
    "Bank of Japan holds rates steady, yen falls to 24-year low",
    "US CPI prints at 3.2 percent, slightly below consensus",
    "UK GDP contracts 0.3 percent in Q1, recession fears grow",
    "Oil prices spike on OPEC production cut announcement",
    "Gold rallies to record high as dollar weakens",
    "Powell strikes hawkish tone in Jackson Hole speech",
    "Lagarde warns of persistent inflation in eurozone",
    "Yields on 10-year Treasury climb above 4.5 percent",
    "Swiss National Bank surprises with 50bp rate cut",
    "Bank of Canada pauses tightening cycle amid mixed data",
    "Reserve Bank of Australia holds rates at decade high",
    "China PMI falls to 48.7, signaling contraction",
    "EUR_USD breaks below 1.05 on weak German factory orders",
    "USD_JPY tests 150 as BoJ maintains ultra-loose policy",
    "GBP_USD rallies on BoE hawkish hold and stronger CPI",
    "Risk-on sentiment lifts AUD and NZD on China stimulus",
    "Crude oil drops 3 percent on demand concerns",
    "FOMC minutes reveal divided opinion on rate path",
    "Eurozone retail sales fall for third consecutive month",
    "Japanese inflation hits 33-year high of 4.3 percent",
    "Canadian jobs report disappoints, loonie tumbles",
    "Australian wage growth accelerates to 4.0 percent",
    "Swiss CPI surprises higher at 1.7 percent year-on-year",
    "Trump tariff threat rattles emerging market currencies",
    "Eurozone PMI rises to 51.2, signaling expansion",
    "BoE Bailey hints at June rate cut if inflation cooperates",
    "Dollar index rebounds after Fed pushes back on cuts",
    "Swiss franc strengthens as risk-off flows return",
    "Mexican peso slides on central bank dovish surprise",
]


def _synthesize_500_headlines() -> List[str]:
    """Build 500 realistic finance headlines for PCA fit.

    Strategy: combinatorial (entity x action x metric x direction) over
    finance-domain vocabulary. Real text shape, not lorem ipsum.
    """
    entities = [
        "Fed", "ECB", "BoE", "BoJ", "SNB", "BoC", "RBA", "RBNZ", "PBoC",
        "Treasury", "OPEC", "IMF",
    ]
    actions = [
        "raises", "cuts", "holds", "signals", "warns of", "hints at",
        "surprises with", "delays", "accelerates", "pauses",
    ]
    metrics = [
        "rates 25bp", "rates 50bp", "QE expansion", "balance sheet runoff",
        "inflation outlook", "growth forecast", "employment target",
        "yield curve", "currency intervention", "policy stance",
    ]
    directions = [
        "amid persistent inflation", "as growth slows", "on hawkish data",
        "in dovish pivot", "after weak jobs report", "before key CPI print",
        "with recession risk rising", "as risk appetite returns",
        "despite market volatility", "on stronger dollar",
    ]
    pairs_actions = [
        "EUR_USD breaks below 1.05", "USD_JPY tests 150 resistance",
        "GBP_USD rallies on hawkish BoE", "AUD_USD slides on weak China data",
        "USD_CAD climbs above 1.40", "NZD_USD holds 0.59 support",
        "USD_CHF rejected at 0.92", "EUR_GBP probes 0.85",
    ]
    contexts = [
        "ahead of NFP", "post-FOMC", "after CPI release", "on intervention fears",
        "during Asian session", "into London fix", "post-ECB presser",
        "ahead of BoE decision",
    ]

    out: List[str] = []
    rng = np.random.default_rng(42)
    # 400 macro/policy combos
    for _ in range(400):
        e = entities[rng.integers(len(entities))]
        a = actions[rng.integers(len(actions))]
        m = metrics[rng.integers(len(metrics))]
        d = directions[rng.integers(len(directions))]
        out.append(f"{e} {a} {m} {d}")
    # 100 price-action combos
    for _ in range(100):
        p = pairs_actions[rng.integers(len(pairs_actions))]
        c = contexts[rng.integers(len(contexts))]
        out.append(f"{p} {c}")
    return out


def _make_events(texts: List[str]) -> List[NewsEvent]:
    base_ts = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    return [
        NewsEvent(
            timestamp=base_ts,
            text=t,
            source="manual",
            category="OTHER",
            relevance_score=1.0,
            pair="EUR_USD",
        )
        for t in texts
    ]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _bench_device(device: str, events: List[NewsEvent]) -> dict:
    emb = FinBERTEmbedder(device=device, batch_size=32, max_length=128)
    # Warmup (load + first-batch JIT cost) — do NOT count this.
    _ = emb.embed(events[:4])

    t0 = time.perf_counter()
    out = emb.embed(events)
    dt = time.perf_counter() - t0

    return {
        "device_requested": device,
        "device_resolved": getattr(emb, "_resolved_device", "?"),
        "n_headlines": len(events),
        "latency_ms_total": dt * 1000.0,
        "latency_ms_per_headline": (dt * 1000.0) / max(1, len(events)),
        "throughput_per_sec": len(events) / max(dt, 1e-9),
        "shape": out.shape,
        "dtype": str(out.dtype),
    }


def main() -> None:
    print("=" * 72)
    print("FinBERT embedder benchmark + PCA decision (Stage 4-A)")
    print("=" * 72)

    sample_events = _make_events(REAL_32_HEADLINES)
    print(f"\nSample size: {len(sample_events)} real finance headlines\n")

    # ---- CPU bench
    print("--- CPU benchmark ---")
    cpu_stats = _bench_device("cpu", sample_events)
    for k, v in cpu_stats.items():
        print(f"  {k}: {v}")

    # ---- MPS bench (if available)
    import torch

    mps_stats = None
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("\n--- MPS (Metal) benchmark ---")
        mps_stats = _bench_device("mps", sample_events)
        for k, v in mps_stats.items():
            print(f"  {k}: {v}")
    else:
        print("\n--- MPS unavailable; skipping ---")

    # Decide which device's number to publish
    best = mps_stats if mps_stats is not None else cpu_stats
    print(
        f"\nDevice picked: {best['device_resolved']} | "
        f"throughput: {best['throughput_per_sec']:.1f} headlines/sec | "
        f"per-headline: {best['latency_ms_per_headline']:.2f} ms"
    )

    # ---- PCA decision
    print("\n" + "=" * 72)
    print("PCA n_components decision (500 synthesized financial headlines)")
    print("=" * 72)

    pca_texts = _synthesize_500_headlines()
    pca_events = _make_events(pca_texts)
    pca_emb = FinBERTEmbedder(
        device=best["device_resolved"], batch_size=32, max_length=128
    )
    print(f"Embedding {len(pca_events)} headlines on {best['device_resolved']}...")
    t0 = time.perf_counter()
    X = pca_emb.embed(pca_events)
    dt_embed = time.perf_counter() - t0
    print(
        f"  embedded in {dt_embed:.1f}s | "
        f"shape={X.shape} | dtype={X.dtype}"
    )

    from sklearn.decomposition import PCA

    print("\nExplained variance ratio (cumulative):")
    print(f"  {'k':>4} | {'cum_explained_variance':>22} | {'compression_ratio':>17}")
    print(f"  {'-'*4} + {'-'*22} + {'-'*17}")
    results: dict = {}
    for k in (16, 32, 64):
        pca = PCA(n_components=k, random_state=42)
        pca.fit(X)
        cum = float(np.cumsum(pca.explained_variance_ratio_)[-1])
        ratio = 768 / k
        results[k] = cum
        print(f"  {k:>4} | {cum:>22.4f} | {ratio:>16.1f}x")

    print("\nDecision rule (per design doc §2 + §4):")
    if results[32] >= 0.95:
        rec = 32
        why = f"k=32 retains {results[32]*100:.1f}% variance (>=95% target); 24x compression."
    elif results[64] >= 0.95:
        rec = 64
        why = f"k=32 below 95% ({results[32]*100:.1f}%); k=64 retains {results[64]*100:.1f}%."
    else:
        rec = 64
        why = f"Neither k hits 95%; k=64 wins as best variance retained ({results[64]*100:.1f}%)."

    print(f"\n>>> RECOMMENDATION: PCA n_components = {rec}")
    print(f">>> JUSTIFICATION:   {why}")


if __name__ == "__main__":
    main()
