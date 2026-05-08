# Phase 5.C — News Fusion First Holdout Result (EUR_USD H1)

> **Status**: Stage 4-A end-to-end pipeline VERIFIED working. Holdout at full
> statistical power (n=1500) shows **no lift** over price-only baseline.
> Training-time gain (+6pp val_balanced) was overfitting on news PCA features.
>
> **Bot stays halted.** Operator decides whether to iterate news config or
> pivot to a different lever.

---

## 1. The numbers (EUR_USD H1, lookahead=24, n=1500 windows)

| Model | Train (val_acc) | Train (val_bal) | Holdout (n=1500) |
|---|---:|---:|---:|
| Price-only (Phase 5.B baseline) | 51.29% | 48.97% | **51.3%** |
| News-fused (Stage 4-A, FinBERT+PCA(64)+24h lookback) | 55.63% | 54.96% | **51.3%** |
| **Lift from news at holdout** | — | — | **0.0pp** |

The first n=50 holdout showed 60.0% (+8.7pp lift) but that was a lucky
test slice; at n=1500 the apparent lift evaporates. Statistical power matters.

## 2. Why training showed gain but holdout didn't (textbook overfitting)

RF feature selector inspection of the saved news model
(`trained_data/models/EUR_USD_news_first_run/transformer_direction.meta.pkl`):

| Feature class | Selected | Total | Selection rate |
|---|---:|---:|---:|
| News PCA (pca_0..63) | 29 | 64 | 45% |
| News event-class counts (ec_0..7) | 1 | 8 | 12% |
| Regime one-hots | 2 | 4 | 50% |
| Price features | 18 | 60 | 30% |
| **Total** | 50 | 136 | — |

RF chose news features more heavily than price features. They had high
*training-time* importance — but the importance came from spurious
correlations between PCA-compressed FinBERT embeddings and training-period
price labels. Those correlations don't transfer to out-of-sample data.

This is the classic high-dimensional-feature failure mode: 64 news PCA
cols + 8 ec cols add ~72 degrees of freedom to a model trained on ~12k
filtered samples, which is enough room for the model to memorize noise
that looks like signal.

## 3. What the result confirms vs invalidates

### Confirms (pipeline works, contracts are honored):

- ✅ Stage 4-A end-to-end pipeline is functionally correct (see Phase 4-A
  unit tests + Phase 5.B contract verification).
- ✅ News data scrape + FinBERT embed + align + PCA fit-on-train + concat
  produces the expected shape (X with 136 cols vs 64 price-only).
- ✅ Trainer-side meta saves `news_pca` + companions; gates loads them.
- ✅ Inference path threads `instrument` through per-pair routing and
  fetches news at scan time. The libomp deadlock fix lets the loop run.
- ✅ The holdout matches val_balanced ± 5pp (51.3% vs 54.96% = ~3.7pp gap),
  which is consistent with a small generalization gap for the train-only
  feature-selection step.

### Invalidates (the original hypothesis):

- ❌ Hypothesis: "FinBERT + PCA(64) + 24h-lookback news fusion lifts
  EUR_USD H1 holdout past the 52% gate." → **Rejected at n=1500.**

## 4. Phase 5.C decision options (operator picks)

The pipeline is now a verified Stage 4-A delivery surface. The remaining
question is what configuration of news fusion (if any) actually adds
signal. Five options ordered by cost:

| Option | Cost | Hypothesis tested | Risk |
|---|---|---|---|
| **5.C.i — Smaller PCA k** (16 or 32) | ~10 min retrain × 1 pair | Less expressive features → less overfitting | k=16 lost 25% variance, k=32 lost 12% — may underfit instead |
| **5.C.ii — Shorter lookback** (4h) | ~10 min retrain | Recent news contains more signal than 24h-old | Smaller weighted-event windows may be too sparse |
| **5.C.iii — Longer lookahead** ablation (6, 12 vs 24) | ~30 min × 3 retrains | Maybe news predicts shorter horizons better | More configs to manage |
| **5.C.iv — Different embedder** (text-embedding-3-large) | ~1h | Paid OpenAI embedder may have stronger domain signal | $0.13/1M tokens; ~10K events × 50 tokens × $0.13 = ~$0.07 per pair |
| **5.C.v — Multi-pair training** with shared news head | Multi-day implementation | More data dilutes per-pair spurious correlations | Significant scope; downstream of 5.C.i-iv |
| **5.C.vi — Shelve news fusion, pivot to alternative lever** | $0 | Maybe price-only at H1 IS the ceiling | Defers news work for ≥3 months until trade volume justifies offline RL (Decision Transformer) |

## 5. Recommendation

**5.C.i (smaller PCA k=32) FIRST.** Cheapest experiment; directly tests
the overfitting hypothesis. If k=32 holdout is materially better than
k=64 (e.g., ≥53%), continue narrowing. If k=32 is identical or worse,
the issue isn't dimensionality — it's a mismatch between FinBERT's
financial-headline embedding distribution and what the H1 trainer can
exploit with this much training data.

Then **5.C.ii in parallel** — shorter lookback could compound with
narrower PCA to find a sweet spot.

If both fail to lift past 52%, **5.C.vi** — the price-only ceiling at H1
is the ceiling, and the next lever is Decision Transformer / offline RL
once we have >5K trades from forward-testing USD_JPY (CLAUDE.md P4).

## 6. Stage 4-A architectural takeaways (preserved regardless of news result)

These are real shipped artifacts even if news fusion doesn't help:

1. **Production-ready FF historical backfill.** 39,853 events across 7
   majors × 36mo, idempotent parquet cache, cross-validated within ±10%
   vs JSON mirror.
2. **Generic news-fusion pipeline.** The wire (compute_normalized_features
   + load_direction_data + trainer + gates) supports ANY NewsSource +
   NewsEmbedder. Future swaps to text-embedding-3 or NewsAPI need only
   subclass + drop-in replace.
3. **Rigorous lookahead-bias guards.** Strict open-interval window in
   align_news_to_bars + 5 explicit lookahead-bias tests guarantee
   the inference path can't accidentally peek ahead.
4. **Two real bugs caught + fixed.** Per-pair routing instrument-stripping
   (Phase 4 echo) and the PyTorch+TensorFlow libomp deadlock would have
   been silent failure modes if not surfaced via this end-to-end run.
5. **Inference-time news fetch + embed + PCA.transform path.** Lazy-init
   FinBERT singleton, parquet cache hit, ~0.74s/prediction at H1.
   Production-deployable as-is.

## 7. What's preserved on disk

- `trained_data/models/EUR_USD/` — restored to price-only baseline (Phase 5.B)
- `trained_data/models/EUR_USD_news_first_run/` — Stage 4-A news-fused EUR_USD
  (val_acc=55.63%, holdout 51.3%; preserved as reference for future
  configuration comparisons)
- `trained_data/news/{pair}_ff_events.parquet` × 7 majors — 36mo backfill
  intact, reusable for Phase 5.C iterations
- `scripts/stage_4a_news_retrain.py` — direct retrain script ready for
  k/lookback sweeps (just change `--news-pca-n` and `--news-lookback-h`)

Halt remains True. USD_JPY H1 demo path unaffected.
