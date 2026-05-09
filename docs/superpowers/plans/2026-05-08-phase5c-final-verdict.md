# Phase 5.C Final Verdict — News Fusion Empirically Rejected at H1 / lookahead=24

> **Status**: Stage 4-A pipeline VERIFIED working (architectural delivery
> preserved). News fusion at H1 lookahead=24 does NOT lift holdout signal
> across 3 different configurations. Hypothesis exhausted; pivot recommended.
>
> **Bot stays halted.** USD_JPY H1 (62%) demo path remains unaffected.

---

## 1. Empirical record (EUR_USD H1, lookahead=24, n=1500)

| Config | val_acc | val_balanced | Holdout n=1500 | Δ baseline |
|---|---:|---:|---:|---:|
| Price-only baseline (Phase 5.B) | 51.29% | 48.97% | **51.3%** | (reference) |
| News k=64, 24h lookback (5.C orig) | 55.63% | 54.96% | 51.3% | 0.0pp |
| News k=32, 24h lookback (5.C.i) | 54.96% | 54.80% | 49.6% | -1.7pp |
| News k=64, 4h lookback (5.C.ii) | 53.80% | 51.43% | 50.8% | -0.5pp |

3 news-fusion configs tested; 0 of 3 beat price-only baseline at the holdout
level. The pattern is consistent: training-time val accuracy is inflated by
overfitting on news features (which RF feature selector preferentially picks
because of spurious correlation with training labels), and the lift fails to
generalize to out-of-sample windows.

## 2. What this rejects

| Hypothesis | Verdict |
|---|---|
| "FinBERT + PCA(64) + 24h lookback news fusion lifts EUR_USD H1 holdout past 52% gate" | ❌ rejected at n=1500 |
| "Smaller PCA k reduces overfitting → better holdout" | ❌ k=32 actually worse |
| "Shorter lookback captures more recent signal → better holdout" | ❌ 4h worse than 24h on both train val and holdout |

## 3. What this does NOT rule out

The Stage 4-A architecture is sound; the immediate news config is the failure.
Untested hypotheses that could still produce lift:

| Untested option | Estimated cost | Expected lift if hypothesis correct |
|---|---|---|
| Different lookahead (6 or 12 vs 24) | ~30 min × 2 retrains + holdouts | Unknown; news may predict shorter horizons better |
| Different embedder (text-embedding-3-large) | ~$0.07 + ~30 min | Possibly stronger domain signal |
| Multi-pair training with shared news head | Multi-day implementation | More data dilutes spurious correlations |
| Aggressive regularization (dropout / L2 specifically on news cols) | ~30 min retrain | Reduces overfitting if that's the bottleneck |

But each of these is a fresh experiment with its own training cost. **At
some point, declining marginal returns dominate**: with 3 negative tests
already and no clear failure pattern in the data, further news-config
sweeps are speculative.

## 4. Decision tree (operator picks)

The honest read of the data: **price-only at H1 is approximately the
ceiling for current architecture + data scale**. CLAUDE.md modernization-
stance section already flagged this in §1: "the price-only direction-
prediction holdout has plateaued at ~70.0% on M15" — that 70% number was
based on broken pipeline; the corrected ceiling at H1 is ~51-52% mean.

Three forward paths:

### Path A — One more cheap experiment, then pivot

Test lookahead ablation: retrain EUR_USD news k=64 24h with `--lookahead 6`
and `--lookahead 12`, holdout each. If either lifts past 53%, news fusion
works at shorter horizons. If neither does, conclude shorter horizons
don't help and pivot.

**Cost**: ~1h. **Reward**: clear yes/no on the lookahead axis.

### Path B — Pivot to forward-testing USD_JPY (5.C.vi)

USD_JPY H1 holdout was **62.0% (9σ above chance)** in Phase 5.B. That
signal IS tradeable. The next experiment that produces a real outcome
isn't ML — it's running USD_JPY on demo for 1-2 weeks and seeing whether
the holdout number translates to live forward-test accuracy.

This produces:
- Real demo trade data feeding the journal (which Phase 6+ Decision
  Transformer can use)
- Validation that the 62% holdout isn't artifact (cross-time generalization)
- Operator confidence in the corrected pipeline before risking real capital

**Cost**: ~10 min config + 1-2 weeks observation. **Reward**: first real
forward-test signal in months.

### Path C — Pursue Decision Transformer / offline RL preparation

CLAUDE.md flags Decision Transformer as P4 "needs >5K trades first."
Forward-testing USD_JPY accelerates the trade-journal accumulation. In
parallel, prep the offline-RL training infrastructure so it's ready when
journal volume is sufficient.

**Cost**: Engineering effort over next month. **Reward**: long-term
tradeable architecture once data is available.

## 5. Recommendation

**B + C in parallel; A only if operator explicitly wants exhaustive news
config sweep.**

Rationale:
- USD_JPY has demonstrated tradeable signal; not exploiting it while
  doing speculative news sweeps is premature optimization.
- Decision Transformer prep work doesn't block the demo forward-test —
  parallel tracks compound learnings.
- Path A's residual probability of success is decreasing fast: with 3
  negative news tests, the prior on news-config-lift is now ~10-15%.

If operator wants the exhaustive sweep, Path A is fine — the
architecture is built and the iteration cost is low (1h per axis).

## 6. What's preserved on disk regardless

- **trained_data/news/{pair}_ff_events.parquet × 7** — 36mo backfill,
  reusable for ANY future news experiment.
- **trained_data/models/EUR_USD_news_first_run/** — k=64, 24h config,
  reference for future architectural changes.
- **trained_data/models/EUR_USD_news_k32/** — k=32 config.
- **trained_data/models/EUR_USD_news_lb4/** — 4h lookback config.
- **scripts/stage_4a_news_retrain.py** — reusable for any pair × news config.
- **Stage 4-A pipeline contract in trainer + gates** — works for any
  NewsSource + NewsEmbedder subclass.

These artifacts are real shipped value even though the immediate
hypothesis was rejected.

## 7. CLAUDE.md update needed

The "Modernization stance" section's ceiling claim was already corrected
(50-52% mean H1 instead of 70% M15). This Phase 5.C verdict adds the
follow-up: **news/macro fusion at H1 lookahead=24 with FinBERT does not
lift past that ceiling for EUR_USD**, across 3 distinct configurations.
Future news experiments should target different lookaheads or different
embedders if the architectural lever is news.

Halt remains True. Main HEAD will be next commit.
