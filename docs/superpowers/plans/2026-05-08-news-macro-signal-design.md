# News/Macro Signal Pipeline — Design (P1)

**Date**: 2026-05-08
**Status**: Design + scaffolding (Phase 1 of 4). No data fetched, no trainer touched, no runtime touched.
**Owner**: AI Engineer agent (controller); operator approves recommendations + answers open questions before Phase 2.
**Promotion source**: CLAUDE.md "Modernization stance" §47 — promoted from P2 to P1 after empirical evidence that price-only models hit a ~70% holdout ceiling on M15 EUR_USD/GBP_USD across architectures (custom Transformer, Chronos-T5-small, Chronos-T5-base).

---

## 0. Executive summary

Direction-prediction holdout accuracy on M15 EUR_USD plateaus at ~70.0% across every architecture we have measured. The bottleneck is no longer model class; the four-bug data alignment fix already extracted the signal that the price tape contains. To break 70%, the model needs information that **is not in the price tape** — scheduled macro events, breaking-news sentiment, and policy-statement language.

This document scopes the pipeline that fuses event/headline data with the existing 50-feature price matrix, recommends one option for each axis (data source / embedder / fusion), and stages the work across four sessions so each session lands an isolated, measurable artifact.

| Axis | Recommendation | Rationale (1-line) |
|---|---|---|
| Data source (prototype) | **ForexFactory event calendar** + existing RSS headlines as secondary | Free, deterministic, already partially wired in `src/scanner/economic_calendar.py` and `_evaluate_news_risk` — fastest path to a measurable holdout number |
| Embedder | **FinBERT (`ProsusAI/finbert`)** | Free, finance-domain-tuned, runs on M1 Metal via `transformers` 4.57 already installed — zero new dependencies; quality matches OpenAI for sentiment-heavy headlines |
| Fusion architecture | **(a) Concatenation** at the feature-DataFrame level, single-tower model | Smallest blast radius on `compute_normalized_features → load_direction_data → train` pipeline; reusable by ridge / LightGBM heads without two-tower retrain |

---

## 1. Data sources comparison

| Source | Latency | Coverage | Cost | License | Integration cost | Signal quality (estimate) |
|---|---|---|---|---|---|---|
| **ForexFactory calendar** (XML/HTML scrape) | T+0 (scheduled releases known days ahead) | NFP, CPI, FOMC, ECB, BoE, BoJ, RBNZ, RBA — all major FX pairs | $0 | "personal use OK"; commercial-redistribution prohibited (we consume, don't republish) | LOW — `src/scanner/economic_calendar.py` already exists; `_evaluate_news_risk` already uses it via `market_intelligence.EconomicCalendar` | HIGH for *scheduled* moves (NFP / FOMC); ZERO for breaking news |
| **NewsAPI.org** | ~5-15min lag on free tier; T+0 on paid | 80k+ sources, query by keyword/symbol | $0 (dev, 100req/day, headlines only) → $449/mo (Business, full text) | Per-tier; commercial OK on paid | MEDIUM — REST API, requires key, rate-limit handling | MEDIUM — broad but noisy; free tier truncates to 100req/day = ~4 pulls/hour |
| **Reuters / Bloomberg** | T+0 institutional | Deepest FX coverage on the planet | $24k+/yr (Reuters Eikon entry); Bloomberg terminal $25k+/yr/seat | Per-license; commercial OK | HIGH — proprietary SDKs, redistribution gates | VERY HIGH — but cost-prohibitive for a prototype |
| **Alpaca News API** | T+0 streaming | Benzinga firehose, US-equity-tilted; FX coverage thin | $0 with brokerage account | Commercial OK | LOW-MEDIUM — Python SDK; needs Alpaca account | LOW for FX (US-equity bias); decent for risk-on/risk-off macro reads |
| **Generic RSS** (Google News, Reuters RSS) | ~30s-5min | Whatever the publisher syndicates | $0 | Per-source; usually OK for non-redistributive consumption | LOW — already wired in `news_features._rss_fallback_fetcher` | LOW — unstructured, dedupe is a project on its own |

**Recommendation: ForexFactory calendar (primary) + RSS headlines (secondary fallback).**

Rationale:
1. **Already partially wired.** `_evaluate_news_risk` in `src/scanner/agents/_team.py:2360-2402` already pulls ForexFactory events via `market_intelligence.EconomicCalendar` and RSS headlines via `news_features.fetch_forex_news`. Phase 2 only needs to add a *historical* fetch path — the runtime side already exists.
2. **Deterministic + reproducible.** Scheduled releases have a fixed timestamp. Backfilling 22 months of training data is a deterministic operation; we can train on the same event tape we'll see in production.
3. **Phase-2 timeboxing.** Avoids API-key procurement, vendor onboarding, and SDK install for Phase 2. NewsAPI/Alpaca slot in cleanly later as new `NewsSource` subclasses if RSS proves too thin.

**Evaluation criterion used: integration cost vs measurable signal at minimum spend.** A free, partially-wired data source that lets us measure holdout lift in one session beats a paid premium source we can't justify until lift is proven.

**Phase 2 follow-up to revisit:** if FF + RSS combined yield <2pp holdout lift over the price-only 70.0% baseline, queue NewsAPI Business tier as a Phase 4 expansion.

---

## 2. Embedding model comparison

| Model | Dim | Latency (M1 Metal) | Cost per 1k headlines | M1 compatible | Finance-domain quality | Notes |
|---|---|---|---|---|---|---|
| **FinBERT** (`ProsusAI/finbert`) | 768 | ~5-10ms/headline | $0 | Yes (transformers 4.57 already in env) | HIGH — pre-trained on Reuters financial PhraseBank | Outputs sentiment logits + last-hidden-state pooling; no API |
| **OpenAI text-embedding-3-small** | 1536 | API roundtrip ~100-300ms | $0.02 / 1M tokens ≈ $0.0004/1k headlines | API only | MEDIUM-HIGH (general-purpose, no finance finetune) | Best generic embedder $/quality; depends on internet + key |
| **OpenAI text-embedding-3-large** | 3072 | API roundtrip ~200-500ms | $0.13 / 1M tokens ≈ $0.003/1k headlines | API only | HIGH | Higher dim → more capacity at fusion layer; $7-15/yr at our scale |
| **all-MiniLM-L6-v2** (sentence-transformers) | 384 | ~2-5ms/headline | $0 | Yes | LOW-MEDIUM (generic web corpus) | Smallest viable embedder; useful as latency-floor benchmark |
| **VoyageAI voyage-finance-2** | 1024 | API ~80-200ms | $0.12 / 1M tokens ≈ $0.0024/1k headlines | API only | HIGH (finance-tuned) | Newer; less battle-tested |
| **Cohere embed v3** | 1024 | API ~100-300ms | $0.10 / 1M tokens | API only | MEDIUM | General-purpose; adequate |

**Recommendation: FinBERT (`ProsusAI/finbert`).**

Rationale:
1. **Zero new dependencies.** `transformers==4.57.6` already in env (verified 2026-05-08 on this worktree). Adding FinBERT is `AutoModel.from_pretrained("ProsusAI/finbert")` plus tokenizer — no `pip install`, no API key, no rate limit, no internet dependency in the trainer hot path.
2. **Domain match.** Pre-trained on Reuters Financial PhraseBank — exactly the corpus shape we'll consume. Generic embedders (MiniLM, OpenAI-small) treat "Fed pivots dovish" and "Fed pivots hawkish" as near-identical because the general web rarely uses those terms; FinBERT separates them.
3. **M1-friendly.** ~110M params; ~250MB on disk; CPU inference at ~5-10ms/headline is acceptable for batch backfill (22 months × ~50 headlines/day = ~33k headlines = ~5min wall-time on M1 CPU; negligible on Metal).
4. **Reproducibility.** Same weights every run. API embedders re-version silently (OpenAI deprecated `text-embedding-ada-002` in 2025; we'd need to re-embed all training data when that happens).

**Evaluation criterion used: total cost-of-ownership for a 4-session prototype.** FinBERT has zero ongoing cost, zero runtime risk, zero API-key management overhead. If FinBERT signal disappoints, OpenAI text-embedding-3-large is a one-line `embedder = OpenAIEmbedder()` swap because the `NewsEmbedder` interface is sealed in this design.

**Phase 2 follow-up to revisit:** if FinBERT lift is 0pp-2pp, run a second-pass with `text-embedding-3-large` ($5-15 total) before declaring news features useless.

---

## 3. Feature fusion architecture

| Architecture | How it fuses | Blast radius on existing trainer | Expected lift | Retraining cost |
|---|---|---|---|---|
| **(a) Concatenation** | News features computed per-bar (mean-pool over lookback window) → joined to price-feature DataFrame as new columns → `load_direction_data` ingests N+M-feature matrix unchanged | LOW — adds columns to DataFrame; no model-class change. Ridge + LightGBM heads consume it free | 1-3pp (low ceiling: model can't attend to news independently of price) | Same cost as price-only retrain (~3min/pair on M1) |
| **(b) Two-tower** | Separate text encoder + price encoder, fused at penultimate layer via concatenation or gating | MEDIUM-HIGH — requires Transformer architecture change in `transformer_trainer.py`; ridge/LightGBM heads cannot consume two-tower outputs directly | 2-5pp | ~2× price-only retrain |
| **(c) Cross-attention** | Text embeddings injected as additional sequence tokens; transformer self-attention learns price↔news interactions | HIGH — sequence-length expansion, attention-mask changes, position-encoding rework | 3-7pp (highest ceiling but most variance) | ~3-5× price-only retrain; risk of OOM on M1 Metal at long sequences |

**Recommendation: (a) Concatenation.**

Rationale:
1. **Smallest blast radius.** The existing trainer pipeline (`compute_normalized_features → load_direction_data → train_transformer/train_lgbm`) is a sequence of DataFrame operations terminating in `df.select_dtypes(include=[np.number]).columns.tolist()`. Adding news columns means the trainer auto-picks them up at line 1729 of `modular_data_loaders.py`. Zero model-class change.
2. **Reusable across heads.** Concatenation features feed ridge confidence, LightGBM momentum, RF risk, and TCN/Transformer direction equally. Two-tower locks news to one model class.
3. **Phase-staging fits.** If Phase 3 lift is meaningful (≥3pp), Phase 4+ can promote to two-tower / cross-attention as an optimization. Concatenation establishes the floor; advanced fusion expands the ceiling.
4. **Honest about the ceiling.** If concatenation gives 1-2pp lift and two-tower would have given 4pp, we'd at least know news contains *some* signal — and could justify the two-tower investment with empirical evidence rather than speculation.

**Evaluation criterion used: how much existing infrastructure must I rewrite?** Concatenation = 0 trainer changes; two-tower = transformer rewrite; cross-attention = transformer rewrite + sequence-length scaling. Phase 1 ships fastest with (a).

---

## 4. Time-alignment problem

**The asymmetry:** price bars are dense + regular (1 bar per 15 minutes, 96/day, ~33k bars over 22 months). News events are sparse + asynchronous (NFP fires monthly, FOMC every ~6 weeks, RSS headlines burst around events but go silent for hours). Naive join leaves 99.5% of bars with no event and produces a feature matrix that is mostly zeros.

**Algorithm (to live in `src/data/news/feature_alignment.py:align_news_to_bars`):**

For each price bar at timestamp `t_b`, build a per-bar news vector by:

1. **Lookback window**: collect every `NewsEvent` with `event.timestamp ∈ [t_b - lookback_window_hours, t_b)`. Default `lookback_window_hours = 24` (one trading day).
2. **Time-decay weighting**: for event at time `t_e`, weight `w_e = exp(-(t_b - t_e) / tau)` where `tau = lookback_window_hours / 3` (decay constant; events 8h+ stale carry < 5% weight). This handles NFP-day burst-then-silence cleanly.
3. **Embed**: each event has a `text_embedding ∈ R^768` from FinBERT.
4. **Per-bar mean-pool**: `bar_embedding[t_b] = sum_e(w_e * event_embedding_e) / max(sum_e(w_e), 1e-6)`. Returns `R^768`.
5. **Event-class one-hot fallback**: in addition to the embedding mean, emit a small `R^8` vector counting events-in-window by class: `[NFP, CPI, GDP, FOMC, ECB, BoE, BoJ, OTHER]`. This gives the model a bias-free counter for "high-impact event imminent" even when no headlines are present.
6. **No-event bar**: `bar_embedding = zeros(768)`, `event_class_count = zeros(8)`. Honest absence-signal.
7. **PCA compression (Phase 3)**: 768-dim per bar × 33k bars = 25M floats. PCA-fit on training fold to ~32 components reduces to 1M floats while retaining ~95% of variance. Stored alongside the existing scaler in `trained_data/scalers/`.

**Insertion point:** `compute_normalized_features` in `src/core/modular_data_loaders.py:692`. Phase 3 wiring will:
1. Add a `news_df` argument (default `None` → no-op for backward compat with all existing call sites — `compute_normalized_features` is called from 8+ loaders).
2. After the existing 186 features compute, if `news_df is not None`, call `align_news_to_bars(news_df, df.index, lookback_window_hours=24)` and join the resulting `R^32` (post-PCA) + `R^8` (event-class) columns.
3. Result: `df` grows from 186 → 226 columns. The dynamic-feature-selection block at `:1772` correlation-prunes redundant features automatically.

**Lookahead-bias guard:** event timestamps must be **release timestamps**, not scrape timestamps. ForexFactory publishes NFP at `release_time`, not `now()`. The `NewsEvent.timestamp` field is the release time; `align_news_to_bars` strictly enforces `event.timestamp < t_b` (open interval, no leak from the current bar).

**Walk-forward validation guard:** PCA must be fit on the training fold ONLY, then applied frozen to val/test. `feature_alignment.py` will accept a pre-fit PCA object via dependency injection, mirroring how the price-feature scaler is handled in `direction_trainer.py`.

---

## 5. Validation strategy

**Hypothesis:** news features add ≥3pp holdout-accuracy lift on M15 EUR_USD direction prediction. Baseline: 70.0% holdout (price-only, 22-month walk-forward, validated 2026-05-08).

**Target:** ≥73.0% news-augmented holdout (3pp = signal-vs-noise threshold; below 3pp = within walk-forward variance, likely noise).

**Cheapest experiment that proves/disproves news lift in <1 hour of compute (Phase 3):**

1. **Backfill** ForexFactory + RSS for EUR_USD, 2024-08 to 2026-05 (~22 months) → `trained_data/news/EUR_USD_events.parquet` (~30s wall time once Phase 2 is wired).
2. **Embed** with FinBERT batch — ~5min on M1 CPU for ~30k events.
3. **PCA-fit** on train fold (first 70%), 768 → 32 components — ~5s.
4. **Align** to 33k M15 bars via `align_news_to_bars` → 33k × (32 + 8) news features.
5. **Train direction head** at M15 with the augmented feature matrix — ~3min on M1 Metal (matches existing baseline trainer time; concatenation adds 40 columns, marginal cost).
6. **Compare** holdout accuracy to 70.0% baseline. **Decision rule**:
   - **≥73.0%**: ship news pipeline; Phase 4 expands to remaining majors.
   - **70.5%-72.9%**: ambiguous; rerun with `text-embedding-3-large` to rule out FinBERT-specific quality issue (~$1 spend).
   - **≤70.5%**: news features (or this fusion architecture) don't help; go back to first principles or retire the work item.

Total wall time: ~10min. Total compute cost: $0. Total operator time to review: ~5min on a single notebook.

**Walk-forward integrity:** the comparison MUST use the same train/val/test splits as the 70.0% baseline. `load_direction_data(split=(0.7, 0.2, 0.1))` is the existing default; do not change it.

**Tracked in W&B:** Phase 3 retrain logs to `wandb_offline/runs/news_baseline_eurusd/`; same project as existing direction-head experiments. Single chart compares `accuracy_holdout` series.

---

## 6. Sequencing

| Phase | Owner | Scope | Output | Touches trainer? | Touches runtime? |
|---|---|---|---|---|---|
| **Phase 1 (this session)** | AI Engineer | Design doc + scaffolding stubs + CLAUDE.md pointer | This doc + `src/data/news/{__init__,source,embedder,feature_alignment}.py` (stubs) + `tests/test_news_pipeline_stubs.py` + CLAUDE.md update | NO | NO |
| **Phase 2 (next session)** | Data Engineer or AI Engineer | Implement `ForexFactoryNewsSource.fetch_events`; implement `FinBERTEmbedder.embed`; manual-fetch sample for EUR_USD; verify shapes; persist sample to `trained_data/news/EUR_USD_sample.parquet` | Working data-fetch + embed pipeline; sample artifact on disk; integration test (`@pytest.mark.integration`, real network, no mocks) | NO | NO |
| **Phase 3 (session after)** | AI Engineer | Implement `align_news_to_bars`; backfill EUR_USD 22 months; PCA-fit; thread `news_df` through `compute_normalized_features`; retrain M15 EUR_USD with news features; compare to 70.0% baseline | Holdout number; W&B run; decision on Phase 4 | YES (additive: optional `news_df` arg) | NO |
| **Phase 4 (later)** | AI Engineer | If Phase 3 lift ≥3pp: backfill + retrain GBP_USD, USD_JPY, USD_CHF, AUD_USD, USD_CAD, NZD_USD; promote per-pair news models | All 7 majors news-augmented at M15 | YES (existing wiring; new pairs only) | YES (`_evaluate_news_risk` may be augmented to surface embedded-news-driven confidence) |

**Phase boundary discipline:** each phase is one session. End-of-session must produce a measurable artifact (file on disk, holdout number, decision). No phase straddles operator's halt-mode boundary — the bot stays halted throughout Phases 1-3 (no live impact); Phase 4 can land while halted and require operator unhalt to validate.

**Gate to start Phase 2:** operator answers the open questions in §7 below.

---

## 7. Decisions (resolved 2026-05-08 by Claude as partner)

Operator delegated decisions on the open questions. Below are the calls + reasoning. Where I diverge from the agent's original recommendation, the divergence is called out explicitly.

| # | Question | Decision | Reasoning |
|---|---|---|---|
| 1 | Calendar source | **Reuse `market_intelligence.EconomicCalendar`** | Same class of bug as Option G if training and runtime use different event tapes. Add `fetch_events_historical(since, until)` if missing. (Confirms agent.) |
| 2 | FinBERT vs OpenAI ambiguous-band experiment | **Yes, $1 on `text-embedding-3-large` if Phase 3 lift is 70.5%-72.9%** | Cheapest disambiguation possible. Rules out "FinBERT was the wrong embedder" before shelving the workstream. (Confirms agent.) |
| 3 | NewsAPI Business subscription | **DEFER** | $449/mo recurring is real cost. Prove signal on free FF+RSS first. (Confirms agent.) |
| 4 | PCA dimensionality | **32 components** | ~95% variance retained; trivially expandable to 64 if Phase 3 holdout disambiguates. (Confirms agent.) |
| 5 | Lookback window | **REVISED — `[4, 24]` for BOTH training AND runtime** | Agent recommended asymmetric (24h train, 4h prod). I disagree: that re-introduces the same train/eval mismatch class as Option G's lookahead bug. Symmetric is non-negotiable. The runtime path can adopt the [4,24] two-window block alongside the existing 4h-only path; 4h stays as fallback when 24h data is missing. |
| 6 | Backfill rate-limit policy | **Accept; cache to `trained_data/news/{pair}_{date_range}.parquet`** | One-time cost per pair. Idempotent fetch wrapper that skips if cached. (Confirms agent.) |

**Net:** 5 of 6 decisions match the agent's recommendation; Q5 is revised to enforce symmetric train/eval features as a hard project invariant.

---

## 7-original. Open questions for operator (preserved for history)

1. **ForexFactory scrape vs market_intelligence.EconomicCalendar?** The existing runtime path uses `market_intelligence.EconomicCalendar`. Phase 2 can either reuse that module's `fetch_events()` (consistent with runtime) or write a fresh historical-backfill scraper (consistent with deterministic backfill). **Recommendation: reuse `market_intelligence.EconomicCalendar`; add a `fetch_events_historical(since, until)` method if not already present.** Reason: training and runtime should consume the same event tape.

2. **FinBERT vs OpenAI cost/quality experiment.** Phase 2 ships FinBERT only. If Phase 3 holdout lift is in the ambiguous 70.5%-72.9% band, do we (a) burn ~$1 on `text-embedding-3-large` to disambiguate, or (b) accept the result and move to next priority? **Recommendation: (a). $1 to know if a $0 model was the wrong choice is the cheapest disambiguation possible.**

3. **NewsAPI subscription tier — defer or commit?** If Phase 3 lift confirms news adds signal but RSS+FF coverage feels thin, NewsAPI Business is $449/mo. **Recommendation: defer — only commit after Phase 4 ships ≥6 majors and we have evidence that headline volume (not just calendar events) is the marginal lever.**

4. **PCA dimensionality — 32 or 64 components?** Tradeoff: 32 = ~95% variance retained, lighter; 64 = ~98% variance retained, slower retrain. **Recommendation: 32, validate via scree plot in Phase 3.** Easy to expand if bottlenecked.

5. **Lookback window — 24h or 4h?** The runtime `_evaluate_news_risk` uses `hours_ahead=4`, suggesting near-term impact dominates. But scheduled events (NFP) move markets the day after the release. **Recommendation: 24h for training (captures multi-day position-building); separate 4h window in production matches existing runtime.** Two windows = two feature blocks; trainer handles both. Add as `lookback_windows = [4, 24]` in Phase 3.

6. **Backfill cost.** ForexFactory may rate-limit historical scrape (Cloudflare). If Phase 2 hits rate limits, we may need to throttle to ~1 month per 10min, extending Phase 2 to 3-4 hours total. **Recommendation: accept; backfill happens once per pair; cache in `trained_data/news/` for reuse.**

---

## 8. Non-goals (explicitly)

- **Don't build a news-driven trading agent yet.** This pipeline feeds the existing direction head; the existing 15-agent consensus + RL loop stays unchanged. News is a feature, not a model.
- **Don't replace `_evaluate_news_risk` runtime.** Phase 4 may *augment* it with embedded-news features, but the runtime VADER-sentiment + calendar-penalty path stays intact as a fallback when historical embeddings are unavailable.
- **Don't ship news features without holdout proof.** Phase 3 must produce a measurable holdout improvement; if it doesn't, the work is shelved, not promoted.
- **Don't fetch real-time news in the runtime hot path.** The runtime `_evaluate_news_risk` already does this and it's fine. The Phase 4 augmentation reads pre-computed embeddings from disk, not on-the-fly inference.
- **Don't add a new database.** Parquet on disk for backfill data; no Pinecone, no Postgres, no migrations.

---

## 9. File layout (new in Phase 1)

```
src/data/news/
├── __init__.py                  # Public surface; re-exports NewsSource, NewsEmbedder, NewsEvent, align_news_to_bars
├── source.py                    # NewsSource ABC + NewsEvent dataclass + ForexFactoryNewsSource (Phase 2 stub)
├── embedder.py                  # NewsEmbedder ABC + FinBERTEmbedder (Phase 2 stub)
└── feature_alignment.py         # align_news_to_bars(events, bar_timestamps, lookback_window_hours) (Phase 3 stub)

tests/
└── test_news_pipeline_stubs.py  # Imports + instantiation + NotImplementedError verification (no mocks)

docs/superpowers/plans/
└── 2026-05-08-news-macro-signal-design.md  # this document
```

Phase-2 additions (anticipated, not landed in this commit):
- `src/data/news/forex_factory_historical.py` — backfill scraper
- `src/data/news/finbert_embedder_impl.py` — FinBERT inference wrapper
- `tests/test_news_source_integration.py` — `@pytest.mark.integration`, real network

Phase-3 additions:
- `src/data/news/pca_compressor.py` — train-fold PCA fit + transform
- Trainer modification: `compute_normalized_features(df, news_df=None)` adds the optional arg

---

## 10. Verification surfaces (how operator audits this work)

| Surface | What to check |
|---|---|
| `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md` | This file exists; sections 1-6 present |
| `src/data/news/*.py` | 4 files; each has docstrings; stub methods raise `NotImplementedError("Phase 2 implementation")` or similar |
| `tests/test_news_pipeline_stubs.py` exit 0 | Tests verify imports + instantiation + exception raising; **no `unittest.mock`** |
| `CLAUDE.md` "News/macro pipeline (P1)" section | Single subsection naming this design doc, the chosen recommendations, and the phase sequencing |
| `git log --oneline -5` | One commit on `news-macro/P1-design-scaffold` branch (or feature branch name) named per project commit-style |

---

## 11. Honesty caveats

- **No news data has been fetched.** Estimates of headline volume (~50/day), embedding latency (~5-10ms/headline on M1), and PCA variance retention (95% at 32 components) are extrapolated from FinBERT public benchmarks and similar embedding-fusion papers. Phase 2 will replace estimates with measurements.
- **The 3pp lift target is a hypothesis, not a guarantee.** Generic-domain news-fusion work in equities-prediction literature reports 1-5pp typical lift over price-only baselines. FX may behave differently because FX is more macro-driven (so news *could* help more) or because FX is more institutional (so news *may* be already priced in faster than retail-equities benchmarks suggest).
- **The 70.0% baseline may itself drift.** If the EUR_USD M15 model is retrained between Phase 1 and Phase 3 with new data or fixes, the comparison must use the contemporaneous baseline, not the 2026-05-08 number.
- **Phase 3 may surface that concatenation is insufficient.** If the holdout lift is 0pp despite the embedder + alignment working correctly, it may signal that two-tower fusion is needed. This is a known acceptable outcome — concatenation is the cheapest way to discover whether news helps at all.
