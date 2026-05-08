# Stage 4-A — News/Macro Fusion at H1 (Spec)

**Date**: 2026-05-08
**Status**: Spec — operator-approved scope (F4 phased), pending operator review of this document before `writing-plans` runs.
**Owner**: Software Architect (this spec) + Agents A/B/C (parallel implementation of NewsSource, FinBERTEmbedder, align_news_to_bars).
**Supersedes**: §5 ("Validation strategy") and §7 ("Decisions") of `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md` insofar as those sections quote a 70% M15 baseline. The corrected baseline is documented in §1 below.

---

## 1. Goal — corrected baseline restatement

Lift the 5 weak H1 majors (EUR_USD 51.3%, GBP_USD 50.2%, USD_CAD 51.3%, USD_CHF 47.9%, AUD_USD 47.2%) above the 52% gate by fusing scheduled-event + headline-sentiment features into the existing 186-feature price matrix at H1. The corrected price-only ceiling — measured 2026-05-08 in `docs/superpowers/plans/2026-05-08-phase5-h1-decision.md` after end-to-end pipeline reconciliation — is **mean 52.3% across 7 H1 majors**, with **USD_JPY tradeable at 62.0%** (9.3σ) and **NZD_USD significant at 56.1%** (4.7σ). The earlier "70% M15" number was produced by the broken pre-Phase-2 pipeline (null scaler + lookahead mismatch + primary_tf misalignment) and is invalid; any references to it in prior docs must be read as bugged-pipeline artifacts. Stage 4-A's hypothesis is that ≥3pp news-feature lift on the 5 weak pairs is the empirical test of whether macro/news context contains tradeable signal beyond the price tape at H1.

---

## 2. Architecture diagram

```
                                            (Stage 4-A scope)
+----------------------+
| Agent A              |     +-------------------+
| ForexFactoryNews-    | --> | NewsEvent[]       |
| Source.fetch_events  |     | (timestamp UTC,   |
| (+ RSS fallback)     |     |  text, classes)   |
+----------------------+     +---------+---------+
                                       |
                                       v
                             +-------------------+
                             | Agent B           |
                             | FinBERTEmbedder   |
                             | .embed(text)      |
                             +---------+---------+
                                       |
                                       v
                       +---------------+---------------+
                       | event_embedding ∈ R^768       |
                       | event_class_count ∈ R^8       |
                       +---------------+---------------+
                                       |
                                       v
                            +----------------------+
                            | Agent C              |
                            | align_news_to_bars   |
                            | (lookback decay,     |
                            |  open-interval guard)|
                            +----------+-----------+
                                       |
                                       v
                            +-----------------------+
                            | news_features_df      |
                            | index = bar_timestamp |
                            | cols = pca_0..pca_63  |
                            |        + ec_0..ec_7   |
                            +----------+------------+
                                       |
                                       v   (THIS SPEC begins here)
+----------------------+    +-----------------------------------+
| existing price_df    | -> | compute_normalized_features(      |
| (186 features, M15→  |    |   df,                             |
|  resampled to H1)    |    |   news_features_df=news_df  <NEW> |
+----------------------+    | ) → 186 + 64 + 8 = 258 columns    |
                            +------------------+----------------+
                                               |
                                               v
                                  +------------------------+
                                  | load_direction_data    |
                                  | walk-forward split     |
                                  +-----------+------------+
                                              |
                                              v
                                  +------------------------+
                                  | TransformerDirection-  |
                                  | Trainer (H1, 60-bar    |
                                  | context, EMA+EWC)      |
                                  +-----------+------------+
                                              |
                                              v
                                  +------------------------+
                                  | per-pair H1 holdout    |
                                  | (1500 windows)         |
                                  +------------------------+
```

Concatenation fusion at the DataFrame level (decision locked by P1 design doc §3). No model-class change. Trainer auto-picks up new columns at `modular_data_loaders.py:1729` via `df.select_dtypes(include=[np.number])`.

---

## 3. Components owned by parallel agents (reference only — do NOT redesign here)

| Component | Owner | Contract this spec assumes | Source of truth |
|---|---|---|---|
| `ForexFactoryNewsSource.fetch_events(start, end, pair) -> List[NewsEvent]` | **Agent A** | Returns release-time-stamped events with `(timestamp_utc, text, event_class, currency_pair)`; release-time, NOT scrape-time | `src/data/news/source.py` (Phase 2 impl) |
| `FinBERTEmbedder.embed(events: List[NewsEvent]) -> np.ndarray (N, 768)` | **Agent B** | Deterministic `from_pretrained("ProsusAI/finbert")`; **mean-pooled** last-hidden-state masked by `attention_mask` (NOT CLS-pooled); M1 Metal batched. Verified empirically: 316 headlines/sec on MPS, 201 on CPU. | `src/data/news/embedder.py` (Phase 2 impl, landed) |
| `align_news_to_bars(events, bar_timestamps, embeddings: np.ndarray, lookback_window_hours=24) -> np.ndarray (n_bars, embedding_dim + 8)` | **Agent C** | Open-interval `event.timestamp < t_b` (no leak from current bar); time-decay weight `exp(-Δsec / tau_seconds) * relevance_score`, `tau = lookback/3`; weighted mean-pool embeddings (denom clamped 1e-6); 8-dim weighted side-counts over `[NFP, CPI, GDP, FOMC, ECB, BoE, BoJ, OTHER]`. **Returns ndarray, NOT DataFrame** — caller (`compute_normalized_features` integration) handles PCA + DataFrame join. Verified: 5 lookahead-bias tests + 3 contract guards pass. | `src/data/news/feature_alignment.py` (Phase 3 impl, landed) |

This spec does NOT redefine those contracts. If Agents A/B/C land surfaces that diverge from the assumptions above, this spec must be amended before integration.

---

## 4. NEW work owned by Stage 4-A integration

### 4.1 `compute_normalized_features` signature change

File: `src/core/modular_data_loaders.py:692`.

Add optional kwarg:

```
compute_normalized_features(
    df: pd.DataFrame,
    *,
    news_features_df: Optional[pd.DataFrame] = None,
    ...existing args...,
) -> Tuple[pd.DataFrame, ...]
```

**Join contract**:
- Join key: `df.index` (bar_timestamp, UTC, tz-aware).
- Join semantics: left-outer join. Bars with no aligned news row → fill `pca_*` and `ec_*` columns with zeros (honest absence-signal — same convention as `align_news_to_bars`'s no-event branch).
- Default `news_features_df=None` → no-op; preserves backward compat for the 8+ existing call sites of `compute_normalized_features` that don't yet pass news.

### 4.2 PCA compression

- Dim: **64 components** (empirically locked 2026-05-08 by Agent B). The P1 design doc estimated k=32 would retain ~95% variance; measurement on real FinBERT financial-headline embeddings showed k=32 → 88.6% only; k=64 → **95.6%** which clears the 95% retention floor. 12× compression (768→64) is still acceptable for the downstream DataFrame join.
- Fit policy: **train-fold only**. The PCA object is constructed inside `load_direction_data` after the train/val/test split, fit on train rows only, then frozen and applied to val + test + (later) inference.
- Persisted to model meta as a new contract field `news_pca: PCA` using the project-standard `joblib` serializer (consistent with how the existing `scaler: StandardScaler` is persisted — no plain-pickle paths added by this spec).

### 4.3 Inference contract additions

The Phase 2.A inference contract (`trained_data/models/{PAIR}/H1/contract.json` + sidecar joblib artifacts) currently has: `feature_names`, `scaler`, `regime_quantiles`, `version`. Stage 4-A adds three required fields:

| Field | Type | Purpose |
|---|---|---|
| `news_pca` | sklearn `PCA` (joblib-serialized) | Transform of the 768-dim FinBERT embedding to 64-dim feature block at inference time |
| `news_event_class_count_columns` | `List[str]` | Names of the 8 one-hot event-class columns; preserves order between train and inference |
| `lookback_window_hours` | `int` | Single-source-of-truth lookback used at training; runtime MUST use the same (P1 §7 Q5 — symmetric train/eval is non-negotiable) |

**Version bump**: `FEATURE_PIPELINE_VERSION` → `"2026-05-08-v2"`. Inference loaders must reject contracts with `version < 2026-05-08-v2` when news features are expected. Models trained without news (e.g., USD_JPY held at price-only — see §6 risks) keep `version = 2026-05-08-v1` and skip the news-loading branch entirely.

### 4.4 Runtime gate path (inference)

`src/scanner/gates.py::evaluate_transformer` currently builds the feature matrix from price-only data. Stage 4-A inserts the news path:

| Step | Call | Notes / latency budget |
|---|---|---|
| 1 | `NewsSource.fetch_events(now - lookback, now, pair)` | **Cache per scan-cycle**: one fetch per pair per cycle (5-min cycle on H1 scan). Budget: 200ms cold, 5ms cached. |
| 2 | `FinBERTEmbedder.embed(events.texts)` | Skip if no new events since last cycle (most cycles). Budget: 50ms for ≤10 fresh events on M1 Metal; 0ms steady-state. |
| 3 | `news_pca.transform(embeddings)` | 768 → 64 matmul. Budget: <1ms. |
| 4 | `align_news_to_bars(events, current_bar_timestamps, pca_embeddings, lookback_window_hours=24)` | PCA done by caller (step 3). Returns ndarray shape (n_bars, 72); caller joins to price_df. Budget: 5-10ms. |
| 5 | `compute_normalized_features(price_df, news_features_df=news_df, ...)` | Existing path with new kwarg. Budget: same as today + ≤2ms. |

**Total worst-case added latency per scan**: ~250ms cold, ~10ms steady-state. Acceptable on H1 (300s scan budget).

**Pre-fetch hook**: on bar-close (`_on_bar_close` if exists, or scanner's pre-scan callback), warm the news cache so the in-cycle path is steady-state. If pre-fetch absent, accept the 200ms cold cost on the first scan after a bar.

---

## 5. Validation protocol (decision rules with explicit numbers)

### 5.1 Per-pair H1 holdout retrain

Each of the 7 H1 majors retrains with news features ON, on the same 25k-training-candle / 5k-holdout-candle / 1500-window split as the Phase 5.B baseline (`docs/superpowers/plans/2026-05-08-phase5-h1-decision.md` §1). Identical splits = honest delta.

### 5.2 Statistical-significance threshold

Binomial std for 1500 windows = √(0.25/1500) ≈ 1.29pp. Therefore:
- **+3pp lift over baseline = ~2.3σ over noise**. This is the minimum bar for "news lifted this pair beyond walk-forward variance".
- A pair claiming "news improved holdout" must beat its baseline by **≥3pp on 1500 windows**. Anything <3pp is reported as "no signal detected" even if the raw delta is positive.

### 5.3 Decision matrix (per pair)

| Per-pair news-augmented holdout | Action |
|---|---|
| **≥58.0%** | **Ship news-fusion to that pair**. Mark for Stage 4-B (replicate to M15 + H4 for the same pair). |
| **52.0% – 57.9%** | **Pipeline correct, signal real but weak.** Do NOT ship to runtime yet. Queue Stage 4-A.2: rerun with `text-embedding-3-large` (per P1 §7 Q2) to disambiguate "FinBERT was the wrong embedder" from "news doesn't help this pair". $1 spend. |
| **<52.0%** | **News fusion does not help this pair.** Shelve for that pair. Document baseline + augmented numbers in the post-mortem; do NOT silently retry. Pursue different lever (lookahead sweep / longer-history retrain / different embedder). |

### 5.4 Stage 4-A overall decision (across the 5 weak majors)

| Outcome | Action |
|---|---|
| **≥3 of 5 pairs reach ≥52%** | Stage 4-A succeeded on the broad hypothesis. Proceed to **Stage 4-B** (replicate to M15 + H4). |
| **1-2 of 5 pairs reach ≥52%** | Partial success. Ship news-fusion to those pairs only at H1; do NOT auto-promote to Stage 4-B until a second pair confirms or a different embedder unlocks more. |
| **0 of 5 pairs reach ≥52%** | News-fusion at H1 with FinBERT + concat does not work for this pipeline. Queue: (a) text-embedding-3-large rerun on EUR_USD only (cheapest disambiguation, $1); (b) if that also fails, shelve news-fusion and pursue different lever. Do NOT escalate to two-tower architecture without intermediate evidence. |

---

## 6. Risks + mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | **News data fetched at inference time blocks the scan loop.** Cold fetch on first call ~200ms; networked rate-limits could spike to >1s. | Cache `NewsSource.fetch_events` results per scan-cycle (5-min TTL on H1). Pre-fetch on bar-close hook. Hard timeout 500ms; on timeout, fall back to zero-vector news features and log `news_fetch_timeout` to `buddy_debug.log`. Trade still gates on price features alone — degraded but not blocked. |
| R2 | **News data sparse → most bars get zero embedding.** ~50 events/day vs 24 H1 bars/day means most bars fall in event-empty windows. | 24h lookback default (per P1 §7 Q5) gives most bars at least one event in window. If Phase 3 measurement shows >40% of bars with all-zero news features, expand to multi-window `lookback_windows = [4, 24, 72]` and concatenate. Budget: +80 columns; trainer auto-handles. |
| R3 | **USD_JPY (62%, 9.3σ) regresses after news fusion.** Adding 72 noisy columns (64 PCA + 8 category counts) to a working feature matrix can hurt as easily as help. | Stage 4-A retrains ALL 7 H1 majors WITH news features, but **holdout USD_JPY is evaluated separately and gated**: if news-augmented USD_JPY drops below **60.0%** (i.e., loses more than 2pp from the 62.0% baseline), revert USD_JPY to price-only model. Persist USD_JPY's price-only contract (`version = 2026-05-08-v1`) as the production artifact regardless of news-augmented number. NZD_USD same logic with floor at **54.0%** (i.e., loses no more than 2pp from 56.1%). |
| R4 | **Train/inference asymmetry on lookback window** (re-introduces the Option-G class of bug). | Single-source-of-truth `lookback_window_hours` field on the inference contract (§4.3). Trainer writes it; runtime reads it; integration test asserts `train.lookback == runtime.lookback`. |
| R5 | **PCA fit leaks val/test data** if Agent C accepts a pre-pooled PCA fit on the full corpus. | Spec requires `pca=None` at train-fold-load time → fit happens inside `load_direction_data` AFTER walk-forward split. Code review on Agent C's `align_news_to_bars` must verify the fit-on-train-only call site. |
| R6 | **ForexFactory rate-limit / scrape failure during backfill.** | Cache to `trained_data/news/{pair}_{date_range}.parquet`; idempotent re-fetch wrapper. P1 §7 Q6 already accepts this cost. Sets the upper bound on backfill duration to ~3-4h per pair worst-case. |

---

## 7. Out of scope for Stage 4-A

Explicitly deferred (do NOT expand scope into these without operator approval):
- **M15 retraining with news features** — Stage 4-B. Only triggers if Stage 4-A 5-pair decision (§5.4) lands in the "≥3 of 5 pairs ≥52%" branch.
- **H4 retraining with news features** — Stage 4-B same gate.
- **Per-TF model ensemble** (e.g., M15 + H1 + H4 news-aware models voted at inference) — Stage 4-C. Requires Stage 4-B to land first.
- **News source expansion** to NewsAPI Business / Bloomberg / Reuters — P1 design doc §1 already defers; only revisit if Stage 4-A ships AND headline volume (not calendar event count) is the marginal lever, AND operator approves $449+/mo recurring spend.
- **Two-tower fusion architecture** — explicitly do NOT escalate from concatenation to two-tower without evidence that concatenation got 0pp lift while embeddings + alignment were verified correct. Premature optimization risk.
- **Replacing the runtime `_evaluate_news_risk` agent** — that path stays as-is on its VADER/RSS heuristic. Stage 4-A may LATER augment that agent with embedded-news features, but Stage 4-A's deliverable is feature-fusion into the direction head, not the news-risk agent.

---

## 8. Done-criteria (Stage 4-A complete when ALL of these are true)

1. `compute_normalized_features` accepts `news_features_df` kwarg; backward-compatible default `None`.
2. PCA fit-on-train-only verified by integration test asserting val/test rows do not influence fitted components.
3. Inference contract version bumped to `2026-05-08-v2` with `news_pca`, `news_event_class_count_columns`, `lookback_window_hours` all required.
4. Per-pair H1 holdout numbers measured for 7 majors with news-augmented features, on identical splits as Phase 5.B.
5. Decision applied per §5.3 + §5.4; result documented in a follow-up post-mortem at `docs/superpowers/plans/2026-05-XX-stage-4a-results.md`.
6. USD_JPY revert-guard fired or not fired, recorded explicitly.
7. CLAUDE.md "News/macro pipeline (P1)" section updated with Stage 4-A outcome and corrected baseline numbers.

---

## 9. Open questions exposed by this spec (operator decisions)

1. **Lookback window default**: P1 §7 Q5 locked `[4, 24]` symmetric for both training AND runtime. This spec defaults to a single `lookback_window_hours = 24`. Operator: keep [4,24] two-block (more columns, more capacity) or single-window 24h (simpler, smaller feature blowup)? **Recommendation: single 24h for Stage 4-A; expand to [4,24] in Stage 4-B if Stage 4-A holdout is in the 52-57% ambiguous band.**
2. **R3 revert thresholds (60% USD_JPY, 54% NZD_USD)**: 2pp guard band is conservative. Operator can tighten to 1pp (4σ stricter) or loosen to 3pp (allow more news exploration on already-working pairs). **Recommendation: 2pp; consistent with the 1.29pp binomial std on 1500 windows.**
3. ~~**PCA n_components**~~: **RESOLVED 2026-05-08 — locked to 64** based on Agent B's empirical variance-retention measurement on real FinBERT financial-headline embeddings (k=64 → 95.6%, k=32 → 88.6%). The P1 doc's theoretical k=32 was wrong.

4. **🚧 BLOCKER — Historical news backfill**: Agent A surfaced that the public ForexFactory JSON mirror (`nfs.faireconomy.media/ff_calendar_thisweek.json`) is **rolling-this-week-only**. There is NO public endpoint for `lastweek`, `lastmonth`, etc. The current implementation supports forward-only weekly accumulation (cron a fetch from today onward), but it CANNOT backfill the 22-month history that price-only training uses. Three paths the operator must choose between:

   | Option | Cost | Time-to-first-Stage-4-A-holdout | Risk |
   |---|---|---|---|
   | **4-A.i — Forward-accumulate from today** | $0; deploy weekly cron; revisit in 4-8 weeks once enough data has accumulated | ~6-12 weeks | Slow; market regimes may shift before we have enough train rows |
   | **4-A.ii — HTML scraper for FF calendar archive** | ~1-2 days engineering; FF's `/calendar?week=last-N` HTML pages are scrapable but rate-limit hostile | ~1 week | Brittle to FF page changes; may need rotating user-agent + proxy |
   | **4-A.iii — Pivot to different historical source** | NewsAPI Business: $449/mo (full history). FRED economic-data API: free but events-only, no headlines. Alpaca Benzinga firehose: free with brokerage account, US-equity-tilted | $0-$449/mo + 2-3 days engineering | Different schema → potential redesign of NewsEvent dataclass and category map |

   **Recommendation**: **4-A.ii — the HTML scraper is the lowest-cost path to a real Stage 4-A holdout.** Operator approval required because it adds ~1-2 days to the timeline and introduces a brittle dependency on FF's page structure. If declined, fallback is 4-A.i (forward-accumulate, defer Stage 4-A holdout to ~July 2026).

5. **Holiday rows** — Agent A's fetcher emits FF holiday/non-economic rows with `relevance_score=0.0`, category=OTHER. They consume ~10% of cache space but contribute zero embedding signal. **Decision**: drop at fetch (saves space) or keep for auditability (spec defaults to keep)?
