# ML Engine — Strategy & Modernization

> Relocated from CLAUDE.md (2026-06-09) to keep the always-loaded instruction file lean.
> This is strategic context, not per-task coding rules. The **load-bearing guardrails**
> distilled from here live in CLAUDE.md ("Strategy guardrails"); read this doc when making
> roadmap / model-architecture / data-pipeline decisions.

## Modernization stance (May 2026 baseline)

**Standing goal:** keep the ML stack within striking distance of SOTA. The bot's edge is the
ML signal, not the heuristics. If we fall behind on time-series foundation models /
news-fused embeddings / calibrated uncertainty, we're optimizing the wrong thing.

### Active gap audit (refresh quarterly or when SOTA shifts)

| Layer | Current | SOTA | Gap |
|---|---|---|---|
| Direction head | Custom Transformer, M15 — **no shippable artifact** (2026-06-10: the 70.0%/56.4% holdouts were the anchored-OBV artifact; honest v2 retrain ~52% val, >10% gap, quarantined) | Chronos / TimesFM / Moirai (60M-700M params) | **SMALL — FM zero-shot underperforms by 20pp+ at this timeframe (empirically tested)** |
| Confidence | Ridge head + calibration JSON, ad-hoc | Conformal prediction (mapie, crepes) | MEDIUM — drop-in wrapper |
| Volatility regime | Heuristic bridge (was leaky TCN) | Heuristic acceptable | SMALL |
| Macro/news signal | NONE | FinBERT / text-embedding-3 fused to price | **LARGE — the only remaining lever; price-only M15 has no shippable edge (2026-06-10 verdict, dad8624)** |
| Pair coverage | EUR_USD, GBP_USD M15-trained | All 16 majors+crosses at M15 | OPERATIONAL — retrain at `--granularity M15 --candles 65000` |
| Sequence length | 60-bar context | 1024+ via state-space models (Mamba) | SMALL — not the bottleneck |
| Sizing/decision | Decoupled prediction + DynamicPositionSizer | Decision Transformer / offline RL on journal | DEFER — needs >5K trades first |

### Investment priority (revised 2026-05-08 on empirical evidence)

1. **News/macro embedding pipeline** — the only remaining lever once data alignment is fixed; price-only M15 confirmed unshippable 2026-06-10 (USD_JPY/EUR_USD/GBP_USD ~52% val with >10% gap after the anchored-feature fix — prior ~70% was the OBV artifact).
2. **Pair expansion at M15** — operational, not architectural. Retrain remaining majors at `--granularity M15 --candles 65000`. ~3 min/pair on M1 Metal.
3. **Conformal prediction confidence layer** — replaces leaky confidence head + calibration JSON in one shot.
4. ~~Foundation-model direction head~~ — **DEMOTED**. Chronos-T5-small (60M, zero-shot, h=24) = 44.5%; Chronos-T5-base (200M, h=24) = 46.0%; current custom Transformer (19k params, M15) = 70.0% as then-measured (later shown to be the anchored-OBV artifact; honest ~52%). Even against the honest baseline Chronos underperforms; 4× model scale produced +1.5pp. The bottleneck was 3 data/training/holdout bugs (scaler null, lookahead mismatch, primary_tf misalignment), not model class.
5. **State-space models** — defer, not the bottleneck.
6. **Decision Transformer** — defer, needs trade-journal volume.

**Key lesson:** we assumed architecture was the gap; the data showed 3 alignment bugs masquerading as a model-quality problem. Always test "we need a better model" against "we have the data wrong" first. A 19k-param model with correct data beat a 200M-param model with wrong data by 20+ pp.

**Non-goals:** don't optimize the custom Transformer beyond bug fixes (wrong horse); don't ship "we built our own" when pretrained alternatives exist; don't research-tour — pick the highest gap, wire the simplest version, measure, iterate.

**Keep:** W&B control plane, walk-forward validation, EMA/SWA tricks, the meta-pipeline / Tier 7 loop, heuristic bridges as fallback, per-pair routing. Modernization replaces models, not infrastructure.

## Key design decisions (rationale)

- **Soft uncertainty blocking (confidence penalty) over a hard circuit breaker.** Uncertainty/disagreement reduce confidence rather than hard-stopping — EXCEPT the explicit staleness case (hard-block on `uncertainty_score > 0.35` when `max_component_age_days > 7`, see CLAUDE.md trading invariants). Don't replace the soft penalty with a blanket hard breaker.
- **Decoupled prediction + sizing.** Direction/regime heads predict; `DynamicPositionSizer` sizes. Don't fuse them.
- **Per-pair routing is the only supported runtime path** (joint dir deprecated). See `.claude/rules/improvement.md`.

## Empirical signal ceiling (do not re-litigate without new evidence)

Price-only EUR_USD direction caps at ~52% intraday (M15/H1/H4), faint at daily (53.9% balanced); GBP_USD/USD_JPY daily ~62%. News fusion tested 2026-05-27 produced no lift. **The P3 news-embedding experiment (2026-06-12) also produced no shippable lift** (see below). **Don't re-run news / FM / more-data experiments** without a materially different setup.

## News/macro pipeline (P1) — TESTED 2026-06-12, NO SHIPPABLE LIFT (shelved)

**VERDICT (2026-06-12, P3 complete):** the news/macro embedding lever — the "only remaining" one — has now been properly tested with real FinBERT semantic embeddings through the production pipeline, and it does **not** unlock M15 EUR_USD. Controlled result, same 65k CSV the price-only baseline used:

| Arm | val_acc | balanced | train_acc | gap | shippable? |
|---|---|---|---|---|---|
| Price-only (a5748e0, baseline) | 0.5192 | — | 0.7637 | 0.2445 | no (quarantined) |
| **News-fused (FinBERT+FF, 32-PCA)** | **0.5288** | 0.5348 | 0.8108 | **0.2820** | **no (quarantined)** |

- **News lift = +0.96pp** (0.5192 → 0.5288) — within single-holdout noise, far below the +3pp ship threshold; val 52.88% < 55% required.
- **Overfitting got WORSE, not better**: gap widened 24.45% → 28.20%. Adding 40 news columns (32 PCA @ 94% explained variance + 8 event-class counts) gave the model more memorization capacity without generalizable directional signal — exactly the failure mode for a head already overfitting at 24%.
- **This is a trustworthy negative.** A latent bug was found+fixed first: `FeatureEngineering.create_features` returns a tz-NAIVE `DatetimeIndex`, so the news block's `fetch_events` call raised and the broad `try/except` silently degraded every prior run to price-only. Fixed in `load_direction_data` (UTC-localize the index). Verified news genuinely activated this run: 6,481 events embedded → (6481,768), 40 cols appended, `news_pca_active=true`.
- **Repro:** `python scripts/train_news_experiment_eur_usd.py` (RESULT line + `logs/news_experiment_*.log`). Artifact auto-quarantined by the 10% gap gate; no live transformer exists (fail-closed correct).
- **Low-value follow-ups (only if operator insists):** `news_pca_n_components=8` (8 comps already capture 93-94% variance — would trim params, unlikely to add 3pp); H1/H4 timeframe; different asset class. Note: the runtime news-inference path is still unimplemented (P4), so even a passing model couldn't serve trades without that work.
- **Bottom line for the live-money goal:** with the last identified ML lever now tested and dry, there is no price/news M15 edge to ship. **The bot stays halted** — this is the correct outcome, not a failure to fix.

### Original P1 plan (for reference — now closed by the verdict above)

The only remaining lever — price-only M15 confirmed unshippable 2026-06-10 (dad8624); instrumentation (cost-aware backtest + v2 feature contract) is now honest enough to measure any lift this pipeline produces.

- **Design doc**: `docs/superpowers/plans/2026-05-08-news-macro-signal-design.md`
- **Recommendations**: ForexFactory calendar (primary) + RSS headlines (secondary) for the prototype; FinBERT (`ProsusAI/finbert`, 768-dim, free, M1-friendly) embedder; concatenation fusion at the DataFrame level (lowest blast radius).
- **Integration point**: `compute_normalized_features` in `src/core/modular_data_loaders.py` — optional `news_df` arg joining per-bar news features after the price features.
- **Runtime path untouched in P1-P3.** The existing `_evaluate_news_risk` agent stays as-is; Phase 4 may *augment* it.
- **Phase plan**: P1 design+stubs (done) → P2 implement `ForexFactoryNewsSource.fetch_events` + `FinBERTEmbedder.embed` → P3 implement `align_news_to_bars`, backfill 22mo EUR_USD, retrain M15, compare to the HONEST price-only baseline (~52% val on the v2 window-invariant pipeline; re-measure at P3 time via the cost-aware backtest, not a single holdout). **Decision rule (recalibrated 2026-06-11)**: ship only on ≥3pp val lift over the re-measured price-only baseline AND a positive-expectancy cost-aware backtest that passes the 10% gap gate; 1.5-3pp lift → disambiguate with `text-embedding-3-large`; less shelves. The old ≥73.0%-vs-70.0% rule was calibrated against the anchored-OBV artifact — void. P4 expand to remaining majors if lift confirmed.
- **Validation hypothesis**: ≥3pp M15 lift over the honest price-only baseline (~52%, v2 pipeline) + positive cost-aware expectancy confirms signal worth shipping.

## Inference contract (Phase 2.A+B)

The model's saved meta sidecar (e.g. `transformer_direction.meta.pkl`) is a contract telling the inference path how to reproduce the training-time feature distribution. Any train↔inference drift produces silent OOD predictions. The enforced **rules** live in `.claude/rules/improvement.md` ("Train↔Inference Contract Gates"); this is the reference detail.

**Required meta keys** (saved by trainer, read by `gates._load_transformer`): `feature_names` (authoritative column order), `scaler` (fitted StandardScaler with REAL per-column stats, never the identity `var_=1.0±1e-9` double-fit fingerprint), `regime_quantiles` (`{q25,q50,q75}` from training-time ATR), `regime_atr_col` (which atr feature drove the quantiles), `feature_pipeline_version` (semver; `gates` refuses a model whose version ≠ the runtime constant in `modular_data_loaders.py`).

**Inference path**: `gates.evaluate_transformer` → `compute_normalized_features` → `_build_transformer_inference_matrix(feature_names)` → per name, take from compute output OR compute regime one-hot from saved quantiles OR **refuse** (no silent zero-fill) → apply saved `scaler.transform()` → keras predict.

**Tripwires**: `_assert_scaler_not_identity` fires ERROR on the double-fit fingerprint; a missing required column logs a contract-gap warning and returns `(None, 0.5)`.

Audit: `docs/superpowers/plans/2026-05-08-pipeline-reconciliation-phase1-audit.md`.
