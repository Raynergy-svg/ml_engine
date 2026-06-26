# Session handoff — 2026-06-16 (infra/architecture audit + retrain)

Audit-and-retrain session. System remains **halted / demo-only**. No live trading occurred.
This doc records what was done, what could NOT be done, and a prioritized TODO so the next
session can pick up without re-deriving context.

## Done this session

- **Full infra/architecture audit** (5 specialist passes): training, data, architecture,
  realized track record, autonomous control plane, SOTA wiring, SOTA model artifacts.
- **Ran `./buddy --demo`** — TUI renders correctly (after installing rich/textual/pandas);
  scanner shows OFFLINE / no heartbeat (consistent with halted state). Live mode not run
  (needs TF/torch; system halted anyway).
- **Retrained 4 per-pair transformer_direction models** (H1, cached candles). Results in
  `docs/2026-06-15-retrain-results.md`. Summary (balanced accuracy, chance=0.50):
  EUR_USD 0.562 (gap 16.6% → QUARANTINED), USD_JPY 0.512, USD_CAD 0.526, AUD_USD 0.627.
- **Confirmed gap-gate works**: EUR_USD auto-quarantined; its prior model removed, not replaced
  → EUR_USD currently has NO live transformer model.

## What I could NOT do (and why)

1. **Pull OANDA trade data / train neural agents.** No OANDA credentials in this environment
   (`OANDA_*` env all unset; only `.env.local.toggles.example` present). Network to
   api-fxpractice.oanda.com:443 is OPEN, so the blocker is purely missing creds. Even with
   creds it would not train agents: OANDA stores fills/P&L, NOT the per-agent verdicts
   `train_neural_agents.py` needs, and the only outcome history is ~16 losing demo trades
   (trainer needs ≥50/agent). Agent training requires the system to run in demo long enough
   to generate verdict-tagged outcomes first.
2. **Train GBP_JPY / EUR_GBP / AUD_NZD.** No local H1 cache in `market_data/`; would need an
   OANDA fetch (blocked by #1).
3. **Validate the AUD_USD outlier.** balanced 0.627 is a SINGLE train/val split, not
   walk-forward OOS. `val>train` flag + best-epoch-3 + prior model was 0.528 → treat as LOW
   confidence / probably a favorable split. NOT run through `walkforward_validation.py` yet.
4. **Run live `./buddy`.** TF/torch not installed for live inference; system halted.

## TODO — next session (prioritized)

1. **[load-bearing] Walk-forward validate AUD_USD.** `walkforward_validation.py` on AUD_USD.
   Decision rule: balanced ≥ ~0.54 across purged folds with CI excluding 0.50 → first
   price-only signal worth investigating. Else (expected ~0.52) → no-edge conclusion stands
   across all 4 approaches (intraday price-only, news fusion, factor portfolio, fresh retrain).
2. **Disarm the `sota_ab` random-weight landmine.** `config.py:456-461` sets
   `use_sota_inference=True` / `use_neural_agents=True`; inference then loads a nonexistent
   `sota_model.keras` and SILENTLY falls back to a random-init `RawSequenceModel`
   (`hybrid_inference.py` ~L82). Not the default profile, but one config selection arms it.
   Make the missing-artifact path REFUSE, not fall back to random weights.
3. **Decide fate of the SOTA modules.** All 11 are untrained shells (random numpy weights,
   no artifacts; TimesFM is an explicit stub returning 0.5). `MetaStrategyAgent` is LIVE in
   the smart profile (`enable_meta_learning=True`) and deactivates agents on an empty
   meta-learning state — either train it or disable `enable_meta_learning` until trained.
4. **Fix orphan config flags.** `use_sac_execution`, `use_cql_rebalancer`, `sac_min_lots`,
   `position_sizing_mode` are read via `getattr(...,False)` but DO NOT EXIST in `config.py`
   → permanently dead. Either add the dataclass fields or remove the dead call sites.
   `DiffusionMarketModel` is behind `QuickBacktester.stress_test` which has zero callers.
5. **Replace the No-Mock-violating SOTA wiring tests.** `test_{sac,cql,hybrid}_*_wiring.py`
   use MagicMock/patch (forbidden by `.claude/rules/improvement.md`) and assert string
   presence, not live behavior. They give false "wired" confidence.
6. **Finish joint-fallback deprecation** (improvement.md steps 3–5 still pending: engine
   startup filter, `_get_pair_evaluator` return None, delete joint loading).
7. **EUR_USD has no live model** (quarantined). Decide: retrain with stronger regularization
   (it overfit to 16.6% gap) or leave unmodelled.

## Environment reproduction (ephemeral container)

This container ships numpy/sklearn only. To retrain / run TUI again:
`pip install tensorflow-cpu lightgbm xgboost rich textual pandas`. Training cache was staged
from `market_data/oanda_<PAIR>_H1_*.csv` → `trained_data/cache/training_data/<PAIR>_H1_25000.csv`;
run trainer with `--skip-preflight` (bypasses the OANDA-cred preflight; uses cache).

## Bottom line for the operator

No demonstrated edge across four independent approaches. The retrain reproduced the ~52%
ceiling on 3/4 pairs; AUD_USD is the only open question and is unverified. The engineering
(measurement harness, gap-gate, factor machinery) is sound; the signal is not — pending the
AUD_USD walk-forward, which is the single cheapest test that could change the verdict.
