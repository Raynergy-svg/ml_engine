# Label Leak Audit — 2026-05-05

Follow-up to `docs/confidence_model_leak_investigation_2026-04-30.md`. Scope:
identify any other supervised head whose label is closed-form derivable from
columns that are also in the feature matrix `X`.

Audit rule (from `.claude/rules/improvement.md`):
> ALWAYS audit any new supervised label for closed-form derivability from
> input features — if the label can be written as `f(x_1, ..., x_k)` where
> `x_i` are also features, it is a leak even if temporal splits are correct.

Six heads in scope (confidence already known leaky and fixed-in-progress).

---

## Summary

| Head | Label file:line | Feature source file:line | Verdict | Notes |
|---|---|---|---|---|
| Direction (Transformer) | `src/core/modular_data_loaders.py:1897-1957` | `src/core/modular_data_loaders.py:1722-1862` (all numeric except OHLCV/target) | **CLEAN** | Label = sign of `close[i+lookahead]/close[i]`; pure forward price, no name overlap, no derivability from row-`t` features |
| Direction baseline (HistGB) | same as above (uses output of `load_direction_data`) | same as above | **CLEAN** | Trainer at `src/training/trainers/histgb_trainer.py:66-100` consumes the same X/y from `load_direction_data` |
| Volatility regime (TCN, 4-class) | `src/core/modular_data_loaders.py:588-658` (`compute_volatility_regime`); applied at `:3586` | `src/core/modular_data_loaders.py:3540-3555` | **CONFIRMED LEAK** | Label = bin(percentile rank of `atr_pct_14[i]` over prior 100 bars); `atr_pct_14` is in X |
| Trend/chop/MR regime (TransformerRegime, 3-class) | `src/core/modular_data_loaders.py:1568-1633` | `src/core/modular_data_loaders.py:1521-1547` | **SUSPICIOUS** | Branch decisions read `adx[i]`, `rsi[i]`, `zscore_20[i]` (all in X); future-return only acts as a confirmation gate, not the primary signal |
| Momentum (LightGBM) `momentum_score` head | `src/core/modular_data_loaders.py:2980-3028` | `src/core/modular_data_loaders.py:2942` → `:1328-1340` | **CONFIRMED LEAK** | Label = `mean(\|returns[i-W:i]\|) / norm_factor`; `returns_1..returns_20` are in X |
| Momentum (LightGBM) `acceleration` head | `src/core/modular_data_loaders.py:3032-3036` | same as above | **CONFIRMED LEAK** | Label = `1[mom_score[i] > mom_score[i-5]]`; uses no future data, deterministic in features |
| Risk (LightGBM) `expected_drawdown_pct` head | `src/core/modular_data_loaders.py:3157-3170` | `src/core/modular_data_loaders.py:3111-3127` → `:1341-1354` | **CLEAN** | Label = max forward MAE/MFE over `drawdown_horizon` future bars; no current-bar derivability. Author's note at `:3146` confirms a previous leak (`atr_pct * 2`) was deliberately replaced |
| Risk (LightGBM) `streak_prob` head | `src/core/modular_data_loaders.py:3199-3210` | same as above | **CONFIRMED LEAK** | Label = `clip((volatility_10[i]/volatility_20[i] - 0.8)/0.4, 0, 1)`; both `volatility_10` and `volatility_20` are in X |
| Meta-labeler (XGBoost, triple-barrier) | `src/risk/triple_barrier.py:99-236` (barrier sim); `src/training/meta_labeling.py:134-205` (meta-labeling) | `src/training/meta_labeling.py:177-197` (features + primary_proba) | **CLEAN** | Triple-barrier labels are pure forward OHLC simulation; meta-label = `1[primary_correct]` is derived from forward outcomes, not current features |

**Heads audited: 7** (direction primary + direction baseline counted as one label-source; multi-output LightGBM heads expanded into momentum_score / acceleration / drawdown / streak; meta-labeler counted once).

**Verdict tally:** 4 CLEAN · 1 SUSPICIOUS · 4 CONFIRMED LEAK (momentum_score, acceleration, streak_prob, volatility_regime).

---

## Confirmed leaks

### Volatility regime (TCN)

**What.** `compute_volatility_regime()` at `src/core/modular_data_loaders.py:615-649`
reads `atr_pct_14[i]` from each row, ranks it against a backward-looking
100-bar window of `atr_pct_14`, and bins the percentile rank into
`{LOW, NORMAL, HIGH, EXTREME}` using fixed cuts at 0.25 / 0.60 / 0.85.

The X matrix at `src/core/modular_data_loaders.py:3540-3555` includes
`atr_pct_5, atr_pct_10, atr_pct_14, atr_pct_20, volatility_5, volatility_10,
volatility_20, bb_width_norm, bb_width_20, high_low_range, returns_1,
returns_5, returns_10, volume_ratio_10, volume_ratio_20, adx`.

`atr_pct_14` is in X **as the exact column** used to derive the label. The
TCN consumes a sliding sequence (`seq_len` bars, see
`src/training/trainers/tcn_trainer.py:642-645`), so each training sample
contains both the current `atr_pct_14[i]` and a window of prior values from
which the percentile cut can be reconstructed almost exactly. The label is
deterministic in features the model can see.

There is one weak guard at `:3571` excluding the `volatility_regime`
column — but only because that column already contains a continuous version
of the same answer (0-1 percentile from FeatureEngineering). The fix did not
extend to the underlying `atr_pct_*` columns that drive the bin assignment.

**Why it matters.** TCN inference will report high accuracy on the
volatility-regime classification task and that accuracy will be largely
mechanical. Live, the regime gate (`min_regime_for_trade=2` at
`VolatilityRegimeConfig`) is acting on a label any rule-based binner of
`atr_pct_14` could compute deterministically — the model is round-tripping
that computation through a TCN forward pass for nothing. Per the confidence
investigation parallel: the model is approximating an arithmetic formula it
already has the inputs for.

This also propagates downstream: the regime tier flows into
`ScannerConfig.atr_sl_multiplier_*` profile selection, position sizing, and
the LOW-regime SL rule (`.claude/rules/trading.md`). All of those decisions
are tracking a label-from-features identity rather than a learned signal.

**Suggested fix path (operator decides).** Replace the rolling-percentile
target with an outcome-derived volatility target (e.g. realized
`high-low_range` or realized return-vol over the next N bars). Or drop
`atr_pct_*` columns from X for this specific head and keep only structural
features (returns, BB-width, volume, ADX). Or — most honest — replace this
head with the rolling-percentile rule directly and remove the model.

### Trend/chop/MR regime (TransformerRegime) — SUSPICIOUS, partial

**What.** Label assignment at `src/core/modular_data_loaders.py:1600-1633`:

- MEAN_REVERT branch: `(rsi[i] < 25 or rsi[i] > 75 or |zscore_20[i]| > 2.0)` AND
  a future-return confirmation
- TREND branch: `(adx[i] > 25 AND consistency[i] > 0.6)` AND a future-return
  confirmation
- CHOP branch: everything else

`adx`, `rsi`, `rsi_norm`, `zscore_20` are all in X (lines 1524-1529). The
gating predicates of the label rule operate on values that are also features;
only the confirmation gate (the `future_return` check) injects forward
information. Practically, the model can perfectly learn the pre-confirmation
buckets (RSI/ADX/zscore thresholds), and the label noise comes only from
the `future_return` filter — meaning the model can score very well by
matching the input thresholds alone, even if the future-confirm gate is
where the real predictive value sits.

This is not as severe as a closed-form leak (the label is not a deterministic
function of features — the future-return confirmation can flip a candidate
TREND/MR back to CHOP), but the input thresholds do most of the work.
Expected behavior: high apparent accuracy on the trivial `adx > 25` /
`rsi extreme` cases, with most of the model's "skill" being threshold
recognition.

**Why it matters.** The 3-class regime head is used to gate trade type
(let directional models trade in TREND, fade in MR, skip in CHOP). If the
model is largely a re-encoder of `adx[i] > 25` and `rsi[i] in extremes`, the
gate can be replaced with an if-then rule. Worse, the *confidence* of the
classification will be inflated — Buddy will be confidently wrong when
those thresholds fire but the future-return confirmation should have
overruled them.

**Suggested fix path (operator decides).** Either (a) drop `adx`, `rsi`,
`rsi_norm`, `zscore_20` from X for this specific head and keep only price
structure / volatility features, forcing the model to learn the
threshold-discovery task itself; or (b) replace the label with a
purely-forward-derived regime classification (e.g. ADX of the next N bars,
or realized trend-vs-chop measure on forward returns).

### Momentum head — `momentum_score` regression

**What.** At `src/core/modular_data_loaders.py:2986-3028`:

```
raw_momentum_all[i] = mean(|returns[i-momentum_window:i]|)   # default W=10
momentum_score[i]   = min(raw_momentum_all[i] / norm_factor, 1.0)
```

The label is a backward-looking rolling mean of absolute returns ending at
row `i`. The X matrix includes (line 2942 → `get_normalized_feature_names()['momentum']`
at `:1328-1340`): `returns_1, returns_2, returns_3, returns_5, returns_10,
returns_20, log_returns_1, atr_pct_5/10/14, volatility_5/10, rsi_norm,
stoch_k_norm, macd_norm, macd_hist_norm, volume_ratio_5/10, volume_zscore`.

`returns_1` is the exact return at row `i`. `returns_10` is `(close[i] -
close[i-10])/close[i-10]`. The label `mean(|returns[i-10:i]|)` is a closed-form
function of (`returns_1`, `returns_2`, ..., `returns_10`) — not exactly the
same statistic, but bounded above and below by feature combinations the
model can recover. With LightGBM's 150 trees of depth 6, this is fitted
trivially.

**Why it matters.** `momentum_score` is consumed by the momentum agent
(weight 0.95-1.0 in `_BASE_WEIGHTS`) and contributes to WVS in trade gating.
A high momentum score therefore trips trades, but the score is a
mechanical re-encoding of features the rest of the system already has. The
agent's `passed=True` is decoupled from outcome and is correlated with
"market is currently moving" by construction, regardless of whether moving
markets are a profitable thing to trade in this regime.

**Suggested fix path (operator decides).** Replace label with a
forward-looking momentum target: e.g. `mean(|returns[i:i+W]|)` (forward
realized momentum), or a forward-return magnitude class. Both make the
label dependent on future bars, breaking the closed-form derivability.
Alternatively, drop `returns_*` columns from this head's X, but that's
brittle because `atr_pct_*` and `volatility_*` are also strongly correlated
with the label.

### Momentum head — `acceleration` classifier

**What.** Same file, lines 3032-3036:

```
acceleration[i] = 1 if momentum_score[i] > momentum_score[i-5] else 0
```

This is a function of two label values that are themselves backward-looking
functions of features. There is no future information at all in this label.
**The label can be computed from features alone, with zero forward
data.**

**Why it matters.** The acceleration head is downstream-consumed as a
boolean ("momentum growing?"). The model is learning to compare two
rolling sums of features the model can already see — a feature-engineering
problem, not a prediction problem. If the binarized output is later treated
as a forward-looking signal ("momentum is accelerating, expect continuation"),
that interpretation is unfounded — the label says nothing about future bars.

**Suggested fix path (operator decides).** Re-target as forward
acceleration: `1[momentum_forward[i:i+W] > momentum_backward[i-W:i]]`. Or
drop the head entirely if the answer is "is current rolling momentum higher
than the rolling momentum 5 bars ago" — that's a one-liner of pandas.

### Risk head — `streak_prob`

**What.** At `src/core/modular_data_loaders.py:3199-3210`:

```
streak_prob[i] = clip((volatility_10[i] / volatility_20[i] - 0.8) / 0.4, 0, 1)
```

The X matrix (line 3111 → `get_normalized_feature_names()['risk']` at
`:1341-1354`) explicitly contains `volatility_5, volatility_10, volatility_20`.
The label is a deterministic linear-and-clipped expression of two columns
that are both in X. LightGBM with 150 trees of depth 6 will recover this
within numerical precision.

This is the same leak class as the confidence head: a deterministic formula
over named features, where the named features are also in X.

**Why it matters.** `streak_prob` flows into the risk agent and into
position sizing. Currently the model is reporting a metric that is
arithmetically equivalent to `clip((vol_10/vol_20 - 0.8)/0.4, 0, 1)`,
which is information already present in the system. Same pattern as
the confidence head — the calibration layer downstream is doing the real
work of mapping the score to outcomes; the model is a redundant
intermediate step.

Note that the **`expected_drawdown_pct` head in the same file is CLEAN**
(forward MAE/MFE, no overlap), so the leak is isolated to the `streak_prob`
output — the bug is in one of the two outputs, not the whole head. The
author's comment at `:3146` even shows awareness of the leak class
("Previous approach used `atr_pct * 2` which is trivially recoverable from
input features. Now we compute actual max adverse excursion") — but the
fix was applied to the drawdown output only, not to streak_prob.

**Suggested fix path (operator decides).** Replace `streak_prob` with a
forward-derived label: e.g. probability of `k` consecutive losing bars
within the next horizon, or forward realized loss-streak length. This
matches the spirit of the drawdown fix and follows the same pattern as
the confidence-leak remediation.

---

## Clean heads — evidence

### Direction (Transformer) and Direction baseline (HistGB)

Label generator at `src/core/modular_data_loaders.py:1924-1952`:

```
pct_change = (close[i+lookahead] - close[i]) / close[i]
y[i] = 1 if pct_change > threshold else 0 if pct_change < -threshold else 0.5
```

The label uses only `close[i+lookahead]`. The feature matrix excludes `close`
explicitly (see exclusion set at `:1724-1725`: `{'open', 'high', 'low',
'close', 'volume', 'time', 'timestamp', 'date', 'target', 'label',
'direction', 'y', 'target_direction'}`) and additionally filters anything
matching `target|label|future|forward` at `:1731-1732`. Returns are present
in X but they are *backward* returns; the label is a *forward* return — no
closed-form overlap.

`temporal_split` is chronological, no shuffle; `RobustScaler` fit on train
only; threshold filtering uses train-only percentiles.

`HistGradientBoostingDirectionTrainer` consumes the same X/y from
`load_direction_data` (per `src/training/model_config.py` registry and
`src/training/trainers/histgb_trainer.py:66-95`), so the same reasoning
applies.

### Risk head — `expected_drawdown_pct`

Label at `:3157-3170`: forward max adverse excursion across the next
`drawdown_horizon` (default 10) bars, computed from `low[i+1:]` and
`high[i+1:]` only. No current-bar derivability. Author's comment at
`:3146` documents that the previous proxy (`atr_pct * 2`) was a leak and
this rewrite is the fix.

### Meta-labeler (XGBoost, triple-barrier)

Triple-barrier at `src/risk/triple_barrier.py:99-236` simulates TP/SL outcome
from `entry_price` forward, using `simulate_tp_sl_outcome` over the next
`max_horizon_candles` bars. Label is `barrier_hit` ∈ {TP, SL, TIMEOUT}
mapped to `direction_label` ∈ {0, 1} and `confidence_label` ∈ [0, 1] —
all functions of forward OHLC only.

`MetaLabeler.generate_meta_labels()` at `src/training/meta_labeling.py:134-205`
labels `meta_y[i] = 1[primary_binary[i] == actual_binary[i]]` where
`actual_outcomes` come from forward simulation. `primary_predictions` IS in
X — but that is the canonical López de Prado setup and not a leak: the meta
label is "is this prediction correct?" given the prediction, which by
construction needs the prediction as input. The forward outcome breaks the
closed-form derivability.

---

## Cross-cutting observations

1. **The same author who wrote the confidence leak wrote three of the four
   confirmed leaks here.** All four leaks (volatility regime, momentum_score,
   acceleration, streak_prob) follow the identical pattern as the confidence
   leak: a deterministic formula over named indicator columns, with those
   same columns in X. The confidence investigation
   (`docs/confidence_model_leak_investigation_2026-04-30.md:42-45`) noted
   the docstring openly admits "This IS learnable because these are
   computed from the same features!" — the same mental model produced
   each of these.

2. **The `expected_drawdown_pct` head shows the author DOES know the
   pattern is wrong.** The comment at `:3146` ("Previous approach used
   `atr_pct * 2` which is trivially recoverable from input features. Now we
   compute actual max adverse excursion over the next `drawdown_horizon`
   bars") is the exact reasoning that needs to extend to the other four
   heads. The fix template is right there in the same file, just not
   applied to the sibling outputs.

3. **Temporal-split correctness is universal across heads.** Every loader
   uses `temporal_split` chronologically, fits scalers/percentiles on
   train only, and respects `gap` parameter. The leaks are all structural
   (label-from-features), not temporal — the same finding as the
   confidence investigation.

4. **Two heads are clean for the right reasons.** The direction head and
   the `expected_drawdown_pct` output both have labels that are functions
   of strictly future bars, with no row-`t` feature derivability. These are
   the templates to apply elsewhere.

5. **One head is "suspicious" but partly defensible.** The trend/chop/MR
   regime label has feature-threshold gating but a future-return
   confirmation that injects genuine forward information. It's not a clean
   leak, not a clean head; it warrants a closer empirical look (suggested
   below) before judgment.

---

## Suggested follow-up empirical checks (do not run in this dispatch)

For each CONFIRMED leak head, the corresponding sanity check is the same
as the confidence-investigation appendix:

1. Compute the label formula directly on val rows (no model). Predict
   that as the val output. Compare against the trained LightGBM/TCN val
   predictions. If MAE/F1 deltas are within a percent or two, the model
   is just approximating the formula.

2. Drop the label-input columns from X. Retrain. R²/F1 should fall to
   the level of "what other features can predict this formula" — usually
   much lower. The drop magnitude is the leakage delta.

3. Replace the label with a forward-derived equivalent (per "Suggested
   fix path" above, per head). Retrain. Expect R²/accuracy to drop sharply
   (this is correct — that drop is the system being honest about how hard
   the real prediction problem is).

For the SUSPICIOUS regime head:

4. Train two variants — full features and features minus `adx, rsi,
   rsi_norm, zscore_20`. Compare F1. The delta tells you how much of the
   model's score is coming from threshold-recognition vs structural
   features (returns, BB, volume, MACD). A small delta means the
   confirmation-gate is doing real work; a large delta means the head is
   essentially a threshold-encoder.

---

## Pointers

- Confidence leak investigation that motivated this audit:
  `docs/confidence_model_leak_investigation_2026-04-30.md`
- Audit doc that flagged the broader risk:
  `docs/ml_architecture_audit_2026-04-30.md:140` (action item #1)
- Direction labels: `src/core/modular_data_loaders.py:1684-2000`
- Volatility regime labels: `src/core/modular_data_loaders.py:580-658`,
  `:3501-3646`
- Trend/chop/MR labels: `src/core/modular_data_loaders.py:1491-1677`
- Momentum labels: `src/core/modular_data_loaders.py:2907-3069`
- Risk labels: `src/core/modular_data_loaders.py:3076-3259`
- Meta-labeler: `src/training/meta_labeling.py:134-205`,
  `src/risk/triple_barrier.py:99-236`
- Improvement rule that mandates this audit:
  `.claude/rules/improvement.md` ("ALWAYS audit any new supervised label
  for closed-form derivability from input features")
