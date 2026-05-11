# 2026-05-11 — `disagreement_hard_block` Root-Cause Diagnosis & Gated Unblock

## Status: APPROVED 2026-05-11 (auto mode); dispatching B and C immediately.

## Problem

After the operator restarted Buddy with the smart-profile loosening commit
(`07a1f96`), the momentum-cascade health fix (`1939e9e`), and the correlation
net-exposure cap (`c38b166`), trading is still 100 % blocked. Brain feed
(`.claude/brain/feed.jsonl`) post-restart shows **10/10 rejections** firing
`circuit_breakers=['disagreement_hard_block']` or `['uncertainty_block']`.

The block does NOT come from actual model-to-model disagreement
(`src/scanner/ensemble_conflict.py` logs `scores=[0. 0. 0.],
disagreement=0.000` — those legacy attrs are dead). It fires from
`src/scanner/agents/_team.py:2167-2169` — a **heuristic-indicator vote**:

```python
oppose = sum(1 for h in heuristics if h < 0)
model_disagreement = oppose / total  # 5 indicators
```

The 5 indicators are close vs SMA_20, RSI vs 50, sign of returns, close vs
SMA_50, MACD histogram sign. The hard floor at `_team.py:2295` reads from
`disagreement_hard_floor` (default `0.50`, NOT overridden in smart profile);
trades hard-block when `≥ 0.50`. With 5 heuristics that's `≥ 3/5 oppose`.

Compounding factor: brain feed shows **30 SHORT vs 4 LONG** signals (88 %
SHORT). Heuristics are trend-followers and trend is still positive (prices
above SMA_20 / SMA_50, MACD histogram positive on most pairs), so every
SHORT signal hits ≥3/5 heuristic opposition → hard-block fires.

Three failure modes could explain the SHORT bias:

1. **Genuine market signal** — model is correctly bearish on USD strength;
   heuristics are lagging trend-followers and are wrong.
2. **Contract drift** — `feature_pipeline_version` mismatch or scaler
   identity-fingerprint regression per CLAUDE.md "Train↔Inference Contract
   Gates"; inference is feeding bad inputs and the model is hallucinating.
3. **Training imbalance** — the trained transformer's OOS holdout was
   already SHORT-skewed (training label imbalance); fix is a balanced
   retrain, not a config tweak.

Choosing between these is the load-bearing question this work answers.

## Design — Two-team parallel dispatch with gated third action

### Team B — SHORT bias diagnostic

**Type**: AI Engineer. **Mode**: READ-ONLY (no code edits, no commits).

Investigates the three failure modes against on-disk evidence:

| Failure mode | Evidence to check | Conclusion if positive |
|---|---|---|
| Genuine market signal | OOS holdout LONG/SHORT distribution per pair; recent macro events (USD strength); ensemble training-set period | VERDICT-MARKET → trigger A |
| Contract drift | `transformer_direction.meta.pkl` `feature_pipeline_version` vs runtime constant in `src/core/modular_data_loaders.py`; `scaler.var_` for identity-fingerprint pattern | VERDICT-CONTRACT-DRIFT → queue inference contract investigation; do NOT apply A |
| Training imbalance | OOS holdout class distribution; `trained_data/models/EUR_USD/transformer_direction.meta.pkl` training-set label counts | VERDICT-TRAINING-IMBALANCE → queue balanced retrain; do NOT apply A |

**Deliverable**: written report (≤ 600 words) with one of the three
verdicts, file:line evidence for each load-bearing claim, and a confidence
level. No code changes.

### Team C — ensemble_conflict legacy attribute fix

**Type**: Backend Architect. **Mode**: CODE + TESTS.

Replaces `_team.py:2245-2247`:
```python
tcn_score   = _safe_float(getattr(ctx.analysis, "tcn_score", 0.0), 0.0)
ridge_score = _safe_float(getattr(ctx.analysis, "ridge_score", 0.0), 0.0)
rf_score    = _safe_float(getattr(ctx.analysis, "rf_score", 0.0), 0.0)
```

These attributes are legacy from the pre-Transformer era. The current
inference path populates Transformer + HistGB + LightGBM-momentum scores
on `ctx.analysis` under different attribute names. Team C must:

1. Locate the modern attribute names by reading `src/core/modular_inference.py`
   (the path that builds `ModularSignal` / writes onto `PairAnalysis`).
2. Replace the three legacy reads with reads against the modern attributes,
   preserving the same sign convention (-1 / 0 / +1 → SHORT / HOLD / LONG).
3. Real no-mock tests covering:
   - Scores all populated and disagreeing → ensemble disagreement > 0.
   - One score missing → graceful fallback (uses the two available, does
     not crash).
   - All three agreeing → ensemble disagreement near 0.
4. Three-layer wiring verification per `.claude/rules/improvement.md`.

**Deliverable**: 1 commit, all new tests green, integration grep proving
the new attribute reads return non-zero values for a representative pair
on at least one cycle.

### Team A — heuristic floor relaxation (GATED)

**NOT dispatched until Team B returns VERDICT-MARKET.**

If B returns VERDICT-CONTRACT-DRIFT or VERDICT-TRAINING-IMBALANCE: A does
NOT fire. Operator is briefed with the alternative recommendation
(contract investigation OR retrain).

If B returns VERDICT-MARKET: dispatch Team A (Backend Architect) to:

1. Add `"disagreement_hard_floor": 0.65` to smart profile (with comment
   citing the B verdict and the 2/5-passes / 4/5-blocks math).
2. Real no-mock tests:
   - `test_smart_profile_relaxed_disagreement_hard_floor` (= 0.65)
   - `test_two_of_five_heuristics_oppose_passes`
   - `test_four_of_five_heuristics_oppose_still_blocks`
   - `test_three_of_five_passes_after_relaxation`
3. Single commit, ≤ 4 production lines + tests.

## Coordination

- B and C run concurrently. B touches no files; C touches `_team.py` and
  a new test file. No file conflict.
- After both return: I read B's verdict and either dispatch A or report
  the alternative path. I do NOT auto-apply A.
- All work lands on `main`. No remote push.

## Out of scope

- Retraining the direction model (separate workstream if B returns
  VERDICT-TRAINING-IMBALANCE).
- Refactoring `_team.py` heuristics (5-indicator vote is preserved as-is;
  only the FLOOR threshold is adjustable in Option A).
- Fixing other code paths that read legacy `tcn_score/ridge_score/rf_score`
  outside `_team.py` (Team C scopes only the ensemble_conflict consumer).

## Success criteria

- B verdict + evidence delivered.
- C commit lands; ensemble_conflict scores are non-zero on at least one
  real scan post-deploy.
- A commit lands only if VERDICT-MARKET.
- Operator restart picks up the changes; brain feed shows
  `disagreement_hard_block` rate drop OR a clear non-A path queued.
