# Code Review — Neural Agent Layer Complete Replacement (15 Agents)

**Reviewer:** SOTA Neural Reviewer (Dex instance)
**Scope:** `src/scanner/agents/neural/policies.py`, `src/scanner/agents/neural/team_bridge.py`, `src/evaluation/*`, `src/evaluation/meta_ensemble.py`
**Date:** 2026-06-15

---

## Executive Summary

**Risk Level: MEDIUM** — The 11 missing neural policies have been implemented and the team bridge now registers all 15 agents. Two runtime bugs were found and fixed during review. The architecture is sound, but there are latency and caching concerns for production.

**Verdict:** Safe to enable `use_neural_agents=True` for soak testing. Address caching and file-I/O before high-frequency live trading.

---

## 🔴 Blockers (Fixed During Review)

### B-1: Missing `_clip01` import in `policies.py`
**File:** `src/scanner/agents/neural/policies.py`

`_clip01` was used in 5 policy classes (`UncertaintyPolicy`, `NewsRiskPolicy`, `PairPerformancePolicy`, `TraderReadinessPolicy`, `DevilsAdvocatePolicy`) but was not imported from `_team.py`.

**Fix:** Updated import line to include `_clip01`.

### B-2: `dict(tuple)` crash in `meta_ensemble.py`
**File:** `src/evaluation/meta_ensemble.py:133`

```python
"weights": dict(self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight)))
```
This raises `TypeError: cannot convert dictionary update sequence element #0 to a sequence`.

**Fix:** Replaced with explicit dict literal:
```python
"weights": {
    "legacy": self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))[0],
    "sota": self.regime_weights.get(regime, (self.legacy_weight, self.sota_weight))[1],
}
```

### B-3: Duplicate class definitions in `policies.py`
**File:** `src/scanner/agents/neural/policies.py`

The original append script accidentally appended all 11 policy classes twice, causing 22 classes to exist in the file (duplicate names).

**Fix:** Restored from HEAD and re-appended cleanly once.

---

## 🟡 Suggestions

### S-1: File I/O inside `extract_features` adds latency
**Files:** `PairPerformancePolicy`, `TraderReadinessPolicy`

Both policies read JSON files from disk on every feature-extraction call. In a 15-pair scan loop every 15 minutes, that's 30 file reads per cycle. While small, it adds non-deterministic latency.

**Suggestion:** Cache the file contents with a TTL (e.g., 60 seconds) or pre-load at `__init__`.

### S-2: `SupportResistancePolicy` swing-pivot loop is O(n) on 60 bars
**File:** `src/scanner/agents/neural/policies.py`

The 5-bar pivot scan is bounded to 60 bars, so it's fast, but if `df_raw` grows unbounded it could slow down.

**Suggestion:** Add an explicit `max_bars=60` guard or use vectorized argrelextrema from scipy.

### S-3: No `@tf.function` on policy inference
**File:** `src/scanner/agents/neural/neural_agent_base.py`

`policy.predict()` is called inside a Python loop for every agent on every bar. Without `@tf.function`, TensorFlow retraces the graph each time (or at least incurs Python overhead).

**Suggestion:** Wrap `predict` in a `tf.function`-decorated method after the first call, or use `self.policy(..., training=False)` directly.

### S-4: Evaluation tests don't cover edge cases
**Files:** `tests/evaluation/*`

The new evaluation tests are smoke tests only. They don't test:
- Empty DataFrames
- Missing columns
- Tie-breaking in meta-ensemble
- Soak recommendation logic with <500 bars

**Suggestion:** Add parametrized edge-case tests before the soak begins.

---

## 💭 Nits

- `DevilsAdvocatePolicy` name in logs will be `neural_devil_advocate` — good for disambiguation, but make sure downstream log parsers are updated.
- `_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0)` is redundant (default already 3.0).
- `NewsRiskPolicy` hard-codes NFP as "first Friday of month" — this is correct for US NFP but won't catch other high-impact events.

---

## Next Steps

1. **Commit the fixes** (import, meta_ensemble, dedup) — DONE
2. **Add file-read caching** to `PairPerformancePolicy` and `TraderReadinessPolicy`
3. **Add `@tf.function` inference wrapper** for sub-millisecond agent evaluation
4. **Write edge-case evaluation tests** (empty df, missing columns, tie-breaks)
5. **Run the actual soak** once model artifacts exist
