# Phase 8 — Frankenstein Audit & Architectural Simplification

> **Status**: PROPOSAL — needs operator approval before any code changes.
>
> Operator's prompt (2026-05-10): "if we did 15min fix and improved above 10% then the bot is a Frankenstein". Correctly diagnosed. This doc captures the audit and proposes the actual fix.

## 1. The smoking gun

A 15-minute hand-tune of 4 multiplicative constants in the Phase 44 calibrator produced **+19% lift** on live confidence scores (raw 0.582 → calibrated 0.111 became 0.582 → 0.132). When arbitrary constants move the system that much, those constants were never derived from anything — they were engineer guesses stacked on engineer guesses.

The patch is itself the disease. Real fix: architectural, not numerical.

## 2. The confidence pipeline today (the Frankenstein)

Path from "transformer outputs raw confidence" to "is this trade tradeable?" passes through **19 layers**:

| # | Layer | File | Operation |
|---|---|---|---|
| 1 | Tiny Transformer prediction | modular_inference | raw output 0.0-1.0 |
| 2 | LightGBM momentum (per-pair) | modular_inference | independent scoring |
| 3 | LightGBM risk (per-pair) | modular_inference | independent scoring |
| 4 | Ridge confidence (per-pair) | modular_inference | overlapping signal |
| 5 | 15-agent weighted vote | _team.py | aggregation |
| 6 | Sub-inference 3-window vote | engine.py:2971-3028 | re-runs models on shifted data |
| 7 | Phase 44 Platt sigmoid | confidence_calibration.py:540 | ×~0.56 |
| 8 | Phase 44 agreement_factor | confidence_calibration.py:707 | ×0.5-1.0 |
| 9 | Phase 44 disagreement_factor | confidence_calibration.py:710 | ×0.85-1.0 |
| 10 | Phase 44 meta_weight | confidence_calibration.py:719 | ×0.7-1.0 |
| 11 | Phase 52 isotonic overlay | _team.py:1705 | another compression |
| 12 | Memory nudges (per-agent, stacking) | agents/memory.py:438 | -0.10 each |
| 13 | Phase 47 expectancy modifiers | _team.py | regime ×0.8 |
| 14 | Tier 6 meta-learner overrides | _team.py | veto ×1.5, conf ×0.6-0.75 |
| 15 | Circuit breaker vetoes | _team.py:1841 | binary block |
| 16 | 3-gate sub-inference (mom/conf/risk) | engine.py:3126 | binary block |
| 17 | Sub-inference vote count gate | engine.py:3019 | needs 2/3 windows |
| 18 | Adaptive sub-inference threshold | adaptive_sub_inference_threshold.py | dynamic vote requirement |
| 19 | Final 52% confidence gate | engine.py:3099 | binary cut |

**Each layer was added to address a specific past failure.** Each made sense individually when added. Together they crush every signal regardless of model quality.

## 3. The three architectural critiques

### Critique 1: Multiplicative compounding is mathematically broken

When N factors are all ≤1, they compound geometrically:

| Factors | Each | Product |
|---|---|---|
| 1 | 0.7 | 0.7 |
| 2 | 0.7 | 0.49 |
| 4 | 0.7 | 0.24 |
| 6 | 0.7 | 0.12 |

The Phase 44 calibrator alone applies 4 factors (Platt × agreement × disagreement × meta_weight). Adding "more safety" via more factors makes signal worse, not better. There is no defensible reason for a 0.582 transformer output to become 0.132 — unless the transformer is wrong, but in that case we shouldn't trust it at all.

**Fix:** Pick ONE calibration mechanism. Multiplying multiple "safety" factors is a category error.

### Critique 2: Overlapping signals from same data are double-counted

Transformer (#1), LightGBM momentum (#2), LightGBM risk (#3), Ridge confidence (#4), and the 15 agents (#5) all predict from the same OHLCV bars + derived features. They cannot independently disagree about a price move and have it count as 5 separate disagreements — that's one disagreement counted 5 times.

The agreement_factor (#8) and disagreement_factor (#9) are computing variance over what is essentially the same signal source filtered through different model architectures. High variance there is structural, not informational.

**Fix:** Pick the load-bearing signal source (transformer, post-Phase-5.D). Use the others as honest features inside it, not as separate voters.

### Critique 3: Hard vetoes mixed with soft penalties is the worst of both

The bot has BOTH:
- **Soft penalties** (multiplicative confidence reduction) from agreement, disagreement, memory nudges
- **Hard vetoes** (binary block_trade) from the same agents

So a borderline-failing agent both reduces confidence (penalty) AND can binary-block (veto). One mechanism would be defensible. Both compound.

**Fix:** Each agent contribution is either a hard gate (block or pass) or a soft input (signal, no gate). Not both.

## 4. The clean architecture

| Concern | Frankenstein (today) | Clean (proposed) |
|---|---|---|
| **Confidence calibration** | Platt × agreement × disagreement × meta_weight (4-factor multiply) + Phase 52 isotonic overlay | Single isotonic regression fit on real recent journal outcomes |
| **Disagreement** | Multiplicative penalty 0.85-1.0, smooth across all signals | Hard gate: block if std > 0.30 across the agents that survived gate-checks |
| **Memory bias** | -0.20 nudge per agent stacking | Hard gate: block trade on pair if recent rolling hit-rate < 0.40 (single check) |
| **Vote aggregation** | Weighted vote + sub-inference + 3-gate × 15 agents | Single weighted average + 1 threshold |
| **Vetoes** | Each of 15 agents can veto + sub-inference vote + 3-gate + circuit breaker | Three explicit hard gates: model_disagreement, recent_hit_rate, calibrated_confidence |
| **Gate threshold** | 52% on heavily-compressed signal | 60% on calibrated signal (no compression) |
| **Number of layers** | 19 | 5 |

In this architecture, the 0.582 transformer output flows: isotonic-mapped on real data → ~0.55. Hard gates checked: low disagreement ✓, recent hit rate ≥0.40 ✓. 60% threshold check: 0.55 < 0.60, no trade. **Honest "no" once.** Frankenstein says "no trade" 13 different ways and you can't tell which one mattered.

## 5. Migration plan

This is a multi-week refactor. Phased to avoid one-shot risk.

### Phase 8.A — Shadow mode (Week 1, ~3 days)
- Build new calibrator (`confidence_calibration_v2.py`) with the clean architecture, but DON'T wire it to gates
- Run in parallel: every scan computes both the Frankenstein output AND the v2 output, logs both
- Compare on weekday data: which one would have traded? Which one wins on outcomes?
- **Reversible**: zero production change; just observation
- **Decision gate**: at end of week, look at shadow results. If v2 doesn't trade more or has worse signal-to-noise than Frankenstein, abandon. If v2 looks better, proceed to 8.B.

### Phase 8.B — Switch with kill-switch (Week 2, ~2 days)
- Add `enable_clean_calibrator: bool = False` config flag
- When True, use v2 calibrator + remove the 5 layers it replaces (Phase 44 Platt × 4-factor multiply, Phase 52 isotonic, memory nudges, agreement-factor multiplicative, disagreement-factor multiplicative)
- Layers retained as hard gates: disagreement, hit-rate, threshold
- Default flag = False (Frankenstein remains active)
- **Operator flips flag** for a week of production observation
- **Reversible**: flip flag back to revert

### Phase 8.C — Decommission (Week 3, ~2 days)
- If 8.B production observation shows v2 is the winner, remove the dead Frankenstein layers
- Clean up code paths
- Update tests
- Document what was removed and why

## 6. Risks

| Risk | Mitigation |
|---|---|
| v2 produces too many trades, drawdown spikes | 8.A shadow mode catches this before production. 8.B kill switch reverts in seconds |
| v2 calibration overfits to recent journal | Use isotonic regression with cross-validation; require ≥500 trades for fitting; fall back to passthrough if insufficient |
| Removed layers were load-bearing for some failure mode I haven't accounted for | Each removal in 8.C is its own small commit with explicit revert path. Operator can revert individual layers |
| Multi-week timeline misses opportunity | The Frankenstein bot is currently NOT TRADING (gates blocking everything). A clean architecture that does trade is a strict upgrade — opportunity cost is "more weeks of zero trades" which is the status quo |

## 7. Open questions for operator

1. **Authorize the work?** Multi-week investment, code that ships will replace ~1500 lines of accumulated patches. Reversible at every phase boundary, but the cumulative change is large.

2. **Hit-rate gate cutoff (0.40)?** Set arbitrarily; could be 0.35, 0.45, or use rolling Sharpe instead. Want operator's view on what "this pair is being a problem" means in concrete terms.

3. **Drop the LightGBM momentum/risk and Ridge confidence per-pair models?** They're vestigial after Phase 5.D made the transformer the load-bearer. If keep, they're features not voters. If drop, simpler but loses some belt-and-suspenders.

4. **Phase 8.A duration?** 3 days seems like enough shadow data to compare; 1 week is more conservative. Operator's risk tolerance.

## 8. Why this matters

The +19% from my 15-minute patches is not the win it looks like. It's the smell of a system that needs structural work, not more parameter tuning. Every previous fix in this codebase that involved "tune the magic number" landed us where we are now: 19 layers, all "necessary," all compounding into a system that won't take a trade on conf=0.582.

Either we keep adding patches and the bot remains paralyzed, or we refactor and find out whether the underlying ML actually has signal. There's no middle ground that doesn't involve more Frankenstein.

---

## Appendix A — Where the 19 layers came from

Each layer landed via a specific "we lost money on X" incident or "this pair is risky" pattern. Tracing them:

- Phase 44 Platt sigmoid: Phase 44 (US-280), 2026-03-XX. Calibration architecture proposal.
- Phase 47 expectancy modifiers: regime-aware risk shrinkage, post-loss-streak.
- Phase 52 isotonic: belt-and-suspenders on Platt, in case Platt is wrong.
- Phase 67 (US-393) adaptive sub-inference threshold: dynamic relaxation when signals near threshold.
- Phase 76 max_model_disagreement: hard gate added after specific incident.
- Phase 91 trend agent hard veto: from 2026-04-15 Trade 1220 ADX=1 incident.
- Phase 91 staleness uncertainty hard-block: from 2026-04-15 10-loss streak.
- Phase 93 MR composite veto: from 2026-04-17 14-loss streak with MR voting NO.
- Phase 98 soft agent gate: penalty path for `agent_passed=False`.
- Memory nudges: from RL feedback design, intent was to learn pair-specific reliability.
- Tier 6 meta-learner: graph-attention agent consensus.

Each was reasonable. Together they're the Frankenstein. None of the layers measure how MUCH they compound with the others before they fire — each is locally optimized.

## Appendix B — Concrete diff sketch (for reviewer estimation)

Files to modify (estimated line counts):
- `src/scanner/confidence_calibration.py` — replace `_combine_confidence_components` (currently 50 lines) with new isotonic-only path (~20 lines)
- `src/scanner/agents/_team.py` — remove Phase 44 + Phase 52 calls, remove memory nudge stacking, remove Phase 47 modifiers (~150 lines net removal)
- `src/scanner/engine.py` — collapse sub-inference vote into a single hard gate (~80 lines net removal)
- `src/scanner/results.py` — drop `circuit_breakers_triggered`, `agent_soft_penalty_applied` fields and downstream consumers (~30 lines)
- `src/scanner/agents/memory.py` — convert nudges to a single `should_block_pair(pair) -> bool` query (~50 lines net removal)

Tests:
- New: `tests/test_clean_calibrator.py` (~200 lines)
- New: `tests/test_phase8_shadow_compare.py` (~150 lines)
- Existing: 5-10 calibration tests need updates (~100 lines diff)

Net effect: codebase shrinks by ~300 lines, adds ~350 lines new, simpler architecture.
