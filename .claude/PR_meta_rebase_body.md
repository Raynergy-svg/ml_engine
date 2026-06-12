## Summary

Rebases the meta-cybernetic change pipeline onto current `main` (was forked from a 30-commit-stale base) and enables it in the `smart` (live) profile with the runtime kept Claude-free.

Three coupled landings:

1. **Rebase** — original branch was forked from `2f7c1c9`, never updated as Phase 91/93/95/96 + the homework system + US-604/605/606 landed on `main`. Cherry-picked the two original commits (`feat(meta)` + `chore(gitignore)`) onto current main; one config conflict in `smart`-profile dict (Phase 91/93 fields adjacent to meta fields), resolved by keeping both halves.

2. **Canary-flow fix** — surfaced during the rebase. Three coupled bugs:
   - `ConfigAdjuster.collect_adjustment` returned `None` always; now returns `Optional[str]` (proposal_id on success).
   - `_write_pending_proposal` short-circuits on duplicate `(source, key, value)` signatures but used to discard the freshly-allocated id, so callers got a phantom id whose row was never written. Now returns the id actually present.
   - `AdjustmentApprover.approve()` doesn't remove proposals from pending — it just flips status to `"approved"`. So a redeploy of the same `change_id` collided with the resident "approved" row and silently no-oped. Fixed by including a per-attempt counter in the canary source string: `meta_manager_canary:{change_id}:attempt_N`. `revert_by_id` still source-substring-matches on `change_id`.
   - `StagedDeployer` now takes `adjustment_approver` kwarg; orchestrator constructs `AdjustmentApprover()` inside the `enable_meta_manager=True` branch and passes it. Canary captures proposal_ids from each `collect_adjustment`, calls `approver.approve(pid)` on each, then runs `apply_adjustments`.

3. **LLM-free + smart-profile enable** — new `meta_manager_use_llm: bool = False` flag. When False (default everywhere), orchestrator passes a no-op `specialist_invoker` to `MetaManager`. The constitution + scorecard + `revert_by_id` are pure-Python and gate every change without Claude. Specialists enrich the change ledger but never gate. `smart` profile flipped to `enable_meta_manager: True`; other profiles keep `False`.

## What's gated by what
- **Constitution** — 7 invariants in `.claude/rules/constitution.json` (no_widen_max_risk, no_disable_supervision, freshness_floor, min_eval_confidence, broker_change_requires_test, rollback_required_for_live, rr_floor_holds). Pure Python.
- **Scorecard** — `ChangeEvalHarness` runs `ReplayValidator` + `QAPipeline` + `pytest`. Pure Python.
- **revert_by_id** — walks `_history` in reverse; resets `setattr(config, key, old_value)`. Pure Python.

## Test plan
- [x] 209 tests across meta + Phase 90/91/93/95/96 + event_bus + adjustment suites pass with no regressions
- [x] 3 new tests for the canary flow: full-chain, redeploy regression, missing-approver guard
- [x] 1 fixed test (`test_revert_by_id_restores_old_values`) — was written against pre-US-508 single-step apply
- [x] End-to-end smoke with Claude-trap sentinel — canary advance + rollback chain runs zero Claude calls
- [ ] First live cycle on practice — restart Buddy, watch `.claude/meta/changes.jsonl` populate (expected: empty until an incident routes through `route_incident`)
- [ ] First incident routed through meta — verify dispatch step `meta_pipeline_drain` runs every 5 cycles (orchestrator step #17, `critical=False`)
- [ ] Verify constitution.json loads cleanly on the first canary attempt

## Pre-existing failures (verified independently against plain main, NOT rebase-induced)
- `test_inbox_e2e.py` × 6 — `InboxScreen._proposals_cache` missing in non-mounted tests, introduced today by `a59e828` (mtime-cache perf commit)
- `test_phase53_pair_weights::test_pair_weights_update_in_sync` × 1 — stale `inspect.getsource` grep
- `tests/scanner/test_circuit_breaker_enforcement` — collection error (`tests.scanner` not a package)

## Operational posture
- Pipeline ships **enabled in `smart` only**. Live profile fires the meta dispatch every 5 cycles.
- LLM is **disabled by default** (`meta_manager_use_llm=False`). Flip to `True` only if you want enrichment text in `.claude/meta/changes.jsonl`.
- Approval queue at `.claude/approval_queue/{pending,approved,rejected}/`.
- Change ledger at `.claude/meta/changes.jsonl` (append-only).
- Constitution at `.claude/rules/constitution.json` — edit without code changes; it's reloaded on each `Constitution()` instantiation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
