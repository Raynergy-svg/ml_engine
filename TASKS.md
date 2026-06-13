# TASKS — safe surface only (see CLAUDE.loop.md for the Danger Zone)
# Source: 2026-06-11 three-agent production audit (Code Reviewer / DevOps / API Tester).
# Done-criteria are immutable once written: build to the criterion, never edit it to fit.
# Ordered by value. Danger-Zone findings from the same audit live in REVIEW-QUEUE.md — not here.

## ★ PRIMARY MISSION — Daily FX factor portfolio (src/factor/)
# Operator-approved strategy (tasks/prd-fx-factor-portfolio.md). Today's honest verdict
# is a FAIL: carry+trend net Sharpe ~ -0.2 on 7 USD-only majors. Work these toward a real
# edge. src/factor/ship_gate.py is OFF LIMITS (pre-registered bar). After ANY src/factor
# change, run `python scripts/run_factor_backtest.py` and paste the VERDICT line as evidence.
# A FAIL is a valid result — report it honestly, never weaken the gate to pass.

- [ ] FP-1 (HIGHEST) — Cross-section breadth: add G10 CROSSES (e.g. AUD_JPY, NZD_JPY, EUR_JPY,
      GBP_JPY, EUR_GBP, AUD_NZD, EUR_CHF) so carry/value rank over a real cross-section instead
      of 7 USD-only legs (the current rank is mostly a disguised USD-cycle bet — this is the
      single change most likely to lift carry's Sharpe; PRD OQ-3). Extend PAIRS/loader to fetch
      + cache + validate the crosses; recompute signals cross-sectionally over the wider set.
      Done: backtest runs over ≥12 instruments incl. crosses; each new instrument's history is
      validated (≥10y, else a named warning); a window-invariance test still passes; the report
      JSON records the full instrument list; VERDICT line pasted in the iteration reply.

- [ ] FP-2 — Value factor (US-004): add src/factor/reer.py fetching BIS REER (monthly, free),
      build value_signal into the book (already a pure function in signals.py). Causality +
      window-invariance tests in the US-002/003 style; no mocks.
      Done: backtest re-run WITH and WITHOUT value, both net-Sharpe numbers recorded in
      docs/factor-portfolio-results.md; new reer.py has ≥4 no-mock tests; VERDICT line pasted.

- [ ] FP-3 — Rebalance cadence study (PRD OQ-4): make rebalance cadence a parameter (weekly vs
      monthly) and report net-of-cost Sharpe + turnover for both. Trend's 48x/yr turnover is a
      prime cost sink. Done: a no-mock test asserts monthly turnover < weekly; results table in
      docs/factor-portfolio-results.md; VERDICT lines for both cadences pasted.

- [ ] FP-4 — Test coverage for the data layers that currently have none: src/factor/rates.py
      and the gap/validation paths of data_loader.py. Real disk, no mocks, no network (use a
      written tmp cache + synthetic frames). Done: tests/test_factor_rates.py with ≥5 tests green.

- [ ] FP-5 — Auto-report: extend scripts/run_factor_backtest.py (or a sibling) to append a
      dated row (date, book, net Sharpe, maxDD, +years, verdict) to a docs/factor-ledger.md
      table on every run, so drift in the honest number is tracked over time. Done: ledger row
      written atomically; a test covers the row formatter.

## ──────────────────────────────────────────────────────────────────────────

- [x] P0 — Fix live-journal wipe hazard in `tests/test_ewma_wiring.py:419`. (2026-06-12: fixture
      added to the test signature at :410; `grep -n "temp_state_dir" tests/test_ewma_wiring.py`
      shows it; file's 32 tests pass. Live journal verified intact — the runner's earlier full-suite
      run was interrupted before reaching this file, by luck.)
      `test_ewma_state_persistence_handles_save_error` writes `trained_data/trade_journal_rl.json`
      via a RELATIVE path but does not take the `temp_state_dir` chdir fixture (defined :84-92),
      so running pytest from repo root overwrites the live RL journal with `[]`.
      Done: test takes the fixture; `grep -n "temp_state_dir" tests/test_ewma_wiring.py` shows it
      in this test's signature; full file's tests pass.

- [ ] P1 — Write a no-mock test suite for `src/scanner/backtest_gate.py` (currently ZERO tests;
      it sits in the model promotion chain). Tests only — do NOT edit backtest_gate.py itself.
      Use real classes + tmp_path per the No-Mock rule. Cover: pass path, fail path, missing/corrupt
      input artifact, and the fail-closed default.
      Done: new tests/test_backtest_gate.py with ≥6 tests, zero mock imports, all green.

- [ ] P1 — Migrate the 4 post-2026-05-01 No-Mock violations to real classes:
      `tests/test_audit_20260512.py` (worst — mocks ScannerConfig/PairAnalysis, financial objects),
      `tests/test_scheduled_jobs.py`, `tests/test_wandb_control_plane.py`,
      `tests/test_online_retrainer_wandb_integration.py`. W&B is an external API: use
      `@pytest.mark.integration`/skip per the rule, not mocks.
      Done: `grep -l "MagicMock\|mock.patch\|Mock(" ` on those 4 files returns nothing; suites green.

- [ ] P2 — Add real assertions to assertion-free smoke tests, starting with
      `tests/test_regime_gates.py` (7 assertion-free funcs), then `tests/test_shutdown_hardening.py` (5).
      Done: every `def test_` in those two files contains ≥1 assert on state or return value.

- [ ] P2 — Investigate the 4 permanent `xfail(strict=False)` at `tests/test_buddy_scan.py:46,102,136,181`.
      Fix the underlying failure if it's in tests/, or document why it can't be (then propose in
      REVIEW-QUEUE.md if the fix needs Danger Zone edits). Done: each is green+strict, or a dated
      justification comment + REVIEW-QUEUE entry exists.

- [ ] P2 — Tests that read real production model artifacts (`tests/test_keras_model_loader.py:99-153`,
      `tests/test_pickle_load_under_tui_stderr.py:29`, `tests/test_phase2b_inference_contract.py:48`)
      break when artifacts are quarantined (current state). Add tiny fixture artifacts under
      tests/fixtures/ or skipif-on-missing. Note: test_phase2b uses the DEPRECATED joint path —
      retarget to per-pair routing while touching it.
      Done: full suite green with trained_data/models/*/transformer_direction.keras absent.

- [ ] P3 — Docs: `docs/supervisor_console_runbook.md` and `docs/tier7-architecture.md` — verify the
      runbook's commands and file paths still match reality post-2026-06-10 (fail-closed runtime,
      v2 feature pipeline); fix stale claims with evidence. Done: every command in the runbook
      verified runnable or corrected.
