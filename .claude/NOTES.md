# NOTES — live working memory (survives compaction)

> This is the **only** file I (Claude) may update without operator approval. It holds *state*, not
> doctrine. New decisions go to INTENT, new failure modes go to LESSONS, new patterns go to a skill
> — all via `/evolve`, with operator approval. Keep this file short and true; prune what's stale.

Last touched: 2026-06-24 by Claude (corrected record: equity four-pillar work is dormant scaffolding,
NOT running — see In-flight + L-017). Note: `state.json halted=true` again (re-flipped; operator
directs it STAYS true). Report rule: every status claim states running:yes/no, verified from disk (L-017).

## Operating mode — delegated authority (2026-06-23, survives fresh sessions)

The operator delegated standing approval authority. From here on:
- **git commit + push are AUTO-APPROVED.** Commit in scoped, well-messaged commits; push when the
  work is verified green. No per-commit approval needed.
- **The orchestrator may approve `/evolve` proposals and similar on the operator's behalf** when they
  are (a) appropriate, (b) improve the bot, and (c) **backed by test/from-disk proof**. Evidence is
  mandatory — no proof, no approval.
- **IMMUTABLE ESCALATIONS — NEVER auto-approved by orchestrator or Claude; ALWAYS escalate to the
  human:** anything that relaxes a Hard NO, touches the per-trade hot path
  (`src/scanner/execution.py`, `scripts/`, `main.py`), changes `oanda_environment` off `"practice"`,
  un-halts trading, or moves real money. Surface these; route to the operator. The `/evolve` loop may
  never propose loosening a Hard NO.

## Schema (keep this file in these sections; prune anything stale every time you touch it)
1. Current runtime state  2. In-flight work  3. Blockers  4. Judgment calls (for veto)
5. Assumptions resolved  6. Active loop status (if mid-loop: cycle #, last verifier verdict, open
load-bearing question). Memory tightening rule: NOTES holds *state only*; the moment a note becomes a
durable decision/failure/pattern, move it to INTENT/LESSONS/skill via `/evolve` and delete it here.

---

## Current runtime state (verify against disk before acting — this is a snapshot)

Source: `.claude/state.json` read 2026-06-24T18:52Z (`last_actor: operator-directed-enable`).

- **`halted: false` — OPERATOR-DIRECTED UNHALT (2026-06-24T18:52:52Z).** The human operator (owns the
  halt) directed "unhalt the bot" on the PRACTICE/demo account; flipped via the sanctioned writer
  `StateEngine().set_halted(False)` (true→false), env-practice confirmed before flipping. Note: the
  morning enable's `halted=false` had since flipped back to `true` (auto-halt circuit-breaker pattern;
  the noisy "16 consecutive losses" lines were unit-test artifacts, not real trades — lifetime journal
  is 26 trades), so this is a re-unhalt.
- **OANDA PRACTICE TREND LANE IS LIVE — PLACED 4 REAL DEMO ORDERS THAT FILLED (2026-06-25, operator lane choice).**
  Operator pivoted from IBKR to OANDA practice (v20). Strategy = NON-directional TREND/managed-futures
  (price vs MA, long-or-flat, shift(1) — the validated drawdown-reducer; NOT the retired directional FX
  transformer, L-016 stands). Built `src/equity/oanda_trend.py` (candles→close panel→trend_sleeve signal→
  units→orders, halt-gated, practice-asserted) + `scripts/run_oanda_trend.py` + `get_instruments()` on the
  practice client. **Token is LIVE** (the old 401 is resolved — `.env` has a working practice PAT). Verified
  PRACTICE: base `api-fxpractice`, account prefix `101`, NAV ~$102k, was flat. Cycle EXECUTED: 4/10 FX majors
  on (GBP_JPY/USD_JPY/USD_CHF/USD_CAD), 4 market orders placed → **openTradeCount=4 confirmed on the account**.
  Account is **FX-only** (68 instruments, NO metals/indices/commodities — operator expected XAU/XAG; not on
  this account). HARD LINE intact: practice-only, hard-pinned URL, no live path.
  **UPDATES (2026-06-25 pm, HEAD f717a49, loop PID rotates):** (1) FX sizing FIXED — per-base-currency
  (`base_to_home_rate`), consistent home-notional (verifier: SIZING CORRECT). (2) Robust v20 layer
  `src/brokers/oanda_v20.py` — streaming (stream-fxpractice + reconnect/backoff), TransactionLedger (audit
  trail + realized P&L → `trained_data/oanda/transactions.jsonl`), `snapshot_account_state` (→ account_state.json
  for TUI), sentiment DATA-ONLY. (3) Safety rails: no-trade band, NAV-drawdown auto-halt (20% from peak),
  nan/nav<=0 guards. (4) LEVERAGE DIAL (operator: 0.5x too timid) — `OANDA_GROSS_LEVERAGE`/`--gross-leverage`,
  default **3x**, cap 15x; at 3x ~$76k/position, marginUsed ~11%. Honest: leverage amplifies variance not edge
  (trend = ~0-Sharpe drawdown-reducer); $1M/6mo on demo = ruin-math live, NON-GOAL. (5) SECURITY-VERIFIER
  (Security Engineer): **SAFE** — practice-only guaranteed, leverage bounded, cannot bleed unbounded, real
  fills truthfully labeled. (6) `docs/dashboard-data-contract.md` for the parallel read-only dashboard
  workstream. `trained_data/oanda/` gitignored; token in `.env.local` (gitignored, never committed, diff-scanned).
  17 no-mock tests; flake8 clean.
  **SENTIMENT LEVER TESTED → NEGATIVE (2026-06-29, ceed4b2, the deferred flag now closed).** Pre-registered
  (docs/experiment-oanda-sentiment-2026-06-29.md, written before results) OANDA position-book contrarian
  fade: S=-(NL-0.5), causal t->t+1, 521 daily books x 7 majors (OANDA serves books >=2y). Pooled IC +0.0005
  (t=0.03 ≈ 0); IS portfolio Sharpe 0.403 (t=0.49 insig.) COLLAPSES to -1.593 OOS, IC flips sign; per-year
  Sharpe +1.25/-0.51/-1.20. Fails all 3 pre-registered bars → NO EDGE, shelved (offline-only, not wired).
  Honest negative = success; closes the last genuinely-untested lever. **Separate verifier (Code Reviewer):
  NEGATIVE CONFIRMED** — causality clean (forward shift), sign clean (momentum=exact negation, |IC|≈0 either
  way), cost-independent (gross OOS Sharpe -1.562 vs net -1.593), eff-N≈521 not 3640 (only more null); even the
  most-favorable per-pair causal-demeaned timing test fails OOS (t=1.64 IS, OOS Sharpe -0.65, sign-flips 3/7).
  No hidden edge. Wrinkle noted: pre-reg+result in one atomic commit (provenance adequate-not-ironclad).
  `src/equity/oanda_sentiment.py` offline; 184MB book cache gitignored.
  **MULTI-ASSET TREND — GATE-CLEARING, the strongest finding of the whole campaign (2026-06-29, ec3182f).**
  Operator ungated a broad new-markets hunt (no spend). Pre-registered (docs/experiment-multi-asset-trend-
  2026-06-29.md, before results): ONE canonical trend rule (price>200d SMA, long-or-flat, shift1) UNIFORM
  across 21 free multi-asset ETF/crypto proxies (7 classes), per-asset streams HRP-combined + 10% vol-target,
  portfolio-level judgment. **CLEARS the ship gate: Sharpe 0.744 full / 0.829 OOS, maxDD 0.18, 26/34 pos yrs,
  34y, all 7 sub-periods positive — the FIRST lever ever to clear the gate on its own** (FX-only was gross≈0).
  Robust: no-crypto 0.79/0.91 PASS, no-overlay 0.69 PASS (not artifact). **BUT ties EW buy-hold on Sharpe
  (0.829 vs 0.895) — the edge over passive is DRAWDOWN control (18% vs 68%, ~4x), NOT alpha.** Same drawdown-
  reducer property as FX trend, now gate-clearing at portfolio breadth. Per the strict relative "+1 vs baseline"
  bar: does NOT clear (tie); per the absolute ship gate: DOES. REPORTED, NOT promoted/traded (new-market
  execution = operator decision). `src/equity/multi_asset_trend.py` offline.
  **Skeptical verifier: GATE-CLEARING CONFIRMED** — causal (triple-shift, reproduced to 3 decimals), uniform/
  no-dredge (losers FXY/UNG included), gate legit (not tautological). CAVEAT (verifier-flagged, substance
  survives): the "34y" oversells — only SPY reaches 1993, TRUE breadth (19-21 assets) only since ~2007;
  read as "~19y genuine multi-asset breadth (2007-26), SPY-anchored before." Verdict survives: post-2007
  diversified = 0.737 PASS, and the whole OOS (2017-26) is post-2007. Survivor-only ETF universe → 0.18 maxDD
  is a FLOOR not a guarantee (delisted assets absent). NET CAMPAIGN SCORECARD: every return/direction lever
  NEGATIVE (intraday 52%, daily factor ~0, multi-horizon, quality/value/lowvol/HRP, carry gate-rejected,
  news, meta-label, order-book sentiment); multi-asset trend is the FIRST + ONLY gate-clear, but it's
  drawdown-control diversification (ties buy-hold Sharpe), NOT alpha. No directional/return edge found anywhere.
  **UPDATES (2026-06-29, HEAD fa856e8):** (7) TP/SL BRACKETS live — ATR-based stopLossOnFill (entry-2*ATR,
  ON) + optional TP (OFF, trend rides) on opening orders; verified real broker-side SL on the account
  (USD_JPY SL 160.579). (8) MARGIN/LIQUIDATION GUARD — margin_scale clamps book to max_margin_util*NAV
  (50% default) regardless of leverage dial; STRESS @15x fired (scale=0.833 -> 47% margin, held); HALT rail
  fired (halted=true -> 0 orders). Verifier (Code Reviewer): SAFE. (9) account_state.json now carries
  per-position stop_loss/take_profit (read-only, for AXIOM SL/TP columns); contract updated. (10) TIER 7
  SELF-HEAL LOOP STARTED — `scripts/run_tier7_loop.py` BOUNDED headless supervisor (no headless scanner
  daemon existed; full Tier 7 is TUI-only). Writes heartbeat + bounded self-heal (config/weight only, tiered/
  budgeted/debounced) + tier7_state each tick. LIVE: PID rotates, tier7_state running:True (heartbeat fresh +
  pid alive). Explore-agent + Security-verifier (Security Engineer) confirmed BOUNDED + SAFE: cannot unhalt
  (only TUI app.py:2724 does), cannot trade (no execution import), cannot flip env, cannot promote/un-quarantine
  a ship-gate-failing artifact (retrain_gates → online_retrainer rewrites only xgb_momentum/rf_risk/
  ridge_confidence sklearn baselines IN-PLACE; zero promotion code). Self-heal events in
  tier7_state.self_heal.recent_events (real apply() results, lie-policy). running:YES honest (fresh≤90s + pid alive).
  **AUTONOMY DISCREPANCY (noted so it's not a future surprise):** `ScannerConfig.self_heal_max_autonomy_level`
  DEFAULTS TO 5 (`config.py:682`) while the self_heal docstring/`is None` branch intend LEVEL_3. At 5 the
  unattended loop would auto-apply LEVEL_5 retrains. FIXED: `run_tier7_loop.py` CLAMPS to 3 by default
  (`--max-autonomy` / `TIER7_MAX_AUTONOMY`, operator-raisable [3,5]); at 3 only reversible operational heals
  auto-apply, retrains REFUSED (unit-tested + live: status=degraded). Commits b69ad30 (SL/TP display) +
  fa856e8 (Tier 7 supervisor) + 03fd7e2 (clamp) pushed. Two background loops: OANDA trend (trades) + Tier 7
  self-heal (operational recovery only, clamped L3).
  **RESOLVED 2026-06-29 (operator-approved): flipped `config.py:682` `self_heal_max_autonomy_level` default
  5->3** so the dataclass matches its own docstring — defense-in-depth, fail-closed for EVERY `ScannerConfig()`
  consumer (not just the supervisor). DOCUMENTED BEHAVIOR CHANGE: the attended TUI scanner's self-heal now
  also defaults to L3 (was 5); operator can still raise explicitly via config / `--max-autonomy` /
  `TIER7_MAX_AUTONOMY`. No profile dict overrides it. verify_gate PASS (28 checks, hard_no_ok=True) +
  risk_monitor GREEN post-flip. This safety-TIGHTENING relaxes no Hard NO.
  **RAISED 2026-06-29 (operator-approved): `run_tier7_loop.py` autonomy default L3->L5 (FULL).** Operator
  ungated full self-heal autonomy on the supervisor. Bounds RE-DERIVED FROM DISK and HOLD at L5: the retrain
  handlers `_handle_retrain_gates` -> repo-root `online_retrainer.py` (writes ONLY xgb_momentum/rf_risk/
  ridge_confidence `.pkl` sklearn baselines via pickle.dump; ZERO halt/env/promote/champion/quarantine/
  execute/order/`.keras` patterns) and `_handle_retrain_rl_position_sizer` (marker file only). So even at L5
  the supervisor CANNOT unhalt, flip env, touch real money, promote/un-quarantine, or write a `.keras` champion
  (directional transformer stays closed). NOTE: the `online_retrainer` self_heal calls is the REPO-ROOT module,
  NOT `src/core/modular_inference.py`'s `.keras` trigger_retrain (different function, not on this path).
  ScannerConfig dataclass default stays 3 (global safe for other consumers); only the supervisor runs L5.
  Supervisor restarted L5 (running:True). Lower again with `--max-autonomy 3` if desired.
- **(SUPERSEDED) Equity-harvester H1 / IBKR-paper lane (2026-06-25 earlier):** harvester shadow loop ran
  (running:YES via oracle, simulated fills) but IBKR paper needs `ib_async`+gateway+login. Operator chose
  OANDA instead; harvester code (`scripts/run_equity_harvester.py`, runner `execute_order` inject) retained,
  not the active demo. Original framing below kept for lineage.
- **EQUITY HARVESTER H1 (lineage):** (2026-06-25, operator-authorized H1 for PAPER only).
  Re-unhalted via `StateEngine().set_halted(False)` (`automation/state_engine.py`); built H1 entrypoint
  `scripts/run_equity_harvester.py` (shadow + ibkr-paper lanes, `--loop`) driving the gated
  `run_shadow_rebalance` (added injectable `execute_order`). Re-validated SHIP_GATE on the current PIT
  universe (was hash-stale → NO_ACT; re-ran `run_equity_harvester_shipgate_pit.py` → gate_pass net_sharpe
  0.908, hash `f550709a` now MATCHES snapshot). **Oracle (`running_status.py`, the fail-closed lie-catcher)
  confirms `running:YES · process:YES · cycles_executed≥3 · live_artifacts:YES`; ledger hash-chain intact;
  20 equal-weight orders FILLED in SIMULATION.** Fills are SHADOW (simulated) — NOT real paper orders yet.
- **ACTUAL paper-broker fills BLOCKED (operator-side, precise):** equity harvester needs IBKR **paper**
  (OANDA is FX-only, can't fill US stocks). Runtime blocker captured: `ImportError: ib_async not installed`.
  To place real demo trades: (1) `pip install ib_async`; (2) IB Gateway/TWS on `127.0.0.1:7497` (paper);
  (3) IBKR paper-account login. The `--broker ibkr-paper` lane connects + places the instant those exist
  (fill path wired via `whole_share_round` + `place_equity_order`, UNVERIFIED until a live gateway tests it).
- **(LEGACY FX, retired per L-016 — not the live path):** old FX blockers (no scanner proc, OANDA 401,
  stale per-pair champions) no longer gate the live strategy; the equity harvester replaced FX direction.
- **`mode: "live"`, `status: "running"`.** mode=live is EXECUTION mode (place orders), NOT real money.
  Orders go to the PRACTICE/paper account: the order client is `OandaPracticeClient`, hard-pinned to
  `PRACTICE_API_URL = api-fxpractice.oanda.com/v3` (`src/utils/oanda_practice.py:117`) and it IGNORES
  `oanda_environment` entirely — there is NO live-URL path in the order client. Verified by separate
  agent: "Can it place a real-money order? NO" (HIGH confidence). This is why mode=live is acceptable.
- `oanda_environment: "practice"` (`src/scanner/config.py:738`) — **immutable Hard NO, untouched.**
- Gates taught (committed): risk_monitor + verify_gate alarm on `mode=live` ONLY when env≠practice;
  **env=live / real-money / ship-gate stay HARD** (env=live+mode=live → double hard alarm). L-014.
- NAV $102,183 · `open_trades: 0` · 6 per-pair transformer champions present but 36d-stale (+19
  quarantined) → bot abstains on staleness/ship-gate, so unhalting unleashes no flood of trades.
- Known residual (non-blocking, pre-existing): static env-tripwire matches `oanda_environment = "live"`
  assignment + git-diff `api-fxtrade`; a future *dict-form* profile override `"oanda_environment":
  "live"` wouldn't be caught by the tripwire — but the practice-pinned order client is the primary
  rail. Add the dict pattern IF env is ever wired into a profile dict / the client honors env.

- **close_trade halt guard (HUMAN-authorized hot-path safety-ADD, 2026-06-24).** `execute_trade` had a
  halt guard (execution.py:2093) but `close_trade` did not — an autonomous close could reach the broker
  while halted. Added a mirror guard: autonomous/programmatic closes are BLOCKED while `halted=true`
  (returns `BLOCKED: state.halted=True` before any broker call); an EXPLICIT operator close passes
  `operator_override=True`. Safety-ADD only (can only make close MORE restrictive). Separate verifier
  PASS, 4 no-mock tests (`tests/test_close_trade_halt_guard_2026_06_24.py`). Env/practice-pin/order-client
  UNTOUCHED — this was authorized for the close-guard ONLY; the "everything to live mode" instruction
  was NOT approved and NOT acted on.
  **TUI follow-up (TUI agent's job, not mine):** `src/tui/screens/trades_screen.py:874` calls
  `close_trade(...)` WITHOUT `operator_override=True`, so the operator's manual TUI close is now also
  blocked while halted (strictly fail-closed). The TUI's halt-aware `c`-key confirm must pass
  `operator_override=True` after the operator confirms. Do NOT wire this from here (outside scope).
- Branch: `ralph/equity-harvester-bot` (an equity-beta harvester workstream is in flight)

## In-flight work (from session memory, not re-verified this turn — confirm before relying)

- **STATUS-ORACLE TRAP FIXED (2026-06-29).** `running_status.py` reported ONLY the dormant equity-harvester
  (IBKR, superseded) lane -> "running:NO" even though the LIVE OANDA-trend + Tier7 lane was running -> caused a
  false "nothing running". Rewrote it to report TWO clearly-LABELED lanes: **LIVE LANE** (OANDA trend + Tier7;
  running:YES if `run_oanda_trend.py` proc alive OR `trained_data/oanda/account_state.json` fresh<=2h OR
  `.claude/heartbeat.json` fresh<=90s + pid alive) and **HARVESTER LANE** (dormant/legacy; "running:NO is
  EXPECTED, not a fault"). `--assert-running` now asserts the LIVE lane. 4 no-mock tests. Also force-killed a
  duplicate uvicorn (46367 — had a stale :8888 SSE connection; canonical listener 53816 kept, health ok).
  **DAILY MONITOR should read the LIVE lane:** `python3 .claude/loop/running_status.py --assert-running`
  (exit 0=up, 3=down); or raw disk: tier7 = `.claude/tier7_state.json` `running:true`; trader =
  `trained_data/oanda/account_state.json` mtime <=2h. NEVER the harvester lane (always NO = dormant).
- **Multi-strategy sleeve program — Phase 0 DONE; no-spend frontier exhausted (2026-06-25, running:NO,
  offline-eval only).** Plan doc `docs/multi-strategy-sleeve-architecture-2026-06-24.md` (commit 3a7038b).
  Phase-0 (commit 65015bd): built Sleeve A = multi-asset trend/managed-futures (long-or-flat ETFs,
  `src/equity/trend_sleeve.py`) + HRP-across-sleeves combiner (`sleeve_combiner.py`) + combined-book gate
  (`SHIP_GATE_book.json`). Falsifiable test {harvester+trend} vs harvester-alone, pre-registered canonical
  params, separate-verifier independently re-derived (byte-for-byte) + multiple-testing audit.
  **VERDICT (verifier-confirmed): NOT a stable Sharpe +1 — it's a DRAWDOWN-only risk-reducer.** Combined
  beats harvester on Sharpe full +0.033 / OOS +0.118 BUT only 2/4 disjoint 5yr blocks (wins=GFC+2021-25,
  loses calm 2011-20); sign-test p≈0.69; OOS ~72% overlaps the winning 2021-25 regime (not independent);
  collapses under N≈5 ideas tried. ROBUST finding: maxDD 0.229→0.166 across ALL regimes (trend goes to
  cash in crashes, corr~0.49). On the operator's "+1 Sharpe OOS" bar this is an HONEST NEGATIVE (=success).
  **Frontier state:** trend was THE strong free/orthogonal candidate (drawdown-only); carry is gate-rejected
  (EM-carry fat tail); quality needs PIT fundamentals = DATA PURCHASE (HARD STOP, operator's card —
  financial-datasets = latest-FY/$0 balance, no free PIT). NOT iterating more sleeves (anti-p-hacking;
  N already burns the significance budget). Surfaced operator decisions: (a) is tail-risk reduction a goal
  worth taking trend toward shadow/live [operator-authorized hot-path]? (b) authorize PIT data purchase to
  unblock the quality sleeve = the next real diversification lever. STOP-DONE on no-spend frontier.

- **Phase-0 trend-sleeve verdict — INDEPENDENTLY RE-VERIFIED 2026-06-25 (trust-downgraded; complete, was
  mid-run at limit).** Fresh bake-off re-run byte-identical to committed `SHIP_GATE_book.json`; separate Code
  Reviewer re-derived from source + multiple-testing audit. VERDICT: **risk-reducer-only** — drawdown
  reduction (0.229→0.166) ROBUST across all 4 sub-periods; Sharpe improvement (full +0.033/OOS +0.118) is
  REGIME-DEPENDENT (2/4 sub-blocks, sign-test p=0.69, OOS 72% overlaps winning 2021-25) → does NOT clear the
  +1 bar. Causality clean (double-shift, exec_lag=1), apples-to-apples, invariants intact. Matches prior
  record (1b1e340); verification now closed.

- **Quality factor — FREE true-PIT pipeline BUILT + RUN; ROBUST NEGATIVE (operator corrected plan to NO paid
  data; running:NO).** Paid-data path (financial-datasets Pro) SUPERSEDED — operator directed a $0 PIT build.
  Built: `src/equity/edgar_fundamentals.py` (SEC EDGAR companyfacts, TRUE-PIT via real `filed` dates, annual
  10-K, synonym-merge + cost-derived gross profit, as-originally-reported = earliest-filed), `sp500_membership.py`
  (survivorship-aware Wikipedia change-log reconstruction — 873 ever-members, 279 since-removed incl. SIVB/
  TWC/AET), `pit_quality_eval.py` (`evaluate_book_wide` = staggered-universe NaN-safe evaluator, baseline
  UNTOUCHED; forces weight=0 where price NaN so fabricated cells inert). quality_data extended for `filed`-date
  PIT + `clip_z`/`components`. **RESULT** (`trained_data/backtests/pit_quality_bakeoff.json`, 624-name PIT
  universe, 2012-2026, cov 95.5%): canonical composite **LOSES** to EW (full −0.600/OOS −0.324) — but that's
  largely a non-robust `−debt_to_equity` outlier artifact (banks/REITs/neg-equity tails; winsorize halves it).
  **Clean margin-only tilt merely MATCHES EW** (full −0.018/OOS +0.026) = **NO shippable edge.** Honest
  negative = success. LABELLED residual bias: delisting returns understated (ffill), yfinance price-availability
  survivorship (189 delisted lack prices), EDGAR XBRL floor ~2010. 23 no-mock tests green; flake8 clean.
  Caches on disk (`market_data/equity/sp500_pit_fundamentals.json`, `sp500_prices.parquet`) for re-derivation.
  **Separate verifier (trust-downgraded) CONFIRMED:** re-derived EVERY headline number exactly from cache
  (EW 0.740, canonical −0.600, margin-only OOS +0.026, 624 names, cov 0.9554); the lookahead validator
  genuinely fires (injected forgery → raised); baselines unmodified (git diff empty); residual survivorship
  is CONSERVATIVE (biases toward EW, cannot manufacture the negative); −0.600 confirmed a d/e-outlier artifact
  (max d/e 990, 303 neg-equity). Verdict: no shippable edge, ROBUST. Commits d6c63bb (receiving end) + c8be8cd
  (EDGAR/membership/eval + result). Diversification-sleeve frontier now genuinely exhausted (trend=drawdown-only,
  carry=gate-rejected, quality=no-edge) — further factor variants = dredging; operator's call.

- **Equity-harvester four-pillar self-improver — SHIPPED-TO-DISK + UNIT-TESTED SCAFFOLDING, NOT RUNNING
  (corrected record 2026-06-24; separate-verifier confirmed; supersedes prior "running/wired/now does X"
  chat framing, which was inaccurate — see L-017).** TRUE status, re-derived from disk independently:
  - Committed (capability only, never executed): N1–N5 shadow runner `src/equity/runner.py` + config flag
    `enable_equity_harvester` (`ea85f8c`); Pillar 3 decision gate `src/equity/decision_gate.py` (`d4d8aa7`);
    Pillar 2 cycle ledger `src/equity/cycle_ledger.py` (`ce74989`). **running in process: NO.**
  - **cycles executed: 0** — `trained_data/equity/` does NOT exist; `cycle_ledger.jsonl` /
    `rebalance_state.json` ABSENT. **state artifacts present (live, non-test): NO.**
  - **invoked by: tests ONLY** — `run_shadow_rebalance`/`decide_cycle`/`append_cycle` called only by their
    test files + internally in `runner.py`; no `main.py`/`src/scanner/*`/`embedded_scanner.py`/`scripts/`
    calls them. The enable flag guards code no live path reaches.
  - 27/27 unit tests pass (decision_gate 11, cycle_ledger 6, runner 10) = "shipped + unit-tested", NOT
    "functioning". Dormant until **H1** (the live invocation) — HELD for operator authorization.
  - Pillars 1 (verifier) + 4 (risk monitor): **not built** (stopped per operator). running:NO / built:NO.
  - Even if invoked now it would `REFUSE` every cycle (`state.json halted=true`). Hot path / env-pin /
    FX legacy untouched (git show --stat on all three commits confirms).
- **Equity-harvester-bot**: 22-story PRD; TUI wired to equity (commits 3a58c6c, eb04687). An
  independent code review flagged **4 CRITICAL + 7 HIGH execution defects** (C1 crash→double-submit,
  C2 fill-detection reads aggregate position as per-order fill, C3 books PENDING as FILLED, C4 can't
  detect a MISSING corp-action split). These are the real blocker to paper/shadow trading. `[unverified — from memory recent.md, re-grep before acting]`
- **FX direction is RETIRED — equity harvester is the live strategy (operator doctrine 2026-06-24, L-016).**
  FX/forex direction hit a hard ~52% ceiling (coin-flip, no shippable edge; confirmed 4+ ways:
  price-only, news, factor, carry, meta-labeling), so the transformers fail the 10% ship gate and stay
  PERMANENTLY quarantined. The product pivoted to the **equity harvester** (equity-beta risk-premium
  harvesting). Consequences for future sessions: the 6 stale FX champions (36d old) are
  **expected/abandoned-by-design, NOT a fixable gap**; do **NOT** propose FX retrains — and the earlier
  "refresh creds + retrain a gate-passing FX model so it trades" suggestion is a **DEAD END** (the
  ceiling is the market, not a bug). Route trading work to the equity harvester. See L-016 + verdict
  docs in `docs/`.
- **L7 (TUI live-run finding) RESOLVED as intentional:** the Trades tab rendering "HARVESTER REBALANCE
  PLAN" instead of FX trades is correct product behavior (FX retired → equity harvester). Not a bug;
  never "fix" it back to FX.

## Blockers

- None for *this* task (context-system build). For trading: the **operator** may unhalt at any time
  (operator-directed override is allowed and was exercised 2026-06-24T18:52Z → `halted=false`). What
  stays doctrine: the **AUTONOMOUS/loop path may NEVER unhalt on its own** (the `/evolve` loop can't
  relax the halt Hard NO; only the human who owns the halt can lift it). And unhalting ≠ trading:
  actual trades are gated on scanner-running + fresh (non-401) OANDA creds + a gate-passing fresh
  model — none of which hold right now, so the bot abstains. See the runtime-state blockers above.

## Judgment calls I made on this build (veto any of these and I'll revise)

1. **CLAUDE.md reconciled, not overwritten.** A rich 14 KB CLAUDE.md already existed. I *prepended*
   a lean "Self-Evolving Context System" block at the top (one-liner, Tier 6/7, Hard NOs w/
   citations, pointers, working rules) and **left all existing FX doctrine below it intact.** Net
   effect: leaner *entry point*, nothing lost. If you'd rather I trim the old body, say so.
2. **`oanda_environment` line is 738, not ~612.** Your brief said ~612; line 612 is the docstring.
   The actual field default is `src/scanner/config.py:738`. I cited 738 everywhere. Reality won.
3. **Session-spawns-agents instruction** ("add so session spawns agents to prompt main AI after
   engineering it") — interpreted as: a SessionStart hook that, on every session, injects a boot
   prompt telling the main AI to read INTENT/NOTES/LESSONS first, surfaces the halted+practice
   state, and recommends dispatching domain specialist sub-agents per the working rules. Implemented
   as `.claude/tools/session_context_boot.sh`, registered as a **second, additive** SessionStart
   hook (the existing tmux hook is untouched). Hooks emit context — they can't literally spawn
   Claude agents — so "spawn agents" is realized by *instructing the main AI to dispatch them*. If
   you meant something more literal (e.g. background `Agent`/cron jobs), tell me and I'll rebuild.
4. **`.claude/commands/evolve.md`** created (the `commands/` dir didn't exist). Invoke with `/evolve`.
5. **Self-improver loop added (2026-06-23 steering).** Per operator steering, folded in: `LOOP.md`
   (cycle + 5 objective stopping conditions incl. anti-stall STOP-CHURN), `verifier.md` + `/verify-task`
   (independent verifier = separate Code Reviewer agent/model, re-derives claims from disk),
   `tools/risk_monitor.sh` (parallel fail-closed safety tripwire — runs GREEN against live repo, exit 0),
   LESSONS recall-trigger index, NOTES schema+pruning rule, DoD item 6 (verify-task PASS + monitor GREEN).
   **Judgment call:** I authored these coherently myself (they cross-reference heavily), then dispatched
   a *separate* Code Reviewer agent as the independent verifier of the whole system — that build→verify
   loop IS the "let an agent create a loop... until proper and complete" instruction, instantiated. The
   loop is a META/dev loop (like Ralph), deliberately NOT runtime code — Claude stays out of the hot path.
   If you wanted the loop wired as a live background daemon instead, say so and I'll build that variant.
6. **Enforcement layer added (2026-06-23 mandate).** Converted the inert/advisory pieces into
   deterministic gates: a `Stop` hook (`stop_gate.sh`) enforcing the risk monitor every turn-end;
   `verify_gate.py` (deterministic half of the verifier — reads disk, immune to a lying agent);
   `loop_gate.py` (stopping conditions from disk); 32 no-mock tests. **Judgment calls for veto:**
   (a) the Stop hook runs at *every* turn-end and **blocks once** on ALARM — I chose blocking (your
   fail-closed ethos) with a loop-guard so it can't trap; if turn-end checks feel heavy, I can scope
   it to risky tool calls instead. (b) The whole layer is **untracked in git** — a `git clean` would
   wipe it and leave `settings.json` pointing at a missing hook. I did NOT commit (your call); the
   verifier flagged this as the #1 residual risk. Say "commit it" and I will. (c) `L-005` body is
   proposed via `/evolve` below and awaits your approval — the LESSONS trigger-row referencing it is
   a forward-ref until you approve.

## Active loop status (2026-06-23 — standing-roadmap cycle, converged)

- `.claude/loop/state.json` has 6 cycles; `loop_gate.py` reads disk → **STOP-DONE** (risk GREEN,
  last cycle no new info, 0 open questions). Took **3 independent verify rounds** to converge.
- **Roadmap hardening landed (all 4 fronts):** verifier ship-gate + halt-guard are now **AST-verified**
  (immune to `.15`/`1e-1`/`+=`/tuple/walrus/expr/non-literal evasion, and dead/commented/inverted
  guards); `lessons_have_triggers` integrity makes memory provably fire at planning time;
  `risk_monitor.sh` widened to the 4th Hard NO (no live env assignment). **47 no-mock tests, all green.**
- **Independent verifier earned its keep:** round 2 returned **GATE FAIL** and found a real gap I
  introduced (AST missed `HARD_MAX_GAP += 0.10` and tuple-unpack → >0.10 passed as PASS) + a latent
  inverted-else halt guard. Fixed both + preempted walrus; round 3 → **GATE PASS, HIGH confidence, no
  residual gap** after the agent invented its own evasions and all were caught. Self-grade would have
  shipped the gap.
- Real-repo `verify_gate.py` PASS (20 checks); hot path (`execution.py`/`scripts/`/`main.py`) untouched.
- Prior-cycle status (enforcement layer build) is folded into LESSONS L-005 + the cycle history.

## Roadmap progress (north star — `INTENT.md` standing roadmap)

- **Committed + pushed:** enforcement layer at `1062064` → `origin/ralph/equity-harvester-bot`
  (settings.json is gitignored/machine-local → `.claude/loop/INSTALL.md` documents the wiring).
- **Verifier front:** AST-verified, fail-closed, 3 independent rounds → PASS. L-006 captures the lesson.
- **Stopping-conditions front:** STOP-CHURN now un-gameable to the self-report limit — rolling-window
  (progress = close a question OR learn a lesson) + 6-cycle absolute backstop. A separate verifier
  brute-forced 200k sequences and found no forever-CONTINUE stall. **51 no-mock tests, all green.**
- **Objective stopping inputs — DONE (2026-06-23).** Stopping signals now derive from observable
  reality, not self-report: `open_questions` counted from tracked `.claude/loop/questions.json`;
  `new_verified_facts`/`new_lessons` are per-cycle deltas of real test count + verify verdict +
  LESSONS.md count, recorded by `.claude/loop/record_cycle.py` (runs the suite); loop_gate re-checks
  the latest cycle's verdict + open-count against LIVE reality → tamper = HALT. **59 no-mock tests.**
  Separate verifier acted as a malicious worker (faked verdict/open-count, wiped history, abused
  blocked flag, malformed JSON) — every attempt failed closed; **no false STOP-DONE possible.**
- **Remaining honest limit (documented in loop_gate.py + L-007):** a worker could still falsify the
  recorded `tests_passed`/`lessons_count` integers themselves — but the live verify_gate re-run is the
  true integrity floor, and human review backstops the counts. This is the irreducible self-report
  edge; closing it further would need signing/attestation of record_cycle output (low priority).
- Operating in delegated-authority mode (commit/push + /evolve auto-approved with proof; immutable
  escalations route to human). See the "Operating mode" section above.

## Irreducible floors — the 2026-06-23 red-team program's residual trust assumptions

A red-team (separate agent, lying + lazy) attacked all four fronts; 4 hardening increments closed
every CLOSABLE cheat with deterministic, fail-closed, tested gates (commits 88b8b83 / c0d8306 /
d5992a8 / + Increment 4). What remains is genuinely irreducible — documented here per the operator's
"don't fake closure" mandate. Each names the residual trust assumption and what closing it would cost.

1. **Gate-script integrity (L-008).** All enforcement scripts + the test suite are SHA-256 hash-pinned
   (`gate_manifest.json`), cross-checked by two independent checkers, coverage-checked vs entry-drop.
   FLOOR: editing a gate script AND regenerating the manifest in one change evades hashing — but the
   diff is git-visible, and a root checker can't hash itself. Closing cost: sign the manifest with a
   key the worker can't access (CI-side), checked out-of-band.
2. **Lesson meaning (L-009).** Lessons are structurally audited (5 fields, length, uniqueness);
   empty/shallow/dup fail closed. FLOOR: a structurally-valid but vacuous lesson passes — a static
   check can't judge meaning. Closing cost: an LLM-judge (itself subject to #5) or human review (current).
3. **Recorded metrics (L-010).** open_questions, verify verdict, lessons_count, and tests_passed are
   ALL re-derived live at decide time and tamper-checked → a hand-edited count HALTs. FLOOR: none for
   these — re-derived from source (tests_passed's source, the suite, is hash-pinned per #1).
4. **Observable work (L-010).** STOP-DONE requires an observable artifact delta OR an explicit
   no_work_needed_attested. FLOOR: that attestation is a human-reviewed self-claim (can't prove intent
   from inside). Closing cost: same as #5.
5. **Separate-agent verification (L-011).** STOP-DONE requires a fresh PASS agent-verdict BOUND to the
   current state — skipping the verifier blocks done; any post-verify change forces re-verify. FLOOR:
   a worker can write `record_verdict.py --gate PASS` WITHOUT dispatching a real agent (the lie). No
   static gate inside the worker-controlled environment can prove an LLM was dispatched or judged
   honestly. The DETERMINISTIC half (verify_gate: live + hash-pinned + cross-protected) is the enforced
   verification floor. Closing cost: an out-of-band trust anchor the worker can't forge — signed
   attestation from a trusted execution environment that actually dispatched the verifier, or
   append-only out-of-band dispatch logging the gate checks instead of a self-written JSON.

Bottom line: every machine-checkable claim and every Hard NO is enforced + fail-closed. What's left
reduces to "a worker with full repo access can lie in a git-visible way (backstopped by human review)
or fabricate an agent dispatch." Closing those requires moving the root of trust OUTSIDE the
worker-controlled environment (CI signing / TEE / out-of-band logging) — operator's call on cost.

### Managed-settings trust anchor (2026-06-23, docs-verified) — moves the root of trust outside the repo
Prepared (operator runs one privileged install step — `.claude/loop/managed/INSTALL.md`): enterprise
managed-settings.json pins the Stop hook to a ROOT-OWNED self-contained wrapper at the system managed
dir (macOS `/Library/Application Support/ClaudeCode/`, Linux `/etc/claude-code/`, Windows
`C:\Program Files\ClaudeCode\` — NOT the deprecated ProgramData). Managed settings are highest
precedence ("can't be overridden"); Claude Code write access is confined to the project tree. The
wrapper re-derives the gate hashes ITSELF (catches a neutered in-repo checker). 90 no-mock tests;
separate verifier red-teamed it (PASS, no new gap; fetched the docs live). Effect on the floors:
- **Floor #1 (gate-tamper) — DISABLE + NEUTER dimensions: CLOSED to "requires admin."** A worker can't
  remove/re-point the Stop hook (managed precedence) nor neuter a checker (wrapper self-hashes).
- **Floor #5 (hook-neuter): CLOSED to "requires admin."** The tripwire can't be disabled by the worker.
- STILL OPEN (documented): edit-script-AND-manifest-together (git-visible; close by baking the manifest
  hash into the root-owned wrapper, re-install per gate change); the verification LIE dimension (L-011,
  advisory agent half); and the cwd/harness-integrity assumption (same as the in-repo stop_gate).
The honest win: this turns "worker can disable the gate" into "worker can't without admin, which the
operator controls" — until the one privileged install step runs, the anchor is advisory and the
in-repo gates + code-level guards remain primary.

**INSTALLED 2026-06-24, then ADVERSARIALLY RE-CHECKED — found 2 bypasses, hardened (RE-INSTALL needed).**
Operator ran the install. Verified from disk: the two files are `root:wheel`, NOT writable by the
worker — a worker CANNOT directly edit/remove/re-point the managed hook (proven: every write returned
Permission denied). BUT the adversarial re-check (operator-requested) found two ways a worker could
still neutralize it via LOCAL settings (which the worker CAN write):
  (1) `disableAllHooks: true` in local settings disables ALL hooks incl. managed — unless managed pins
      `disableAllHooks: false` (precedence). NOW fixed in the in-repo managed-settings.json.
  (2) interpreter PATH-shadow of bare `python3`/`bash`. NOW fixed: absolute `/usr/bin/python3` + `/bin/bash`.
`verify_managed_anchor.py` now REQUIRES `disableAllHooks:false` and currently reports **NOT-ACTIVE,
exit 1** against the installed OLD file — honest: **the operator must RE-INSTALL** the hardened file
for the closure to take effect. **I correct my prior "CLOSED in effect" claim:** until re-install, the
direct-edit vector is closed but the disableAllHooks/PATH vectors are open. Separate verifier red-teamed
the hardened anchor → PASS, no new bypass. Runtime "managed false wins" is docs-precedence-confirmed
(verbatim "can't be overridden"), not empirically reproduced (needs a live session). LOW residual noted:
risk_monitor.sh's own unqualified git/grep (not a current bypass; documented in INSTALL.md).

- 3 cycles recorded in `.claude/loop/state.json`. `loop_gate.py` computes **STOP-DONE** from disk
  (risk GREEN, last cycle no new info, 0 open questions). Build→verify→fix ran twice.
- **What is now ENFORCED (deterministic, not compliance):** Stop hook `stop_gate.sh` runs
  `risk_monitor.sh` at every turn-end, fail-closed, blocks on ALARM, loop-guarded (proven: real-repo
  exit 0; synthetic live-env → exit 2 HALT-SAFETY; `stop_hook_active` → exit 0 no-trap).
  `verify_gate.py` re-derives the Hard NOs from disk (19 checks, immune to narrative), now also asserts
  the Stop hook is REGISTERED and catches non-default live assignments. `loop_gate.py` computes the
  stopping decision. All covered by `.claude/loop/tests/test_loop_enforcement.py` — **32/32 no-mock
  tests pass** (`python3 .claude/loop/tests/test_loop_enforcement.py`).
- **Independent verifier (separate Code Reviewer agent): GATE PASS, 8/8 claims, zero Hard-NO.** It
  found 3 real hardening gaps; I fixed the 2 high-value ones (Stop-hook-registration check + stronger
  live-flip detection) and documented the 3rd (regex evasion surface). Hot path untouched
  (`execution.py`/`scripts/`/`main.py` unmodified — verified via `git status`).
- Earlier context-system cycle verdict (PASS, risk_monitor scope fix) is folded into L-001..L-004.
- Watch-item: equity-harvester C1–C4 defects below are memory-sourced `[unverified]` — re-grep first.

## Assumptions resolved on my own this turn

- Tier 6 = meta-learning ensemble (MetaLearner + Bayesian adapter + ensemble weighter, shadow);
  Tier 7 = autonomous control loop (incident→propose→gate→soak→promote→close), Claude never in the
  hot path. Source: CLAUDE.md + `docs/tier7-architecture.md`.
