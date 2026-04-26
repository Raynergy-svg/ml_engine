# Trade Homework System — Design Spec

**Status:** APPROVED — pending operator final review before implementation plan
**Date:** 2026-04-25
**Author:** Claude (in collaboration with operator)
**Phase:** Phase 96 candidate (rename suggested: "Apprenticeship — Buddy Studies His Trades")
**Predecessor work:** Phase 91/92/93/94/95 (hard gates, supervisor console, MR veto, futures, persistence)

---

## 1. Why This Exists

### 1.1 The two-year insight

Buddy is a *student*, not a worker. The operator (you) is the master. The activity is **studying past trades to internalize patterns** — not executing live trades autonomously. Closed trades are homework material; the operator grades each homework; corrections become training signal. The bot earns the right to "fish" by first proving he can analyze the catch correctly.

This vision was articulated 2026-04-25 after a 2-year detour through "autonomous quant fund" framing that did not match the codebase the operator actually built. Architectural evidence supporting the apprenticeship framing:

- 8-screen TUI workbench (observation surfaces, not headless execution)
- Phase 92 Supervisor Console with operator hotkeys, typed-confirmation modals
- `.claude/brain/` — externalized reasoning files for the master
- 15-agent verdict system designed to be inspectable, not just aggregated
- Pre-trade veto window (Phase 92 US-511) — explicit human checkpoint
- Adjustment Inbox (Phase 92 US-509) — human approval gate for config changes

### 1.2 The current gap

Buddy in dry-run mode evaluates setups but never sees outcomes, so no learning signal flows. The 17 trade journal entries from the 2026-04-15 catastrophic streak sit unanalyzed. Reflection cycles run periodically but produce no homework — they produce telemetry.

**No outcome feedback = no learning.** A student doing practice problems with no answer key.

### 1.3 What we're building

A **Trade Homework System** that:

1. Picks up every closed trade (real or backtest-replay)
2. Pulls outcome data from OANDA via the existing US-605 backfill pipeline
3. Generates a structured analysis using a deterministic heuristic engine (no LLM in hot path)
4. Lands the analysis in the existing F2 Inbox alongside config adjustments
5. Lets the operator approve / reject / edit / snooze each homework item
6. Translates the operator's grade into RL training signal (agent weight updates, regime priors)
7. Allows on-demand batch generation (CLI) for catching up on unstudied trades
8. Adds a two-pane (queue + live preview) inbox layout so review is fast and pleasant

---

## 2. Architecture

### 2.1 Module layout

```
src/scanner/automation/homework/
  ├── __init__.py
  ├── generator.py     HomeworkGenerator   — closed trade → markdown analysis
  ├── store.py         HomeworkStore       — .jsonl pending + history (atomic writes)
  ├── reviewer.py      HomeworkReviewer    — A/R/E/S transitions + RL signal emit
  └── heuristics.py    HEURISTIC_CATALOG   — ~25 hand-coded patterns across 6 categories, growable

src/tui/screens/
  └── inbox_screen.py  ← EXTENDED          — two-pane layout, type-aware rendering

buddy_scanner.py CLI:
  └── homework subcommand                  — on-demand generation: --generate-batch [...]

State files (new, gitignored):
  .claude/homework_pending.jsonl
  .claude/homework_history.jsonl
```

### 2.2 Component responsibilities

| Component | Responsibility | Mirrors |
|---|---|---|
| `HomeworkGenerator` | Pure function: `(trade_dict, outcome_dict) → HomeworkEntry`. Runs heuristic catalog, picks primary lesson, renders markdown. **No LLM call.** | New pattern |
| `HomeworkStore` | Atomic file IO for `homework_pending.jsonl` + `homework_history.jsonl`. Read/append/move-to-history operations. | `AdjustmentApprover._load_pending` / `_save_approved` |
| `HomeworkReviewer` | Apply operator decision (approve / reject / edit / snooze). Move entry pending → history with grade. Emit training signal. | Merged from `HomeworkApprover` + `TrainingSignalEmitter` per operator's call (sequential anyway) |
| `HEURISTIC_CATALOG` | List of `Heuristic(name, predicate, lesson_template, confidence)`. Hand-coded initially; grows organically as operator rejects analyses with corrections. | New pattern |
| `inbox_screen.py` (extended) | Renders unified queue (homework + adjustments) with type icons, filter pills, two-pane (queue + detail) layout, A/R/E/S hotkeys acting on focused row. | `InboxScreen` from Phase 92 US-509 |
| `homework` CLI subcommand | `python buddy_scanner.py homework --generate-batch [--last N] [--since DATE]`. Bootstrap mode for catching up on unstudied trades. | `learn` subcommand pattern |

### 2.3 Three trigger paths

| Trigger | When it fires | How |
|---|---|---|
| **Real-time** | Trade closes on OANDA | `OutcomeBackfill` (US-605) populates outcome → fires `homework.generate` event → `HomeworkGenerator` subscribes |
| **Batch overnight** | 03:00 UTC daily | `cycle_autonomy.py` adds homework sweep — generates analysis for any closed trades not yet in inbox |
| **On-demand CLI** | Operator runs `buddy_scanner.py homework --generate-batch` | Bootstrap. Studies the 17 unstudied trades from the 04-15 streak (or any historical window). |

The on-demand CLI is the immediate-value mode: operator can populate the inbox tonight without waiting for live trades.

---

## 3. Data Model

### 3.1 HomeworkEntry dataclass

Lives in `src/scanner/automation/homework/__init__.py` (re-exported for downstream).

**Fields fall into 4 groups:**

1. **Identity** — `homework_id` (uuid4), `trade_id`, `schema_version=1`, `generated_at` (iso8601 utc)
2. **Trade snapshot (denormalized)** — `pair`, `direction`, `entry_price`, `sl_price`, `tp_price`, `rr_ratio`, `confidence`, `weighted_vote_score`, `regime`, `agent_verdicts` (15 entries: name, score, passed, weight, reason)
3. **Outcome (from OANDA backfill)** — `close_time`, `close_price`, `realized_pl`, `close_reason` (TP|SL|MANUAL), `duration_minutes`, `mfe_pips`, `mae_pips`
4. **Analysis (Buddy's homework)** — `analysis_markdown`, `proposed_lesson`, `confidence_in_analysis`, `agents_to_reinforce` (list of agent names), `agents_to_penalize`
5. **Review state (mutable via reviewer.transition)** — `status` (pending|approved|rejected|edited|snoozed_until_X), `operator_grade`, `operator_note`, `operator_edits`, `reviewed_at`

The dataclass is `frozen=True` for the immutable core; review state is updated by writing a *new* entry to history, not by mutating the original.

### 3.2 Storage layout

```
.claude/homework_pending.jsonl
  ├─ One JSON object per line
  ├─ Append-only via tmp + rename
  ├─ Read on demand by inbox_screen and reviewer
  └─ Schema-validated on read

.claude/homework_history.jsonl
  ├─ Same schema + populated review fields
  ├─ Append-only forever (audit trail)
  └─ Source of truth for "what has the operator graded"
```

Storage decision: keep separate from `.claude/config_adjustments.json`. Phase 92 adjustment code stays untouched. The inbox screen merges both streams at render time.

---

## 4. The Heuristic Engine (the novel piece)

### 4.1 Why no LLM

Per CLAUDE.md and operator instruction, no Claude/LLM call in the runtime hot path. Buddy's analysis is generated by deterministic heuristics that pattern-match over the trade record. The operator's review (and its corrections) IS the intelligence layer — Buddy surfaces the structured facts and proposes a candidate lesson from a known catalog.

### 4.2 Heuristic shape

A `Heuristic` is a 4-tuple:
- `name`: string identifier
- `predicate`: `Callable[[Trade, Outcome], bool]` — does this pattern apply?
- `lesson_template`: string with `{}` placeholders for trade-specific values
- `confidence`: float 0-1, used to rank when multiple heuristics fire

Multiple heuristics may fire on the same trade. The highest-confidence match becomes `proposed_lesson`. All matches appear in the markdown analysis under "Detected patterns."

### 4.3 Initial catalog (~25 heuristics across 6 categories)

Living in `src/scanner/automation/homework/heuristics.py`. Each heuristic is independently testable. The catalog is organized around **5 categories used by professional discretionary trade review** (Steenbarger, Bellafiore, Seykota, Raschke) plus **1 category specific to algorithmic systems** (López de Prado meta-labeling). Sources cited in §4.6.

Every closed trade is evaluated against every heuristic. Multiple matches are ranked by confidence; the highest becomes the primary lesson. All matches appear in the markdown analysis.

#### A — Setup Validity (was the entry signal genuine?)
*Bellafiore: "Was I right and lucky, or right and skilled?"*

| ID | Heuristic | Confidence |
|---|---|---|
| A1 | `setup_adx_trend_mismatch` — directional trade with ADX<10 (no trend present, wrong setup type for a momentum/trend strategy) | 0.85 |
| A2 | `setup_volatility_regime_mismatch` — strategy doesn't fit current `regime` (e.g., mean-reversion taken in trending HIGH regime, or directional in ranging LOW regime) | 0.80 |
| A3 | `setup_session_mismatch` — pair traded in wrong session (e.g., AUD/NZD trade taken during illiquid Tokyo→London handoff) | 0.65 |
| A4 | `setup_rsi_neutral` — directional entry while RSI in 45–55 zone (no momentum bias either direction; setup quality low) | 0.70 |

#### B — Risk Calibration
*Kelly Criterion / Ed Seykota / Phase 91 lessons*

| ID | Heuristic | Confidence |
|---|---|---|
| B1 | `risk_rr_below_breakeven` — R:R ratio combined with implied win-rate produces negative expectancy (Kelly violation) | 0.90 |
| B2 | `risk_sl_too_tight_for_atr` — `sl_pips / atr_pips < 1.0` (stop inside normal noise; whipsaw nearly guaranteed) | 0.85 |
| B3 | `risk_low_regime_sl_violation` — regime=LOW + `sl_mult<1.2` (the Phase 91 promoted rule — ranging markets need wider stops, not tighter) | 0.95 |
| B4 | `risk_correlated_double_exposure` — another correlated pair already open at trade entry (effective leverage doubled vs intended) | 0.75 |

#### C — Agent Consensus Quality
*Bayesian voting integrity / Phase 91+93 promoted patterns*

| ID | Heuristic | Confidence |
|---|---|---|
| C1 | `consensus_trend_veto_unhonored` — SL outcome + trend agent `passed=False` on directional trade (Phase 91 pattern: WVS arithmetic should not override individual hard-veto signals) | 0.90 |
| C2 | `consensus_mr_composite_match` — SL + MR `passed=False` + `model_disagreement>0.25` (Phase 93 composite veto fingerprint — would have caught this trade) | 0.90 |
| C3 | `consensus_disagreement_at_floor` — SL + `model_disagreement` within 0.03 of `disagreement_hard_floor` (boundary case — gate may need tightening) | 0.70 |
| C4 | `consensus_high_winner` — TP + WVS>0.75 + ≥10 agents passed (high-confluence pattern; reinforce) | 0.80 |
| C5 | `consensus_single_agent_dragged` — outcome opposite to majority vote, dominated by one high-weight agent's contrarian read (audit that agent's weight) | 0.65 |

#### D — Execution Quality
*Bellafiore "tape reading" / Steenbarger deliberate-practice categories*

| ID | Heuristic | Confidence |
|---|---|---|
| D1 | `exec_mfe_zero_directional_loss` — SL + `mfe_pips/atr_pips < 0.2` (price never moved in our favor — the 04-15 catastrophic-streak fingerprint; entry was directionally wrong from tick 1) | 0.85 |
| D2 | `exec_whipsaw_reversal` — SL hit, then price reversed past entry within 2× ATR over the next 60 minutes (stop was too tight; trade thesis was correct, exit was premature) | 0.80 |
| D3 | `exec_slow_tp_widening_candidate` — TP + `duration_minutes > 4 × expected_hold_minutes` (TP hit slowly; consider wider `tp_mult` to capture more) | 0.60 |
| D4 | `exec_fast_sl_bad_timing` — SL + `duration_minutes < 5` (stopped almost immediately — likely news event, bad fill, or pre-trade signal stale) | 0.70 |
| D5 | `exec_slippage_cost_winner` — winner where realized fill price differed from expected entry by > 0.3 × `sl_pips` (slippage ate edge; review broker latency) | 0.55 |

#### E — Regime / Context Drift
*Raschke regime-fingerprint analysis / Phase 91 staleness-block lesson*

| ID | Heuristic | Confidence |
|---|---|---|
| E1 | `context_stale_models` — SL + `oldest_age_days > 7` (Phase 91 staleness threshold; models predicting old regime into new market) | 0.85 |
| E2 | `context_regime_transition` — entered in regime X, closed in regime Y (regime shift mid-trade — was thesis still valid after shift?) | 0.65 |
| E3 | `context_news_window` — known high-impact news (NFP, FOMC, CPI, ECB) during trade duration; outcome may reflect news shock not setup quality | 0.70 |
| E4 | `context_correlated_co_loss` — ≥2 correlated pairs closed at SL within 30 min of each other (regime issue not single-trade issue; consider regime-pause heuristic) | 0.75 |

#### F — Meta-Patterns (algorithmic system specific)
*López de Prado "Advances in Financial ML" Ch. 3 (meta-labeling) and Ch. 4 (sample weights)*

| ID | Heuristic | Confidence |
|---|---|---|
| F1 | `meta_repeat_fingerprint` — last N≥3 trades sharing this gate-combination all lost (suggests structural hole in the gate logic itself, not bad luck) | 0.85 |
| F2 | `meta_lucky_winner` — TP, but `mae_pips > 0.7 × sl_pips` (trade nearly stopped before reversing; profit depended on luck not skill — don't reinforce confidently) | 0.70 |
| F3 | `meta_underrepresented_setup` — setup-type cluster has <10 examples in training data (model was extrapolating; outcome carries low evidential weight either way) | 0.60 |

The catalog is **growable**: when the operator rejects an analysis with a correction note that doesn't match an existing pattern, that note becomes a candidate for a new heuristic. A future phase will surface high-frequency rejection-notes for review and promote them into formal `Heuristic` entries.

### 4.6 Sources

The catalog is grounded in established trade-review literature, not invented:

- **Brett Steenbarger** — *Trading Psychology 2.0* (2015): deliberate-practice categories (setup quality, execution quality, risk management, emotional state). Maps to categories A, D.
- **Mike Bellafiore** — *One Good Trade* (2010): prop-trader review framework — reading the tape, market context, setup, R:R calibration. Maps to A, B, D.
- **Linda Raschke** — *Street Smarts* (1996): regime-fingerprint analysis, time-of-day patterns, correlation drag. Maps to E.
- **Marcos López de Prado** — *Advances in Financial Machine Learning* (2018): meta-labeling (Ch. 3), sample weights (Ch. 4), triple-barrier method. Maps to F1, F3.
- **Ed Seykota** — *Market Wizards* interviews: position sizing under Kelly, slippage analysis, whipsaw detection. Maps to B, D2, D5.
- **Ernest Chan** — *Algorithmic Trading: Winning Strategies and Their Rationale* (2013): backtest review — performance by regime, drawdown clustering. Maps to E1, E2, F1.
- **Phase 91 / 93 promoted rules** — `.claude/rules/trading.md`: in-house lessons from the 04-15 catastrophic streak (trend hard-veto, MR composite veto, LOW regime SL floor, staleness block). Maps to B3, C1, C2, E1.

Each heuristic in `heuristics.py` carries a `source` field referencing the originating literature or rule, so the catalog stays auditable.

### 4.4 Agent reinforcement scoring

`agents_to_reinforce` and `agents_to_penalize` are computed per-trade by a separate function (not part of heuristic catalog):

- **Reinforce**: any agent whose `passed` aligned with the outcome (passed=True on TP, passed=False on SL)
- **Penalize**: any agent whose `passed` was opposite to the outcome
- Each suggestion includes a delta (e.g. "+0.02") that the reviewer applies if approved

This is a deterministic mapping. The operator can override in the edit modal.

### 4.5 Training signal payload (concrete shape)

When `HomeworkReviewer.transition()` processes an approved homework entry, it emits a `TrainingSignal` to the existing RL update queue. Payload fields:

- `homework_id`, `trade_id`, `outcome` (TP|SL|MANUAL)
- `agent_weight_deltas`: `dict[agent_name, float]` — applied to `trained_data/models/agent_weights.json` on next persistence cycle
- `regime_prior_deltas`: `dict[regime_name, dict[gate_name, float]]` — applied to regime-specific gate calibration
- `heuristic_fired`: name of primary heuristic (for accumulating evidence on which heuristics are useful)
- `operator_action`: approved | edited | rejected
- `operator_note`: verbatim text from operator (rejected/edited only)

**Approved** payloads apply deltas as Buddy proposed. **Edited** payloads apply the operator's edited deltas instead. **Rejected** payloads emit a *negative* signal: deltas are discarded, but the heuristic that fired is recorded in a `rejected_heuristics_log.jsonl` for future tuning. The operator's note is stored as a candidate seed for new heuristics (a future phase will surface these for review).

The signal queue is the existing RL update path used by `agent_weights.json` save cycles. No new infrastructure is introduced — homework just becomes another producer alongside the per-trade RL learning that's already wired.

### 4.5 Markdown rendering template

The output markdown is rendered via Jinja2-style template embedded in `generator.py`. Sections:

```
# Trade #{trade_id} {pair} {direction} — {close_reason_label}

## Outcome
- {realized_pl} | held {duration} | MFE {mfe_pips} | MAE {mae_pips}

## Setup at entry
[table: confidence, wvs, regime, ADX, RSI, ATR]

## Agent verdicts
[table: 15 rows, name | score | passed | weight | reason]

## Detected patterns
{matches from heuristic catalog, sorted by confidence}

## Buddy's analysis
{primary lesson rendered with values from trade}

## Proposed adjustments
- Reinforce: {agents_to_reinforce}
- Penalize: {agents_to_penalize}
- Confidence in analysis: {confidence_in_analysis}
```

The exact template string lives in `homework/generator.py` — not in this spec, per "code lives in files."

---

## 5. UI — Two-Pane Inbox (Option C from operator selection)

### 5.1 Layout

| Region | Content |
|---|---|
| Top filter pills | `[ All N ]` `[ 📚 Homework N ]` `[ 🔧 Adjustments N ]` — Tab cycles |
| Left column (queue) | Type icon, timestamp, subject, outcome P/L (homework only). Scrollable with arrow keys. |
| Right column (detail) | Auto-renders detail of focused row. Markdown for homework, value-comparison for adjustments. |
| Bottom hint bar | `↑↓ navigate · A approve · R reject · E edit · S snooze · Tab filter · / search` |

Visual differentiation:
- 📚 prefix + cyan accent for homework
- 🔧 prefix + magenta accent for adjustments
- Green P/L for wins, red for losses (homework only)

### 5.2 Interaction model

- Cursor in queue moves with ↑↓ — detail pane re-renders on every move
- A/R/E/S act on focused row
- After action, focused row is removed; cursor *stays at same position* so next entry slides into focus
- This enables A-A-A-A-style fast triage

### 5.3 Edge cases

| Situation | Behavior |
|---|---|
| Queue empty | Both panes show "No pending items" placeholder |
| 200+ entries | Auto-paginate; PgUp/Dn jumps 10 rows; scrollbar shows position |
| Reject modal cancel | Esc → no action, focus returns to queue |
| Edit modal long text | Multiline text area, max 2000 chars |
| Filter change | Cursor resets to most recent in new filter; previous focus saved per filter |
| Double-tap A in 100ms | Second press ignored if first hasn't processed |
| New entry arrives during review | Appears at top of queue with subtle highlight; cursor stays put |

### 5.4 Implementation surface

Textual framework already supports `Horizontal` container with two children, `DataTable` with row-focus events, and `Markdown` widget. Expected line count for the screen change: ~200 LOC net add.

The wiring path that matters (US-604 lesson): `inbox_screen.py` is registered in `app.py` BINDINGS. Both the existing inbox path AND the new homework path must hit the same screen. Static + behavioral wiring tests are mandatory (see §7).

---

## 6. Data Flow End-to-End

### 6.1 Real-time path

```
1. Trade closes on OANDA → ORDER_FILL transaction
2. OutcomeBackfill (US-605) polls /transactions/sinceid every cycle
3. Match by trade_id; populate journal entry: realized_pl, close_time, close_reason, mfe, mae
4. Backfill emits homework.outcome_ready event
5. HomeworkGenerator subscribes; constructs HomeworkEntry
6. HomeworkStore.add() → atomic append to homework_pending.jsonl
7. Inbox screen (if open) refreshes — new entry appears at top
8. Operator presses V/A/R/E/S
9. HomeworkReviewer.transition(homework_id, action, note?)
   - Read pending.jsonl, find entry, mark with grade
   - Atomic move: pending.jsonl → history.jsonl
   - Emit training signal payload to RL queue
10. RL update applies on next agent_weights save cycle (existing infrastructure)
```

### 6.2 Batch overnight path

`cycle_autonomy.py` adds a 03:00 UTC sweep. Pseudocode in `homework/__init__.py`:

```
def overnight_sweep():
    for trade in journal_since_last_sweep():
        if trade.has_outcome and not trade.has_homework:
            entry = generator.generate(trade, trade.outcome)
            store.add(entry)
```

### 6.3 On-demand CLI path

`python buddy_scanner.py homework --generate-batch [--last N] [--since DATE] [--include-graded]`

Modes:
- `--last 17` → study last N closed trades
- `--since 2026-04-15` → study trades from a specific date
- `--include-graded` → re-study trades that already have homework history (for evolving heuristics)

This is the **bootstrap mode** for operator's first session: studies the 17 unstudied 04-15 streak trades, populates the inbox, ready for review.

---

## 7. Error Handling

### 7.1 Principles

- **Homework subsystem is best-effort.** Generator failures must never crash the scanner.
- **Atomic writes everywhere.** Both jsonl files use `tmp + rename` pattern for crash safety.
- **Schema validation on every read.** Corrupt jsonl entry → quarantine to `.claude/homework_quarantine.jsonl`, log warning, continue.
- **Operator notes required for reject/edit.** TUI input validation enforces.
- **Concurrent operator action protection.** `fcntl` lock on history.jsonl during reviewer.transition.

### 7.2 Specific failure modes + handling

| Failure | Handler |
|---|---|
| Trade has no outcome yet | Skip; will retry on next cycle |
| OANDA API timeout during backfill | Existing US-605 retry logic; homework generation deferred |
| Heuristic predicate raises | Try/except per heuristic; log and skip individual heuristic, continue with others |
| Markdown render template error | Fallback to minimal template (raw fields); log error |
| pending.jsonl corruption | Quarantine corrupt lines; rebuild file from valid lines |
| Inbox full (>500 pending) | Hard cap; FIFO eviction with warning to brain log |
| Reviewer called on non-existent homework_id | Log + return False; UI shows "entry no longer exists" |
| RL training signal queue full | Backpressure; reviewer succeeds but signal is lost (logged) — operator can re-grade if needed |

---

## 8. Testing Strategy

### 8.1 Test files (~30 tests across 6 files)

| File | Coverage |
|---|---|
| `tests/test_homework_generator.py` | Each ~20 heuristics fires on its expected pattern (parametrized); primary lesson selection logic; markdown rendering against snapshot |
| `tests/test_homework_store.py` | Add / list / move-to-history / atomic write under interrupt / schema validation / quarantine flow |
| `tests/test_homework_reviewer.py` | A/R/E/S transitions; training signal payload shape; rate-limit prevents double-grade; concurrent operator action |
| `tests/test_homework_integration.py` | E2E: journal entry + outcome → generator → store → reviewer → agent_weights.json updated. Real journal fixture. |
| `tests/test_homework_inbox_wiring.py` | Static (grep + class introspection) + behavioral (TUI screen invocation) regression. Mirrors US-604 wiring pattern. |
| `tests/test_inbox_screen_two_pane.py` | TUI snapshot tests: queue scrolls, detail updates on row focus change, A/R/E/S act on focused row, filter pills work |

### 8.2 Test data

- Fixture: `tests/fixtures/homework_trades.json` — anonymized synthetic trades with diverse outcomes (TP, SL, BE, manual close, all 4 regimes, varied agent vote distributions)
- Fixture: `tests/fixtures/homework_outcomes.json` — corresponding outcome data
- Fixture: real 04-15 streak trades (17 entries from `trained_data/trade_journal_rl.json`) — used in integration test as the canonical "first-day curriculum"

### 8.3 What gets locked by the tests

- **Heuristic predicate coverage**: each heuristic's truth table is asserted (positive + 1+ negative case)
- **Markdown stability**: render snapshot tests catch unintended template drift
- **Wiring symmetry**: same defense-in-depth pattern as US-604 — `validation_stats` taught us this once
- **Training signal integrity**: approving a reinforce-X recommendation MUST result in agent X's weight increasing in the next save

---

## 9. Out of Scope (deferred to future phases)

- **LLM-augmented analysis.** No Claude/Anthropic API calls in this system. Heuristics only.
- **Backtest replay generator.** Wiring real OANDA historical bars through the scanner to produce thousands of practice trades — interesting but separate work.
- **Operator UI polish beyond C.** Animations, transitions, keyboard shortcuts beyond the spec, multi-row select, bulk actions — all post-MVP.
- **Cross-trade pattern aggregation.** "These 5 losses share fingerprint X" — that's a meta-analysis layer we add after the per-trade homework loop is proven.
- **Automated heuristic discovery from rejection notes.** Operator's correction notes are stored verbatim for now; auto-extracting them into new heuristics is future work.

---

## 10. Acceptance Criteria

The system is considered complete when:

1. The 17 04-15 streak trades each have a HomeworkEntry in pending.jsonl after `python buddy_scanner.py homework --generate-batch --last 17` runs without error.
2. The F2 Inbox shows all 17 entries in the queue column (with adjustments mixed in correctly per filter), and the detail pane renders the analysis markdown for the focused row.
3. Operator can A/R/E/S any entry and observe:
   - Entry disappears from pending.jsonl
   - Entry appears in history.jsonl with grade populated
   - Cursor stays at same position
4. Approving an entry that recommends "reinforce trend agent" results in `trained_data/models/agent_weights.json` showing trend's weight increased on the next persistence cycle.
5. Rejecting with operator note "you missed X" results in the note appearing verbatim in history.jsonl.
6. All 30 tests pass. Wiring tests pass. Integration test passes.
7. CLAUDE.md and `.claude/brain/strategic_log.md` updated to reflect the new subsystem and how to use it.

---

## 11. Build Plan Preview (handed to writing-plans skill)

The implementation will be parallelized across specialist subagents. Story-level breakdown will be produced by the `writing-plans` skill. Anticipated PR-sized stories:

1. `HomeworkEntry` dataclass + `HomeworkStore` (foundation; no UI dependencies)
2. `HEURISTIC_CATALOG` + `HomeworkGenerator` (pure logic; testable in isolation)
3. `HomeworkReviewer` + training signal emission (depends on store)
4. CLI `homework` subcommand (depends on generator + store)
5. Inbox two-pane refactor (depends on store + reviewer for read paths)
6. End-to-end integration test (depends on all of the above)
7. Wiring regression suite (mirrors US-604 pattern)
8. Documentation: CLAUDE.md update, runbook section, strategic_log entry

Stories 1, 2, 3, 4 can run in parallel by separate specialist subagents. 5 needs 3 done first. 6, 7, 8 are sequential close-out.

Specialist allocation:
- Software Architect — overall design supervision + story 2 architecture
- Senior Developer — stories 1, 3, 4
- Frontend Developer + UI Designer — story 5
- AI Engineer — training signal emission portion of story 3
- API Tester + Test Results Analyzer — stories 6, 7
- Technical Writer — story 8
- Code Reviewer — final gate

Estimated wall-clock: hours, not days, with parallel subagents. No two-week bullshit.

---

## 12. Open Questions

None blocking implementation. All major design choices are locked.

Soft questions for future iteration (not implementation blockers):

1. Should homework history support tag-based search? (e.g., "show me all rejected analyses from EUR_USD in HIGH regime") — defer to v2.
2. Should we support operator drafting their own homework (master writes, then later compares to buddy's)? — defer to v2.
3. Should rejected analyses contribute negative training signal (penalize the heuristic that fired), or only the operator's correction become the signal? — defaulting to "operator note becomes signal," can revisit.

---

## 13. References

- Phase 91 PRD (hard gates): `.claude/ralph/archive/2026-04-16-phase-91-hard-gates/prd.json`
- Phase 92 PRD (Supervisor Console): `.claude/ralph/archive/2026-04-17-phase-92-supervisor-console/prd.json`
- Phase 95 US-602 (state reconciler): `src/scanner/automation/state_reconciler.py`
- Phase 95 US-605 (outcome backfill): `src/scanner/automation/outcome_backfill.py`
- US-604 wiring regression pattern: `tests/test_embedded_scanner_validation_wiring.py`
- AdjustmentApprover (template for HomeworkReviewer): `src/scanner/automation/adjustment_approver.py`
- InboxScreen (template for two-pane refactor): `src/tui/screens/inbox_screen.py`
- 04-15 catastrophic streak journal entries: `trained_data/trade_journal_rl.json`
- CLAUDE.md token discipline rule (prompted this spec format): `CLAUDE.md` § "Token Discipline — Code Lives in Files, Not Chat"
