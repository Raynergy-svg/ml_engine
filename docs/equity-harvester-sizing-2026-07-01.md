# Equity Harvester ("Track A") — Sizing & Expectation-Setting Note (2026-07-01)

Follow-up action item from `docs/equity-harvester-verdict-2026-07-01-independent-audit.md`
finding (b). This note exists to make sure nobody sizes positions, sets risk budget, or sets
return expectations off the curated-pool 0.908 Sharpe. **Documentation only — no code, no
threshold, no `SHIP_GATE.json` change made as part of this note.**

## 1. The number to use for sizing/expectations

For position-sizing and expectation-setting purposes, treat the equity harvester as a
**beta sleeve with honest net Sharpe ≈0.74 full-sample / ≈0.36 out-of-sample (OOS)** —
not the 0.908 figure in the currently-shipped `SHIP_GATE.json`.

| Metric | Curated 20-name pool (shipped) | Broad PIT S&P universe (honest) |
|---|---|---|
| Net Sharpe, full sample | 0.908 | **0.739** (reproduced) / 0.74 (artifact) |
| Net Sharpe, OOS (2019+ / 2021-07+) | n/a (SHIP_GATE reports full-sample only) | **0.354** (reproduced) / 0.355 (artifact) |
| Max drawdown | 0.229 | 0.195 (both full + OOS) |
| Gate pass | PASS | Full: PASS. OOS: **FAIL** |
| Universe size | ~20 tickers, top-N-by-ADV | 624 evaluated (873 ever-members, PIT reconstruction) |

## 2. What this sleeve is — and is not

This is a **risk-premium / beta-harvesting sleeve**, not an alpha strategy. It holds a broad,
equal-weighted basket of liquid US equities and mechanically resizes total exposure using two
trailing signals — realized volatility (vol-target 12%/yr) and drawdown-from-peak (soft
de-risk at 10%, hard de-risk to cash at 20%). There is no forecasting and no stock-picking;
the edge, such as it is, is exposure to the equity risk premium with a defensive overlay. Per
the original audit (`docs/equity-harvester-verdict-2026-07-01-independent-audit.md` section 1),
the 0.908 number itself is mechanically real and not fabricated (HIGH confidence,
independently re-run twice by a separate agent to 0.907/0.921) — the issue is which universe
the number describes, not whether the computation is honest.

## 3. Why 0.908 is not the number to size against

`trained_data/backtests/SHIP_GATE.json` reports `net_sharpe: 0.908`, `max_dd: 0.229`,
`gate_pass: true`, `total_years: 17` on a curated ~20-name mega-cap-tilted pool (top-N-by-ADV
from `scripts/run_equity_harvester_shipgate_pit.py:54-66`'s `CANDIDATE_POOL`, which — after
point-in-time top-N-by-ADV selection — concentrates hard into AAPL/MSFT/GOOGL/AMZN/NVDA per the
audit's finding (b)). Concentrating a 20-name equal-weight book into the decade's best-performing
mega-caps mechanically inflates trailing Sharpe; it is a **selection/concentration effect**, not
model skill, since the same construction on a wider, survivorship-corrected universe produces a
materially lower number (below). This is not a bug, not leakage, not p-hacking — it's the same
code, same overlay params, same gate function, run on two different universes.

## 4. The honest number and its provenance

Source artifact: `trained_data/backtests/pit_quality_bakeoff.json` (built 2026-06-25, confirmed
present on disk 2026-07-01). Producing code: `src/equity/pit_quality_eval.py`,
`run_pit_quality_bakeoff()` (line 160), which calls `evaluate_book_wide()` (line 65) — the
**identical** construction as the shipped harvester (EW baseline via
`build_equal_weight_panel`, vol/DD overlay via `baseline_overlay` = `src/equity/backtest.py`'s
`overlay` with `target_vol=0.12`/`dd_soft=0.10`/`dd_hard=0.20`
[`src/equity/ship_gate.py:75-77`], gated via `src.factor.ship_gate.evaluate_gate`
[`src/factor/ship_gate.py:32`]) — but evaluated on the wide, survivorship-aware PIT S&P
membership reconstruction (`src/equity/sp500_membership.py`, Wikipedia change-log based),
with cheaper costs (flat 2bps only, `DEFAULT_COST_BPS=2.0` at `src/equity/ship_gate.py:82`,
no ADV slippage — see `evaluate_book_wide` call at `pit_quality_eval.py:96-99`,
`slippage_bps_per_pct_adv=0.0`).

Artifact fields (`trained_data/backtests/pit_quality_bakeoff.json`):
- `full_sample.baseline_ew.net_sharpe` = **0.74**, `max_dd` = **0.195**, `gate_pass` = **true**,
  `positive_years`/`total_years` = **12/15** (lines 17-23 of the JSON).
- `out_of_sample.baseline_ew.net_sharpe` = **0.355**, `max_dd` = **0.195**, `gate_pass` =
  **false**, `positive_years`/`total_years` = **5/6** (lines 36-42 of the JSON). OOS window
  is `2021-07-20` to `2026-06-24` (`out_of_sample.start`/`end`, lines 43/54), i.e. the trailing
  `oos_fraction=0.34` slice of the common window (`run_pit_quality_bakeoff(..., oos_fraction:
  float = 0.34)`, `pit_quality_eval.py:164`) — the audit doc's "OOS 2019+" is a loose paraphrase;
  the artifact's actual OOS start is 2021-07-20.
- `universe.evaluated_with_price_and_pit_fundamentals` = **624**,
  `universe.ever_members_in_window` = **873** (JSON lines 110-111) — matches the audit's
  "624-873 tickers" claim.
- `sub_period_stability` (JSON lines 83-108) shows the decay the audit cites: baseline net
  Sharpe **0.863** (2012-2016) → **0.974** (2017-2021) → **0.353** (2022-2026).

### Independent reproduction (this task, 2026-07-01)

Re-ran `run_pit_quality_bakeoff()` directly (not a re-read of the JSON) against the on-disk
cached price/fundamentals data (`market_data/equity/sp500_prices.parquet`,
`market_data/equity/sp500_pit_fundamentals.json`, both last refreshed 2026-06-25 and matching
the artifact's `common_window.end = 2026-06-24`). Result:

- Full sample: `net_sharpe=0.739`, `max_dd=0.195`, `positive_years=12/15`, `gate_pass=True`
- OOS: `net_sharpe=0.354`, `max_dd=0.195`, `positive_years=5/6`, `gate_pass=False`

This matches the artifact (0.74/0.355) and the audit's claim (0.740/0.355) to within float
rounding (~0.001) — **no material discrepancy**. Confidence: HIGH that the artifact numbers are
real and reproducible from the same code path. Caveat: this rerun used the cached parquet/JSON
data files rather than a fresh yfinance/EDGAR network pull, so it verifies "the code produces
this number from this data" rather than "the data hasn't drifted since 2026-06-25" — the
2026-07-01 independent audit's separate agent already did a fresh-network rerun of the
*curated-pool* script (`run_equity_harvester_shipgate_pit.py`) and got 0.907/0.921, so day-to-day
data-revision noise is known to be small (<0.02 Sharpe) for this pipeline.

## 5. Operator direction for this sleeve

Per operator direction (this task), the OOS 0.354/0.355 Sharpe **fails** the canonical ship gate
(`MIN_NET_SHARPE = 0.40` at `src/factor/ship_gate.py:18`) but is **acceptable to run as a
risk-controlled beta sleeve** — not because the gate is being waived in code (it is not; no
threshold or gate file was touched by this note), but because the operator has decided the
sleeve's function is honest risk-premium harvesting under an explicit drawdown overlay, not a
strategy required to independently clear the alpha ship-gate bar. This is an expectation-setting
and position-sizing distinction, not a code change: `SHIP_GATE.json`'s `gate_pass: true` remains
accurate for what it measures (the curated pool), and any future automated gate check against
`SHIP_GATE.json` will continue to see a PASS — this note does not alter that. What it does is
flag, in writing, that operators/consumers of this system should NOT read "gate_pass: true,
net_sharpe: 0.908" as the expected real-world return profile. Expect the broad-universe
full-sample number (~0.74) as the plausible long-run behavior, and treat the OOS figure (~0.35)
as the more conservative planning number for recent-regime (post-2021) performance, given the
sub-period decay already visible in the artifact (0.863 → 0.974 → 0.353).

## 6. Items flagged, not changed (out of scope for this note)

- `src/equity/harvester_strategy.py:86` / `:208-209` (`max_lev` default 1.0, configurable via
  `equity_harvester_max_lev`) and `src/equity/risk_agents.py:101` (`max_lev: float = 1.0`) are
  fixed constants, not calibrated to a specific Sharpe value — grepped, found no config file or
  leverage cap that implicitly assumes the 0.908 number. No sizing code found that needs
  flagging as "silently assuming the curated-pool Sharpe."
- Parameter provenance gap (audit finding (c)): shipped overlay params (`target_vol=12%,
  dd_soft=10%, dd_hard=20%`) don't trace to the one visible grid-search artifact
  (`trained_data/backtests/equity_harvester_20260618T202412Z.json`), which was run on the
  *sector* book, not the single-stock book, and whose actual winning config differs from what's
  shipped. Worth closing (re-run the grid on the actual PIT single-stock universe) but is a
  separate follow-up, not addressed here.
- `LiveGate.arm()` bypass in `--broker ibkr-paper` path (audit section 3, item 5) — separate
  wiring gap, not addressed here.

## Sources / verification trail

- `trained_data/backtests/SHIP_GATE.json` — curated-pool 0.908 net Sharpe, read fresh this turn.
- `trained_data/backtests/pit_quality_bakeoff.json` — broad-universe 0.74/0.355, read fresh this
  turn (lines cited above).
- `src/equity/pit_quality_eval.py` — producing code, read in full this turn (323 lines).
- `scripts/run_equity_harvester_shipgate_pit.py` — curated-pool shipgate driver, read in full
  this turn (227 lines).
- `src/equity/ship_gate.py:75-82` — overlay defaults (`DEFAULT_TARGET_VOL=0.12`,
  `DEFAULT_DD_SOFT=0.10`, `DEFAULT_DD_HARD=0.20`, `DEFAULT_COST_BPS=2.0`).
- `src/factor/ship_gate.py:18-21` — canonical gate thresholds
  (`MIN_NET_SHARPE=0.40`, `MIN_POSITIVE_YEARS=6`, `MIN_TOTAL_YEARS=10`, `MAX_DRAWDOWN=0.25`).
- Live re-run of `run_pit_quality_bakeoff()` against cached `market_data/equity/sp500_prices.parquet`
  (mtime 2026-06-25 07:23, 3639 rows × 684 tickers, index 2012-01-03 to 2026-06-24) and
  `market_data/equity/sp500_pit_fundamentals.json` (mtime 2026-06-25 07:28) — executed this turn,
  output: `net_sharpe=0.739` full / `0.354` OOS, `gate_pass=True` / `False` respectively.
- `docs/equity-harvester-verdict-2026-07-01-independent-audit.md` — original independent audit
  this note operationalizes.
