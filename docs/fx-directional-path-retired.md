# FX directional path — RETIRED (2026-06-21, US-020)

## Status

The FX directional engine path and its 15 specialist agents are **retired
from the active runtime**. The autonomous bot now runs the equity-beta
harvester (`src/equity/control_loop.py`) — see
`tasks/prd-equity-harvester-bot.md`.

**Nothing has been hard-deleted.** Every FX module under `src/scanner/`,
every model under `trained_data/models/`, every entry in
`trained_data/trade_journal_rl.json`, and every backtest artefact remains
on disk and in git history. The retirement is enforced as an explicit
*opt-in* guard, not as a removal — see "How the retirement is enforced"
below.

## Verdicts that closed the loop

1. **`docs/fx-edge-search-final-verdict-2026-06-18.md`** — own-data 22-yr
   walk-forward daily direction balanced accuracy = 50.3% (effectively
   random). Majors ≈ random walk; published claims of 58%+ provably
   artefacts of leakage, 2022-trend bleed, or feature-search bias.
2. **`docs/equity-harvester-verdict-2026-06-18.md`** + **`docs/harvester-verdict-2026-06-18.md`**
   — the equity harvester (vol-managed beta + drawdown circuit-breaker)
   clears the pre-registered ship gate on a single-stock universe:
   - Net Sharpe 0.92 (2010-2026), 0.80 GFC-inclusive
   - MaxDD 0.229 both windows
   - 13 positive years out of 17 (≥ 6 required)
3. **`tasks/prd-equity-harvester-bot.md`** — operator decision on
   2026-06-18 to retire the FX path non-destructively and re-point the
   autonomous machinery (control plane, drawdown guardian, self-heal,
   state persistence, `AgentVerdict`) onto the harvester.

The verdict is **risk-premium harvesting, not directional prediction.**
That ends the FX directional thesis at this data scale.

## The 15 retired directional agents

Source of truth: `src.scanner.agents._team.ScannerAgentTeam._BASE_WEIGHTS`.
Mirror in `src.scanner.fx_retired.FX_DIRECTIONAL_AGENT_NAMES`. A canary
test asserts the two stay in sync.

| Agent | Base weight | Flag |
|-------|-------------|------|
| trend | 1.15 | `enable_trend_agent` |
| mean_reversion | 0.90 | `enable_mean_reversion_agent` |
| volatility | 1.00 | `enable_volatility_agent` |
| risk_sentinel | 1.25 | `enable_risk_sentinel_agent` |
| uncertainty | 1.10 | `enable_uncertainty_agent` |
| execution_quality | 1.05 | `enable_execution_quality_agent` |
| news_risk | 0.95 | `enable_news_risk_agent` |
| multi_timeframe | 1.10 | `enable_multi_timeframe_agent` |
| pair_performance | 0.85 | `enable_pair_performance_agent` |
| momentum | 1.05 | `enable_momentum_agent` |
| session_timing | 0.80 | `enable_session_timing_agent` |
| support_resistance | 1.00 | `enable_support_resistance_agent` |
| order_flow | 0.95 | `enable_order_flow_agent` |
| trader_readiness | 0.50 | `enable_trader_readiness_agent` |
| devil_advocate | 1.30 | `enable_devil_advocate_agent` (+ `enable_devil_advocate` legacy) |

## How the retirement is enforced

`src/scanner/fx_retired.py` is the single, machine-checkable retirement
entry point:

1. **`disable_fx_directional_path(config)`** — flips every agent flag
   on a `ScannerConfig` (or any attribute-bag) to `False`. Idempotent.
   Returns the list of flags actually flipped.
2. **`enforce_harvester_ship_gate(expected_universe_hash, ship_gate_path=None)`**
   — refuses to allow the retired FX path to start as a runtime unless
   `trained_data/backtests/SHIP_GATE.json` exists, contains
   `gate_pass == true`, and has a `universe_hash` matching the
   harvester universe the operator is deploying. Missing / false /
   mismatched → `FXPathRetiredError`. Behaviour is byte-for-byte
   identical to `src.equity.control_loop._enforce_ship_gate`.
3. **No runtime path still depends on the retired agents.** The
   harvester engine path (`src/equity/`) does not import any of the 15
   directional agent evaluators. The equity layer imports the
   *type only* (`AgentVerdict` dataclass from
   `src.scanner.agents._team`) — that's a shared verdict shape, not a
   live agent. A canary test asserts this stays true.

## What the FX code is still good for

- **Reproducing prior backtests** — the FX scanner remains runnable in
  test / research notebooks. It just cannot be invoked as a live
  autonomous runtime; the ship-gate guard catches every realistic
  start path.
- **Audit trail** — `trained_data/trade_journal_rl.json` and the
  agent-weights snapshots stay on disk; reflections and post-mortems
  that cite "what FX did" can still resolve.

## Re-activating the FX path

Don't. The decision record is in
`tasks/prd-equity-harvester-bot.md` (US-020). If a future operator has
new evidence that overturns the 2026-06-18 verdicts, they must:

1. Open a new PRD that cites the contradicting evidence.
2. Re-run the daily-direction walk-forward from
   `scripts/experiment_daily_direction_oos.py` and show ≥ 53% balanced
   accuracy out-of-sample.
3. **Then** add a flag to opt back in — do not silently flip
   `disable_fx_directional_path` to a no-op.

## See also

- `tasks/prd-equity-harvester-bot.md` — full PRD (US-020 in §
  "Acceptance Criteria")
- `src/scanner/fx_retired.py` — the retirement module
- `tests/test_fx_path_retired.py` — the no-mock acceptance tests
- `src/equity/control_loop.py` — the new runtime
