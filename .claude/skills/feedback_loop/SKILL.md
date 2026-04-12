---
name: feedback_loop
description: "Post-trade feedback loop system: shaped-reward logging, diagnostics, and self-heal. Use when investigating trade outcomes, checking system health, or understanding automated corrective actions."
user-invocable: false
---

# Post-Trade Feedback Loop

Automated post-trade analysis pipeline that logs shaped rewards, runs health diagnostics, and applies corrective actions without human intervention.

---

## Loop Diagram

```
Trade Close (OANDA)
       |
       v
PostTradeLoop.run(trade_result)
       |
       +---> [1] Idempotency check (trade_id dedup)
       |
       +---> [2] Load journal entry from trade_journal_rl.json
       |
       +---> [3] Compute shaped reward
       |
       +---> [4] Append to logs/trade_feedback.jsonl
       |
       +---> [5] Append learning if |reward| > 2R
       |
       +---> [6] PostTradeDiagnostics.run()
       |            |
       |            v
       |     Structured health report
       |     {status: HEALTHY|DEGRADED|CRITICAL}
       |            |
       |            v (if not HEALTHY)
       +---> [7] SelfHeal.apply(diag_result)
                     |
                     v
              Decision log + corrective actions
```

---

## When It Fires

- **Per trade close**: Hooked into `close_trade_oanda()` in `src/scanner/execution.py`. Every confirmed OANDA trade close triggers `PostTradeLoop.run()`.
- **Per cycle**: The orchestrator dispatches feedback processing each scan cycle via `run_cycle()`.
- **Manual**: Via MCP tools (`trigger_rl_update`, `trigger_diagnostics`, `trigger_self_heal`) or CLI (`python main.py feedback-status`).

---

## What It Does NOT Do

The feedback loop does **NOT** update agent weights. Weight updates remain on the existing batch path in `execution.sync_closed_trades_rl()`. This is a hard scope rule (Parity Q5 compliance). The loop only logs shaped rewards and triggers diagnostics/self-heal.

---

## Shaped Reward Formula

```
shaped_reward = pnl_pips - slippage_pips - 0.5 * mae_pips
```

| Component | Source | Description |
|-----------|--------|-------------|
| `pnl_pips` | `outcome.pnl_pips` or `entry.pnl_pips` | Gross P/L in pips |
| `slippage_pips` | `entry.slippage_pips` | Entry slippage cost |
| `mae_pips` | `outcome.mae_pips` | Maximum adverse excursion (drawdown proxy) |

Coefficients are class constants on `PostTradeLoop`: `SLIP_WEIGHT = 1.0`, `MAE_WEIGHT = 0.5`.

If `|shaped_reward| > 2 * sl_pips`, the trade is considered a significant outcome and an entry is appended to `.claude/learnings.md`.

---

## Diagnostics: The 5 Health Checks

`PostTradeDiagnostics.run()` returns a structured report with status `HEALTHY`, `DEGRADED`, or `CRITICAL`.

### Status Aggregation

- **HEALTHY**: No non-skipped issues found.
- **DEGRADED**: At least one `warning`-severity issue, no `critical`.
- **CRITICAL**: At least one `critical`-severity issue.

### Check Details

| # | Check | What It Detects | Warning Threshold | Critical Threshold |
|---|-------|-----------------|--------------------|--------------------|
| 1 | `direction_accuracy` | Win rate over last 20 trades | < 0.52 | < 0.48 |
| 2 | `gate_firing_rate` | Per-gate pass rate over last 50 entries | < 0.05 (stuck off) or > 0.95 (stuck on) | N/A (warning only) |
| 3 | `agent_weight_boundary` | Agent weights clamped at min (0.1) or max (2.0) | Any agent at boundary | N/A (warning only) |
| 4 | `drawdown_streak` | Consecutive losses at journal tail | >= 5 streak | >= 8 streak |
| 5 | `rl_model_staleness` | Age of `rl_position_sizer.zip` model file | > 48 hours | > 168 hours (1 week) |

All thresholds are overridable in `config/config_improved_H1.yaml` under `feedback_loop.diagnostics.*`.

### Result Schema

```json
{
  "status": "DEGRADED",
  "issues": [
    {
      "check": "direction_accuracy",
      "severity": "warning",
      "detail": "win rate 0.500 below warning threshold",
      "value": 0.5,
      "threshold": 0.52
    }
  ],
  "recommended_actions": ["retrain_gates"],
  "metadata": {
    "trades_analyzed": 150,
    "timestamp": "2026-04-12T10:30:00+00:00",
    "entry_trade_id": "12345"
  }
}
```

---

## Self-Heal Decision Table

When diagnostics returns a non-HEALTHY status, `SelfHeal.apply()` processes each `recommended_actions` entry through its dispatch table.

| Diagnostic Check | Recommended Action | What Self-Heal Does |
|------------------|--------------------|---------------------|
| `direction_accuracy` | `retrain_gates` | Calls `OnlineRetrainer.trigger_retrain()` (respects 60-min cooldown) |
| `gate_firing_rate` (stuck off) | `reset_gate_threshold_to_default:<gate>` | Resets gate threshold to yaml default in `config_adjustments.json` |
| `gate_firing_rate` (stuck on) | `tighten_gate_threshold:<gate>` | Multiplies gate threshold by 1.10x in `config_adjustments.json` |
| `agent_weight_boundary` | `soft_reset_agent_weight:<agent>` | Resets agent weight to 1.0 baseline in `agent_weights.json` (atomic write) |
| `drawdown_streak` | `reduce_risk_per_trade_pct` | Applies 0.75x risk multiplier in `config_adjustments.json` |
| `rl_model_staleness` | `retrain_rl_position_sizer` | Writes a retrain-request marker file to `trained_data/retrain_requests/` |

### Return Schema

```json
{
  "status": "applied",
  "actions_taken": [
    {"action": "retrain_gates", "success": true, "detail": "retrain_status:completed"}
  ],
  "files_modified": [],
  "degraded_mode": false,
  "error": null
}
```

---

## Degraded Mode

When a CRITICAL issue is detected or any self-heal handler raises an exception, the system enters degraded mode.

- **Marker file**: `trained_data/self_heal_degraded.flag` (JSON with timestamp, reason, source).
- **Effect**: The next scan cycle reads this flag and runs in conservative fallback (reduced position sizing, tighter gates).
- **Clearing**: Delete the flag file manually or let a subsequent healthy diagnostics cycle clear it.

```bash
# Check if degraded mode is active
cat trained_data/self_heal_degraded.flag

# Clear degraded mode manually
rm trained_data/self_heal_degraded.flag
```

---

## Manual Triggers

### Via MCP Tools

```
trigger_diagnostics        # Run the 5-check diagnostics suite
trigger_self_heal          # Run diagnostics + apply corrective actions
trigger_rl_update(trade_id)  # Run full feedback loop for one trade
```

### Via CLI

```bash
python main.py feedback-status   # Show current feedback loop state
```

---

## Log Files

| File | Format | Contents |
|------|--------|----------|
| `logs/trade_feedback.jsonl` | JSONL | One record per processed trade: shaped_reward, pnl_pips, slippage, mae, regime, source |
| `logs/system_decisions.jsonl` | JSONL | One record per self-heal action: trigger, issue, action_taken, files_modified, degraded_mode |
| `logs/unhandled_errors.jsonl` | JSONL | Safety-net errors from post_trade_loop or self_heal modules |
| `.claude/learnings.md` | Markdown | Significant outcomes (>= 2R) appended as dated bullet entries |

---

## Idempotency

The feedback loop deduplicates by `trade_id`. Before processing, `PostTradeLoop._already_processed()` scans the last 500 lines of `logs/trade_feedback.jsonl` for a matching `trade_id` string. If found, the trade is skipped with `{"status": "skipped", "reason": "already_processed"}`.

This means you can safely call `PostTradeLoop.run()` multiple times for the same trade without producing duplicate records.

---

## Escalation Criteria

Alert a human when:

1. **CRITICAL status on two consecutive cycles** -- the self-heal actions are not resolving the underlying issue.
2. **Repeated self-heal failures** -- multiple `"success": false` entries in `logs/system_decisions.jsonl` for the same action type.
3. **Degraded mode persisting > 1 hour** -- check the timestamp in `trained_data/self_heal_degraded.flag`. If it is older than 1 hour and still present, manual investigation is required.

---

## Key Source Files

- `src/scanner/feedback/post_trade_loop.py` -- `PostTradeLoop.run()` entry point
- `src/scanner/feedback/diagnostics.py` -- `PostTradeDiagnostics.run()` health checks
- `src/scanner/feedback/self_heal.py` -- `SelfHeal.apply()` corrective actions
- `src/scanner/execution.py` -- Hook site in `close_trade_oanda()`
