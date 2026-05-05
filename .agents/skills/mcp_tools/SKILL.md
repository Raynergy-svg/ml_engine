---
name: mcp_tools
description: "Buddy MCP server: 15 tools for system observability, feedback loop control, OANDA state, and knowledge management. Use when querying system health, investigating trades, or triggering diagnostics/self-heal."
user-invocable: false
---

# Buddy MCP Tools

15 tools via Model Context Protocol (stdio transport) for orchestrator observability, feedback loop control, OANDA queries, and knowledge management.

---

## Server Launch

```bash
python src/mcp/buddy_server.py    # stdio transport
```

Register in Codex Desktop `mcp.json`:
```json
{
  "mcpServers": {
    "buddy": {
      "command": "python",
      "args": ["src/mcp/buddy_server.py"],
      "env": {"OANDA_API_TOKEN": "your-token", "OANDA_ACCOUNT_ID": "your-account-id"}
    }
  }
}
```

---

## Tool Reference

### 1. `health_check` -- Aggregate health snapshot

**Params**: None. **Use**: Start of any investigation, morning check, post-deployment.

```json
{"diagnostics": {"status": "HEALTHY", "issues": []}, "last_feedback": {"trade_id": "12345", "shaped_reward": 8.5}, "last_decision": null, "agent_weights_count": 48, "rl_model_age_hours": 12.3, "overall": "HEALTHY"}
```

### 2. `get_system_status` -- Orchestrator module/session/improvement status

**Params**: None. **Use**: When you need module availability or session stats beyond `health_check`.

### 3. `get_trade_feedback_log` -- Tail of trade feedback JSONL

**Params**: `n: int = 50`. **Use**: After trade close, shaped reward trend analysis, post-loss review.

```json
{"count": 1, "records": [{"trade_id": "12345", "pair": "EUR_USD", "shaped_reward": 8.5, "pnl_pips": 12.0, "slippage_pips": 1.5, "mae_pips": 4.0, "regime": "NORMAL"}]}
```

### 4. `get_system_decisions` -- Tail of self-heal decision JSONL

**Params**: `n: int = 50`. **Use**: After self-heal runs, verifying corrective actions, auditing decisions.

```json
{"count": 1, "records": [{"trigger": "drawdown_streak", "issue": "5 consecutive losses", "action_taken": "reduce_risk_per_trade_pct", "degraded_mode": false}]}
```

### 5. `get_agent_weights` -- Current agent weights by regime bucket

**Params**: None. **Use**: Investigating agent performance, post self-heal reset, post-loss analysis.

```json
{"NORMAL": {"trend": 1.2, "mean_reversion": 0.8}, "_meta": {"last_updated": "2026-04-12T08:00:00+00:00"}}
```

### 6. `get_gate_health` -- Gate firing rate issues only

**Params**: None. **Use**: When gates seem too tight (blocking all) or too loose (passing all).

```json
{"status": "DEGRADED", "gate_issues": [{"check": "gate_firing_rate", "severity": "warning", "detail": "gate 'confidence_passed' pass rate 0.980 above ceiling"}], "thresholds": {"gate_broken_off_lt": 0.05, "gate_broken_on_gt": 0.95, "gate_window": 50}}
```

### 7. `get_learnings` -- Read .Codex/learnings.md

**Params**: None. **Use**: Reviewing insights, checking patterns, following up on `health_check` issues. Returns `"truncated": true` if file exceeds 100 KB.

### 8. `trigger_diagnostics` -- Run the 5-check diagnostics suite

**Params**: None. **Use**: Get a health report without applying corrections. Use before `trigger_self_heal` to review first.

```json
{"status": "DEGRADED", "issues": [{"check": "drawdown_streak", "severity": "warning", "value": 5, "threshold": 5}], "recommended_actions": ["reduce_risk_per_trade_pct"], "metadata": {"trades_analyzed": 150}}
```

### 9. `trigger_self_heal` -- Diagnostics + corrective actions

**Params**: `diag_result: dict | null = null` (if null, runs diagnostics first). **Use**: Automated correction of flagged issues.

```json
{"diagnostics": {"status": "DEGRADED"}, "heal": {"status": "applied", "actions_taken": [{"action": "reduce_risk_per_trade_pct", "success": true}], "files_modified": [".Codex/config_adjustments.json"], "degraded_mode": false}}
```

### 10. `trigger_rl_update` -- Feedback loop for one trade

**Params**: `trade_id: str` (required). **Use**: Manually process a specific trade through shaped-reward logging, diagnostics, self-heal. Does NOT update agent weights.

```json
{"status": "processed", "trade_id": "12345", "shaped_reward": 8.5, "diagnostics": {"status": "HEALTHY"}, "self_heal": null}
```

### 11. `trigger_gate_retrain` -- Retrain gate models

**Params**: None. **Use**: When direction accuracy is degraded. Respects 60-minute cooldown. Returns `{"error": "online_retrainer not available"}` if module missing.

### 12. `write_learning` -- Append to learnings.md

**Params**: `content: str` (required, max 5000 chars). **Use**: Record a pattern or insight from trade analysis.

```json
{"status": "appended", "path": ".Codex/learnings.md"}
```

### 13. `write_rule` -- Create a rule file

**Params**: `filename: str` (must match `[a-z0-9_]+.md`), `content: str` (required). **Use**: Promote a 3x-observed learning to a rule. Cannot overwrite `trading.md` or `improvement.md`.

### 14. `get_open_positions` -- OANDA open positions

**Params**: None. **Use**: Check current exposure, pre-trade risk assessment. **Requires**: `OANDA_API_TOKEN` + `OANDA_ACCOUNT_ID` env vars.

```json
{"trades": [{"tradeID": "12345", "instrument": "EUR_USD", "currentUnits": "1000", "unrealizedPL": "3.50"}]}
```

### 15. `get_closed_trades` -- OANDA closed trade history

**Params**: `count: int = 50` (max 500). **Use**: Review recent history, correlate with feedback log. **Requires**: `OANDA_API_TOKEN` + `OANDA_ACCOUNT_ID` env vars.

```json
{"trades": [{"tradeID": "12340", "instrument": "GBP_USD", "realizedPL": "-5.20", "closeTime": "2026-04-12T09:15:00Z"}]}
```

---

## Tool Chains

### Morning Health Check
`health_check` --> review status --> if issues: `get_learnings` to check for recurring patterns.

### Post-Loss Investigation
`get_trade_feedback_log(5)` --> `trigger_diagnostics` --> `get_agent_weights` --> decide: `trigger_self_heal` or escalate to human.

### Gate Tune
`get_gate_health` --> `trigger_self_heal` if issues --> `get_system_decisions` to verify actions applied.

### Full Reset
`trigger_diagnostics` --> `trigger_self_heal` --> `trigger_gate_retrain` --> `health_check` to confirm HEALTHY.

---

## OANDA Tools Caveat

Tools 14 and 15 require `OANDA_API_TOKEN` + `OANDA_ACCOUNT_ID` env vars. If missing, they return `{"error": "credentials_missing"}`. All other tools work without credentials.

## Error Handling

Every tool returns structured JSON even on failure. Check for the `"error"` key in every response before acting on data. No tool raises an unhandled exception to the MCP client.

---

## Key Source File

- `src/mcp/buddy_server.py` -- All 15 tool definitions, Pydantic return models, FastMCP server setup
