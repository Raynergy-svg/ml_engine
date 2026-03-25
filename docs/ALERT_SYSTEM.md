# Real-Time Alert System — ML Engine

## Overview

The Alert System provides operational monitoring for the ML Engine trading bot. It fires threshold-based alerts when key risk metrics are exceeded, allowing operators to intervene before large drawdowns or other critical events.

**Status:** OPERATIONAL
**Location:** `src/scanner/automation/alert_manager.py`
**Integration:** Continuous scanner (`src/scanner/automation/continuous.py`, step 5e-i)

## Alert Types

### 1. Drawdown Alert
**Severity:** CRITICAL
**Default Threshold:** 5% (0.05)
**Cooldown:** 60 minutes

Fires when portfolio equity falls below 95% of the peak NAV.

```python
# Example: Portfolio hits $9,400 from peak of $10,000
drawdown = (10000 - 9400) / 10000 = 0.06 (6%)
# Alert fires because 6% > 5% threshold
```

**Action:** Check for consecutive losses, review recent trades, consider pausing new entries.

---

### 2. Consecutive Losses Alert
**Severity:** WARNING
**Default Threshold:** 3 consecutive losses
**Cooldown:** 60 minutes

Fires when the last N trades are all losses, indicating a potential regime shift.

```python
# Example: Last 3 closed trades all lost money
recent_trades = [
    {"outcome": {"trade_won": False}},  # Loss
    {"outcome": {"trade_won": False}},  # Loss
    {"outcome": {"trade_won": False}},  # Loss
]
# Alert fires
```

**Action:** Review agent weights, check market conditions, consider reducing position size.

---

### 3. Win Rate Drop Alert
**Severity:** WARNING
**Default Threshold:** 10% drop over rolling 20-trade window
**Cooldown:** 120 minutes

Fires when win rate over the last 20 trades has dropped 10% from the overall average.

```python
# Example: Overall win rate 60%, recent 20 trades win rate 50%
drop = 0.60 - 0.50 = 0.10 (10%)
# Alert fires because drop == threshold
```

**Action:** Review configuration tuning, check if market regime has changed, audit learnings.

---

### 4. Weight Instability Alert
**Severity:** WARNING
**Default Threshold:** 0.30 max agent weight change in one cycle
**Cooldown:** 60 minutes

Fires when any agent's weight changed by more than 0.30 in a single update cycle, indicating aggressive RL learning.

```python
# Example: Trend agent weight changed from 0.5 to 0.8
change = abs(0.8 - 0.5) = 0.30
# Alert fires at threshold
```

**Action:** Review RL weight updates, verify trade outcomes feeding back correctly.

---

## Configuration

### Default Alert Thresholds

```python
ALERT_CONFIGS = {
    "drawdown": {
        "threshold": 0.05,          # 5% drawdown
        "severity": "CRITICAL",
        "cooldown_minutes": 60,
        "message_template": "..."
    },
    "consecutive_losses": {
        "threshold": 3,             # 3 losses in a row
        "severity": "WARNING",
        "cooldown_minutes": 60,
    },
    "win_rate_drop": {
        "threshold": 0.10,          # 10% drop
        "severity": "WARNING",
        "cooldown_minutes": 120,
    },
    "weight_instability": {
        "threshold": 0.30,          # 0.30 weight change
        "severity": "WARNING",
        "cooldown_minutes": 60,
    },
}
```

### Customizing Thresholds

Create a custom `AlertManager` with looser thresholds:

```python
from src.scanner.automation.alert_manager import AlertManager

custom_config = {
    "drawdown": {
        "threshold": 0.10,  # 10% instead of 5%
        "severity": "WARNING",
        "cooldown_minutes": 30,
        "message_template": "Portfolio drawdown {value:.1%} exceeds {threshold:.1%}",
    }
}

alert_mgr = AlertManager(custom_configs=custom_config)
```

---

## Usage

### In the Continuous Scanner

Alerts are automatically checked every scan cycle in the learning loop:

```python
# Step 5e-i: Run alert checks (in continuous.py _run_learning_loop)
alert_mgr = AlertManager()

fired_alerts = alert_mgr.check_all(
    nav=portfolio_nav,
    peak_nav=peak_nav,
    recent_trades=recent_trades,
    current_weights=current_weights,
    previous_weights=previous_weights,
)

if fired_alerts:
    for alert in fired_alerts:
        logger.warning(f"[{alert.severity}]: {alert.message}")
```

### Standalone Usage

```python
from src.scanner.automation.alert_manager import AlertManager

# Initialize
alert_mgr = AlertManager(
    alert_log_path="trained_data/alerts.jsonl",
    state_path=".claude/alert_state.json"
)

# Check individual metrics
drawdown_alert = alert_mgr.check_drawdown(
    current_nav=9500,
    peak_nav=10000
)

loss_alert = alert_mgr.check_consecutive_losses(recent_trades)

# Get summary
summary = alert_mgr.get_summary()
print(summary)

# Acknowledge alerts
alert_mgr.acknowledge("drawdown")

# Get active unacknowledged alerts
active = alert_mgr.get_active_alerts()
```

---

## State Persistence

Alert state is persisted to `.claude/alert_state.json` for cross-session continuity:

```json
{
  "last_fired": {
    "drawdown": "2026-03-19T10:30:00.123456",
    "consecutive_losses": null
  },
  "active_alerts": [
    {
      "alert_type": "drawdown",
      "severity": "CRITICAL",
      "message": "Portfolio drawdown 6.0% exceeds 5.0% threshold",
      "timestamp": "2026-03-19T10:30:00.123456",
      "value": 0.06,
      "threshold": 0.05,
      "pair": "",
      "acknowledged": false
    }
  ],
  "last_updated": "2026-03-19T10:30:00.123456"
}
```

**File Locking:** All writes use `fcntl` locking to prevent concurrent writes from scanner processes.

---

## Alert Output

### Log Files

#### `trained_data/alerts.jsonl`
JSONL-formatted alert log (one JSON object per line):

```jsonl
{"alert_type":"drawdown","severity":"CRITICAL","message":"Portfolio drawdown 6.0% exceeds 5.0% threshold","timestamp":"2026-03-19T10:30:00.123456","value":0.06,"threshold":0.05,"pair":"","acknowledged":false}
{"alert_type":"consecutive_losses","severity":"WARNING","message":"3 consecutive losses (threshold: 3)","timestamp":"2026-03-19T10:45:00.234567","value":3.0,"threshold":3.0,"pair":"","acknowledged":false}
```

#### Console Output
In the continuous scanner (with Rich formatting):

```
⚠ ALERT [CRITICAL]: Portfolio drawdown 6.0% exceeds 5.0% threshold
⚠ ALERT [WARNING]: 3 consecutive losses (threshold: 3)
```

#### Log Messages
Via Python logging:

```python
logger.warning(f"[CRITICAL] drawdown: Portfolio drawdown 6.0% exceeds 5.0% threshold")
```

---

## Cooldown System

The cooldown system prevents alert spam by suppressing duplicate alerts for a configured period.

### How It Works

1. First alert fires → cooldown timer starts
2. Same alert type triggered again within cooldown period → **suppressed**
3. Cooldown expires → next alert can fire

### Cooldown Periods (Default)

| Alert Type | Cooldown |
|---|---|
| Drawdown | 60 minutes |
| Consecutive Losses | 60 minutes |
| Win Rate Drop | 120 minutes |
| Weight Instability | 60 minutes |

### Override Cooldowns (Testing)

```python
# Clear all cooldowns
alert_mgr.clear_cooldowns()

# Or set custom cooldowns during init
alert_mgr = AlertManager(
    custom_configs={
        "drawdown": {
            "threshold": 0.05,
            "severity": "CRITICAL",
            "cooldown_minutes": 5,  # Short cooldown for testing
            "message_template": "..."
        }
    }
)
```

---

## Error Handling

The alert system is designed to **never crash** the scanner:

- All alert checks wrapped in try/except
- File I/O errors logged but don't halt scanning
- Corrupted state files gracefully restart fresh
- Malformed JSON entries skipped with logging

```python
try:
    fired_alerts = alert_mgr.check_all(...)
except Exception as e:
    logger.warning(f"Alert check error: {e}")
    # Scanner continues running
```

---

## Integration with Trading Rules

Per `.claude/rules/trading.md`:

- **Drawdown Guardian:** Runs every scan cycle (non-negotiable)
  - Drawdown alert triggers when guardian's thresholds are exceeded

- **Maximum Portfolio Risk:** 15% of NAV across all open positions
  - Win rate drop alert indicates risk management drift

- **Agent Consensus:** Higher vote scores correlate with better outcomes
  - Weight instability alert indicates RL learning turbulence

---

## Monitoring Dashboard

### Alert Health Metrics

To monitor alert system health:

```python
from pathlib import Path
import json

alerts_log = Path("trained_data/alerts.jsonl")
lines = alerts_log.read_text().strip().split('\n')
alerts = [json.loads(line) for line in lines if line]

# Alert count by type
by_type = {}
for alert in alerts:
    t = alert['alert_type']
    by_type[t] = by_type.get(t, 0) + 1

print(f"Total alerts: {len(alerts)}")
print(f"By type: {by_type}")
print(f"CRITICAL: {sum(1 for a in alerts if a['severity'] == 'CRITICAL')}")
print(f"WARNING: {sum(1 for a in alerts if a['severity'] == 'WARNING')}")
```

---

## Best Practices

### 1. Review Alerts Regularly
- Check `trained_data/alerts.jsonl` weekly
- Look for patterns: are certain pairs triggering consecutive loss alerts?
- Identify if thresholds need adjustment

### 2. Acknowledge Alerts After Resolution
```python
alert_mgr.acknowledge("drawdown")  # Marks as handled
```

### 3. Adjust Thresholds Based on Account Size
Larger accounts may need looser thresholds (higher drawdown tolerance):
```python
if account_size > 10000:
    custom_config["drawdown"]["threshold"] = 0.10  # 10% for larger accounts
```

### 4. Monitor Weight Instability
High weight instability can indicate:
- RL learning is too aggressive
- Recent trades are highly contradictory
- Market regime shift

### 5. Pair Win Rate Drops with Market Analysis
Win rate drops often correspond to:
- Major economic news events
- Market regime shifts (trend → consolidation)
- New time zones opening (volatility changes)

---

## Testing

Run the validation suite:

```bash
cd /path/to/ml_engine
python src/scanner/automation/test_alert_manager.py
```

Expected output:
```
✓ Test 1: Initialization
✓ Test 2: Drawdown detection
✓ Test 3: Consecutive losses detection
✓ Test 4: Weight instability detection
✓ Test 5: Cooldown blocking
✓ Test 6: State persistence
✓ Test 7: Active alerts tracking
✓ Test 8: Alert acknowledgment
✓ Test 9: Check all alerts simultaneously
✓ Test 10: Alert summary
✓ Test 11: JSONL alert logging

ALL TESTS PASSED ✓
```

---

## Files

| File | Purpose |
|---|---|
| `src/scanner/automation/alert_manager.py` | Core alert system |
| `src/scanner/automation/test_alert_manager.py` | Unit tests |
| `src/scanner/automation/continuous.py` | Integration point (step 5e-i) |
| `trained_data/alerts.jsonl` | Alert log (JSONL format) |
| `.claude/alert_state.json` | Cooldown + active alert state |
| `docs/ALERT_SYSTEM.md` | This document |

---

## Troubleshooting

### No Alerts Firing

1. Check if cooldown is active:
   ```python
   print(alert_mgr._last_fired)
   ```

2. Verify threshold values:
   ```python
   print(alert_mgr._configs["drawdown"]["threshold"])
   ```

3. Check log level (must be WARNING or higher):
   ```python
   import logging
   logging.getLogger("src.scanner.automation.alert_manager").setLevel(logging.WARNING)
   ```

### State File Corruption

Delete `.claude/alert_state.json` to restart fresh:
```bash
rm .claude/alert_state.json
```

The alert manager will gracefully reinitialize on next run.

### File Permission Issues

Ensure write access to `trained_data/` and `.claude/`:
```bash
chmod -R 755 trained_data/ .claude/
```

---

## Future Enhancements

Potential additions (not yet implemented):

1. **Webhook Alerts** — POST to Discord/Slack on critical alerts
2. **Email Notifications** — Send email on CRITICAL alerts
3. **Custom Alert Triggers** — User-defined threshold expressions
4. **Alert History Dashboard** — Web UI showing alert trends
5. **Adaptive Thresholds** — Auto-adjust based on market regime

---

## Related Documentation

- `.claude/rules/trading.md` — Trading execution gates
- `docs/ARCHITECTURE_UPGRADE_PLAN.md` — System roadmap
- `src/scanner/execution.py` — Drawdown guardian implementation

