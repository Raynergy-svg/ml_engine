# Alert System — Quick Start Guide

## What is it?

The Alert System monitors your ML Engine trading bot for critical risk events:

- **Drawdown** — Portfolio drops more than 5%
- **Consecutive Losses** — 3+ losses in a row
- **Win Rate Drop** — Win rate falls 10% over last 20 trades
- **Weight Instability** — Agent weights change too rapidly

## How it works

Alerts automatically run **every scan cycle** in the continuous scanner. They fire alerts to the console and log to `trained_data/alerts.jsonl`.

## Files

```
src/scanner/automation/
├── alert_manager.py           # Core alert system
├── continuous.py              # Integration (step 5e-i)
└── test_alert_manager.py      # Tests

trained_data/
└── alerts.jsonl               # Alert log

.claude/
└── alert_state.json           # Cooldown + state

docs/
└── ALERT_SYSTEM.md            # Full documentation
```

## Enable it (automatic)

Alerts are **enabled by default** when running continuous scan:

```bash
python -m buddy_scanner scan --watch --interval 5
```

You'll see alerts in the console:

```
⚠ ALERT [CRITICAL]: Portfolio drawdown 6.0% exceeds 5.0% threshold
⚠ ALERT [WARNING]: 3 consecutive losses (threshold: 3)
```

## Customize thresholds

Edit thresholds in `continuous.py` (around line 435) or create custom config:

```python
from src.scanner.automation.alert_manager import AlertManager

custom = {
    "drawdown": {
        "threshold": 0.10,  # 10% instead of 5%
        "cooldown_minutes": 30,  # Faster re-alert
        # ... rest of config
    }
}

alert_mgr = AlertManager(custom_configs=custom)
```

## View alert history

Check the alert log:

```bash
tail -20 trained_data/alerts.jsonl
```

Each line is a JSON object:
```json
{"alert_type":"drawdown","severity":"CRITICAL","message":"Portfolio drawdown 6.0% exceeds 5.0% threshold","timestamp":"2026-03-19T10:30:00","value":0.06,"threshold":0.05}
```

## Check active alerts

```python
from src.scanner.automation.alert_manager import AlertManager

alert_mgr = AlertManager()
active = alert_mgr.get_active_alerts()
print(alert_mgr.get_summary())
```

Output:
```
⚠ 2 active alert(s):
  [CRITICAL] drawdown: Portfolio drawdown 6.0% exceeds 5.0% threshold
  [WARNING] consecutive_losses: 3 consecutive losses (threshold: 3)
```

## Acknowledge an alert

Mark as reviewed:

```python
alert_mgr.acknowledge("drawdown")  # Removes from active list
```

## Reset cooldowns (testing only)

```python
alert_mgr.clear_cooldowns()  # Allow same alert to fire again
```

## Key Points

✓ **Never crashes** — All errors wrapped in try/except
✓ **File-locked writes** — Safe concurrent access with fcntl
✓ **Graceful degradation** — Corrupted state files auto-reinitialize
✓ **Cooldown prevention** — No spam (60-120 min cooldowns by default)
✓ **Cross-session state** — Cooldowns persist in `.claude/alert_state.json`

## Alert Thresholds (Default)

| Alert | Threshold | Cooldown | Severity |
|-------|-----------|----------|----------|
| Drawdown | 5% | 60 min | CRITICAL |
| Consecutive Losses | 3 trades | 60 min | WARNING |
| Win Rate Drop | 10% | 120 min | WARNING |
| Weight Instability | 0.30 change | 60 min | WARNING |

## Troubleshooting

**No alerts firing?**
- Check cooldown: `print(alert_mgr._last_fired)`
- Verify threshold: `print(alert_mgr._configs["drawdown"]["threshold"])`
- Check log level is WARNING or higher

**State file corrupted?**
```bash
rm .claude/alert_state.json
# Will reinit on next scan
```

## Next Steps

1. Read **Full Documentation**: `docs/ALERT_SYSTEM.md`
2. Review **Alert Rules**: `.claude/rules/trading.md`
3. Monitor **Continuous Scanner**: `buddy_scanner scan --watch`
4. Check **Alert Log Weekly**: `tail trained_data/alerts.jsonl`

---

For detailed info, see `docs/ALERT_SYSTEM.md`.
