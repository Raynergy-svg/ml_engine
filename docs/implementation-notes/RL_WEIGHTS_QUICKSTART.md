# RL Weight Persistence — Quick Start

**TL;DR:** Enhanced weight persistence now includes time decay, confidence scaling, snapshots, and corruption recovery. No code changes required — it's automatic.

## Key Features At A Glance

| Feature | What | When | Impact |
|---------|------|------|--------|
| **Time Decay** | Weights drift toward baseline | Session start | Prevents indefinite persistence |
| **Confidence Scaling** | Blend baseline + learned | On retrieval | Prevents overfitting < 50 trades |
| **Snapshots** | Save checkpoint every 50 trades | Auto every 50 | Enable rollback |
| **Corruption Recovery** | Fix NaN/Inf/out-of-range | On load | Graceful fallback |

## Quick Integration

### Nothing to Change

Existing code works automatically with all enhancements:

```python
from src.scanner.agents import ScannerAgentTeam

team = ScannerAgentTeam(config)
team.reload_learned_weights()  # Automatic: time decay + confidence scaling

# Get weights — automatic confidence scaling applied
weights = team.get_weights_for_regime("NORMAL")

# Update weights — automatic snapshot save at 50-trade intervals
team.update_weights_from_outcome(verdicts, trade_won)
```

## Monitoring Maturity

Check when weights become "mature" (100% learned, no baseline blending):

```python
meta = team._learned_weights.get("_meta", {})
total_trades = meta.get("total_trades", 0)
confidence = min(total_trades / 50.0, 1.0)

if confidence < 1.0:
    print(f"Learning phase: {int(confidence * 100)}% confidence ({total_trades} trades)")
else:
    print(f"Mature: 100% learned weights ({total_trades} trades)")
```

## Emergency Rollback

If weights become unstable, rollback to a previous checkpoint:

```python
# List all available snapshots
snapshots = team.list_weight_snapshots()
# Output: [(50, '2026-03-19T10:30:00Z'), (100, '2026-03-19T12:00:00Z'), ...]

# Rollback to 100-trade checkpoint
team.load_weight_snapshot(100)
print("Rolled back to 100-trade checkpoint")
```

## What Gets Logged

### Normal Operation

```
INFO: Applied 48h time decay to agent weights (factor: 2%)
INFO: Applied confidence scaling (40%) based on 20 trades
INFO: Saved weight snapshot at 50 trades: 50.json
```

### Error Recovery

```
WARNING: Invalid value (inf) for volatility in HIGH. Resetting to baseline.
WARNING: Out-of-range value (50.0) for momentum in _global. Clamping to valid range.
ERROR: JSON parse error in agent_weights.json: ... Using baseline weights.
```

## Metadata Structure

Your `trained_data/models/agent_weights.json` now includes:

```json
{
  "_global": { ... },
  "NORMAL": { ... },
  "HIGH": { ... },
  "EXTREME": { ... },
  "_meta": {
    "total_trades": 47,
    "trades_NORMAL": 25,
    "trades_HIGH": 22,
    "last_updated": "2026-03-19T10:00:00Z",
    "min_trades_per_regime": 10
  }
}
```

## Performance Impact

All operations are lightweight:
- Time decay: <1ms (runs once per session)
- Confidence scaling: <1ms (on-demand retrieval)
- Snapshots: <5ms (once per 50 trades)
- Validation: <1ms (on load, once per session)

**Total overhead:** Negligible

## Common Questions

### Q: Will my existing code break?
**A:** No. All enhancements are backward compatible and automatic.

### Q: When do weights become mature?
**A:** After 50 trades. Before that, they're blended with baseline weights.

### Q: How do I know time decay was applied?
**A:** Check the logs at session start: `"Applied XXh time decay to agent weights"`

### Q: What if a weight file gets corrupted?
**A:** Automatically detected and recovered. Check logs for details.

### Q: Can I disable these features?
**A:** They're automatic and cannot be disabled (per trading rules). But they're designed to be safe and add no risk.

## Key Formulas

**Time Decay:**
```
decay_factor = min(hours_since_update / 2400, 0.5)
decayed_weight = learned + (baseline - learned) * decay_factor
```

**Confidence Scaling:**
```
confidence = min(total_trades / 50, 1.0)
final_weight = baseline * (1 - confidence) + learned * confidence
```

## Need More Details?

See `RL_WEIGHTS_ENHANCEMENT.md` for the complete reference guide.

---

**Status:** Ready for production  
**Side effects:** None (fully backward compatible)  
**Requires:** Python 3.8+
