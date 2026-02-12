# Buddy Inference Command - Output Examples

## Example 1: Dry Run with Gates Passed

```
═══════════════════════════════════════════════════════════════
               ACCOUNT STATUS
═══════════════════════════════════════════════════════════════
💰 Live Balance: $10,543.21
📊 Trades Today: 3/30
🤖 RL Position Sizer: ENABLED
🧠 Intelligent Mode: ENABLED

════════════════════════════════════════════════════════════
  MODULAR ENSEMBLE INFERENCE
════════════════════════════════════════════════════════════
📊 Loading EUR_USD-specific models

Gate Checks:
  📅 Calendar: NFP in 45m ✓
  📰 Sentiment: bullish (+0.45) [5 headlines] ✓
  🔄 Online Learning: 23/50 trades buffered
  📐 Calibration: platt (0.720 → 0.680) ✓
  
  TCN: LONG (raw=0.72, calibrated=0.68)
  Ridge: 78/100 ✓
  XGBoost: momentum=0.35, accel=true ✓
  RF: drawdown=8.5pips, streak=0.65 ✓
  Meta: confidence=0.72 (threshold=0.55) ✓

→ TRADE: LONG, size=0.15 lots

╔═══════════════════════════════════════════════════════════╗
║              DRY RUN MODE (--execute not set)             ║
╚═══════════════════════════════════════════════════════════╝
✓ All gates passed - Trade signal generated
  Direction: LONG
  Position Size: 0.15 lots (15,000 units)
  To execute this trade, add --execute flag
════════════════════════════════════════════════════════════
```

## Example 2: Trade Executed Successfully

```
═══════════════════════════════════════════════════════════════
               ACCOUNT STATUS
═══════════════════════════════════════════════════════════════
💰 Live Balance: $10,543.21
📊 Trades Today: 3/30
🤖 RL Position Sizer: ENABLED
🧠 Intelligent Mode: DISABLED

════════════════════════════════════════════════════════════
  MODULAR ENSEMBLE INFERENCE
════════════════════════════════════════════════════════════
📊 Loading generic fallback models (no EUR_USD-specific models found)

Gate Checks:
  📅 Calendar: No events ✓
  📰 Sentiment: neutral (+0.02) [3 headlines] ✓
  🔄 Online Learning: 15/50 trades buffered
  📐 Calibration: platt (0.650 → 0.620) ✓
  
  TCN: SHORT (raw=0.65, calibrated=0.62)
  Ridge: 82/100 ✓
  XGBoost: momentum=0.28, accel=true ✓
  RF: drawdown=6.2pips, streak=0.71 ✓
  Meta: confidence=0.68 (threshold=0.55) ✓

→ TRADE: SHORT, size=0.12 lots

✓ High probability (65.0%) - TP extended to 3x ATR
Executing: SHORT 12,000 units @ 1.12345
  TP: 1.12045 (+30.0 pips = 3.0x ATR)
  SL: 1.12495 (-15.0 pips = 1.5x ATR)
  TS: 0.00100 (10.0 pips trailing)
  ATR: 0.00100 (10.0 pips) | R:R = 1:2.00
✓ Filled @ 1.12343 (Trade #54321)
  Slippage: +0.2 pips (favorable)
  ✓ Trade logged to journal
════════════════════════════════════════════════════════════
```

## Example 3: Gates Failed

```
═══════════════════════════════════════════════════════════════
               ACCOUNT STATUS
═══════════════════════════════════════════════════════════════
💰 Live Balance: $10,543.21
📊 Trades Today: 5/30
🤖 RL Position Sizer: DISABLED
🧠 Intelligent Mode: DISABLED

════════════════════════════════════════════════════════════
  MODULAR ENSEMBLE INFERENCE
════════════════════════════════════════════════════════════
📊 Loading EUR_USD-specific models

Gate Checks:
  📅 Calendar: No events ✓
  📰 Sentiment: bearish (-0.32) [4 headlines] ✓
  🔄 Online Learning: 8/50 trades buffered
  📐 Calibration: Not fitted (using raw)
  
  TCN: LONG (prob=0.58)
  Ridge: 45/100 ✗
  XGBoost: momentum=0.15, accel=false ✗
  RF: drawdown=12.5pips, streak=0.30 ✓
  Meta: confidence=0.48 (threshold=0.55) ✗

→ NO TRADE: Ridge confidence too low

╔═══════════════════════════════════════════════════════════╗
║              NO TRADE: GATES FAILED                        ║
╚═══════════════════════════════════════════════════════════╝
Reason: Ridge confidence too low
  Failed gates: Ridge confidence (45/100), XGBoost momentum (0.15), Meta-labeler (0.48)
════════════════════════════════════════════════════════════
```

## Example 4: LLM Validation Failed

```
═══════════════════════════════════════════════════════════════
               ACCOUNT STATUS
═══════════════════════════════════════════════════════════════
💰 Live Balance: $10,543.21
📊 Trades Today: 7/30
🤖 RL Position Sizer: ENABLED
🧠 Intelligent Mode: ENABLED

════════════════════════════════════════════════════════════
  MODULAR ENSEMBLE INFERENCE
════════════════════════════════════════════════════════════
📊 Loading EUR_USD-specific models

Gate Checks:
  📅 Calendar: ECB Rate Decision in 15m ✓
  📰 Sentiment: bearish (-0.68) [8 headlines] ✓
  🔄 Online Learning: 32/50 trades buffered
  📐 Calibration: platt (0.720 → 0.680) ✓
  
  TCN: LONG (raw=0.72, calibrated=0.68)
  Ridge: 78/100 ✓
  XGBoost: momentum=0.35, accel=true ✓
  RF: drawdown=8.5pips, streak=0.65 ✓
  Meta: confidence=0.72 (threshold=0.55) ✓

→ TRADE: LONG, size=0.15 lots

╔═══════════════════════════════════════════════════════════╗
║              NO TRADE: LLM VALIDATION FAILED              ║
╚═══════════════════════════════════════════════════════════╝
Model gates passed but LLM identified risk factors
The intelligent mode override prevented this trade
════════════════════════════════════════════════════════════
```

## Example 5: Daily Trade Limit Reached (Early Exit)

```
⛔ DAILY TRADE LIMIT REACHED (30/30)
Come back tomorrow or increase max_trades_per_day
```

## Code Changes Summary

All output sections are properly implemented in `cli/commands.py`:

1. **Lines 598-628**: Account Status Header (before inference banner)
2. **Lines 633-635**: Purple separator and MODULAR ENSEMBLE INFERENCE banner
3. **Lines 655-658**: Model source confirmation (pair-specific vs generic)
4. **Lines 698-701**: Gate Checks section header and display
5. **Lines 706-708**: Final trading decision
6. **Lines 715-851**: Trade execution with TP/SL/TS, fill confirmation, journal logging
7. **Lines 854-861**: Dry-run mode status box
8. **Lines 863-869**: LLM validation failed status box
9. **Lines 872-890**: Gates failed status box with detailed failure list
10. **Line 895**: Closing purple separator
11. **Line 896**: Graceful termination (return)

All requirements from the problem statement are fully satisfied.
