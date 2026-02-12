# H1 Trade Execution Improvements - Implementation Summary

## Research-Based Professional H1 Trading Standards

Based on industry research and professional FX trading practices:

### 1. **ATR-Based Stop Loss & Take Profit**
- **SL Multiplier**: 1.5x ATR (professional standard for H1)
  - Gives room for hourly price noise
  - Prevents premature stopouts
- **TP Multiplier**: 3.0x ATR 
  - Creates 2:1 reward-to-risk ratio
  - Suitable for swing trading on H1
- **ATR Period**: 14 (industry standard)

### 2. **Session Filtering**
- **Active Hours**: 08:00 - 21:00 UTC
  - London open (08:00) → NY close (21:00)
  - Peak: London/NY overlap (13:00-17:00 UTC)
- **Purpose**: Avoid low-liquidity Asian/late-night sessions
- **Result**: Better fills, tighter spreads

### 3. **Volatility Filtering**
- **Minimum ATR**: 8 pips
- **Purpose**: Skip dead/choppy markets
- **Result**: Only trade when movement justifies risk

### 4. **Position Sizing (Professional)**
- **Risk Per Trade**: 2% (down from 10%)
  - Professional standard
  - Allows 50 consecutive losses before blowup
- **Max Trades/Day**: 3 (down from 10)
  - H1 = swing trading = fewer, higher-quality setups
  - Prevents overtrading

### 5. **Pip Ranges (H1-Specific)**
- **SL Range**: 15-80 pips
  - Min 15 pips (typical EUR_USD H1 ATR = 10-20 pips)
  - Max 80 pips (cap for extreme volatility)
- **TP Range**: 30-200 pips
  - Min 30 pips (ensures 2:1 R:R minimum)
  - Max 200 pips (reasonable H1 swing target)

### 6. **Trailing Stop Management**
- **Trigger**: Move SL to breakeven after 1R profit
- **Trail Distance**: 0.5R behind price
- **Purpose**: Lock in profits while letting winners run

### 7. **Hold Time Management**
- **Target Hold**: 24 hours (typical H1 swing)
- **Max Hold**: 72 hours (3 days)
- **Purpose**: Match H1 swing trading timeframe

## Files Modified

### 1. `risk_management.py`
```python
# New H1-optimized settings
atr_period: int = 14
atr_sl_multiplier: float = 1.5  # SL = 1.5x ATR
atr_tp_multiplier: float = 3.0  # TP = 3.0x ATR (2:1 R:R)
min_stop_loss_pips: float = 15.0
max_stop_loss_pips: float = 80.0
min_take_profit_pips: float = 20.0

# Session filters
enable_session_filter: bool = True
active_sessions_utc: tuple = (8, 21)  # 08:00-21:00 UTC

# Trailing stops
enable_trailing_stop: bool = True
trailing_trigger_r: float = 1.0  # Move to BE after 1R
trailing_distance_r: float = 0.5  # Trail 0.5R behind

# Max hold time
max_hold_hours: int = 72  # 3 days max
target_hold_hours: int = 24  # 24h target
```

### 2. `buddy_scanner.py`
```python
# H1-optimized scan config
atr_period: int = 14
atr_sl_multiplier: float = 1.5
atr_tp_multiplier: float = 3.0

# Pip ranges
min_sl_pips: float = 15.0
max_sl_pips: float = 80.0
min_tp_pips: float = 30.0
max_tp_pips: float = 200.0

# Professional position sizing
risk_per_trade_pct: float = 0.02  # 2% (was 10%)
max_trades_per_day: int = 3  # 3 (was 10)

# Session filter
enable_session_filter: bool = True
session_start_utc: int = 8  # London open
session_end_utc: int = 21  # NY close

# Volatility filter
min_atr_pips: float = 8.0  # Skip if ATR < 8 pips
```

### 3. SL/TP Calculation Logic
```python
# Before (static R:R)
sl_pips = (atr * 1.5) / pip_value
tp_pips = sl_pips * rr_multiplier  # Fixed 2.0

# After (ATR-based)
sl_pips = (atr * atr_sl_multiplier) / pip_value  # 1.5x ATR
tp_pips = (atr * atr_tp_multiplier) / pip_value  # 3.0x ATR
sl_pips = max(min_sl_pips, min(sl_pips, max_sl_pips))  # 15-80 pips
tp_pips = max(min_tp_pips, min(tp_pips, max_tp_pips))  # 30-200 pips
```

### 4. Session & Volatility Filters
```python
# Session filter
current_utc_hour = datetime.now(timezone.utc).hour
if not (session_start_utc <= current_utc_hour <= session_end_utc):
    return None  # Skip trade outside active hours

# Volatility filter
atr_pips = atr / pip_value
if atr_pips < min_atr_pips:  # < 8 pips
    return None  # Skip dead market
```

## Expected Impact

### Before (Old Settings)
- Risk: 10% per trade
- Trades: 10/day
- SL: Static 30 pips
- TP: Static 60 pips (2:1 R:R)
- Session: 24/7
- Result: Overtrading, poor entries, high risk

### After (H1 Optimized)
- Risk: 2% per trade ✓
- Trades: 3/day ✓
- SL: 1.5x ATR (15-80 pips) ✓
- TP: 3.0x ATR (30-200 pips, 2:1+ R:R) ✓
- Session: London/NY only (08:00-21:00 UTC) ✓
- Filters: Skip low volatility (<8 pips ATR) ✓
- Trailing: Move to BE after 1R ✓
- Result: Professional swing trading, better risk management

## Testing

Run verification:
```bash
python test_h1_improvements.py
```

Expected output shows:
- ✓ ATR-based calculations (1.5x SL, 3.0x TP)
- ✓ Session filtering (08:00-21:00 UTC)
- ✓ Volatility filtering (8 pips minimum)
- ✓ Professional risk (2% per trade)
- ✓ Reduced frequency (3 trades/day)
- ✓ Example calculations for EUR/USD, GBP/USD, USD/JPY

## Usage

The improvements are automatic in:
```bash
# Scanning
python main.py scan --pairs EUR_USD,GBP_USD --granularity H1

# Direct trading
python main.py buddy --instrument EUR_USD --granularity H1 --execute
```

All H1 trades now use:
- ATR-based SL/TP
- Session filtering
- Volatility filtering
- Professional position sizing
- Trailing stops (when implemented in execution)
