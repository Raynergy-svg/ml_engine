# Dynamic Pair Rotation System — Portfolio Optimizer Guide

## Overview

The `PortfolioOptimizer` implements a rolling Sharpe ratio-based pair rotation system that dynamically prioritizes FX pairs based on recent risk-adjusted performance. It identifies underperforming pairs and moves them to "observe-only" mode (scanned for learning but not traded) while promoting high-Sharpe pairs to active trading.

**Key features:**
- Rolling Sharpe ratio calculation across last N trades per pair
- Smart edge-case handling (new pairs, insufficient history, insufficient data)
- USD exposure clustering detection to prevent concentration
- Graceful degradation on corrupted/missing trade journal
- File locking (fcntl) for concurrent access safety
- ObservationLog integration for audit trail

## Architecture

### File Locations

```
src/scanner/automation/portfolio_optimizer.py    # Core optimizer class
src/scanner/automation/continuous.py             # Integration into scanner loop
trained_data/models/pair_rankings.json           # Persisted rankings (daily snapshots)
trained_data/observations.jsonl                  # Rotation decision audit trail
```

### Core Classes

#### `PortfolioOptimizer`

Main class responsible for pair ranking and rotation logic.

**Constructor parameters:**
```python
PortfolioOptimizer(
    trade_journal_path: str = "trained_data/trade_journal_rl.json",
    pair_rankings_path: str = "trained_data/models/pair_rankings.json",
    rolling_window: int = 50,          # Last N trades per pair
    max_active_pairs: int = 7,          # Top N by Sharpe to actively trade
    max_usd_exposure: float = 0.40,     # Max 40% of active pairs USD-based
    rotation_interval: int = 10,        # Check every 10 scan cycles
)
```

#### `PairRanking`

Data class representing a single pair's performance metrics.

```python
@dataclass
class PairRanking:
    pair: str                # Pair symbol (e.g., "EUR_USD")
    sharpe_ratio: float      # Rolling Sharpe (NaN → 0.0 if insufficient data)
    rolling_pnl: float       # Sum of last N trades P/L
    trade_count: int         # Total closed trades for this pair
    win_rate: float          # Wins / total trades
    status: str              # "active" or "observe"
    reason: str              # Why this status (e.g., "Sharpe 1.23" or "Insufficient history")
```

## How It Works

### 1. Rolling Sharpe Calculation

The system calculates Sharpe ratio for each pair using the last `rolling_window` closed trades:

```
Sharpe = (mean_return - risk_free_rate) / std_dev * sqrt(annualization_factor)
```

**Edge cases handled:**
- **Empty returns**: Returns 0.0
- **Single trade**: Insufficient data; returns 0.0
- **Zero volatility**: Returns 0.0 (can't divide by zero)
- **NaN results**: Coerced to 0.0

### 2. Pair Ranking Algorithm

For each pair in the trade journal:

1. Filter to closed trades only (outcome != None)
2. Extract last N trades (rolling_window)
3. Calculate returns list: [trade1.pnl, trade2.pnl, ...]
4. Calculate Sharpe ratio and win rate
5. Assign preliminary status:
   - **"observe"** if trade_count < 5 (insufficient history)
   - **"active"** if trade_count >= 5 (candidate for trading)
6. Sort by Sharpe ratio descending

### 3. Active Pair Selection

From ranked pairs:

1. Filter to "active" status candidates (5+ trades)
2. If fewer than `max_active_pairs` active, pad with "observe" pairs until reaching limit
3. Limit to top `max_active_pairs` pairs
4. Apply USD exposure check:
   - If USD pairs > `max_usd_exposure` × active count, demote lowest-Sharpe USD pair
5. Return final active list

### 4. USD Exposure Check

USD pairs identified:
```python
USD_PAIRS = {
    "EUR_USD", "GBP_USD", "USD_JPY",
    "AUD_USD", "NZD_USD", "USD_CAD", "USD_CHF"
}
```

**Rationale**: All 7 major pairs involve USD. If all 7 pairs are active, portfolio is heavily correlated. The check prevents this without hard-blocking (demotes lowest-Sharpe USD pair instead).

**Example:**
- Active pairs: [EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, USD_CHF, EUR_GBP]
- USD count: 7 / 8 = 87.5% (exceeds 40% limit)
- Action: Demote lowest-Sharpe USD pair (e.g., USD_CAD if Sharpe = 0.15)
- Result: USD count now 6 / 8 = 75% (still exceeds 40%, but optimization continues)

## Integration with ContinuousScanner

### Initialization

In `ContinuousScanner.__init__`:
```python
self._portfolio_optimizer = PortfolioOptimizer()
```

### Pair Rotation Check (Every 10 Scan Cycles)

In `ContinuousScanner.run()`:
```python
self._apply_pair_rotation(scan_pairs)  # Checks if time to rotate
```

### Trade Filtering

After scan completes, tradeable analyses are filtered to remove observe-only pairs:
```python
observe_pairs = self._portfolio_optimizer.get_observe_only_pairs(all_pairs=scan_pairs)
tradeable = [a for a in result.analyses if a.pair not in observe_pairs]
```

**Result**: Observe-only pairs are scanned (for learning and observation logging) but execution is skipped.

### Output

When rotation triggers:
```
📊 PAIR ROTATION (Cycle 10)
  🟢 Active (7): EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, EUR_GBP
  🟡 Observe-only (8): USD_CHF, EUR_JPY, GBP_JPY, AUD_JPY, EUR_AUD, GBP_AUD, EUR_CHF, GBP_CHF
```

## Observability & Logging

### Console Output

```
[cyan]📊 PAIR ROTATION (Cycle 10)[/cyan]
[green]Active (7): EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, EUR_GBP[/green]
[yellow]Observe-only (8): USD_CHF, EUR_JPY, GBP_JPY, AUD_JPY, EUR_AUD, GBP_AUD, EUR_CHF, GBP_CHF[/yellow]
```

### Observation Log

Rotation decisions are logged to `trained_data/observations.jsonl` with category `pair_rotation`:

```json
{
  "timestamp": "2026-03-19T12:30:45.123456+00:00",
  "pair": "PORTFOLIO",
  "category": "pair_rotation",
  "description": "Pair rotation at cycle 10: 7 active, 8 observe",
  "metadata": {
    "scan_cycle": 10,
    "active_pairs": ["EUR_USD", "GBP_USD", ...],
    "observe_pairs": ["USD_CHF", ...],
    "active_count": 7,
    "observe_count": 8
  }
}
```

### Pair Rankings File

Saved to `trained_data/models/pair_rankings.json` with timestamp:

```json
{
  "timestamp": "2026-03-19T12:30:45.123456+00:00",
  "rankings": [
    {
      "pair": "EUR_USD",
      "sharpe_ratio": 1.25,
      "rolling_pnl": 345.67,
      "trade_count": 18,
      "win_rate": 0.6111,
      "status": "active",
      "reason": "Sharpe 1.25"
    },
    {
      "pair": "GBP_JPY",
      "sharpe_ratio": null,
      "rolling_pnl": 0.0,
      "trade_count": 2,
      "win_rate": 0.0,
      "status": "observe",
      "reason": "Insufficient history (2/5 min)"
    }
  ]
}
```

## Configuration

### Default Settings (from `PortfolioOptimizer.__init__`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rolling_window` | 50 | Use last 50 closed trades per pair for Sharpe |
| `max_active_pairs` | 7 | Trade only top 7 pairs (rest observe-only) |
| `max_usd_exposure` | 0.40 | USD pairs ≤ 40% of active set |
| `rotation_interval` | 10 | Check rotation every 10 scan cycles (~50 min at 5-min intervals) |

### Recommended Adjustments

**For aggressive trading:**
```python
PortfolioOptimizer(
    max_active_pairs=10,        # Trade more pairs
    max_usd_exposure=0.50,      # Allow higher USD correlation
    rotation_interval=5,         # Rotate more frequently
)
```

**For conservative trading:**
```python
PortfolioOptimizer(
    rolling_window=100,         # Longer lookback period
    max_active_pairs=4,         # Trade fewer pairs
    max_usd_exposure=0.30,      # Stricter USD limit
    rotation_interval=20,       # Rotate less frequently
)
```

## API Reference

### Core Methods

#### `calculate_sharpe(returns, risk_free_rate=0.0, annualization_factor=252.0) → float`

Calculate Sharpe ratio from returns list.

```python
sharpe = optimizer.calculate_sharpe([10.0, -5.0, 15.0, 8.0])  # → 0.742
sharpe = optimizer.calculate_sharpe([])                       # → 0.0
sharpe = optimizer.calculate_sharpe([5.0, 5.0, 5.0])          # → 0.0 (zero std dev)
```

#### `rank_pairs(trade_journal=None) → list[PairRanking]`

Rank all pairs by rolling Sharpe ratio.

```python
rankings = optimizer.rank_pairs()  # Loads from file if trade_journal=None
for r in rankings:
    print(f"{r.pair}: Sharpe={r.sharpe_ratio:.2f}, status={r.status}")
```

#### `get_active_pairs(trade_journal=None) → list[str]`

Get list of pairs to actively trade.

```python
active = optimizer.get_active_pairs()
# → ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD", "USD_CAD", "EUR_GBP"]
```

#### `get_observe_only_pairs(trade_journal=None, all_pairs=None) → list[str]`

Get list of observe-only pairs (for learning, not trading).

```python
observe = optimizer.get_observe_only_pairs(all_pairs=DEFAULT_PAIRS)
# → ["USD_CHF", "EUR_JPY", "GBP_JPY", "AUD_JPY", ...]
```

#### `should_rotate(scan_cycle: int) → bool`

Check if it's time to re-evaluate pair rankings.

```python
if optimizer.should_rotate(10):  # rotation_interval=10
    # Run rotation
```

#### `save_rankings(rankings: list[PairRanking]) → None`

Persist rankings to JSON file (with fcntl file locking).

```python
rankings = optimizer.rank_pairs()
optimizer.save_rankings(rankings)
```

#### `load_rankings() → list[PairRanking]`

Load saved rankings from JSON file.

```python
rankings = optimizer.load_rankings()
```

#### `log_rotation_decision(scan_cycle, active_pairs, observe_pairs) → None`

Log rotation decision to ObservationLog.

```python
optimizer.log_rotation_decision(
    scan_cycle=10,
    active_pairs=["EUR_USD", "GBP_USD"],
    observe_pairs=["USD_CHF", "EUR_JPY"]
)
```

## Error Handling

All methods implement graceful degradation:

| Error | Behavior |
|-------|----------|
| Missing trade journal file | Returns empty list; logs debug message |
| Corrupted JSON | Catches JSONDecodeError; logs error; returns empty list |
| File lock timeout | Uses non-blocking read/write; graceful fallback |
| Invalid trade data | Skips malformed entries; continues processing |
| Sharpe calculation failure | Returns 0.0; logs debug message |
| ObservationLog unavailable | Logs debug message; doesn't crash scanner |

## Example Usage

### Standalone

```python
from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    rolling_window=50,
    max_active_pairs=7,
    rotation_interval=10
)

# Check if time to rotate
if optimizer.should_rotate(scan_cycle=50):
    # Rank all pairs
    rankings = optimizer.rank_pairs()

    # Get active pairs for trading
    active = optimizer.get_active_pairs()
    print(f"Active: {active}")

    # Get observe-only pairs
    observe = optimizer.get_observe_only_pairs()
    print(f"Observe: {observe}")

    # Save rankings
    optimizer.save_rankings(rankings)

    # Log decision
    optimizer.log_rotation_decision(50, active, observe)
```

### With ContinuousScanner

Integration happens automatically. In `continuous.py`:

1. `__init__` creates PortfolioOptimizer instance
2. `_apply_pair_rotation()` is called each scan cycle
3. Tradeable analyses are filtered before execution
4. Observe-only pairs are still scanned (for learning)

## Testing

Run a quick test to verify installation:

```bash
cd /path/to/ml_engine

# Test import
python3 -c "from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer; print('✓ Import OK')"

# Test instantiation
python3 << 'EOF'
from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer
optimizer = PortfolioOptimizer()
print(f"✓ Initialized: {optimizer}")
print(f"✓ Rolling window: {optimizer.rolling_window}")
print(f"✓ Max active pairs: {optimizer.max_active_pairs}")
print(f"✓ Rotation interval: {optimizer.rotation_interval}")
EOF

# Test with sample trade journal
python3 << 'EOF'
from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer
import json

optimizer = PortfolioOptimizer()

# Load real trade journal
rankings = optimizer.rank_pairs()
print(f"✓ Ranked {len(rankings)} pairs")

if rankings:
    active = optimizer.get_active_pairs()
    print(f"✓ Active pairs ({len(active)}): {active}")
EOF
```

## Future Enhancements

1. **Multi-timeframe correlation**: Rank pairs not just by solo Sharpe, but by correlation with other active pairs
2. **Regime-aware rotation**: Adjust `max_active_pairs` based on volatility regime (more pairs in LOW regime, fewer in EXTREME)
3. **Equity curve filtering**: Exclude pairs where equity curve is below 30-day MA
4. **Performance decay**: Penalize Sharpe for pairs with declining recent performance
5. **Watchlist promotion**: Promote observe-only pairs if they exceed active median Sharpe
6. **Pair clustering**: Group pairs by correlation; limit to N from each cluster

## Troubleshooting

**Q: Why is my pair stuck in "observe" status?**
A: Check trade count. Need ≥5 closed trades to be considered "active". Run a few scans on that pair.

**Q: Why was a high-Sharpe pair demoted?**
A: USD exposure check. If your active set is 7 USD pairs + 1 cross, the lowest-Sharpe USD pair gets demoted to balance exposure.

**Q: Pair rankings not updating?**
A: Check:
1. Trade journal has recent closed trades (outcome != None)
2. Rotation check is triggering (scan_cycle % rotation_interval == 0)
3. File permissions on `trained_data/models/pair_rankings.json`
4. No JSON lock contention (fcntl timeout)

**Q: Why aren't observe-only pairs being scanned?**
A: They are! They're in the scan result, but not executed. Check scan results display or `observations.jsonl` for evidence.

**Q: How do I manually force a rotation?**
A: Run `_apply_pair_rotation()` directly or set `rotation_interval=1` for testing.

## References

- **File locations**: See "Architecture" section
- **Trade journal format**: `src/scanner/execution.py`
- **Observation logging**: `src/scanner/automation/observation_log.py`
- **Continuous scanner**: `src/scanner/automation/continuous.py`
