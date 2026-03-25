# Portfolio Optimizer — Quick Start

## 30-Second Overview

The `PortfolioOptimizer` dynamically rotates FX pairs based on rolling Sharpe ratio:

1. **Ranks** all pairs by 50-trade rolling Sharpe ratio
2. **Selects** top 7 pairs for active trading
3. **Demotes** underperformers to observe-only (scanned, not traded)
4. **Checks** USD exposure to prevent over-concentration
5. **Logs** decisions to ObservationLog

**Active** pairs are traded when signal conditions are met.
**Observe-only** pairs are scanned for learning but execution is skipped.

## Usage

### Automatic (Recommended)

Enabled by default in `ContinuousScanner`:

```python
# src/scanner/automation/continuous.py
scanner = ContinuousScanner(engine)
scanner.run(pairs=DEFAULT_PAIRS)  # Portfolio optimizer runs every 10 cycles
```

### Manual

```python
from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()

# Every rotation_interval cycles:
if optimizer.should_rotate(scan_cycle):
    active = optimizer.get_active_pairs()      # → Top 7 pairs
    observe = optimizer.get_observe_only_pairs()  # → Remaining pairs

    # Filter tradeable analyses
    tradeable = [a for a in analyses if a.pair not in observe]
```

## Configuration

```python
PortfolioOptimizer(
    rolling_window=50,          # Last 50 trades per pair
    max_active_pairs=7,         # Top N by Sharpe
    max_usd_exposure=0.40,      # Max 40% USD pairs
    rotation_interval=10,       # Check every 10 scan cycles
)
```

## Outputs

### Console
```
📊 PAIR ROTATION (Cycle 10)
🟢 Active (7): EUR_USD, GBP_USD, USD_JPY, AUD_USD, NZD_USD, USD_CAD, EUR_GBP
🟡 Observe-only (8): USD_CHF, EUR_JPY, GBP_JPY, AUD_JPY, ...
```

### Files
- `trained_data/models/pair_rankings.json` — Sharpe rankings snapshot
- `trained_data/observations.jsonl` — Rotation decision audit trail

## Example: 2-Trade Journal

```json
[
  {
    "pair": "EUR_USD",
    "direction": "LONG",
    "entry_price": 1.085,
    "outcome": {
      "realized_pl": 20.0,
      "trade_won": true
    }
  },
  {
    "pair": "EUR_USD",
    "direction": "SHORT",
    "entry_price": 1.090,
    "outcome": {
      "realized_pl": -15.0,
      "trade_won": false
    }
  }
]
```

**Ranking for EUR_USD:**
```
Trade count: 2
Rolling P/L: [20.0, -15.0]
Mean: 2.5
Std Dev: 17.5
Sharpe: 0.14 (annualized)
Win rate: 50%
Status: "observe" (2 < 5 min trades)
```

## Key Rules

1. **Need ≥5 closed trades** to be considered "active" ✓
2. **Sharpe = 0.0** if insufficient data or zero volatility ✓
3. **USD exposure check** demotes lowest-Sharpe USD pair if >40% ✓
4. **Observe-only pairs** still scanned (for learning) ✓
5. **File locking (fcntl)** prevents corruption in concurrent access ✓

## Troubleshooting

| Problem | Solution |
|---------|----------|
| All pairs observe-only | Most have <5 trades. Run more scans to build history. |
| Pairs not rotating | Check rotation_interval value. Must be > 0. |
| High Sharpe pair demoted | USD exposure triggered. Another USD pair has higher Sharpe. |
| Rankings file not updating | Check file permissions. Verify trade journal has closed trades. |

## See Also

- **Full Guide**: `docs/PORTFOLIO_OPTIMIZER_GUIDE.md`
- **Implementation**: `src/scanner/automation/portfolio_optimizer.py`
- **Integration**: `src/scanner/automation/continuous.py`
