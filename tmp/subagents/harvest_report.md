# Data Harvest Backfill Report

## Summary
Attempted to run the ML Engine's automated data harvest backfill for all currency pairs. The dry-run failed due to a source code bug in `src/data/harvest.py` that incorrectly formats the instrument symbol when querying OANDA. Because the bug prevents any pair from being successfully harvested, the full backfill was **not executed** per instructions.

## What I Did
1. **Syntax validation**: Ran `python -m py_compile src/data/harvest.py` — passed (exit code 0).
2. **Dry-run**: Ran dry-run for `EUR_USD`:
   ```
   PYTHONPATH=/Users/buddy/Documents/ml_engine python -m src.data.harvest --dry-run --pairs EUR_USD
   ```
3. **Result**: Dry-run completed without crashing, but the pair harvest failed with an instrument-not-found error.

## Dry-Run Results

### Command
```bash
cd /Users/buddy/Documents/ml_engine
PYTHONPATH=/Users/buddy/Documents/ml_engine python -m src.data.harvest --dry-run --pairs EUR_USD
```

### Output
```
2026-06-15 17:42:44,552 INFO __main__ HarvestScheduler: connected to OANDA
2026-06-15 17:42:44,552 INFO __main__ Harvesting EUR_USD M15
2026-06-15 17:42:44,552 INFO src.brokers.registry Initialized default registry: 24 FX, 6 FUTURES
2026-06-15 17:42:44,552 WARNING __main__ Harvest failed for EUR_USD: 'Instrument not found: EUR/USD'
2026-06-15 17:42:44,663 INFO __main__ HarvestScheduler: disconnected
EUR_USD: {'status': 'error', 'error': "'Instrument not found: EUR/USD'"}
```

### Root Cause Analysis
The bug is in `src/data/harvest.py` at line 205:
```python
broker_symbol = pair.replace("_", "/")  # OANDA uses EUR/USD
```

This converts `EUR_USD` → `EUR/USD`. However:
- The instrument registry (`src/brokers/registry.py`) stores FX pairs with **underscore** format (e.g., `broker_symbol="EUR_USD"`).
- The OANDA v20 REST API also expects **underscore** format in the URL path (e.g., `/instruments/EUR_USD/candles`).
- After the incorrect conversion, `get_registry().get("EUR/USD")` fails with `KeyError`, which is raised *before* the fallback logic inside the try/except block, causing the pair to error out.

This means **all pairs** will fail identically because they all use the same `_` → `/` conversion.

## Per-Pair Harvest Statistics
No pairs were successfully harvested. The `trained_data/harvest/` directory is empty.

## Errors Encountered
- **Critical**: `Instrument not found: EUR/USD` — affects every FX pair due to incorrect symbol formatting in `src/data/harvest.py:205`.
- **Impact**: Full backfill blocked; zero parquet files created.

## Total Disk Usage
```
trained_data/harvest/  0B
```

## Recommended Fix
Change line 205 in `src/data/harvest.py` from:
```python
broker_symbol = pair.replace("_", "/")
```
to:
```python
broker_symbol = pair
```

Alternatively, move the `get_registry().get()` call inside the `try` block in `_fetch_chunk()` so that lookup failures gracefully fall back to the generic `Instrument.fx()` builder instead of raising an uncaught `KeyError`.

## Next Steps
Once the symbol-formatting bug is fixed:
1. Re-run the dry-run for `EUR_USD`.
2. If successful, proceed with the full backfill:
   ```bash
   PYTHONPATH=/Users/buddy/Documents/ml_engine python -m src.data.harvest --backfill-all --granularity M15
   ```
