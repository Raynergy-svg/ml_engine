# Runbook: OANDA Tick Capture — This Week

## Prerequisites
```bash
export OANDA_API_TOKEN="your-token-here"
export OANDA_ACCOUNT_ID="001-001-..."
export OANDA_ENVIRONMENT="practice"   # or "live" if funded
```

## Step 1: Verify Connectivity
```bash
python scripts/verify_oanda_env.py
```
Expected: ✅ Connected. Balance: ... USD

## Step 2: Backfill S5 Historical Candles (finest granularity available)
```bash
# Single pair
python -m src.data.harvest --pairs EUR_USD --granularity S5 --backfill-all

# All 15 majors — run overnight
python -m src.data.harvest --pairs EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD,EUR_GBP,EUR_JPY,GBP_JPY,EUR_CHF,GBP_CHF,AUD_JPY,EUR_AUD,GBP_AUD --granularity S5 --backfill-all
```

## Step 3: Start Live Tick Capture (tmux/screen)
```bash
# Core 3 pairs (lightweight, ~1 tick/sec each during London/NY)
python scripts/run_tick_capture.py --pairs EUR_USD,GBP_USD,USD_JPY --buffer 5000

# All majors (heavier CPU/network, ~5-10 ticks/sec total)
python scripts/run_tick_capture.py --pairs ALL_FX --buffer 10000
```

Leave this running in a detached tmux session:
```bash
tmux new-session -d -s tickcap "python scripts/run_tick_capture.py --pairs ALL_FX"
```

## Step 4: Verify Ticks Landing
```bash
# After 60 seconds, check parquet output
ls -la trained_data/ticks/EUR_USD/$(date +%Y)/$(date +%m)/$(date +%d).parquet
```

## Step 5: Aggregate to M15 for Model Training
```bash
python -c "
from src.data.tick_aggregate import aggregate_ticks_to_candles
df = aggregate_ticks_to_candles('EUR_USD', 'M15', start='$(date +%Y-%m-%d)')
print(f'M15 bars today: {len(df)}')
print(df.tail())
"
```

## Expected Data Growth
| Granularity | Rate | Daily Volume | Annual |
|-------------|------|--------------|--------|
| Raw ticks | ~5–10/sec | ~500K ticks/pair | ~180M ticks |
| S5 candles | 1 bar/5s | ~17K bars/pair | ~6M bars |
| M15 candles | 1 bar/15min | ~96 bars/pair | ~35K bars |

Disk: raw ticks ~50 MB/day for 15 pairs (parquet+zstd). Manageable.

## Monitoring
Watch `logs/autonomous_trainer.jsonl` for any tick-capture alerts.
