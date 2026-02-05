# Scheduled Retraining Setup Guide

## Overview

The ML Engine scanner now uses **joint-trained models exclusively**. Models are automatically retrained on **Mon/Wed/Fri at 10 AM UTC** during the London trading session.

## Quick Setup

### 1. Install the launchd job

```bash
# Copy the plist to LaunchAgents
cp scripts/com.mlengine.retrain.plist ~/Library/LaunchAgents/

# Load the job (starts scheduling immediately)
launchctl load ~/Library/LaunchAgents/com.mlengine.retrain.plist

# Verify it's loaded
launchctl list | grep mlengine
```

### 2. Configure email alerts (optional)

Edit `scripts/scheduled_retrain.py` and set these environment variables:

```bash
export RETRAIN_EMAIL_SENDER="your-email@domain.com"
export RETRAIN_SMTP_HOST="smtp.gmail.com"
export RETRAIN_SMTP_PORT="587"
```

Or add them to the plist `EnvironmentVariables` section.

### 3. Test the setup

```bash
# Dry run (no actual training)
python scripts/scheduled_retrain.py --dry-run

# Run manually once
python scripts/scheduled_retrain.py --pairs EUR_USD,GBP_USD,USD_JPY

# Check logs
tail -f logs/retrain_*.log
```

## Managing the Schedule

### Stop the scheduled job
```bash
launchctl unload ~/Library/LaunchAgents/com.mlengine.retrain.plist
```

### Restart the job
```bash
launchctl unload ~/Library/LaunchAgents/com.mlengine.retrain.plist
launchctl load ~/Library/LaunchAgents/com.mlengine.retrain.plist
```

### Run immediately (for testing)
```bash
launchctl start com.mlengine.retrain
```

### Check job status
```bash
launchctl list | grep mlengine
```

## Schedule Details

- **Days**: Monday, Wednesday, Friday
- **Time**: 10:00 AM UTC (London session)
- **Pairs**: All 15 scanner pairs by default
- **Models saved to**: `trained_data/models/joint/`

## Timezone Notes

The plist uses **local system time**. Adjust the `Hour` values in the plist based on your timezone:

| Your Timezone | 10 AM UTC = |
|--------------|-------------|
| UTC | 10:00 |
| CET (Winter) | 11:00 |
| CEST (Summer) | 12:00 |
| PST | 02:00 |
| PDT | 03:00 |
| EST | 05:00 |
| EDT | 06:00 |

## Email Alerts

- **Only sent on FAILURE** (no spam on success)
- Recipient: `dcertan84@gmail.com`
- Fallback: If email fails, alert written to `logs/ALERT_*.txt`

## Logs

- **Per-run logs**: `logs/retrain_YYYYMMDD_HHMMSS.log`
- **launchd stdout**: `logs/launchd_retrain_stdout.log`
- **launchd stderr**: `logs/launchd_retrain_stderr.log`

## Troubleshooting

### Job not running?
```bash
# Check if loaded
launchctl list | grep mlengine

# Check launchd error
cat ~/Library/Logs/com.mlengine.retrain.stderr.log

# Manual test
python scripts/scheduled_retrain.py --dry-run
```

### Models not updating?
```bash
# Check if joint models exist
ls -la trained_data/models/joint/

# Run training manually
python main.py train-joint --instruments EUR_USD,GBP_USD,USD_JPY
```

### Permission issues?
```bash
# Ensure script is executable
chmod +x scripts/scheduled_retrain.py

# Ensure logs directory exists
mkdir -p logs
```
