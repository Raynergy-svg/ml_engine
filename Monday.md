# Monday Workflow - Start of Week

## Overview
Monday is your **full retrain day**. Start the week fresh with a model trained on the latest data.

---

## Morning Routine (Before Market Open)

### 1. Check Current Model Status
```bash
buddy status
```
**What to look for:**
- Current production model accuracy
- Candidate model (if exists) accuracy
- Model age (when was it last trained?)

### 2. Retrain Model with Fresh Data
```bash
buddy train --oanda-live --candles 15000 --granularity H1
```
**Why 15,000 candles?**
- ~2-3 years of H1 data
- Recent enough to capture current market regime
- Not too old to include outdated patterns

**Expected time:** 5-15 minutes depending on your M1 chip performance.

**What to watch:**
- Validation direction accuracy should be > 50%
- Loss should decrease smoothly
- Early stopping will trigger if model isn't improving

### 3. Analyze Feature Importance
```bash
buddy analyze --top 30
```
**What to look for:**
- Which features are most important this week?
- Has the feature ranking changed from last week?
- Are volatility features rising? (might indicate regime change)

**Save plots for later comparison:**
```bash
buddy analyze --top 30 --save
```
Plots saved to: `trained_data/visualizations/shap_importance.png`

### 4. Test New Model on Recent Data
```bash
buddy test --candles 100 --min-confidence 0.6
```
**What to look for:**
- Direction accuracy on recent 100 candles
- Win rate with confidence filter
- Expected value per trade

**Good signs:**
- Accuracy > 52% (better than random)
- Win rate > 50% with confidence filter
- Positive expected value

### 5. Promote Candidate Model (If Better)
```bash
buddy promote
```
**Only promote if:**
- Candidate model accuracy > production model
- Test results show improvement
- You're confident in the new model

---

## After Training - Market Analysis

### 6. Scan All Major Pairs
```bash
buddy scan
```
**What to look for:**
- Which pairs show the strongest signals?
- Are there clear LONG or SHORT opportunities?
- What's the confidence level?

**Top opportunities this week:**
- Note the top 3 pairs
- Check their trend strength and volatility
- These are your watchlist for the week

### 7. Detailed Analysis of Top Pairs
```bash
# Replace PAIR with your top opportunity
buddy predict -i GBP_USD -g H1 -v
```
**What to look for:**
- Signal strength
- Market regime (trending vs ranging)
- Tier-1 and Tier-2 confidence
- Expected value

---

## Weekly Planning

### Create Your Watchlist
Based on scan results, create a watchlist:

1. **Primary pair:** Highest confidence opportunity
2. **Secondary pair:** Second best opportunity
3. **Backup pair:** Third option if others don't trigger

### Set Your Trading Rules for the Week
- Minimum confidence threshold: `0.60` (60%)
- Risk per trade: Check your config (`buddy status` shows current settings)
- Timeframes: H1 recommended for clearer signals

---

## Quick Reference

```bash
# Full Monday routine (run in order)
buddy status
buddy train --oanda-live --candles 15000 --granularity H1
buddy analyze --top 30 --save
buddy test --candles 100 --min-confidence 0.6
buddy promote  # Only if candidate is better
buddy scan
buddy predict -i [TOP_PAIR] -g H1 -v
```

---

## Troubleshooting

### Model training fails
- Check OANDA credentials: `echo $OANDA_API_TOKEN`
- Try fewer candles: `buddy train --oanda-live --candles 10000`
- Check disk space: `df -h`

### Low validation accuracy (< 50%)
- Market might be in transition
- Try different timeframe: `--granularity H4`
- Check if features are working: `buddy analyze`

### No good opportunities in scan
- Markets might be ranging (low volatility)
- Wait for higher volatility periods
- Check different pairs: `buddy scan --pairs EUR_USD,GBP_USD`

---

## Notes Section

**Week of:** _______________

**Model Performance:**
- Production accuracy: _____%
- Candidate accuracy: _____%
- Promoted: Yes / No

**Top Features This Week:**
1. ________________
2. ________________
3. ________________

**Top Opportunities:**
1. Pair: _____ | Signal: _____ | Confidence: _____%
2. Pair: _____ | Signal: _____ | Confidence: _____%
3. Pair: _____ | Signal: _____ | Confidence: _____%

**Weekly Goals:**
- [ ] Retrain model
- [ ] Analyze features
- [ ] Test model
- [ ] Promote if better
- [ ] Identify top 3 opportunities

