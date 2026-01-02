# Wednesday Workflow - Mid-Week Check

## Overview
Wednesday is your **quick check day**. Verify model performance and optionally retrain with more recent data.

---

## Morning Routine (Before Market Open)

### 1. Check Model Status
```bash
buddy status
```
**What to look for:**
- Is the model still performing well?
- Any drift in accuracy?
- When was it last trained?

### 2. Quick Scan of Major Pairs
```bash
buddy scan --top 5
```
**What to look for:**
- Are opportunities still aligned with Monday's analysis?
- Have market conditions changed?
- New pairs showing up in top 5?

### 3. Test Model on Recent Data
```bash
buddy test --candles 50 --min-confidence 0.6
```
**What to look for:**
- Is accuracy holding up?
- Win rate still positive?
- Any degradation in performance?

---

## Optional: Quick Retrain (If Needed)

### When to Retrain Mid-Week
- Market regime has clearly changed (e.g., volatility spike)
- Model accuracy dropped significantly
- Major economic event occurred (e.g., Fed meeting, NFP)

### Quick Retrain Command
```bash
buddy train --oanda-live --candles 10000 --granularity H1
```
**Why 10,000 candles?**
- Faster training (5-10 minutes)
- Focuses on more recent data
- Good for mid-week adjustments

**After retraining:**
```bash
buddy test --candles 50 --min-confidence 0.6
buddy promote  # Only if significantly better
```

---

## Market Analysis

### 4. Review Your Watchlist
Check the pairs you identified on Monday:

```bash
# Check your primary pair
buddy predict -i [PRIMARY_PAIR] -g H1 -v

# Check your secondary pair
buddy predict -i [SECONDARY_PAIR] -g H1 -v
```

**What to look for:**
- Are signals still valid?
- Has confidence changed?
- Any new market regime indicators?

### 5. Scan for New Opportunities
```bash
buddy scan --pairs EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD
```
**Compare with Monday:**
- Same pairs in top 3? (market is stable)
- Different pairs? (market regime shift)
- Confidence levels changed?

---

## Performance Review

### 6. Check Feature Importance (Optional)
```bash
buddy analyze --top 20
```
**Compare with Monday:**
- Are the same features still important?
- New features rising? (market adapting)
- Volatility features changing? (regime shift)

### 7. Review Trading Performance
If you've been trading this week:
- Check your trade log
- Compare actual results vs. model predictions
- Note any patterns (e.g., model struggles in certain conditions)

---

## Quick Reference

```bash
# Full Wednesday routine
buddy status
buddy scan --top 5
buddy test --candles 50 --min-confidence 0.6
buddy predict -i [PRIMARY_PAIR] -g H1 -v
buddy scan --pairs EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD

# Optional: Quick retrain if needed
buddy train --oanda-live --candles 10000 --granularity H1
buddy test --candles 50 --min-confidence 0.6
buddy promote  # Only if better
```

---

## Decision Points

### Should I Retrain?
**Yes, if:**
- [ ] Model accuracy dropped > 5% from Monday
- [ ] Market volatility changed significantly
- [ ] Major economic event occurred
- [ ] Feature importance shifted dramatically

**No, if:**
- [ ] Model still performing well
- [ ] Market conditions stable
- [ ] No major events
- [ ] Monday's model still accurate

### Should I Change Watchlist?
**Yes, if:**
- [ ] New pairs showing higher confidence
- [ ] Original pairs no longer in top 5
- [ ] Market regime clearly shifted

**No, if:**
- [ ] Same pairs still top opportunities
- [ ] Confidence levels stable
- [ ] Market conditions unchanged

---

## Troubleshooting

### Model accuracy dropped
- Quick retrain might help: `buddy train --oanda-live --candles 10000`
- Check if it's temporary: `buddy test --candles 100`
- Might be market transition - wait for Friday retrain

### No clear opportunities
- Markets might be ranging
- Check different timeframes: `buddy scan -g H4`
- Wait for volatility to increase

### Feature importance changed
- Market regime might be shifting
- Consider quick retrain
- Monitor closely for Friday full retrain

---

## Notes Section

**Date:** _______________

**Model Status:**
- Current accuracy: _____%
- Changed from Monday: + / - _____%
- Retrained: Yes / No

**Market Conditions:**
- Volatility: High / Medium / Low
- Trend: Strong / Weak / Ranging
- Regime: Changed / Stable

**Top Opportunities:**
1. Pair: _____ | Signal: _____ | Confidence: _____%
2. Pair: _____ | Signal: _____ | Confidence: _____%
3. Pair: _____ | Signal: _____ | Confidence: _____%

**Actions Taken:**
- [ ] Quick retrain
- [ ] Model promoted
- [ ] Watchlist updated
- [ ] Trading rules adjusted

**Observations:**
_________________________________________________
_________________________________________________
_________________________________________________

