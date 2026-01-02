# Friday Workflow - End of Week Review

## Overview
Friday is your **comprehensive review and retrain day**. End the week with a fully updated model and plan for next week.

---

## Morning Routine (Before Market Open)

### 1. Check Model Status
```bash
buddy status
```
**What to look for:**
- Current model performance
- Age of model (when trained?)
- Candidate model status

### 2. Full Retrain with Week's Data
```bash
buddy train --oanda-live --candles 20000 --granularity H1
```
**Why 20,000 candles?**
- Includes full week's data plus historical context
- More robust training
- Better generalization

**Expected time:** 10-20 minutes

**What to watch:**
- Validation accuracy trend
- Loss convergence
- Early stopping behavior

### 3. Comprehensive Feature Analysis
```bash
buddy analyze --top 20 --save
```
**What to analyze:**
- Which features dominated this week?
- How did they change from Monday?
- What does this tell you about market regime?

**Save for comparison:**
- Plots saved to: `trained_data/visualizations/shap_importance.png`
- Compare with Monday's analysis

### 4. Thorough Model Testing
```bash
buddy test --candles 200 --min-confidence 0.55
```
**What to evaluate:**
- Overall accuracy on 200 recent candles
- Win rate with confidence filter
- Expected value per trade
- Compare with Monday's test results

**Good metrics:**
- Accuracy > 52%
- Win rate > 50% with confidence filter
- Positive expected value
- Consistent performance

### 5. Promote Candidate Model (If Better)
```bash
buddy promote
```
**Promote if:**
- Candidate accuracy > production by > 2%
- Test results show clear improvement
- Feature importance makes sense
- You're confident in the model

---

## Weekly Performance Review

### 6. Scan All Major Pairs
```bash
buddy scan
```
**Compare with Monday:**
- Same pairs in top 3? (stable week)
- Different pairs? (regime shift)
- Confidence levels changed?

### 7. Analyze Top Opportunities
```bash
# Check top 3 pairs in detail
buddy predict -i [TOP_PAIR_1] -g H1 -v
buddy predict -i [TOP_PAIR_2] -g H1 -v
buddy predict -i [TOP_PAIR_3] -g H1 -v
```

**What to document:**
- Best opportunities for next week
- Confidence levels
- Market regime indicators
- Risk/reward setup

---

## Weekly Summary

### 8. Review Trading Performance
If you traded this week:
- Total trades: _____
- Win rate: _____%
- Average confidence: _____%
- Best performing pair: _____
- Worst performing pair: _____

### 9. Feature Importance Comparison
Compare Monday vs Friday:

**Monday's Top Features:**
1. ________________
2. ________________
3. ________________

**Friday's Top Features:**
1. ________________
2. ________________
3. ________________

**Changes observed:**
- [ ] Features stable (market regime stable)
- [ ] Features shifted (regime change)
- [ ] Volatility features rising/falling
- [ ] Momentum features changing

### 10. Market Regime Analysis
**This week's market characteristics:**
- [ ] High volatility / Low volatility
- [ ] Strong trends / Ranging markets
- [ ] Risk-on / Risk-off
- [ ] Dollar strength / Dollar weakness

**Regime changes detected:**
- [ ] Yes - Document below
- [ ] No - Market stable

---

## Plan for Next Week

### 11. Create Next Week's Watchlist
Based on Friday's scan:

**Primary Pair:** _______________
- Signal: _____
- Confidence: _____%
- Rationale: _______________

**Secondary Pair:** _______________
- Signal: _____
- Confidence: _____%
- Rationale: _______________

**Backup Pair:** _______________
- Signal: _____
- Confidence: _____%
- Rationale: _______________

### 12. Set Trading Parameters
**For next week:**
- Minimum confidence: _____%
- Risk per trade: _____%
- Preferred timeframe: _____
- Max trades per day: _____

### 13. Identify Key Events
**Upcoming economic events:**
- Monday: _______________
- Wednesday: _______________
- Friday: _______________

**How to prepare:**
- [ ] Retrain after major events
- [ ] Adjust confidence thresholds
- [ ] Monitor volatility changes

---

## Quick Reference

```bash
# Full Friday routine (run in order)
buddy status
buddy train --oanda-live --candles 20000 --granularity H1
buddy analyze --top 20 --save
buddy test --candles 200 --min-confidence 0.55
buddy promote  # Only if candidate is better
buddy scan
buddy predict -i [TOP_PAIR_1] -g H1 -v
buddy predict -i [TOP_PAIR_2] -g H1 -v
buddy predict -i [TOP_PAIR_3] -g H1 -v
```

---

## Weekly Metrics Tracking

### Model Performance
| Metric | Monday | Wednesday | Friday | Change |
|--------|--------|-----------|--------|--------|
| Production Accuracy | _____% | _____% | _____% | _____% |
| Candidate Accuracy | _____% | _____% | _____% | _____% |
| Test Accuracy (100 candles) | _____% | _____% | _____% | _____% |
| Win Rate (>60% conf) | _____% | _____% | _____% | _____% |

### Top Opportunities
| Day | Pair 1 | Pair 2 | Pair 3 |
|-----|--------|--------|--------|
| Monday | _____ | _____ | _____ |
| Wednesday | _____ | _____ | _____ |
| Friday | _____ | _____ | _____ |

### Feature Importance Evolution
**Top 5 Features:**
- Monday: 1. _____ 2. _____ 3. _____ 4. _____ 5. _____
- Friday: 1. _____ 2. _____ 3. _____ 4. _____ 5. _____

---

## Troubleshooting

### Model not improving
- Check data quality: `buddy test --candles 200`
- Try different timeframe: `--granularity H4`
- Review feature importance: `buddy analyze`
- Market might be too efficient (normal for FX)

### Low test accuracy
- FX markets are efficient - 52-55% is good
- Focus on confidence filtering
- Use tier-2 gating (already enabled in config)
- Consider ensemble model: `buddy train -m ensemble`

### Feature importance unstable
- Market regime is changing
- This is normal - retrain weekly
- Focus on consistent top features
- Remove very low importance features

---

## Notes Section

**Week of:** _______________

**Model Performance:**
- Production accuracy: _____%
- Candidate accuracy: _____%
- Promoted: Yes / No
- Test accuracy: _____%

**Top Features This Week:**
1. ________________
2. ________________
3. ________________
4. ________________
5. ________________

**Top Opportunities for Next Week:**
1. Pair: _____ | Signal: _____ | Confidence: _____%
2. Pair: _____ | Signal: _____ | Confidence: _____%
3. Pair: _____ | Signal: _____ | Confidence: _____%

**Market Regime:**
- Volatility: High / Medium / Low
- Trend: Strong / Weak / Ranging
- Dominant features: ________________

**Trading Performance (if applicable):**
- Total trades: _____
- Win rate: _____%
- Best pair: _____
- Lessons learned: _______________

**Next Week Goals:**
- [ ] Focus on top 3 pairs
- [ ] Monitor feature changes
- [ ] Adjust confidence thresholds if needed
- [ ] Retrain after major events

**Weekly Reflection:**
_________________________________________________
_________________________________________________
_________________________________________________

---

## Archive

**Model files location:**
- Production: `trained_data/models/buddy_tf.keras`
- Candidate: `trained_data/models/buddy_tf_candidate.keras`
- Visualizations: `trained_data/visualizations/`

**Backup recommendation:**
- Copy model files weekly
- Save feature importance plots
- Document weekly metrics

