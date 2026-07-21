# LLM Macro Agent — Historical Shadow Validation Report

**Generated:** 2026-07-08T09:57:20.904068
**Method:** Pre-event transcripts fed to LLM Macro Agent + CB Audio Processor (tone analysis)

## Summary

| Metric | Value |
|--------|-------|
| Events tested | 4 |
| Threshold passes | 0 / 4 |
| Avg regime_shift_probability | 0.400 |

## Results by Event

### July 2024 BoJ Hike (2024-07-31)

- **Pair:** USD_JPY
- **Direction bias:** SHORT
- **Regime shift probability:** 0.400
- **Threshold:** False
- **Confidence:** 0.500
- **Tone score:** -0.333
- **Key drivers:** degraded_mode_heuristic
- **Notes:** LLM unavailable; heuristic fallback used
- **Status:** ❌ FAIL

### September 2024 Fed Cut (2024-09-18)

- **Pair:** EUR_USD
- **Direction bias:** LONG
- **Regime shift probability:** 0.200
- **Threshold:** False
- **Confidence:** 0.500
- **Tone score:** -0.333
- **Key drivers:** degraded_mode_heuristic
- **Notes:** LLM unavailable; heuristic fallback used
- **Status:** ❌ FAIL

### March 2023 SVB Collapse (2023-03-10)

- **Pair:** USD_JPY
- **Direction bias:** SHORT
- **Regime shift probability:** 0.500
- **Threshold:** False
- **Confidence:** 0.500
- **Tone score:** 0.000
- **Key drivers:** degraded_mode_heuristic
- **Notes:** LLM unavailable; heuristic fallback used
- **Status:** ❌ FAIL

### February 2022 Russia-Ukraine Invasion (2022-02-24)

- **Pair:** USD_CHF
- **Direction bias:** LONG
- **Regime shift probability:** 0.500
- **Threshold:** False
- **Confidence:** 0.500
- **Tone score:** 0.000
- **Key drivers:** degraded_mode_heuristic
- **Notes:** LLM unavailable; heuristic fallback used
- **Status:** ❌ FAIL

## Conclusion

This report validates whether the LLM Macro Agent (in degraded or full mode) would have flagged regime shifts before known historical macro events. A passing score means the agent's regime_shift_probability exceeded the expected threshold for that event.

**Next steps:**
- If >=3/4 events pass: consider promoting LLM Macro Agent from shadow to active voting.
- If <2/4 events pass: investigate why the agent misses regime shifts (macro context gaps, threshold tuning).
