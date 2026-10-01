# Corrected Axiom Development Replay — 2026-10-01

## Result

A new preregistered development experiment was run after the adversarial research repairs. It did not reuse the old 36-equity result as evidence and did not access the genuine final holdout.

Experiment: phase1-price-only-corrected-replay-20261001  
Code checkpoint: e9d229c1a2abcc26ec7e42c570c295bb7462ef1a  
Universe: 11 securities continuously present in the local PIT membership archive from 2022-01-03 through 2024-06-28  
Observations: 625 price dates; 6,875 feature rows; 6,765 complete feature/label rows  
Walk-forward folds: 5  
Registered LightGBM configurations: 4  
Portfolio: top 5, equal weight, five-session holding/rebalance, no overlapping cohorts  
Feature families: price_return only

Metrics:

- mean per-session cross-sectional rank IC: +0.0028677098
- after-cost mean portfolio return per evaluated rebalance: +0.07806002%
- aligned momentum baseline net return: +0.00245957%
- average one-way turnover: 0.54443026
- maximum drawdown including initial wealth: -8.95338%
- positive-IC folds: 2/5 = 40%
The preregistered development gate remained unchanged: rank IC >= 0.01, net return >= 0, max drawdown >= -15%, and positive-IC fold fraction >= 60%. The replay failed the rank-IC and fold-stability requirements. No candidate was frozen.

## Evidence limitations

The local price archive does not contain authoritative source-availability timestamps. For this replay, each daily price was conservatively assigned next-day availability solely to exercise the corrected row-time pipeline. Availability provenance therefore remains UNVERIFIED and this run cannot establish final point-in-time data evidence.

The stored archive also lacks historical volume and sector reference feeds required to reproduce all four original feature families without inventing inputs. The corrected feature factory now requires only inputs for selected families, so this replay preregistered price_return only rather than fabricating volume or sector history. That makes it a new development experiment, not a direct corrected rerun of the old four-family campaign.

The benchmark is an equal-weight daily-return index of the same 11 continuously observed members. Membership selection used only securities present on every archived PIT membership date in the window, avoiding the earlier whole-window pre-membership leakage.

## Consequence

The previous 36-equity +0.02637 IC / +0.54765% development result remains historical exploratory evidence, not validated evidence for the repaired kernel. The corrected price-only experiment does not pass the fixed development gate, so it cannot be frozen or advanced to the genuine final holdout.

No threshold was weakened and no additional model family was searched. The genuine final holdout remains untouched.

Raw local replay evidence is retained at ~/.local/share/axiom/corrected-price-replay-20261001/result.json.

This outcome blocks claims of a validated current edge. It does not block implementation/testing of non-capital execution infrastructure with synthetic evidence, but capital eligibility remains closed.

## Scientific qualification added during checkpoint resumption

The continuous-member restriction above removes the particular pre-membership use it targeted, but it does not establish a causally selected broad equity universe. Conditioning the research panel on staying present through the entire window uses future membership information. A synthetic perturbation confirmed that the whole-window intersection changes when a later removal changes, while the existing per-date PIT selector correctly leaves earlier membership unchanged. This is a limitation of the retrospective sample selection, not a regression in the repaired selector; its effect on the reported return or IC has not been quantified.

The assigned next-day availability remains an assumption, not source certification. The next empirical orchestrator must additionally bind decision cutoff to executable entry and actual label-end times. A synthetic composition of current feature/label helpers can otherwise pair a label's starting price with a later feature decision. That probe did not reconstruct or establish the historical replay's actual entry-time convention.

Subtracting one common benchmark return from all securities on a date cannot change that date's cross-sectional ranks. The old-to-new IC change therefore cannot be attributed to common-benchmark subtraction alone; this was not a one-variable experiment. No original metric, gate outcome or evidence artifact was changed by this qualification.

See `NEXT_RESEARCH_EXPERIMENT_2026-10-01.md` for the capability comparison, synthetic diagnostics and proposed source-qualified fixed-family ablation. Its empirical campaign is NOT REGISTERED OR RUN and remains blocked on source/timing admission.
