# FX Edge Search — Final Verdict (2026-06-18)

**Question the operator posed:** "Why is my system the only one on the internet stuck at 52%?
Get me out of the 50% coin-flip."

**Answer (calibrated HIGH):** The system is not broken. ~52% directional accuracy on liquid FX
majors is the genuine efficient-market wall, confirmed three independent ways below. The systems
claiming 60–95% are search/leakage artifacts; the players who actually profit do not win on
directional accuracy. There is no forward-deployable directional edge at this data/feature scale.
Staying halted is the correct, evidenced outcome — a closed question with evidence, per the PRD.

## 1. Every lever tested (this session + prior)

| Lever | Verdict | Evidence |
|---|---|---|
| Price-only direction M15/H1/H4 | dead | ~52%, 4 confirmations |
| Daily direction (quick) | ~54% | prior |
| **Daily 1-day-ahead, rigorous 22-yr walk-forward** | **dead: 50.3% balanced** | this session, `daily_direction_oos_*.json` |
| News/macro fusion | dead | no lift, gap worsened |
| **Meta-labeling reframe (3 majors)** | **dead: meta-AUC ≤0.53** | this session, `phase_a_meta_*.json` |
| Cross-sectional carry+trend 2014–2026 | dead | gross Sharpe ≈0 |
| **Pre-2014 factor window** | **real then, dead now** | this session: trend +0.46 (99–14) → −0.55 (20–26); carry decayed to +0.09 |
| EM / global carry | fails gate | prior: net Sharpe ~0.50 but 40% DD (carry crash) |
| SOTA capacity scaling | rejected | signal-bound, not capacity-bound |

## 2. The decisive own-data test (refutes the "58% daily" literature)

Built `scripts/experiment_daily_direction_oos.py`: HistGradientBoosting, causal daily features,
walk-forward by calendar year (train on years < Y, predict Y), 7 majors pooled, 2005–2026.

- Mean raw acc **0.5027**, mean **balanced acc 0.5029**, median balanced **0.5024**.
- Only 3/22 years beat 52% balanced; best (2012) = 53.2%, isolated, no persistence.
- Up-day rate ~0.50 every year → raw ≈ balanced → no imbalance to inflate → true coin flip.

This explains the credible-looking counter-claims rather than contradicting them:
- [arXiv 2409.04471](https://arxiv.org/abs/2409.04471): 58.52% one-day EUR/USD — but headline return
  is **2022 alone** (a strong USD-trend year). My 2012 hits 53% in isolation too; the 22-yr mean is 50.3%.
- [MDPI BFSA, Mar 2026](https://www.mdpi.com/3042-5042/3/1/6): 55–60% win rate at H=1, **negative by
  H=2/3**, authors concede it is "exaggerated by feature selection, causing structural directional
  imbalance" — i.e. the annealing feature search overfits a window (the verified 5.12× backtest-
  inflation effect, [arXiv 2604.15531](https://arxiv.org/html/2604.15531v1)).

## 3. What the literature actually says (deep-research, 24 primary sources)

- **Consensus ceiling:** short-horizon nominal FX for majors ≈ random walk; predictability only at
  2–3 yr horizons ([ECB WP 088](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp088.pdf),
  [J. Int. Econ.](https://www.sciencedirect.com/science/article/abs/pii/S0022199602000600),
  Meese–Rogoff 1983). The one "FX is predictable" result is long-horizon + time-varying-parameter
  ([arXiv 1403.0627](https://arxiv.org/pdf/1403.0627), verified) — episodic, not intraday skill.
- **High-accuracy claims are artifacts (verified):** spec-search produces significant backtests on
  pure noise; K=1000 configs inflate in-sample 5.12×; flaws = full-sample normalization leakage,
  random CV, overlapping windows, tune+test on same data ([arXiv 2604.15531](https://arxiv.org/html/2604.15531v1)).
  Backtest overfitting is mathematically inevitable under multiple trials (max Sharpe ≈ √(2 ln I))
  ([Bailey/López de Prado SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253),
  [GARP](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf)).
- **Order flow — the real lever you can't access:** *interdealer* order flow predicts daily FX OOS
  at Sharpe ~1.52 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022199609000464),
  [Evans–Lyons](https://faculty.georgetown.edu/evansm1/wpapers_files/orderflow.pdf)), but the
  *commercially/retail-available* version (OANDA-style position books — the bot's existing
  `order_flow` proxy) has "doubtful practical value" ([Caveat Emptor](https://www.researchgate.net/publication/5168708)).
  The dealers' edge is an information asset (their clients' orders) + spread capture, structurally
  unavailable to a retail OANDA bot.
- **The reframe (confirmed):** profitable systematic FX = carry/risk premia + market-making/spread +
  execution/latency, NOT >52% directional prediction. Directional accuracy is the wrong scoreboard.

## 4. Implication for the bot

Directional ML is the wrong horse for retail spot majors. The only things that ever showed positive
expectancy here — carry / pre-2014 trend / EM carry — are fat-tailed risk premia that fail the 25%
max-drawdown gate, and capturing the trend component requires timing big FX moves (the coin flip
again). Recommended posture: **stay halted (fail-closed)**; this is a success state, not a failure.

Deep-research verification was cut off by the session API limit (resets 04:20 ET) with ~20 claims
still un-voted (abstained, not refuted); re-run to confirm the remaining claims if desired. Core
consensus claims above are established economics and independently solid.
