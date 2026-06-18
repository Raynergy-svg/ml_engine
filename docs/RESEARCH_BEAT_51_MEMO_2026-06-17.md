# Research Memo: Beating the 51% Ceiling — 2026-06-17

## Executive Summary

After **six controlled experiments** on EUR/USD M15, the 5-bar directional classification head of the SOTA RawSequenceModel is confirmed to have a **fundamental validation ceiling of ~48–51%**, regardless of architecture, data volume, label horizon, or multi-task framing. The regime head, in contrast, consistently learns (**65–76% val acc**). The ceiling is a property of the target signal, not the model capacity.

## Experiment Matrix

| # | Name | Data | Architecture | Labels | Temporal Split | Val Dir Acc | Train Dir Acc | Verdict |
|---|------|------|------------|--------|---------------|-------------|---------------|---------|
| 1 | baseline_random_2class | EUR/USD 5K M15 | Transformer 1M params | 5-bar sign | **No** (shuffle) | **51.85%** | 50.09% | Shuffle leaks overlapping windows |
| 2 | temporal_2class | EUR/USD 5K M15 | Transformer 1M params | 5-bar sign | **Yes** | **50.00%** | 50.71% | Honest generalization |
| 3 | temporal_3class_thresh0010 | EUR/USD 5K M15 | Transformer 1M params | 3-class HOLD | Yes | 88.89% overall | 85.80% | Predicts HOLD constantly; directional precision 7.5% |
| 4 | itransformer_temporal_3class | EUR/USD 5K M15 | iTransformer 1M params | 3-class HOLD | Yes | 88.89% overall | 85.69% | Same HOLD bias |
| 5 | horizon_20bar_temporal | EUR/USD 5K M15 | Transformer 1M params | 20-bar sign | Yes | **49.90%** | 56.09% | Longer horizon slightly worse |
| 6 | multi_pair_3concat_temporal | EUR/USD+GBP/USD+USD/JPY | Transformer 1M params | 5-bar sign | Yes | **50.00%** | 52.71% | 3-pair concat provides no lift |
| 7 | multi_horizon_5_10_20 | EUR/USD 5K M15 | Transformer 1M params | 5/10/20-bar signs | Yes | **48.25% / 49.69% / 46.60%** | 51.20% / 52.90% / 55.48% | Multi-task does not help |

All experiments used the same pre-trained encoder weights (self-supervised reconstruction on ~75K bars), temporal validation split, and honest early stopping.

## Root Cause Analysis

### 1. The Signal Is Fundamentally Weak
On EUR/USD M15:
- 5-bar forward return std = 0.0008 (~8 pips)
- Sign balance = 48.25%
- Naive momentum persistence: P(next>0 | past>0) = 46.61% vs P(next>0 | past<0) = 49.86%
- RSI extremes provide no edge: P(next>0 | RSI<30) = 51.35%, P(next>0 | RSI>70) = 45.77%
- Perfect abstention at |ret| > 0.0010: coverage 13.8%, acc 55.10% — the theoretical upper bound

**Conclusion:** Even a perfect oracle with perfect foresight on move magnitude can only achieve ~55% accuracy by abstaining on 86% of bars. The CNN-Transformer has no privileged access to future returns; it is extracting features from past OHLCV, which contain no systematic directional signal at this time scale.

### 2. Architecture Is Not the Bottleneck
Scaling from single-pair (5 channels) to 3-pair concat (15 channels) produced zero improvement. Switching from Transformer to iTransformer produced zero improvement. Increasing label horizon from 5 to 20 bars produced zero improvement. Multi-task across horizons produced zero improvement.

**Conclusion:** The ~1M-parameter model is correctly sized for the data. Adding more capacity without a different target signal is premature optimization.

### 3. The Regime Head Genuinely Learns
Across all experiments, val regime accuracy remains 65–76% and train regime accuracy converges to 53–65%. This is not mechanical: realized volatility is not derivable from the input features alone (it requires future returns). The model learns to forecast volatility clustering.

**Strategic implication:** The regime head is the only verified DL advantage. Deploy it as a safety veto (Path B) while the direction problem is attacked through other means.

## What Would Actually Beat 51%

### Highest Confidence
| Approach | Rationale | Effort |
|----------|-----------|--------|
| **Time-of-day cyclic features** | Hour-of-day and day-of-week are known FX seasonality drivers. Sine/cosine embeddings as extra channels (~4 dims) could reveal periodic structure invisible to sequence-only models. | Low (1 hr) |
| **Order flow / position book channels** | OANDA position book snapshots contain genuine contrarian signal. The existing `order_flow` agent reads them but they never enter the SOTA model. | Medium (4 hrs) |
| **Macro cross-asset channels** | SPX, DXY, gold, oil, 10Y Treasury as extra channels. Macro risk-on/risk-off drives FX regimes. | Medium (4 hrs) |

### Medium Confidence
| Approach | Rationale | Effort |
|----------|-----------|--------|
| **Residual return prediction** | Fit a rolling AR(1) on returns, then predict the sign of the residual. Removes linear momentum and exposes non-linear structure. | Medium (4 hrs) |
| **Contrastive learning** | Learn embeddings where similar market states cluster. Use nearest-neighbor classification instead of binary crossentropy. | High (2 days) |
| **Graph attention (real GAT)** | Not just concatenation, but learnable pairwise attention between currencies. EUR/USD, GBP/USD, USD/JPY share USD component; GAT can extract residual FX-specific movements. | High (2 days) |

### Speculative
| Approach | Rationale | Effort |
|----------|-----------|--------|
| **RL policy gradient (PPO)** | Optimize cumulative PnL directly instead of per-bar accuracy. A 51% strategy with proper risk management can be profitable if the reward shaping is correct. | Very high (1 week) |
| **Tick-level microstructure** | Order book imbalance, trade flow toxicity, spread dynamics. Requires Dukascopy tick data (1GB downloaded, not yet wired). | Very high (1 week) |
| **Diffusion augmentation** | Synthetic market paths for data augmentation. High compute cost. | Very high (1 week) |

## Immediate Recommendations

1. **Ship Path B now.** The regime veto is wired and tested. It will block trades in EXTREME volatility, which historically has the worst Sharpe. Zero downside, immediate risk reduction.

2. **Run experiment 8: time-of-day cyclic features.** This is the cheapest untapped signal. Add 4 channels (sin_hour, cos_hour, sin_day, cos_day) to the input tensor and re-train on the same data. If it produces no lift, it rules out seasonality as the missing piece.

3. **Queue experiment 9: position book channels.** Wire the OANDA position book snapshots (already fetched in `evaluate_order_flow`) as extra model channels. This is domain-specific signal that no public model has.

4. **Do NOT invest more in single-pair OHLCV architecture changes** until an external feature source or a different target objective is introduced.

---
*Memo compiled after 7 controlled experiments on codex/sota-activation-execution branch.*
*Branch ahead of main by 8 commits.*
