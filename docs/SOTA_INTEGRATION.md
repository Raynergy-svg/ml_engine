# SOTA Integration — Goals 1 & 2

## Overview

This document describes the two parallel workstreams that modernize Buddy's
signal generation and agent consensus layers.

| Goal | What | Status |
|------|------|--------|
| 1 | End-to-end deep learning signal core (replaces classical ensemble) | Implemented |
| 2 | Neural agent policies (replaces rule-based heuristic agents) | Implemented |

Both systems are **opt-in** via config flags and fall back to the legacy
implementation when disabled.

---

## Goal 1: SOTA Raw-Sequence Inference (`src/sota_core/`)

### What changed

The classical pipeline computes hand-engineered features (SMA, RSI, ADX, MACD,
BB width, etc.) and feeds them into an ensemble of Transformer + XGBoost +
Ridge + Random Forest models.  Each model votes, and a gate evaluator blocks
trades that don't pass threshold checks.

The SOTA pipeline feeds **raw OHLCV** directly into a single neural network:

```
OHLCV (5 channels, 128 bars)
  → Variable Selection Network      (learns which channels matter per timestep)
  → 1D CNN blocks                   (learned local patterns, replaces SMA/RSI)
  → Positional Encoding + Transformer (long-range dependencies)
  → Global Average Pooling
  → Gated Residual Network
  → [direction head] sigmoid        (LONG vs SHORT probability)
  → [regime head] softmax           (LOW / NORMAL / HIGH / EXTREME)
```

No human-designed features.  The model discovers trend, momentum, mean-reversion,
volatility, and support/resistance concepts in latent space.

### Training

Two-phase training is required:

**Phase 1 — Self-supervised pre-training**
- Task 1: Masked candle reconstruction (fill in masked OHLCV values)
- Task 2: Next-bar return regression
- Data: years of unlabeled OHLCV history
- Output: encoder weights saved to `trained_data/models/sota_pretrain/`

**Phase 2 — Supervised fine-tuning**
- Labels: future 5-bar direction + realized volatility regime
- Data: labeled trade outcomes or forward-return labels
- Output: fine-tuned model saved to `trained_data/models/sota_finetuned/sota_model.keras`

### Usage

```python
from src.scanner.config import ScannerConfig
from src.scanner.sota_integration import resolve_inference_engine

cfg = ScannerConfig()
cfg.use_sota_inference = True
cfg.sota_model_path = "trained_data/models/sota_finetuned/sota_model.keras"

engine = resolve_inference_engine(cfg)
signal = engine.predict(pair="EUR_USD", df_raw=df)
# signal.direction → "LONG" | "SHORT" | "HOLD"
# signal.confidence → float [0, 1]
```

### Training script

```bash
# Pre-train on raw historical data
python scripts/train_sota_and_neural.py \
  --phase pretrain \
  --data data/historical/*.csv

# Fine-tune on labeled outcomes
python scripts/train_sota_and_neural.py \
  --phase finetune \
  --data data/historical/*.csv
```

---

## Goal 2: Neural Agent Policies (`src/scanner/agents/neural/`)

### What changed

The legacy ScannerAgentTeam has 15 specialist agents, each evaluating a trade
setup with hand-coded heuristics:

- `trend`: `close > sma_20 >= sma_50`
- `mean_reversion`: `RSI > 58` vs `RSI < 42`
- `momentum`: MACD histogram + ROC alignment
- `volatility`: fixed ATR thresholds

NeuralAgentTeam replaces these with **learned policies** — small MLPs that
receive the same context (price history, features, analysis) and output a
score in [0, 1].  Each policy has ~1,000–5,000 parameters and trains in
real-time from trade outcomes.

### Architecture

Each agent policy is a feedforward network:

```
feature vector (20–120 dims)
  → Dense(64, relu) → Dropout(0.2)
  → Dense(32, relu) → Dropout(0.2)
  → Dense(1, sigmoid)  # score in [0, 1]
```

Policies extract features differently (the network learns to specialize):
- **TrendPolicy**: sees raw OHLCV window + SMA ratios + ADX
- **MomentumPolicy**: sees 20-bar returns + MACD + ROC
- **MeanReversionPolicy**: sees deviation from SMA + RSI + BB width
- **VolatilityPolicy**: sees ATR history + regime one-hot

### Training

After each trade closes, the trainer builds a supervised dataset:

```
features  →  target_score
  where target_score = 1.0 if (agent voted correctly)
                      = 0.0 if (agent voted incorrectly)
```

Online updates run every 100 trades (1 epoch, 32 batch).  Offline batch
retraining can be triggered periodically on the full journal.

### Usage

```python
from src.scanner.config import ScannerConfig
from src.scanner.sota_integration import resolve_agent_team

cfg = ScannerConfig()
cfg.use_neural_agents = True

team = resolve_agent_team(cfg)
analysis = team.evaluate(analysis, df_raw, df_feat, gate_details)
# analysis.weighted_vote_score → float
# analysis.agent_passed → bool
```

### Training script

```bash
# Train neural agents from journal outcomes
python scripts/train_sota_and_neural.py \
  --phase neural \
  --journal trained_data/journal.jsonl
```

---

## Integration with Scanner Engine

The patch is applied at runtime via `src/scanner/sota_engine_patch.py`:

```python
from src.scanner.engine import Scanner
from src.scanner.sota_engine_patch import patch_scanner_for_sota

scanner = Scanner(config)
patch_scanner_for_sota(scanner)  # swaps engine/agents if flags are True
```

No modifications to `engine.py` are required.  The patch is:
- **Reversible**: set `use_sota_inference=False` and `use_neural_agents=False`
- **Safe**: falls back to legacy implementations on any error
- **Isolated**: all new code lives in `src/sota_core/` and `src/scanner/agents/neural/`

---

## Config Flags

Add to your `config/config_improved_H1.yaml` (or set on `ScannerConfig`):

```yaml
# Goal 1: End-to-end DL signal core
use_sota_inference: false
sota_model_path: "trained_data/models/sota_finetuned/sota_model.keras"

# Goal 2: Neural agent policies
use_neural_agents: false
neural_agent_save_dir: "trained_data/models/neural_agents"
```

Set both to `true` to activate the full SOTA stack.  Set both to `false`
for the legacy behavior.

---

## File Map

```
src/
  sota_core/
    __init__.py
    raw_sequence_model.py      # VariableSelectionNetwork + CNN-Transformer
    trainer.py                  # Two-phase training pipeline
    inference.py                # SOTAInference (scanner-compatible wrapper)
  scanner/
    agents/
      neural/
        __init__.py
        neural_agent_base.py   # Base class + online update logic
        policies.py             # Trend, Momentum, MR, Volatility policies
        trainer.py              # Multi-agent batch trainer
        team_bridge.py          # NeuralAgentTeam (mirror of ScannerAgentTeam)
    sota_integration.py         # Factory functions
    sota_engine_patch.py        # Runtime monkey-patch for Scanner
    config.py                   # ← new flags added here
tests/
  sota_core/
    test_smoke.py               # 8 tests (model build, forward, pretrain, inference)
  scanner/agents/neural/
    test_neural_agents.py       # 8 tests (policy eval, team E2E, trainer)
scripts/
  train_sota_and_neural.py     # CLI entry point for training both systems
```

---

## Why this is state-of-the-art

| Capability | Legacy | SOTA (this PR) |
|-----------|--------|----------------|
| Signal source | Hand-engineered features (SMA, RSI, MACD) | Raw OHLCV → learned representations |
| Model architecture | Separate Transformer/XGB/Ridge/RF ensemble | Single end-to-end CNN-Transformer |
| Pre-training | None (supervised only) | Self-supervised masked reconstruction + next-return |
| Agent logic | Fixed heuristic formulas | Small neural policies trained on outcomes |
| Agent specialization | Human-assigned categories | Emerges from data via feature extraction + gradient descent |
| Weight learning | Scalar arithmetic (+0.10/-0.15) | Actual gradient-based policy updates |
| Regime awareness | Fixed lookup tables | Learned regime head (4-class softmax) |

---

## Next Steps / Roadmap

1. **Collect training data**: Run the scanner in legacy mode for 2–4 weeks to
   accumulate labeled trade outcomes and raw OHLCV history.
2. **Pre-train SOTA encoder**: Run `scripts/train_sota_and_neural.py --phase pretrain`.
3. **Fine-tune SOTA model**: Run `scripts/train_sota_and_neural.py --phase finetune`.
4. **Train neural agents**: Run `scripts/train_sota_and_neural.py --phase neural`.
5. **A/B test**: Enable SOTA flags on a paper account; compare PnL vs legacy.
6. **Iterate**: Retrain both systems monthly as new data arrives.
