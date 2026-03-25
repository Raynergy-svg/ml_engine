# Buddy ML Engine — Full Training Pipeline (M1 Mac)

## Quick Start

```bash
# 1. Install dependencies
pip install tensorflow-macos tensorflow-metal
pip install lightgbm xgboost scikit-learn pandas numpy requests
pip install stable-baselines3 gymnasium  # For RL suite (Tier 3)

# 2. Ensure .env.local has your OANDA credentials
cat .env.local
# OANDA_PRACTICE_TOKEN=your_token
# OANDA_PRACTICE_ACCOUNT_ID=your_account_id

# 3. Run the full pipeline
chmod +x scripts/run_full_training.sh
./scripts/run_full_training.sh
```

## Pipeline Architecture

```
Tier 1: Core Ensemble ──────────────────────────────────
  8 models × 5 master pairs = 40 model trainings
  Models: transformer, tcn, lgbm_momentum, lgbm_risk,
          ridge, histgb, transformer_regime, tcn_vol_regime
  Pairs: GBP_JPY, EUR_GBP, AUD_USD, USD_CAD, AUD_NZD

Tier 2: Transfer Learning ──────────────────────────────
  Master → Target via warm-start + EWC + layer freezing
  Groups: yen_carry, euro_basket, sterling_cluster,
          aussie_kiwi_commodity, usd_safe_haven, crosses

Tier 3: RL Suite ───────────────────────────────────────
  PPO Position Sizer → optimal position sizing
  SAC Gate Thresholds → adaptive confidence/momentum/risk gates
  PPO Optimal Exits → when to TP/SL
  Ridge Reward Model → learned reward shaping

Tier 4: Meta-Labeler ──────────────────────────────────
  XGBoost secondary model: predicts WHEN to trust signals

Tier 5: Confidence Calibrator ─────────────────────────
  Platt scaling: raw confidence → P(win|confidence)

Final: Validation Report ──────────────────────────────
  PASS/FAIL per model, <6% gap enforcement
```

## Selective Training

```bash
# Train only one master pair
./scripts/run_full_training.sh --pair GBP_JPY

# Train only a specific tier
./scripts/run_full_training.sh --tier 1    # Core ensemble only
./scripts/run_full_training.sh --tier 2    # Transfer learning only
./scripts/run_full_training.sh --tier 3    # RL suite only

# Skip a tier
./scripts/run_full_training.sh --skip-tier 3  # Skip RL

# Custom candle count
./scripts/run_full_training.sh --candles 10000

# Train a single model for a single pair
python scripts/train_single_model_m1.py --instrument GBP_JPY --model transformer --candles 25000
python scripts/train_single_model_m1.py --instrument GBP_JPY --model all --candles 25000
```

## Wandb-Optimized Hyperparameters

Each master pair uses sweep-optimized parameters:

| Pair    | seq_len | batch | LR       | patience | TCN filters | kernel |
|---------|---------|-------|----------|----------|-------------|--------|
| GBP_JPY | 120     | 128   | 0.000508 | 10       | 256         | 3      |
| EUR_GBP | 90      | 128   | 0.000415 | 25       | 256         | 5      |
| AUD_USD | 90      | 128   | 0.000630 | 20       | 64          | 5      |
| USD_CAD | 120     | 128   | 0.000552 | 25       | 128         | 5      |
| AUD_NZD | 90      | 128   | 0.000376 | 20       | 256         | 7      |

## Correlation Groups

| Group                 | Master  | Targets                           |
|-----------------------|---------|-----------------------------------|
| Yen Carry             | GBP_JPY | EUR_JPY, AUD_JPY, NZD_JPY, USD_JPY |
| Euro Basket           | EUR_GBP | EUR_USD, EUR_JPY, EUR_AUD         |
| Sterling Cluster      | EUR_GBP | GBP_USD, GBP_AUD                  |
| Aussie/Kiwi Commodity | AUD_USD | NZD_USD, AUD_JPY, NZD_JPY         |
| USD Safe-Haven        | USD_CAD | USD_JPY, USD_CHF                  |
| Crosses               | AUD_NZD | GBP_AUD, EUR_AUD                  |

## Output Structure

```
trained_data/
├── models/
│   ├── GBP_JPY/
│   │   ├── transformer_direction.keras
│   │   ├── tcn_volatility.keras
│   │   ├── lgbm_momentum.pkl
│   │   ├── lgbm_risk.pkl
│   │   ├── ridge_confidence.pkl
│   │   ├── histgb_direction.pkl
│   │   ├── transformer_regime.keras
│   │   ├── tcn_volatility_regime.keras
│   │   └── training_summary.json
│   ├── EUR_GBP/ ...
│   ├── EUR_JPY/ ...  (transfer learned)
│   ├── joint/
│   │   └── meta_labeler.pkl
│   ├── rl_position_sizer.zip
│   ├── sac_gate_thresholds.zip
│   ├── ppo_optimal_exit.zip
│   ├── reward_model.pkl
│   └── confidence_calibration.json
├── cache/training_data/
│   ├── GBP_JPY_H1_25000.csv  (cached OANDA data)
│   └── ...
├── logs/
│   ├── tier1_GBP_JPY.log
│   ├── tier2_transfer.log
│   ├── tier3_rl.log
│   └── tier4_5_meta.log
└── validation_report.json
```

## Validation Criteria

- Direction models (transformer, tcn, histgb): gap < 6% between train/val accuracy
- All models must load without error
- RL models must produce valid position sizing output
- Meta-labeler overfitting gap < 8%

## Troubleshooting

**tf-metal not detected:**
```bash
pip install --upgrade tensorflow-macos tensorflow-metal
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**OOM on M1:**
Reduce batch_size or seq_len in `train_single_model_m1.py` MASTER_PARAMS.

**OANDA fetch errors:**
Check `.env.local` credentials. The scripts cache data to CSV so you only fetch once.

**stable-baselines3 conflicts:**
RL training uses subprocess isolation to avoid TF/PyTorch Metal conflicts.
