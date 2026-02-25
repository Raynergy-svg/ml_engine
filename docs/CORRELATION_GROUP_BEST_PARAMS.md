# FX Pair Correlation Groups - Best Training Parameters

## Summary Table

| Group | Pairs That Move Together | Avg Corr (H1) | Why They Sync | Best "Master" Pair to Train On |
|-------|--------------------------|---------------|---------------|-------------------------------|
| Euro Basket | EUR/USD, EUR/JPY, EUR/GBP, EUR/AUD | 0.85–0.92 | Euro strength/weakness drives them all | EUR/USD (most liquid, cleanest signal) |
| Yen Carry / Risk-On | EUR/JPY, GBP/JPY, AUD/JPY, NZD/JPY, USD/JPY | 0.75–0.90 | Yen weakens on risk-on, strengthens on fear | EUR/JPY (your 70% winner — transfers best) |
| Sterling Cluster | GBP/USD, GBP/JPY, GBP/AUD, EUR/GBP | 0.78–0.88 | UK data + Brexit noise, but tracks USD & yen | GBP/USD (highest volume, least noise) |
| Aussie / Kiwi Commodity | AUD/USD, NZD/USD, AUD/JPY, NZD/JPY | 0.80–0.90 | Commodity prices + China growth | AUD/USD (more volume than NZD) |
| USD Safe-Haven | USD/JPY, USD/CHF, USD/CAD | 0.70–0.85 | USD strength vs risk-off (yen, franc) | USD/JPY (most responsive to sentiment) |
| Crosses (less common) | GBP/AUD, EUR/AUD, AUD/NZD | 0.65–0.80 | Commodity + risk-on, but noisier | AUD/JPY (if you want yen exposure) |

---

## Best Parameters by Group (From Sweep Results)

### 1. Yen Carry / Risk-On Group
**Pairs:** EUR/JPY, GBP/JPY, AUD/JPY, NZD/JPY, USD/JPY

**Best Performing Pair from Sweeps: GBP/JPY**
- Combined Score: 0.5656
- Direction Accuracy: 66.55%
- Direction F1: 0.6724
- Val Loss: 0.3222

**Optimal Parameters (GBP/JPY - Run: r31a4419):**
```yaml
batch_size: 128
seq_len: 120
tcn_nb_filters: 256
tcn_kernel_size: 3
tcn_dilations: [1, 2, 4, 8, 16, 32]
dropout_rate: 0.236
lr: 0.000508
l2_reg: 0.000149
direction_weight: 0.491
confidence_weight: 0.276
volatility_weight: 0.175
patience: 10
mixed_precision: True
train_smoothing: False
top_features: 150
```

**Also Available in Group:**
| Pair | Combined Score | Direction Acc | Val Loss |
|------|----------------|---------------|----------|
| AUD/JPY | 0.5600 | 64.71% | 0.3224 |

---

### 2. Sterling Cluster Group
**Pairs:** GBP/USD, GBP/JPY, GBP/AUD, EUR/GBP

**Best Performing Pair from Sweeps: EUR/GBP**
- Combined Score: 0.5724
- Direction Accuracy: 66.58%
- Direction F1: 0.6627
- Val Loss: 0.4332

**Optimal Parameters (EUR/GBP - Run: yxjlclgt):**
```yaml
batch_size: 128
seq_len: 90
tcn_nb_filters: 256
tcn_kernel_size: 5
tcn_dilations: [1, 2, 4, 8, 16, 32]
dropout_rate: 0.363
lr: 0.000415
l2_reg: 0.000098
direction_weight: 0.493
confidence_weight: 0.238
volatility_weight: 0.297
patience: 25
mixed_precision: False
train_smoothing: True
top_features: 150
```

**Also Available in Group:**
| Pair | Combined Score | Direction Acc | Val Loss |
|------|----------------|---------------|----------|
| GBP/JPY | 0.5656 | 66.55% | 0.3222 |
| GBP/AUD | 0.5720 | 65.76% | 0.3414 |

---

### 3. Aussie / Kiwi Commodity Group
**Pairs:** AUD/USD, NZD/USD, AUD/JPY, NZD/JPY

**Best Performing Pair from Sweeps: AUD/USD**
- Combined Score: 0.5683
- Direction Accuracy: 65.69%
- Direction F1: 0.6642
- Val Loss: 0.3365

**Optimal Parameters (AUD/USD - Run: bancfgtj):**
```yaml
batch_size: 128
seq_len: 90
tcn_nb_filters: 64
tcn_kernel_size: 5
tcn_dilations: [1, 2, 4, 8, 16, 32]
dropout_rate: 0.183
lr: 0.000630
l2_reg: 0.000405
direction_weight: 0.530
confidence_weight: 0.346
volatility_weight: 0.145
patience: 20
mixed_precision: True
train_smoothing: True
top_features: None  # Uses all features
```

**Also Available in Group:**
| Pair | Combined Score | Direction Acc | Val Loss |
|------|----------------|---------------|----------|
| AUD/JPY | 0.5600 | 64.71% | 0.3224 |

---

### 4. USD Safe-Haven Group
**Pairs:** USD/JPY, USD/CHF, USD/CAD

**Best Performing Pair from Sweeps: USD/CAD**
- Combined Score: 0.5725
- Direction Accuracy: 66.37%
- Direction F1: 0.6574
- Val Loss: 0.3989

**Optimal Parameters (USD/CAD - Run: gshma2xk):**
```yaml
batch_size: 128
seq_len: 120
tcn_nb_filters: 128
tcn_kernel_size: 5
tcn_dilations: [1, 2, 4, 8, 16]
dropout_rate: 0.468
lr: 0.000552
l2_reg: 0.000013
direction_weight: 0.435
confidence_weight: 0.325
volatility_weight: 0.264
patience: 25
mixed_precision: False
train_smoothing: True
top_features: 150
```

---

### 5. Crosses Group
**Pairs:** GBP/AUD, EUR/AUD, AUD/NZD

**Best Performing Pair from Sweeps: AUD/NZD** (Highest Combined Score)
- Combined Score: 0.5804
- Direction Accuracy: 66.36%
- Direction F1: 0.6445
- Val Loss: 0.3273

**Optimal Parameters (AUD/NZD - Run: 908esr1t):**
```yaml
batch_size: 128
seq_len: 90
tcn_nb_filters: 256
tcn_kernel_size: 7
tcn_dilations: [1, 2, 4, 8, 16]
dropout_rate: 0.420
lr: 0.000376
l2_reg: 0.000208
direction_weight: 0.764
confidence_weight: 0.117
volatility_weight: 0.101
patience: 20
mixed_precision: True
train_smoothing: True
top_features: 100
```

**Also Available in Group:**
| Pair | Combined Score | Direction Acc | Val Loss |
|------|----------------|---------------|----------|
| GBP/AUD | 0.5720 | 65.76% | 0.3414 |

---

## Quick Reference: Best Params by Master Pair

| Master Pair | Group | seq_len | filters | kernel | dropout | lr | direction_w | patience |
|-------------|-------|---------|---------|--------|---------|-----|-------------|----------|
| EUR/USD* | Euro Basket | - | - | - | - | - | - | - |
| EUR/JPY* | Yen Carry | - | - | - | - | - | - | - |
| **GBP/JPY** | Yen Carry | 120 | 256 | 3 | 0.24 | 0.00051 | 0.49 | 10 |
| **EUR/GBP** | Sterling | 90 | 256 | 5 | 0.36 | 0.00042 | 0.49 | 25 |
| **AUD/USD** | Commodity | 90 | 64 | 5 | 0.18 | 0.00063 | 0.53 | 20 |
| **USD/CAD** | USD Safe | 120 | 128 | 5 | 0.47 | 0.00055 | 0.44 | 25 |
| **AUD/NZD** | Crosses | 90 | 256 | 7 | 0.42 | 0.00038 | 0.76 | 20 |

*Not in current sweep data - use similar pair params for transfer learning

---

## Transfer Learning Recommendations

1. **For EUR/USD (Euro Basket):** Use EUR/GBP params (similar EUR exposure)
2. **For EUR/JPY (Yen Carry):** Use GBP/JPY params (both JPY crosses)
3. **For NZD pairs:** Use AUD/USD params (commodity correlation)

---

*Generated from wandb sweep results - Entity: tencylinder8310-smartdebtflow-com, Project: ml_engine_fx*
