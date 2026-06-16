# SOTA Model Training Report

Generated: 2026-06-15 20:18 EDT
Environment: macOS M1, TensorFlow (Metal/CPU hybrid)

---

## Data Discovery Summary

- Discovered **15 parquet files** in `trained_data/harvest/*.parquet`
- All 15 files loaded successfully after adding parquet support to `_load_candles_csv`
- Original data loader assumed CSV format only; parquet files are binary and caused UTF-8 decode errors
- Total windows extracted: **1,836 train / 229 val** for Phase 1; **58,404 train / 7,300 val** for Phase 2

---

## Phase 1: Self-Supervised Pre-Training

**Status:** ✅ Completed successfully

- Early stopping triggered at **epoch 21/50**
- Restored best weights from epoch 21 (lowest val_loss: **0.0294**)
- Pre-trained weights saved to `trained_data/models/sota_pretrain/pretrain_weights.weights.h5` (8.5 MB)

### Loss Curves (Validation)
| Epoch | val_loss | val_recon_head_loss | val_next_return_loss |
|-------|----------|---------------------|----------------------|
| 1     | 0.2924   | 0.2492              | 0.0110               |
| 2     | 0.2069   | 0.1790              | 0.0011               |
| 3     | 0.1789   | 0.1542              | 0.0026               |
| 4     | 0.1439   | 0.1254              | 6.3e-04              |
| 5     | 0.1421   | 0.1218              | 0.0060               |
| 6     | 0.1212   | 0.1058              | 4.6e-05              |
| 7     | 0.1079   | 0.0934              | 0.0015               |
| 8     | 0.0985   | 0.0860              | 0.0011               |
| 9     | 0.0785   | 0.0679              | 8.1e-04              |
| 10    | 0.0622   | 0.0543              | 1.4e-04              |

**Success criterion met:** Validation MSE decreased monotonically for the first 10 epochs.

### Runtime
- ~20 minutes total
- ~30–60s per epoch

---

## Phase 2: Supervised Fine-Tuning

**Status:** ✅ Completed (with caveats)

- Ran **10 epochs** (default is 30; reduced due to time constraints — full 30 would take ~5 hours)
- Final model saved to `trained_data/models/sota_finetuned/sota_model.keras` (9.2 MB)
- Restored best weights from epoch 5 (lowest val_direction_loss: **0.6928**)

### Final Metrics
| Metric | Train | Validation |
|--------|-------|------------|
| direction_loss | 0.6932 | 0.6928 |
| direction_accuracy | 50.29% | 51.59% |
| regime_loss | 0.8304 | 0.8329 |
| loss | 0.9424 | 0.9430 |

**Note:** Direction accuracy remained near random-chance (~50%) throughout all epochs. The model did not learn discriminative directional signals from the pre-training or fine-tuning data under the current configuration.

### Runtime
- ~10 minutes per epoch (58,404 samples, batch_size=32)
- ~100 minutes total for 10 epochs

---

## Errors & Adjustments

### Bug Fixes Applied (src/sota_core/trainer.py)

1. **Parquet loading failure** (`utf-8 decode error`)
   - Root cause: `_load_candles_csv` assumed all files were CSV
   - Fix: Added `pd.read_parquet()` branch for `.parquet` suffix

2. **`TrainerConfig has no attribute 'seq_len'`**
   - Root cause: `_generator` referenced `cfg.seq_len` instead of `self.model.cfg.seq_len`
   - Fix: Updated reference to `self.model.cfg.seq_len`

3. **`class_weight` unsupported for multi-output models**
   - Root cause: Keras 3 rejects `class_weight` on models with multiple outputs (`direction` + `regime`)
   - Attempted `sample_weight` workaround but hit additional Keras 3 structural mismatch errors (`KeyError: 0`)
   - Resolution: Removed class balancing entirely for Phase 2. Dataset is naturally balanced (~50.5%), so impact is minimal.

4. **`ReduceLROnPlateau` / `EarlyStopping` require explicit `mode='min'`**
   - Root cause: Keras 3 cannot auto-infer minimize/maximize for custom metric names like `val_direction_loss`
   - Fix: Added `mode="min"` to both callbacks

### Pre-Trained Weight Transfer Issue

When loading `pretrain_weights.weights.h5` into the fine-tuning model, **all 44 layers were skipped**:
- Root cause: The pre-training model wraps the encoder as a sub-model (`sota_encoder`), giving weight paths like `sota_encoder/conv_0/kernel`
- The fine-tuning model flattens the same layers directly (`conv_0/kernel`), so Keras cannot match the names even with `skip_mismatch=True`
- **Impact:** Fine-tuning ran from scratch; pre-training did not transfer to Phase 2. This is likely a major contributor to the lack of convergence.

---

## Recommendations

1. **Fix weight transfer architecture**  
   Make the fine-tuning model also use the `sota_encoder` sub-model explicitly, so weight paths align between pre-train and fine-tune checkpoints. Alternatively, save only `self._encoder.save_weights()` during pre-training and load only those weights during fine-tuning.

2. **Reduce dataset redundancy**  
   Phase 2 uses `stride=1`, generating ~58k windows with massive overlap. Increasing `stride` (e.g., to `seq_len // 4` as in Phase 1) would cut training time by 4× with minimal information loss.

3. **Investigate GPU/Metal utilization**  
   ~300ms/step on M1 suggests CPU execution. Verify `tensorflow-metal` plugin is installed and active. CPU fallback makes 30-epoch fine-tuning impractical.

4. **Tune learning rate / architecture**  
   Near-random accuracy for 10 epochs from scratch on a balanced binary task usually signals either:
   - Learning rate too low for the architecture depth
   - Architecture capacity insufficient for raw OHLCV → direction mapping
   - Label noise (next-5-bar return direction is a weak signal)
   Consider curriculum learning or a stronger auxiliary loss.

5. **Resume for full 30 epochs if desired**  
   The script supports `--finetune-only --epochs-finetune 30` once the above issues are resolved.

---

## Success Criteria Checklist

| Criterion | Status |
|-----------|--------|
| Phase 1 val MSE decreases monotonically first 10 epochs | ✅ PASS (0.2924 → 0.0622) |
| Phase 2 model file exists at expected path | ✅ PASS (9.2 MB saved) |
| No TensorFlow crashes | ✅ PASS (script completed) |

---

*All temporary code fixes have been reverted per constraint instructions.*
