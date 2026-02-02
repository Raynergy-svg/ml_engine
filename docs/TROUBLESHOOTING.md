# Troubleshooting Guide

Common issues and solutions for the ML Engine FX Trading Bot.

---

## Table of Contents

1. [TensorFlow / Protobuf Compatibility](#tensorflow--protobuf-compatibility)
2. [Conda Environment Issues](#conda-environment-issues)
3. [Apple Silicon (Metal) GPU Issues](#apple-silicon-metal-gpu-issues)
4. [Model Loading / Keras Migration](#model-loading--keras-migration)
5. [OANDA API Issues](#oanda-api-issues)
6. [Training Performance Issues](#training-performance-issues)

---

## TensorFlow / Protobuf Compatibility

### Error: `Assertion failed: down_cast` / `Abort trap: 6`

**Full error:**
```
Assertion failed: (f == nullptr || dynamic_cast<To>(f) != nullptr), function down_cast, file external/local_tsl/tsl/platform/default/casts.h, line 58.
Abort trap: 6
```

**Cause:** Known bug in TensorFlow 2.18.0 on macOS. The conda-forge build has an internal type casting issue during `model.fit()`.

**Solution:** Downgrade to TensorFlow 2.16.2:

```bash
# In your active conda environment
pip install tensorflow==2.16.2
```

**Prevention:** Environment files now pin `tensorflow>=2.16.1,<2.17` to avoid 2.18.x.

### Error: `'google._upb._message.FieldDescriptor' object has no attribute 'is_repeated'`

**Cause:** TensorFlow 2.16+ requires `protobuf>=4.23.0`. The installed protobuf version is incompatible.

**Solution:**

```bash
# Quick fix - install compatible protobuf in your active environment
pip install 'protobuf>=4.23.0,<5.0.0'

# Or recreate the environment with correct dependencies
conda env remove -n intel
conda env create -f environment_intel_mac.yml
conda activate intel
```

**Prevention:** The environment files now pin `protobuf>=4.23.0,<5.0.0`. If you manually install packages, ensure protobuf stays in this range.

### Error: `No module named 'tensorflow'`

**Cause:** TensorFlow not installed or wrong conda environment activated.

**Solution:**

```bash
# Check which environment is active
conda info --envs

# Intel Mac
conda activate intel

# Apple Silicon (M1/M2/M3)
conda activate tf-metal
```

---

## Conda Environment Issues

### Error: Environment name mismatch

The `bin/Buddy` script expects specific environment names:

| Platform | Expected Environment |
|----------|---------------------|
| Intel Mac (x86_64) | `intel` |
| Apple Silicon (arm64) | `tf-metal` |

**Override:** Set `BUDDY_CONDA_ENV` to use a different name:

```bash
export BUDDY_CONDA_ENV=my_custom_env
./bin/Buddy EUR_USD
```

### Environment creation fails

```bash
# Remove old environment and recreate
conda env remove -n intel  # or tf-metal

# Intel Mac
conda env create -f environment_intel_mac.yml

# Apple Silicon
conda env create -f environment_tf_metal.yml
```

---

## Apple Silicon (Metal) GPU Issues

### Metal GPU not detected

**Symptoms:** Training runs on CPU despite having Apple Silicon.

**Check GPU availability:**

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Solutions:**

1. Ensure `tensorflow-metal` is installed:
   ```bash
   pip install 'tensorflow-metal>=1.1.0,<1.3'
   ```

2. Update macOS to 12.1+ (Metal plugin requirement)

3. Check TensorFlow version compatibility:
   | TensorFlow | tensorflow-metal |
   |------------|------------------|
   | 2.14-2.18 | 1.1.0 |
   | 2.18+ | 1.2.0 |

### Extremely slow training on Metal

**Cause:** `recurrent_dropout > 0` causes 10x slowdown on Metal GPU.

**Solution:** Ensure config has:

```yaml
model:
  recurrent_dropout: 0.0  # CRITICAL: Must be 0 for Metal performance
```

### Metal XLA compilation errors

**Solution:** Disable JIT compilation in config:

```yaml
training:
  jit_compile: false  # Metal doesn't fully support XLA
  mixed_precision: false  # Metal FP16 support is limited
```

---

## Model Loading / Keras Migration

### Error: `Unable to restore custom object` or `Unknown layer`

**Cause:** Keras 2.x models incompatible with Keras 3.x (TF 2.16+).

**Solutions:**

1. **Automatic migration** (built-in):
   The model loader at `main.py:399` (`_migrate_keras2_to_keras3()`) attempts automatic migration.

2. **Manual migration script:**
   ```bash
   python migrate_keras_models.py
   ```

3. **Retrain from scratch:**
   ```bash
   rm -rf trained_data/models/EUR_USD/
   ./bin/Buddy train -i EUR_USD --oanda-live
   ```

### Error: `ValueError: Cannot migrate: architecture incompatible`

**Cause:** Model architecture changed between versions.

**Solution:** Delete old models and retrain:

```bash
rm trained_data/models/transformer_direction.keras
rm trained_data/models/transformer_direction.meta.pkl
./bin/Buddy train -i EUR_USD --oanda-live
```

---

## OANDA API Issues

### Error: `401 Unauthorized`

**Solution:** Check `.env` file has valid credentials:

```bash
OANDA_API_TOKEN=your_practice_api_token
OANDA_ACCOUNT_ID=your_account_id
```

### Error: `Rate limit exceeded`

**Cause:** Too many API requests in short period.

**Solution:** Wait 1-2 minutes, or reduce candle count:

```bash
./bin/Buddy train -c 5000 -i EUR_USD --oanda-live  # Instead of 12000
```

### Error: `Instrument not found`

**Solution:** Use OANDA format for instrument names:

```bash
# Correct
./bin/Buddy EUR_USD

# Wrong
./bin/Buddy EURUSD
./bin/Buddy EUR/USD
```

---

## Training Performance Issues

### Training stuck or extremely slow

1. **Check batch size** - Reduce if running out of memory:
   ```yaml
   training:
     batch_size: 32  # Default is 64
   ```

2. **Check GPU utilization:**
   ```bash
   # macOS
   sudo powermetrics --samplers gpu_power -i 1000
   ```

3. **Reduce model complexity:**
   ```yaml
   transformer:
     d_model: 16  # Reduce from 32
     num_layers: 1  # Reduce from 2
   ```

### Out of memory errors

```bash
# Reduce candle count
./bin/Buddy train -c 5000 -i EUR_USD --oanda-live

# Or reduce sequence length in config
# config/config_improved_H1.yaml
model:
  seq_len: 30  # Reduce from 60
```

---

## Still Having Issues?

1. **Check logs:** `logs/buddy_training.log`
2. **Run with verbose:** `./bin/Buddy train -i EUR_USD --oanda-live --verbose`
3. **Validate structure:** `python validate_structure.py`
4. **Check config:** Ensure using `config/config_improved_H1.yaml`

---

## Related Documentation

- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Environment installation guide
- [README.md](../README.md) - Project overview
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
