# Environment Setup Guide

Complete setup instructions for the ML Engine FX Trading Bot across different platforms.

---

## Quick Start

### Intel Mac (x86_64)

```bash
cd ~/Desktop/ml_engine
conda env create -f environment_intel_mac.yml
conda activate intel
```

### Apple Silicon (M1/M2/M3)

```bash
cd ~/Desktop/ml_engine
conda env create -f environment_tf_metal.yml
conda activate tf-metal
```

---

## TensorFlow Version Compatibility

All environments now target **TensorFlow 2.16+** for consistency. This version includes Keras 3.x by default.

### Version Matrix

| Environment | Python | TensorFlow | tensorflow-metal | protobuf |
|-------------|--------|------------|------------------|----------|
| `intel` | 3.12 | ≥2.16, <2.19 | N/A | ≥4.23.0, <5.0.0 |
| `tf-metal` | 3.11 | ≥2.16, <2.19 | ≥1.1.0, <1.3 | ≥4.23.0, <5.0.0 |

### Why TensorFlow 2.16+?

1. **Keras 3.x** - Better model serialization, improved `.keras` format
2. **Performance** - Optimized ops for both Intel and Apple Silicon
3. **Consistency** - Same API across all platforms
4. **Active support** - Latest security patches and bug fixes

### protobuf Requirement

TensorFlow 2.16+ requires `protobuf>=4.23.0`. Without this, you'll see:

```
AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'is_repeated'
```

This is now pinned in all environment files.

---

## Platform-Specific Details

### Intel Mac Setup

**File:** `environment_intel_mac.yml`  
**Environment name:** `intel`

```bash
# Create environment
conda env create -f environment_intel_mac.yml

# Activate
conda activate intel

# Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.16.x or 2.17.x or 2.18.x
```

**Notes:**
- Uses Python 3.12
- No GPU acceleration (CPU only)
- TensorFlow uses Intel MKL optimizations automatically

### Apple Silicon Setup

**File:** `environment_tf_metal.yml`  
**Requirements:** `requirements_tf_metal.txt`  
**Environment name:** `tf-metal`

```bash
# Create environment
conda env create -f environment_tf_metal.yml

# Activate
conda activate tf-metal

# Verify TensorFlow and Metal GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Notes:**
- Uses Python 3.11 (best Metal compatibility)
- Requires macOS 12.1+ for Metal plugin
- `tensorflow-metal` provides GPU acceleration

### Critical Metal Settings

For optimal performance on Apple Silicon, ensure these settings in `config/config_improved_H1.yaml`:

```yaml
model:
  recurrent_dropout: 0.0  # NON-ZERO CAUSES 10x SLOWDOWN

training:
  batch_size: 64          # Optimal for Metal GPU
  mixed_precision: false  # Metal doesn't fully support FP16
  jit_compile: false      # Avoid Metal XLA issues
  steps_per_execution: 10 # Reduces Python overhead
```

---

## Environment Files

### environment_intel_mac.yml

- **Purpose:** Full development environment for Intel Macs
- **Includes:** All ML packages, Jupyter, testing, visualization
- **Size:** ~2GB installed

### environment_tf_metal.yml

- **Purpose:** Minimal training environment for Apple Silicon
- **Includes:** Core TensorFlow stack, references requirements_tf_metal.txt
- **Size:** ~1.5GB installed

### requirements_tf_metal.txt

- **Purpose:** Pip requirements for Metal environment
- **Use:** Can be installed standalone with `pip install -r requirements_tf_metal.txt`

### requirements.txt

- **Purpose:** General requirements (CI/CD, development)
- **Note:** May overlap with conda environments

---

## Updating Environments

### Add a new package

```bash
# Intel
conda activate intel
pip install <package>
# Then add to environment_intel_mac.yml

# Metal
conda activate tf-metal
pip install <package>
# Then add to requirements_tf_metal.txt
```

### Upgrade TensorFlow

```bash
# Check current version
python -c "import tensorflow as tf; print(tf.__version__)"

# Upgrade (respects version pins)
pip install --upgrade 'tensorflow>=2.16,<2.19'

# Apple Silicon: also upgrade Metal plugin
pip install --upgrade 'tensorflow-metal>=1.1.0,<1.3'

# Verify protobuf compatibility
pip install --upgrade 'protobuf>=4.23.0,<5.0.0'
```

### Recreate environment from scratch

```bash
# Remove old
conda env remove -n intel  # or tf-metal

# Recreate
conda env create -f environment_intel_mac.yml  # or environment_tf_metal.yml
```

---

## Custom Environment Names

The `bin/Buddy` script auto-detects platform and activates:
- `intel` for x86_64
- `tf-metal` for arm64

To use a custom name:

```bash
export BUDDY_CONDA_ENV=my_custom_env
./bin/Buddy EUR_USD
```

Or set permanently in your shell config:

```bash
# ~/.zshrc or ~/.bashrc
export BUDDY_CONDA_ENV=my_custom_env
```

---

## Verifying Installation

### Quick health check

```bash
# Activate your environment first
conda activate intel  # or tf-metal

# Run validation
python validate_structure.py
```

### Test TensorFlow

```python
import tensorflow as tf

# Check version
print(f"TensorFlow: {tf.__version__}")

# Check devices
print(f"Devices: {tf.config.list_physical_devices()}")

# Quick computation test
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print(tf.matmul(a, b))
```

### Test training pipeline

```bash
# Small test run
./bin/Buddy train -c 500 -i EUR_USD --oanda-live --verbose
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## Related Documentation

- [README.md](../README.md) - Project overview
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Error solutions
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
