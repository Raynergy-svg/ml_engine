# ML Engine Setup and Usage Guide

Welcome to the **ML Engine** repository! This tool is designed to simplify training and testing of various machine learning models. Dive into the setup instructions and commands below to get started in your M1 Mac environment.

---

## 🌟 Supported Models
Our ML Engine supports the following models:

1. **Temporal Convolutional Networks (TCN)**
2. **XGBoost**
3. **Gradient Boosting**
4. **Ridge Regression**
5. **Ensemble Techniques**

---

## 🛠️ Setting Up the Environment (Optimized for M1 Mac using Conda)

### Prerequisites
- macOS with an Apple M1 chip (or later).
- Conda installed. We recommend **Miniforge**, as it simplifies installation and supports TensorFlow-metal for Apple Silicon optimization.

### Steps

1. **Install Miniforge**:
    ```bash
    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    bash Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
    ```

2. **Create a new Conda environment**:
    ```bash
    conda create -n ml_env python=3.9
    conda activate ml_env
    ```

3. **Install Required Packages**:
    ```bash
    pip install tensorflow tensorflow-metal xgboost scikit-learn
    ```

4. **Verify TensorFlow-metal Integration**:
    Run the following Python code to ensure `tensorflow-metal` is leveraging the GPU:
    ```python
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    ```
    This should output at least one GPU device.

---

## ⚙️ Commands

### Training a Model
To train a supported model, use the `buddy train` subcommand. Example:

```bash
buddy train --model TCN --data path/to/training_data.csv
```

Replace `TCN` with any of the other supported models (`XGBoost`, `GradientBoosting`, `RidgeRegression`, `Ensemble`).

### Testing a Model
To test a trained model, use the `buddy test` subcommand. Example:

```bash
buddy test --model TCN --data path/to/testing_data.csv --model-path path/to/saved/model
```

### Additional Commands
- **Scan buddy compatibility**:
    ```bash
    buddy scan
    ```
- **Check buddy status**:
    ```bash
    buddy status
    ```

---

Stay productive and enjoy building intelligent models with ML Engine! 🚀