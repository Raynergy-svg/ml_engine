# ML Engine - Forex Trading Bot

## ⚠️ **WARNING: DISCLAIMER**

**WARNING: This is experimental code. Not financial advice. Trading Forex involves substantial risk of loss. Use only money you can afford to lose. Test on demo first. I am not responsible for any outcomes.**

**"It works for me. Might work for you. Probably won't make you rich. But it might teach you something."**

---

## What Is This?

A Python-based H1 modular ensemble Forex trading bot. The system employs specialized machine learning models for:

- **Direction**: TCN/Transformer network to predict movement.
- **Momentum**: XGBoost for strength measurement.
- **Risk/Drawdown**: RandomForest.
- **Confidence**: Ridge regression.

It uses gated decision logic to ensure all components align before trades are made. Features include daily instrument scanning and adaptive retraining for individual Forex pairs. The bot integrates with Oanda for end-to-end trade execution.

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 or later
- Conda (recommended: **Miniforge** for M1 Mac)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Raynergy-svg/ml_engine.git
    cd ml_engine
    ```

2. **Create a virtual environment**:
    ```bash
    # Using Conda (recommended for M1 Mac):
    conda create -n ml_env python=3.9
    conda activate ml_env
    
    # Or using venv:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
    *Note: On macOS (M1/M2), `tensorflow-metal` is automatically installed for GPU acceleration.*

---

## 🚀 Usage

### Train the Models

```bash
python main.py buddy train
```

### Scan for Pairs

```bash
python main.py buddy scan
```

### Check Bot Status

```bash
python main.py buddy status
```

### Run the Bot

```bash
python main.py buddy --config config_tuned.yaml
```

---

## 🔧 Configuration

The provided `config_tuned.yaml` file is the central configuration for:
- **Trading Pairs**: Define which Forex pairs to trade.
- **Risk Parameters**: Set risk levels for trades.
- **Model Settings**: Adjust H1-specific tuning parameters.
- **Oanda API Credentials**: Add your account details here.

Customize this file to optimize the bot for your needs. Each section is well-commented for clarity.

---

## 📚 Learn More

Useful documentation to get started or delve deeper:
- **[CONFIDENCE_SYSTEM_DOCUMENTATION.md](CONFIDENCE_SYSTEM_DOCUMENTATION.md)** - Learn how direction, momentum, and confidence systems interact.
- **[risk_management/](risk_management)** - Risk and drawdown handling mechanisms.

---

**Use at your own risk. Always test on demo accounts first.**