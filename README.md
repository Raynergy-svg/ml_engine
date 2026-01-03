# ML Engine - Forex Trading Bot

## ⚠️ **WARNING: DISCLAIMER**

**WARNING: This is experimental code. Not financial advice. Trading Forex involves substantial risk of loss. Use only money you can afford to lose. Test on demo first. I am not responsible for any losses.**

**"It works for me. Might work for you. Probably won't make you rich. But it might teach you something."**

---

## What Is This?

Python-based ML engine with TCN/Transformer + XGBoost + RandomForest + Ridge modular ensemble (separate roles for direction, momentum, risk, confidence), gated decision logic, daily pair scanning, and end-to-end Oanda integration.

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

4. **For M1 Mac users** (optional - GPU acceleration):
    ```bash
    pip install tensorflow-metal
    ```

---

## 🚀 Getting Started

### Train the Model

```bash
python main.py train-buddy --csv path/to/market_data.csv
```

### Run the Bot

```bash
python main.py buddy --config config_tuned.yaml
```

### Configuration

Edit `config_tuned.yaml` to customize:
- Trading pairs
- Risk parameters
- Model settings
- Oanda API credentials

---

## 📚 Learn More

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[CONFIDENCE_SYSTEM_DOCUMENTATION.md](CONFIDENCE_SYSTEM_DOCUMENTATION.md)** - How the confidence system works
- **[FX_TIER1_GUARDRAILS_PLAN.md](FX_TIER1_GUARDRAILS_PLAN.md)** - Risk management details

---

**Use at your own risk. Always test on demo accounts first.**