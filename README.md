# ML Engine
A simple machine learning tool for predictions in the financial markets, with features tuned for FX, algorithmic trading, and Oanda integration.

## Features
- Predict **stock** and **FX markets** trends using advanced algorithms.
- Unified trading via `buddy` command-line workflows.
- Real-time dashboard for tracking model accuracy and trades.

## Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Raynergy-svg/ml_engine.git && cd ml_engine
   ```

2. Set up your Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

Train a model:
```bash
python main.py train-buddy --config config.yaml
```

Run forecasts:
```bash
python main.py buddy runtime --api-key {your-key}
```

---
Stay ahead, take action, and make profitable decisions with ML Engine!
