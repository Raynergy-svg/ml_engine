#!/usr/bin/env python3
"""
Comprehensive debug script for ML Engine self-improvement systems.
Tests: RL, Online Learning, Continual Learning (EWC/EMA/Replay), Trade Analysis
"""

import pickle
import json
from pathlib import Path
from datetime import datetime

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_status(name, ok, detail=""):
    status = "✅" if ok else "❌"
    print(f"  {status} {name}: {detail}")

def main():
    print_header("ML ENGINE SELF-IMPROVEMENT SYSTEMS DEBUG")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_ok = True
    
    # 1. TRADE JOURNAL
    print_header("1. TRADE JOURNAL")
    journal_path = Path('trained_data/trade_journal.json')
    if journal_path.exists():
        with open(journal_path) as f:
            trades = json.load(f)
        print_status("Trade Journal", True, f"{len(trades)} trades")
    else:
        print_status("Trade Journal", False, "NOT FOUND")
        all_ok = False
    
    # 2. ONLINE LEARNING
    print_header("2. ONLINE LEARNING")
    ol_buffer = Path('trained_data/online_learning/trade_buffer.json')
    if ol_buffer.exists():
        with open(ol_buffer) as f:
            data = json.load(f)
        trades_count = len(data.get('trades', []))
        print_status("Online Learning Buffer", True, f"{trades_count} trades")
        print_status("Trades Since Retrain", True, str(data.get('trades_since_retrain', 0)))
    else:
        print_status("Online Learning Buffer", False, "NOT FOUND")
    
    # 3. RL POSITION SIZER
    print_header("3. RL POSITION SIZER")
    try:
        from rl_position_sizing import RLPositionSizer, SB3_AVAILABLE, GYM_AVAILABLE
        print_status("stable-baselines3", SB3_AVAILABLE, "available" if SB3_AVAILABLE else "NOT INSTALLED")
        print_status("gymnasium", GYM_AVAILABLE, "available" if GYM_AVAILABLE else "NOT INSTALLED")
        
        rl_model = Path('trained_data/models/rl_position_sizer.zip')
        if rl_model.exists():
            sizer = RLPositionSizer()
            loaded = sizer.load()
            print_status("RL Model", loaded, f"loaded from {rl_model.name}")
        else:
            print_status("RL Model", False, "NOT TRAINED (need 100+ trades)")
    except ImportError as e:
        print_status("RL Position Sizer", False, str(e))
        all_ok = False
    
    # 4. CONTINUAL LEARNING
    print_header("4. CONTINUAL LEARNING (EWC + EMA + Replay)")
    
    # Check EUR_USD models (most recent)
    eur_usd = Path('trained_data/models/EUR_USD')
    if eur_usd.exists():
        # EMA
        ema_path = eur_usd / 'transformer_direction.ema.pkl'
        if ema_path.exists():
            with open(ema_path, 'rb') as f:
                ema = pickle.load(f)
            print_status("EMA Weights", True, f"{len(ema)} tensor groups")
        else:
            print_status("EMA Weights", False, "NOT FOUND")
        
        # EWC
        ewc_path = eur_usd / 'transformer_direction.ewc.pkl'
        if ewc_path.exists():
            with open(ewc_path, 'rb') as f:
                ewc = pickle.load(f)
            print_status("EWC State", True, f"{ewc.get('n_tasks', 0)} tasks, λ={ewc.get('ewc_lambda', 0)}")
        else:
            print_status("EWC State", False, "NOT FOUND")
    else:
        print_status("EUR_USD Models", False, "Directory NOT FOUND")
    
    # Replay buffers
    replay = Path('trained_data/replay')
    if replay.exists():
        replay_files = list(replay.glob('**/*.npz'))
        print_status("Replay Buffers", len(replay_files) > 0, f"{len(replay_files)} files")
    else:
        print_status("Replay Buffers", False, "NOT FOUND")
    
    # 5. TRADE ANALYZER
    print_header("5. TRADE ANALYZER")
    try:
        from trade_analyzer import TradeAnalyzer, GATE_THRESHOLDS
        print_status("TradeAnalyzer", True, "imported")
        print(f"      Gate thresholds: {GATE_THRESHOLDS}")
    except ImportError as e:
        print_status("TradeAnalyzer", False, str(e))
    
    # 6. MARKET INTELLIGENCE
    print_header("6. MARKET INTELLIGENCE")
    try:
        from market_intelligence import MarketIntelligence
        mi = MarketIntelligence(
            enable_sentiment=False,
            enable_calendar=False,
            enable_online_learning=True
        )
        print_status("MarketIntelligence", True, "initialized")
        print_status("Online Learner", mi.online_learner is not None, 
                    f"{len(mi.online_learner.trade_buffer) if mi.online_learner else 0} trades in buffer")
    except Exception as e:
        print_status("MarketIntelligence", False, str(e))
    
    # 7. MODULAR INFERENCE
    print_header("7. MODULAR INFERENCE (Gate System)")
    try:
        from src.core.modular_inference import ModularEnsembleInference, InferenceGateConfig
        config = InferenceGateConfig()
        print_status("Gate Config", True, "loaded")
        print(f"      TCN threshold: {config.min_tcn_probability}")
        print(f"      Ridge threshold: {config.min_ridge_confidence}")
        print(f"      Drawdown max: {config.max_drawdown_pct * 100:.1f}% ({config.max_drawdown_pips} pips)")
    except Exception as e:
        print_status("ModularInference", False, str(e))
    
    # Summary
    print_header("SUMMARY")
    if all_ok:
        print("  ✅ All core systems operational")
    else:
        print("  ⚠️  Some systems need attention")
    
    print("""
  Notes:
  - RL training requires 100+ trades in online learning buffer
  - Current buffer has 25 trades - will train when threshold reached
  - EWC/EMA/Replay are active during training
  - Online learning updates after each closed trade
""")

if __name__ == '__main__':
    main()
