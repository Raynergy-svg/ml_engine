#!/usr/bin/env python3
"""Debug script to check all self-improvement and RL systems."""

from pathlib import Path
import json


def main():
    print("=" * 60)
    print("🔍 ML ENGINE SELF-IMPROVEMENT SYSTEMS DEBUG")
    print("=" * 60)
    
    # 1. Trade Journal
    print("\n📊 1. TRADE JOURNAL")
    journal_path = Path('trained_data/trade_journal.json')
    if journal_path.exists():
        with open(journal_path) as f:
            trades = json.load(f)
        print(f"   ✓ Found: {len(trades)} trades")
        if trades:
            last = trades[-1]
            print(f"   Last trade: {last.get('instrument', 'N/A')} @ {last.get('entry_time', 'N/A')}")
    else:
        print("   ✗ NOT FOUND")
    
    # 2. RL Buffer
    print("\n🤖 2. RL POSITION SIZER")
    rl_buffer = Path('trained_data/rl_sizer/trade_buffer.json')
    if rl_buffer.exists():
        with open(rl_buffer) as f:
            buffer = json.load(f)
        print(f"   ✓ Buffer: {len(buffer)} trades (need 100 for training)")
    else:
        print("   ✗ Buffer NOT FOUND")
    
    rl_model = Path('trained_data/rl_sizer/ppo_position_sizer.zip')
    if rl_model.exists():
        print(f"   ✓ Model: EXISTS ({rl_model.stat().st_size / 1024:.1f} KB)")
    else:
        print("   ✗ Model: NOT TRAINED YET")
    
    # 3. Online Learning Buffer
    print("\n🔄 3. ONLINE LEARNING")
    ol_buffer = Path('trained_data/online_learning_buffer.json')
    if ol_buffer.exists():
        with open(ol_buffer) as f:
            buffer = json.load(f)
        print(f"   ✓ Buffer: {len(buffer)} trades")
    else:
        print("   ✗ Buffer NOT FOUND")
    
    # 4. Replay Buffers (Continual Learning)
    print("\n♻️  4. CONTINUAL LEARNING (EWC + Replay)")
    replay = Path('trained_data/replay')
    if replay.exists():
        files = list(replay.glob('**/*.npz'))
        print(f"   ✓ Replay buffers: {len(files)} files")
        for f in files[:5]:
            print(f"     - {f.relative_to(replay)}")
    else:
        print("   ✗ Replay buffers NOT FOUND")
    
    # 5. EWC State
    ewc_files = list(Path('trained_data/models').glob('**/ewc_state.json'))
    if ewc_files:
        print(f"   ✓ EWC state files: {len(ewc_files)}")
    else:
        print("   ✗ EWC state NOT FOUND")
    
    # 6. EMA Weights
    ema_files = list(Path('trained_data/models').glob('**/ema_weights.npz'))
    if ema_files:
        print(f"   ✓ EMA weight files: {len(ema_files)}")
    else:
        print("   ✗ EMA weights NOT FOUND")
    
    # 7. Check RL module
    print("\n🧪 5. MODULE IMPORTS")
    try:
        from src.training.rl.position_sizer import RLPositionSizer  # noqa: F401
        print("   ✓ RLPositionSizer imported")
    except ImportError as e:
        print(f"   ✗ RLPositionSizer import failed: {e}")
    
    try:
        from market_intelligence import MarketIntelligence  # noqa: F401
        print("   ✓ MarketIntelligence imported")
    except ImportError as e:
        print(f"   ✗ MarketIntelligence import failed: {e}")
    except ImportError as e:
        print(f"   ✗ MarketIntelligence import failed: {e}")
    
    try:
        from trade_analyzer import TradeAnalyzer  # noqa: F401
        print("   ✓ TradeAnalyzer imported")
    except ImportError as e:
        print(f"   ✗ TradeAnalyzer import failed: {e}")
    
    # 8. Check stable-baselines3
    print("\n📦 6. DEPENDENCIES")
    try:
        import stable_baselines3
        print(f"   ✓ stable-baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print("   ✗ stable-baselines3 NOT INSTALLED")
        print("     Run: pip install stable-baselines3")
    
    try:
        import gymnasium
        print(f"   ✓ gymnasium: {gymnasium.__version__}")
    except ImportError:
        print("   ✗ gymnasium NOT INSTALLED")
    
    # 9. Test RL Environment
    print("\n🎮 7. RL ENVIRONMENT TEST")
    try:
        from src.training.rl.position_sizer import RLPositionSizer as _RLPositionSizer
        sizer = _RLPositionSizer()
        print("   ✓ RLPositionSizer initialized")
        print(f"   Model path: {sizer.model_path}")
        print(f"   Trade buffer: {len(sizer.trade_buffer)} trades")
    except Exception as e:
        print(f"   ✗ RLPositionSizer init failed: {e}")
    
    # 10. Test Online Learning in MarketIntelligence
    print("\n🧠 8. MARKET INTELLIGENCE ONLINE LEARNING")
    try:
        from market_intelligence import MarketIntelligence as _MarketIntelligence
        mi = _MarketIntelligence(
            enable_sentiment=False,
            enable_calendar=False,
            enable_online_learning=True
        )
        print("   ✓ MarketIntelligence initialized")
        print(f"   Online learning: {mi.enable_online_learning}")
        print(f"   Trade buffer: {len(mi.trade_buffer)} trades")
    except Exception as e:
        print(f"   ✗ MarketIntelligence init failed: {e}")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
