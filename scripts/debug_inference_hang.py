#!/usr/bin/env python3
"""
Debug script to isolate inference bottleneck.
Replicates exact buddy command flow.
"""

import time
import os
import sys

# Add project root to path
sys.path.insert(0, '/Users/davidcertan/Desktop/ml_engine')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def timed(label):
    """Context manager to time a block."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            print(f"⏱️ Starting: {label}...", flush=True)
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"✅ {label}: {elapsed:.2f}s", flush=True)
    return Timer()

if __name__ == "__main__":
    print("=" * 60)
    print("BUDDY COMMAND FLOW DEBUGGER (WITH LLM)")
    print("=" * 60)
    
    # == STEP 1: LLM INITIALIZATION (matches buddy command) ==
    print("\n--- STEP 1: LLM Initialization ---")
    with timed("Initialize LLM provider (Claude)"):
        try:
            from llm_providers import initialize_buddy_llm, check_provider_status
            provider = initialize_buddy_llm(provider=None)  # Auto-select
            print(f"  Provider: {provider.name}, Available: {provider.is_available}")
        except Exception as e:
            print(f"  ❌ LLM init error: {e}")
    
    # == STEP 2: JOURNAL SYNC (matches buddy command) ==
    print("\n--- STEP 2: Journal Sync ---")
    with timed("Sync trade journal with OANDA"):
        try:
            from trade_journal import TradeJournal
            from oanda_practice import OandaPracticeClient
            journal = TradeJournal()
            client = OandaPracticeClient.from_env()
            sync_results = journal.sync_open_trades(client)
            print(f"  Closed updated: {sync_results.get('closed_updated', 0)}")
        except Exception as e:
            print(f"  ⚠ Sync error: {e}")
    
    # == STEP 3: ONLINE LEARNING BACKFILL (matches buddy command) ==
    print("\n--- STEP 3: Online Learning Backfill ---")
    with timed("Backfill online learning from journal"):
        try:
            from trade_journal import TradeJournal
            from market_intelligence import MarketIntelligence
            journal = TradeJournal()
            added = journal.backfill_online_learning()
            intel = MarketIntelligence(enable_online_learning=True)
            buffer_size = len(intel.online_learner.trade_buffer) if intel.online_learner else 0
            print(f"  Buffer size: {buffer_size}, Added: {added}")
        except Exception as e:
            print(f"  ⚠ Backfill error: {e}")
    
    # == STEP 4: FETCH LIVE BALANCE (matches buddy command) ==
    print("\n--- STEP 4: Fetch OANDA Balance ---")
    with timed("Fetch live account balance"):
        try:
            from oanda_practice import OandaPracticeClient
            client = OandaPracticeClient.from_env()
            summary = client.get_account_summary()
            nav = float(summary.get('account', {}).get('NAV', 0))
            print(f"  NAV: ${nav:,.2f}")
        except Exception as e:
            print(f"  ⚠ Balance error: {e}")
    
    # == STEP 5: LOAD CONFIG (matches buddy command) ==
    print("\n--- STEP 5: Load Config ---")
    with timed("Load config file"):
        from src.utils import load_config
        cfg = load_config("config/config_intel_optimized.yaml")
        print(f"  Config loaded")
    
    # == STEP 6: CREATE ENSEMBLE (matches buddy command) ==
    print("\n--- STEP 6: Create ModularEnsembleInference ---")
    with timed("Import modular_inference"):
        from src.core.modular_inference import ModularEnsembleInference, InferenceConfig
    
    with timed("Create ensemble instance (with RL + LLM)"):
        ensemble = ModularEnsembleInference(
            instrument="EUR_USD",
            use_rl_sizer=True,
            enable_llm_integration=True,  # ENABLED - matches user's flow
        )
    
    # == STEP 7: LOAD MODELS (bottleneck suspected here) ==
    print("\n--- STEP 7: Load Models (DETAILED) ---")
    
    # Manually time each part of load_models
    print("  7a. Loading Transformer...")
    t_start = time.perf_counter()
    
    from pathlib import Path
    from src.training.modular_trainers import TransformerDirectionTrainer, XGBoostTrainer, RandomForestTrainer, RidgeTrainer
    
    tf_path = Path("trained_data/models/EUR_USD/transformer_direction.keras")
    with timed("Load Transformer"):
        if tf_path.exists():
            trainer = TransformerDirectionTrainer()
            trainer.load(str(tf_path))
    
    with timed("Load XGBoost"):
        xgb_path = Path("trained_data/models/EUR_USD/xgb_momentum.pkl")
        if xgb_path.exists():
            xgb = XGBoostTrainer()
            xgb.load(str(xgb_path))
    
    with timed("Load RandomForest"):
        rf_path = Path("trained_data/models/EUR_USD/rf_risk.pkl")
        if rf_path.exists():
            rf = RandomForestTrainer()
            rf.load(str(rf_path))
    
    with timed("Load Ridge"):
        ridge_path = Path("trained_data/models/EUR_USD/ridge_confidence.pkl")
        if ridge_path.exists():
            ridge = RidgeTrainer()
            ridge.load(str(ridge_path))
    
    # RL Sizer - this was 4 seconds in our test
    print("\n  7b. Loading RL Position Sizer...")
    with timed("Load RL Position Sizer"):
        try:
            from rl_position_sizing import RLPositionSizer
            rl_sizer = RLPositionSizer()
            result = rl_sizer.load()
            print(f"    RL loaded: {result}")
        except Exception as e:
            print(f"    ❌ RL error: {e}")
    
    # == STEP 8: FULL ENSEMBLE LOAD ==
    print("\n--- STEP 8: Full ensemble.load_models() ---")
    with timed("Full ensemble.load_models()"):
        ensemble2 = ModularEnsembleInference(
            instrument="EUR_USD",
            use_rl_sizer=True,
            enable_llm_integration=True,
        )
        ensemble2.load_models()
    
    # == STEP 9: FETCH CANDLES ==
    print("\n--- STEP 9: Fetch Candles ---")
    with timed("Fetch 500 H1 candles from OANDA"):
        from src.utils.oanda_practice import OandaPracticeClient
        from fx_paper import candles_to_ohlcv_df
        client = OandaPracticeClient.from_env()
        resp = client.get_candles("EUR_USD", granularity="H1", count=500, price="MBA")
        df = candles_to_ohlcv_df(resp)
        print(f"  Fetched {len(df)} candles")
    
    # == STEP 10: FEATURE ENGINEERING ==
    print("\n--- STEP 10: Feature Engineering ---")
    import numpy as np
    with timed("Feature engineering"):
        from feature_engineering import FeatureEngineering
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(df, include_all=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0.0)
        print(f"  Features: {len(df.columns)}")
    
    # == STEP 11: INFERENCE (LLM calls here) ==
    print("\n--- STEP 11: Run Inference (predict_verbose) ---")
    with timed("predict_verbose (includes LLM calls if edge case)"):
        result = ensemble2.predict_verbose(df, instrument="EUR_USD")
        print(f"  Decision: {result.get('decision', 'N/A')[:50]}...")
    
    print("\n" + "=" * 60)
    print("COMPLETE - No hang detected")
    print("=" * 60)
